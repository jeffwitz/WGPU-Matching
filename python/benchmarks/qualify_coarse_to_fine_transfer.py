"""Re-solve fine DIC windows using validated coarse affine states."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from wgpu_matching.dic_synthetic import generate_speckle_pair
from wgpu_matching.seed_propagation import transfer_to_fine_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--discontinuity-boundary-fraction", type=float, default=.5)
    parser.add_argument("--diagnostic-boundary-x", type=float)
    parser.add_argument("--oracle-region-boundary-x", type=float,
                        help="Synthetic diagnostic only: supply the true boundary to the initializer")
    args = parser.parse_args()
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=32,
                      subset_stride=16, interpolation="linear", motion_model="affine",
                      outlier_policy="disabled")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    paths = sorted(args.fine.glob("windows_seed*/*.npz"))
    if not paths:
        parser.error("no fine-grid archives")
    for path in paths:
        seed_report = args.fine / f"{path.parent.name.removeprefix('windows_')}.json"
        if seed_report.exists():
            protocol = json.loads(seed_report.read_text())["protocol"]
            if protocol.get("discontinuity_boundary_fraction", .5) != args.discontinuity_boundary_fraction:
                raise ValueError("fine-grid archive uses a different discontinuity boundary")
        with np.load(path) as fine, np.load(args.coarse / path.name) as coarse:
            centres = fine["centres_xy_native"].copy()
            visible = fine["visible"].astype(bool)
            truth = fine["truth_parameters"].copy()
            direct = fine["icgn_correct_lt2px"].astype(bool)
            coarse_centres = coarse["centres"]
            coarse_centres = coarse_centres.copy()
            coarse_shape = (len(np.unique(coarse_centres[:, 1])), len(np.unique(coarse_centres[:, 0])))
            region_options = {}
            if args.oracle_region_boundary_x is not None:
                boundary = args.oracle_region_boundary_x
                def labels(points, half_width, boundary=boundary):
                    x = points[:, 0]
                    return np.where(x + half_width < boundary, 0,
                                    np.where(x - half_width >= boundary, 1, -1))
                region_options = {"coarse_region_labels": labels(coarse_centres, 31.5),
                                  "fine_region_labels": labels(centres, 15.5)}
            initial = transfer_to_fine_grid(coarse_centres, coarse_shape, coarse["parameters"],
                                            coarse["accepted"], centres, **region_options)
        mode = path.stem.split("_")[0]
        texture = int(path.stem.split("texture")[1].split("_")[0])
        warp = int(path.stem.split("warp")[1])
        pair = generate_speckle_pair(texture, warp, 3840, 2160, motion_mode=mode,
                                     discontinuity_boundary_fraction=args.discontinuity_boundary_fraction,
                                     texture_options={"characteristic_size_px": 7., "coverage": .31,
                                                      "blur_sigma_px": .9, "softness": .612, "noise_std": .025})
        solver.set_reference(np.rint(pair.reference * 255).astype(np.uint8),
                             iterations=30, tolerance=1e-3, zncc_min=.65)
        state = np.zeros((len(centres), 8), np.float32)
        active = initial["available"]
        state[active, :6] = initial["parameters"][active]
        state[active, 6] = 1
        result = solver._register_seeded_frame(np.rint(pair.target * 255).astype(np.uint8),
                                               state, active.astype(np.uint32))
        params = np.asarray(result.parameters).reshape(-1, 6)
        accepted = (np.asarray(result.status).reshape(-1) > .5) & active
        error = np.linalg.norm(params[:, [0, 3]] - truth[:, [0, 3]], axis=1)
        correct = accepted & visible & (error < 2)
        support_crossing = np.zeros(len(centres), dtype=bool)
        opposite_source = np.zeros(len(centres), dtype=bool)
        if args.diagnostic_boundary_x is not None:
            x = centres[:, 0]
            boundary = args.diagnostic_boundary_x
            support_crossing = (x - 15.5 < boundary) & (x + 15.5 >= boundary)
            valid_source = np.flatnonzero(active)
            opposite_source[valid_source] = ((x[valid_source] < boundary)
                                               != (coarse_centres[initial["source"][valid_source], 0] < boundary))
        rows.append({"pair": path.stem, "visible": int(visible.sum()),
                     "available": int((active & visible).sum()), "accepted": int((accepted & visible).sum()),
                     "correct": int(correct.sum()), "false_accepted": int((accepted & visible & (error >= 2)).sum()),
                     "accepted_invisible": int((accepted & ~visible).sum()), "direct_correct": int((direct & visible).sum()),
                     "accepted_crossing_support": int((accepted & support_crossing).sum()),
                     "visible_opposite_source": int((visible & opposite_source).sum())})
        np.savez_compressed(args.output_dir / path.name, centres=centres, visible=visible, available=active,
                            source=initial["source"], initial=initial["parameters"], parameters=params,
                            accepted=accepted, correct=correct, error=error,
                            diagnostic_support_crossing=support_crossing, diagnostic_opposite_source=opposite_source)
        print(f"Completed {path.stem}", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"pairs": rows}, indent=2) + "\n")
    lines = ["# Transfer from 64 px accepted windows to the 32 px grid", "",
             ("Each fine-grid initial affine state is transported from its nearest solver-accepted "
              "64 px neighbour within 96 px. It counts as a fine result only if the WebGPU "
              "inverse-compositional Gauss–Newton (IC-GN) solver accepts it. The source mask "
              "does not use analytical visibility. These synthetic pairs were already inspected."), "",
             f"Visible fine windows: {sum(r['visible'] for r in rows)}.",
             f"Available initializations: {sum(r['available'] for r in rows)}.",
             (f"Correct fine results: {sum(r['correct'] for r in rows)}; "
              f"false accepted: {sum(r['false_accepted'] for r in rows)} / {sum(r['accepted'] for r in rows)}."),
             f"Direct fine-grid baseline correct: {sum(r['direct_correct'] for r in rows)}.", "",
             ("Per-pair results and per-window source indices are retained in JSON and NPZ files. "
              + ("The true material boundary is supplied to the initializer: this is an oracle control, "
                 "not an automatic segmentation method. " if args.oracle_region_boundary_x is not None else
                 "No material-boundary constraint is imposed; this experiment does not validate transfers across cracks. ")
              + "Its timings are not performance measurements."), "", "Reproduce:", "", "```bash",
             "PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_coarse_to_fine_transfer.py \\",
             f"  --coarse {args.coarse} --fine {args.fine} \\",
             f"  --live-dic-source {args.live_dic_source} --output-dir {args.output_dir} \\",
             f"  --discontinuity-boundary-fraction {args.discontinuity_boundary_fraction:g}"
             + (f" --diagnostic-boundary-x {args.diagnostic_boundary_x:g}" if args.diagnostic_boundary_x is not None else "")
             + (f" --oracle-region-boundary-x {args.oracle_region_boundary_x:g}" if args.oracle_region_boundary_x is not None else ""), "```", ""]
    if args.diagnostic_boundary_x is not None:
        lines.extend([f"Diagnostic boundary x = {args.diagnostic_boundary_x:g} px: "
                      f"{sum(r['accepted_crossing_support'] for r in rows)} accepted windows cross the boundary, "
                      f"and {sum(r['visible_opposite_source'] for r in rows)} visible initialized windows use "
                      "a coarse source on the opposite side. "
                      + (f"The initializer uses the oracle boundary x = {args.oracle_region_boundary_x:g} px."
                         if args.oracle_region_boundary_x is not None else "The boundary is not used for inference."), ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

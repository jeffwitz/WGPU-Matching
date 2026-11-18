"""Secondary image-only silhouette proposal, with unchanged solver confirmation.

This comparator is specific to the supplied complete constant-background objects.
It cannot establish accuracy in the absence of associated displacement truth.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from hudge_strains_data import material_mask, silhouette_proposal
from PIL import Image
from wgpu_matching.seed_initialization import initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["FEA", "ClosedFormExpression"], required=True)
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--subset", type=int, choices=[32, 64], default=64)
    parser.add_argument("--output", type=Path, default=Path("results/hudge_strains_20261114"))
    args = parser.parse_args()
    sys.path.insert(0, "/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src")
    from live_dic import ICGNWGPU
    directory = Path("/home/jeff/CNRS/DIC/HudgeStrains") / args.family
    reference = np.asarray(Image.open(directory / "Img_F.tif"))
    mask = material_mask(reference)
    subset = args.subset
    solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=subset,
                      subset_stride=subset, interpolation="linear", motion_model="affine",
                      outlier_policy="disabled")
    started = time.perf_counter()
    solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=.65, roi_mask=mask)
    reference_time = time.perf_counter() - started
    height, width = reference.shape
    first = (subset - 1) / 2
    xx, yy = np.meshgrid(np.arange(first, width-first, subset), np.arange(first, height-first, subset))
    centres = np.column_stack((xx.ravel(), yy.ravel()))
    valid = np.asarray(solver.roi.selected).ravel().astype(bool)
    count = len(centres)
    assert len(valid) == count
    pattern = "Img_wa_g*.tif" if args.family == "FEA" else "Img_G*.tif"
    output = args.output / f"silhouette_{subset}" / args.family
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(directory.glob(pattern)):
        level = int(path.stem[-3:])
        if args.levels and level not in args.levels:
            continue
        record_path = output / f"level_{level:03d}.json"
        if record_path.exists():
            rows.append(json.loads(record_path.read_text()))
            continue
        started = time.perf_counter()
        target = np.asarray(Image.open(path))
        proposal, geometry = silhouette_proposal(reference, target, centres)
        calls = []

        def solve(indices, values, target=target, calls=calls):
            states = np.zeros((count, 8), np.float32)
            states[indices, :6] = values
            states[indices, 6] = 1
            active = np.zeros(count, np.uint32)
            active[indices] = 1
            result = solver._register_seeded_frame(target, states, active.reshape(solver.roi.selected.shape))
            calls.append(len(indices))
            return (np.asarray(result.parameters).reshape(-1, 6)[indices],
                    np.asarray(result.status).ravel()[indices] > .5)

        result = initialize(centres, xx.shape, valid,
                            [("zero", np.zeros_like(proposal)), ("silhouette", proposal)], solve,
                            sparse_stride=2, tau_u=8., tau_g=.25, minimum_anchors=3,
                            batch_confirmation=True)
        elapsed = time.perf_counter() - started
        row = {"family": args.family, "level": level, "subset_and_stride": subset, "valid_roi": int(valid.sum()),
               "accepted": int(result["accepted"].sum()), "seeds": int(result["confirmed_seeds"].sum()),
               "hypotheses": result["hypothesis_acceptances"], "selected": result["selected_hypothesis"],
               "passes": result["passes"], "solver_calls": len(calls), "attempted_windows": sum(calls),
               "target_wall_s": elapsed, "reference_prepare_s": reference_time, **geometry,
               "truth_status": "unassociated", "temporal_state_reused": False}
        np.savez_compressed(output / f"level_{level:03d}.npz", centres=centres, valid_roi=valid,
                            initial_parameters=proposal, **{k: result[k] for k in
                            ["parameters", "accepted", "confirmed_seeds", "generation", "parent", "attempts"]})
        record_path.write_text(json.dumps(row, indent=2)+"\n")
        rows.append(row)
        (output / "summary.json").write_text(json.dumps(rows, indent=2)+"\n")
        print(args.family, level, row["accepted"], "/", row["valid_roi"], elapsed, flush=True)


if __name__ == "__main__":
    main()

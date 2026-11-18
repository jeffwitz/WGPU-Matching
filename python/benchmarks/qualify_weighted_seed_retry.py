"""Re-solve weighted neighbour estimates on the held-out synthetic images."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from wgpu_matching.dic_synthetic import generate_speckle_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    args.output_dir.mkdir(parents=True, exist_ok=True)
    solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=64,
                      subset_stride=64, interpolation="linear", motion_model="affine",
                      outlier_policy="disabled")
    rows = []
    for path in sorted(args.estimates.glob("*.npz")):
        mode = path.stem.split("_")[0]
        texture = int(path.stem.split("texture")[1].split("_")[0])
        warp = int(path.stem.split("warp")[1])
        seed = texture // 1000000 * 1000000
        baseline = args.archive / "affine_ncc" / f"windows_seed{seed}" / path.name
        with np.load(path) as source, np.load(baseline) as truth:
            available = source["available"].astype(bool)
            visible = source["visible"].astype(bool)
            parameters = source["parameters"].copy()
            truth_parameters = truth["truth_parameters"].copy()
            oracle_correct = truth["oracle_icgn_correct_lt2px"].astype(bool)
        pair = generate_speckle_pair(texture, warp, 3840, 2160, motion_mode=mode,
                                     texture_options={"characteristic_size_px": 7., "coverage": .31,
                                                      "blur_sigma_px": .9, "softness": .612,
                                                      "noise_std": .025})
        solver.set_reference(np.rint(pair.reference * 255).astype(np.uint8),
                             iterations=30, tolerance=1e-3, zncc_min=.65)
        state = np.zeros((len(available), 8), np.float32)
        state[available, :6] = parameters[available]
        state[available, 6] = 1
        start = time.perf_counter()
        result = solver._register_seeded_frame(np.rint(pair.target * 255).astype(np.uint8),
                                               state, available.astype(np.uint32))
        elapsed = time.perf_counter() - start
        solved = np.asarray(result.parameters).reshape(-1, 6)
        accepted = (np.asarray(result.status).reshape(-1) > .5) & available
        error = np.linalg.norm(solved[:, [0, 3]] - truth_parameters[:, [0, 3]], axis=1)
        correct = accepted & visible & (error < 2)
        rows.append({"pair": path.stem, "attempted": int(available.sum()),
                     "attempted_visible": int((available & visible).sum()),
                     "accepted_visible": int((accepted & visible).sum()), "correct": int(correct.sum()),
                     "false_accepted": int((accepted & visible & (error >= 2)).sum()),
                     "accepted_invisible": int((accepted & ~visible).sum()), "solve_wall_s": elapsed})
        rows[-1]["attempted_visible_oracle_correct"] = int((available & visible & oracle_correct).sum())
        np.savez_compressed(args.output_dir / path.name, available=available, visible=visible,
                            accepted=accepted, parameters=solved, error_px=error, correct=correct)
        print(f"Completed {path.stem}", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"pairs": rows}, indent=2) + "\n")
    lines = ["# Weighted estimates followed by a solver retry", "",
             ("The WebGPU inverse-compositional Gauss–Newton (IC-GN) solver retries weighted estimates "
              "using unchanged settings. The active mask comes exclusively from the estimates; "
              "analytical visibility is used only for scoring."), "",
             "| Family | Attempted visible | Accepted visible | Correct <2 px | False accepted | Accepted invisible |",
             "|---|---:|---:|---:|---:|---:|"]
    for family in ("heterogeneous", "translation"):
        group = [r for r in rows if r["pair"].startswith(family + "_")]
        lines.append(f"| {family} | " + " | ".join(str(sum(r[k] for r in group)) for k in
                     ("attempted_visible", "accepted_visible", "correct", "false_accepted", "accepted_invisible")) + " |")
    for family in ("heterogeneous", "translation"):
        group = [r for r in rows if r["pair"].startswith(family + "_")]
        lines.append(f"\n{family}: {sum(r['attempted_visible_oracle_correct'] for r in group)} "
                     "attempted visible windows converge correctly under the analytical oracle.\n")
    lines.extend(["", ("These retries are a post-hoc experiment on the previously inspected test set; "
                  "they do not change the frozen G2 results. See `summary.json` and the per-window archives."), ""])
    lines.extend(["Reproduce from the repository root:", "", "```bash",
                  "PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_weighted_seed_retry.py \\",
                  f"  --estimates {args.estimates} --archive {args.archive} \\",
                  f"  --live-dic-source {args.live_dic_source} --output-dir {args.output_dir}", "```", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

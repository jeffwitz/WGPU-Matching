"""Re-solve failed shear-band windows from analytical states as an oracle diagnostic."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from solver_image_cache import read_solver_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.base / "g2.0/inputs/affine_ncc"
    evaluation = args.base / "g2.0/evaluation"
    solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=64, subset_stride=64,
                      interpolation="linear", motion_model="affine", outlier_policy="disabled")
    rows = []
    try:
        for f in sorted(evaluation.glob("*.npz")):
            archive = source / "windows_seed59000000" / f.name
            reference, target = read_solver_images(source / "solver_images_seed59000000", archive)
            with np.load(f) as result, np.load(archive) as original:
                bad = result["accepted"].astype(bool) & result["visible"].astype(bool) & (result["error"] >= 2)
                truth = original["truth_parameters"].copy()
                previously_correct = original["oracle_icgn_correct_lt2px"].astype(bool)
                solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=.65)
                initial = np.zeros((len(truth), 8), np.float32)
                initial[:, :6] = truth
                initial[:, 6] = 1
                oracle = solver._register_seeded_frame(target, initial, bad.astype(np.uint32))
                params = np.asarray(oracle.parameters).reshape(-1, 6)
                accepted = np.asarray(oracle.status).reshape(-1) > .5
                errors = np.linalg.norm(params[:, [0, 3]] - truth[:, [0, 3]], axis=1)
                correct = accepted & (errors < 2)
                if np.any(bad & (correct != previously_correct)):
                    raise ValueError("oracle re-solve disagrees with archived oracle scoring")
                selected = errors[bad & accepted]
                row = {"pair": f.stem, "false_windows": int(bad.sum()),
                       "oracle_accepted": int((bad & accepted).sum()),
                       "oracle_rejected": int((bad & ~accepted).sum()),
                       "oracle_correct": int((bad & correct).sum()),
                       "oracle_accepted_error_median_px": float(np.median(selected)) if len(selected) else None,
                       "oracle_accepted_error_max_px": float(np.max(selected)) if len(selected) else None}
                np.savez_compressed(args.output_dir / f.name, false_window=bad, oracle_parameters=params,
                                    oracle_accepted=accepted, oracle_error_px=errors)
                rows.append(row)
    finally:
        solver.close()
    (args.output_dir / "summary.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

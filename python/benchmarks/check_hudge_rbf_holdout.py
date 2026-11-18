"""Leave one accepted site out; solver observations are not mechanical truth."""

import json
from pathlib import Path

import numpy as np
from hudge_rbf_restart import fit_seed_field


def main():
    rows = []
    root = Path("results/hudge_affine_restart_20261115/all_levels/rectified_native")
    for level in range(1, 192):
        d = np.load(root/f"level_{level:03d}.npz")
        for index in np.flatnonzero(d["accepted"]):
            keep = d["accepted"].copy()
            keep[index] = False
            for method in ("affine", "rbf_position", "rbf_gradient"):
                field, _ = fit_seed_field(d["centres"], d["parameters"], keep, method=method)
                error = field.evaluate(d["centres"][[index]])[0] - d["parameters"][index]
                rows.append({"level": level, "index": int(index), "method": method,
                             "du": float(np.linalg.norm(error[[0, 3]])),
                             "dg": float(np.linalg.norm(error[[1, 2, 4, 5]]))})
    output = Path("results/hudge_rbf_restart_20261115")
    (output/"holdout.json").write_text(json.dumps(rows))
    summary = []
    for method in ("affine", "rbf_position", "rbf_gradient"):
        group = [r for r in rows if r["method"] == method]
        summary.append({"method": method, "count": len(group),
                        **{f"{key}_{metric}": float(function([r[key] for r in group]))
                           for key in ("du", "dg") for metric, function in
                           (("median", np.median), ("rms", lambda x: np.sqrt(np.mean(np.square(x)))))}})
    (output/"holdout_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

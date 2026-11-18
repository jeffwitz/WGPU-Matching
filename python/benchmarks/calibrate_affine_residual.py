"""Fit frozen image-residual alert thresholds on an explicitly labeled development set."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.development.glob("*.npz"))
    if not paths:
        raise ValueError("no development residual files")
    correlations, valid, good, bad = [], [], [], []
    hashes = {}
    for path in paths:
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:
            correlations.append(data["correlations"])
            valid.append(data["valid_score"].astype(bool))
            good.append(data["correct"].astype(bool))
            bad.append(data["false_accepted"].astype(bool))
    scores, valid, good, bad = (
        np.concatenate(correlations),
        np.concatenate(valid),
        np.concatenate(good),
        np.concatenate(bad),
    )
    if not good.any() or not bad.any():
        raise ValueError("development needs both correct and incorrect accepted windows")
    candidates = {}
    for name in ("global", "min_quadrant"):
        value = (scores[:, 0] if name == "global" else np.min(scores[:, 1:], axis=1)).astype(
            np.float64
        )
        mandatory = ~valid
        max_false_alerts = int(np.floor(0.05 * good.sum()))
        if int((mandatory & good).sum()) > max_false_alerts:
            candidates[name] = {
                "available": False,
                "reason": "unavailable scores already exceed alert budget",
            }
            continue
        finite = np.unique(value[valid & (good | bad)].astype(np.float64))
        thresholds = np.concatenate((finite[:1], np.nextafter(finite, np.inf)))
        best = None
        for threshold in thresholds:
            flagged = mandatory | (valid & (value < threshold))
            false_alerts = int((flagged & good).sum())
            true_alerts = int((flagged & bad).sum())
            if false_alerts <= max_false_alerts:
                rank = (true_alerts, -false_alerts, -float(threshold))
                if best is None or rank > best[0]:
                    best = (rank, float(threshold), true_alerts, false_alerts)
        if best is None:
            candidates[name] = {"available": False, "reason": "no threshold meets alert budget"}
            continue
        candidates[name] = {
            "available": True,
            "threshold": best[1],
            "incorrect_detected": best[2],
            "incorrect_total": int(bad.sum()),
            "correct_flagged": best[3],
            "correct_total": int(good.sum()),
            "unavailable_correct": int((mandatory & good).sum()),
            "unavailable_incorrect": int((mandatory & bad).sum()),
        }
    eligible = [name for name in ("global", "min_quadrant") if candidates[name]["available"]]
    selected = (
        max(eligible, key=lambda name: (candidates[name]["incorrect_detected"], name == "global"))
        if eligible
        else None
    )
    result = {
        "purpose": "development-only residual alert; not a confidence guarantee",
        "comparison": "score < threshold",
        "invalid_score_policy": "suspect",
        "max_correct_alert_fraction": 0.05,
        "development_hashes_sha256": hashes,
        "candidates": candidates,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

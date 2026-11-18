"""Score the frozen P2 held-out qualification with image-pair bootstrap."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--frozen-parameters", type=Path, required=True)
    parser.add_argument("--local-validation", type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_parameters.read_text())["selected"]
    report = json.loads(args.summary.read_text())
    protocol = report["protocol"]
    for setting, key in (("tau_u", "tau_u"), ("tau_g", "tau_g"),
                         ("sparse_stride", "sparse_stride"), ("minimum_anchors", "minimum_anchors")):
        if protocol[setting] != frozen[key]:
            raise ValueError(f"test setting {setting} differs from frozen development value")
    rows = report["rows"]
    families = {family: [r for r in rows if r["pair"].startswith(family + "_")]
                for family in ("heterogeneous", "translation")}
    if any(len(group) != 20 for group in families.values()):
        raise ValueError("gate requires 20 pairs of each family")
    h = families["heterogeneous"]
    t = families["translation"]
    diff = np.array([(r["correct"] - r["baseline_correct"]) / r["visible"] for r in h])
    control_diff = np.array([(r["correct"] - r["control_correct"]) / r["visible"] for r in h])
    rng = np.random.default_rng(20261111)
    lo, hi = np.quantile(rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1), [.025, .975])
    control_lo, control_hi = np.quantile(
        rng.choice(control_diff, size=(10000, len(control_diff)), replace=True).mean(axis=1), [.025, .975]
    )
    correct, visible = sum(r["correct"] for r in h), sum(r["visible"] for r in h)
    accepted, bad = sum(r["accepted"] for r in h), sum(r["false_accepted"] for r in h)
    below = sum(r["correct"] / r["visible"] < .2 for r in h)
    trans_delta = sum(r["correct"] - r["baseline_correct"] for r in t)
    baseline_cases = []
    seed_offsets = sorted({int(row["pair"].split("texture")[1].split("_")[0]) // 1000000 * 1000000
                           for row in rows})
    for seed in seed_offsets:
        baseline_report = Path(protocol["archive"]) / "affine_ncc" / f"combined_seed{seed}.json"
        baseline_cases.extend(json.loads(baseline_report.read_text())["cases"])
    translation_cases = [case for case in baseline_cases if case["mode"] == "translation"]
    rejected_prefits = sum(not case["global_affine_prefit_used"] for case in translation_cases)
    passed = (correct / visible >= .85 and lo > 0 and below <= 1
              and bad / max(accepted, 1) <= .001 and trans_delta >= 0)
    rejected = lo <= 0 <= hi or bad / max(accepted, 1) > .005
    verdict = "PASS" if passed else "REJECT" if rejected else "INDETERMINATE"
    numerical_verdict = "INVALID BASELINE" if rejected_prefits else verdict
    limitations = []
    if not protocol.get("local_affine_escalation", False):
        limitations.append("local affine hypothesis source not evaluated")
    if not protocol.get("same_budget_control", False) or not protocol.get("escalation", False):
        limitations.append("matched re-solve control or regional escalation missing")
    if not protocol.get("full_pipeline_timing", False):
        limitations.append("end-to-end hypothesis-generation timing not measured")
    if args.local_validation:
        local = json.loads(args.local_validation.read_text())
        local_rows = local["rows"]
        local_protocol = local["protocol"]
        if (len(local_rows) != 20 or not local_protocol.get("local_affine_validation")
                or local_protocol.get("sparse_stride") != frozen["sparse_stride"]
                or local_protocol.get("tau_u") != frozen["tau_u"]
                or local_protocol.get("tau_g") != frozen["tau_g"]
                or sum(row["false_accepted"] for row in local_rows) > .001 * max(1, sum(row["accepted"] for row in local_rows))):
            raise ValueError("local fallback development validation is incomplete or fails its guard")
    else:
        limitations.append("local affine fallback not validated on the full development set")
    if rejected_prefits:
        limitations.append("translation baseline rejected one or more global affine prefits")
    if limitations:
        verdict = "INCOMPLETE"
    lines = ["# Held-out image-only initialization", "",
             ("The thresholds and sampling stride were frozen on the development group before these test cases. "
             "The inverse-compositional Gauss–Newton (IC-GN) solver accepts anchors; analytic truth "
             "is used only after selection and propagation for scoring."), "",
             (f"G2: **{verdict}**. Heterogeneous correct {correct}/{visible} ({correct / visible:.2%}); "
             f"paired mean gain {diff.mean():+.2%}, 95% pair bootstrap [{lo:+.2%}, {hi:+.2%}]; "
             f"{below}/20 pairs below 20%; {bad}/{accepted} false accepted; "
             f"translation correct-count difference {trans_delta:+d}."), "",
             f"G2 numerical criteria: **{numerical_verdict}**; the qualification limitations below are reported separately.", "",
             (f"Translation baseline diagnostic: {rejected_prefits}/{len(translation_cases)} global affine "
              + ("prefits were rejected; this can discard translations and invalidates the non-regression comparison."
                 if rejected_prefits else "prefits were rejected; the near-identity affine retention option was used.")), "",
             "Qualification limitations: " + "; ".join(limitations) + ".", "",
             (f"Heterogeneous mean gain versus the matched re-solve control: {control_diff.mean():+.2%}, "
              f"95% pair bootstrap [{control_lo:+.2%}, {control_hi:+.2%}]. "
              f"Accepted windows outside analytical visibility across both families: "
              f"{sum(row['accepted_invisible'] for row in rows)}."), "",
             "| Pair | Selected | Confirmed seeds | Correct / visible | Baseline | Matched control | False accepted |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['pair']} | {row['selected']} | {row['confirmed_seeds']} | "
                     f"{row['correct']}/{row['visible']} | {row['baseline_correct']} | "
                     f"{row['control_correct']} | {row['false_accepted']} |")
    maps = args.summary.parent / "paired_maps"
    maps.mkdir(exist_ok=True)
    for row in rows:
        texture = int(row["pair"].split("texture")[1].split("_")[0])
        seed = texture // 1000000 * 1000000
        baseline_path = Path(protocol["archive"]) / "affine_ncc" / f"windows_seed{seed}" / f"{row['pair']}.npz"
        with np.load(args.summary.parent / f"{row['pair']}.npz") as candidate, np.load(baseline_path) as baseline:
            centres = candidate["centres"]
            xs, ys = np.unique(centres[:, 0]), np.unique(centres[:, 1])
            visible = candidate["visible"].astype(bool)
            good = candidate["correct"].astype(bool)
            previous = baseline["icgn_correct_lt2px"].astype(bool)
            colour = np.full((len(centres), 3), (80, 80, 80), dtype=np.uint8)
            colour[previous & good] = (0, 180, 80)
            colour[previous & ~good] = (220, 0, 0)
            colour[~previous & good] = (0, 120, 255)
            colour[~visible] = (0, 0, 0)
            rgb = np.zeros((len(ys), len(xs), 3), dtype=np.uint8)
            rgb[np.searchsorted(ys, centres[:, 1]), np.searchsorted(xs, centres[:, 0])] = colour
            Image.fromarray(rgb).resize((len(xs) * 8, len(ys) * 8), Image.Resampling.NEAREST).save(
                maps / f"{row['pair']}.png")
    lines.extend(["", ("The saved JSON and NPZ outputs give the per-hypothesis counts, window states and provenance. "
                  "Reported wall times exclude generation of the input hypotheses and are not full pipeline times."),
                  "", ("Paired maps: green means both methods correct, red means only the baseline correct, "
                       "blue means only propagation correct, gray means neither correct, and black is invisible."), ""])
    (args.summary.parent / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

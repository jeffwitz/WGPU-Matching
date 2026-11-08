"""Controlled projection ablation on the exact archived observed pairs.

No detector is rerun and no analytic error enters the fit. This intentionally
omits common local geometric verification while retaining native acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from wgpu_matching import CorrespondenceSet, WGPUMatching
from wgpu_matching.images import load_gray

from qualification_visuals import assess_parameters, render_pair
from qualify_all_initializers import (
    LABELS,
    archive_model,
    cases,
    compact_diagnostics,
    fit,
    sha256,
    write_json,
)


def run(root, output, selected_cases):
    main_summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    methods = manifest["protocol"]["methods"]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with WGPUMatching(device_type="gpu") as matcher:
        for case in cases():
            if "all" not in selected_cases and case["name"] not in selected_cases:
                continue
            provenance = next(
                v for v in manifest["protocol"]["images"] if v["case"] == case["name"]
            )
            for name in ("reference", "target"):
                assert sha256(case[name]) == provenance[name + "_sha256"], (
                    "Image changed since main run"
                )
            reference, target = load_gray(case["reference"]), load_gray(case["target"])
            case["truth"] = provenance["analytic_truth"]
            first = next(r for r in main_summary["rows"] if r["case"] == case["name"])
            seeds = np.asarray([[v["x"], v["y"]] for v in first["assessment"]["per_seed"]], float)
            for method in methods:
                cache = root / case["name"] / method / "pair_cache.npz"
                if not cache.exists():
                    continue
                print(f"ABLATION {case['name']} / {method}", flush=True)
                with np.load(cache) as data:
                    observations = CorrespondenceSet(
                        data["source"],
                        data["target"],
                        case["native_size"],
                        method,
                        accepted=data["upstream_accepted"],
                        coordinate_mapping={
                            "source": "same archived direct pairs; common verification omitted"
                        },
                    )
                bases = (
                    ("affine", "gaussian_rbf")
                    if case["name"] == "tensile_60pct" or case["name"].startswith("adrien")
                    else ("affine",)
                )
                for basis in bases:
                    destination = output / case["name"] / method / basis
                    model, patches, error = None, [], None
                    try:
                        model, patches, stages = fit(matcher, observations, seeds, basis)
                    except (RuntimeError, ValueError) as exc:
                        error, stages = str(exc), {}
                    assessment = assess_parameters(
                        [p.parameters for p in patches] if model else [None] * len(seeds),
                        seeds,
                        case["truth"],
                        case["native_size"],
                    )
                    verdict = (
                        "NO ANALYTIC TRUTH"
                        if case["truth"] is None
                        else "PASS"
                        if assessment["pass_below_5px"]
                        else "FAIL"
                    )
                    visual = render_pair(
                        reference,
                        target,
                        model,
                        destination,
                        label=f"{case['name']} | {LABELS[method]} | RAW-PAIR ABLATION | {basis}",
                        verdict=verdict,
                        detail="No common local consensus; same original observations",
                        mask=case["mask"],
                        zoom=case["name"] == "tensile_60pct" or case["name"].startswith("adrien"),
                    )
                    if model:
                        archive_model(destination, model)
                    verified = next(
                        (
                            r
                            for r in main_summary["rows"]
                            if r["case"] == case["name"]
                            and r["method"] == method
                            and r["basis_requested"] == basis
                        ),
                        None,
                    )
                    record = {
                        "case": case["name"],
                        "method": method,
                        "basis_requested": basis,
                        "basis_effective": model.effective_basis if model else None,
                        "assessment": assessment,
                        "fit_error": error,
                        "diagnostics": compact_diagnostics(model),
                        "visualizations": visual,
                        "verified_assessment": verified["assessment"] if verified else None,
                        "cache_sha256": sha256(cache),
                        "source_protocol_sha256": manifest["protocol_sha256"],
                        "ablation_source_sha256": hashlib.sha256(
                            Path(__file__).read_bytes()
                        ).hexdigest(),
                        "stage_times_s": stages,
                        "timing_scope": "one projection call; no extraction or repeated end-to-end measurement",
                    }
                    write_json(destination / "result.json", record)
                    records.append(record)
                    write_json(output / "summary.json", {"records": records})
    lines = [
        "# Does robust projection alone reject the false matches?",
        "",
        "This controlled ablation uses the exact cached observations from the complete qualification. "
        "Native acceptance masks are retained; only the additional common local affine consensus is "
        "omitted. No detector is rerun, no truth filters pairs and the same spatial solver-capacity "
        "reduction applies. This is a diagnostic comparison, not a replacement for the main protocol.",
        "",
        "The affine ablation covers every case. Both basis requests are included for 60 percent "
        "and Adrien. On sparse detectors, the fixed local minimum of 8 candidates can reject "
        "otherwise useful pairs; this ablation also exposes that coverage cost rather than "
        "assuming the filter always improves the field.",
        "",
        "A raw robust field can converge numerically while being physically wrong. Results concern "
        "smooth initialization fields; they do not evaluate IC-GN convergence. The Adrien pair has no "
        "analytic error or accuracy verdict.",
        "",
        "| Case | Method | Requested → effective | Raw-projection correct <5 px / requested | With common verification | Raw RMSE (px) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in records:
        a, v = row["assessment"], row["verified_assessment"]
        raw = "—" if a["correct_lt5"] is None else f"{a['correct_lt5']}/{a['requested']}"
        good = (
            "—" if v is None or v["correct_lt5"] is None else f"{v['correct_lt5']}/{v['requested']}"
        )
        rmse = a.get("errors_px", {}).get("rmse")
        text_rmse = "—" if rmse is None else f"{rmse:.3f}"
        lines.append(
            f"| {row['case']} | {LABELS[row['method']]} | {row['basis_requested']} → "
            f"{row['basis_effective'] or 'none'} | {raw} | {good} | {text_rmse} |"
        )
    for row in records:
        folder = Path(row["case"]) / row["method"] / row["basis_requested"]
        a = row["assessment"]
        status = (
            "No analytic truth"
            if a["pass_below_5px"] is None
            else "Pass"
            if a["pass_below_5px"]
            else "Fail"
        )
        lines += [
            "",
            f"## {row['case']} / {LABELS[row['method']]} / {row['basis_requested']}",
            "",
            f"**{status}.** Effective basis: {row['basis_effective'] or 'none'}.",
            "",
            f"![Raw-pair projection, including failures]({folder.as_posix()}/comparison.gif)",
            "",
            *(
                [f"![Fixed native-pixel crop]({folder.as_posix()}/zoom_comparison.gif)", ""]
                if (output / folder / "zoom_comparison.gif").exists()
                else []
            ),
            f"[Full metrics and provenance]({folder.as_posix()}/result.json)",
            "",
        ]
    (output / "index.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["all"])
    args = parser.parse_args()
    run(args.run, args.output, args.cases)


if __name__ == "__main__":
    main()

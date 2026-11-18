"""Summarize status-only and optional residual-guided small-subset retries."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--guided-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=64001111)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = json.loads((args.status_dir / "summary.json").read_text())["cases"]
    guided = (
        {
            row["pair"]: row
            for row in json.loads((args.guided_dir / "summary.json").read_text())["cases"]
        }
        if args.guided_dir
        else {}
    )
    families = sorted({row["family"] for row in status})
    groups, incidents, pair_scores = [], [], []
    for row in status:
        if guided and row["pair"] not in guided:
            raise ValueError("guided and status-only input pairs differ")
        path = args.status_dir / f"{row['pair']}.npz"
        with np.load(path) as ref:
            truth, visible = ref["truth"], ref["visible"].astype(bool)
            results = {}
            for name, parameter_key, status_key in (
                ("64", "coarse_parameters", "coarse_accepted"),
                ("32", "stage32_parameters", "stage32_accepted"),
                ("16", "parameters", "accepted"),
            ):
                params, accepted = ref[parameter_key], ref[status_key].astype(bool)
                error = np.linalg.norm(params[:, [0, 3]] - truth[:, [0, 3]], axis=1)
                results[name] = (
                    visible & accepted & (error < 2),
                    visible & accepted & (error >= 2),
                )
            item = {
                "pair": row["pair"],
                "family": row["family"],
                "visible": int(visible.sum()),
                "status_metrics": row["metrics"],
                "attempts_32": row["attempts_32"],
                "attempts_16": row["attempts_16"],
                "accepted_32": row["accepted_32"],
                "accepted_16": row["accepted_16"],
                "retry_wall_s": row["retry_wall_s"],
                "stage_timings": row["stage_timings"],
                "status_new_correct": int((results["16"][0] & ~results["64"][0]).sum()),
                "status_lost_correct": int((results["64"][0] & ~results["16"][0]).sum()),
            }
            if guided:
                with np.load(args.guided_dir / path.name) as alt:
                    if not np.array_equal(visible, alt["visible"].astype(bool)) or not np.allclose(
                        truth, alt["truth"]
                    ):
                        raise ValueError("status-only and guided scoring truth differ")
                    error = np.linalg.norm(alt["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1)
                    accepted = alt["accepted"].astype(bool)
                    good = visible & accepted & (error < 2)
                    bad = visible & accepted & (error >= 2)
                    suspect = alt["suspect"].astype(bool)
                    old_bad, old_good = results["64"][1], results["64"][0]
                    item.update(
                        {
                            "guided_metrics": guided[row["pair"]]["metrics"],
                            "residual": guided[row["pair"]]["residual"],
                            "guided_new_correct": int((good & ~results["64"][0]).sum()),
                            "guided_lost_correct": int((results["64"][0] & ~good).sum()),
                            "guided_gained_vs_status": int((good & ~results["16"][0]).sum()),
                            "guided_lost_vs_status": int((results["16"][0] & ~good).sum()),
                            "flagged_bad_recovered": int((old_bad & suspect & good).sum()),
                            "flagged_bad_unresolved": int((old_bad & suspect & ~accepted).sum()),
                            "flagged_good_recovered": int((old_good & suspect & good).sum()),
                            "flagged_good_unresolved": int((old_good & suspect & ~accepted).sum()),
                            "guided_attempts_32": guided[row["pair"]]["attempts_32"],
                            "guided_attempts_16": guided[row["pair"]]["attempts_16"],
                            "guided_accepted_32": guided[row["pair"]]["accepted_32"],
                            "guided_accepted_16": guided[row["pair"]]["accepted_16"],
                            "guided_retry_wall_s": guided[row["pair"]]["retry_wall_s"],
                        }
                    )
                    for i in np.flatnonzero(bad):
                        incidents.append(
                            {
                                "pair": row["pair"],
                                "center_xy": alt["centres"][i].tolist(),
                                "error_px": float(error[i]),
                                "stage": int(alt["stage"][i]),
                                "suspect": bool(alt["suspect"][i]),
                                "source": int(alt["source"][i]),
                            }
                        )
            pair_scores.append(item)
    for family in families:
        cases = [item for item in pair_scores if item["family"] == family]
        agg = {
            "family": family,
            "pairs": len(cases),
            "visible": sum(item["visible"] for item in cases),
        }
        for stage in ("64", "32", "16"):
            agg[f"status_{stage}"] = {
                key: sum(item["status_metrics"][stage][key] for item in cases)
                for key in ("correct", "false", "accepted_visible", "accepted_invisible")
            }
        agg["status_attempts_16"] = sum(item["attempts_16"] for item in cases)
        agg["status_accepted_16"] = sum(item["accepted_16"] for item in cases)
        if guided:
            agg["guided_16"] = {
                key: sum(item["guided_metrics"]["16"][key] for item in cases)
                for key in ("correct", "false", "accepted_visible", "accepted_invisible")
            }
            agg["residual"] = {
                key: sum(item["residual"][key] for item in cases)
                for key in (
                    "true_bad",
                    "flagged_bad",
                    "true_good",
                    "flagged_good",
                    "unavailable_accepted",
                )
            }
            agg["guided_new_correct"] = sum(item["guided_new_correct"] for item in cases)
            agg["guided_lost_correct"] = sum(item["guided_lost_correct"] for item in cases)
            for key in (
                "guided_gained_vs_status",
                "guided_lost_vs_status",
                "flagged_bad_recovered",
                "flagged_bad_unresolved",
                "flagged_good_recovered",
                "flagged_good_unresolved",
            ):
                agg[key] = sum(item[key] for item in cases)
            agg["guided_attempts_16"] = sum(item["guided_attempts_16"] for item in cases)
            agg["guided_accepted_16"] = sum(item["guided_accepted_16"] for item in cases)
            random = np.random.default_rng(args.bootstrap_seed)
            samples = random.integers(0, len(cases), size=(10000, len(cases)))
            gains = np.array(
                [
                    item["guided_metrics"]["16"]["correct"]
                    - item["status_metrics"]["16"]["correct"]
                    for item in cases
                ],
                dtype=float,
            )
            visible = np.array([item["visible"] for item in cases], dtype=float)
            agg["paired_gain_95pct"] = np.quantile(
                gains[samples].sum(axis=1) / visible[samples].sum(axis=1), [0.025, 0.975]
            ).tolist()
        groups.append(agg)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "families": groups,
                "pairs": pair_scores,
                "false_acceptances": incidents,
                "bootstrap_seed": args.bootstrap_seed,
            },
            indent=2,
        )
        + "\n"
    )
    names = [g["family"] for g in groups]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 4))
    for offset, stage, label in (
        (-0.25, "64", "64 only"),
        (0, "32", "status: +32"),
        (0.25, "16", "status: +16"),
    ):
        ax.bar(
            x + offset,
            [100 * g[f"status_{stage}"]["correct"] / max(1, g["visible"]) for g in groups],
            width=0.24,
            label=label,
        )
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("Correct visible windows (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "coverage.png", dpi=150)
    plt.close(fig)
    lines = [
        "# Adaptive 64→32→16 pixel retry",
        "",
        (
            "All runs use the same original grid centers, image-derived coarse sources, and unchanged solver "
            "and consistency thresholds. Status-only preserves all accepted coarse states; a false accepted "
            "coarse state is not automatically detected. Truth is used only after inference. "
            "These archived data are retrospective unless the surrounding protocol explicitly states otherwise."
        ),
        "",
        "| Family | Visible | 64 correct / false | Status +32 correct / false | Status +16 correct / false |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        values = [
            f"{group[f'status_{stage}']['correct']} / {group[f'status_{stage}']['false']}"
            for stage in ("64", "32", "16")
        ]
        lines.append(f"| {group['family']} | {group['visible']} | {' | '.join(values)} |")
    if guided:
        lines.extend(
            [
                "",
                "| Family | Residual flags: wrong / wrong, correct / correct | Guided +16 correct / false | Gained / lost vs 64 | Gained / lost vs status +16 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for group in groups:
            r, g = group["residual"], group["guided_16"]
            lines.append(
                f"| {group['family']} | {r['flagged_bad']} / {r['true_bad']}, {r['flagged_good']} / {r['true_good']} | "
                f"{g['correct']} / {g['false']} | {group['guided_new_correct']} / {group['guided_lost_correct']} | "
                f"{group['guided_gained_vs_status']} / {group['guided_lost_vs_status']} |"
            )
        lines.extend(
            [
                "",
                "| Family | Flagged wrong recovered / unresolved | Flagged correct recovered / unresolved |",
                "| --- | ---: | ---: |",
            ]
        )
        for group in groups:
            lines.append(
                f"| {group['family']} | {group['flagged_bad_recovered']} / {group['flagged_bad_unresolved']} | "
                f"{group['flagged_good_recovered']} / {group['flagged_good_unresolved']} |"
            )
        lines.extend(
            [
                "",
                "| Pair | Visible | 64 correct / false | Status +16 correct / false | Guided +16 correct / false | Alerts wrong / correct | Residual / retry seconds |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in pair_scores:
            base, status_end, end, r = (
                item["status_metrics"]["64"],
                item["status_metrics"]["16"],
                item["guided_metrics"]["16"],
                item["residual"],
            )
            lines.append(
                f"| {item['pair']} | {item['visible']} | {base['correct']} / {base['false']} | "
                f"{status_end['correct']} / {status_end['false']} | {end['correct']} / {end['false']} | "
                f"{r['flagged_bad']} / {r['flagged_good']} | {r['wall_s_including_upload_and_readback']:.3f} / {item['guided_retry_wall_s']:.3f} |"
            )
        lines.extend(
            [
                "",
                (
                    "The residual classifier uses one threshold frozen on three development shear pairs. "
                    "Its image-only alerts are not a guarantee that the flagged windows are wrong or that "
                    "unflagged windows are correct. Each reported visible false acceptance is in `analysis.json`."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "",
            (
                "The original G3 criterion is unchanged. A small support can yield false accepted results, "
                "and the final 64-pixel model can still be biased by deformation within its support. "
                "The measured retry wall times include Python orchestration, solver creation, CPU-side "
                "reference preparation and host/device transfers; they are not pure GPU times or a complete pipeline benchmark."
            ),
            "",
            "![Coverage comparison](coverage.png)",
            "",
            (
                "Per-pair metrics, all timing records, GPU stage counts and "
                "window-level provenance are retained in `analysis.json` and the source NPZ files."
            ),
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

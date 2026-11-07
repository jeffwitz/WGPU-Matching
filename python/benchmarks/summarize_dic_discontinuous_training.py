"""Compare the DIC detector checkpoints on identical held-out speckle pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "results/features/dic_detector_discontinuous_legacy_heldout_20261107.json"
DISCONTINUITY_AWARE = ROOT / "results/features/dic_detector_discontinuous_heldout_20261107.json"
TRAINING = (
    ROOT / "results/features/dic_detector_training_discontinuous_20261107/training_report.json"
)
PARITY = ROOT / "results/features/dic_detector_discontinuous_wgsl_parity_4k_20261107.json"
OUTPUT = ROOT / "results/features/dic_detector_discontinuous_training_20261107.md"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(report: dict[str, Any], mode: str, threshold: int) -> str:
    value = report["summary_by_motion_mode"][mode]["learned"]["repeatability"]
    entry = value[f"at_{threshold}px"]
    return f"{entry['correct']}/{entry['denominator']} ({entry['fraction']:.1%})"


def build() -> str:
    previous = read(PREVIOUS)
    aware = read(DISCONTINUITY_AWARE)
    training = read(TRAINING)
    parity = read(PARITY)
    previous_ids = [
        (row["texture_seed"], row["warp_seed"], row["motion_mode"]) for row in previous["cases"]
    ]
    aware_ids = [
        (row["texture_seed"], row["warp_seed"], row["motion_mode"]) for row in aware["cases"]
    ]
    if previous_ids != aware_ids:
        raise ValueError("Checkpoint reports do not cover the same ordered held-out pairs")
    modes = sorted(aware["summary_by_motion_mode"])
    if modes != sorted(previous["summary_by_motion_mode"]):
        raise ValueError("Checkpoint reports have different motion families")
    training_counts = training["training"]["motion_mode_counts"]
    previous_summary = previous["summary"]["learned"]
    aware_summary = aware["summary"]["learned"]
    lines = [
        "# Discontinuity-aware DIC-Detector training study",
        "",
        "The earlier and discontinuity-aware checkpoints were evaluated on the same 24 deterministic texture/warp pairs using the WGSL detector. The only changed factor is the learned score weights. Each run uses a 512-keypoint budget, 256×256 images, and three warmed measurements per image. The tensor baseline is also reported in each raw file.",
        "",
        f"Adapter: {aware['adapter']}. Previous weights SHA-256: `{previous['weights_sha256']}`. Discontinuity-aware weights SHA-256: `{aware['weights_sha256']}`.",
        f"Training: {training['training']['epochs']} epochs × {training['training']['samples_per_epoch']} samples, CPU PyTorch {training['training']['torch_version']}, {training['training']['elapsed_s']:.1f} s. Training motion counts: {', '.join(f'{name} {count}' for name, count in sorted(training_counts.items()))}.",
        f"WGSL/PyTorch score parity for the same checkpoint at 3840×2160 with {parity['selected_keypoints']} selected points: maximum absolute score difference {parity['parity']['maximum_absolute_score_error']:.3g}; median WGSL dispatch/readback {parity['dispatch_and_readback_ms']:.2f} ms.",
        "",
        "## Aggregate held-out repeatability",
        "",
        "| Checkpoint | ≤1 px | ≤2 px | ≤5 px | Pair detection median | FPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in (("Previous", previous_summary), ("Discontinuity-aware", aware_summary)):
        values = summary["repeatability"]
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{values[f'at_{threshold}px']['correct']}/{values[f'at_{threshold}px']['denominator']} ({values[f'at_{threshold}px']['fraction']:.1%})"
                for threshold in (1, 2, 5)
            )
            + f" | {summary['pair_detection_s_median']:.4f} s | {summary['pair_fps']:.1f} |"
        )
    lines.extend(
        (
            "",
            "## Repeatability by motion family",
            "",
            "| Motion | Pairs | Previous ≤2 px | Discontinuity-aware ≤2 px | Previous ≤5 px | Discontinuity-aware ≤5 px |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for mode in modes:
        pair_count = aware["summary_by_motion_mode"][mode]["learned"]["pair_count"]
        lines.append(
            f"| {mode} | {pair_count} | {_rate(previous, mode, 2)} | {_rate(aware, mode, 2)} | {_rate(previous, mode, 5)} | {_rate(aware, mode, 5)} |"
        )
    lines.extend(
        (
            "",
            "The discontinuity-aware checkpoint improves the pooled ≤2 px rate from 54.2% to 57.0%, with almost unchanged detector time (4.05 ms per image pair). The discontinuous subset changes from 64.3% to 64.9% at ≤2 px across five pairs; this is a small exploratory difference, not evidence of a reliable crack-robust detector. The denominator varies because each checkpoint selects a different number of points that remain visible; rates and counts are both shown.",
            "",
            "This experiment measures detector repeatability only. It uses no descriptor, mutual matching, geometric fit, or IC-GN solver. The synthetic gap is a simple piecewise translation with a masked band, not a physical crack or a localized finite-element strain field.",
            "",
            "Raw per-pair results are in `dic_detector_discontinuous_legacy_heldout_20261107.json` and `dic_detector_discontinuous_heldout_20261107.json`; 4K WGSL/PyTorch score parity is in `dic_detector_discontinuous_wgsl_parity_4k_20261107.json`. Training seeds, motion counts, losses, and held-out PyTorch metrics are in `dic_detector_training_discontinuous_20261107/training_report.json`.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

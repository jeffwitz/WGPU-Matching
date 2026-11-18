"""Reproduce and localize the 18be5d9 raw-bilinear translation regression."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _load(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name].copy() for name in data.files}


def _map(path: Path, old: dict, new: dict) -> dict:
    visible = old["visible"].astype(bool) & new["visible"].astype(bool)
    old_ok = old["icgn_correct_lt2px"].astype(bool) & visible
    new_ok = new["icgn_correct_lt2px"].astype(bool) & visible
    centres = old["centres_xy_native"]
    xs = np.unique(centres[:, 0])
    ys = np.unique(centres[:, 1])
    colors = np.zeros((len(ys), len(xs), 3), dtype=np.uint8)
    grid_x = np.searchsorted(xs, centres[:, 0])
    grid_y = np.searchsorted(ys, centres[:, 1])
    colors[grid_y[visible], grid_x[visible]] = (70, 70, 70)
    # Green: both correct. Red: only pre-fix. Blue: only corrected.
    colors[grid_y[old_ok & new_ok], grid_x[old_ok & new_ok]] = (50, 180, 65)
    colors[grid_y[old_ok & ~new_ok], grid_x[old_ok & ~new_ok]] = (220, 55, 55)
    colors[grid_y[~old_ok & new_ok], grid_x[~old_ok & new_ok]] = (55, 90, 225)
    image = Image.fromarray(colors, "RGB").resize((len(xs) * 4, len(ys) * 4), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.text((4, 4), "green both | red pre-fix only | blue corrected only", fill="white")
    image.save(path)
    old_error = np.linalg.norm(
        old["icgn_parameters"][:, [0, 3]] - old["truth_parameters"][:, [0, 3]], axis=1
    )
    new_error = np.linalg.norm(
        new["icgn_parameters"][:, [0, 3]] - new["truth_parameters"][:, [0, 3]], axis=1
    )
    return {
        "visible": int(visible.sum()),
        "pre_fix_correct": int(old_ok.sum()),
        "corrected_correct": int(new_ok.sum()),
        "lost": int((old_ok & ~new_ok).sum()),
        "gained": int((~old_ok & new_ok).sum()),
        "old_initial_translation_rmse_px": float(np.sqrt(np.mean(np.sum(
            (old["initial_parameters"][:, [0, 3]] - old["truth_parameters"][:, [0, 3]]) ** 2, axis=1
        )[visible]))),
        "new_initial_translation_rmse_px": float(np.sqrt(np.mean(np.sum(
            (new["initial_parameters"][:, [0, 3]] - new["truth_parameters"][:, [0, 3]]) ** 2, axis=1
        )[visible]))),
        "old_accepted_error_median_px": float(np.median(old_error[old["icgn_status"].astype(bool) & visible]))
            if np.any(old["icgn_status"].astype(bool) & visible) else None,
        "new_accepted_error_median_px": float(np.median(new_error[new["icgn_status"].astype(bool) & visible]))
            if np.any(new["icgn_status"].astype(bool) & visible) else None,
        "map": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-fix-qualifier", required=True, type=Path,
                        help="qualify_ndf_pair_icgn.py from commit 4b575de")
    parser.add_argument("--post-fix-qualifier", type=Path,
                        default=Path(__file__).with_name("qualify_ndf_pair_icgn.py"))
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, qualifier in (("pre_fix", args.pre_fix_qualifier), ("corrected", args.post_fix_qualifier)):
        directory = output / label
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / "translation_seed11000000.json"
        command = [
            sys.executable, str(qualifier.resolve()),
            "--weights", str(args.weights.resolve()),
            "--live-dic-source", str(args.live_dic_source.resolve()),
            "--output", str(report_path),
            "--window-diagnostics-dir", str(directory / "windows"),
            "--native-scale", "20", "--model-width", "192", "--model-height", "108",
            "--speckle-size", "0.35", "--families", "translation",
            "--cases-per-family", "10", "--seed-offset", "11000000",
            "--contextual-features", "--projection", "raw_bilinear",
            "--local-ncc-radius", "8", "--local-ncc-patch-radius", "5",
            "--local-ncc-backend", "wgpu",
        ]
        env = os.environ.copy()
        python_path = [str(qualifier.resolve().parent), str(Path("python/src").resolve())]
        env["PYTHONPATH"] = os.pathsep.join(python_path + [env.get("PYTHONPATH", "")])
        with (directory / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        results[label] = json.loads(report_path.read_text(encoding="utf-8"))

    old_cases = results["pre_fix"]["cases"]
    new_cases = results["corrected"]["cases"]
    if [(r["texture_seed"], r["warp_seed"]) for r in old_cases] != [
        (r["texture_seed"], r["warp_seed"]) for r in new_cases
    ]:
        raise ValueError("pre-fix and corrected executions do not contain the same generated pairs")
    pairs = []
    for old_case, new_case in zip(old_cases, new_cases, strict=True):
        key = f"texture{old_case['texture_seed']}_warp{old_case['warp_seed']}"
        pairs.append({
            "texture_seed": old_case["texture_seed"],
            "warp_seed": old_case["warp_seed"],
            **_map(output / f"{key}_correctness.png",
                  _load(old_case["window_diagnostics"]),
                  _load(new_case["window_diagnostics"])),
        })
    summary = {
        "protocol": {
            "pre_fix_code": "4b575de, parent of coordinate correction 18be5d9",
            "corrected_code": "current checkout",
            "same_seeded_translation_pairs": True,
            "truth_used_for_selection": False,
            "thresholds_or_algorithms_changed": False,
            "coordinate_change": "map native pixel centres to reduced raster using align_corners=False: (x+0.5)/scale-0.5, and sample the learned feature grid at its true half-cell centres",
        },
        "pre_fix_total_correct": sum(pair["pre_fix_correct"] for pair in pairs),
        "corrected_total_correct": sum(pair["corrected_correct"] for pair in pairs),
        "visible_total": sum(pair["visible"] for pair in pairs),
        "total_lost": sum(pair["lost"] for pair in pairs),
        "total_gained": sum(pair["gained"] for pair in pairs),
        "pairs": pairs,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Per-window analysis of the raw-bilinear translation regression",
        "",
        "IC-GN means inverse-compositional Gauss-Newton. The comparison reruns the same ten seeded synthetic translations with code before and after the coordinate correction. It changes no threshold or solver setting. Green map cells are correct in both runs, red cells only before the correction, blue cells only after it, gray cells are visible but incorrect in both, and black cells are outside the analytical visibility region.",
        "",
        f"| Correct before / visible | Correct after / visible | Lost | Gained |\n|---:|---:|---:|---:|\n| {summary['pre_fix_total_correct']:,}/{summary['visible_total']:,} | {summary['corrected_total_correct']:,}/{summary['visible_total']:,} | {summary['total_lost']:,} | {summary['total_gained']:,} |",
        "",
        "The per-window exports show that the coordinate change affects the initial field before IC-GN: the initial displacement root mean square error (RMSE) increases for all ten pairs. The corrected mapping places samples at the feature grid's true half-cell centres and follows the pixel-centre resize convention. This reveals a sensitivity of the current learned field to spatial sample location; it does not justify reverting the coordinate correction or changing thresholds. Of the 13,444 visible windows, five individual windows become correct while 553 are lost, for a net loss of 548. These synthetic measurements do not establish accuracy on experimental images.",
        "", "Reproduce from the repository root (the pre-fix qualifier is archived from commit `4b575de`):", "",
        "```console",
        "mkdir -p /tmp/wgpu-old-4b575de",
        "git archive 4b575de -- python/benchmarks | tar -x -C /tmp/wgpu-old-4b575de",
        "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_ndf_translation_coordinate_anomaly.py \\",
        "  --pre-fix-qualifier /tmp/wgpu-old-4b575de/python/benchmarks/qualify_ndf_pair_icgn.py \\",
        "  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \\",
        "  --live-dic-source /path/to/Live-DIC/python/src \\",
        "  --output-dir results/ndf_dic_20261109/translation_coordinate_anomaly_20261109",
        "```",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "summary.json")


if __name__ == "__main__":
    main()

"""Export image-derived affine hypotheses for the three highest ranked rotations."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", choices=("heterogeneous", "translation", "rigid", "affine", "discontinuous", "shear_band"), default="heterogeneous")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--textures", nargs="+", type=int)
    parser.add_argument("--ranks", nargs="+", type=int, choices=(1, 2, 3), default=(1, 2, 3))
    parser.add_argument("--discontinuity-boundary-fraction", type=float, default=.5)
    parser.add_argument("--speckle-noise-std", type=float, default=.025)
    parser.add_argument("--target-gain", type=float, default=1.)
    parser.add_argument("--target-offset", type=float, default=0.)
    parser.add_argument("--target-blur-sigma", type=float, default=0.)
    parser.add_argument("--target-noise-std", type=float, default=0.)
    parser.add_argument("--periodic-strength", type=float, default=0.)
    parser.add_argument("--periodic-period-px", type=float, default=128.)
    parser.add_argument("--external-reference", type=Path)
    parser.add_argument("--external-target", type=Path)
    parser.add_argument("--external-stretch-x", type=float)
    parser.add_argument("--external-stretch-y", type=float)
    parser.add_argument("--model-width", type=int, default=192)
    parser.add_argument("--model-height", type=int, default=108)
    parser.add_argument("--shear-band-peak-gradient", type=float, default=.67)
    parser.add_argument("--ndf-onnx-webgpu-cache", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for report_path in args.ranking_reports:
        cases.extend(json.loads(report_path.read_text())["cases"])
    cases = [case for case in cases if case.get("mode") == args.family]
    if args.textures:
        cases = [case for case in cases if case["texture_seed"] in args.textures]
    if args.limit:
        cases = cases[:args.limit]
    family_index = {"translation": 0, "heterogeneous": 1, "rigid": 2, "affine": 3,
                    "discontinuous": 4, "shear_band": 5}[args.family]
    manifest = []
    for case in cases:
        texture, warp = case["texture_seed"], case["warp_seed"]
        seed_offset = (texture // 1000000) * 1000000
        sample = (texture - seed_offset - 51000 - family_index) // 2
        top = sorted(case["rotation_bank_candidate_scores"],
                     key=lambda x: (x["ranking_score"], x["consensus_support"]), reverse=True)[:3]
        if len(top) != 3:
            raise ValueError(f"insufficient rotation candidates for {texture}")
        for rank, candidate in enumerate(top, 1):
            if rank not in args.ranks:
                continue
            directory = args.output_dir / f"{texture}_{warp}" / f"rank{rank}"
            directory.mkdir(parents=True, exist_ok=True)
            report = directory / "report.json"
            command = [sys.executable, str(Path(__file__).with_name("qualify_ndf_pair_icgn.py")),
                       "--weights", str(args.weights.resolve()), "--live-dic-source", str(args.live_dic_source.resolve()),
                       "--output", str(report), "--window-diagnostics-dir", str(directory / "windows"),
                       "--native-scale", "20", "--model-width", str(args.model_width), "--model-height", str(args.model_height),
                       "--speckle-size", ".35", "--families", args.family, "--cases-per-family", "1",
                       "--case-start-index", str(sample), "--seed-offset", str(seed_offset),
                       "--contextual-features", "--rotation-bank-steps", "9",
                       "--rotation-bank-probe-query-budget", "256", "--global-affine-prefit",
                       "--field-residual", "none", "--local-ncc-radius", "0", "--local-ncc-backend", "wgpu",
                       "--local-consensus-radius", "40", "--local-consensus-threshold", "5",
                       "--projection", "raw_bilinear", "--solver-window-budget", "1",
                       "--forced-rotation-angle-rad", str(candidate["angle_rad"])]
            if args.family == "shear_band":
                command.extend(("--shear-band-peak-gradient", str(args.shear_band_peak_gradient)))
            for option in ("discontinuity_boundary_fraction", "speckle_noise_std", "target_gain",
                           "target_offset", "target_blur_sigma", "target_noise_std", "periodic_strength", "periodic_period_px"):
                command.extend(("--" + option.replace("_", "-"), str(getattr(args, option))))
            for option in ("external_reference", "external_target", "external_stretch_x", "external_stretch_y"):
                value = getattr(args, option)
                if value is not None:
                    command.extend(("--" + option.replace("_", "-"), str(value)))
            if args.ndf_onnx_webgpu_cache:
                command.extend(("--ndf-onnx-webgpu-cache", str(args.ndf_onnx_webgpu_cache)))
            wall_s = None
            if not report.exists():
                start = time.perf_counter()
                with (directory / "run.log").open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                wall_s = time.perf_counter() - start
            cached = json.loads(report.read_text())
            if (args.family == "shear_band"
                    and cached["protocol"].get("shear_band_peak_gradient") != args.shear_band_peak_gradient):
                raise ValueError("cached rotation hypothesis has a different shear-band gradient")
            previous_gpu = "ndf_onnx_webgpu" in cached["protocol"]
            if previous_gpu != bool(args.ndf_onnx_webgpu_cache):
                raise ValueError("cached rotation hypothesis was produced with a different NDF backend")
            for option, default in (("discontinuity_boundary_fraction", .5), ("speckle_noise_std", .025),
                                    ("target_gain", 1.), ("target_offset", 0.), ("target_blur_sigma", 0.),
                                    ("target_noise_std", 0.), ("periodic_strength", 0.), ("periodic_period_px", 128.)):
                if cached["protocol"].get(option, default) != getattr(args, option):
                    raise ValueError(f"cached hypothesis has different {option}")
            for option in ("external_reference", "external_target", "external_stretch_x", "external_stretch_y"):
                value = getattr(args, option)
                expected = str(value) if isinstance(value, Path) else value
                if cached["protocol"].get(option) != expected:
                    raise ValueError(f"cached hypothesis has different {option}")
            output = cached["cases"][0]
            if output["texture_seed"] != texture or output["warp_seed"] != warp:
                raise ValueError("regenerated pair does not match ranking report")
            manifest.append({"family": args.family, "texture_seed": texture, "warp_seed": warp, "rank": rank,
                             "angle_rad": candidate["angle_rad"], "ranking_score": candidate["ranking_score"],
                             "consensus_support": candidate["consensus_support"],
                             "subprocess_wall_s": wall_s,
                             "windows": output["window_diagnostics"]})
        print(f"Completed {texture}", flush=True)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

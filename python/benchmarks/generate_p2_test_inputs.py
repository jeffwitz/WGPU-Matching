"""Generate frozen-protocol, held-out image-derived P2 hypothesis inputs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-parameters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-offsets", type=int, nargs="+", default=(21000000, 22000000))
    parser.add_argument("--retain-translation-affine", action="store_true")
    parser.add_argument("--shared-prefit", action="store_true",
                        help="experimentally export affine-only and affine-plus-NCC states from one inference")
    parser.add_argument("--export-solver-images", action="store_true",
                        help="optionally cache exact quantized images for the verification script")
    parser.add_argument("--families", nargs="+", choices=("translation", "heterogeneous", "rigid", "affine", "discontinuous", "shear_band"),
                        default=("translation", "heterogeneous"))
    parser.add_argument("--cases-per-family", type=int, default=10)
    parser.add_argument("--case-start-index", type=int, default=0)
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
    parser.add_argument("--ndf-onnx-webgpu-cache", type=Path,
                        help="optional fixed-shape ONNX Runtime WebGPU matcher graphs")
    args = parser.parse_args()
    if args.export_solver_images and not args.shared_prefit:
        parser.error("--export-solver-images requires --shared-prefit")
    if not args.frozen_parameters.is_file():
        parser.error("development parameters must be frozen before test generation")
    qualifier = Path(__file__).with_name("qualify_ndf_pair_icgn.py")
    variants = (("affine_ncc", 8),) if args.shared_prefit else (("affine_only", 0), ("affine_ncc", 8))
    for variant, radius in variants:
        for seed in args.seed_offsets:
            directory = args.output_dir / variant
            directory.mkdir(parents=True, exist_ok=True)
            report = directory / f"combined_seed{seed}.json"
            command = [sys.executable, str(qualifier), "--weights", str(args.weights.resolve()),
                       "--live-dic-source", str(args.live_dic_source.resolve()), "--output", str(report),
                       "--window-diagnostics-dir", str(directory / f"windows_seed{seed}"),
                       "--native-scale", "20", "--model-width", str(args.model_width), "--model-height", str(args.model_height),
                       "--speckle-size", ".35", "--families", *args.families,
                       "--cases-per-family", str(args.cases_per_family), "--case-start-index", str(args.case_start_index),
                       "--seed-offset", str(seed), "--contextual-features",
                       "--rotation-bank-steps", "9", "--rotation-bank-probe-query-budget", "256",
                       "--global-affine-prefit", "--field-residual", "none", "--local-ncc-radius", str(radius),
                       "--local-ncc-patch-radius", "4", "--local-ncc-backend", "wgpu",
                       "--local-ncc-geometry", "translation", "--local-consensus-radius", "40",
                       "--local-consensus-threshold", "5", "--projection", "raw_bilinear"]
            if "shear_band" in args.families:
                command.extend(("--shear-band-peak-gradient", str(args.shear_band_peak_gradient)))
            for option in ("discontinuity_boundary_fraction", "speckle_noise_std", "target_gain",
                           "target_offset", "target_blur_sigma", "target_noise_std", "periodic_strength", "periodic_period_px"):
                command.extend(("--" + option.replace("_", "-"), str(getattr(args, option))))
            if args.retain_translation_affine:
                command.append("--retain-translation-affine")
            if args.ndf_onnx_webgpu_cache:
                command.extend(("--ndf-onnx-webgpu-cache", str(args.ndf_onnx_webgpu_cache)))
            if args.shared_prefit:
                command.append("--export-pre-ncc")
            if args.export_solver_images:
                command.extend(("--solver-image-cache-dir", str(directory / f"solver_images_seed{seed}")))
            for option in ("external_reference", "external_target", "external_stretch_x", "external_stretch_y"):
                value = getattr(args, option)
                if value is not None:
                    command.extend(("--" + option.replace("_", "-"), str(value)))
            if report.exists():
                previous = json.loads(report.read_text())
                previous_gpu = "ndf_onnx_webgpu" in previous["protocol"]
                if previous_gpu != bool(args.ndf_onnx_webgpu_cache):
                    raise ValueError("cached prefit was produced with a different NDF backend")
                if ("shear_band" in args.families
                        and previous["protocol"].get("shear_band_peak_gradient") != args.shear_band_peak_gradient):
                    raise ValueError("cached prefit has a different shear-band gradient")
            if not report.exists():
                with (directory / f"seed{seed}.log").open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            elif args.shared_prefit:
                import numpy as np
                files = list((directory / f"windows_seed{seed}").glob("*.npz"))
                if not files:
                    raise ValueError("cached shared-prefit report has no window archives")
                for path in files:
                    with np.load(path) as data:
                        if "pre_ncc_initial_parameters" not in data:
                            raise ValueError("cached report does not contain shared pre-NCC inputs")
                if args.export_solver_images:
                    from solver_image_cache import read_solver_images

                    sources = (args.external_reference, args.external_target) if args.external_reference else None
                    for path in files:
                        read_solver_images(directory / f"solver_images_seed{seed}", path, source_files=sources)
            print(f"Completed {variant} seed {seed}", flush=True)


if __name__ == "__main__":
    main()

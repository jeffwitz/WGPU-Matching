"""Run the frozen held-out image-only seed qualification from prepared inputs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command):
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--frozen-parameters", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-offsets", type=int, nargs="+", default=(21000000, 22000000))
    parser.add_argument("--families", nargs="+", choices=("heterogeneous", "translation", "rigid", "affine", "discontinuous", "shear_band"),
                        default=("heterogeneous", "translation"))
    parser.add_argument("--cases-per-family", type=int, default=10)
    parser.add_argument("--shear-band-peak-gradient", type=float, default=.67)
    parser.add_argument("--skip-g2-summary", action="store_true")
    parser.add_argument("--lazy-rotations", action="store_true",
                        help="experimentally defer forced rotations until verified anchors are insufficient")
    parser.add_argument("--ndf-onnx-webgpu-cache", type=Path,
                        help="use ONNX Runtime WebGPU when generating rotation hypotheses")
    parser.add_argument("--use-cached-solver-images", action="store_true",
                        help="reuse the verified quantized pixels exported by the shared-prefit preparation")
    parser.add_argument("--batch-anchor-confirmation", action="store_true",
                        help="experimentally batch neighbouring anchor-confirmation solves")
    parser.add_argument("--gpu-resident-selection", action="store_true",
                        help="keep the selected sparse field on GPU")
    parser.add_argument("--gpu-hypothesis-selection", action="store_true",
                        help="also select sparse hypotheses on GPU")
    parser.add_argument("--gpu-anchor-frontier", action="store_true",
                        help="also confirm sparse anchors on GPU")
    parser.add_argument("--gpu-frontier", action="store_true",
                        help="experimentally propagate on the existing solver's WebGPU device")
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
    parser.add_argument("--roi-mask", type=Path)
    parser.add_argument("--model-width", type=int, default=192)
    parser.add_argument("--model-height", type=int, default=108)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_parameters.read_text())["selected"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).parent
    perturbation_args = []
    for option in ("discontinuity_boundary_fraction", "speckle_noise_std", "target_gain",
                   "target_offset", "target_blur_sigma", "target_noise_std", "periodic_strength", "periodic_period_px"):
        perturbation_args.extend(("--" + option.replace("_", "-"), str(getattr(args, option))))
    external_args = []
    for option in ("external_reference", "external_target", "external_stretch_x", "external_stretch_y"):
        value = getattr(args, option)
        if value is not None:
            external_args.extend(("--" + option.replace("_", "-"), str(value)))
    rotation_args = []
    if args.lazy_rotations:
        for seed in args.seed_offsets:
            for family in args.families:
                windows = list((args.inputs / "affine_ncc" / f"windows_seed{seed}").glob(f"{family}_*.npz"))
                if len(windows) != args.cases_per_family:
                    parser.error(f"expected {args.cases_per_family} {family} cases for seed group {seed}, found {len(windows)}")
        rotation_args = ["--lazy-rotation-weights", str(args.weights),
                         "--lazy-rotation-output", str(args.output_dir / "lazy_rotations")]
    else:
        manifest = []
        for family in args.families:
            directory = args.output_dir / f"rotation_{family}"
            reports = [args.inputs / "affine_ncc" / f"combined_seed{seed}.json"
                       for seed in args.seed_offsets]
            if not all(path.is_file() for path in reports):
                parser.error("independent input reports are incomplete")
            ndf_args = (["--ndf-onnx-webgpu-cache", str(args.ndf_onnx_webgpu_cache)]
                        if args.ndf_onnx_webgpu_cache else [])
            run([sys.executable, str(scripts / "generate_rotation_seed_hypotheses.py"),
                 "--ranking-reports", *map(str, reports), "--family", family,
                 "--weights", str(args.weights), "--live-dic-source", str(args.live_dic_source),
                 "--output-dir", str(directory), "--model-width", str(args.model_width),
                 "--model-height", str(args.model_height), "--shear-band-peak-gradient", str(args.shear_band_peak_gradient),
                 *perturbation_args, *external_args, *ndf_args])
            manifest.extend(json.loads((directory / "manifest.json").read_text()))
        manifest_path = args.output_dir / "rotation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        expected = 3 * len(args.seed_offsets) * len(args.families) * args.cases_per_family
        if len(manifest) != expected:
            raise ValueError(f"expected {expected} rotation hypotheses, found {len(manifest)}")
        rotation_args = ["--rotation-manifest", str(manifest_path)]
    evaluation = args.output_dir / "evaluation"
    pilot_external_args = []
    if args.ndf_onnx_webgpu_cache:
        pilot_external_args.extend(("--ndf-onnx-webgpu-cache", str(args.ndf_onnx_webgpu_cache)))
    if args.use_cached_solver_images:
        pilot_external_args.append("--use-cached-solver-images")
    if args.batch_anchor_confirmation:
        pilot_external_args.append("--batch-anchor-confirmation")
    if args.gpu_frontier:
        pilot_external_args.append("--gpu-frontier")
    if args.gpu_anchor_frontier:
        pilot_external_args.append("--gpu-anchor-frontier")
    if args.gpu_hypothesis_selection:
        pilot_external_args.append("--gpu-hypothesis-selection")
    if args.gpu_resident_selection:
        pilot_external_args.append("--gpu-resident-selection")
    if not args.gpu_frontier:
        pilot_external_args.append("--same-budget-control")
    for option in ("external_reference", "external_target", "roi_mask"):
        value = getattr(args, option)
        if value is not None:
            pilot_external_args.extend(("--" + option.replace("_", "-"), str(value)))
    run([sys.executable, str(scripts / "qualify_image_seed_pilot.py"), "--archive", str(args.inputs),
         *rotation_args, "--live-dic-source", str(args.live_dic_source),
         "--output-dir", str(evaluation), "--frozen-parameters", str(args.frozen_parameters),
         "--tau-u", str(frozen["tau_u"]), "--tau-g", str(frozen["tau_g"]),
         "--sparse-stride", str(frozen["sparse_stride"]),
         "--minimum-anchors", str(frozen["minimum_anchors"]),
         "--shear-band-peak-gradient", str(args.shear_band_peak_gradient),
         "--escalation", "--local-affine-escalation",
         "--families", *args.families, "--seed-offsets", *map(str, args.seed_offsets),
         *perturbation_args, *pilot_external_args])
    if not args.skip_g2_summary:
        run([sys.executable, str(scripts / "summarize_image_seed_test.py"),
             str(evaluation / "summary.json"), "--frozen-parameters", str(args.frozen_parameters)])


if __name__ == "__main__":
    main()

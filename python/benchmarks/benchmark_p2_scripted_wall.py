"""Measure a complete scripted one-pair P2 run including diagnostic overhead."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--frozen-parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=25000000)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).parent
    frozen = json.loads(args.frozen_parameters.read_text())["selected"]
    texture = args.seed_offset + 51001 + 2 * args.sample
    measurements = []
    for repetition in range(args.repetitions + 1):
        folder = args.output_dir / f"run{repetition}"
        folder.mkdir(parents=True, exist_ok=True)
        starts = time.perf_counter()
        for variant, radius in (("affine_only", 0), ("affine_ncc", 8)):
            directory = folder / variant
            directory.mkdir(exist_ok=True)
            command = [sys.executable, str(scripts / "qualify_ndf_pair_icgn.py"),
                       "--weights", str(args.weights), "--live-dic-source", str(args.live_dic_source),
                       "--output", str(directory / "report.json"),
                       "--window-diagnostics-dir", str(directory / f"windows_seed{args.seed_offset}"),
                       "--native-scale", "20", "--model-width", "192", "--model-height", "108",
                       "--speckle-size", ".35", "--families", "heterogeneous", "--cases-per-family", "1",
                       "--case-start-index", str(args.sample), "--seed-offset", str(args.seed_offset),
                       "--contextual-features", "--rotation-bank-steps", "9", "--rotation-bank-probe-query-budget", "256",
                       "--global-affine-prefit", "--field-residual", "none", "--retain-translation-affine",
                       "--local-ncc-radius", str(radius), "--local-ncc-patch-radius", "4",
                       "--local-ncc-backend", "wgpu", "--local-ncc-geometry", "translation",
                       "--local-consensus-radius", "40", "--local-consensus-threshold", "5", "--projection", "raw_bilinear"]
            with (directory / "run.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        after_prefits = time.perf_counter()
        rotations = folder / "rotations"
        subprocess.run([sys.executable, str(scripts / "generate_rotation_seed_hypotheses.py"),
                        "--ranking-reports", str(folder / "affine_ncc/report.json"),
                        "--weights", str(args.weights), "--live-dic-source", str(args.live_dic_source),
                        "--output-dir", str(rotations)], check=True, stdout=subprocess.DEVNULL)
        after_rotations = time.perf_counter()
        subprocess.run([sys.executable, str(scripts / "qualify_image_seed_pilot.py"),
                        "--archive", str(folder), "--rotation-manifest", str(rotations / "manifest.json"),
                        "--live-dic-source", str(args.live_dic_source), "--output-dir", str(folder / "evaluation"),
                        "--frozen-parameters", str(args.frozen_parameters), "--tau-u", str(frozen["tau_u"]),
                        "--tau-g", str(frozen["tau_g"]), "--sparse-stride", str(frozen["sparse_stride"]),
                        "--minimum-anchors", str(frozen["minimum_anchors"]), "--escalation", "--local-affine-escalation"],
                       check=True, stdout=subprocess.DEVNULL)
        done = time.perf_counter()
        measurements.append({"warmup": repetition == 0, "prefits_wall_s": after_prefits - starts,
                             "rotation_hypotheses_wall_s": after_rotations - after_prefits,
                             "verification_propagation_wall_s": done - after_rotations,
                             "full_script_wall_s": done - starts})
        print(f"Completed timing run {repetition}, texture {texture}", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"measurements": measurements,
        "scope": "Includes repeated synthetic-image generation, process/model setup, oracle diagnostics and disk output; not production pipeline latency."}, indent=2) + "\n")


if __name__ == "__main__":
    main()

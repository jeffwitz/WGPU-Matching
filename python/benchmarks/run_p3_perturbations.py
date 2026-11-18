"""Run fixed P3 perturbation cohorts with the frozen image-only initializer."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--frozen-parameters", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cohorts", nargs="+", choices=("shifted_opening", "doubled_noise", "photometric", "defocus", "camera_noise", "periodic"))
    args = parser.parse_args()
    scripts = Path(__file__).parent
    cohorts = [("shifted_opening", 27000000, "discontinuous", ["--discontinuity-boundary-fraction", ".503125"]),
               ("doubled_noise", 28000000, "heterogeneous", ["--speckle-noise-std", ".05"]),
               ("photometric", 29000000, "heterogeneous", ["--target-gain", "1.2", "--target-offset", "-.05"]),
               ("defocus", 30000000, "heterogeneous", ["--target-blur-sigma", "1.2"]),
               ("camera_noise", 31000000, "heterogeneous", ["--target-noise-std", ".05"]),
               ("periodic", 32000000, "heterogeneous", ["--periodic-strength", ".9", "--periodic-period-px", "128"])]
    for name, seed, family, parameters in cohorts:
        if args.cohorts and name not in args.cohorts:
            continue
        inputs = args.output_dir / name / "inputs"
        output = args.output_dir / name / "qualification"
        common = ["--weights", str(args.weights), "--live-dic-source", str(args.live_dic_source),
                  "--frozen-parameters", str(args.frozen_parameters)]
        subprocess.run([sys.executable, str(scripts / "generate_p2_test_inputs.py"), *common,
                        "--output-dir", str(inputs), "--seed-offsets", str(seed),
                        "--families", family, "--cases-per-family", "5", "--retain-translation-affine",
                        *parameters], check=True)
        subprocess.run([sys.executable, str(scripts / "run_image_seed_test.py"), *common,
                        "--inputs", str(inputs), "--output-dir", str(output), "--seed-offsets", str(seed),
                        "--families", family, "--cases-per-family", "5", "--skip-g2-summary", *parameters],
                       check=True)
        print(f"Completed cohort {name}", flush=True)


if __name__ == "__main__":
    main()

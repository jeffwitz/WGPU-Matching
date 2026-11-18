"""Direct, independent reference-to-target qualification of supplied image sets.

No displacement files or preceding target states enter inference. Truth association
is a separate audit: an accepted solve is not asserted to be a correct measurement.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/home/jeff/CNRS/DIC/HudgeStrains"))
    parser.add_argument("--output", type=Path, default=Path("results/hudge_strains_20261114"))
    parser.add_argument("--family", choices=["FEA", "ClosedFormExpression"], required=True)
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    scripts = Path(__file__).parent
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    weights = root / "results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt"
    frozen = root / "results/seed_propagation_20261111/p2_calibration/frozen_parameters.json"
    solver = Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src")
    # Isotropic model rasters; both remain bounded below 6000 feature tokens.
    size = (192, 192) if args.family == "FEA" else (128, 704)
    for level in args.levels:
        name = f"Img_wa_g{level:03d}.tif" if args.family == "FEA" else f"Img_G{level:03d}.tif"
        directory = args.output / args.family / f"level_{level:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "completed.json").exists():
            continue
        common = ["--weights", str(weights), "--live-dic-source", str(solver),
                  "--frozen-parameters", str(frozen), "--seed-offsets", "70000000",
                  "--families", "heterogeneous", "--cases-per-family", "1",
                  "--model-width", str(size[0]), "--model-height", str(size[1]),
                  "--external-reference", str(args.data / args.family / "Img_F.tif"),
                  "--external-target", str(args.data / args.family / name)]
        commands = [
            [sys.executable, str(scripts / "generate_p2_test_inputs.py"), *common,
             "--output-dir", str(directory / "inputs"), "--retain-translation-affine",
             "--shared-prefit", "--export-solver-images"],
            [sys.executable, str(scripts / "run_image_seed_test.py"), *common,
             "--inputs", str(directory / "inputs"), "--output-dir", str(directory / "qualification"),
             "--skip-g2-summary", "--lazy-rotations", "--use-cached-solver-images",
             "--batch-anchor-confirmation", "--roi-mask",
             str(args.output / f"{args.family}_reference_mask.png")],
        ]
        (directory / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        timings = []
        try:
            for i, command in enumerate(commands):
                started = time.perf_counter()
                with (directory / f"stage{i}.log").open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=1800, env={**os.environ, "OMP_NUM_THREADS": "4"})
                timings.append(time.perf_counter() - started)
            report = json.loads((directory / "qualification/evaluation/summary.json").read_text())
            record = {"family": args.family, "level": level, "model_size": size,
                      "reference": common[-3], "target": common[-1],
                      "stage_wall_seconds": timings, "row": report["rows"][0],
                      "truth_status": "unassociated; acceptance only",
                      "temporal_state_reused": False}
            (directory / "completed.json").write_text(json.dumps(record, indent=2) + "\n")
            print(args.family, level, timings, flush=True)
        except (subprocess.SubprocessError, OSError, KeyError) as error:
            (directory / "failure.json").write_text(json.dumps({"error": str(error), "timings": timings}, indent=2))
            print(args.family, level, "FAILED", error, flush=True)


if __name__ == "__main__":
    main()

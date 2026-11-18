"""Qualify frozen image-only seeds on tensile frames and the masked Adrien pair."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--frozen-parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensile-dir", type=Path, default=Path("results/uniaxial/tensile_60pct"))
    parser.add_argument("--frames-dir", type=Path, default=Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile"))
    parser.add_argument("--adrien-dir", type=Path, default=Path("/home/jeff/CNRS/DIC/AdrienVariational"))
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    scripts = Path(__file__).parent
    manifest = json.loads((args.tensile_dir / "manifest.json").read_text())
    cases = [{"name": f"tensile_{round(100*s['requested_engineering_strain']):02d}",
              "reference": args.tensile_dir / "reference.png", "target": args.tensile_dir / s["file"],
              "stretch_x": s["stretch_x"], "stretch_y": s["stretch_y"]} for s in manifest["snapshots"]]
    with (args.frames_dir / "machine_truth.csv").open(newline="") as handle:
        truth = {int(row["frame"]): row for row in csv.DictReader(handle)}
    cases.extend({"name": f"frame_{frame:03d}", "reference": args.frames_dir / "reference.png",
                  "target": args.frames_dir / f"frame_{frame:03d}.png",
                  "stretch_x": float(truth[frame]["stretch_x"]), "stretch_y": float(truth[frame]["stretch_y"])}
                 for frame in (40, 100, 199))
    cases.append({"name": "adrien_masked", "reference": args.adrien_dir / "img_000000_0.09422.tiff",
                  "target": args.adrien_dir / "img_000075_1608.88508.tiff", "mask": args.adrien_dir / "Mask.tiff"})
    if args.only:
        cases = [case for case in cases if case["name"] in args.only]
        if not cases:
            parser.error("no requested external case was found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, case in enumerate(cases):
        reference = case["reference"].resolve()
        target = case["target"].resolve()
        with Image.open(reference) as image:
            width, height = image.size
        model_height = round(192 * height / width)
        seed = 34000000 + index * 1000000
        directory = args.output_dir / case["name"]
        inputs = directory / "inputs"
        qualification = directory / "qualification"
        external = ["--external-reference", str(reference), "--external-target", str(target)]
        if "stretch_x" in case:
            external.extend(("--external-stretch-x", str(case["stretch_x"]),
                             "--external-stretch-y", str(case["stretch_y"])))
        common = ["--weights", str(args.weights.resolve()), "--live-dic-source", str(args.live_dic_source.resolve()),
                  "--frozen-parameters", str(args.frozen_parameters.resolve()), "--seed-offsets", str(seed),
                  "--families", "heterogeneous", "--cases-per-family", "1", "--model-width", "192",
                  "--model-height", str(model_height), *external]
        subprocess.run([sys.executable, str(scripts / "generate_p2_test_inputs.py"), *common,
                        "--output-dir", str(inputs), "--retain-translation-affine"], check=True)
        command = [sys.executable, str(scripts / "run_image_seed_test.py"), *common,
                   "--inputs", str(inputs), "--output-dir", str(qualification), "--skip-g2-summary"]
        if case.get("mask"):
            command.extend(("--roi-mask", str(case["mask"].resolve())))
        subprocess.run(command, check=True)
        evaluation = qualification / "evaluation"
        row = json.loads((evaluation / "summary.json").read_text())["rows"][0]
        row.update({"case": case["name"], "reference": str(reference), "target": str(target),
                    "reference_sha256": sha256(reference), "target_sha256": sha256(target),
                    "mask_sha256": sha256(case["mask"]) if case.get("mask") else None,
                    "native_size": [width, height], "model_size": [192, model_height]})
        rows.append(row)
        with np.load(evaluation / f"{row['pair']}.npz") as data:
            centres = data["centres"]
            xs, ys = np.unique(centres[:, 0]), np.unique(centres[:, 1])
            colours = np.zeros((len(centres), 3), np.uint8)
            colours[data["valid_roi"].astype(bool)] = (90, 90, 90)
            colours[data["accepted"].astype(bool)] = (0, 185, 90)
            colours[data["confirmed_seeds"].astype(bool)] = (40, 120, 255)
            rgb = np.zeros((len(ys), len(xs), 3), np.uint8)
            rgb[np.searchsorted(ys, centres[:, 1]), np.searchsorted(xs, centres[:, 0])] = colours
            Image.fromarray(rgb).resize((len(xs) * 4, len(ys) * 4), Image.Resampling.NEAREST).save(directory / "accepted_map.png")
        (args.output_dir / "summary.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
        print(f"Completed external case {case['name']}", flush=True)


if __name__ == "__main__":
    main()

"""Compare verification with regenerated or fingerprinted cached solver pixels."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--frozen-parameters", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("at least one repetition is required")
    frozen = json.loads(args.frozen_parameters.read_text())["selected"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pilot = Path(__file__).with_name("qualify_image_seed_pilot.py")
    rows = []
    for repetition in range(args.repetitions + 1):
        folders = {mode: args.output_dir / f"run{repetition}" / mode for mode in ("regenerated", "cached")}
        times, reports = {}, {}
        for mode in (("regenerated", "cached") if repetition % 2 == 0 else ("cached", "regenerated")):
            folder = folders[mode]
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"output already exists: {folder}; choose a new directory")
            folder.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(pilot), "--archive", str(args.inputs), "--live-dic-source", str(args.live_dic_source),
                   "--output-dir", str(folder), "--lazy-rotation-weights", str(args.weights),
                   "--lazy-rotation-output", str(folder / "rotations"), "--frozen-parameters", str(args.frozen_parameters),
                   "--tau-u", str(frozen["tau_u"]), "--tau-g", str(frozen["tau_g"]),
                   "--sparse-stride", str(frozen["sparse_stride"]), "--minimum-anchors", str(frozen["minimum_anchors"]),
                   "--escalation", "--local-affine-escalation"]
            if mode == "cached":
                cmd.append("--use-cached-solver-images")
            start = time.perf_counter()
            with (folder / "run.log").open("w") as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
            times[mode] = time.perf_counter() - start
            reports[mode] = json.loads((folder / "summary.json").read_text())
            if len(reports[mode]["rows"]) != 1:
                raise ValueError("benchmark requires one image pair")
        left, right = (reports[x]["rows"][0] for x in ("regenerated", "cached"))
        if left["pair"] != right["pair"]:
            raise ValueError("pair identities differ")
        with (np.load(folders["regenerated"] / f"{left['pair']}.npz") as a,
              np.load(folders["cached"] / f"{left['pair']}.npz") as b):
            keys = ("centres", "parameters", "accepted", "parent", "origin", "generation", "confirmed_seeds", "correct")
            same = {key: bool(np.array_equal(a[key], b[key], equal_nan=True)) for key in keys}
        rows.append({"warmup": repetition == 0, "seconds": times, "same": same,
                     "results": {mode: reports[mode]["rows"][0] for mode in reports}})
        print(f"Completed repetition {repetition}: {times}", flush=True)
    measured = [row for row in rows if not row["warmup"]]
    medians = {key: statistics.median(row["seconds"][key] for row in measured) for key in ("regenerated", "cached")}
    (args.output_dir / "summary.json").write_text(json.dumps({"runs": rows, "medians_s": medians}, indent=2) + "\n")
    exact = all(all(row["same"].values()) for row in rows)
    lines = ["# Reuse of fingerprinted solver images", "",
             ("One 3840 × 2160 synthetic pair, one warm-up and measured alternating-order repetitions. "
              "Each run starts a fresh process and graphics context. The cached image files were created earlier; "
              "these verification-only times exclude their creation and the initial matching. "
              "They are not end-to-end production or GPU-resident timings."), "",
             (f"Median verification process: regenerated {medians['regenerated']:.3f} s; "
              f"cached {medians['cached']:.3f} s. Exact state, status and provenance parity: {exact}."), "",
             "| Run | Warm-up | Regenerated (s) | Cached (s) | Exact parity |", "|---:|---|---:|---:|---|"]
    for index, row in enumerate(rows):
        lines.append(f"| {index} | {row['warmup']} | {row['seconds']['regenerated']:.3f} | "
                     f"{row['seconds']['cached']:.3f} | {all(row['same'].values())} |")
    lines.extend(["", ("The source window archive and both image files are checked by SHA-256 "
                       "(Secure Hash Algorithm with a 256-bit digest) before reuse. Cached values are solver inputs, "
                       "not accepted measurements. Full stage details are in the saved JavaScript Object Notation (JSON) file."), ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))
    if not exact:
        raise SystemExit("cached image parity failed")


if __name__ == "__main__":
    main()

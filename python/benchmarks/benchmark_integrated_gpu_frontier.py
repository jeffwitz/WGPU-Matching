"""Compare host and GPU frontiers inside the image-only initialization pilot."""

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
    parser.add_argument("--external-reference", type=Path)
    parser.add_argument("--external-target", type=Path)
    parser.add_argument("--roi-mask", type=Path)
    parser.add_argument("--gpu-resident-selection", action="store_true")
    parser.add_argument("--compare-nonresident", action="store_true",
                        help="compare against GPU selection with the original CPU readback")
    parser.add_argument("--gpu-hypothesis-selection", action="store_true")
    parser.add_argument("--gpu-anchor-frontier", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1 or bool(args.external_reference) != bool(args.external_target):
        parser.error("provide at least one repetition and either both external images or neither")
    if args.compare_nonresident and not (args.gpu_resident_selection and args.gpu_anchor_frontier
                                         and args.gpu_hypothesis_selection):
        parser.error("nonresident comparison requires all three GPU selection options")
    frozen = json.loads(args.frozen_parameters.read_text())["selected"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pilot = Path(__file__).with_name("qualify_image_seed_pilot.py")
    rows = []
    for repetition in range(args.repetitions + 1):
        folders = {mode: args.output_dir / f"run{repetition}" / mode for mode in ("host", "gpu")}
        times, reports = {}, {}
        for mode in (("host", "gpu") if repetition % 2 == 0 else ("gpu", "host")):
            folder = folders[mode]
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"output already exists: {folder}; choose a new directory")
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(pilot), "--archive", str(args.inputs), "--live-dic-source", str(args.live_dic_source),
                       "--output-dir", str(folder), "--lazy-rotation-weights", str(args.weights),
                       "--lazy-rotation-output", str(folder / "rotations"), "--frozen-parameters", str(args.frozen_parameters),
                       "--tau-u", str(frozen["tau_u"]), "--tau-g", str(frozen["tau_g"]),
                       "--sparse-stride", str(frozen["sparse_stride"]), "--minimum-anchors", str(frozen["minimum_anchors"]),
                       "--escalation", "--local-affine-escalation", "--use-cached-solver-images", "--batch-anchor-confirmation"]
            for option in ("external_reference", "external_target", "roi_mask"):
                value = getattr(args, option)
                if value is not None:
                    command.extend(("--" + option.replace("_", "-"), str(value)))
            if mode == "gpu":
                command.append("--gpu-frontier")
                if args.gpu_anchor_frontier:
                    command.append("--gpu-anchor-frontier")
                if args.gpu_hypothesis_selection:
                    command.append("--gpu-hypothesis-selection")
                if args.gpu_resident_selection:
                    command.append("--gpu-resident-selection")
            elif args.compare_nonresident:
                command.extend(("--gpu-frontier", "--gpu-anchor-frontier", "--gpu-hypothesis-selection"))
            start = time.perf_counter()
            with (folder / "run.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            times[mode] = time.perf_counter() - start
            reports[mode] = json.loads((folder / "summary.json").read_text())["rows"]
            if len(reports[mode]) != 1:
                raise ValueError("benchmark requires exactly one image pair")
            reports[mode] = reports[mode][0]
        arow, brow = reports["host"], reports["gpu"]
        if arow["pair"] != brow["pair"]:
            raise ValueError("pair identities differ")
        with (np.load(folders["host"] / f"{arow['pair']}.npz") as a,
              np.load(folders["gpu"] / f"{arow['pair']}.npz") as b):
            keys = ("centres", "accepted", "parent", "origin", "generation", "attempts", "rejection_reason", "confirmed_seeds")
            exact = {key: bool(np.array_equal(a[key], b[key])) for key in keys}
            common = a["accepted"].astype(bool) & b["accepted"].astype(bool)
            diff = np.abs(a["parameters"][common] - b["parameters"][common])
            displacement_max = float(diff[:, [0, 3]].max()) if len(diff) else 0.
            gradient_max = float(diff[:, [1, 2, 4, 5]].max()) if len(diff) else 0.
            changed = int(np.count_nonzero(np.any(diff != 0, axis=1)))
        rows.append({"warmup": repetition == 0, "seconds": times, "exact_metadata": exact,
                     "changed_common_parameter_windows": changed, "max_displacement_difference_px": displacement_max,
                     "max_gradient_difference": gradient_max, "results": reports})
        print(f"Completed repetition {repetition}: {times}", flush=True)
    measured = [row for row in rows if not row["warmup"]]
    medians = {mode: {"full_s": statistics.median(row["seconds"][mode] for row in measured),
                      "verification_s": statistics.median(row["results"][mode]["wall_s_excluding_hypothesis_generation"]
                                                          for row in measured)} for mode in ("host", "gpu")}
    (args.output_dir / "summary.json").write_text(json.dumps({"runs": rows, "medians": medians}, indent=2) + "\n")
    lines = ["# Host versus graphics processing unit (GPU) frontier in the initialization pilot", "",
             ("One image pair, one warm-up and alternating-order measured repetitions. The two paths use identical cached "
              "solver images and batched anchor confirmation. Host times include process startup, image-cache verification, "
              "device/reference setup and result writes. Input hypotheses and cached images were prepared earlier; "
              "these are not complete end-to-end timings. No GPU timestamp measurement was available."), "",
             (f"GPU anchor confirmation enabled: {args.gpu_anchor_frontier}; "
              f"GPU hypothesis selection enabled: {args.gpu_hypothesis_selection}; "
              f"resident selection enabled: {args.gpu_resident_selection}."), "",
             ("In the tables, host denotes the previous GPU path with host readback; GPU denotes resident selection."
              if args.compare_nonresident else "In the tables, host denotes the host confirmation and propagation path."), "",
             (f"Median verification and propagation: host {medians['host']['verification_s']:.3f} s; "
              f"GPU frontier {medians['gpu']['verification_s']:.3f} s. Median process: "
              f"{medians['host']['full_s']:.3f} / {medians['gpu']['full_s']:.3f} s."), "",
             "| Run | Warm-up | Process host / GPU (s) | Metadata exact | Max displacement difference (px) | Max gradient difference |",
             "|---:|---|---:|---|---:|---:|"]
    for index, row in enumerate(rows):
        lines.append(f"| {index} | {row['warmup']} | {row['seconds']['host']:.3f} / {row['seconds']['gpu']:.3f} | "
                     f"{all(row['exact_metadata'].values())} | {row['max_displacement_difference_px']:.9g} | "
                     f"{row['max_gradient_difference']:.9g} |")
    lines.extend(["", ("The GPU frontier reads one small counter per pass, then reads the final state and provenance buffers once each. "
                       "Initial image matching and intermediate transfers remain host-mediated; "
                       "the complete pipeline is not GPU-resident. Consult the saved JavaScript Object Notation (JSON) file for per-pass transfer counts "
                       "and all output metadata. A real image without truth cannot establish accuracy."), ""])
    if args.gpu_hypothesis_selection and not args.gpu_resident_selection:
        lines.extend([("The selected hypothesis field is read back to the host and uploaded again for anchor "
                       "confirmation. The target image is uploaded separately for selection and propagation."), ""])
    if args.gpu_resident_selection:
        lines.extend(["The selected field and target remain on the solver device through anchor confirmation.", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

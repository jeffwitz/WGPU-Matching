"""Independently audit completeness, analytic metrics and every qualification GIF."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def validate(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads((root / "summary.json").read_text())["rows"]
    protocol = manifest["protocol"]
    expected = {
        (case, method, basis)
        for case in protocol["cases"]
        for method in protocol["methods"]
        for basis in ("affine", "gaussian_rbf")
    }
    actual = [(r["case"], r["method"], r["basis_requested"]) for r in rows]
    assert len(actual) == len(set(actual)), "Duplicated result key"
    assert set(actual) == expected, f"Missing or unexpected rows: {set(actual) ^ expected}"
    images = {v["case"]: v for v in protocol["images"]}
    repository = Path(__file__).resolve().parents[2]
    for name, expected_hash in protocol["source_files"].items():
        assert hashlib.sha256((repository / name).read_bytes()).hexdigest() == expected_hash, name
    checks = {
        "rows": len(rows),
        "whole_image_gifs": 0,
        "native_crop_gifs": 0,
        "execution_errors": [],
        "nonconverged_fields": [],
        "nonrepeatable_pairs": [],
        "verified_pair_archives": 0,
    }
    for row in rows:
        key = (row["case"], row["method"], row["basis_requested"])
        folder = root.joinpath(*key)
        assert (folder / "result.json").is_file(), key
        assert (root / f"case_{row['case']}.md").is_file(), key
        visuals = row["visualizations"]
        with Image.open(folder / visuals["reference_png"]) as reference:
            native_preview = reference.size
        with Image.open(folder / visuals["corrected_png"]) as corrected:
            assert corrected.size == native_preview, key
        for flag, counter in (("gif", "whole_image_gifs"), ("zoom_gif", "native_crop_gifs")):
            if flag not in visuals:
                continue
            with Image.open(folder / visuals[flag]) as gif:
                assert gif.n_frames == 2, (key, flag, gif.n_frames)
                size = gif.size
                for frame in range(2):
                    gif.seek(frame)
                    gif.load()
                    assert gif.size == size, key
                assert gif.info.get("duration", 0) > 0, key
            checks[counter] += 1
        a = row["assessment"]
        assert len(a["per_seed"]) == a["requested"], key
        available = [v for v in a["per_seed"] if v["estimated"]]
        assert len(available) == a["estimated"], key
        truth = images[row["case"]]["analytic_truth"]
        if truth is None:
            assert a["pass_below_5px"] is None and a["correct_lt5"] is None, key
        else:
            size = np.asarray(images[row["case"]]["size"], float)
            errors = []
            for v in available:
                expected_displacement = (np.asarray([v["x"], v["y"]]) - size / 2) * (
                    np.asarray([truth["stretch_x"], truth["stretch_y"]]) - 1
                )
                error = np.linalg.norm(np.asarray(v["parameters"][:2]) - expected_displacement)
                assert np.isclose(error, v["error_px"], atol=1e-8), key
                errors.append(error)
            errors = np.asarray(errors)
            assert a["correct_le2"] == int(np.count_nonzero(errors <= 2)), key
            assert a["correct_lt5"] == int(np.count_nonzero(errors < 5)), key
            assert a["false_estimates_ge5"] == int(np.count_nonzero(errors >= 5)), key
            assert a["pass_below_5px"] == bool(
                len(errors) == a["requested"] and np.all(errors < 5)
            ), key
            if len(errors):
                e = a["errors_px"]
                for metric, value in (
                    ("rmse", np.sqrt(np.mean(errors**2))),
                    ("median", np.median(errors)),
                    ("p95", np.percentile(errors, 95)),
                    ("maximum", np.max(errors)),
                ):
                    assert np.isclose(e[metric], value, atol=1e-8), (key, metric)
            else:
                assert all(v is None for v in a["errors_px"].values()), key
        if row["status"] == "execution_error":
            checks["execution_errors"].append(key)
            continue
        assert len(row["trials"]) == protocol["repeats"], key
        times = np.asarray(row["affine_pipeline_wall_s"])
        assert np.all(times > 0) and len(times) == protocol["repeats"], key
        assert np.isclose(np.median(times), row["affine_pipeline_median_s"]), key
        if row.get("visualizations", {}).get("diagnostic_only"):
            assert a["estimated"] == 0 and row["basis_effective"] is None, key
        if row["basis_effective"]:
            assert (folder / "model.json.gz").exists(), key
            if not row["diagnostics"]["irls_converged"]:
                checks["nonconverged_fields"].append(key)
        if row["basis_requested"] == "affine":
            digests = {(t["pair_digest"], t["acceptance_digest"]) for t in row["trials"]}
            if len(digests) != 1:
                checks["nonrepeatable_pairs"].append(key)
            with gzip.open(folder.parent / "correspondences.csv.gz", "rt") as stream:
                observations = list(csv.DictReader(stream))
            stats = row["direct_pairs"]
            assert len(observations) == stats["raw_count"], key
            assert sum(v["accepted"] == "True" for v in observations) == stats["retained_count"], (
                key
            )
            assert (
                sum(v["upstream_accepted"] == "True" for v in observations)
                == stats["upstream_retained_count"]
            ), key
            checks["verified_pair_archives"] += 1
    checks["protocol_sha256"] = manifest["protocol_sha256"]
    checks["validator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checks["status"] = "complete; metric and artifact assertions passed"
    (root / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    validate(args.run)


if __name__ == "__main__":
    main()

"""Verify that Sphinx serves the exact measured GIFs and compressed archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.gifs = []
        self.archives = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "img" and urlsplit(attrs.get("src", "")).path.endswith(".gif"):
            self.gifs.append(attrs["src"])
        if tag == "a" and urlsplit(attrs.get("href", "")).path.endswith(".gz"):
            self.archives.append(attrs["href"])


def identical(page, urls, expected):
    assert len(urls) == len(expected), (page, len(urls), len(expected))
    for url, source in zip(urls, expected, strict=True):
        copied = (page.parent / unquote(urlsplit(url).path)).resolve()
        assert copied.is_file(), copied
        assert (
            hashlib.sha256(copied.read_bytes()).digest()
            == hashlib.sha256(source.read_bytes()).digest()
        ), (page, source, copied)


def check(html, run, ablation):
    records = json.loads((run / "summary.json").read_text())["rows"]
    manifest = json.loads((run / "manifest.json").read_text())
    rendered = html / "_generated_results" / run.resolve().relative_to(ROOT / "results")
    counts = {"main_gifs": 0, "main_archive_links": 0, "ablation_gifs": 0}
    for case in manifest["protocol"]["cases"]:
        rows = [r for r in records if r["case"] == case]
        page = rendered / f"case_{case}.html"
        parser = Assets()
        parser.feed(page.read_text())
        images, archives = [], []
        for row in rows:
            folder = run / case / row["method"] / row["basis_requested"]
            images.append(folder / row["visualizations"]["gif"])
            if "zoom_gif" in row["visualizations"]:
                images.append(folder / row["visualizations"]["zoom_gif"])
            for file in (
                folder.parent / "correspondences.csv.gz",
                folder / "model.json.gz",
                folder / "rejected_diagnostic_model.json.gz",
            ):
                if file.is_file():
                    archives.append(file)
        identical(page, parser.gifs, images)
        identical(page, parser.archives, archives)
        counts["main_gifs"] += len(images)
        counts["main_archive_links"] += len(archives)
    rows = json.loads((ablation / "summary.json").read_text())["records"]
    page = (
        html
        / "_generated_results"
        / ablation.resolve().relative_to(ROOT / "results")
        / "index.html"
    )
    parser = Assets()
    parser.feed(page.read_text())
    images = []
    for row in rows:
        folder = ablation / row["case"] / row["method"] / row["basis_requested"]
        images.append(folder / row["visualizations"]["gif"])
        if "zoom_gif" in row["visualizations"]:
            images.append(folder / row["visualizations"]["zoom_gif"])
    identical(page, parser.gifs, images)
    counts["ablation_gifs"] = len(images)
    counts["status"] = "all compiled assets equal their original measured artifacts"
    print(json.dumps(counts, indent=2))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=ROOT / "docs/_build/html")
    parser.add_argument(
        "--run", type=Path, default=ROOT / "results/qualification/all_methods_20261107"
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=ROOT / "results/qualification/raw_projection_ablation_20261107",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/web-asset-review.json")
    args = parser.parse_args()
    result = check(args.html, args.run, args.ablation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

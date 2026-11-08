"""Build the curated Sphinx/LaTeX handbook and copy its final PDF artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sphinx-build", default="sphinx-build")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/pdf/WGPU-Matching-Technical-Handbook.pdf"
    )
    args = parser.parse_args()
    executable = shutil.which(args.sphinx_build)
    if executable is None:
        raise SystemExit("Install the documentation extra and activate that environment first")
    for program in ("latexmk", "xelatex"):
        if shutil.which(program) is None:
            raise SystemExit(f"Missing system tool: {program}")
    build = ROOT / "docs/_build/print"
    subprocess.run(
        [executable, "-M", "latexpdf", str(ROOT / "docs"), str(build), "-E", "-W", "--keep-going"],
        cwd=ROOT,
        check=True,
    )
    pdf = build / "latex/WGPU-Matching.pdf"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, args.output)
    metadata = {
        "pdf_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "source_root": "docs/pdf/index.rst",
        "theme": "A4 technical handbook; Pagella text/math, Heros headings",
        "animations": "static crop pairs in print; complete alternating GIFs in web galleries",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree": "The PDF includes current working-tree documentation and qualification results",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

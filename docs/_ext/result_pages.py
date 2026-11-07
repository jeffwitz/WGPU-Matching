"""Generate Sphinx pages for Markdown reports and raw tabular results."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

PREVIEW_ROWS = 50


def _write_csv_page(source: Path, page_source: Path, relative: Path) -> None:
    page_source.parent.mkdir(parents=True, exist_ok=True)
    raw_copy = page_source.parent / source.name
    preview_copy = page_source.parent / f"_{source.stem}_preview.csv"
    shutil.copy2(source, raw_copy)

    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))

    if rows:
        header, data = rows[0], rows[1:]
        preview = [header, *data[:PREVIEW_ROWS]]
    else:
        preview, data = [], []

    with preview_copy.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(preview)

    title = f"{source.name} — raw data"
    body = [
        title,
        "=" * len(title),
        "",
        f"Source file: ``{relative.as_posix()}``.",
        "",
        f"This page previews the first {min(PREVIEW_ROWS, len(data))} of {len(data)} data rows.",
        f"Download the complete file: :download:`{source.name} <{source.name}>`.",
        "",
    ]
    if header:
        body.extend(
            [
                f".. csv-table:: CSV preview ({len(header)} columns)",
                f"   :file: {preview_copy.name}",
                "   :header-rows: 1",
                "   :widths: auto",
                "",
            ]
        )
    else:
        body.extend(["The CSV file is empty.", ""])

    page_source.write_text("\n".join(body), encoding="utf-8")


def _rewrite_markdown_links(
    text: str, report: Path, results_root: Path, repo_root: Path
) -> str:
    link_pattern = re.compile(r"(?<!!)\[([^\]]+)\]\(([^\s)]+)([^)]*)\)")

    def rewrite(match: re.Match[str]) -> str:
        label, target, title = match.group(1), match.group(2), match.group(3)
        if "://" in target or target.startswith(("#", "mailto:")):
            return match.group(0)
        path_text, separator, fragment = target.partition("#")
        candidate = (report.parent / path_text).resolve()
        if candidate.suffix.lower() == ".csv":
            return f"{{doc}}`{label} <{path_text}{separator}{fragment}>`"
        if candidate.suffix.lower() == ".json":
            return f"{{doc}}`{label} <{path_text}{separator}{fragment}>`"
        if candidate.suffix.lower() == ".py" and candidate.is_file():
            try:
                repo_path = candidate.relative_to(repo_root.resolve()).as_posix()
            except ValueError:
                return match.group(0)
            url = f"https://github.com/jeffwitz/WGPU-Matching/blob/main/{repo_path}{separator}{fragment}"
            return f"[{label}]({url}{title})"
        try:
            candidate.relative_to(results_root.resolve())
        except ValueError:
            return match.group(0)
        if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"} and candidate.is_file():
            return f"{{download}}`{label} <{path_text}{separator}{fragment}>`"
        if candidate.suffix.lower() == ".md" and candidate.is_file():
            return match.group(0)
        return match.group(0)

    return link_pattern.sub(rewrite, text)


def generate_result_pages(app) -> None:
    repo_root = Path(app.confdir).resolve().parent
    results_root = repo_root / "results"
    generated_root = Path(app.confdir) / "_generated_results"
    if generated_root.exists():
        shutil.rmtree(generated_root)
    generated_root.mkdir(parents=True)

    report_sources = sorted(results_root.rglob("*.md"))
    csv_sources = sorted(results_root.rglob("*.csv"))
    json_sources = sorted(results_root.rglob("*.json"))
    report_docs: list[str] = []
    data_docs: list[str] = []

    for source in report_sources:
        relative = source.relative_to(results_root)
        destination = generated_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        original_markdown = source.read_text(encoding="utf-8")
        for image in re.findall(r"!?\[[^]]*\]\(([^\s)]+)", original_markdown):
            image_path = (source.parent / image.split("#", 1)[0]).resolve()
            try:
                image_relative = image_path.relative_to(results_root.resolve())
            except ValueError:
                continue
            if image_path.is_file() and image_path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".svg",
                ".gif",
            }:
                image_destination = generated_root / image_relative
                image_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, image_destination)

        markdown = _rewrite_markdown_links(original_markdown, source, results_root, repo_root)
        destination.write_text(markdown, encoding="utf-8")
        report_docs.append(destination.relative_to(generated_root).with_suffix("").as_posix())

    for source in csv_sources:
        relative = source.relative_to(results_root)
        _write_csv_page(source, generated_root / f"{relative.as_posix()}.rst", relative)
        data_docs.append(relative.as_posix())

    for source in json_sources:
        relative = source.relative_to(results_root)
        page_source = generated_root / f"{relative.as_posix()}.rst"
        page_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, page_source.parent / source.name)
        title = f"{source.name} — raw data"
        page_source.write_text(
            "\n".join(
                [
                    title,
                    "=" * len(title),
                    "",
                    f"Source file: ``{relative.as_posix()}``.",
                    "",
                    f"Download the complete file: :download:`{source.name} <{source.name}>`.",
                    "",
                    f".. literalinclude:: {source.name}",
                    "   :language: json",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        data_docs.append(page_source.relative_to(generated_root).with_suffix("").as_posix())

    index_lines = [
        "Reports and raw data",
        "=====================",
        "",
        "Markdown reports and machine-readable result files are rendered as pages. CSV pages show a compact preview and provide the complete source file for download.",
        "",
        "Reports",
        "-------",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
    ]
    index_lines.extend(f"   {doc}" for doc in report_docs)
    index_lines.extend(["", "Raw data", "--------", "", ".. toctree::", "   :maxdepth: 2", ""])
    index_lines.extend(f"   {doc}" for doc in data_docs)
    (generated_root / "index.rst").write_text("\n".join(index_lines), encoding="utf-8")


def setup(app):
    app.connect("builder-inited", generate_result_pages)
    return {"version": "1.0", "parallel_read_safe": False}

"""Generate Sphinx pages for Markdown reports and raw tabular results."""

from __future__ import annotations

import csv
import itertools
import os
import re
import shutil
from pathlib import Path

PREVIEW_ROWS = 50
PREVIEW_JSON_LINES = 160


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
        ":orphan:",
        "",
        title,
        "=" * len(title),
        "",
        ":doc:`Raw-data catalogue </_generated_results/data>`.",
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


def _rewrite_markdown_links(text: str, report: Path, results_root: Path, repo_root: Path) -> str:
    link_pattern = re.compile(r"(?<!!)\[([^\]]+)\]\(([^\s)]+)([^)]*)\)")

    def rewrite(match: re.Match[str]) -> str:
        label, target, title = match.group(1), match.group(2), match.group(3)
        if "://" in target or target.startswith(("#", "mailto:")):
            return match.group(0)
        path_text, separator, fragment = target.partition("#")
        candidate = (report.parent / path_text).resolve()
        if candidate.name.endswith((".csv.gz", ".json.gz", ".npz")) and candidate.is_file():
            return f"{{download}}`{label} <{path_text}{separator}{fragment}>`"
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
        current_report = results_root / "qualification/all_methods_20261107/index.md"
        if relative.parts[0] != "qualification" and current_report.is_file():
            current_link = Path(os.path.relpath(current_report, source.parent)).as_posix()
            banner = (
                "\n\n> **Study scope.** This focused report retains its original protocol and measurements. "
                f"For the refreshed complete initializer comparison, use the [current qualification]({current_link}).\n"
            )
            heading = re.search(r"(?m)^# .*$", original_markdown)
            if heading:
                original_markdown = (
                    original_markdown[: heading.end()] + banner + original_markdown[heading.end() :]
                )
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
                ".gz",
                ".npz",
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
        preview_name = f"_{source.stem}_preview.json"
        with source.open(encoding="utf-8") as stream:
            preview_lines = list(itertools.islice(stream, PREVIEW_JSON_LINES + 1))
        truncated = len(preview_lines) > PREVIEW_JSON_LINES
        (page_source.parent / preview_name).write_text(
            "".join(preview_lines[:PREVIEW_JSON_LINES]), encoding="utf-8"
        )
        title = f"{source.name} — raw data"
        page_source.write_text(
            "\n".join(
                [
                    ":orphan:",
                    "",
                    title,
                    "=" * len(title),
                    "",
                    ":doc:`Raw-data catalogue </_generated_results/data>`.",
                    "",
                    f"Source file: ``{relative.as_posix()}``.",
                    "",
                    f"Download the complete file: :download:`{source.name} <{source.name}>`.",
                    "",
                    (
                        f"Preview: first {PREVIEW_JSON_LINES} lines only; this fragment is not a complete JSON document."
                        if truncated
                        else "Complete content preview."
                    ),
                    "",
                    f".. literalinclude:: {preview_name}",
                    "   :language: json",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        data_docs.append(page_source.relative_to(generated_root).with_suffix("").as_posix())

    # Keep thousands of raw arrays out of every page's global navigation.
    # They remain themed, indexed and directly linked, with a separate catalogue.
    categories = {
        "features": "Feature detectors and learned matchers",
        "uniaxial": "Uniaxial traction",
        "projection": "Field projection",
        "cpm": "CPM diagnostics",
        "qualification": "Complete initializer qualification",
    }
    directories = {Path(".")}
    reports_by_directory: dict[Path, list[str]] = {}
    for doc in report_docs:
        path = Path(doc)
        reports_by_directory.setdefault(path.parent, []).append(path.name)
        directories.update([path.parent, *path.parent.parents])
    children: dict[Path, list[Path]] = {}
    for directory in directories:
        if directory != Path("."):
            children.setdefault(directory.parent, []).append(directory)
    for directory in sorted(directories):
        local = reports_by_directory.get(directory, [])
        nested = sorted(children.get(directory, []))
        if directory == Path("."):
            title = "Reports and raw data"
        else:
            words = directory.name.replace("pct", " percent").replace("_", " ").split()
            abbreviations = {
                t.lower(): t for t in ("DIC", "WGSL", "ONNX", "GPU", "CPU", "RBF", "HORG", "CPM")
            }
            words = [abbreviations.get(word.lower(), word) for word in words]
            title = categories.get(directory.as_posix(), " ".join(words).capitalize())
            if directory.as_posix() not in categories:
                title = " ".join(words)
                title = title[:1].upper() + title[1:]
        lines = [title, "=" * len(title), ""]
        if directory == Path("."):
            lines += [
                (
                    "Reports are grouped by study. Large arrays remain available as themed data "
                    "pages with bounded previews and complete-file downloads; they are listed in "
                    "the separate raw-data catalogue rather than in every page's navigation."
                ),
                "",
            ]
        lines += [".. toctree::", "   :maxdepth: 2", ""]
        lines += [f"   {name}" for name in sorted(local)]
        lines += [f"   {child.name}/_reports" for child in nested]
        if directory == Path("."):
            lines += ["   data", ""]
            destination = generated_root / "index.rst"
        else:
            destination = generated_root / directory / "_reports.rst"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    catalogue = [
        "Raw-data catalogue",
        "==================",
        "",
        (
            "Use the study reports for interpretation. The links below open compact themed "
            "previews with complete-file downloads. Compressed correspondence/model archives "
            "are linked directly from their study galleries."
        ),
        "",
    ]
    grouped: dict[str, list[str]] = {}
    for doc in data_docs:
        grouped.setdefault(Path(doc).parts[0], []).append(doc)
    for category, documents in sorted(grouped.items()):
        title = categories.get(category, category.replace("_", " "))
        catalogue += [title, "-" * len(title), ""]
        catalogue += [f"* :doc:`{doc} <{doc}>`" for doc in sorted(documents)]
        catalogue += [""]
    (generated_root / "data.rst").write_text("\n".join(catalogue), encoding="utf-8")


def setup(app):
    app.connect("builder-inited", generate_result_pages)
    return {"version": "1.0", "parallel_read_safe": False}

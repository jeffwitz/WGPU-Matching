"""Keep scientific downloads complete while bounding rendered raw-data previews."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[2] / "docs/_ext/result_pages.py"
spec = importlib.util.spec_from_file_location("result_pages", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_result_pages_bound_json_and_preserve_compressed_downloads(tmp_path):
    docs = tmp_path / "docs"
    results = tmp_path / "results/study"
    docs.mkdir()
    results.mkdir(parents=True)
    data = json.dumps({"observations": list(range(500))}, indent=2)
    (results / "large.json").write_text(data)
    (results / "pairs.csv.gz").write_bytes(b"compressed-fixture")
    (results / "model.json.gz").write_bytes(b"model-fixture")
    (results / "report.md").write_text(
        "# Study\n\n[Pairs](pairs.csv.gz) and [model](model.json.gz).\n"
    )
    module.generate_result_pages(SimpleNamespace(confdir=str(docs)))
    generated = docs / "_generated_results/study"
    assert (generated / "large.json").read_text() == data
    assert len((generated / "_large_preview.json").read_text().splitlines()) == 160
    page = (generated / "large.json.rst").read_text()
    assert "not a complete JSON document" in page
    assert "literalinclude:: _large_preview.json" in page
    assert "download:`large.json <large.json>`" in page
    assert (generated / "pairs.csv.gz").read_bytes() == b"compressed-fixture"
    assert (generated / "model.json.gz").read_bytes() == b"model-fixture"
    assert "{download}`Pairs <pairs.csv.gz>`" in (generated / "report.md").read_text()
    assert page.startswith(":orphan:")
    assert "study/large.json" in (docs / "_generated_results/data.rst").read_text()
    assert "large.json" not in (generated / "_reports.rst").read_text()
    assert "study/_reports" in (docs / "_generated_results/index.rst").read_text()
    (generated / "stale.rst").write_text("stale")
    module.generate_result_pages(SimpleNamespace(confdir=str(docs)))
    assert not (generated / "stale.rst").exists()

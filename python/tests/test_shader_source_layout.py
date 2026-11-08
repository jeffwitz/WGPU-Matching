from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from wgpu_matching import CPMOptions, DeepMatchingOptions, HORGOptions, ProjectionOptions

ROOT = Path(__file__).resolve().parents[2]
SHADERS = ROOT / "python/src/wgpu_matching/shaders"


def test_browser_and_python_share_the_packaged_wgsl_sources() -> None:
    browser_sources = [
        ROOT / name
        for name in (
            "deepmatching.js",
            "horg.js",
            "cpm.js",
            "matcher-contracts.js",
        )
    ]
    referenced = set()
    for source in browser_sources:
        text = source.read_text(encoding="utf-8")
        referenced.update(re.findall(r"([a-z0-9-]+\.wgsl)", text))

    assert referenced
    assert '"./python/src/wgpu_matching/matcher-contracts.json"' in (
        ROOT / "matcher-contracts.js"
    ).read_text(encoding="utf-8")
    assert all((SHADERS / name).is_file() for name in referenced)
    assert not list(ROOT.glob("*.wgsl")), (
        "WGSL copies at the repository root would drift from the package"
    )


def test_every_packaged_shader_is_included_by_setuptools_package_data() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'wgpu_matching = ["py.typed", "matcher-contracts.json", "shaders/*.wgsl"]' in pyproject
    assert list(SHADERS.glob("*.wgsl"))


def test_python_option_defaults_come_from_the_shared_matcher_contract() -> None:
    contract = json.loads((ROOT / "python/src/wgpu_matching/matcher-contracts.json").read_text())
    instances = {
        "deepmatching": asdict(DeepMatchingOptions()),
        "horg": asdict(HORGOptions()),
        "cpm": asdict(CPMOptions()),
        "projection": asdict(ProjectionOptions()),
    }
    for algorithm, spec in contract["algorithms"].items():
        for name, expected in spec["defaults"].items():
            python_name = spec["option_keys"][name]["python"]
            actual = instances[algorithm][python_name]
            if isinstance(expected, list):
                expected = tuple(expected)
            assert actual == expected

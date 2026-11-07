"""Sphinx configuration for the English Diátaxis documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

project = "WGPU-Matching"
author = "WGPU-Matching contributors"
copyright = "2026, WGPU-Matching contributors"
language = "en"
version = "0.1.0"
release = version

extensions = [
    "myst_parser",
    "result_pages",
    "define_acronyms",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
root_doc = "index"
nitpicky = True
nitpick_ignore = [("py:class", "wgpu_matching.features._TorchKeypointDetector")]
autosummary_generate = True
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_mock_imports = ["wgpu"]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_title = "WGPU-Matching"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "prev_next_buttons_location": "bottom",
}
html_context = {
    "github_user": "jeffwitz",
    "github_repo": "WGPU-Matching",
    "github_version": "main",
    "doc_path": "docs",
}

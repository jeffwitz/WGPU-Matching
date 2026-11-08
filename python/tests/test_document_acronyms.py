"""Readability checks for first-use scientific abbreviations in Sphinx prose."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sphinx")
from docutils import nodes
from docutils.utils import new_document

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "define_acronyms", ROOT / "docs/_ext/define_acronyms.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def transform(text):
    document = new_document("fixture")
    document.settings.env = SimpleNamespace(
        app=SimpleNamespace(srcdir=str(ROOT / "docs")),
        docname="fixture",
        note_dependency=lambda *args: None,
    )
    section = nodes.section()
    section += nodes.title(text="Example")
    section += nodes.paragraph(text=text)
    document += section
    module.DefineAcronyms(document).apply()
    return document.astext()


@pytest.mark.parametrize(
    "term,text",
    [
        ("IC-GN", "inverse\ncompositional Gauss–Newton (IC-GN)"),
        ("IC-GN", "inverse-compositional Gauss–Newton (IC-GN)"),
        ("RBF", "radial basis\nfunction (RBF)"),
        ("DIC", "digital image correlation (DIC)"),
    ],
)
def test_authored_definition_across_source_lines_is_not_duplicated(term, text):
    rendered = transform(text)
    assert rendered.count(term) == 1
    assert term + " (" not in rendered


def test_detector_word_does_not_falsely_define_dedode():
    rendered = transform("The detector DeDoDe produces points.")
    assert "DeDoDe (Detect, Don’t Describe" in rendered


def test_repeated_abbreviation_has_one_definition():
    rendered = transform("DIC initializes DIC windows.")
    assert rendered.count("digital image correlation") == 1


def test_plural_defines_the_singular_abbreviation_too():
    rendered = transform("POIs are requested. One POI is returned.")
    assert rendered.count("points of interest") == 1
    assert "POI (" not in rendered

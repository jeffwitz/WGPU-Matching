"""Define technical abbreviations independently on every rendered page."""

from __future__ import annotations

import json
import re
from pathlib import Path

from docutils import nodes
from sphinx.transforms import SphinxTransform
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)


class DefineAcronyms(SphinxTransform):
    """Expand first prose uses, including autodoc and generated reports."""

    default_priority = 750

    def apply(self, **kwargs):
        if self.env.docname == "reference/glossary":
            return
        registry = Path(self.app.srcdir) / "acronyms.json"
        self.env.note_dependency(str(registry))
        terms = json.loads(registry.read_text(encoding="utf-8"))
        reported = set()

        def audit(value, location):
            for token in re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", value):
                if token not in terms and token not in reported:
                    LOGGER.warning(
                        "Undefined abbreviation %s; add its meaning to docs/acronyms.json",
                        token,
                        location=location,
                        type="acronym",
                        subtype="undefined",
                    )
                    reported.add(token)

        for heading in self.document.findall(nodes.title):
            audit(heading.astext(), heading)
        pattern = re.compile(
            r"(?<![\w])("
            + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
            + r")(?![\w])"
        )
        defined = set()
        title = next(self.document.findall(nodes.title), None)
        if title is not None:
            used = list(
                dict.fromkeys(
                    term
                    for heading in self.document.findall(nodes.title)
                    for term in pattern.findall(heading.astext())
                )
            )
            if used:
                introduction = nodes.paragraph(classes=["title-term-definitions"])
                for index, term in enumerate(used):
                    if index:
                        introduction += nodes.Text("; ")
                    introduction += nodes.strong(text=term)
                    introduction += nodes.Text(" — " + terms[term])
                    defined.add(term)
                introduction += nodes.Text(".")
                title.parent.insert(title.parent.index(title) + 1, introduction)

        # Snapshot before mutation. Literal identifiers, equations, URLs and
        # executable snippets retain their exact spelling and semantics.
        for text in list(self.document.findall(nodes.Text)):
            parent = text.parent
            ancestors = []
            node = parent
            while node is not None:
                ancestors.append(node)
                node = node.parent
            if any(
                isinstance(
                    node,
                    (
                        nodes.literal,
                        nodes.literal_block,
                        nodes.math,
                        nodes.math_block,
                        nodes.raw,
                        nodes.title,
                        nodes.abbreviation,
                    ),
                )
                for node in ancestors
            ):
                continue
            if any(
                "title-term-definitions" in node.get("classes", [])
                for node in ancestors
                if isinstance(node, nodes.Element)
            ):
                continue
            if not any(
                isinstance(node, (nodes.paragraph, nodes.caption, nodes.rubric))
                for node in ancestors
            ):
                continue
            value = str(text)
            audit(value, parent)
            replacement = []
            offset = 0
            for match in pattern.finditer(value):
                replacement.append(nodes.Text(value[offset : match.start()]))
                term = match.group()
                replacement.append(nodes.abbreviation(term, term, explanation=terms[term]))
                if term not in defined:
                    # Preserve an author's existing “full name (ACRONYM)”.
                    full_name = terms[term].split(":", 1)[0].split(",", 1)[0].casefold()
                    preceding = value[max(0, match.start() - len(full_name) - 12) : match.start()]
                    if full_name not in preceding.casefold():
                        replacement.append(nodes.Text(" (" + terms[term] + ")"))
                    defined.add(term)
                offset = match.end()
            if replacement:
                replacement.append(nodes.Text(value[offset:]))
                parent.replace(text, replacement)


def setup(app):
    app.add_transform(DefineAcronyms)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}

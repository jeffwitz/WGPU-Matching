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


# A curated print root keeps profiler dumps and thousands of raw data lines
# downloadable in HTML while retaining every scientific explanation in print.
latex_engine = "xelatex"
latex_documents = [
    ("pdf/index", "WGPU-Matching.tex", "WGPU-Matching Technical Handbook", author, "manual", True),
]
latex_show_urls = "footnote"
latex_domain_indices = False
latex_use_xindy = False
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "classoptions": ",oneside,openany",
    "fncychap": "",
    "printindex": r"\begingroup\footnotesize\raggedright\printindex\endgroup",
    "fontpkg": r"""
\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}
\setsansfont{TeX Gyre Heros}[Scale=MatchLowercase]
\setmonofont{DejaVu Sans Mono}[Scale=0.78]
\usepackage{unicode-math}
\setmathfont{TeX Gyre Pagella Math}
\setmathfont{TeX Gyre Pagella Math}[version=bold]
""",
    "geometry": r"\usepackage[top=24mm,bottom=25mm,left=23mm,right=23mm,headheight=15pt]{geometry}",
    "sphinxsetup": "TitleColor={RGB}{19,52,77},InnerLinkColor={RGB}{0,99,137},OuterLinkColor={RGB}{0,99,137},VerbatimColor={RGB}{245,248,250},VerbatimBorderColor={RGB}{210,221,228},verbatimwithframe=true,verbatimforcewraps=true,div.seealso_title-icon={},TableRowColorHeader={RGB}{228,237,243},TableRowColorOdd={RGB}{246,249,251},TableRowColorEven={RGB}{255,255,255}",
    "preamble": r"""
\usepackage{titlesec}
\AtBeginDocument{\hypersetup{
  pdftitle={WGPU-Matching Technical Handbook},
  pdfauthor={WGPU-Matching contributors},
  pdfsubject={Feature detection, robust field projection and digital image correlation initialization},
  pdfkeywords={digital image correlation, feature matching, WebGPU, affine initialization},
  pdfdisplaydoctitle=true}}
\usepackage{truncate}
\usepackage{needspace}
\definecolor{HandbookNavy}{RGB}{16,36,54}
\definecolor{HandbookTeal}{RGB}{0,112,138}
\definecolor{HandbookMuted}{RGB}{91,109,123}
\titleformat{\chapter}[display]
  {\normalfont\sffamily\bfseries\color{HandbookNavy}\raggedright\hyphenpenalty=10000\exhyphenpenalty=10000}
  {\color{HandbookTeal}\large\chaptertitlename\ \thechapter}
  {8pt}{\Huge}
\titlespacing*{\chapter}{0pt}{0pt}{26pt}
\titleformat{\section}{\Large\sffamily\bfseries\color{HandbookNavy}\raggedright\hyphenpenalty=10000}{\thesection}{0.7em}{}
\titleformat{\subsection}{\large\sffamily\bfseries\color{HandbookNavy}\raggedright\hyphenpenalty=10000}{\thesubsection}{0.7em}{}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 1pt}
\linespread{1.06}
\setcounter{tocdepth}{1}
\setcounter{secnumdepth}{2}
\emergencystretch=2em
\fancypagestyle{normal}{
  \fancyhf{}
  \fancyhead[L]{\sffamily\small\color{HandbookMuted}WGPU-Matching}
  \fancyhead[R]{\sffamily\small\color{HandbookMuted}\truncate{0.66\textwidth}{\nouppercase{\leftmark}}}
  \fancyfoot[L]{\sffamily\footnotesize Technical handbook | November 2026}
  \fancyfoot[R]{\sffamily\small\thepage}
  \renewcommand{\headrulewidth}{0.3pt}
  \renewcommand{\footrulewidth}{0pt}
}
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[L]{\sffamily\footnotesize WGPU-Matching | Technical handbook}
  \fancyfoot[R]{\sffamily\small\thepage}
  \renewcommand{\headrulewidth}{0pt}
}
""",
    "maketitle": r"""
\begin{titlepage}
\pagecolor{HandbookNavy}\color{white}
\sffamily
\vspace*{18mm}
{\large RESEARCH SOFTWARE / SCIENTIFIC COMPUTING}\par
\vspace{25mm}
{\fontsize{40}{46}\selectfont\bfseries WGPU-Matching}\par
\vspace{9mm}
{\fontsize{25}{32}\selectfont Technical handbook}\par
\vspace{14mm}
{\Large Feature detection, correspondence search,\\
robust field projection and DIC initialization}\par
\vspace{7mm}
{\normalsize DIC means digital image correlation.\\
WGPU names the project's WebGPU implementation.\\
WebGPU is a portable graphics processing unit interface.}\par
\vfill
{\large Python-first learning path}\par
\vspace{5mm}
Tutorials / How-to guides / Explanations / Reference\par
\vspace{12mm}
{\large November 2026 edition}\par
\vspace{4mm}
Measured evidence through 60 percent synthetic traction\par
Experimental image comparisons and explicit failure cases\par
\vspace{14mm}
{\small WGPU-Matching contributors\\
{\hypersetup{urlcolor=white}\url{https://github.com/jeffwitz/WGPU-Matching}}}\par
\end{titlepage}
\clearpage\nopagecolor\color{black}
""",
}


def normalize_print_dashes(app, doctree, docname):
    """Keep print punctuation portable without changing code or equations."""
    if app.builder.format != "latex":
        return
    from docutils import nodes

    # The shared transform already inserts per-page definitions. Suppress
    # the LaTeX writer's separate global expansion to avoid duplicate wording.
    for abbreviation in doctree.findall(nodes.abbreviation):
        abbreviation["explanation"] = ""
    translation = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-"})
    for node in list(doctree.findall(nodes.Text)):
        ancestor = node.parent
        protected = False
        while ancestor is not None:
            if isinstance(
                ancestor, (nodes.literal, nodes.literal_block, nodes.math, nodes.math_block)
            ):
                protected = True
                break
            ancestor = ancestor.parent
        if not protected and str(node).translate(translation) != str(node):
            node.parent.replace(node, nodes.Text(str(node).translate(translation)))


def select_print_root(app):
    # Keep the alternate print toctree from changing HTML breadcrumbs.
    if app.builder.format != "latex":
        app.config.exclude_patterns = [*app.config.exclude_patterns, "pdf/index.rst"]


def setup(app):
    app.connect("builder-inited", select_print_root)
    app.connect("doctree-resolved", normalize_print_dashes)

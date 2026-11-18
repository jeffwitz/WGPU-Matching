#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 paper/audit_revision.py
python3 paper/make_figures.py
python3 paper/make_method_figures.py
# The existing field oracle is experimental; production inference is unchanged.
.venv/bin/python paper/check_revision.py --geometry
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex > supplement-build.log 2>&1
cp main.pdf manuscript_revised.pdf
cd ..
python3 paper/check_revision.py

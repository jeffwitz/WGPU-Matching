#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
PYTHON=.venv/bin/python
"$PYTHON" python/benchmarks/qualify_hudge_strains.py --family FEA --levels 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 50 75 100 125 150 175 191
"$PYTHON" python/benchmarks/qualify_hudge_strains.py --family ClosedFormExpression --levels 0 1 10 25 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 60 75 100 150 200 250 300 400 500
for support in 64 32; do
  for family in FEA ClosedFormExpression; do
    "$PYTHON" python/benchmarks/qualify_hudge_silhouette.py --family "$family" --subset "$support"
  done
done
"$PYTHON" python/benchmarks/audit_hudge_strains.py
python3 python/benchmarks/summarize_hudge_strains.py
python3 python/benchmarks/plot_hudge_strains.py
# Existing result directories are resumed, not silently overwritten.
# To perform a clean rerun, pass a new --output/--root to each script.
python3 python/benchmarks/report_hudge_strains.py

"$PYTHON" -m pytest -q python/benchmarks/tests/test_hudge_benchmark.py
python3 paper/make_silhouette_method_figure.py
python3 paper/make_silhouette_explanation_figures.py
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error hudge_strains_addendum.tex)

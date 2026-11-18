#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
.venv/bin/python -m pytest -q python/benchmarks/tests/test_hudge_affine_restart.py python/benchmarks/tests/test_hudge_benchmark.py
.venv/bin/python python/benchmarks/qualify_hudge_affine_restart.py --levels $(seq 1 191) --output results/hudge_affine_restart_20261115/rerun/all_levels
python3 python/benchmarks/report_hudge_affine_restart.py --root results/hudge_affine_restart_20261115/rerun

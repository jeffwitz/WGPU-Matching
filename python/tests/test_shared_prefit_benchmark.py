"""Test the paired-prefit comparison detects distinct output states."""

import numpy as np
from benchmark_shared_prefit import parity


def test_shared_prefit_parity_checks_affine_and_ncc_states(tmp_path):
    seed = 123
    folder = f"windows_seed{seed}"
    separate, shared = tmp_path / "separate", tmp_path / "shared"
    paths = [separate / "affine_only" / folder, separate / "affine_ncc" / folder,
             shared / "affine_ncc" / folder]
    for path in paths:
        path.mkdir(parents=True)
    base = {"initial_parameters": np.zeros((2, 6)), "icgn_status": np.ones(2),
            "icgn_parameters": np.zeros((2, 6)), "visible": np.ones(2), "truth_parameters": np.zeros((2, 6))}
    for path in paths[:2]:
        np.savez(path / "pair.npz", **base)
    np.savez(paths[2] / "pair.npz", **base, pre_ncc_initial_parameters=base["initial_parameters"],
             pre_ncc_icgn_status=base["icgn_status"], pre_ncc_icgn_parameters=base["icgn_parameters"])
    assert all(item["exact"] for item in parity(separate, shared, seed).values())
    changed = dict(base)
    changed["initial_parameters"] = np.ones((2, 6))
    np.savez(paths[2] / "pair.npz", **changed, pre_ncc_initial_parameters=base["initial_parameters"],
             pre_ncc_icgn_status=base["icgn_status"], pre_ncc_icgn_parameters=base["icgn_parameters"])
    result = parity(separate, shared, seed)
    assert result["affine_initial"]["exact"]
    assert result["ncc_initial"]["differing_values"] == 12

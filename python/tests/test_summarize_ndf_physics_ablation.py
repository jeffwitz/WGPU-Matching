from __future__ import annotations

import pytest
from summarize_ndf_physics_ablation import summarize


def _case(mode: str, texture_seed: int, correct: int) -> dict[str, object]:
    return {
        "mode": mode,
        "texture_seed": texture_seed,
        "warp_seed": texture_seed + 4_000,
        "visible_windows": 100,
        "correct_converged_lt2px": correct,
        "inference_s_cpu": 0.2,
        "icgn_s": 0.3,
    }


def test_summary_uses_paired_visible_counts_and_complete_timed_pipeline():
    control = {
        ("heterogeneous", 1, 4_001): _case("heterogeneous", 1, 10),
        ("heterogeneous", 2, 4_002): _case("heterogeneous", 2, 20),
    }
    multiscale = {
        ("heterogeneous", 1, 4_001): _case("heterogeneous", 1, 15),
        ("heterogeneous", 2, 4_002): _case("heterogeneous", 2, 20),
    }

    result = summarize(control, multiscale)["families"]["heterogeneous"]

    assert result["visible_windows"] == 200
    assert result["control_correct"] == 30
    assert result["multiscale_correct"] == 35
    assert result["positive_pairs"] == 1
    assert result["equal_pairs"] == 1
    assert result["negative_pairs"] == 0
    assert result["control_seconds_sum"] == pytest.approx(1.0)
    assert result["multiscale_correct_per_second"] == pytest.approx(35.0)


def test_summary_rejects_unpaired_cases():
    control = {("translation", 1, 4_001): _case("translation", 1, 10)}
    multiscale = {("translation", 2, 4_002): _case("translation", 2, 10)}

    with pytest.raises(ValueError, match="Unpaired inputs"):
        summarize(control, multiscale)

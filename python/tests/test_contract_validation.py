from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from wgpu_matching import CPMOptions, DeepMatchingOptions, HORGOptions, ProjectionOptions
from wgpu_matching.contracts import validate_matcher_options

ROOT = Path(__file__).resolve().parents[2]


def test_contract_constraints_cover_every_declared_option() -> None:
    contract = json.loads((ROOT / "python/src/wgpu_matching/matcher-contracts.json").read_text())
    options = {
        "deepmatching": asdict(DeepMatchingOptions()),
        "horg": asdict(HORGOptions()),
        "cpm": asdict(CPMOptions()),
        "projection": asdict(ProjectionOptions()),
    }
    for algorithm, spec in contract["algorithms"].items():
        assert set(spec["constraints"]) <= set(spec["defaults"])
        for parameter in spec["constraints"]:
            assert spec["option_keys"][parameter]["python"] in options[algorithm]


@pytest.mark.parametrize(
    ("algorithm", "options", "message"),
    [
        (
            "deepmatching",
            {**asdict(DeepMatchingOptions()), "downscale_levels": 9},
            "downscale_levels",
        ),
        (
            "deepmatching",
            {**asdict(DeepMatchingOptions()), "search_radius_px": -1},
            "search_radius_px",
        ),
        ("deepmatching", {**asdict(DeepMatchingOptions()), "minimum_score": 1.1}, "minimum_score"),
        ("deepmatching", {**asdict(DeepMatchingOptions()), "patch_size": 12.5}, "patch_size_px"),
        ("horg", {**asdict(HORGOptions()), "max_features": 0}, "max_features"),
        ("horg", {**asdict(HORGOptions()), "reduction": 1.5}, "reduction"),
        ("horg", {**asdict(HORGOptions()), "seed_points": ((-1, 10),)}, "inside the image"),
        (
            "horg",
            {**asdict(HORGOptions()), "seed_points": ((float("nan"), 10),)},
            "inside the image",
        ),
        ("cpm", {**asdict(CPMOptions()), "matching_support_px": 4}, "matching_support_px"),
        ("cpm", {**asdict(CPMOptions()), "initial_hypotheses": 65}, "initial_hypotheses"),
        ("cpm", {**asdict(CPMOptions()), "seed": 2**32}, "seed"),
        ("projection", {**asdict(ProjectionOptions()), "basis": "thin_plate"}, "basis"),
        (
            "projection",
            {**asdict(ProjectionOptions()), "max_irls_iterations": 0},
            "max_irls_iterations",
        ),
        ("projection", {**asdict(ProjectionOptions()), "irls_damping": 1.0}, "irls_damping"),
    ],
)
def test_invalid_options_are_rejected_by_the_shared_contract(
    algorithm: str, options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_matcher_options(algorithm, options, width=512, height=384)


def test_manual_seed_limit_is_shared() -> None:
    seeds = tuple((float(index), 10.0) for index in range(513))
    options = {**asdict(DeepMatchingOptions()), "seed_points": seeds}
    with pytest.raises(ValueError, match="at most 512"):
        validate_matcher_options("deepmatching", options, width=1024, height=384)


def test_empty_manual_points_are_allowed_to_fall_back_to_grid() -> None:
    for algorithm, options in (
        ("deepmatching", asdict(DeepMatchingOptions(seed_points=()))),
        ("horg", asdict(HORGOptions(seed_points=()))),
        ("cpm", asdict(CPMOptions(seed_points=()))),
    ):
        validate_matcher_options(algorithm, options, width=512, height=384)

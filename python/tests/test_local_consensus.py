"""Check that geometric verification retains observed points, not fitted ones."""

import numpy as np
import pytest
from wgpu_matching import (
    CorrespondenceSet,
    CPMOptions,
    DeepMatchingOptions,
    WGPUMatching,
)

from python.benchmarks.geometric_consensus import LocalConsensus, bounded_pairs


def test_capacity_reduction_is_deterministic_spatial_and_preserves_measurements():
    xx, yy = np.meshgrid(np.arange(1, 129), np.arange(1, 97))
    source = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    observed = CorrespondenceSet(
        source, source + [3, 2], (160, 128), "fixture", accepted=np.ones(len(source), bool)
    )
    first, info = bounded_pairs(observed, limit=80, tiles=(4, 3))
    second, _ = bounded_pairs(observed, limit=80, tiles=(4, 3))
    assert len(first.source_xy) == 80
    assert first.match_ids == second.match_ids
    np.testing.assert_array_equal(first.source_xy, source[list(first.match_ids)])
    np.testing.assert_array_equal(first.target_xy, observed.target_xy[list(first.match_ids)])
    assert info["applied"]
    assert len(np.unique((first.source_xy / [40, 43]).astype(int), axis=0)) >= 9


def device():
    try:
        return WGPUMatching(device_type="gpu")
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_exporting_local_affine_coefficients_preserves_default_results():
    xx, yy = np.meshgrid(np.linspace(20, 200, 14), np.linspace(20, 160, 12))
    source = np.column_stack((xx.ravel(), yy.ravel()))
    matrix = np.asarray([[1.02, 0.03], [-0.01, 1.04]])
    target = source @ matrix.T + [6, -3]
    pairs = np.column_stack((source, target))
    seeds = np.asarray([[64, 64], [128, 96], [176, 128]], float)
    with device() as matcher:
        original = matcher.fit_local_affine(pairs, seeds)
        exported = matcher.fit_local_affine(pairs, seeds, include_affine_models=True)
    assert all(m.usable for m in exported.matches)
    np.testing.assert_allclose(
        [[m.dx, m.dy] for m in original.matches],
        [[m.dx, m.dy] for m in exported.matches],
        atol=1e-6,
    )
    for centre, raw in zip(seeds, exported.affine_models, strict=True):
        np.testing.assert_allclose(raw[:2], matrix[0], atol=2e-4)
        np.testing.assert_allclose(raw[4:6], matrix[1], atol=2e-4)
        np.testing.assert_allclose([raw[2], raw[6]], centre @ matrix.T + [6, -3], atol=2e-3)


def test_gpu_filter_removes_false_pairs_without_inventing_measurements():
    xx, yy = np.meshgrid(np.linspace(20, 200, 14), np.linspace(20, 160, 12))
    source = np.column_stack((xx.ravel(), yy.ravel()))
    target = source @ np.asarray([[1.02, -0.01], [0.03, 1.04]]) + [6, -3]
    wrong = np.arange(len(source)) % 4 == 0
    target[wrong] += [40, 45]
    observed = CorrespondenceSet(
        source,
        target,
        (320, 240),
        "fixture",
        accepted=np.ones(len(source), bool),
        rejection_reasons=(None,) * len(source),
    )
    with device() as matcher:
        verified, diagnostics = LocalConsensus(matcher).filter(
            observed, np.asarray([[64, 64], [128, 96], [176, 128]], float)
        )
    assert not verified.accepted[wrong].any()
    assert verified.accepted[~wrong].all()
    np.testing.assert_array_equal(verified.source_xy, observed.source_xy)
    np.testing.assert_array_equal(verified.target_xy, observed.target_xy)
    assert diagnostics["local_seed_models"] == 3


def test_native_direct_exports_preserve_initializer_outputs():
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    with device() as matcher:
        for name, options in (
            (
                "match_deepmatching",
                DeepMatchingOptions(
                    patch_size=32,
                    root_patch_size=32,
                    downscale_levels=0,
                    search_radius_px=16,
                    seed_points=((63.5, 63.5),),
                ),
            ),
            (
                "match_cpm",
                CPMOptions(
                    patch_size=32,
                    matching_support_px=16,
                    downscale_levels=0,
                    search_radius_px=16,
                    work_grid_stride=8,
                    levels=2,
                    seed_points=((63.5, 63.5),),
                ),
            ),
        ):
            plain = getattr(matcher, name)(image, image, options=options)
            exported = getattr(matcher, name)(
                image, image, options=options, include_correspondences=True
            )
            assert plain.matches == exported.matches
            observed = CorrespondenceSet.from_match_field(exported)
            assert len(observed.source_xy) > len(exported.matches)
            assert "direct_" in observed.coordinate_mapping["source"]

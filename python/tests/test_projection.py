"""Analytic checks for the method-independent displacement projection."""

from types import SimpleNamespace

import numpy as np
import pytest
from wgpu_matching.projection import (
    CorrespondenceSet,
    ProjectionOptions,
    fit_initialization_field,
)


def _affine_pairs() -> tuple[np.ndarray, tuple[int, int]]:
    x, y = np.meshgrid(np.linspace(20, 480, 18), np.linspace(20, 380, 14))
    source = np.column_stack((x.ravel(), y.ravel()))
    gradient = np.asarray([[0.04, 0.02], [-0.01, -0.03]])
    displacement = source @ gradient.T + np.asarray([8.0, -4.0])
    return np.column_stack((source, source + displacement)), (512, 420)


def test_affine_projection_reproduces_six_patch_parameters() -> None:
    pairs, size = _affine_pairs()
    model = fit_initialization_field(pairs, size, options=ProjectionOptions(basis="affine"))
    patches = model.initialize_patches(np.asarray([[100.0, 100.0]]), 128)
    assert model.effective_basis == "affine"
    np.testing.assert_allclose(
        patches[0].parameters, [14.0, -8.0, 0.04, 0.02, -0.01, -0.03], atol=1e-8
    )
    assert patches[0].support_rms_px < 1e-8


def test_gaussian_rbf_selection_preserves_exact_affine_field() -> None:
    pairs, size = _affine_pairs()
    model = fit_initialization_field(pairs, size, options=ProjectionOptions(basis="gaussian_rbf"))
    patch = model.initialize_patches(np.asarray([[100.0, 100.0]]), 128)[0]
    assert model.effective_basis in {"affine", "gaussian_rbf"}
    np.testing.assert_allclose(patch.parameters, [14.0, -8.0, 0.04, 0.02, -0.01, -0.03], atol=1e-7)
    assert len(model.diagnostics["candidate_scores"]) == 28
    assert model.effective_basis == "affine"


def test_correspondence_coordinates_are_validated_and_deduplicated() -> None:
    pairs, size = _affine_pairs()
    pairs = np.vstack((pairs, pairs[0], [np.nan, 1.0, 2.0, 3.0]))
    model = fit_initialization_field(pairs, size, options=ProjectionOptions(basis="affine"))
    assert model.diagnostics["input_count"] == len(pairs)
    assert model.diagnostics["retained_count"] == 246
    assert model.diagnostics["duplicate_count"] == 1
    assert model.diagnostics["excluded_rows"][-1]["reason"] == "non_finite_coordinate"


def test_correspondence_set_keeps_provenance_and_excludes_inferred_rows() -> None:
    source = np.asarray([[10.0, 10.0], [20.0, 10.0], [10.0, 20.0], [20.0, 20.0]])
    target = source + [3.0, -2.0]
    data = CorrespondenceSet(
        source_xy=source,
        target_xy=target,
        image_size=(64, 64),
        matcher_id="fixture",
        match_ids=("a", "b", "c", "d"),
        measurement_kind=("direct_measured", "direct_measured", "direct_measured", "interpolated"),
        raw_scores=np.asarray([0.9, 0.8, 0.7, 1.0]),
    )
    model = fit_initialization_field(data, (64, 64), options=ProjectionOptions(basis="affine"))
    assert model.source_matcher == "fixture"
    assert set(model.correspondence_ids) == {"a", "b", "c"}
    assert dict(zip(model.correspondence_ids, model.diagnostics["retained_raw_scores"], strict=True)) == {
        "a": 0.9,
        "b": 0.8,
        "c": 0.7,
    }
    assert model.diagnostics["excluded_rows"] == [
        {"row": 3, "match_id": "d", "reason": "measurement_kind:interpolated"}
    ]


def test_match_field_adapter_requires_measured_target_coordinates() -> None:
    matches = [
        SimpleNamespace(x=10.0, y=10.0, target_x=13.0, target_y=8.0, usable=True, score=0.9),
        SimpleNamespace(x=20.0, y=10.0, target_x=23.0, target_y=8.0, usable=True, score=0.8),
        SimpleNamespace(x=10.0, y=20.0, target_x=13.0, target_y=18.0, usable=True, score=0.7),
        SimpleNamespace(x=20.0, y=20.0, target_x=None, target_y=None, usable=False, score=0.0),
    ]
    field = SimpleNamespace(
        algorithm="DeepMatching test",
        width=64,
        height=64,
        scale=1,
        metadata={},
        matches=matches,
    )
    pairs = CorrespondenceSet.from_match_field(field)
    model = fit_initialization_field(pairs, (64, 64), options=ProjectionOptions(basis="affine"))
    assert set(model.correspondence_ids) == {0, 1, 2}
    assert model.diagnostics["excluded_rows"] == [
        {"row": 3, "match_id": 3, "reason": "no_direct_target_coordinate"}
    ]


def test_horg_raw_match_metadata_preserves_ratio_rejections() -> None:
    field = SimpleNamespace(
        algorithm="HORG-wgpu-py",
        width=64,
        height=64,
        scale=1,
        metadata={
            "raw_correspondences_xyxy": [
                [10.0, 10.0, 12.0, 8.0],
                [20.0, 10.0, 22.0, 8.0],
                [10.0, 20.0, 12.0, 18.0],
            ],
            "raw_correspondence_ids": [5, 7, 9],
            "raw_match_scores": [0.4, 0.9, 0.6],
            "raw_correspondence_accepted": [True, False, True],
            "raw_correspondence_rejection_reasons": [None, "lowe_ratio_rejected", None],
        },
        matches=[],
    )
    pairs = CorrespondenceSet.from_match_field(field)
    assert pairs.match_ids == (5, 7, 9)
    assert pairs.accepted.tolist() == [True, False, True]
    assert pairs.rejection_reasons[1] == "lowe_ratio_rejected"


def test_preprocessed_pairs_restore_anisotropic_scale_padding_and_crop() -> None:
    native_source = np.asarray([[40.0, 60.0]])
    native_target = np.asarray([[50.0, 70.0]])
    source_processed = native_source * [0.5, 0.25] + [9.0, 19.25]
    target_processed = native_target * [0.4, 0.2] + [7.0, 15.0]
    pairs = CorrespondenceSet.from_preprocessed_pairs(
        source_processed,
        target_processed,
        (128, 128),
        "fixture",
        source_scale_xy=(0.5, 0.25),
        target_scale_xy=(0.4, 0.2),
        source_padding_xy=(10.0, 20.0),
        target_padding_xy=(8.0, 16.0),
        source_crop_origin_xy=(2.0, 3.0),
        target_crop_origin_xy=(2.5, 5.0),
    )
    np.testing.assert_allclose(pairs.source_xy, native_source)
    np.testing.assert_allclose(pairs.target_xy, native_target)
    assert pairs.coordinate_mapping["source_scale_xy"] == [0.5, 0.25]


def test_webgpu_affine_fit_and_patch_projection() -> None:
    from wgpu_matching import WGPUMatching

    pairs, size = _affine_pairs()
    try:
        matcher = WGPUMatching(device_type="gpu")
    except RuntimeError as error:
        pytest.skip(f"no hardware WebGPU adapter: {error}")
    with matcher:
        model = matcher.fit_initialization_field(
            pairs, size, options=ProjectionOptions(basis="affine")
        )
        patches = model.initialize_patches(np.asarray([[100.0, 100.0]]), (128, 96))
    assert model.diagnostics["backend"].startswith("webgpu_")
    assert patches[0].status == "estimated_from_field"
    np.testing.assert_allclose(
        patches[0].parameters, [14.0, -8.0, 0.04, 0.02, -0.01, -0.03], atol=5e-5
    )
    assert patches[0].support_rms_px < 2e-5


def test_webgpu_gaussian_rbf_projection_and_patch_parameters() -> None:
    from wgpu_matching import WGPUMatching

    x, y = np.meshgrid(np.linspace(16, 240, 9), np.linspace(16, 176, 7))
    source = np.column_stack((x.ravel(), y.ravel()))
    radial = source - np.asarray([128.0, 96.0])
    bump = np.exp(-np.sum(radial**2, axis=1) / (2.0 * 45.0**2))
    displacement = np.column_stack(
        (5.0 + 0.01 * source[:, 0] + 30.0 * bump,
         -3.0 + 0.02 * source[:, 1] + 15.0 * bump)
    )
    pairs = CorrespondenceSet(
        source,
        source + displacement,
        (256, 192),
        "analytic-localized-deformation",
    )
    options = ProjectionOptions(
        basis="gaussian_rbf",
        center_counts=(8,),
        width_factors=(1.0, 2.0, 4.0),
        ridge_values=(1e-3,),
    )
    try:
        matcher = WGPUMatching(device_type="gpu")
    except RuntimeError as error:
        pytest.skip(f"no hardware WebGPU adapter: {error}")
    with matcher:
        model = matcher.fit_initialization_field(pairs, (256, 192), options=options)
        patches = model.initialize_patches(np.asarray([[128.0, 96.0]]), 64)
        predicted = model.evaluate(source)
    assert model.effective_basis == "gaussian_rbf"
    assert model.diagnostics["backend"].startswith("webgpu_")
    assert model.diagnostics["selected_center_count"] == 8
    assert patches[0].status == "estimated_from_field"
    assert np.isfinite(predicted).all()
    assert np.isfinite(patches[0].parameters).all()
    assert patches[0].support_rms_px is not None

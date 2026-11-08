"""Scientific safeguards for the unified benchmark and its visualizations."""

from types import SimpleNamespace

import numpy as np
from PIL import Image
from wgpu_matching import CorrespondenceSet

from python.benchmarks.qualification_visuals import (
    assess_parameters,
    render_pair,
    restore_centres,
    warp_preview,
)


def test_inverse_resize_preserves_pixel_centres_and_anisotropic_coordinates():
    points = np.asarray([[0, 0], [319.5, 179.5]])
    restored = restore_centres(points, (3840, 2160), (640, 360))
    np.testing.assert_allclose(restored, [[2.5, 2.5], [1919.5, 1079.5]])
    np.testing.assert_allclose(
        restore_centres(np.asarray([[0, 0]]), (128, 96), (64, 24)), [[0.5, 1.5]]
    )


def test_five_pixel_threshold_is_strict_and_missing_estimates_reduce_coverage():
    seeds = np.asarray([[10, 10], [20, 10], [30, 10]], float)
    truth = {"stretch_x": 1.0, "stretch_y": 1.0}
    result = assess_parameters(
        [[4.999, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0], None], seeds, truth, (64, 64)
    )
    assert not result["pass_below_5px"]
    assert result["correct_lt5"] == 1
    assert result["false_estimates_ge5"] == 1
    assert result["estimated"] == 2
    assert result["correct_coverage"] == 1 / 3
    assert result["conditional_precision"] == 1 / 2
    assert assess_parameters([None] * 3, seeds, None, (64, 64))["pass_below_5px"] is None


def test_warp_uses_forward_displacement_to_sample_target_without_truth():
    reference = np.arange(32 * 24, dtype=np.uint16).reshape(24, 32).astype(np.uint8)
    target = np.zeros_like(reference)
    target[:, 3:] = reference[:, :-3]

    class Translation:
        def evaluate(self, points):
            return np.tile([3.0, 0.0], (len(points), 1))

    ref, corrected, metadata = warp_preview(reference, target, Translation(), width=32)
    np.testing.assert_array_equal(corrected[:, :-3], ref[:, :-3])
    assert not corrected[:, -3:].any()
    assert metadata["valid_warp_fraction_of_reference_roi"] == 29 / 32


def test_failed_visualization_is_annotated_and_retains_two_frames(tmp_path):
    reference = np.zeros((24, 32), dtype=np.uint8)
    target = np.full_like(reference, 100)
    result = render_pair(
        reference, target, None, tmp_path, label="test", verdict="FAIL", detail="no field", width=32
    )
    assert result["visualization_field"] == "identity_no_model"
    assert result["corrected_png"] == "image1_def_uncorrected.png"
    with Image.open(tmp_path / "comparison.gif") as gif:
        assert gif.n_frames == 2
        assert gif.info["loop"] == 0


def test_cpm_seed_affine_estimates_are_not_direct_correspondences():
    field = SimpleNamespace(
        algorithm="CPM-WebGPU",
        width=64,
        height=64,
        metadata={},
        scale=1,
        matches=[SimpleNamespace(x=20, y=20, target_x=22, target_y=19, usable=True, score=0.9)],
    )
    pairs = CorrespondenceSet.from_match_field(field)
    assert not pairs.accepted.any()
    assert pairs.measurement_kind == ("unavailable",)


def test_direct_work_grid_export_preserves_source_and_rejection_provenance():
    field = SimpleNamespace(
        algorithm="CPM-WebGPU",
        width=64,
        height=64,
        matches=[],
        metadata={
            "raw_correspondences_xyxy": [[8, 8, 10, 7], [16, 8, 18, 7]],
            "raw_correspondence_accepted": [True, False],
            "raw_correspondence_kind": "direct_CPM_work_grid_matches",
            "raw_correspondence_rejection_reasons": [None, "work_grid_rejected"],
        },
    )
    pairs = CorrespondenceSet.from_match_field(field)
    np.testing.assert_array_equal(pairs.source_xy, [[8, 8], [16, 8]])
    assert pairs.accepted.tolist() == [True, False]
    assert "CPM" in pairs.coordinate_mapping["source"]

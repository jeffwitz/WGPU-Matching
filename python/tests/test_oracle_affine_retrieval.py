import numpy as np
from benchmark_oracle_affine_descriptor_retrieval import (
    canonical_points,
    ranking_metrics,
    warp_target_to_material_frame,
)


def test_target_point_is_mapped_to_its_source_frame_coordinate():
    jacobian = np.asarray([[1.2, 0.15], [-0.08, 0.9]], dtype=np.float64)
    source_anchor = np.asarray([30.0, 42.0])
    target_anchor = np.asarray([110.0, 76.0])
    source_point = source_anchor + np.asarray([7.0, -5.0])
    target_point = target_anchor + jacobian @ (source_point - source_anchor)

    mapped = canonical_points(
        target_point[None], source_anchor, target_anchor, jacobian
    )

    np.testing.assert_allclose(mapped[0], source_point, atol=1e-6)


def test_affine_pullback_uses_forward_map_and_transforms_validity():
    height, width = 96, 128
    yy, xx = np.mgrid[:height, :width]
    target = (xx + 2 * yy).astype(np.float32)
    valid = np.ones((height, width), dtype=bool)
    jacobian = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    anchor = np.asarray([width / 2, height / 2])
    target_anchor = anchor + np.asarray([3.0, -2.0])

    canonical, canonical_valid = warp_target_to_material_frame(
        target, valid, anchor, target_anchor, jacobian
    )

    np.testing.assert_allclose(canonical[40, 50], target[38, 53], atol=1e-5)
    assert canonical_valid[40, 50]
    assert not canonical_valid[1, 1]


def test_retrieval_scores_use_positive_rank_and_visible_query_denominators():
    source = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    target = np.asarray([[50.0, 50.0], [10.5, 0.0], [20.0, 0.0]])
    similarities = np.asarray([[0.9, 0.1, 0.0], [0.1, 0.9, 0.2], [0.1, 0.2, 0.9]])

    result = ranking_metrics(similarities, source, target)

    assert result["queries"] == 3
    assert result["repeatable_le5px"] == 2
    assert result["top1_correct_le2px"] == 2
    assert result["top5_retrieval"] == 2

from __future__ import annotations

import numpy as np
import pytest
from wgpu_matching.features import (
    DescribedFeatures,
    FeatureSet,
    mutual_nearest_cosine,
)


def test_feature_set_preserves_float_pixel_centres() -> None:
    features = FeatureSet(
        xy=np.array([[10.25, 20.75], [31.0, 42.0]], dtype=np.float64),
        scores=np.array([0.9, 0.8]),
    )
    np.testing.assert_array_equal(features.xy, [[10.25, 20.75], [31.0, 42.0]])
    assert features.xy.dtype == np.float32
    assert features.valid is not None and features.valid.all()


def test_feature_set_rejects_misaligned_scores() -> None:
    with pytest.raises(ValueError, match="scores"):
        FeatureSet(xy=np.zeros((2, 2)), scores=np.zeros(1))


def test_described_features_require_one_descriptor_per_keypoint() -> None:
    features = FeatureSet(xy=np.zeros((2, 2)), scores=np.ones(2))
    with pytest.raises(ValueError, match="aligned"):
        DescribedFeatures(features=features, descriptors=np.zeros((1, 8)))


def test_mutual_cosine_matching_is_reciprocal_and_chunk_invariant() -> None:
    coordinates_a = np.array([[1.25, 2.5], [10.0, 12.0], [30.0, 32.0]], np.float32)
    coordinates_b = np.array([[2.25, 3.5], [11.0, 13.0], [31.0, 33.0]], np.float32)
    descriptors_a = np.eye(3, dtype=np.float32)
    descriptors_b = np.eye(3, dtype=np.float32)
    a = DescribedFeatures(FeatureSet(coordinates_a, np.ones(3)), descriptors_a)
    b = DescribedFeatures(FeatureSet(coordinates_b, np.ones(3)), descriptors_b)

    one_at_a_time = mutual_nearest_cosine(a, b, source_chunk_size=1)
    all_at_once = mutual_nearest_cosine(a, b, source_chunk_size=16)

    np.testing.assert_array_equal(
        one_at_a_time.points, np.column_stack((coordinates_a, coordinates_b))
    )
    np.testing.assert_array_equal(one_at_a_time.points, all_at_once.points)
    np.testing.assert_array_equal(one_at_a_time.cosine_similarity, np.ones(3))


def test_mutual_cosine_matching_honours_similarity_threshold() -> None:
    coordinates = np.array([[0.0, 0.0], [1.0, 1.0]], np.float32)
    descriptors = np.eye(2, dtype=np.float32)
    described = DescribedFeatures(FeatureSet(coordinates, np.ones(2)), descriptors)
    result = mutual_nearest_cosine(described, described, min_cosine_similarity=1.0)
    assert result.points.shape == (2, 4)


def test_horg_keypoints_are_exposed_for_a_common_descriptor() -> None:
    from wgpu_matching import WGPUMatching

    rng = np.random.default_rng(81)
    image = rng.integers(0, 256, size=(160, 192), dtype=np.uint8)
    try:
        matcher = WGPUMatching(device_type="cpu")
    except RuntimeError as error:
        pytest.skip(f"No software WebGPU adapter is available: {error}")
    with matcher:
        features = matcher.detect_horg(image, max_keypoints=64, reduction=4)

    assert features.xy.ndim == 2 and features.xy.shape[1] == 2
    assert len(features.xy) <= 64
    assert features.metadata["detector"] == "HORG-SIFT-DoG"
    assert np.isfinite(features.xy).all()
    assert np.isfinite(features.scores).all()
    assert features.scales is not None

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


def test_webgpu_mutual_cosine_matches_numpy_reference() -> None:
    from wgpu_matching import WGPUMatching
    from wgpu_matching.features import mutual_nearest_cosine_wgpu

    rng = np.random.default_rng(402)
    descriptors_a = rng.normal(size=(37, 24)).astype(np.float32)
    descriptors_b = descriptors_a.copy()
    descriptors_b += rng.normal(scale=0.01, size=descriptors_b.shape).astype(np.float32)
    coordinates = rng.uniform(0, 256, size=(37, 2)).astype(np.float32)
    reference = DescribedFeatures(FeatureSet(coordinates, np.ones(37)), descriptors_a)
    target = DescribedFeatures(FeatureSet(coordinates + [2.5, -1.25], np.ones(37)), descriptors_b)
    expected = mutual_nearest_cosine(reference, target, min_cosine_similarity=0.2)
    try:
        matcher = WGPUMatching(device_type="cpu")
    except RuntimeError as error:
        pytest.skip(f"No software WebGPU adapter is available: {error}")
    with matcher:
        actual = mutual_nearest_cosine_wgpu(
            matcher.device, reference, target, min_cosine_similarity=0.2
        )
    np.testing.assert_array_equal(actual.points, expected.points)
    np.testing.assert_array_equal(actual.reference_indices, expected.reference_indices)
    np.testing.assert_array_equal(actual.target_indices, expected.target_indices)
    np.testing.assert_allclose(actual.cosine_similarity, expected.cosine_similarity, atol=2e-6)


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


def test_horg_compact_descriptor_preserves_supplied_points() -> None:
    from wgpu_matching import WGPUMatching

    rng = np.random.default_rng(8108)
    image = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    features = FeatureSet(
        xy=np.array([[40.25, 41.75], [64.0, 64.0], [88.5, 77.25]], dtype=np.float32),
        scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
    )
    try:
        matcher = WGPUMatching(device_type="cpu")
    except RuntimeError as error:
        pytest.skip(f"No software WebGPU adapter is available: {error}")
    with matcher:
        described = matcher.describe_horg_keypoints(image, features, reduction=2)

    assert described.descriptors.shape == (3, 64)
    assert described.metadata["descriptor"] == "HORG-64 supplied-keypoint variant"
    np.testing.assert_array_equal(described.features.xy, features.xy)
    assert np.isfinite(described.descriptors).all()
    np.testing.assert_allclose(np.linalg.norm(described.descriptors, axis=1), 1.0, atol=2e-5)


def test_trained_compact_dic_descriptor_runs_and_matches_numpy(tmp_path) -> None:
    import json

    from wgpu_matching import WGPUMatching

    rng = np.random.default_rng(223)
    image = rng.integers(0, 256, size=(160, 160), dtype=np.uint8)
    features = FeatureSet(
        xy=np.array([[55.25, 56.75], [80.0, 80.0], [105.5, 101.25]], dtype=np.float32),
        scores=np.ones(3, dtype=np.float32),
    )
    matrix = rng.normal(0, 0.08, (64, 64)).astype(np.float32)
    bias = rng.normal(0, 0.01, 64).astype(np.float32)
    weights_path = tmp_path / "compact.npz"
    np.savez_compressed(
        weights_path,
        metadata_json=np.frombuffer(
            json.dumps(
                {
                    "architecture": "linear",
                    "image_reductions": [4],
                    "input_dimension": 64,
                    "hidden_dimension": 0,
                    "output_dimension": 64,
                }
            ).encode(),
            dtype=np.uint8,
        ),
        first_weight=matrix,
        first_bias=bias,
        second_weight=np.empty((0, 0), dtype=np.float32),
        second_bias=np.empty((0,), dtype=np.float32),
    )
    try:
        matcher = WGPUMatching(device_type="cpu")
    except RuntimeError as error:
        pytest.skip(f"No software WebGPU adapter is available: {error}")
    with matcher:
        horg = matcher.describe_horg_keypoints(image, features, reduction=4)
        actual = matcher.describe_dic_keypoints(image, features, weights_path=weights_path)

    expected = horg.descriptors @ matrix.T + bias
    expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-12)
    np.testing.assert_allclose(actual.descriptors, expected, rtol=2e-5, atol=2e-5)
    assert actual.metadata["architecture"] == "linear"
    assert actual.metadata["weights_sha256"]


@pytest.mark.parametrize(
    ("architecture", "input_dim", "hidden_dim"),
    [("residual_mlp", 64, 96), ("multiscale_mlp", 192, 96)],
)
def test_compact_dic_mlp_variants_match_numpy(
    tmp_path, architecture, input_dim, hidden_dim
) -> None:
    import json

    from wgpu_matching.dic_descriptor import CompactDescriptorBackend, CompactDescriptorWeights

    rng = np.random.default_rng(input_dim)
    first_weight = rng.normal(0, 0.03, (hidden_dim, input_dim)).astype(np.float32)
    first_bias = rng.normal(0, 0.01, hidden_dim).astype(np.float32)
    second_weight = rng.normal(0, 0.03, (64, hidden_dim)).astype(np.float32)
    second_bias = rng.normal(0, 0.01, 64).astype(np.float32)
    path = tmp_path / f"{architecture}.npz"
    np.savez_compressed(
        path,
        metadata_json=np.frombuffer(
            json.dumps(
                {
                    "architecture": architecture,
                    "image_reductions": [2, 4, 8] if input_dim == 192 else [4],
                    "input_dimension": input_dim,
                    "hidden_dimension": hidden_dim,
                    "output_dimension": 64,
                }
            ).encode(),
            dtype=np.uint8,
        ),
        first_weight=first_weight,
        first_bias=first_bias,
        second_weight=second_weight,
        second_bias=second_bias,
    )
    inputs = rng.normal(size=(11, input_dim)).astype(np.float32)
    try:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync(feature_level="core")
        device = adapter.request_device_sync()
    except RuntimeError as error:
        pytest.skip(f"No software WebGPU adapter is available: {error}")
    weights = CompactDescriptorWeights.load(path)
    actual = CompactDescriptorBackend(device).project(inputs, weights)
    hidden = np.maximum(inputs @ first_weight.T + first_bias, 0.0)
    expected = hidden @ second_weight.T + second_bias
    if architecture == "residual_mlp":
        expected += 0.25 * inputs
    expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-12)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_dic_patch_cnn_wgsl_matches_numpy_reference(tmp_path) -> None:
    import json

    from wgpu_matching import WGPUMatching

    rng = np.random.default_rng(644)
    arrays = {
        "conv1_weight": rng.normal(0, 0.06, (8, 1, 3, 3)).astype(np.float32),
        "conv1_bias": rng.normal(0, 0.02, 8).astype(np.float32),
        "depthwise_weight": rng.normal(0, 0.06, (8, 1, 3, 3)).astype(np.float32),
        "depthwise_bias": rng.normal(0, 0.02, 8).astype(np.float32),
        "projection_weight": rng.normal(0, 0.04, (64, 128)).astype(np.float32),
        "projection_bias": rng.normal(0, 0.02, 64).astype(np.float32),
    }
    path = tmp_path / "patch_cnn.npz"
    np.savez_compressed(
        path,
        metadata_json=np.frombuffer(
            json.dumps({"architecture": "dic_patch_cnn_v1", "support_px": 64}).encode(),
            dtype=np.uint8,
        ),
        **arrays,
    )
    image = rng.integers(0, 256, size=(160, 160), dtype=np.uint8)
    point = np.array([[80.0, 80.0]], dtype=np.float32)
    features = FeatureSet(point, np.ones(1, dtype=np.float32))
    try:
        matcher = WGPUMatching(device_type="cpu")
    except RuntimeError as error:
        pytest.skip(f"No software WebGPU adapter is available: {error}")
    with matcher:
        actual = matcher.describe_dic_cnn_keypoints(image, features, weights_path=path)

    source = image.astype(np.float32) / 255.0
    conv1 = np.zeros((8, 16, 16), dtype=np.float32)
    for channel in range(8):
        for oy in range(16):
            for ox in range(16):
                value = arrays["conv1_bias"][channel]
                for ky in range(3):
                    for kx in range(3):
                        patch_y, patch_x = 4 * oy + ky - 1, 4 * ox + kx - 1
                        if 0 <= patch_y < 64 and 0 <= patch_x < 64:
                            iy = int(point[0, 1]) + patch_y - 31
                            ix = int(point[0, 0]) + patch_x - 31
                            value += arrays["conv1_weight"][channel, 0, ky, kx] * source[iy, ix]
                conv1[channel, oy, ox] = max(value, 0.0)
    conv2 = np.zeros_like(conv1)
    for channel in range(8):
        for oy in range(16):
            for ox in range(16):
                value = arrays["depthwise_bias"][channel]
                for ky in range(3):
                    for kx in range(3):
                        iy, ix = oy + ky - 1, ox + kx - 1
                        if 0 <= iy < 16 and 0 <= ix < 16:
                            value += (
                                arrays["depthwise_weight"][channel, 0, ky, kx]
                                * conv1[channel, iy, ix]
                            )
                conv2[channel, oy, ox] = max(value, 0.0)
    pooled = np.empty((128,), dtype=np.float32)
    for channel in range(8):
        for cy in range(4):
            for cx in range(4):
                pooled[channel * 16 + cy * 4 + cx] = conv2[
                    channel, cy * 4 : cy * 4 + 4, cx * 4 : cx * 4 + 4
                ].mean()
    expected = arrays["projection_weight"] @ pooled + arrays["projection_bias"]
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(actual.descriptors[0], expected, rtol=3e-4, atol=3e-4)
    assert actual.features.valid[0]
    assert actual.metadata["support_px"] == 64


def test_dic_patch_cnn_wgsl_matches_pytorch_reference(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    import json

    from wgpu_matching import WGPUMatching

    class TorchReference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=4, padding=1)
            self.depthwise = torch.nn.Conv2d(8, 8, 3, padding=1, groups=8)
            self.projection = torch.nn.Linear(128, 64)

        def forward(self, value):
            value = torch.relu(self.conv1(value))
            value = torch.relu(self.depthwise(value))
            value = torch.nn.functional.avg_pool2d(value, 4)
            value = self.projection(value.flatten(1))
            return torch.nn.functional.normalize(value, dim=1, eps=1e-8)

    rng = np.random.default_rng(711)
    arrays = {
        "conv1_weight": rng.normal(0, 0.06, (8, 1, 3, 3)).astype(np.float32),
        "conv1_bias": rng.normal(0, 0.02, 8).astype(np.float32),
        "depthwise_weight": rng.normal(0, 0.06, (8, 1, 3, 3)).astype(np.float32),
        "depthwise_bias": rng.normal(0, 0.02, 8).astype(np.float32),
        "projection_weight": rng.normal(0, 0.04, (64, 128)).astype(np.float32),
        "projection_bias": rng.normal(0, 0.02, 64).astype(np.float32),
    }
    path = tmp_path / "patch_cnn_torch.npz"
    np.savez_compressed(
        path,
        metadata_json=np.frombuffer(
            json.dumps({"architecture": "dic_patch_cnn_v1", "support_px": 64}).encode(),
            dtype=np.uint8,
        ),
        **arrays,
    )
    image = rng.integers(0, 256, size=(192, 192), dtype=np.uint8)
    points = np.array([[80.0, 80.0], [103.25, 96.75]], dtype=np.float32)
    features = FeatureSet(points, np.ones(len(points), dtype=np.float32))
    reference = TorchReference().eval()
    with torch.no_grad():
        reference.conv1.weight.copy_(torch.from_numpy(arrays["conv1_weight"]))
        reference.conv1.bias.copy_(torch.from_numpy(arrays["conv1_bias"]))
        reference.depthwise.weight.copy_(torch.from_numpy(arrays["depthwise_weight"]))
        reference.depthwise.bias.copy_(torch.from_numpy(arrays["depthwise_bias"]))
        reference.projection.weight.copy_(torch.from_numpy(arrays["projection_weight"]))
        reference.projection.bias.copy_(torch.from_numpy(arrays["projection_bias"]))
        source = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]
        offsets = torch.arange(64, dtype=torch.float32) - 31.0
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        patches = []
        for x, y in points:
            grid_x = (x + xx + 0.5) * (2.0 / image.shape[1]) - 1.0
            grid_y = (y + yy + 0.5) * (2.0 / image.shape[0]) - 1.0
            grid = torch.stack((grid_x, grid_y), dim=-1)[None]
            patches.append(
                torch.nn.functional.grid_sample(
                    source, grid, mode="bilinear", align_corners=False
                )
            )
        expected = reference(torch.cat(patches, dim=0)).numpy()

    try:
        matcher = WGPUMatching(device_type="cpu")
    except RuntimeError as error:
        pytest.skip(f"No software WebGPU adapter is available: {error}")
    with matcher:
        actual = matcher.describe_dic_cnn_keypoints(image, features, weights_path=path)
    np.testing.assert_allclose(actual.descriptors, expected, rtol=3e-4, atol=3e-4)

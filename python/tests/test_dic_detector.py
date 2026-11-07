from __future__ import annotations

import numpy as np
import pytest
from wgpu_matching import DICDetectorOptions, WGPUMatching


def test_wgsl_dic_detector_is_deterministic_and_respects_mask_support() -> None:
    rng = np.random.default_rng(54)
    image = rng.integers(0, 256, size=(128, 160), dtype=np.uint8)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[20:108, 24:136] = 255
    options = DICDetectorOptions(
        stride_px=4,
        tensor_radius_px=4,
        max_keypoints=48,
        tile_columns=4,
        tile_rows=3,
        minimum_eigenvalue=1e-6,
    )
    try:
        matcher = WGPUMatching(device_type="cpu")
    except RuntimeError as error:
        pytest.skip(f"No software WebGPU adapter is available: {error}")
    with matcher:
        first = matcher.detect_dic(image, options=options, roi_mask=mask)
        second = matcher.detect_dic(image, options=options, roi_mask=mask)

    assert 20 <= len(first.features.xy) <= options.max_keypoints
    np.testing.assert_array_equal(first.features.xy, second.features.xy)
    np.testing.assert_array_equal(first.features.scores, second.features.scores)
    assert np.all(first.features.xy[:, 0] >= 24 + 4 + 1 - options.stride_px / 2)
    assert np.all(first.features.xy[:, 1] >= 20 + 4 + 1 - options.stride_px / 2)
    assert np.all(first.features.xy[:, 0] < 136 - 4 - 1 + options.stride_px / 2)
    assert np.all(first.features.xy[:, 1] < 108 - 4 - 1 + options.stride_px / 2)
    assert np.all(first.condition_numbers >= 1.0)
    assert first.metadata["readback_bytes"] == options.max_keypoints * 32
    assert first.features.metadata["quota_tiles"] == [4, 3]
    assert first.features.metadata["selection_scope"] == "WGSL per-tile top-score candidate pool"


def test_wgsl_dic_detector_rejects_invalid_mask_shape() -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="same shape"):
        from wgpu_matching.dic_detector import DICDetectorBackend

        DICDetectorBackend(object()).compute(image, roi_mask=np.zeros((31, 32)))


def test_wgsl_dic_detector_guards_pathological_uhd_global_scan() -> None:
    image = np.zeros((2160, 3840), dtype=np.uint8)
    try:
        matcher = WGPUMatching(device_type="cpu")
    except RuntimeError as error:
        pytest.skip(f"No software WebGPU adapter is available: {error}")
    with matcher, pytest.raises(ValueError, match="impractically large serial scan"):
        matcher.detect_dic(
            image,
            options=DICDetectorOptions(
                stride_px=4,
                tensor_radius_px=4,
                max_keypoints=512,
                tile_columns=1,
                tile_rows=1,
            ),
        )


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-np.clip(value, -20.0, 20.0)))


def _numpy_dic_scores(image: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    source = image.astype(np.float32) / 255.0
    padded = np.pad(source, 1)
    first = np.empty((8, *source.shape), dtype=np.float32)
    for channel in range(8):
        value = np.full(source.shape, weights["conv1.bias"][channel], dtype=np.float32)
        for ky in range(3):
            for kx in range(3):
                value += (
                    padded[ky : ky + source.shape[0], kx : kx + source.shape[1]]
                    * weights["conv1.weight"][channel, 0, ky, kx]
                )
        first[channel] = _silu(value)

    padded = np.pad(first, ((0, 0), (1, 1), (1, 1)))
    second = np.empty_like(first)
    for channel in range(8):
        value = np.full(source.shape, weights["depthwise.bias"][channel], dtype=np.float32)
        for ky in range(3):
            for kx in range(3):
                value += (
                    padded[channel, ky : ky + source.shape[0], kx : kx + source.shape[1]]
                    * weights["depthwise.weight"][channel, 0, ky, kx]
                )
        second[channel] = _silu(value)
    logits = weights["score.bias"].reshape(()) + np.einsum(
        "chw,c->hw", second, weights["score.weight"][0, :, 0, 0]
    )
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))


def test_wgsl_learned_detector_matches_numpy_oracle(tmp_path) -> None:
    rng = np.random.default_rng(6301)
    weights = {
        "conv1.weight": rng.normal(0.0, 0.12, size=(8, 1, 3, 3)).astype(np.float32),
        "conv1.bias": rng.normal(0.0, 0.1, size=8).astype(np.float32),
        "depthwise.weight": rng.normal(0.0, 0.12, size=(8, 1, 3, 3)).astype(np.float32),
        "depthwise.bias": rng.normal(0.0, 0.1, size=8).astype(np.float32),
        "score.weight": rng.normal(0.0, 0.12, size=(1, 8, 1, 1)).astype(np.float32),
        "score.bias": rng.normal(0.0, 0.1, size=1).astype(np.float32),
    }
    weights_path = tmp_path / "dic-weights.npz"
    np.savez_compressed(weights_path, **weights)
    image = rng.integers(0, 256, size=(64, 80), dtype=np.uint8)
    options = DICDetectorOptions(
        stride_px=4,
        tensor_radius_px=4,
        max_keypoints=32,
        tile_columns=4,
        tile_rows=3,
        minimum_score=0.0,
    )
    try:
        matcher = WGPUMatching(device_type="cpu")
    except RuntimeError as error:
        pytest.skip(f"No software WebGPU adapter is available: {error}")
    with matcher:
        result = matcher.detect_dic(image, options=options, learned_weights_path=weights_path)

    scores = _numpy_dic_scores(image, weights)
    grid_x = np.rint((result.features.xy[:, 0] - 2.0) / options.stride_px).astype(int)
    grid_y = np.rint((result.features.xy[:, 1] - 2.0) / options.stride_px).astype(int)
    expected_candidates = []
    for candidate_x, candidate_y in zip(grid_x, grid_y, strict=True):
        nearby = [
            scores[
                (candidate_y + dy) * options.stride_px + 2,
                (candidate_x + dx) * options.stride_px + 2,
            ]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if 0 <= (candidate_x + dx) * options.stride_px + 2 < image.shape[1]
            and 0 <= (candidate_y + dy) * options.stride_px + 2 < image.shape[0]
        ]
        expected_candidates.append(
            min(
                nearby,
                key=lambda score: abs(
                    float(score) - float(result.features.scores[len(expected_candidates)])
                ),
            )
        )
    expected = np.asarray(expected_candidates, dtype=np.float32)
    np.testing.assert_allclose(result.features.scores, expected, rtol=2e-5, atol=2e-6)
    assert result.metadata["detector"] == "DIC-Detector-learned-v0"
    assert result.metadata["weights_sha256"]
    assert np.all(np.isfinite(result.minimum_eigenvalues))

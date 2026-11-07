from __future__ import annotations

import numpy as np
from wgpu_matching.features import FeatureSet

from python.benchmarks.experiment_learned_detectors import (
    keypoint_distribution,
    repeatability,
    select_indices,
)


def test_repeatability_uses_analytic_affine_and_pixel_centres() -> None:
    source = np.array([[20.0, 20.0], [80.0, 30.0], [40.0, 70.0]], np.float32)
    width = height = 100
    stretch_x, stretch_y = 1.1, 0.9
    target = np.column_stack(
        (
            width / 2 + stretch_x * (source[:, 0] - width / 2),
            height / 2 + stretch_y * (source[:, 1] - height / 2),
        )
    ).astype(np.float32)
    result = repeatability(source, target, width, height, stretch_x, stretch_y)
    assert result["at_1px"] == 3
    assert result["repeatability_1px"] == 1.0


def test_spatial_quotas_select_from_more_than_one_region() -> None:
    xy = np.array([[10, 10], [11, 12], [20, 20], [90, 90]], np.float32)
    features = FeatureSet(xy=xy, scores=np.array([1.0, 0.9, 0.8, 0.7]))
    selected = select_indices(features, 2, "spatial_quotas", 100, 100)
    assert len(selected) == 2
    assert keypoint_distribution(features.xy[selected], 100, 100)["coverage_cells_occupied"] == 2


def test_repeatability_returns_zero_for_empty_visible_support() -> None:
    result = repeatability(
        np.empty((0, 2), np.float32),
        np.empty((0, 2), np.float32),
        100,
        100,
        1.0,
        1.0,
    )
    assert result["denominator_min_visible"] == 0
    assert result["repeatability_5px"] == 0.0

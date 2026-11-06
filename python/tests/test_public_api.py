from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from wgpu_matching import DeepMatchingOptions, HORGOptions, Match, MatchField, load_gray
from wgpu_matching.api import _as_gray, _select_adapter


def test_option_defaults_are_documented_api() -> None:
    assert HORGOptions() == HORGOptions(128, 512, 4, 1024, None)
    assert DeepMatchingOptions() == DeepMatchingOptions(
        128, 512, 512, 2, 256, 0.0, 0.0, None, "inria"
    )
    assert DeepMatchingOptions(mode="direct").mode == "direct"


def test_array_input_is_clipped_and_contiguous() -> None:
    image = np.array([[0.0, 128.4], [255.0, 300.0]], dtype=np.float32)[:, ::-1]
    result = _as_gray(image)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, [[128, 0], [255, 255]])


def test_array_input_requires_two_dimensions() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        _as_gray(np.zeros((4, 4, 3), dtype=np.uint8))


def test_array_input_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        _as_gray(np.array([[np.nan]], dtype=np.float32))


def test_load_gray_reads_rgb_as_contiguous_uint8(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.new("RGB", (2, 1), color=(255, 0, 0)).save(path)
    result = load_gray(path)
    assert result.shape == (1, 2)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    assert result[0, 0] == 54


def test_match_field_keeps_rejected_matches_and_serializes() -> None:
    accepted = Match(x=10.0, y=20.0, dx=3.0, dy=-2.0, usable=True)
    rejected = Match(x=30.0, y=40.0, dx=0.0, dy=0.0, usable=False)
    field = MatchField(
        algorithm="example",
        width=64,
        height=48,
        patch_size=16,
        stride=16,
        scale=1,
        grid_width=2,
        grid_height=1,
        matches=[accepted, rejected],
        metadata={"backend": "test"},
    )
    assert field.accepted == [accepted]
    result = field.as_dict()
    assert result["count"] == 2
    assert result["matches"][1]["usable"] is False
    assert result["backend"] == "test"


class Adapter:
    def __init__(self, backend: str, kind: str, name: str) -> None:
        self.info = {"backend_type": backend, "adapter_type": kind, "device": name}


def test_adapter_selection_prefers_vulkan_discrete_for_high_performance() -> None:
    metal = Adapter("Metal", "DiscreteGPU", "other")
    integrated = Adapter("Vulkan", "IntegratedGPU", "integrated")
    discrete = Adapter("Vulkan", "DiscreteGPU", "discrete")
    selected = _select_adapter([metal, integrated, discrete], "gpu", "high-performance")
    assert selected is discrete


def test_adapter_selection_requires_a_compatible_adapter() -> None:
    with pytest.raises(RuntimeError, match="No compatible"):
        _select_adapter([], "gpu", "high-performance")

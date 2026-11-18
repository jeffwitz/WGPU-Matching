"""Numerical parity of the optional ONNX Runtime WebGPU matcher backend."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxruntime_ep_webgpu")

from ndf_onnx_webgpu import NDFOnnxWebGPU
from ndf_pair_model import NDFPairMatcher


def test_pair_and_encoder_parity_on_webgpu(tmp_path):
    torch.manual_seed(24)
    model = NDFPairMatcher(contextual_features=True).eval()
    weights = tmp_path / "weights.pt"
    torch.save(model.state_dict(), weights)
    backend = NDFOnnxWebGPU(model, weights, 32, 40, tmp_path / "onnx")
    reference = torch.randn(1, 1, 32, 40)
    target = torch.randn(1, 1, 32, 40)
    with torch.no_grad():
        expected = model(reference, target)
        actual = backend(reference, target)
        expected_encoder = model.encoder(reference)
    np.testing.assert_allclose(backend.encoder(reference), expected_encoder, atol=2e-5, rtol=2e-5)
    for name in ("displacement_xy", "target_xy", "hard_displacement_xy", "hard_target_xy", "confidence"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=3e-4, rtol=3e-4)
    with pytest.raises(ValueError, match="fixed image shape"):
        backend.encoder(reference[..., :-1])
    with pytest.raises(ValueError, match="topk8"):
        backend(reference, target, include_topk=True)
    assert not backend.metadata()["intermediate_buffers_shared_with_wgpu_py"]

import numpy as np
from wgpu_matching import WGPUMatching
from wgpu_matching.affine_residual import AffineResidualWGPU


def test_wgpu_affine_residual_matches_numpy_with_subpixel_sampling():
    rng = np.random.default_rng(23)
    reference = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    target = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    parameters = np.zeros((4, 6), np.float32)
    parameters[0, [0, 3]] = [0.3, 0.2]
    accepted = np.array([True, False, False, False])
    with WGPUMatching(device_type="auto") as matcher:
        result = AffineResidualWGPU(matcher.device).evaluate(
            reference, target, parameters, accepted
        )
    yy, xx = np.mgrid[0:64, 0:64]
    fx, fy = float(parameters[0, 0]), float(parameters[0, 3])
    target_float = target.astype(np.float64) / 255
    warped = (
        (1 - fx) * (1 - fy) * target_float[yy, xx]
        + fx * (1 - fy) * target_float[yy, xx + 1]
        + (1 - fx) * fy * target_float[yy + 1, xx]
        + fx * fy * target_float[yy + 1, xx + 1]
    )
    ref = reference[:64, :64].astype(np.float64) / 255
    regions = [
        (slice(None), slice(None)),
        (slice(0, 32), slice(0, 32)),
        (slice(0, 32), slice(32, 64)),
        (slice(32, 64), slice(0, 32)),
        (slice(32, 64), slice(32, 64)),
    ]
    expected = [
        np.corrcoef(ref[region].ravel(), warped[region].ravel())[0, 1] for region in regions
    ]
    np.testing.assert_allclose(result["correlations"][0], expected, atol=2e-5, rtol=0)
    np.testing.assert_array_equal(result["valid"], accepted)
    assert np.isnan(result["correlations"][1:]).all()


def test_wgpu_affine_residual_rejects_out_of_bounds_support():
    image = np.random.default_rng(3).integers(0, 256, (64, 64), dtype=np.uint8)
    parameters = np.zeros((1, 6), np.float32)
    parameters[0, 0] = 2
    with WGPUMatching(device_type="auto") as matcher:
        result = AffineResidualWGPU(matcher.device).evaluate(image, image, parameters, [True])
    assert not result["valid"][0]


def test_wgpu_affine_residual_uses_affine_gradient_and_center():
    rng = np.random.default_rng(81)
    reference = rng.integers(0, 256, (192, 192), dtype=np.uint8)
    target = rng.integers(0, 256, (192, 192), dtype=np.uint8)
    parameters = np.zeros((9, 6), np.float32)
    parameters[4] = [0.7, 0.08, -0.04, -0.3, 0.05, 0.06]
    accepted = np.zeros(9, bool)
    accepted[4] = True
    with WGPUMatching(device_type="auto") as matcher:
        result = AffineResidualWGPU(matcher.device).evaluate(
            reference, target, parameters, accepted
        )
    yy, xx = np.mgrid[64:128, 64:128].astype(np.float64)
    p = parameters[4].astype(np.float64)
    x = xx + p[0] + p[1] * (xx - 95.5) + p[2] * (yy - 95.5)
    y = yy + p[3] + p[4] * (xx - 95.5) + p[5] * (yy - 95.5)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - ix, y - iy
    image = target.astype(np.float64) / 255
    warped = (
        (1 - fx) * (1 - fy) * image[iy, ix]
        + fx * (1 - fy) * image[iy, ix + 1]
        + (1 - fx) * fy * image[iy + 1, ix]
        + fx * fy * image[iy + 1, ix + 1]
    )
    expected = np.corrcoef(reference[64:128, 64:128].ravel(), warped.ravel())[0, 1]
    assert result["valid"][4]
    np.testing.assert_allclose(result["correlations"][4, 0], expected, atol=2e-5, rtol=0)


def test_wgpu_affine_residual_rejects_constant_support():
    image = np.full((64, 64), 128, dtype=np.uint8)
    with WGPUMatching(device_type="auto") as matcher:
        result = AffineResidualWGPU(matcher.device).evaluate(
            image, image, np.zeros((1, 6), np.float32), [True]
        )
    assert not result["valid"][0]

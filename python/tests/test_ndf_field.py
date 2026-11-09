"""Analytical checks for the optional neural deformation-field prototype."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ndf_field import BSplineDisplacement, sample_image
from ndf_pair_model import (
    NDFPairMatcher,
    pair_matching_loss,
    refine_local_features,
    refine_local_ncc,
)
from qualify_ndf_pair_icgn import (
    compose_affine_residual_flow,
    compose_field_residual_flow,
    fit_global_affine_ransac,
    fit_local_bspline_prefit,
    rectify_target_with_affine,
    robust_affine_projection,
)
from train_ndf_pair_matcher import load_or_make_dataset, local_affine_consensus


def test_bilinear_sampling_coordinate_gradient_matches_finite_difference():
    image = torch.arange(20 * 24, dtype=torch.float64).reshape(20, 24).square()
    point = torch.tensor([[8.31, 11.27]], dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(sample_image(image, point).sum(), point)[0]
    epsilon = 1e-5
    dx = torch.tensor([[epsilon, 0.0]], dtype=point.dtype)
    dy = torch.tensor([[0.0, epsilon]], dtype=point.dtype)
    finite = torch.stack(
        (
            (sample_image(image, point.detach() + dx) - sample_image(image, point.detach() - dx))
            / (2 * epsilon),
            (sample_image(image, point.detach() + dy) - sample_image(image, point.detach() - dy))
            / (2 * epsilon),
        ),
        dim=1,
    )
    torch.testing.assert_close(derivative, finite, rtol=1e-8, atol=1e-7)


def test_cubic_spline_analytic_jacobian_matches_autodiff():
    coefficients = torch.randn((9, 11, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    points = torch.tensor([[41.37, 32.29], [109.41, 77.62]], dtype=torch.float64, requires_grad=True)
    field = BSplineDisplacement((coefficients,), (160, 120))
    displacement, jacobian = field.evaluate(points)
    derivatives = torch.stack(
        [
            torch.autograd.grad(displacement[:, component].sum(), points, retain_graph=True)[0]
            for component in range(2)
        ],
        dim=1,
    )
    torch.testing.assert_close(jacobian - torch.eye(2, dtype=points.dtype), derivatives, atol=1e-12, rtol=1e-12)


def test_image_derived_affine_ransac_rejects_outliers_and_recovers_map():
    y, x = torch.meshgrid(torch.linspace(0, 90, 10), torch.linspace(0, 120, 12), indexing="ij")
    source = torch.stack((x, y), dim=-1).reshape(-1, 2).numpy()
    expected = np.asarray([[17.0, -9.0], [1.16, 0.11], [-0.07, 0.91]])
    target = np.column_stack((np.ones(len(source)), source)) @ expected
    target[::7] += np.asarray([31.0, -27.0])
    coefficients, inliers, median = fit_global_affine_ransac(
        source, target, threshold=0.5, min_inliers=50, iterations=256
    )
    assert coefficients is not None
    assert inliers.sum() >= len(source) - len(source[::7])
    np.testing.assert_allclose(coefficients, expected, atol=1e-10, rtol=0)
    assert median is not None and median < 1e-10


def test_affine_rectification_identity_preserves_image():
    image = torch.rand((1, 1, 19, 27), generator=torch.Generator().manual_seed(22))
    identity = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    rectified = rectify_target_with_affine(image, identity)
    torch.testing.assert_close(rectified, image, atol=1e-6, rtol=0)


def test_affine_residual_composition_matches_direct_map():
    coefficients = np.asarray([[7.0, -3.0], [1.2, 0.1], [-0.05, 0.9]])
    residual = torch.zeros((1, 3, 4, 2), dtype=torch.float64)
    residual[..., 0] = 2.0
    residual[..., 1] = -1.5
    result = compose_affine_residual_flow(residual, coefficients, (40, 30))[0]
    y, x = torch.meshgrid(torch.arange(3, dtype=torch.float64), torch.arange(4, dtype=torch.float64), indexing="ij")
    source = torch.stack(((x + 0.5) * 10 - 0.5, (y + 0.5) * 10 - 0.5), dim=-1)
    q = source + residual[0]
    expected = torch.cat((torch.ones_like(q[..., :1]), q), dim=-1) @ torch.as_tensor(coefficients)
    torch.testing.assert_close(result, expected - source, atol=1e-12, rtol=0)


def test_local_bspline_prefit_and_residual_composition():
    class Prediction:
        pass

    width, height = 128, 96
    rows, columns = 24, 32
    yy, xx = torch.meshgrid(torch.arange(rows), torch.arange(columns), indexing="ij")
    x = (xx + 0.5) * width / columns - 0.5
    y = (yy + 0.5) * height / rows - 0.5
    prediction = Prediction()
    prediction.hard_displacement_xy = torch.stack(
        (3.0 + 0.1 * x + 0.02 * y, 2.0 - 0.03 * x + 0.05 * y), dim=-1
    )[None]
    field, diagnostics = fit_local_bspline_prefit(
        prediction,
        (width, height),
        radius=32.0,
        threshold=0.5,
        min_support=96,
        bending=0.1,
    )
    assert field is not None
    assert diagnostics["support"] >= 96
    residual = torch.zeros((1, 12, 16, 2), dtype=torch.float32)
    composed = compose_field_residual_flow(residual, field, (width, height))
    ys = torch.tensor([2, 6, 8])
    xs = torch.tensor([2, 8, 12])
    sample = torch.stack(
        ((xs + 0.5) * width / 16 - 0.5, (ys + 0.5) * height / 12 - 0.5), dim=-1
    )
    expected = torch.stack(
        (3.0 + 0.1 * sample[:, 0] + 0.02 * sample[:, 1],
         2.0 - 0.03 * sample[:, 0] + 0.05 * sample[:, 1]),
        dim=-1,
    )
    torch.testing.assert_close(composed[0, ys, xs], expected, atol=0.15, rtol=0)


def test_spline_field_can_represent_a_translation_and_its_jacobian():
    points = torch.rand((128, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    points *= torch.tensor([255.0, 191.0], dtype=points.dtype)
    coefficients = torch.zeros((10, 14, 2), dtype=points.dtype)
    coefficients[..., 0] = 75.0
    coefficients[..., 1] = -23.0
    displacement, jacobian = BSplineDisplacement((coefficients,), (256, 192)).evaluate(points)
    torch.testing.assert_close(displacement, coefficients.new_tensor([75.0, -23.0]).expand_as(displacement), atol=1e-10, rtol=0)
    torch.testing.assert_close(jacobian, torch.eye(2, dtype=points.dtype).expand_as(jacobian), atol=1e-10, rtol=0)


def test_pair_matcher_has_global_coarse_output_and_differentiable_loss():
    model = NDFPairMatcher(feature_dimensions=8)
    reference = torch.rand((2, 1, 64, 96))
    target = torch.rand_like(reference)
    flow = torch.zeros((2, 2, 64, 96))
    flow[:, 0] = 8.0
    visible = torch.ones((2, 64, 96), dtype=torch.bool)
    prediction = model(reference, target)
    assert prediction.displacement_xy.shape == (2, 16, 24, 2)
    assert prediction.hard_displacement_xy.shape == (2, 16, 24, 2)
    assert prediction.topk_displacement_xy is None
    assert prediction.confidence.shape == (2, 16, 24)
    topk_prediction = model(reference, target, include_topk=True)
    assert topk_prediction.topk_displacement_xy.shape == (2, 16, 24, 2)
    assert torch.isfinite(topk_prediction.topk_displacement_xy).all()
    loss, metrics = pair_matching_loss(
        model,
        reference,
        target,
        flow,
        visible,
        physical_weight=0.01,
        physical_pyramid_levels=3,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["top1_grid_accuracy"])
    assert torch.isfinite(metrics["physical_loss"])
    assert model.encoder.layers[0].weight.grad is not None
    local = refine_local_features(
        model, reference[:1], target[:1], prediction.target_xy[:1], search_radius_cells=2
    )
    assert local.displacement_xy.shape == (1, 16, 24, 2)
    assert torch.isfinite(local.displacement_xy).all()


def test_contextual_pair_matcher_preserves_feature_grid_shape():
    model = NDFPairMatcher(feature_dimensions=8, contextual_features=True)
    image = torch.rand((1, 1, 64, 96))
    prediction = model(image, image)
    assert prediction.displacement_xy.shape == (1, 16, 24, 2)
    assert model.encoder.context is not None


def test_synthetic_training_cache_reuses_pairs_and_checks_manifest(tmp_path):
    cache = tmp_path / "training.pt"
    first = load_or_make_dataset(cache, 1, 0, 32, 32, 1, 3.5)
    second = load_or_make_dataset(cache, 1, 0, 32, 32, 1, 3.5)
    torch.testing.assert_close(first[0][0], second[0][0])
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_or_make_dataset(cache, 1, 0, 32, 32, 2, 3.5)


def test_local_ncc_refinement_recovers_subpixel_translation():
    torch.manual_seed(32)
    reference = torch.rand((1, 1, 64, 80))
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(80), indexing="ij")
    dx, dy = 2.25, -1.4
    grid = torch.stack(
        (2 * (xx - dx + 0.5) / 80 - 1, 2 * (yy - dy + 0.5) / 64 - 1), dim=-1
    ).float()[None]
    target = torch.nn.functional.grid_sample(
        reference, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    initial = torch.zeros((1, 16, 20, 2))
    result = refine_local_ncc(
        reference, target, initial, search_radius=4, patch_radius=4
    )
    interior = result.valid[0, 2:-2, 2:-2]
    error = torch.linalg.vector_norm(
        result.displacement_xy[0, 2:-2, 2:-2]
        - torch.tensor([dx, dy]), dim=-1
    )
    assert interior.float().mean() > 0.95
    assert torch.median(error[interior]) < 0.3


def test_affine_compensated_local_ncc_uses_predicted_patch_jacobian():
    height = width = 96
    torch.manual_seed(62)
    reference = torch.rand((1, 1, height, width))
    reference = torch.nn.functional.avg_pool2d(reference, 3, stride=1, padding=1)
    matrix = torch.tensor([[1.42, 0.24], [-0.18, 0.76]])
    translation = torch.tensor([2.0, -1.0])
    inverse = torch.linalg.inv(matrix)
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    target_xy = torch.stack((xx, yy), dim=-1).float()
    source_xy = (target_xy - translation) @ inverse.T
    grid = torch.stack(
        (2 * (source_xy[..., 0] + 0.5) / width - 1,
         2 * (source_xy[..., 1] + 0.5) / height - 1), dim=-1
    )[None]
    target = torch.nn.functional.grid_sample(
        reference, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    field_height = field_width = 16
    source_x = (torch.arange(field_width) + 0.5) * (width / field_width) - 0.5
    source_y = (torch.arange(field_height) + 0.5) * (height / field_height) - 0.5
    gy, gx = torch.meshgrid(source_y, source_x, indexing="ij")
    centres = torch.stack((gx, gy), dim=-1)
    flow = centres @ (matrix - torch.eye(2)).T + translation
    jacobian = matrix.expand(field_height, field_width, 2, 2)
    corrected = refine_local_ncc(
        reference, target, flow[None], search_radius=4, patch_radius=7,
        initial_jacobian=jacobian, geometry="affine",
    )
    unwarped = refine_local_ncc(
        reference, target, flow[None], search_radius=4, patch_radius=7
    )
    hybrid = refine_local_ncc(
        reference, target, flow[None], search_radius=4, patch_radius=7,
        initial_jacobian=jacobian, geometry="hybrid",
    )
    interior = corrected.valid[0, 2:-2, 2:-2] & unwarped.valid[0, 2:-2, 2:-2]
    corrected_error = torch.linalg.vector_norm(
        corrected.displacement_xy[0, 2:-2, 2:-2] - flow[2:-2, 2:-2], dim=-1
    )
    unwarped_error = torch.linalg.vector_norm(
        unwarped.displacement_xy[0, 2:-2, 2:-2] - flow[2:-2, 2:-2], dim=-1
    )
    assert interior.float().mean() > 0.5
    assert torch.median(corrected_error[interior]) < 0.5
    assert torch.median(unwarped_error[interior]) > torch.median(corrected_error[interior]) + 0.5
    hybrid_error = torch.linalg.vector_norm(
        hybrid.displacement_xy[0, 2:-2, 2:-2] - flow[2:-2, 2:-2], dim=-1
    )
    assert torch.median(hybrid_error[interior]) <= torch.median(unwarped_error[interior])
    np = pytest.importorskip("numpy")
    from wgpu_matching import WGPUMatching

    with WGPUMatching(device_type="cpu") as matcher:
        gpu = matcher.refine_ndf_local_ncc(
            reference[0, 0].numpy(), target[0, 0].numpy(), flow.numpy(),
            search_radius=4, patch_radius=7, initial_jacobian=jacobian.numpy(),
            geometry="affine",
        )
    assert np.array_equal(gpu.valid, corrected.valid[0].numpy())
    valid = gpu.valid
    endpoint_error = np.linalg.norm(
        gpu.displacement_xy[valid] - corrected.displacement_xy[0].numpy()[valid], axis=-1
    )
    assert np.quantile(endpoint_error, 0.95) < 0.1
    with WGPUMatching(device_type="cpu") as matcher:
        gpu_hybrid = matcher.refine_ndf_local_ncc(
            reference[0, 0].numpy(), target[0, 0].numpy(), flow.numpy(),
            search_radius=4, patch_radius=7, initial_jacobian=jacobian.numpy(),
            geometry="hybrid",
        )
    assert np.array_equal(gpu_hybrid.valid, hybrid.valid[0].numpy())
    hybrid_endpoint_error = np.linalg.norm(
        gpu_hybrid.displacement_xy[hybrid.valid[0].numpy()]
        - hybrid.displacement_xy[0].numpy()[hybrid.valid[0].numpy()], axis=-1
    )
    assert np.quantile(hybrid_endpoint_error, 0.95) < 0.25


def test_webgpu_local_ncc_matches_torch_reference():
    np = pytest.importorskip("numpy")
    from wgpu_matching import WGPUMatching

    torch.manual_seed(15)
    reference = torch.rand((1, 1, 64, 80))
    target = torch.zeros_like(reference)
    target[:, :, 2:, 3:] = reference[:, :, :-2, :-3]
    initial = torch.zeros((1, 16, 20, 2))
    cpu = refine_local_ncc(reference, target, initial, search_radius=4, patch_radius=4)
    with WGPUMatching(device_type="cpu") as matcher:
        gpu = matcher.refine_ndf_local_ncc(
            reference[0, 0].numpy(), target[0, 0].numpy(), initial[0].numpy(),
            search_radius=4, patch_radius=4,
        )
        serial = matcher.refine_ndf_local_ncc(
            reference[0, 0].numpy(), target[0, 0].numpy(), initial[0].numpy(),
            search_radius=4, patch_radius=4, implementation="serial",
        )
    assert np.array_equal(gpu.valid, cpu.valid[0].numpy())
    valid = gpu.valid
    flow_error = np.linalg.norm(
        gpu.displacement_xy[valid] - cpu.displacement_xy[0].numpy()[valid], axis=-1
    )
    score_error = np.abs(gpu.score[valid] - cpu.score[0].numpy()[valid])
    assert np.quantile(flow_error, 0.95) < 0.1
    assert np.quantile(score_error, 0.95) < 0.1
    serial_error = np.linalg.norm(gpu.displacement_xy[valid] - serial.displacement_xy[valid], axis=-1)
    assert np.quantile(serial_error, 0.95) < 0.1


def test_local_affine_consensus_recovers_a_known_image_only_field():
    y, x = torch.meshgrid(torch.arange(0, 48, 8), torch.arange(0, 64, 8), indexing="ij")
    source = torch.stack((x, y), dim=-1).reshape(-1, 2).numpy().astype(float)
    matrix = torch.tensor([[1.1, 0.08], [-0.03, 0.92]], dtype=torch.float64).numpy()
    translation = torch.tensor([13.0, -7.0], dtype=torch.float64).numpy()
    target = source @ matrix.T + translation
    parameters, usable = local_affine_consensus(source, target, 96, 64, radius=32, threshold=0.1)
    assert usable.mean() > 0.8
    torch.testing.assert_close(
        torch.from_numpy(parameters[usable, 1:3].mean(axis=0)),
        torch.tensor([0.1, 0.08], dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )


def test_robust_affine_projection_rejects_correspondence_outliers():
    axis = torch.linspace(0.0, 100.0, 12, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    source = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    matrix = torch.tensor([[1.08, 0.13], [-0.07, 0.94]], dtype=torch.float64)
    translation = torch.tensor([17.0, -9.0], dtype=torch.float64)
    mapped = source @ matrix.T + translation
    displacement = mapped - source
    displacement[::4] += torch.tensor([180.0, -130.0], dtype=torch.float64)

    predicted, gradient = robust_affine_projection(source, displacement, source)

    torch.testing.assert_close(gradient[0], matrix, atol=2.5e-3, rtol=0)
    torch.testing.assert_close(predicted[0], translation, atol=3e-3, rtol=0)

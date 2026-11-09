"""Validate NDF-DIC interpolation gradients and non-linear field capacity.

Run with the project development environment and ``PYTHONPATH=python/src``.
This is a CPU PyTorch research diagnostic, not an inference-time dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from ndf_field import (
    BSplineDisplacement,
    DenseBilinearDisplacement,
    advected_photometric_losses,
    dense_bilinear_design_matrix,
    fit_bspline_field,
    recommended_control_shape,
    sample_image,
    uniform_points,
)
from wgpu_matching.dic_synthetic import generate_speckle_pair

DTYPE = torch.float64


def analytic_field(name: str, points: torch.Tensor, image_size: tuple[int, int]):
    """Return analytic displacement and F for reproducible representability cases."""
    width, height = image_size
    center = points.new_tensor([0.5 * (width - 1), 0.5 * (height - 1)])
    if name == "translation_300":
        matrix = torch.eye(2, dtype=DTYPE)
        translation = points.new_tensor([230.0, -125.0])
    elif name == "rotation_120":
        angle = math.radians(120.0)
        matrix = points.new_tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        translation = points.new_tensor([125.0, -60.0])
    elif name == "affine_large":
        angle = math.radians(23.0)
        rotation = points.new_tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        matrix = rotation @ points.new_tensor([[1.55, 0.28], [-0.08, 0.73]])
        translation = points.new_tensor([115.0, -55.0])
    elif name in {"heterogeneous_256", "heterogeneous_128", "localized_band_24", "discontinuous"}:
        matrix = torch.eye(2, dtype=DTYPE)
        translation = points.new_tensor([210.0, -95.0])
    else:
        raise ValueError(f"Unknown analytic field: {name}")

    relative = points - center
    transformed = relative @ matrix.T + center + translation
    displacement = transformed - points
    jacobian = matrix.expand(len(points), 2, 2).clone()
    x, y = points[:, 0], points[:, 1]
    if name.startswith("heterogeneous_"):
        wavelength = float(name.rsplit("_", 1)[1])
        kx = 2 * math.pi / wavelength
        ky = 2 * math.pi / wavelength
        ax, ay = (25.0, 18.0) if wavelength >= 192.0 else (12.0, 9.0)
        phase_x, phase_y = kx * x, ky * y
        displacement[:, 0] += ax * torch.sin(phase_y) * torch.cos(phase_x)
        displacement[:, 1] += ay * torch.sin(phase_x) * torch.cos(phase_y)
        jacobian[:, 0, 0] += -ax * kx * torch.sin(phase_y) * torch.sin(phase_x)
        jacobian[:, 0, 1] += ax * ky * torch.cos(phase_y) * torch.cos(phase_x)
        jacobian[:, 1, 0] += ay * kx * torch.cos(phase_x) * torch.cos(phase_y)
        jacobian[:, 1, 1] += -ay * ky * torch.sin(phase_x) * torch.sin(phase_y)
    elif name == "localized_band_24":
        y0, width_band, amplitude = 0.53 * height, 24.0, 34.0
        z = (y - y0) / width_band
        tanh = torch.tanh(z)
        sech2 = 1 - tanh.square()
        displacement[:, 0] += amplitude * tanh
        jacobian[:, 0, 1] += amplitude / width_band * sech2
    elif name == "discontinuous":
        displacement[:, 0] += 36.0 * (x >= 0.5 * width)
    return displacement, jacobian


def affine_fit(points: torch.Tensor, displacement: torch.Tensor):
    design = torch.stack(
        (torch.ones_like(points[:, 0]), points[:, 0], points[:, 1]), dim=1
    )
    coefficients = torch.linalg.lstsq(design, displacement).solution
    return coefficients


def affine_jacobian(coefficients: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    jacobian = torch.eye(2, dtype=points.dtype).expand(len(points), 2, 2).clone()
    jacobian[:, :, 0] += coefficients[1].reshape(1, 2)
    jacobian[:, :, 1] += coefficients[2].reshape(1, 2)
    return jacobian


def summarize_field(
    predicted_u: torch.Tensor,
    predicted_f: torch.Tensor,
    true_u: torch.Tensor,
    true_f: torch.Tensor,
) -> dict[str, float]:
    translation_error = torch.linalg.vector_norm(predicted_u - true_u, dim=1)
    gradient_error = torch.linalg.matrix_norm(predicted_f - true_f, ord="fro", dim=(1, 2))
    predicted_green = 0.5 * (predicted_f.transpose(1, 2) @ predicted_f - torch.eye(2, dtype=DTYPE))
    true_green = 0.5 * (true_f.transpose(1, 2) @ true_f - torch.eye(2, dtype=DTYPE))
    green_error = torch.linalg.matrix_norm(predicted_green - true_green, ord="fro", dim=(1, 2))
    determinant = torch.linalg.det(predicted_f)
    return {
        "displacement_rmse_px": float(torch.sqrt(torch.mean(translation_error.square()))),
        "displacement_median_px": float(translation_error.median()),
        "displacement_p95_px": float(torch.quantile(translation_error, 0.95)),
        "displacement_max_px": float(translation_error.max()),
        "jacobian_frobenius_rmse": float(torch.sqrt(torch.mean(gradient_error.square()))),
        "jacobian_frobenius_p95": float(torch.quantile(gradient_error, 0.95)),
        "green_strain_frobenius_rmse": float(torch.sqrt(torch.mean(green_error.square()))),
        "determinant_min": float(determinant.min()),
        "negative_determinant_fraction": float((determinant <= 0).double().mean()),
    }


def evaluate_field(name: str, width: int, height: int, fit_spacing: int, eval_spacing: int):
    image_size = (width, height)
    fit_points = uniform_points(image_size, fit_spacing)
    eval_points = uniform_points(image_size, eval_spacing)
    fit_displacement, _ = analytic_field(name, fit_points, image_size)
    true_displacement, true_jacobian = analytic_field(name, eval_points, image_size)
    result: dict[str, object] = {
        "case": name,
        "fit_points": len(fit_points),
        "evaluation_points": len(eval_points),
        "fit_sample_spacing_px": fit_spacing,
        "image_size_px": [width, height],
        "representations": {},
    }

    started = time.perf_counter()
    affine_coefficients = affine_fit(fit_points, fit_displacement)
    affine_at_eval = torch.stack(
        (torch.ones_like(eval_points[:, 0]), eval_points[:, 0], eval_points[:, 1]), dim=1
    ) @ affine_coefficients
    affine_f = affine_jacobian(affine_coefficients, eval_points)
    result["representations"]["global_affine"] = {
        **summarize_field(affine_at_eval, affine_f, true_displacement, true_jacobian),
        "fit_s": time.perf_counter() - started,
        "coefficients": 6,
    }

    spacings = (128.0, 64.0, 32.0)
    spline_shapes = tuple(recommended_control_shape(image_size, spacing) for spacing in spacings)
    started = time.perf_counter()
    spline = fit_bspline_field(fit_points, fit_displacement, image_size, spline_shapes)
    spline_u, spline_f = spline.evaluate(eval_points)
    spline_parameters = sum(h * w * 2 for h, w in spline_shapes)
    result["representations"]["cubic_bspline_128_64_32"] = {
        **summarize_field(spline_u, spline_f, true_displacement, true_jacobian),
        "fit_s": time.perf_counter() - started,
        "coefficients": spline_parameters,
        "control_shapes_yx": [list(shape) for shape in spline_shapes],
    }

    dense_shapes = tuple(
        (max(2, math.ceil(height / spacing) + 1), max(2, math.ceil(width / spacing) + 1))
        for spacing in spacings
    )
    started = time.perf_counter()
    dense_design = torch.cat(
        [
            # Each grid is independently predicted; sum-of-levels is a
            # multi-resolution direct displacement parameterization.
            dense_bilinear_design_matrix(fit_points, image_size, shape)
            for shape in dense_shapes
        ],
        dim=1,
    )
    normal = dense_design.T @ dense_design + 1e-7 * torch.eye(
        dense_design.shape[1], dtype=DTYPE
    )
    dense_coefficients = torch.linalg.solve(normal, dense_design.T @ fit_displacement)
    dense_u_parts = []
    dense_f_parts = []
    offset = 0
    for shape in dense_shapes:
        count = shape[0] * shape[1]
        model = DenseBilinearDisplacement(
            dense_coefficients[offset : offset + count].reshape(*shape, 2), image_size
        )
        u, f = model.evaluate(eval_points)
        dense_u_parts.append(u)
        # Jacobian of additive multi-resolution field: sum(F_l - I) + I.
        dense_f_parts.append(f)
        offset += count
    dense_u = torch.stack(dense_u_parts).sum(dim=0)
    dense_f = torch.stack(dense_f_parts).sum(dim=0) - (len(dense_shapes) - 1) * torch.eye(
        2, dtype=DTYPE
    )
    result["representations"]["direct_bilinear_128_64_32"] = {
        **summarize_field(dense_u, dense_f, true_displacement, true_jacobian),
        "fit_s": time.perf_counter() - started,
        "coefficients": sum(h * w * 2 for h, w in dense_shapes),
        "control_shapes_yx": [list(shape) for shape in dense_shapes],
    }
    return result


def validate_interpolation_gradients() -> dict[str, float]:
    torch.manual_seed(107)
    image = torch.randn((64, 80), dtype=DTYPE).cumsum(0).cumsum(1)
    image = image / image.std()
    points = torch.rand((48, 2), dtype=DTYPE)
    points[:, 0] = torch.floor(points[:, 0] * 69) + 4.37
    points[:, 1] = torch.floor(points[:, 1] * 53) + 4.42
    points.requires_grad_(True)
    sampled = sample_image(image, points)
    derivative = torch.autograd.grad(sampled.sum(), points)[0]
    epsilon = 1e-4
    finite_x = (sample_image(image, points.detach() + points.new_tensor([epsilon, 0])) - sample_image(image, points.detach() - points.new_tensor([epsilon, 0]))) / (2 * epsilon)
    finite_y = (sample_image(image, points.detach() + points.new_tensor([0, epsilon])) - sample_image(image, points.detach() - points.new_tensor([0, epsilon]))) / (2 * epsilon)
    finite = torch.stack((finite_x, finite_y), dim=1)
    return {
        "bilinear_autodiff_vs_finite_difference_max_abs": float((derivative - finite).abs().max()),
        "bilinear_autodiff_vs_finite_difference_rmse": float(torch.sqrt(torch.mean((derivative - finite).square()))),
    }


def validate_spline_jacobian() -> dict[str, float]:
    torch.manual_seed(207)
    coefficients = (torch.randn((11, 15, 2), dtype=DTYPE) * 0.3).requires_grad_(True)
    points = torch.rand((128, 2), dtype=DTYPE)
    points[:, 0] *= 510
    points[:, 1] *= 382
    points.requires_grad_(True)
    field = BSplineDisplacement((coefficients,), (512, 384))
    displacement, jacobian = field.evaluate(points)
    autodiff = []
    for component in range(2):
        autodiff.append(torch.autograd.grad(displacement[:, component].sum(), points, retain_graph=True)[0])
    grad = torch.stack(autodiff, dim=1)
    analytic_gradient = jacobian - torch.eye(2, dtype=DTYPE)
    eps = 1e-3
    basis_x = points.detach() + points.new_tensor([eps, 0])
    basis_xm = points.detach() - points.new_tensor([eps, 0])
    basis_y = points.detach() + points.new_tensor([0, eps])
    basis_ym = points.detach() - points.new_tensor([0, eps])
    finite = torch.stack(
        (
            (field.evaluate(basis_x)[0] - field.evaluate(basis_xm)[0]) / (2 * eps),
            (field.evaluate(basis_y)[0] - field.evaluate(basis_ym)[0]) / (2 * eps),
        ),
        dim=2,
    )
    result = {
        "analytic_vs_autodiff_max_abs": float((analytic_gradient - grad).detach().abs().max()),
        "analytic_vs_autodiff_rmse": float(torch.sqrt(torch.mean((analytic_gradient - grad).detach().square()))),
        "analytic_vs_finite_difference_max_abs": float((analytic_gradient - finite).detach().abs().max()),
        "analytic_vs_finite_difference_rmse": float(torch.sqrt(torch.mean((analytic_gradient - finite).detach().square()))),
    }
    # Verify that the image-pair physical objective differentiates through the
    # field coefficients and the bilinear target sampler.
    height, width = 96, 128
    pair = generate_speckle_pair(331, 441, width, height, motion_mode="affine")
    reference = torch.from_numpy(pair.reference.copy()).to(DTYPE)
    target = torch.from_numpy(pair.target.copy()).to(DTYPE)
    sample_points = uniform_points((width, height), 4)[100:-100]
    small_coefficients = torch.zeros((8, 11, 2), dtype=DTYPE, requires_grad=True)
    small_field = BSplineDisplacement((small_coefficients,), (width, height))
    small_u, small_f = small_field.evaluate(sample_points)
    losses = advected_photometric_losses(reference, target, sample_points, small_u, small_f)
    total = losses["intensity"] + losses["advected_gradient"]
    coefficient_gradient = torch.autograd.grad(total, small_coefficients)[0]
    result["physical_loss"] = {
        "intensity": float(losses["intensity"].detach()),
        "advected_gradient": float(losses["advected_gradient"].detach()),
        "coefficient_gradient_norm": float(torch.linalg.vector_norm(coefficient_gradient)),
        "valid_fraction": float(losses["valid_fraction"]),
    }
    oracle_pair = generate_speckle_pair(551, 661, width, height, motion_mode="affine")
    oracle_reference = torch.from_numpy(oracle_pair.reference.copy()).to(DTYPE)
    oracle_target = torch.from_numpy(oracle_pair.target.copy()).to(DTYPE)
    oracle_points = uniform_points((width, height), 2)
    ix = oracle_points[:, 0].long()
    iy = oracle_points[:, 1].long()
    oracle_u = torch.from_numpy(oracle_pair.flow_xy[iy.numpy(), ix.numpy()].copy()).to(DTYPE)
    oracle_f = torch.as_tensor(oracle_pair.metadata["affine_matrix"], dtype=DTYPE)
    oracle_f = oracle_f.expand(len(oracle_points), 2, 2)
    oracle_mask = torch.from_numpy(oracle_pair.visible.copy())
    oracle_loss = advected_photometric_losses(
        oracle_reference,
        oracle_target,
        oracle_points,
        oracle_u,
        oracle_f,
        valid_mask=oracle_mask,
    )
    translation_perturbed = advected_photometric_losses(
        oracle_reference,
        oracle_target,
        oracle_points,
        oracle_u + oracle_u.new_tensor([4.0, -3.0]),
        oracle_f,
        valid_mask=oracle_mask,
    )
    jacobian_perturbed = advected_photometric_losses(
        oracle_reference,
        oracle_target,
        oracle_points,
        oracle_u,
        oracle_f @ oracle_f.new_tensor([[1.2, 0.0], [0.0, 0.85]]),
        valid_mask=oracle_mask,
    )
    result["known_affine_advection_control"] = {
        "oracle_intensity": float(oracle_loss["intensity"].detach()),
        "oracle_advected_gradient": float(oracle_loss["advected_gradient"].detach()),
        "translation_error_intensity": float(translation_perturbed["intensity"].detach()),
        "translation_error_advected_gradient": float(translation_perturbed["advected_gradient"].detach()),
        "jacobian_error_intensity": float(jacobian_perturbed["intensity"].detach()),
        "jacobian_error_advected_gradient": float(jacobian_perturbed["advected_gradient"].detach()),
        "valid_fraction": float(oracle_loss["valid_fraction"].detach()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fit-spacing", type=int, default=8)
    parser.add_argument("--evaluation-spacing", type=int, default=3)
    args = parser.parse_args()
    torch.set_default_dtype(DTYPE)
    results = {
        "protocol": {
            "purpose": "P0 analytic derivative/advection checks and P1 non-linear displacement representation capacity",
            "pytorch_version": torch.__version__,
            "device": "CPU reference; inference backend is not implemented in this phase",
            "coordinate_convention": "pixel-centre coordinates (x,y); array index (0,0) denotes upper-left pixel centre",
            "field_resolution": [args.width, args.height],
            "fit_spacing_px": args.fit_spacing,
            "evaluation_spacing_px": args.evaluation_spacing,
            "fields": ["translation_300", "rotation_120", "affine_large", "heterogeneous_256", "heterogeneous_128", "localized_band_24", "discontinuous"],
        },
        "interpolation_gradient_checks": validate_interpolation_gradients(),
        "spline_jacobian_and_physics_gradient_checks": validate_spline_jacobian(),
        "capacity": [
            evaluate_field(name, args.width, args.height, args.fit_spacing, args.evaluation_spacing)
            for name in (
                "translation_300",
                "rotation_120",
                "affine_large",
                "heterogeneous_256",
                "heterogeneous_128",
                "localized_band_24",
                "discontinuous",
            )
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

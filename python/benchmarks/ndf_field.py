"""Research prototype for differentiable non-linear DIC deformation fields.

This optional benchmark module depends on PyTorch; it is intentionally outside
the install-time ``wgpu_matching`` package. Coordinates follow the repository
convention: ``(x, y)`` in pixel-centre coordinates, where image-array index
``(0, 0)`` is the centre of the upper-left pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as torch_functional


def _cubic_weights(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cardinal cubic B-spline basis and derivatives with respect to ``t``."""
    t2 = t * t
    t3 = t2 * t
    weights = torch.stack(
        (
            (1 - t) ** 3 / 6,
            (3 * t3 - 6 * t2 + 4) / 6,
            (-3 * t3 + 3 * t2 + 3 * t + 1) / 6,
            t3 / 6,
        ),
        dim=-1,
    )
    derivatives = torch.stack(
        (
            -0.5 * (1 - t) ** 2,
            1.5 * t2 - 2 * t,
            -1.5 * t2 + t + 0.5,
            0.5 * t2,
        ),
        dim=-1,
    )
    return weights, derivatives


def _spline_geometry(
    points_xy: torch.Tensor, image_size: tuple[int, int], shape: tuple[int, int]
) -> tuple[torch.Tensor, ...]:
    width, height = image_size
    control_height, control_width = shape
    if min(width, height, control_width, control_height) < 4:
        raise ValueError("Image and cubic B-spline control-grid dimensions must be at least four")
    spacing_x = (width - 1) / (control_width - 3)
    spacing_y = (height - 1) / (control_height - 3)
    # One virtual control interval before the image gives a centred cardinal
    # spline basis at the boundary; indices are clamped for edge extension.
    grid_x = (points_xy[:, 0] + spacing_x) / spacing_x
    grid_y = (points_xy[:, 1] + spacing_y) / spacing_y
    floor_x = torch.floor(grid_x)
    floor_y = torch.floor(grid_y)
    frac_x = grid_x - floor_x
    frac_y = grid_y - floor_y
    wx, dwx = _cubic_weights(frac_x)
    wy, dwy = _cubic_weights(frac_y)
    ix = floor_x.long()[:, None] + torch.tensor(
        [-1, 0, 1, 2], device=points_xy.device
    )[None, :]
    iy = floor_y.long()[:, None] + torch.tensor(
        [-1, 0, 1, 2], device=points_xy.device
    )[None, :]
    ix = ix.clamp(0, control_width - 1)
    iy = iy.clamp(0, control_height - 1)
    return wx, wy, dwx / spacing_x, dwy / spacing_y, ix, iy


def bspline_design_matrix(
    points_xy: torch.Tensor,
    image_size: tuple[int, int],
    shape: tuple[int, int],
) -> torch.Tensor:
    """Dense design matrix for fitting one scalar displacement component."""
    control_height, control_width = shape
    wx, wy, _, _, ix, iy = _spline_geometry(points_xy, image_size, shape)
    point_count = len(points_xy)
    matrix = torch.zeros(
        (point_count, control_height * control_width),
        dtype=points_xy.dtype,
        device=points_xy.device,
    )
    weights = wy[:, :, None] * wx[:, None, :]
    indices = iy[:, :, None] * control_width + ix[:, None, :]
    matrix.scatter_add_(1, indices.reshape(point_count, 16), weights.reshape(point_count, 16))
    return matrix


def _bending_matrix(shape: tuple[int, int], *, device: torch.device, dtype: torch.dtype):
    """Second-difference matrix that leaves affine control nets unpenalized."""
    height, width = shape
    row_count = height * max(width - 2, 0) + max(height - 2, 0) * width
    matrix = torch.zeros((row_count, height * width), device=device, dtype=dtype)
    row = 0
    for y in range(height):
        for x in range(1, width - 1):
            matrix[row, y * width + x - 1 : y * width + x + 2] = matrix.new_tensor([1.0, -2.0, 1.0])
            row += 1
    for y in range(1, height - 1):
        for x in range(width):
            matrix[row, (y - 1) * width + x] = 1.0
            matrix[row, y * width + x] = -2.0
            matrix[row, (y + 1) * width + x] = 1.0
            row += 1
    return matrix


@dataclass
class BSplineDisplacement:
    """Sum of one or more additive cubic B-spline displacement lattices."""

    coefficients: tuple[torch.Tensor, ...]
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("At least one B-spline displacement lattice is required")
        for coefficient in self.coefficients:
            if coefficient.ndim != 3 or coefficient.shape[-1] != 2:
                raise ValueError("Each coefficient lattice must have shape (height, width, 2)")
            if coefficient.shape[0] < 4 or coefficient.shape[1] < 4:
                raise ValueError("Cubic B-spline lattices must have at least 4 x 4 controls")

    @property
    def device(self) -> torch.device:
        return self.coefficients[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.coefficients[0].dtype

    def evaluate(self, points_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate displacement and analytic transformation gradient.

        Returns ``u`` with shape ``(N, 2)`` and ``F = I + grad(u)`` with
        shape ``(N, 2, 2)``. Coefficients remain differentiable through both.
        """
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy must have shape (N, 2)")
        count = len(points_xy)
        displacement = torch.zeros((count, 2), device=self.device, dtype=self.dtype)
        gradient = torch.zeros((count, 2, 2), device=self.device, dtype=self.dtype)
        for coefficient in self.coefficients:
            wx, wy, dwx, dwy, ix, iy = _spline_geometry(
                points_xy, self.image_size, coefficient.shape[:2]
            )
            values = coefficient[iy[:, :, None], ix[:, None, :]]
            basis = wy[:, :, None, None] * wx[:, None, :, None]
            derivative_x = wy[:, :, None, None] * dwx[:, None, :, None]
            derivative_y = dwy[:, :, None, None] * wx[:, None, :, None]
            displacement += (values * basis).sum(dim=(1, 2))
            gradient[:, :, 0] += (values * derivative_x).sum(dim=(1, 2))
            gradient[:, :, 1] += (values * derivative_y).sum(dim=(1, 2))
        identity = torch.eye(2, device=self.device, dtype=self.dtype).expand(count, 2, 2)
        return displacement, identity + gradient


def fit_bspline_field(
    points_xy: torch.Tensor,
    displacement_xy: torch.Tensor,
    image_size: tuple[int, int],
    control_shapes: tuple[tuple[int, int], ...],
    *,
    ridge: float = 1e-10,
    bending: float = 1e-3,
) -> BSplineDisplacement:
    """Fit displacement by least squares with affine-preserving bending control."""
    if displacement_xy.shape != points_xy.shape:
        raise ValueError("Displacement samples must have the same shape as point coordinates")
    blocks = [bspline_design_matrix(points_xy, image_size, shape) for shape in control_shapes]
    design = torch.cat(blocks, dim=1)
    constraints = []
    for shape in control_shapes:
        constraints.append(
            _bending_matrix(shape, device=design.device, dtype=design.dtype)
        )
    constraint_widths = [shape[0] * shape[1] for shape in control_shapes]
    total_width = sum(constraint_widths)
    rows = sum(item.shape[0] for item in constraints)
    bending_matrix = torch.zeros((rows, total_width), device=design.device, dtype=design.dtype)
    row_offset = column_offset = 0
    for constraint, width in zip(constraints, constraint_widths, strict=True):
        bending_matrix[
            row_offset : row_offset + constraint.shape[0], column_offset : column_offset + width
        ] = constraint
        row_offset += constraint.shape[0]
        column_offset += width
    design_blocks = [design]
    target_blocks = [displacement_xy]
    if bending:
        design_blocks.append(bending_matrix * bending**0.5)
        target_blocks.append(torch.zeros((rows, 2), device=design.device, dtype=design.dtype))
    if ridge:
        design_blocks.append(torch.eye(total_width, device=design.device, dtype=design.dtype) * ridge**0.5)
        target_blocks.append(torch.zeros((total_width, 2), device=design.device, dtype=design.dtype))
    coefficients = torch.linalg.lstsq(
        torch.cat(design_blocks, dim=0),
        torch.cat(target_blocks, dim=0),
        driver="gelsd" if design.device.type == "cpu" else None,
    ).solution
    result = []
    offset = 0
    for control_height, control_width in control_shapes:
        count = control_height * control_width
        result.append(coefficients[offset : offset + count].reshape(control_height, control_width, 2))
        offset += count
    return BSplineDisplacement(tuple(result), image_size)


def dense_bilinear_design_matrix(
    points_xy: torch.Tensor,
    image_size: tuple[int, int],
    shape: tuple[int, int],
) -> torch.Tensor:
    """Design matrix for a displacement control grid with bilinear evaluation."""
    width, height = image_size
    grid_height, grid_width = shape
    x = points_xy[:, 0] * (grid_width - 1) / (width - 1)
    y = points_xy[:, 1] * (grid_height - 1) / (height - 1)
    x0, y0 = torch.floor(x).long(), torch.floor(y).long()
    tx, ty = x - x0, y - y0
    x1, y1 = (x0 + 1).clamp(max=grid_width - 1), (y0 + 1).clamp(max=grid_height - 1)
    x0, y0 = x0.clamp(0, grid_width - 1), y0.clamp(0, grid_height - 1)
    ids = torch.stack(
        (y0 * grid_width + x0, y0 * grid_width + x1, y1 * grid_width + x0, y1 * grid_width + x1),
        dim=1,
    )
    weights = torch.stack(((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty), dim=1)
    matrix = torch.zeros(
        (len(points_xy), grid_height * grid_width), dtype=points_xy.dtype, device=points_xy.device
    )
    matrix.scatter_add_(1, ids, weights)
    return matrix


def fit_dense_bilinear_field(
    points_xy: torch.Tensor,
    displacement_xy: torch.Tensor,
    image_size: tuple[int, int],
    shape: tuple[int, int],
    *,
    ridge: float = 1e-7,
) -> DenseBilinearDisplacement:
    design = dense_bilinear_design_matrix(points_xy, image_size, shape)
    gram = design.T @ design
    if ridge:
        gram = gram + ridge * torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
    coefficients = torch.linalg.solve(gram, design.T @ displacement_xy)
    return DenseBilinearDisplacement(coefficients.reshape(*shape, 2), image_size)


@dataclass
class DenseBilinearDisplacement:
    """Directly predicted displacement grid with bilinear interpolation."""

    coefficients: torch.Tensor
    image_size: tuple[int, int]
    grid_origin_xy: tuple[float, float] | None = None
    grid_spacing_xy: tuple[float, float] | None = None

    def evaluate(self, points_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        width, height = self.image_size
        grid_height, grid_width = self.coefficients.shape[:2]
        if self.grid_origin_xy is None:
            origin_x, origin_y = 0.0, 0.0
        else:
            origin_x, origin_y = self.grid_origin_xy
        if self.grid_spacing_xy is None:
            spacing_x = (width - 1) / (grid_width - 1)
            spacing_y = (height - 1) / (grid_height - 1)
        else:
            spacing_x, spacing_y = self.grid_spacing_xy
        if spacing_x <= 0 or spacing_y <= 0:
            raise ValueError("Displacement-grid spacing must be positive")
        x = (points_xy[:, 0] - origin_x) / spacing_x
        y = (points_xy[:, 1] - origin_y) / spacing_y
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        tx, ty = x - x0, y - y0
        x1, y1 = (x0 + 1).clamp(max=grid_width - 1), (y0 + 1).clamp(max=grid_height - 1)
        x0, y0 = x0.clamp(0, grid_width - 1), y0.clamp(0, grid_height - 1)
        c00 = self.coefficients[y0, x0]
        c10 = self.coefficients[y0, x1]
        c01 = self.coefficients[y1, x0]
        c11 = self.coefficients[y1, x1]
        displacement = (
            c00 * ((1 - tx) * (1 - ty))[:, None]
            + c10 * (tx * (1 - ty))[:, None]
            + c01 * ((1 - tx) * ty)[:, None]
            + c11 * (tx * ty)[:, None]
        )
        sx, sy = 1.0 / spacing_x, 1.0 / spacing_y
        dx = ((c10 - c00) * (1 - ty)[:, None] + (c11 - c01) * ty[:, None]) * sx
        dy = ((c01 - c00) * (1 - tx)[:, None] + (c11 - c10) * tx[:, None]) * sy
        gradient = torch.stack((dx, dy), dim=-1)
        identity = torch.eye(2, dtype=points_xy.dtype, device=points_xy.device)
        return displacement, gradient + identity


def feature_grid_displacement_field(
    coefficients: torch.Tensor, image_size: tuple[int, int]
) -> DenseBilinearDisplacement:
    """Wrap a network output lattice using its pixel-centred feature geometry."""
    if coefficients.ndim != 3 or coefficients.shape[-1] != 2:
        raise ValueError("Feature displacement grid must have shape (height, width, 2)")
    width, height = image_size
    grid_height, grid_width = coefficients.shape[:2]
    spacing_xy = (width / grid_width, height / grid_height)
    origin_xy = (spacing_xy[0] / 2 - 0.5, spacing_xy[1] / 2 - 0.5)
    return DenseBilinearDisplacement(
        coefficients,
        image_size,
        grid_origin_xy=origin_xy,
        grid_spacing_xy=spacing_xy,
    )


def sample_image(image: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample an ``(H,W)`` image at pixel-centre coordinates."""
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale tensor")
    height, width = image.shape
    normalized = torch.stack(
        (2 * points_xy[:, 0] / max(width - 1, 1) - 1, 2 * points_xy[:, 1] / max(height - 1, 1) - 1),
        dim=-1,
    )
    grid = normalized.reshape(1, 1, -1, 2)
    return torch_functional.grid_sample(
        image[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0, 0, 0]


def image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference image gradients in x and y pixel coordinates."""
    gradient_y, gradient_x = torch.gradient(image, dim=(0, 1), edge_order=1)
    return gradient_x, gradient_y


def robust_mean(residual: torch.Tensor, *, kind: str = "charbonnier", delta: float = 0.02) -> torch.Tensor:
    if kind == "charbonnier":
        return torch.sqrt(residual.square() + delta * delta).mean()
    if kind == "huber":
        absolute = residual.abs()
        return torch.where(absolute <= delta, 0.5 * residual.square() / delta, absolute - 0.5 * delta).mean()
    raise ValueError("Robust loss kind must be 'charbonnier' or 'huber'")


def advected_photometric_losses(
    reference: torch.Tensor,
    target: torch.Tensor,
    points_xy: torch.Tensor,
    displacement_xy: torch.Tensor,
    jacobian: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    robust: str = "charbonnier",
) -> dict[str, torch.Tensor]:
    """Compute intensity and physically transported-gradient residuals.

    ``jacobian[n, i, j] = d phi_i / d x_j``. Target gradients are sampled at
    ``phi(x)`` and mapped to the reference frame using ``F.T @ grad(I1)``.
    The optional validity mask is supplied by training data, never inferred
    from analytic ground truth during image-pair inference.
    """
    if reference.shape != target.shape:
        raise ValueError("Reference and target images must have equal shapes")
    warped_xy = points_xy + displacement_xy
    height, width = reference.shape
    valid = (
        (warped_xy[:, 0] >= 0)
        & (warped_xy[:, 0] <= width - 1)
        & (warped_xy[:, 1] >= 0)
        & (warped_xy[:, 1] <= height - 1)
    )
    if valid_mask is not None:
        if valid_mask.shape != reference.shape:
            raise ValueError("valid_mask must have the same shape as the images")
        ix = points_xy[:, 0].round().long().clamp(0, width - 1)
        iy = points_xy[:, 1].round().long().clamp(0, height - 1)
        valid &= valid_mask[iy, ix].bool()
    if not torch.any(valid):
        raise ValueError("No valid advected sample points remain")
    ref = reference[points_xy[:, 1].round().long().clamp(0, height - 1), points_xy[:, 0].round().long().clamp(0, width - 1)]
    warped_intensity = sample_image(target, warped_xy)
    target_gradient_x, target_gradient_y = image_gradients(target)
    warped_gradient_x = sample_image(target_gradient_x, warped_xy)
    warped_gradient_y = sample_image(target_gradient_y, warped_xy)
    reference_gradient_x, reference_gradient_y = image_gradients(reference)
    ref_gradient = torch.stack(
        (
            reference_gradient_x[
                points_xy[:, 1].round().long().clamp(0, height - 1),
                points_xy[:, 0].round().long().clamp(0, width - 1),
            ],
            reference_gradient_y[
                points_xy[:, 1].round().long().clamp(0, height - 1),
                points_xy[:, 0].round().long().clamp(0, width - 1),
            ],
        ),
        dim=1,
    )
    target_gradient = torch.stack((warped_gradient_x, warped_gradient_y), dim=1)
    transported_gradient = torch.einsum("nji,nj->ni", jacobian, target_gradient)

    # Per-pair global gain/offset nuisance fit. Clamp gain to prevent the
    # photometric model from hiding large geometric errors.
    ref_valid, target_valid = ref[valid], warped_intensity[valid]
    variance = ref_valid.var(unbiased=False).clamp_min(1e-6)
    gain = (((ref_valid - ref_valid.mean()) * (target_valid - target_valid.mean())).mean() / variance).clamp(0.5, 2.0)
    offset = (target_valid.mean() - gain * ref_valid.mean()).clamp(-0.5, 0.5)
    intensity_scale = reference.std(unbiased=False).clamp_min(1e-3)
    gradient_scale = torch.sqrt(ref_gradient[valid].square().mean()).clamp_min(1e-3)
    intensity_loss = robust_mean(
        (warped_intensity[valid] - gain * ref_valid - offset) / intensity_scale, kind=robust
    )
    gradient_loss = robust_mean(
        (transported_gradient[valid] - gain * ref_gradient[valid]) / gradient_scale, kind=robust
    )
    return {
        "intensity": intensity_loss,
        "advected_gradient": gradient_loss,
        "gain": gain,
        "offset": offset,
        "valid_fraction": valid.to(reference.dtype).mean(),
    }


def uniform_points(
    image_size: tuple[int, int], spacing: int, *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    width, height = image_size
    x = torch.arange(0, width, spacing, dtype=torch.float64, device=device)
    y = torch.arange(0, height, spacing, dtype=torch.float64, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def feature_grid_points(
    image_size: tuple[int, int], grid_shape: tuple[int, int], *,
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return pixel-centre coordinates for an ``align_corners=False`` feature grid."""
    width, height = image_size
    grid_height, grid_width = grid_shape
    if min(width, height, grid_width, grid_height) < 1:
        raise ValueError("Image and feature-grid dimensions must be positive")
    stride_x, stride_y = width / grid_width, height / grid_height
    x = (torch.arange(grid_width, dtype=dtype, device=device) + 0.5) * stride_x - 0.5
    y = (torch.arange(grid_height, dtype=dtype, device=device) + 0.5) * stride_y - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def resize_pixel_center_coordinates(
    points_xy: torch.Tensor, scale_xy: tuple[float, float]
) -> torch.Tensor:
    """Map pixel-centre coordinates through an ``align_corners=False`` resize."""
    scale = points_xy.new_tensor(scale_xy)
    if torch.any(scale <= 0):
        raise ValueError("Resize scale factors must be positive")
    return (points_xy + 0.5) / scale - 0.5


def recommended_control_shape(image_size: tuple[int, int], spacing: float) -> tuple[int, int]:
    width, height = image_size
    return max(4, ceil((height - 1) / spacing) + 3), max(4, ceil((width - 1) / spacing) + 3)

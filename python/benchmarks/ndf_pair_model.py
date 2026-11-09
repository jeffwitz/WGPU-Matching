"""Coarse global image-pair matcher used for the first NDF-DIC inference gate.

This is a research baseline, not a production dense matcher. Shared learned
features are compared exhaustively at one coarse level; soft correspondences
form a spatial displacement field that is subsequently projected on splines.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from ndf_field import advected_photometric_losses
from torch import nn
from torch.nn import functional as F


@dataclass
class PairPrediction:
    displacement_xy: torch.Tensor
    target_xy: torch.Tensor
    hard_displacement_xy: torch.Tensor
    hard_target_xy: torch.Tensor
    topk_displacement_xy: torch.Tensor | None
    topk_target_xy: torch.Tensor | None
    confidence: torch.Tensor


@dataclass
class LocalNCCRefinement:
    displacement_xy: torch.Tensor
    score: torch.Tensor
    margin: torch.Tensor
    valid: torch.Tensor


@dataclass
class LocalFeatureRefinement:
    displacement_xy: torch.Tensor
    score: torch.Tensor
    margin: torch.Tensor
    valid: torch.Tensor


def refine_local_ncc(
    reference: torch.Tensor,
    target: torch.Tensor,
    initial_displacement_xy: torch.Tensor,
    *,
    search_radius: int = 6,
    patch_radius: int = 4,
    initial_jacobian: torch.Tensor | None = None,
    geometry: str = "translation",
) -> LocalNCCRefinement:
    """Refine a coarse field with local zero-mean normalized cross-correlation.

    The search is centred on the supplied image-pair estimate. It is therefore
    a coarse-to-fine refinement, not a global matcher. Coordinates follow the
    pixel-centre convention and ``initial_displacement_xy`` is sampled on the
    model's stride-four feature grid.
    """
    if reference.shape != target.shape or reference.ndim != 4 or reference.shape[:2] != (1, 1):
        raise ValueError("local NCC expects one monochrome image pair with matching [1, 1, H, W] shapes")
    if search_radius < 1 or patch_radius < 1:
        raise ValueError("search_radius and patch_radius must be positive")
    if geometry not in {"translation", "affine", "hybrid"}:
        raise ValueError("geometry must be 'translation', 'affine', or 'hybrid'")
    _, _, image_height, image_width = reference.shape
    _, field_height, field_width, components = initial_displacement_xy.shape
    if components != 2:
        raise ValueError("initial displacement must end with x/y components")
    device, dtype = reference.device, reference.dtype
    if initial_jacobian is None:
        initial_jacobian = torch.eye(2, device=device, dtype=dtype).expand(
            field_height, field_width, 2, 2
        )
    if initial_jacobian.shape != (field_height, field_width, 2, 2):
        raise ValueError("initial_jacobian must have shape [field_height, field_width, 2, 2]")
    if not torch.isfinite(initial_jacobian).all():
        raise ValueError("initial_jacobian must contain finite values")
    identity = torch.eye(2, device=device, dtype=dtype).expand(field_height, field_width, 2, 2)
    mode_jacobians = [identity] if geometry == "translation" else [initial_jacobian]
    if geometry == "hybrid":
        mode_jacobians.insert(0, identity)
    ys, xs = torch.meshgrid(
        torch.arange(field_height, device=device, dtype=dtype),
        torch.arange(field_width, device=device, dtype=dtype),
        indexing="ij",
    )
    stride_x, stride_y = image_width / field_width, image_height / field_height
    source_xy = torch.stack(((xs + 0.5) * stride_x - 0.5,
                             (ys + 0.5) * stride_y - 0.5), dim=-1).reshape(-1, 2)
    base_target_xy = source_xy + initial_displacement_xy.reshape(-1, 2)
    offset_y, offset_x = torch.meshgrid(
        torch.arange(-search_radius, search_radius + 1, device=device, dtype=dtype),
        torch.arange(-search_radius, search_radius + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    candidate_offsets = torch.stack((offset_x.flatten(), offset_y.flatten()), dim=-1)
    candidates = base_target_xy[:, None, :] + candidate_offsets[None, :, :]
    patch_y, patch_x = torch.meshgrid(
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=dtype),
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    patch_offsets = torch.stack((patch_x.flatten(), patch_y.flatten()), dim=-1)
    source_patch_xy = source_xy[:, None, :] + patch_offsets[None, :, :]
    source_grid = torch.stack(
        (2 * (source_patch_xy[..., 0] + 0.5) / image_width - 1,
         2 * (source_patch_xy[..., 1] + 0.5) / image_height - 1), dim=-1
    )
    source_patches = torch.nn.functional.grid_sample(
        reference, source_grid[None], mode="bilinear", padding_mode="zeros", align_corners=False
    )[0, 0]
    source_centered = source_patches - source_patches.mean(dim=-1, keepdim=True)
    source_norm = torch.linalg.vector_norm(source_centered, dim=-1).clamp_min(1e-6)
    source_valid = (
        (source_xy[:, 0] >= patch_radius) & (source_xy[:, 0] < image_width - patch_radius)
        & (source_xy[:, 1] >= patch_radius) & (source_xy[:, 1] < image_height - patch_radius)
    )
    mode_scores = []
    mode_valid = []
    for jacobian in mode_jacobians:
        warped_offsets = torch.einsum("hwij,pj->hwpi", jacobian, patch_offsets)
        warped_offsets = warped_offsets.reshape(-1, patch_offsets.shape[0], 2)
        candidate_patch_xy = candidates[:, :, None, :] + warped_offsets[:, None, :, :]
        target_grid = torch.stack(
            (2 * (candidate_patch_xy[..., 0] + 0.5) / image_width - 1,
             2 * (candidate_patch_xy[..., 1] + 0.5) / image_height - 1), dim=-1
        )
        target_patches = torch.nn.functional.grid_sample(
            target, target_grid.reshape(1, -1, patch_offsets.shape[0], 2),
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )[0, 0].reshape(len(source_xy), len(candidate_offsets), -1)
        target_centered = target_patches - target_patches.mean(dim=-1, keepdim=True)
        target_norm = torch.linalg.vector_norm(target_centered, dim=-1).clamp_min(1e-6)
        scores = torch.einsum("np,ncp->nc", source_centered, target_centered)
        scores = scores / (source_norm[:, None] * target_norm)
        extent_x = jacobian[..., 0, :].abs().sum(dim=-1).reshape(-1, 1) * patch_radius
        extent_y = jacobian[..., 1, :].abs().sum(dim=-1).reshape(-1, 1) * patch_radius
        candidate_valid = (candidates[..., 0] >= extent_x) & (candidates[..., 0] < image_width - extent_x)
        candidate_valid &= (candidates[..., 1] >= extent_y) & (candidates[..., 1] < image_height - extent_y)
        mode_scores.append(scores.masked_fill(~(candidate_valid & source_valid[:, None]), -2.0))
        mode_valid.append(candidate_valid)
    scores = torch.stack(mode_scores, dim=1)
    candidate_valid = torch.stack(mode_valid, dim=1)
    best_flat = scores.reshape(len(source_xy), -1).argmax(dim=-1)
    best_mode = best_flat // len(candidate_offsets)
    best_index = best_flat % len(candidate_offsets)
    sorted_scores = scores.reshape(len(source_xy), -1).topk(
        k=min(2, len(mode_jacobians) * scores.shape[-1]), dim=-1
    ).values
    best_score = sorted_scores[:, 0]
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    best_offset = candidate_offsets[best_index]
    displacement = initial_displacement_xy.reshape(-1, 2) + best_offset
    score_grid = scores.reshape(
        len(source_xy), len(mode_jacobians), 2 * search_radius + 1, 2 * search_radius + 1
    )[torch.arange(len(source_xy), device=device), best_mode]
    best_y = best_index // (2 * search_radius + 1)
    best_x = best_index % (2 * search_radius + 1)
    row = torch.arange(len(source_xy), device=device)
    inner_x = (best_x > 0) & (best_x < 2 * search_radius)
    inner_y = (best_y > 0) & (best_y < 2 * search_radius)
    sxm = score_grid[row, best_y, (best_x - 1).clamp_min(0)]
    sx0 = best_score
    sxp = score_grid[row, best_y, (best_x + 1).clamp_max(2 * search_radius)]
    sym = score_grid[row, (best_y - 1).clamp_min(0), best_x]
    sy0 = best_score
    syp = score_grid[row, (best_y + 1).clamp_max(2 * search_radius), best_x]
    delta_x = (0.5 * (sxm - sxp) / (sxm - 2 * sx0 + sxp).clamp_max(-1e-6)).clamp(-0.5, 0.5)
    delta_y = (0.5 * (sym - syp) / (sym - 2 * sy0 + syp).clamp_max(-1e-6)).clamp(-0.5, 0.5)
    displacement = displacement + torch.stack(
        (torch.where(inner_x, delta_x, torch.zeros_like(delta_x)),
         torch.where(inner_y, delta_y, torch.zeros_like(delta_y))), dim=-1
    )
    valid = source_valid & candidate_valid[row, best_mode, best_index]
    return LocalNCCRefinement(
        displacement.reshape(1, field_height, field_width, 2),
        best_score.reshape(1, field_height, field_width),
        margin.reshape(1, field_height, field_width),
        valid.reshape(1, field_height, field_width),
    )


class _FeatureEncoder(nn.Module):
    def __init__(self, dimensions: int = 32, contextual_features: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv2d(24, dimensions, 3, padding=1),
        )
        self.context = (
            nn.Sequential(
                nn.Conv2d(dimensions, dimensions, 3, padding=1),
                nn.GroupNorm(4, dimensions),
                nn.GELU(),
                nn.Conv2d(dimensions, dimensions, 3, padding=2, dilation=2),
                nn.GroupNorm(4, dimensions),
            )
            if contextual_features
            else None
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.layers(image)
        if self.context is not None:
            features = features + 0.25 * self.context(features)
        return F.normalize(features, dim=1, eps=1e-6)


class NDFPairMatcher(nn.Module):
    """Learned global coarse correspondence predictor from an image pair.

    The exhaustive correlation is bounded to the stride-four feature grid. It
    retains large translations, unlike a local search around zero displacement.
    """

    def __init__(
        self,
        feature_dimensions: int = 32,
        temperature: float = 0.08,
        contextual_features: bool = False,
    ):
        super().__init__()
        self.encoder = _FeatureEncoder(feature_dimensions, contextual_features)
        self.field_refiner = nn.Sequential(
            nn.Conv2d(2 * feature_dimensions + 3, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, 3, padding=1),
        )
        nn.init.zeros_(self.field_refiner[-1].weight)
        nn.init.zeros_(self.field_refiner[-1].bias)
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    def _correlate(self, reference: torch.Tensor, target: torch.Tensor):
        # Normalize each frame independently before feature extraction. The
        # synthetic generator and real cameras both vary in offset and gain;
        # this keeps that photometric nuisance out of the correspondence
        # embedding. Physical image losses still use the unmodified inputs.
        reference_centered = reference - reference.mean(dim=(-2, -1), keepdim=True)
        target_centered = target - target.mean(dim=(-2, -1), keepdim=True)
        reference_scale = reference_centered.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-4)
        target_scale = target_centered.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-4)
        first = self.encoder(reference_centered / reference_scale)
        second = self.encoder(target_centered / target_scale)
        source = first.flatten(2).transpose(1, 2)
        candidates = second.flatten(2).transpose(1, 2)
        temperature = self.log_temperature.exp().clamp(0.025, 0.5)
        logits = torch.bmm(source, candidates.transpose(1, 2)) / temperature
        probabilities = logits.softmax(dim=-1)
        return first, second, logits, probabilities

    def forward(
        self, reference: torch.Tensor, target: torch.Tensor, *, include_topk: bool = False
    ) -> PairPrediction:
        if reference.shape != target.shape or reference.ndim != 4:
            raise ValueError("reference and target must have matching [B, 1, H, W] shapes")
        if reference.shape[1] != 1:
            raise ValueError("the first pair matcher expects monochrome images")
        first, second, _, probabilities = self._correlate(reference, target)
        batch, _, feature_height, feature_width = first.shape
        candidates = second.flatten(2).transpose(1, 2)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(feature_height, dtype=reference.dtype, device=reference.device),
            torch.arange(feature_width, dtype=reference.dtype, device=reference.device),
            indexing="ij",
        )
        candidate_xy = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)
        expected_xy = probabilities @ candidate_xy
        hard_xy = candidate_xy[probabilities.argmax(dim=-1)]
        topk_target_xy = None
        topk_displacement_xy = None
        source_xy = candidate_xy.unsqueeze(0).expand(batch, -1, -1)
        stride = reference.new_tensor([reference.shape[-1] / feature_width,
                                       reference.shape[-2] / feature_height])
        source_xy = (source_xy + 0.5) * stride - 0.5
        target_xy = (expected_xy + 0.5) * stride - 0.5
        hard_target_xy = (hard_xy + 0.5) * stride - 0.5
        field_shape = (batch, feature_height, feature_width)
        if include_topk:
            top_probabilities, top_indices = probabilities.topk(
                k=min(8, probabilities.shape[-1]), dim=-1
            )
            top_weights = top_probabilities / top_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            topk_xy = (candidate_xy[top_indices] * top_weights[..., None]).sum(dim=-2)
            topk_target_xy = (topk_xy + 0.5) * stride - 0.5
            topk_displacement_xy = (topk_target_xy - source_xy).reshape(*field_shape, 2)
        confidence = probabilities.max(dim=-1).values
        raw_flow = (target_xy - source_xy).reshape(*field_shape, 2)
        expected_features = torch.bmm(probabilities, candidates).transpose(1, 2).reshape_as(first)
        refiner_input = torch.cat(
            (first, expected_features, raw_flow.permute(0, 3, 1, 2), confidence.reshape(*field_shape, 1).permute(0, 3, 1, 2)),
            dim=1,
        )
        refined_flow = raw_flow + self.field_refiner(refiner_input).permute(0, 2, 3, 1)
        return PairPrediction(
            displacement_xy=refined_flow,
            target_xy=source_xy.reshape(*field_shape, 2) + refined_flow,
            hard_displacement_xy=(hard_target_xy - source_xy).reshape(*field_shape, 2),
            hard_target_xy=hard_target_xy.reshape(*field_shape, 2),
            topk_displacement_xy=topk_displacement_xy,
            topk_target_xy=topk_target_xy,
            confidence=confidence.reshape(*field_shape),
        )


@torch.no_grad()
def refine_local_features(
    model: NDFPairMatcher,
    reference: torch.Tensor,
    target: torch.Tensor,
    initial_target_xy: torch.Tensor,
    *,
    search_radius_cells: int = 8,
) -> LocalFeatureRefinement:
    """Refine pair correspondences by searching learned features locally.

    The search is centred on the coarse pair prediction. Candidate descriptors
    are sampled from the shared CNN feature maps, so this step reuses the
    representation learned for global matching without building another dense
    all-pairs cost matrix.
    """
    if reference.shape != target.shape or reference.ndim != 4 or reference.shape[:2] != (1, 1):
        raise ValueError("local feature refinement expects one monochrome image pair")
    if search_radius_cells < 1:
        raise ValueError("search_radius_cells must be positive")
    reference_centered = reference - reference.mean(dim=(-2, -1), keepdim=True)
    target_centered = target - target.mean(dim=(-2, -1), keepdim=True)
    reference_scale = reference_centered.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-4)
    target_scale = target_centered.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-4)
    first = model.encoder(reference_centered / reference_scale)
    second = model.encoder(target_centered / target_scale)
    _, _, image_height, image_width = reference.shape
    _, _, feature_height, feature_width = first.shape
    stride_x, stride_y = image_width / feature_width, image_height / feature_height
    source_features = first.flatten(2).transpose(1, 2)[0]
    target_features = second.flatten(2).transpose(1, 2)[0]
    ys, xs = torch.meshgrid(
        torch.arange(feature_height, device=reference.device, dtype=reference.dtype),
        torch.arange(feature_width, device=reference.device, dtype=reference.dtype),
        indexing="ij",
    )
    source_xy = torch.stack(((xs + 0.5) * stride_x - 0.5,
                             (ys + 0.5) * stride_y - 0.5), dim=-1).reshape(-1, 2)
    target_xy = initial_target_xy.reshape(-1, 2)
    base_x = torch.round((target_xy[:, 0] + 0.5) / stride_x - 0.5).long()
    base_y = torch.round((target_xy[:, 1] + 0.5) / stride_y - 0.5).long()
    offset_y, offset_x = torch.meshgrid(
        torch.arange(-search_radius_cells, search_radius_cells + 1, device=reference.device),
        torch.arange(-search_radius_cells, search_radius_cells + 1, device=reference.device),
        indexing="ij",
    )
    offsets = torch.stack((offset_x.flatten(), offset_y.flatten()), dim=-1)
    candidate_x = base_x[:, None] + offsets[None, :, 0]
    candidate_y = base_y[:, None] + offsets[None, :, 1]
    valid = (candidate_x >= 0) & (candidate_x < feature_width)
    valid &= (candidate_y >= 0) & (candidate_y < feature_height)
    candidate_ids = candidate_y.clamp(0, feature_height - 1) * feature_width + candidate_x.clamp(0, feature_width - 1)
    candidate_features = target_features[candidate_ids]
    logits = (candidate_features * source_features[:, None, :]).sum(dim=-1)
    logits = logits.masked_fill(~valid, -1e4)
    temperature = model.log_temperature.exp().clamp(0.025, 0.5)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    candidate_xy = torch.stack(
        ((candidate_x.to(reference.dtype) + 0.5) * stride_x - 0.5,
         (candidate_y.to(reference.dtype) + 0.5) * stride_y - 0.5), dim=-1
    )
    candidate_xy = torch.where(valid[..., None], candidate_xy, torch.zeros_like(candidate_xy))
    expected_xy = (probabilities[..., None] * candidate_xy).sum(dim=1)
    sorted_logits = logits.topk(k=min(2, logits.shape[-1]), dim=-1).values
    best_score = sorted_logits[:, 0]
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    any_valid = valid.any(dim=-1)
    displacement = expected_xy - source_xy
    return LocalFeatureRefinement(
        displacement.reshape(1, feature_height, feature_width, 2),
        best_score.reshape(1, feature_height, feature_width),
        margin.reshape(1, feature_height, feature_width),
        any_valid.reshape(1, feature_height, feature_width),
    )


def correspondence_targets(
    flow_xy: torch.Tensor, visible: torch.Tensor, feature_shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample analytic flow on feature-cell centres and return discrete targets."""
    batch, _, image_height, image_width = flow_xy.shape
    feature_height, feature_width = feature_shape
    flow = F.interpolate(flow_xy, size=feature_shape, mode="bilinear", align_corners=False)
    valid = F.interpolate(visible[:, None].float(), size=feature_shape, mode="nearest")[:, 0] > 0.5
    ys, xs = torch.meshgrid(
        torch.arange(feature_height, device=flow_xy.device, dtype=flow_xy.dtype),
        torch.arange(feature_width, device=flow_xy.device, dtype=flow_xy.dtype),
        indexing="ij",
    )
    stride_x, stride_y = image_width / feature_width, image_height / feature_height
    source_x = (xs + 0.5) * stride_x - 0.5
    source_y = (ys + 0.5) * stride_y - 0.5
    target_x = source_x[None] + flow[:, 0]
    target_y = source_y[None] + flow[:, 1]
    candidate_x = torch.round((target_x + 0.5) / stride_x - 0.5).long()
    candidate_y = torch.round((target_y + 0.5) / stride_y - 0.5).long()
    valid &= (candidate_x >= 0) & (candidate_x < feature_width)
    valid &= (candidate_y >= 0) & (candidate_y < feature_height)
    targets = candidate_y.clamp(0, feature_height - 1) * feature_width + candidate_x.clamp(0, feature_width - 1)
    return targets.reshape(batch, -1), valid.reshape(batch, -1), flow.permute(0, 2, 3, 1)


def pair_matching_loss(
    model: NDFPairMatcher,
    reference: torch.Tensor,
    target: torch.Tensor,
    flow_xy: torch.Tensor,
    visible: torch.Tensor,
    *,
    physical_weight: float = 0.0,
    physical_pyramid_levels: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train descriptor discrimination and coarse displacement jointly."""
    first, second, logits, probability = model._correlate(reference, target)
    batch, _, height, width = first.shape
    candidates = second.flatten(2).transpose(1, 2)
    targets, valid, true_flow = correspondence_targets(
        flow_xy, visible, (height, width)
    )
    if not torch.any(valid):
        raise ValueError("the synthetic pair has no valid coarse correspondences")
    cross_entropy = F.cross_entropy(logits[valid], targets[valid])
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=reference.dtype, device=reference.device),
        torch.arange(width, dtype=reference.dtype, device=reference.device),
        indexing="ij",
    )
    candidate_xy = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)
    expected = probability @ candidate_xy
    source_xy = candidate_xy.unsqueeze(0).expand(batch, -1, -1)
    stride = reference.new_tensor([reference.shape[-1] / width,
                                   reference.shape[-2] / height])
    raw_flow = ((expected - source_xy) * stride).reshape(batch, height, width, 2)
    expected_features = torch.bmm(probability, candidates).transpose(1, 2).reshape_as(first)
    confidence = probability.max(dim=-1).values.reshape(batch, 1, height, width)
    refiner_input = torch.cat(
        (first, expected_features, raw_flow.permute(0, 3, 1, 2), confidence), dim=1
    )
    predicted_flow = raw_flow + model.field_refiner(refiner_input).permute(0, 2, 3, 1)
    flow_l1 = (predicted_flow - true_flow).abs().sum(dim=-1)
    flow_loss = flow_l1.reshape(batch, -1)[valid].mean()
    physical_loss = reference.new_zeros(())
    if physical_weight:
        if physical_weight < 0:
            raise ValueError("physical_weight must be non-negative")
        if physical_pyramid_levels < 1:
            raise ValueError("physical_pyramid_levels must be positive")
        spacing_x = reference.shape[-1] / width
        spacing_y = reference.shape[-2] / height
        gradient_y_x, gradient_x_x = torch.gradient(
            predicted_flow[..., 0], spacing=(spacing_y, spacing_x), dim=(1, 2)
        )
        gradient_y_y, gradient_x_y = torch.gradient(
            predicted_flow[..., 1], spacing=(spacing_y, spacing_x), dim=(1, 2)
        )
        jacobian = torch.stack(
            (
                torch.stack((1 + gradient_x_x, gradient_y_x), dim=-1),
                torch.stack((gradient_x_y, 1 + gradient_y_y), dim=-1),
            ),
            dim=-2,
        )
        sample_y, sample_x = torch.meshgrid(
            torch.arange(1, height - 1, 2, device=reference.device),
            torch.arange(1, width - 1, 2, device=reference.device),
            indexing="ij",
        )
        sample_ids = (sample_y * width + sample_x).flatten()
        source_x = (sample_x.to(reference.dtype) + 0.5) * spacing_x - 0.5
        source_y = (sample_y.to(reference.dtype) + 0.5) * spacing_y - 0.5
        points_xy = torch.stack((source_x, source_y), dim=-1).reshape(-1, 2)
        physical_terms = []
        for index in range(batch):
            level_terms = []
            for level in range(physical_pyramid_levels):
                factor = 2**level
                if level == 0:
                    reference_level = reference[index, 0]
                    target_level = target[index, 0]
                    visible_level = visible[index]
                else:
                    reference_level = F.avg_pool2d(
                        reference[index : index + 1], factor, stride=factor
                    )[0, 0]
                    target_level = F.avg_pool2d(
                        target[index : index + 1], factor, stride=factor
                    )[0, 0]
                    visible_level = (
                        F.avg_pool2d(
                            visible[index : index + 1, None].float(),
                            factor,
                            stride=factor,
                        )[0, 0]
                        >= 1.0
                    )
                level_points = (points_xy + 0.5) / factor - 0.5
                level_flow = predicted_flow[index].reshape(-1, 2)[sample_ids] / factor
                physical = advected_photometric_losses(
                    reference_level,
                    target_level,
                    level_points,
                    level_flow,
                    jacobian[index].reshape(-1, 2, 2)[sample_ids],
                    valid_mask=visible_level,
                )
                level_terms.append(physical["intensity"] + physical["advected_gradient"])
            physical_terms.append(torch.stack(level_terms).mean())
        physical_loss = torch.stack(physical_terms).mean()
    loss = cross_entropy + 0.10 * flow_loss + physical_weight * physical_loss
    with torch.no_grad():
        top1 = logits.argmax(dim=-1)
        accuracy = (top1[valid] == targets[valid]).float().mean()
        flow_error = torch.linalg.vector_norm(
            predicted_flow - true_flow, dim=-1
        ).reshape(batch, -1)[valid].median()
    return loss, {
        "cross_entropy": cross_entropy.detach(),
        "flow_l1_px": flow_loss.detach(),
        "top1_grid_accuracy": accuracy,
        "median_flow_error_px": flow_error,
        "valid_fraction": valid.float().mean(),
        "physical_loss": physical_loss.detach(),
        "physical_pyramid_levels": reference.new_tensor(float(physical_pyramid_levels)),
    }

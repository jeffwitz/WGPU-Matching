"""Keep measured inliers of the existing local WGSL affine consensus models."""

from __future__ import annotations

import math
import struct
import time

import numpy as np
import wgpu
from wgpu_matching import CorrespondenceSet
from wgpu_matching._gpu import (
    ResourceScope,
    create_buffer,
    dispatch,
    make_bind_group,
    make_pipeline,
    read_array,
    shader_source,
    uniform_buffer,
    upload_buffer,
)


class LocalConsensus:
    def __init__(self, matcher, radius=256.0, threshold=3.0):
        self.matcher, self.radius, self.threshold = matcher, radius, threshold
        self.device = matcher.device
        self.pipeline = make_pipeline(
            self.device,
            self.device.create_shader_module(code=shader_source("local-consensus-filter.wgsl")),
            "retain",
            "Measured local-consensus pair filter",
        )

    def filter(self, pairs, seeds):
        started = time.perf_counter()
        ids = np.flatnonzero(pairs.accepted)
        keep = np.zeros(len(pairs.source_xy), dtype=bool)
        field = None
        if len(ids) >= 3:
            observed = pairs.pairs_xyxy[ids]
            field = self.matcher.fit_local_affine(
                observed,
                seeds,
                radius_px=self.radius,
                threshold_px=self.threshold,
                min_inliers=6,
                min_candidates=8,
                min_inlier_fraction=0.2,
                max_iterations=256,
                include_affine_models=True,
            )
            models = np.zeros((len(seeds), 3, 4), dtype=np.float32)
            models[:, 0, :2] = seeds
            models[:, 0, 2] = [m.usable for m in field.matches]
            models[:, 1:] = field.affine_models.reshape(-1, 2, 4)
            with ResourceScope() as scope:
                point_buffer = upload_buffer(
                    self.device,
                    scope,
                    observed.astype(np.float32),
                    wgpu.BufferUsage.STORAGE,
                    "Observed pairs",
                )
                model_buffer = upload_buffer(
                    self.device, scope, models, wgpu.BufferUsage.STORAGE, "Local affine models"
                )
                output = create_buffer(
                    self.device,
                    scope,
                    len(ids) * 4,
                    wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                    "Local consensus flags",
                )
                params = uniform_buffer(
                    self.device,
                    scope,
                    struct.pack("<2I2f", len(ids), len(seeds), self.radius, self.threshold),
                    "Consensus filter parameters",
                )
                group = make_bind_group(
                    self.device,
                    self.pipeline,
                    [
                        {"binding": i, "resource": {"buffer": buffer}}
                        for i, buffer in enumerate((point_buffer, model_buffer, output, params))
                    ],
                )
                encoder = self.device.create_command_encoder()
                dispatch(encoder, self.pipeline, group, math.ceil(len(ids) / 64))
                self.device.queue.submit([encoder.finish()])
                keep[ids] = read_array(self.device, output, np.dtype(np.uint32), len(ids)) != 0
        reasons = list(pairs.rejection_reasons)
        for i in ids:
            if not keep[i]:
                reasons[i] = "local_consensus_rejected"
        result = CorrespondenceSet(
            pairs.source_xy,
            pairs.target_xy,
            pairs.image_size,
            pairs.matcher_id,
            match_ids=pairs.match_ids,
            accepted=keep,
            raw_scores=pairs.raw_scores,
            rejection_reasons=tuple(reasons),
            coordinate_mapping=pairs.coordinate_mapping,
        )
        diagnostics = {
            "backend": "WGSL local RANSAC and WGSL pair residual filtering",
            "radius_px": self.radius,
            "threshold_px": self.threshold,
            "minimum_inliers": 6,
            "minimum_candidates": 8,
            "minimum_inlier_fraction": 0.2,
            "upstream_accepted": len(ids),
            "retained_original_pairs": int(keep.sum()),
            "local_seed_models": 0 if field is None else len(field.accepted),
            "wall_s": time.perf_counter() - started,
            "seed_models": []
            if field is None
            else [
                {
                    "x": m.x,
                    "y": m.y,
                    "dx": m.dx,
                    "dy": m.dy,
                    "usable": m.usable,
                    "inliers": m.inliers,
                    "candidates": m.candidates,
                    "residual": m.residual,
                }
                for m in field.matches
            ],
        }
        return result, diagnostics


def bounded_pairs(pairs, limit=8192, tiles=(16, 9)):
    """Source-space balanced reduction for a bounded solver; no scores or truth."""
    flags = (
        np.ones(len(pairs.source_xy), bool)
        if pairs.accepted is None
        else np.asarray(pairs.accepted, bool).copy()
    )
    flags &= np.isfinite(pairs.pairs_xyxy).all(axis=1)
    for points in (pairs.source_xy, pairs.target_xy):
        flags &= (points[:, 0] >= 0) & (points[:, 0] < pairs.image_size[0])
        flags &= (points[:, 1] >= 0) & (points[:, 1] < pairs.image_size[1])
    accepted = np.flatnonzero(flags)
    if len(accepted) <= limit:
        return pairs, {"input": len(accepted), "selected": len(accepted), "applied": False}
    source = pairs.source_xy[accepted]
    cells = np.floor(source / np.asarray(pairs.image_size) * tiles).astype(int)
    cells = np.clip(cells, 0, np.asarray(tiles) - 1)
    buckets = cells[:, 1] * tiles[0] + cells[:, 0]
    occupied, counts = np.unique(buckets, return_counts=True)
    quotas = np.zeros(len(occupied), int)
    remaining = limit
    while remaining:
        active = np.flatnonzero(quotas < counts)
        increment = max(1, remaining // len(active))
        for k in active:
            take = min(increment, int(counts[k] - quotas[k]), remaining)
            quotas[k] += take
            remaining -= take
            if not remaining:
                break
    selected = []
    for bucket, quota in zip(occupied, quotas, strict=True):
        ids = accepted[buckets == bucket]
        positions = np.lexsort((ids, pairs.source_xy[ids, 0], pairs.source_xy[ids, 1]))
        ids = ids[positions]
        selected.extend(ids[np.linspace(0, len(ids) - 1, int(quota)).astype(int)])
    selected = np.asarray(sorted(selected), dtype=int)
    reduced = CorrespondenceSet(
        pairs.source_xy[selected],
        pairs.target_xy[selected],
        pairs.image_size,
        pairs.matcher_id,
        match_ids=tuple(pairs.match_ids[i] for i in selected),
        accepted=np.ones(len(selected), bool),
        raw_scores=None if pairs.raw_scores is None else pairs.raw_scores[selected],
        coordinate_mapping=pairs.coordinate_mapping,
    )
    return reduced, {
        "input": len(accepted),
        "selected": len(selected),
        "applied": True,
        "tiles": tiles,
        "criterion": "balanced source-space quotas; uniform within each cell",
    }

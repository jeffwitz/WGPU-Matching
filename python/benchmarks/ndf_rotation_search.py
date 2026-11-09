"""Pure NumPy helpers for coarse-to-fine rotation hypothesis selection."""

from __future__ import annotations

import numpy as np


def rotation_refinement_candidates(
    candidate_scores: list[dict[str, float | int]],
    *,
    coarse_steps: int,
    top_k: int,
) -> list[float]:
    """Return half-step angles around the best coarse hypotheses.

    Scores must already be computed from image-derived matching. The returned
    angles are wrapped to [-pi, pi) and exclude angles already evaluated.
    """
    if coarse_steps < 1 or top_k < 1:
        return []
    step = 2.0 * np.pi / coarse_steps
    ranked = sorted(
        candidate_scores,
        key=lambda item: (float(item["ranking_score"]), int(item["consensus_support"])),
        reverse=True,
    )[:top_k]
    existing = [float(item["angle_rad"]) for item in candidate_scores]
    refined: list[float] = []
    for item in ranked:
        center = float(item["angle_rad"])
        for offset in (-0.5 * step, 0.5 * step):
            angle = (center + offset + np.pi) % (2.0 * np.pi) - np.pi
            if any(
                abs((angle - old + np.pi) % (2.0 * np.pi) - np.pi) < 1e-8
                for old in (*existing, *refined)
            ):
                continue
            refined.append(float(angle))
    return refined


def rotation_bank_angles(steps: int) -> np.ndarray:
    """Build the uniform angular grid and guarantee that identity is tested."""
    if steps < 1:
        raise ValueError("rotation-bank steps must be positive")
    return np.unique(np.r_[np.linspace(-np.pi, np.pi, steps, endpoint=False), 0.0])

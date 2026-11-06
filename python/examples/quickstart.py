"""Build a small reproducible speckle pair and run every Python matcher."""

from __future__ import annotations

import numpy as np
from wgpu_matching import DeepMatchingOptions, HORGOptions, WGPUMatching


def speckle_pair() -> tuple[np.ndarray, np.ndarray]:
    """Return a 512 x 512 image and a target shifted by (7, 5) pixels."""
    rng = np.random.default_rng(7)
    reference = rng.integers(0, 256, (128, 128), dtype=np.uint8).repeat(4, axis=0).repeat(4, axis=1)
    target = np.zeros_like(reference)
    target[5:, 7:] = reference[:-5, :-7]
    return reference, target


def main() -> None:
    reference, target = speckle_pair()
    seeds = ((128.0, 128.0), (256.0, 256.0), (384.0, 384.0))
    methods = (
        ("DeepMatching direct", "direct"),
        ("DeepMatching INRIA", "inria"),
        ("HORG-DIC", "horg"),
    )

    with WGPUMatching() as matcher:
        print(f"Adapter: {matcher.device_name}")
        for label, method in methods:
            if method == "horg":
                field = matcher.match_horg(
                    reference,
                    target,
                    options=HORGOptions(
                        patch_size=64,
                        stride=128,
                        reduction=2,
                        max_features=512,
                        seed_points=seeds,
                    ),
                )
            else:
                field = matcher.match_deepmatching(
                    reference,
                    target,
                    options=DeepMatchingOptions(
                        mode=method,
                        patch_size=64,
                        root_patch_size=128,
                        stride=128,
                        downscale_levels=1,
                        search_radius_px=48,
                        seed_points=seeds,
                    ),
                )

            print(f"\n{label}")
            for match in field.matches:
                print(
                    f"  ({match.x:.1f}, {match.y:.1f}) -> "
                    f"({match.dx:+.2f}, {match.dy:+.2f}) px; "
                    f"usable={match.usable}"
                )


if __name__ == "__main__":
    main()

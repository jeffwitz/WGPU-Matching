"""Plot accuracy and complete-call latency from an already measured campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "deepmatching": "DeepMatching",
    "horg": "HORG",
    "cpm": "CPM-HOG",
    "xfeat": "XFeat",
    "efficientloftr": "EfficientLoFTR",
    "raco": "RaCo-ALIKED-LightGlue+",
    "dad_b": "DaD + DeDoDe-B",
    "dedode_b": "DeDoDe v2-L + DeDoDe-B",
    "dic_tensor_b": "DIC tensor + DeDoDe-B",
    "dic_learned_b": "Learned DIC + DeDoDe-B",
    "horg_b": "HORG/SIFT + DeDoDe-B",
}


def save(fig, root, name):
    fig.savefig(root / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(root / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot(root):
    rows = json.loads((root / "summary.json").read_text())["rows"]
    manifest = json.loads((root / "manifest.json").read_text())
    methods = manifest["protocol"]["methods"]
    affine = {(r["case"], r["method"]): r for r in rows if r["basis_requested"] == "affine"}
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#243b4c",
            "text.color": "#243b4c",
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
        }
    )
    strains = [10, 20, 30, 40, 50, 60]
    coverage = np.zeros((len(methods), len(strains)))
    counts = {}
    for i, method in enumerate(methods):
        for j, strain in enumerate(strains):
            a = affine[(f"tensile_{strain}pct", method)]["assessment"]
            coverage[i, j] = 100 * a["correct_lt5"] / a["requested"]
            counts[i, j] = f"{a['correct_lt5']}/{a['requested']}"
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    image = ax.imshow(coverage, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    for (i, j), label in counts.items():
        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="white" if coverage[i, j] >= 85 or coverage[i, j] < 15 else "#152d3c",
        )
    ax.set(
        xticks=np.arange(len(strains)),
        xticklabels=[f"{v}%" for v in strains],
        yticks=np.arange(len(methods)),
        yticklabels=[LABELS[m] for m in methods],
        xlabel="Axial engineering strain",
        title="Verified observations → global affine initialization",
    )
    fig.colorbar(image, ax=ax, label="Correct coverage strictly below 5 pixels (%)", shrink=0.8)
    fig.subplots_adjust(left=0.27, bottom=0.12, right=0.95)
    save(fig, figures, "correct_coverage_by_strain")
    y = np.arange(len(methods))
    fig, (ax, timing) = plt.subplots(
        1, 2, figsize=(12, 6.5), sharey=True, gridspec_kw={"width_ratios": [1.3, 1]}
    )
    stress = [affine[("tensile_60pct", method)] for method in methods]
    ax.barh(
        y - 0.16,
        [r["assessment"]["correct_le2"] for r in stress],
        height=0.29,
        color="#137a91",
        label="At most 2 pixels",
    )
    ax.barh(
        y + 0.16,
        [r["assessment"]["correct_lt5"] for r in stress],
        height=0.29,
        color="#57b69c",
        label="Strictly below 5 pixels",
    )
    ax.set(
        yticks=y,
        yticklabels=[LABELS[m] for m in methods],
        xlim=(0, 17),
        xticks=[0, 4, 8, 12, 16],
        xlabel="Correct centres / 16 visible requests",
        title="60% traction: primary affine field",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.10), frameon=False, fontsize=9)
    times = [r.get("affine_pipeline_median_s", np.nan) for r in stress]
    low = [
        t - min(r["affine_pipeline_wall_s"]) if r.get("affine_pipeline_wall_s") else 0
        for r, t in zip(stress, times, strict=True)
    ]
    high = [
        max(r["affine_pipeline_wall_s"]) - t if r.get("affine_pipeline_wall_s") else 0
        for r, t in zip(stress, times, strict=True)
    ]
    timing.barh(y, times, height=0.55, color="#385772", xerr=[low, high], capsize=3)
    timing.set(
        xscale="log",
        xlabel="Seconds per complete affine call (log scale)",
        title="Median and observed three-call range",
    )
    timing.grid(axis="x", alpha=0.2)
    for i, value in enumerate(times):
        timing.text(
            max(stress[i]["affine_pipeline_wall_s"]) * 1.13,
            i,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    timing.set_xlim(0.05, max(times) * 2.7)
    fig.subplots_adjust(left=0.23, wspace=0.12, bottom=0.15)
    save(fig, figures, "stress_60_accuracy_and_latency")
    provenance = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "source_protocol_sha256": manifest["protocol_sha256"],
        "scope": "Primary affine field, exact main-campaign summary; no new measurements",
        "count_thresholds": "inclusive <=2 pixels; strictly <5 pixels",
        "timing": "median and observed min/max of three complete affine calls",
    }
    (figures / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    plot(args.run)


if __name__ == "__main__":
    main()

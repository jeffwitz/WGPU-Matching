"""Report actual support sizes, native/quadratic remaps and mixed seed exports."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from qualify_hudge_adaptive_support import sample_patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("results/hudge_adaptive_support_20261115")
FIG = ROOT/"figures"
MODELS = ["affine32", "quadratic32_zero", "quadratic32_rbf", "affine16_dense"]
LABELS = ["Affine 32", "Quadratique 32, courbure initiale nulle",
          "Quadratique 32, courbure RBF", "Affine 16, pas 8"]


def save(fig, name):
    fig.savefig(FIG/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG/f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    combined = ROOT/"mixed_centres"
    combined.mkdir(exist_ok=True)
    summary = []
    for run in ["all_levels", "all_levels_continuation4"]:
        rows = json.loads((ROOT/run/"summary.json").read_text())
        assert len(rows) == 764
        for model in MODELS:
            group = [r for r in rows if r["model"] == model]
            correlations = []
            for r in group:
                d = np.load(ROOT/run/model/f"level_{r['level']:03d}.npz")
                correlations.extend(d["correlation"][d["accepted"]].tolist())
            summary.append({"run": run, "model": model,
                            "old_centres_recovered": sum(r["old_centres_recovered"] for r in group),
                            "accepted_candidates": sum(r["accepted_candidates"] for r in group),
                            "candidates": sum(r["new_candidates"] for r in group),
                            "area_fraction": sum(r["covered_pixels"] for r in group)/334848,
                            "median_correlation": float(np.median(correlations)),
                            "minimum_correlation": min(correlations)})
    (ROOT/"comparison.json").write_text(json.dumps(summary, indent=2))

    # Image-only deterministic union, never selected by unknown analytical truth.
    # This is an assembled candidate seed field, not another solver execution.
    mixed_sizes = {16: 0, 32: 0}
    for level in range(1, 192):
        base = np.load(Path("results/hudge_rbf_restart_20261115/all_levels/rbf_gradient_g0.50")/f"level_{level:03d}.npz")
        a = np.load(ROOT/"all_levels/affine32"/f"level_{level:03d}.npz")
        f = np.load(ROOT/"all_levels/affine16_dense"/f"level_{level:03d}.npz")
        params, accepted = base["parameters"].copy(), base["accepted"].copy()
        size = np.where(accepted, 32, 0)
        origin = np.where(accepted, 0, -1)
        for i, global_index in enumerate(a["failed_indices"]):
            if a["accepted"][i]:
                params[global_index] = a["parameters"][i]
                accepted[global_index], size[global_index], origin[global_index] = True, 32, 1
                mixed_sizes[32] += 1
            else:
                index = np.flatnonzero(np.all(np.isclose(f["centres"], base["centres"][global_index]), axis=1))
                if len(index) and f["accepted"][index[0]]:
                    params[global_index] = f["parameters"][index[0]]
                    accepted[global_index], size[global_index], origin[global_index] = True, 16, 2
                    mixed_sizes[16] += 1
        assert np.all(accepted[base["valid_roi"]])
        np.savez_compressed(combined/f"level_{level:03d}.npz", centres=base["centres"],
                            parameters=params, accepted=accepted, valid_roi=base["valid_roi"],
                            support_size=size, source=origin)
    (ROOT/"mixed_summary.json").write_text(json.dumps({"new_32": mixed_sizes[32], "new_16": mixed_sizes[16],
                                                       "total_centres": 6876, "policy": "baseline, else accepted affine32, else accepted fine16 at exact same centre"}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(4)
    for j, run in enumerate(["all_levels", "all_levels_continuation4"]):
        group = [r for r in summary if r["run"] == run]
        axes[0].bar(x+(j-.5)*.36, [r["old_centres_recovered"] for r in group], .36,
                    label="1 appel par proposition" if j == 0 else "Jusqu'à 4 appels par proposition")
        axes[1].bar(x+(j-.5)*.36, [100*r["area_fraction"] for r in group], .36)
    for ax in axes:
        ax.set_xticks(x, ["Affine 32", "Quadratique 32\nzéro", "Quadratique 32\nRBF", "Affine 16\npas 8"])
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Anciens centres récupérés / 327")
    axes[1].set_ylabel("Surface des régions en échec couverte (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "comparison")

    reference = np.asarray(Image.open("/home/jeff/CNRS/DIC/HudgeStrains/FEA/Img_F.tif"))
    target = np.asarray(Image.open("/home/jeff/CNRS/DIC/HudgeStrains/FEA/Img_wa_g191.tif"))
    base = np.load("results/hudge_rbf_restart_20261115/all_levels/rbf_gradient_g0.50/level_191.npz")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, model in zip(axes, ["affine32", "quadratic32_zero", "affine16_dense"], strict=True):
        d = np.load(ROOT/"all_levels_continuation4"/model/"level_191.npz")
        ax.imshow(reference[:225, :440], cmap="gray", vmin=0, vmax=255)
        for c in d["failed_centres"]:
            ax.add_patch(Rectangle(c-16, 32, 32, fill=False, ec="#ff9b2f", lw=2))
        for i, c in enumerate(d["centres"]):
            color = "#22ef58" if d["accepted"][i] else "#ff2222"
            ax.plot(*c, ".", color=color, ms=5)
        ax.set_title(f"{model}\n{d['accepted'].sum()}/{len(d['centres'])} candidats retenus")
        ax.set_xlim(0, 440)
        ax.set_ylim(225, 0)
    fig.suptitle("Niveau 191 — régions fixes orange ; points verts retenus ; rouges non convergés")
    fig.tight_layout()
    save(fig, "regions_191")

    for window in [65, 164]:
        centre = base["centres"][window]
        ref32 = sample_patch(reference, centre, np.zeros(6), 32)[0]
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes[0, 0].imshow(ref32, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title("Référence, 32×32")
        axes[1, 0].axis("off")
        for col, model in enumerate(["affine32", "quadratic32_zero", "affine16_dense"], 1):
            d = np.load(ROOT/"all_levels_continuation4"/model/"level_191.npz")
            i = np.flatnonzero(np.all(np.isclose(d["centres"], centre), axis=1))[0]
            size = int(d["size"])
            patch = sample_patch(target, centre, d["parameters"][i], size)[0]
            canvas = np.full((32, 32), np.nan)
            residual = canvas.copy()
            start = (32-size)//2
            canvas[start:start+size, start:start+size] = patch
            residual[start:start+size, start:start+size] = np.abs(patch-ref32[start:start+size, start:start+size])
            axes[0, col].set_facecolor("#ccc")
            axes[1, col].set_facecolor("#ccc")
            axes[0, col].imshow(canvas, cmap="gray", vmin=0, vmax=255)
            axes[1, col].imshow(residual, cmap="magma", vmin=0, vmax=128)
            axes[0, col].set_title(f"{model}\ncorrélation {d['correlation'][i]:.3f} ; statut {int(d['status'][i])}")
            axes[1, col].set_title("Résidu absolu, 0–128 / 255")
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Fenêtre {window}, niveau 191 — sorties convergées après continuation\nGris uniforme : hors support 16×16, non évalué")
        fig.tight_layout()
        save(fig, f"patch_{window}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

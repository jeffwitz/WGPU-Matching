"""Paired aggregate and remapped images for the frozen RBF restart experiment."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from diagnose_hudge_remaps import correlation, sample
from hudge_rbf_restart import SeedField
from PIL import Image
from scipy.ndimage import map_coordinates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("results/hudge_rbf_restart_20261115")
RUN = ROOT / "all_levels"
FIG = ROOT / "figures"
NAMES = {"affine": "Affine", "rbf_position": "RBF déplacements", "rbf_gradient": "RBF déplacements + gradients"}
COLORS = ["#566573", "#d35400", "#16867b"]


def save(fig, name):
    fig.savefig(FIG/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG/f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def field_from_archive(path):
    d = np.load(path)
    return SeedField(d["matrix"], d["translation"], d["sites"], d["coefficients"], float(d["length"]))


def remap(target, field):
    y, x = np.mgrid[:225, :440]
    points = np.column_stack((x.ravel(), y.ravel()))
    q = points + field.evaluate(points)[:, [0, 3]]
    return map_coordinates(target.astype(float), [q[:, 1], q[:, 0]], order=1,
                           mode="constant", cval=float(target[-1, -1]), prefilter=False).reshape(x.shape)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    images = ROOT/"images"
    images.mkdir(exist_ok=True)
    rows = json.loads((RUN/"summary.json").read_text())
    assert len(rows) == 191*9, "do not report an incomplete campaign"
    summary = []
    lines = ["# Affine puis RBF avec gradients : comparaison appariée", "",
             "191 cibles comparées indépendamment à la même référence. 36 fenêtres par paire.",
             "Les 6 876 observations ne sont pas des échantillons indépendants : une seule plaque.",
             "Aucune vérité de déplacement associée : il s'agit d'acceptation, pas de couverture correcte démontrée.", "",
             "| Champ | Seuil gradient | Retenues / 6 876 | Gain sur phase affine | Médiane reprise (s) |",
             "|---|---:|---:|---:|---:|"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, tau in zip(axes, [.25, .5, 1.], strict=True):
        for method, color in zip(NAMES, COLORS, strict=True):
            group = sorted([r for r in rows if r["method"] == method and r["tau_g"] == tau], key=lambda r: r["level"])
            count = sum(r["final"] for r in group)
            start = sum(r["stage1"] for r in group)
            item = {"method": method, "tau_g": tau, "accepted": count, "stage1": start,
                    "added": count-start, "median_wall_s": float(np.median([r["wall_s"] for r in group])),
                    "below_065": sum(r["below_065"] for r in group),
                    "out_of_bounds": sum(r["out_of_bounds"] for r in group),
                    "nonpositive_jacobians": sum(r["nonpositive_jacobians"] for r in group),
                    "full_pairs": sum(r["final"] == 36 for r in group),
                    "round_limit_pairs": sum(len(r["rounds"]) == 5 and r["rounds"][-1]["after"] > r["rounds"][-1]["before"] for r in group)}
            summary.append(item)
            lines.append(f"| {NAMES[method]} | {tau:.2f} | {count} | +{count-start} | {item['median_wall_s']:.3f} |")
            ax.plot([r["level"] for r in group], [r["final"] for r in group], label=NAMES[method], color=color, lw=1.1)
        ax.plot([r["level"] for r in group], [r["stage1"] for r in group], color="black", ls=":", label="Phase affine archivée")
        ax.set_title(f"Seuil de cohérence du gradient : {tau:.2f}")
        ax.set_xlabel("Indice de la cible (pas un taux de déformation)")
        ax.set_ylim(16, 37)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Fenêtres retenues / 36")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "coverage")
    (ROOT/"comparison.json").write_text(json.dumps(summary, indent=2))
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n")

    reference = np.asarray(Image.open("/home/jeff/CNRS/DIC/HudgeStrains/FEA/Img_F.tif"))
    for level in (50, 150, 191):
        target = np.asarray(Image.open(f"/home/jeff/CNRS/DIC/HudgeStrains/FEA/Img_wa_g{level:03d}.tif"))
        fig, axes = plt.subplots(3, 3, figsize=(14, 8))
        for row, method in enumerate(NAMES):
            for col, tau in enumerate([.25, .5, 1.]):
                d = np.load(RUN/f"{method}_g{tau:.2f}/level_{level:03d}.npz")
                ax = axes[row, col]
                ax.imshow(reference[:225, :440], cmap="gray", vmin=0, vmax=255)
                for i in np.flatnonzero(d["valid_roi"]):
                    x, y = d["centres"][i]
                    color = "#44d2ff" if d["baseline_accepted"][i] else ("#48ff48" if d["accepted"][i] else "#ff6d30")
                    ax.add_patch(Rectangle((x-16, y-16), 32, 32, fill=False, ec=color))
                ax.set_title(f"{NAMES[method]} ; seuil {tau:.2f} : {d['accepted'].sum()}/36", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"Niveau {level} : bleu = phase affine ; vert = ajouté ; orange = non retenu")
        fig.tight_layout()
        save(fig, f"map_{level}")

    level = 191
    target = np.asarray(Image.open(f"/home/jeff/CNRS/DIC/HudgeStrains/FEA/Img_wa_g{level:03d}.tif"))
    folder = RUN/"rbf_gradient_g1.00"
    rounds = sorted(folder.glob("level_191_round_*.npz"))
    stages = [("Référence", reference[:225, :440])]
    affine = field_from_archive(RUN/"affine_g1.00/level_191_round_1.npz")
    stages.append(("Proposition affine", remap(target, affine)))
    for i, p in enumerate(rounds):
        stages.append((f"RBF + gradients, tour {i+1}", remap(target, field_from_archive(p))))
    fig, axes = plt.subplots(2, len(stages), figsize=(4.2*len(stages), 5.4))
    for j, (title, image) in enumerate(stages):
        axes[0, j].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(title)
        axes[1, j].imshow(np.abs(image-reference[:225, :440]), cmap="magma", vmin=0, vmax=128)
        axes[1, j].set_title("Résidu absolu, 0–128 / 255")
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
        Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255))).save(images/f"191_{j:02d}.png")
    fig.tight_layout()
    save(fig, "remap_191")

    d = np.load(folder/"level_191.npz")
    last = np.load(rounds[-1])
    patch_values = []
    calls = np.load(folder/"level_191_calls.npz")
    for index in (102, 65, 164):
        centre = d["centres"][index]
        ref = sample(reference, centre, np.zeros(6))
        states = [("Affine proposée", affine.evaluate([centre])[0]),
                  ("RBF, première proposition", np.load(rounds[0])["proposal"][index]),
                  ("RBF, dernière proposition", last["proposal"][index])]
        if d["accepted"][index]:
            states.append(("État finalement retenu", d["parameters"][index]))
        else:
            ids = np.flatnonzero(calls["indices"] == index)
            if len(ids):
                states.append((f"Dernière sortie, statut {int(calls['raw_status'][ids[-1]])}", calls["outputs"][ids[-1]]))
        fig, axes = plt.subplots(2, len(states)+1, figsize=(14, 5.8))
        axes[0, 0].imshow(ref, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title("Référence")
        axes[1, 0].axis("off")
        values = []
        for j, (title, state) in enumerate(states, 1):
            image = sample(target, centre, state)
            c = correlation(ref, image)
            values.append({"stage": title, "correlation": c})
            axes[0, j].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[0, j].set_title(f"{title}\ncorrélation {c:.3f}", fontsize=10)
            axes[1, j].imshow(np.abs(image-ref), cmap="magma", vmin=0, vmax=128)
            axes[1, j].set_title("Résidu absolu, 0–128 / 255", fontsize=9)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Niveau 191 — fenêtre {index} — {'retenue' if d['accepted'][index] else 'NON retenue'}")
        fig.tight_layout()
        save(fig, f"patch_191_{index}")
        patch_values.append({"index": index, "accepted": bool(d["accepted"][index]), "values": values,
                             "merged_reason": int(last["merged_reason"][index]),
                             "restart_reason": int(last["restart_reason"][index])})
    (ROOT/"patches.json").write_text(json.dumps(patch_values, indent=2))


if __name__ == "__main__":
    main()

"""Figures and a local gallery for the observed-contour remapping experiment."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    root = Path("results/hudge_dense_remap_20261115")
    output = root/"figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 3, figsize=(14, 7.6), constrained_layout=True)
    for row, level in enumerate((50, 150, 191)):
        folder = root/"refined"/f"level_{level:03d}"
        data = json.loads((folder/"summary.json").read_text())
        crop = data["crop_xyxy"]
        for col, (name, title) in enumerate((("image1_ref.png", "Reference"), ("dense_def_cor.png", "Dense 16-pixel supports"),
                                             ("image2_def_cor.png", "Dense supports + observed contours"))):
            axes[row, col].imshow(Image.open(folder/name).crop(crop), cmap="gray", vmin=0, vmax=255)
            axes[row, col].axis("off")
            subtitle = f"Level {level}; {data['anchors']}/979 anchors" if col == 0 else (
                f"Material overlap {100*data['scores'][col-1]['material_iou']:.2f}%")
            axes[row, col].set_title(title+"\n"+subtitle)
    fig.savefig(output/"remapping_stages.pdf")
    fig.savefig(output/"remapping_stages.png", dpi=180)
    plt.close(fig)
    folder = root/"refined"/"level_191"
    data = json.loads((folder/"summary.json").read_text())
    field = np.load(folder/"field.npz")
    ref_mask, mapped_mask = field["reference_material"], field["remapped_material"]
    corrected = np.asarray(Image.open(folder/"image2_def_cor.png"))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, hole in zip(axes.ravel(), data["scores"][1]["holes"], strict=True):
        cx, cy = hole["reference"]["centre"]
        radius = max(20, np.sqrt(hole["reference"]["area"]/np.pi)+12)
        x0, x1 = max(0, int(cx-radius)), min(1024, int(cx+radius+1))
        y0, y1 = max(0, int(cy-radius)), min(1024, int(cy+radius+1))
        ax.imshow(corrected[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
        ax.contour(ref_mask[y0:y1, x0:x1], levels=[.5], colors=["#f6b526"], linewidths=1.5)
        ax.contour(mapped_mask[y0:y1, x0:x1], levels=[.5], colors=["#38dbef"], linewidths=1.)
        ax.set_title(f"Centre ({cx:.1f}, {cy:.1f}) px\nAxis ratio {hole['corrected']['axis_ratio']:.3f}; overlap {100*hole['iou']:.1f}%")
        ax.axis("off")
    fig.suptitle("Level 191: raw remapped pixels; gold = reference contour, cyan = corrected contour")
    fig.savefig(output/"holes_191.pdf")
    fig.savefig(output/"holes_191.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, name, title in zip(axes, ("image1_ref.png", "image0_def.png", "image2_def_cor.png"),
                                ("Reference", "Original deformed image", "Corrected image"), strict=True):
        ax.imshow(Image.open(folder/name), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title+" — 1024 × 1024 pixels")
        ax.set_xlabel("x (native pixels)")
        ax.set_ylabel("y (native pixels)")
    fig.savefig(output/"native_images.pdf")
    fig.savefig(output/"native_images.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    validation = root/"offset_validation_refined"
    for ax, name in zip(axes, ("coarse36", "dense16", "dense16_contours"), strict=True):
        a = np.load(validation/f"{name}.npz")
        ax.imshow(Image.open(folder/"image1_ref.png"), cmap="gray", vmin=0, vmax=255)
        for value, color in ((False, "#e37528"), (True, "#1682c4")):
            c = a["centres"][a["accepted"] == value]
            ax.scatter(c[:, 0], c[:, 1], s=7, color=color)
        ax.set(xlim=(0, 440), ylim=(225, 0), title=f"{name}: {a['accepted'].sum()}/{len(a['accepted'])}")
        ax.set_aspect("equal")
    fig.suptitle("24-pixel supports at offset centres: blue = retained; orange = not retained")
    fig.savefig(output/"offset_validation.pdf")
    fig.savefig(output/"offset_validation.png", dpi=180)
    plt.close(fig)
    sections = []
    for level in (50, 150, 191):
        prefix = f"refined/level_{level:03d}"
        sections.append(f'''<section><h2>Niveau {level}</h2>
<p>Alternance référence / image déformée remappée. Agrandissement ×2 sans lissage d'affichage.</p>
<img src="{prefix}/reference_corrected_zoom.gif" alt="Alternance niveau {level}">
<p><a href="{prefix}/image1_ref.png">Référence native</a> ·
<a href="{prefix}/image2_def_cor.png">Déformée corrigée native</a> ·
<a href="{prefix}/image0_def.png">Déformée originale</a> ·
<a href="{prefix}/initialization.npz">Initialisations à six paramètres</a></p></section>''')
    (root/"index.html").write_text('''<!doctype html><html lang="fr"><meta charset="utf-8">
<title>Plaque trouée — remappings natifs</title><style>
body{font:17px/1.5 system-ui;max-width:1200px;margin:35px auto;padding:0 20px;color:#203040}
img{max-width:100%;height:auto}section{margin:35px 0;border-top:1px solid #ccc}a{color:#176696}
</style><h1>Plaque trouée : initialisation dense et remapping</h1>
<section id="diagnostic-pres-des-trous" style="border:2px solid #a32;padding:16px;background:#fff3e8">
<h2>Diagnostic du mouchetis près des trous — niveau 191</h2>
<p><strong>La correction des contours n’est pas une initialisation qualifiée près des trous.</strong>
Elle rapproche les formes mais dégrade le mouchetis entre 4 et 16 pixels des trous.
Aucun des 825 centres de contrôle n’est situé à moins de 8 pixels d’un trou.</p>
<table style="border-collapse:collapse;width:100%;text-align:left">
<caption>Corrélation des intensités dans les bandes de matière de référence</caption>
<thead><tr><th>Distance aux trous</th><th>Avant correction</th><th>Après correction</th></tr></thead>
<tbody><tr><td>4–8 pixels</td><td>0,700</td><td><strong>0,614</strong></td></tr>
<tr><td>8–16 pixels</td><td>0,864</td><td><strong>0,774</strong></td></tr></tbody></table>
<p>Ces corrélations regroupent les pixels de chaque bande : ce ne sont pas des scores de fenêtres du solveur.</p>
<a href="near_hole_audit/texture_residuals.png"><img src="near_hole_audit/texture_residuals.png"
alt="Comparaison du mouchetis et des résidus avant et après correction des contours"></a>
<p>En haut : référence, champ dense avant correction, puis correction des contours.
En bas : couronne de matière examinée et écarts absolus d’intensité sur une échelle commune.
Les trous blancs dans les cartes de résidus sont hors matière et ne sont pas évalués.</p>
<p><a href="near_hole_audit/texture_residuals.png">Ouvrir la figure en pleine résolution</a> ·
<a href="near_hole_audit/README.md">Diagnostic détaillé</a> ·
<a href="near_hole_audit/summary.json">Résultats bruts</a></p></section>
<p>Les pixels proviennent exclusivement de l'image déformée originale, échantillonnée avec le champ estimé.
Aucun cercle ni fond n'est redessiné. Les contours observés participent à l'ajustement : leur bon accord
n'est donc pas une validation mécanique indépendante.</p>
<p>Au niveau 191, le contrôle sur 825 centres décalés retient 764 fenêtres avec le champ grossier,
780 avec le champ dense et 782 avec la correction de contour. Il reste des échecs.</p>
<p><a href="report.pdf">Document explicatif en PDF</a> · <a href="README.md">Protocole et fichiers</a></p>
<img src="figures/remapping_stages.png" alt="Étapes des remappings">'''+"".join(sections)+'''
<section><h2>Les six trous au niveau 191</h2><img src="figures/holes_191.png" alt="Contours des trous">
<p>Or : contour de référence. Cyan : contour après remapping. Le petit trou reste mal restitué.</p></section>
<section><h2>Reprise du solveur</h2><img src="figures/offset_validation.png" alt="Convergence aux centres décalés">
<p>Les supports de 24 pixels recouvrent partiellement les observations de 16 pixels : ce contrôle
n'est pas un test sur des images indépendantes. Les statuts ne certifient pas l'exactitude du déplacement.</p></section></html>''')


if __name__ == "__main__":
    main()

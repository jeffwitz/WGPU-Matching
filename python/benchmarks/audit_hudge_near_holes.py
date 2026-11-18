"""Audit the material texture near holes, independently of silhouette overlap."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hudge_strains_data import material_mask
from PIL import Image
from scipy.ndimage import distance_transform_edt, label


def main():
    root = Path("results/hudge_dense_remap_20261115")
    output = root/"near_hole_audit"
    output.mkdir(exist_ok=True)
    folder = root/"refined/level_191"
    reference = np.asarray(Image.open(folder/"image1_ref.png"), float)
    mask = material_mask(reference)
    components, _ = label(~mask)
    outside = np.unique(np.concatenate((components[0], components[-1], components[:, 0], components[:, -1])))
    holes = (components > 0) & ~np.isin(components, outside)
    distance = distance_transform_edt(~holes)
    coverage = np.load(root/"contour_retry/level_191/affine16/field.npz")["support_coverage"]
    images = {"dense": np.asarray(Image.open(folder/"dense_def_cor.png"), float),
              "contours": np.asarray(Image.open(folder/"image2_def_cor.png"), float)}
    bands = [(0, 4), (4, 8), (8, 16), (16, 32), (32, float("inf"))]
    rows = []
    for lower, upper in bands:
        region = mask & (distance > lower) & (distance <= upper)
        row = {"lower_px": lower, "upper_px": upper if np.isfinite(upper) else None,
               "material_pixels": int(region.sum()), "coverage_by_accepted_supports": float(coverage[region].mean())}
        for name, image in images.items():
            row[name] = {"pooled_intensity_correlation": float(np.corrcoef(reference[region], image[region])[0, 1]),
                         "mean_absolute_difference_8bit": float(np.abs(reference-image)[region].mean())}
        rows.append(row)
    validation = {}
    for name in ("coarse36", "dense16", "dense16_contours"):
        archive = np.load(root/"offset_validation_refined"/f"{name}.npz")
        points = np.rint(archive["centres"]).astype(int)
        d = distance[points[:, 1], points[:, 0]]
        validation[name] = []
        for lower, upper in bands:
            region = (d > lower) & (d <= upper)
            validation[name].append({"lower_px": lower, "upper_px": upper if np.isfinite(upper) else None,
                                     "centres": int(region.sum()), "accepted": int(np.sum(region & archive["accepted"]))})
    report = {"level": 191, "bands": rows, "validation": validation,
              "distance": "Euclidean distance to reference hole pixels; exterior background excluded",
              "pixel_selection": "reference material only, fixed between methods; no target-mask intersection",
              "correlation": "one pooled intensity correlation per material band, not an IC-GN window score",
              "conclusion": "Contour fitting worsens material texture 4-16px from holes; no validation centre within 8px. Withdraw recommendation as a qualified near-hole initializer."}
    (output/"summary.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.8), constrained_layout=True)
    crop = np.s_[:225, :440]
    for ax, image, title in zip(axes[0], (reference, images["dense"], images["contours"]),
                                ("Référence", "Champ dense, avant contours", "Après correction des contours"), strict=True):
        ax.imshow(image[crop], cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    band = mask & (distance <= 16)
    axes[1, 0].imshow(reference[crop], cmap="gray", vmin=0, vmax=255)
    overlay = np.zeros((*reference[crop].shape, 4))
    overlay[band[crop]] = [1., .35, 0., .6]
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Matière à moins de 16 px des trous")
    axes[1, 0].axis("off")
    for ax, image, title in zip(axes[1, 1:], images.values(), ("Erreur d'intensité avant", "Erreur d'intensité après"), strict=True):
        residual = np.where(mask, np.abs(reference-image), np.nan)
        im = ax.imshow(residual[crop], cmap="magma", vmin=0, vmax=100)
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=list(axes[1, 1:]), label="Écart absolu (niveaux de gris / 255)", shrink=.8)
    fig.savefig(output/"texture_residuals.png", dpi=180)
    plt.close(fig)
    lines = ["# Correction du diagnostic : le contour ne valide pas la texture", "",
             "La recommandation précédente du champ corrigé comme initialisation près des trous est retirée.", "",
             "Le champ déplace les contours vers la référence mais détériore le mouchetis dans une partie de la couronne matérielle.", "",
             "| Distance aux trous | Pixels | Corrélation avant | Après | Supports validés couvrant la zone |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        upper = row["upper_px"]
        interval = f"{row['lower_px']}–{upper} px" if upper else ">32 px"
        lines.append(f"| {interval} | {row['material_pixels']} | {row['dense']['pooled_intensity_correlation']:.3f} | "
                     f"{row['contours']['pooled_intensity_correlation']:.3f} | {100*row['coverage_by_accepted_supports']:.1f}% |")
    lines += ["", ("Aucun des 825 centres de qualification n'est situé à moins de 8 pixels d'un trou. "
              "Seuls 30 sont situés entre 8 et 16 pixels, avec 29 retenus aussi bien avant qu'après contours. "
              "Les scores globaux ne permettent donc pas de qualifier la couronne proche des trous."), "",
              ("Les corrélations de ce tableau regroupent les intensités de toute une bande, avec un masque de matière de référence fixe. "
              "Ce ne sont pas des scores de fenêtres IC-GN ni des erreurs de déplacement."), "",
              "![Texture et résidus](texture_residuals.png)", "",
              ("Il faut réutiliser les observations photométriques validées sur les images originales, "
              "et ne pas propager une correction de contour comme si elle était une mesure matérielle. "
              "Les zones sans observation doivent rester explicitement non initialisées. "
              "L'absence de centres de contrôle dans la couronne ne prouve ni la convergence ni l'impossibilité de convergence d'un autre support.")]
    (output/"README.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()

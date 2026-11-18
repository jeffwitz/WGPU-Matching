"""Illustrate the saved image-only proposal, without rerunning inference."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/benchmarks"))
from summarize_hudge_strains import photometric_check


def main():
    result = ROOT / "results/hudge_strains_20261114"
    record = result / "silhouette_32/FEA/level_191.json"
    archive = record.with_suffix(".npz")
    metadata = json.loads(record.read_text())
    source = Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA")
    reference = np.asarray(Image.open(source / "Img_F.tif"), dtype=float) / 255
    target = np.asarray(Image.open(source / "Img_wa_g191.tif"), dtype=float) / 255
    y, x = np.mgrid[:225, :440]
    sx, sy = metadata["extent_ratio_xy"]
    tx, ty = metadata["translation_xy"]
    proposal = map_coordinates(target, [sy*y+ty, sx*x+tx], order=1,
                               mode="constant", cval=np.nan, prefilter=False)
    with np.load(archive) as data:
        _, _, corrected = photometric_check(reference, target, data["centres"],
                                            data["parameters"], data["accepted"], 32)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#e3a1a1")
    for ax, image, bounds, title in [
        (axes[0, 0], reference[:225, :440], metadata["reference_bounds"], "(a) Reference and observed box"),
        (axes[0, 1], target[:, :440], metadata["target_bounds"], "(b) Target 191 and observed box"),
    ]:
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
        lx, ly, hx, hy = bounds
        ax.add_patch(Rectangle((lx, ly), hx-lx, hy-ly, fill=False, edgecolor="#e58300", linewidth=1.6))
        ax.set_title(title)
    axes[1, 0].imshow(proposal, cmap=cmap, vmin=0, vmax=1)
    axes[1, 0].set_title("(c) Box proposal only\nNo local solve; not an accepted field")
    axes[1, 1].imshow(corrected[:225, :440], cmap=cmap, vmin=0, vmax=1)
    axes[1, 1].set_title("(d) After local solves and propagation\n18/36 accepted; pink = unavailable")
    for ax in axes.ravel():
        ax.set(xlabel="x (pixels)", ylabel="y (pixels)")
    fig.suptitle(f"Image-only extent proposal: sx = {sx:g}, sy = {sy:.5f}, ty = {ty:.5f} px\n"
                 "Panels (a), (c), (d): reference coordinates; panel (b): target coordinates", fontsize=11)
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(result / f"figures/silhouette_method.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    provenance = {"record": str(record), "archive": str(archive),
                  "reference": str(source / "Img_F.tif"), "target": str(source / "Img_wa_g191.tif"),
                  "reference_crop_xy": [0, 0, 440, 225], "truth_used": False,
                  "proposal": "bilinear target sampling at S*x+t; saved box geometry",
                  "final": "saved accepted six-parameter local states; no gap filling"}
    (result / "figures/silhouette_method_sources.json").write_text(json.dumps(provenance, indent=2)+"\n")


if __name__ == "__main__":
    main()

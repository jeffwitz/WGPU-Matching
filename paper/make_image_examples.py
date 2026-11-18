"""Image-based manuscript illustrations from archived solves, not new benchmarks.

The local pullback uses only archived final parameters, never analytical truth.
Full images are downsampled for printing; local patches use native pixels.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
BASE = ROOT / "results/seed_propagation_20261111"
STUDY = BASE / "p3_fine_transfer_multifamily_independent"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                    "pdf.fonttype": 42, "axes.spines.top": False,
                    "axes.spines.right": False})
manifest = []


def read_image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.


def sample(image, points):
    return map_coordinates(image, [points[..., 1], points[..., 0]],
                           order=1, mode="constant", cval=np.nan,
                           prefilter=False)


def make_case(name, pair, real=False):
    if real:
        archive = BASE / "p3_external_experimental/adrien_masked/qualification/evaluation" / (pair + ".npz")
        inputs = Path("/home/jeff/CNRS/DIC/AdrienVariational")
        reference_path = inputs / "img_000000_0.09422.tiff"
        target_path = inputs / "img_000075_1608.88508.tiff"
        size = 64
    else:
        archive = STUDY / "fine" / (pair + ".npz")
        inputs = STUDY / "inputs/affine_ncc/solver_images_seed62000000"
        reference_path = inputs / (pair + "_reference.png")
        target_path = inputs / (pair + "_target.png")
        size = 32
    a = np.load(archive)
    ref, target = read_image(reference_path), read_image(target_path)
    xy, parameters = a["centres"], a["parameters"]
    h, w = ref.shape
    if real:
        domain = a["valid_roi"].astype(bool)
        success = a["accepted"].astype(bool) & domain
        groups = [(domain & ~success, "#D55E00", "Not accepted"),
                  (success, "#009E73", "Accepted (accuracy unknown)")]
        summary = f"{success.sum():,} accepted / {domain.sum():,} masked centres"
    else:
        domain = a["visible"].astype(bool)
        success = a["correct"].astype(bool) & domain
        prior = a["prior_correct"].astype(bool)
        groups = [(domain & ~success, "#D55E00", "Unresolved / incorrect"),
                  (success & prior, "#0072B2", "Correct at both supports"),
                  (success & ~prior, "#009E73", "Recovered at 32 pixels")]
        summary = f"{(prior & domain).sum():,} to {success.sum():,} correct / {domain.sum():,} visible"
    # An ordinary accepted example: nearest the image centre, not best residual.
    eligible = success & np.isfinite(parameters).all(axis=1)
    if not real:
        eligible &= ~a["support_crossing"].astype(bool)
    idx = np.flatnonzero(eligible)
    index = int(idx[np.argmin(np.sum(((xy[idx] - [w/2, h/2]) / [w, h])**2, axis=1))])
    centre = xy[index].astype(float)
    p = parameters[index].astype(float)
    F = np.array([[1+p[1], p[2]], [p[4], 1+p[5]]])
    mapped_centre = centre + p[[0, 3]]
    t = np.arange(size) - (size-1)/2
    yy, xx = np.meshgrid(t, t, indexing="ij")
    offsets = np.stack([xx, yy], axis=-1)
    reference_patch = sample(ref, centre + offsets)
    target_patch = sample(target, mapped_centre + offsets)
    warped_patch = sample(target, mapped_centre + offsets @ F.T)
    residual = np.abs(reference_patch - warped_patch)
    assert np.isfinite(residual).all(), "Selected patch must be fully in the image"

    fig = plt.figure(figsize=(7.15, 5.7))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=.075, right=.985, top=.86, bottom=.16, hspace=.70)
    upper = outer[0].subgridspec(1, 3, wspace=.24)
    lower = outer[1].subgridspec(1, 4, wspace=.34)
    top_axes = [fig.add_subplot(upper[0, i]) for i in range(3)]
    # Antialiased full-frame views; display reduction never enters the pullback.
    for ax, image, label in zip(top_axes, [ref, target, ref],
                               ["(a) Reference image", "(b) Deformed image", "(c) Solver outcome"]):
        thumb = Image.fromarray(np.uint8(np.clip(image, 0, 1)*255))
        thumb.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        ax.imshow(thumb, cmap="gray", vmin=0, vmax=255,
                  extent=(-.5, w-.5, h-.5, -.5))
        ax.set(title=label, xlabel="x (pixels)", ylim=(h-.5, -.5))
        ax.tick_params(labelsize=6)
        ax.title.set_fontsize(8)
    top_axes[0].set_ylabel("y (pixels)")
    for mask, color, label in groups:
        top_axes[2].scatter(*xy[mask].T, s=2.1, color=color,
                            marker="s", linewidths=0, label=label,
                            rasterized=True)
    handles, labels = top_axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.54, .48),
               fontsize=7, frameon=False, markerscale=2, ncol=len(groups))
    # Crosshairs locate a true-size outlined patch, even in a full 8K view.
    for ax, c in [(top_axes[0], centre), (top_axes[1], mapped_centre)]:
        ax.add_patch(Rectangle(c-size/2, size, size, fill=False,
                               edgecolor="#CC79A7", linewidth=1))
        ax.plot(*c, marker="+", color="#CC79A7", markersize=12, mew=.9)
    patches = [reference_patch, target_patch, warped_patch, residual]
    labels = ["(d) Reference support", "(e) Translation only",
              "(f) Affine pullback", "(g) Absolute residual"]
    for i, (patch, label) in enumerate(zip(patches, labels)):
        ax = fig.add_subplot(lower[0, i])
        image = ax.imshow(patch, cmap="gray" if i < 3 else "magma",
                          vmin=0, vmax=1 if i < 3 else .25,
                          interpolation="nearest",
                          extent=(-size/2, size/2, size/2, -size/2))
        ax.set(title=label, xlabel="Local x (pixels)", xticks=[-size/2, 0, size/2],
               yticks=[-size/2, 0, size/2])
        ax.tick_params(labelsize=6)
        ax.title.set_fontsize(7.5)
        if i == 0:
            ax.set_ylabel("Local y (pixels)")
        if i == 3:
            cax = fig.add_axes([.805, .065, .17, .012])
            cb = fig.colorbar(image, cax=cax, orientation="horizontal",
                              ticks=[0, .125, .25], extend="max")
            cb.set_label("Intensity difference (0–1)", fontsize=6)
            cb.ax.tick_params(labelsize=6)
    fig.suptitle(name + "\n" + summary, fontsize=10)
    stem = "experimental_speckle" if real else ("spatial_transfer" if pair.startswith("heterogeneous") else "discontinuous_speckle")
    fig.savefig(OUT / "figures" / (stem + ".pdf"), dpi=350)
    fig.savefig(OUT / "review" / (stem + ".png"), dpi=180)
    plt.close(fig)
    record = {"figure": stem, "pair": pair, "native_shape": [h, w],
              "requested_centres": len(xy), "evaluation_centres": int(domain.sum()),
              "correct_or_accepted": int(success.sum()), "truth_available": not real,
              "patch_index": index, "patch_centre": centre.tolist(),
              "patch_size": size, "parameters": p.tolist(),
              "selection": "accepted/correct non-crossing centre nearest image centre; no residual ranking",
              "difference": "absolute raw intensity difference, no photometric fitting; scale clipped at 0.25",
              "sources": []}
    for path in [archive, reference_path, target_path]:
        record["sources"].append({"path": str(path),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest.append(record)
    print(stem, summary, "patch", index)


make_case("Synthetic heterogeneous deformation", "heterogeneous_texture62051005_warp62091005")
make_case("Synthetic displacement discontinuity", "discontinuous_texture62051004_warp62091004")
make_case("Experimental speckle: Adrien, frames 000000–000075", "heterogeneous_texture34051001_warp34091001", real=True)
(OUT / "data/image_examples.json").write_text(json.dumps(manifest, indent=2)+"\n")

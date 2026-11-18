"""Publication-style diagnostics from saved independent-pair solves only."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hudge_strains_data import material_mask
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from summarize_hudge_strains import photometric_check


def save(figure, directory, name):
    for extension in ("png", "pdf", "svg"):
        figure.savefig(directory/f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/hudge_strains_20261114"))
    args = parser.parse_args()
    rows = json.loads((args.root/"summary.json").read_text())
    figures = args.root/"figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    labels = {"learned": "Learned proposals, 64 px", "silhouette_64": "Silhouette, 64 px",
              "silhouette_32": "Silhouette, 32 px"}
    for ax, family in zip(axes, ["FEA", "ClosedFormExpression"], strict=True):
        for method, label in labels.items():
            subset = sorted([r for r in rows if r["method"] == method and r["family"] == family], key=lambda r: r["level"])
            ax.plot([r["level"] for r in subset], [100*r["accepted_fraction"] for r in subset],
                    ".-", ms=4, lw=1, label=label)
        ax.set(title=family, xlabel="Target image index (not nominal strain)",
               ylabel="Solver-accepted / reference-mask windows (%)", ylim=(-3, 103))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Independent reference-to-target pairs; displacement truth not associated", fontsize=11)
    save(fig, figures, "acceptance_curves")
    # Fixed diagnostic levels, including failures and the last supplied image.
    wanted = {"FEA": [1, 25, 191], "ClosedFormExpression": [25, 50, 500]}
    sources = []
    for row in rows:
        if row["level"] not in wanted[row["family"]] or row["method"] not in ["learned", "silhouette_32"]:
            continue
        family, level, subset = row["family"], row["level"], row["subset"]
        base = Path("/home/jeff/CNRS/DIC/HudgeStrains")/family
        filename = f"Img_wa_g{level:03d}.tif" if family == "FEA" else f"Img_G{level:03d}.tif"
        reference = np.asarray(Image.open(base/"Img_F.tif"), dtype=np.float32)/255
        target = np.asarray(Image.open(base/filename), dtype=np.float32)/255
        with np.load(row["archive"]) as z:
            centres, parameters, accepted, valid = (z[k] for k in ["centres", "parameters", "accepted", "valid_roi"])
            accepted = accepted.astype(bool)
            correlation, _, pullback = photometric_check(reference, target, centres, parameters, accepted, subset)
        mask = material_mask(np.rint(reference*255).astype(np.uint8))
        yy, xx = np.nonzero(mask)
        xmax, ymax = min(reference.shape[1], xx.max()+10), min(reference.shape[0], yy.max()+10)
        reference_crop, pullback_crop = reference[:ymax, :xmax], pullback[:ymax, :xmax]
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
        axes[0, 0].imshow(reference_crop, cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("(a) Reference material (crop)")
        axes[0, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title("(b) Native target image")
        axes[0, 2].imshow(reference_crop, cmap="gray", vmin=0, vmax=1)
        axes[0, 2].scatter(*centres[valid & ~accepted].T, s=14, marker="x", c="#d95f02", label="Rejected")
        axes[0, 2].scatter(*centres[accepted].T, s=10, c="#1b9e77", label="Accepted")
        axes[0, 2].set(xlim=(0,xmax), ylim=(ymax,0), title="(c) Numerical status\nDots accepted; crosses rejected")
        for ax, component, label in [(axes[1, 0], 0, "(d) Estimated u (pixels)"), (axes[1, 1], 3, "(e) Estimated v (pixels)")]:
            ax.imshow(reference_crop, cmap="gray", vmin=0, vmax=1, alpha=.3)
            if accepted.any():
                plot = ax.scatter(*centres[accepted].T, c=parameters[accepted, component], s=18, cmap="viridis")
                fig.colorbar(plot, ax=ax, shrink=.8)
            ax.set(xlim=(0,xmax), ylim=(ymax,0), title=label)
        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad("#e3a1a1")
        axes[1, 2].imshow(pullback_crop, cmap=cmap, vmin=0, vmax=1)
        axes[1, 2].set_title("(f) Affine pullbacks\nPink: no accepted support")
        for ax in axes.flat:
            ax.set_xlabel("x (pixels)")
            ax.set_ylabel("y (pixels)")
        fig.suptitle(f"{family}, reference → {level:03d}; {labels[row['method']]}\n"
                     f"{row['accepted']}/{row['valid_roi']} accepted; no displacement-accuracy claim", fontsize=11)
        name = f"{row['method']}_{family}_{level:03d}"
        save(fig, figures, name)
        usable = np.flatnonzero(accepted & np.isfinite(correlation))
        if len(usable):
            # Deliberately show the least photometrically similar accepted window,
            # not the most attractive match. This is a visualization rule only.
            chosen = usable[np.argmin(correlation[usable])]
            x, y = centres[chosen]
            u, ux, uy, v, vx, vy = parameters[chosen]
            dx, dy = np.meshgrid(np.arange(subset)-(subset-1)/2, np.arange(subset)-(subset-1)/2)
            patch0 = map_coordinates(reference, [y+dy,x+dx], order=1)
            translated = map_coordinates(target, [y+v+dy,x+u+dx], order=1)
            corrected = map_coordinates(target, [y+v+vx*dx+(1+vy)*dy,x+u+(1+ux)*dx+uy*dy], order=1)
            fig, axes = plt.subplots(1,4,figsize=(10,2.8),constrained_layout=True)
            for ax, array, title in zip(axes[:3], [patch0,translated,corrected],
                                        ["Reference", "Centre translation only", "Final affine pullback"],strict=True):
                ax.imshow(array,cmap="gray",vmin=0,vmax=1)
                ax.set(title=title,xlabel="Patch x (pixels)",ylabel="Patch y (pixels)")
            picture = axes[3].imshow(np.abs(patch0-corrected),cmap="magma",vmin=0,vmax=.25)
            axes[3].set(title="Absolute intensity residual",xlabel="Patch x (pixels)",ylabel="Patch y (pixels)")
            fig.colorbar(picture,ax=axes[3],shrink=.8)
            fig.suptitle(f"{family} {level:03d}, {labels[row['method']]}: least-correlated accepted support\n"
                         f"Centre ({x:.1f}, {y:.1f}) px; final correlation {correlation[chosen]:.3f}; not a displacement-error test",fontsize=10)
            save(fig,figures,"detail_"+name)
        # Same physical reference crop in both frames. No interpolation across rejects.
        frames = []
        for array, title in [(reference_crop, "Reference"), (pullback_crop, "Corrected accepted supports; pink = unavailable")]:
            rgb = np.repeat(np.rint(np.nan_to_num(array)*255).astype(np.uint8)[..., None], 3, axis=2)
            rgb[~np.isfinite(array)] = [227,161,161]
            image = Image.fromarray(rgb)
            image.thumbnail((1000,600))
            canvas = Image.new("RGB", (image.width, image.height+30), "white")
            canvas.paste(image, (0,30))
            ImageDraw.Draw(canvas).text((8,8),title,fill="black")
            frames.append(canvas)
        frames[0].save(figures/f"{name}.gif", save_all=True, append_images=frames[1:], duration=900, loop=0)
        sources.append({"figure": name, "archive": row["archive"], "reference": str(base/"Img_F.tif"),
                        "target": str(base/filename), "crop": [0,0,int(xmax),int(ymax)],
                        "accepted_final_zncc": correlation[accepted].tolist()})
    (figures/"sources.json").write_text(json.dumps(sources,indent=2)+"\n")


if __name__ == "__main__":
    main()

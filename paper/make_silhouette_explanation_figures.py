"""Explanatory figures from the tested plate images and archived solver states."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/benchmarks"))
from hudge_strains_data import material_mask

OUT = ROOT / "results/hudge_strains_20261114/figures"
BLUE, ORANGE, GREEN = "#2469a0", "#d47300", "#087e64"


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def image_axes(ax, array, title):
    ax.imshow(array, cmap="gray", vmin=0, vmax=255)
    ax.set(title=title, xlabel="x (native pixels)", ylabel="y (native pixels)")


def main():
    source = Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA")
    ref_path, target_path = source / "Img_F.tif", source / "Img_wa_g191.tif"
    reference, target = [np.asarray(Image.open(p)) for p in [ref_path, target_path]]
    record = OUT.parent / "silhouette_32/FEA/level_191.json"
    archive = record.with_suffix(".npz")
    meta = json.loads(record.read_text())
    with np.load(archive) as data:
        points, states, accepted, seeds, generation, parent, valid = [data[k] for k in
            ["centres", "parameters", "accepted", "confirmed_seeds", "generation", "parent", "valid_roi"]]
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    mask = material_mask(reference)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), constrained_layout=True)
    image_axes(axes[0], reference[:225, :440], "(a) Reference speckle image")
    image_axes(axes[1], mask[:225, :440]*255, "(b) Image-derived material mask")
    image_axes(axes[2], reference[:225, :440], "(c) Eligible 32-pixel supports")
    for x, y in points[valid]:
        axes[2].add_patch(Rectangle((x-16, y-16), 32, 32, fill=False, lw=.7, edgecolor=GREEN))
    for ax in axes:
        ax.set(xlim=(-.5,439.5), ylim=(224.5,-.5))
    fig.suptitle("Background components are removed; holes are not matched as landmarks", fontsize=11)
    save(fig, "silhouette_mask_steps")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), constrained_layout=True)
    for ax, box, title, ymax in [(axes[0],meta["reference_bounds"],"Reference box",240),
                                 (axes[1],meta["target_bounds"],"Target box",1080)]:
        lx, ly, hx, hy = box
        ax.add_patch(Rectangle((lx,ly),hx-lx,hy-ly,facecolor="#eef3f6",edgecolor=BLUE,lw=2))
        ax.scatter([lx,hx],[ly,hy],c=BLUE,s=25)
        ax.annotate(f"lower edge = {ly:g}",(lx,ly),xytext=(12,10),textcoords="offset points",fontsize=9)
        ax.annotate(f"upper edge = {hy:g}",(hx,hy),xytext=(-5,-16),ha="right",textcoords="offset points",fontsize=9)
        ax.annotate("",xy=(hx+30,ly),xytext=(hx+30,hy),arrowprops={"arrowstyle":"<->","color":ORANGE})
        ax.text(hx+45,(hy+ly)/2,f"{hy-ly:g} px",color=ORANGE,rotation=90,va="center")
        ax.set(xlim=(-20,540),ylim=(ymax,-25),title=title,xlabel="x (pixels)",ylabel="y (pixels)")
    fig.suptitle(r"$s_y=1015/211=4.81043,\quad t_y=3.5-s_y\,3.5=-13.33649$ px"+"\n"
                 r"$y'=3.5+s_y(y-3.5)$; separate vertical scales for legibility",fontsize=11)
    save(fig,"silhouette_box_geometry")

    # Deterministic illustration: accepted centre closest to the ROI centroid.
    ids = np.flatnonzero(accepted)
    chosen = ids[np.argmin(np.linalg.norm(points[ids]-points[valid].mean(0),axis=1))]
    centre, p = points[chosen], states[chosen]
    offsets = np.array([[-16,-16],[16,-16],[16,16],[-16,16]],float)
    scale = np.diag(meta["extent_ratio_xy"])
    translation = np.array(meta["translation_xy"])
    box_centre = scale @ centre + translation
    final_centre = centre+p[[0,3]]
    gradient = np.array([[p[1],p[2]],[p[4],p[5]]])
    box_polygon = (centre+offsets) @ scale.T + translation
    final_polygon = final_centre + offsets @ (np.eye(2)+gradient).T
    fig, axes = plt.subplots(1,2,figsize=(9,4.2),constrained_layout=True)
    image_axes(axes[0],reference,"(a) Reference window")
    axes[0].add_patch(Polygon(centre+offsets,fill=False,edgecolor=BLUE,lw=2))
    axes[0].scatter(*centre,c=BLUE,s=25)
    axes[0].set(xlim=(centre[0]-35,centre[0]+35),ylim=(centre[1]+35,centre[1]-35))
    image_axes(axes[1],target,"(b) Same support in target coordinates")
    for polygon, xy, color, label in [(box_polygon,box_centre,ORANGE,"Box proposal"),
                                      (final_polygon,final_centre,GREEN,"Solved local affine")]:
        axes[1].add_patch(Polygon(polygon,fill=False,edgecolor=color,lw=2,label=label))
        axes[1].scatter(*xy,c=color,s=25)
    combined=np.vstack([box_polygon,final_polygon]); lo=combined.min(0)-12;hi=combined.max(0)+12
    axes[1].set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]))
    axes[1].legend(loc="upper left",bbox_to_anchor=(1.01,1),fontsize=9)
    fig.suptitle(r"$q_i(\xi)=x_i+d_i+(I+G_i)\xi$"+f"\nSaved accepted centre ({centre[0]:.1f}, {centre[1]:.1f}) px; no truth used",fontsize=11)
    save(fig,"silhouette_local_support")

    fig, axes=plt.subplots(1,3,figsize=(10,3.5),constrained_layout=True)
    for ax,title in zip(axes,["(a) Confirmed seeds","(b) Accepted by generation 2","(c) Final states and parent links"],strict=True):
        image_axes(ax,reference[:225,:440],title)
        ax.scatter(*points[valid & ~accepted].T,marker="x",c=ORANGE,s=22)
    axes[0].scatter(*points[seeds].T,c=GREEN,s=25)
    early=accepted & (generation<=2)
    axes[1].scatter(*points[early].T,c=GREEN,s=25)
    for child in np.flatnonzero(accepted & (parent>=0)):
        origin=parent[child]
        assert accepted[origin] and generation[origin]<generation[child]
        axes[2].annotate("",xy=points[child],xytext=points[origin],arrowprops={"arrowstyle":"->","color":BLUE,"lw":.8})
    plotted=axes[2].scatter(*points[accepted].T,c=generation[accepted],cmap="viridis",vmin=0,vmax=max(generation),s=22)
    fig.colorbar(plotted,ax=axes[2],label="Acceptance generation",shrink=.65)
    for ax in axes:
        ax.set(xlim=(-.5,439.5),ylim=(224.5,-.5))
    fig.suptitle(f"Plate 191: {seeds.sum()} seeds, {early.sum()} accepted by generation 2, {accepted.sum()}/{valid.sum()} final\n"
                 "Orange crosses: final rejects; arrows: archived parent links, not displacement vectors",fontsize=11)
    save(fig,"silhouette_propagation_steps")
    manifest={"inputs":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ref_path,target_path,record,archive]},
              "selected_window_index":int(chosen),"selected_window_parameters":states[chosen].tolist(),
              "accepted_by_generation_2":int(early.sum()),"truth_used":False,
              "interpretation":"Mask/box geometry and saved solver states; propagation links are not displacement vectors"}
    (OUT/"silhouette_explanation_sources.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    main()

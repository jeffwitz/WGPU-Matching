"""Side-specific endpoint diagnostics, with optional image-sequence tracking controls.

Intermediate frames only enter the offline controls. The input two-image field
and its archived solver statuses remain untouched.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from hudge_strains_data import material_mask, profiles
from PIL import Image
from scipy.ndimage import binary_fill_holes, map_coordinates
from scipy.spatial import cKDTree


def sample(a, p):
    return map_coordinates(
        a.astype(np.float32), [p[..., 1], p[..., 0]], order=1, prefilter=False, mode="nearest"
    )


def ncc(a, b):
    a = a.astype(float) - a.mean()
    b = b.astype(float) - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def track(folder, reference, p, stride):
    prev = reference
    xy = p.astype("f").reshape(-1, 1, 2)
    valid = np.ones(len(p), bool)
    opts = dict(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
    )
    for i in [*range(1, 191, stride), 191]:
        target = np.array(Image.open(folder / f"Img_wa_g{i:03}.tif"))
        tgt, st, _ = cv2.calcOpticalFlowPyrLK(prev, target, xy, None, **opts)
        back, rev, _ = cv2.calcOpticalFlowPyrLK(target, prev, tgt, None, **opts)
        valid &= (
            st.ravel().astype(bool)
            & rev.ravel().astype(bool)
            & (np.linalg.norm((back - xy)[:, 0], axis=1) < 0.75)
        )
        xy = tgt
        prev = target
    return xy[:, 0], valid


def main(root, folder, out):
    cv2.setNumThreads(4)
    out.mkdir(exist_ok=True, parents=True)
    ref = np.asarray(Image.open(folder / "Img_F.tif"))
    target = np.asarray(Image.open(folder / "Img_wa_g191.tif"))
    eligible = np.load(root / "prior/seeds.npz")["accepted"]
    z = np.load(root / "comparison/narrow_seeds.npz")
    field = np.load(root / "boundary_narrow/map.npz")["coordinates"]
    prior = np.load(root / "prior/map.npz")["coordinates"]
    h, w = field.shape[:2]
    y, x = np.indices((h, w))
    mask = material_mask(ref)[:h, :w]
    before = sample(target, prior)
    after = sample(target, field)
    xy = z["centres"][eligible]
    curves = []
    for axis, profile in profiles(binary_fill_holes(material_mask(ref))):
        t = np.flatnonzero(np.isfinite(profile))
        pts = np.empty((len(t), 2))
        pts[:, 1 - axis], pts[:, axis] = t, profile[t]
        curves.append(cKDTree(pts))

    def sides(p):
        distances = np.stack([tree.query(p)[0] for tree in curves])
        return distances.argmin(0), distances.min(0)

    index, d = sides(xy)
    pix = np.c_[x[mask], y[mask]]
    ip, dp = sides(pix)
    band = dp <= 8
    points = pix[band]
    # Track all boundary-band material pixels and all eligible window centres.
    probes = np.vstack([points, xy])
    results = []
    for stride in [2, 5]:
        file = out / f"tracks_{stride}.npz"
        if not file.exists():
            pos, ok = track(folder, ref, probes, stride)
            np.savez_compressed(file, positions=pos, valid=ok, probes=probes)
        data = np.load(file)
        np.testing.assert_array_equal(probes, data["probes"])
        results.append((data["positions"], data["valid"]))
    p2, v2 = results[0]
    p5, v5 = results[1]
    agreed = v2 & v5 & (np.linalg.norm(p2 - p5, axis=-1) < 1)
    n = len(points)
    target_mask = material_mask(target)
    inside_fraction = sample(target_mask.astype(float), field)
    # Count a pixel as mapped outside if most of its target interpolation
    # footprint is background. Also count footprints containing only background.
    from PIL import ImageDraw

    greyscale = np.clip(after[:225, :440], 0, 255).astype(np.uint8)
    overlay = np.array(Image.fromarray(greyscale).convert("RGB"))
    scores = []
    for side, name in enumerate(["left", "right", "top", "bottom"]):
        use = (index == side) & (d <= 8)
        pp = ip[band] == side
        indices = points[pp]
        ii = indices[:, 1], indices[:, 0]
        for xx, yy in indices:
            if inside_fraction[yy, xx] < 0.5:
                overlay[yy, xx] = [241, 65, 54]
        common = pp & agreed[:n]
        jj = points[common, 1], points[common, 0]
        u = use & agreed[n:]
        old = z["previously_accepted"][eligible]
        added = z["accepted"][eligible] & ~old
        native = xy + z["parameters"][eligible][:, [0, 3]]
        err = np.linalg.norm(native - p2[n:], axis=-1)
        row = dict(
            side=name,
            eligible=int(use.sum()),
            old=int((use & old).sum()),
            accepted=int((use & z["accepted"][eligible]).sum()),
            added_with_tracks=int((u & added).sum()),
            added_within_2=int((u & added & (err < 2)).sum()),
            band_pixels=int(pp.sum()),
            mapped_background=int((inside_fraction[ii] < 0.5).sum()),
            full_background_footprint=int((inside_fraction[ii] < 1e-6).sum()),
            photometry_prior=ncc(ref[ii], before[ii]),
            photometry_final=ncc(ref[ii], after[ii]),
            common_pixels=int(common.sum()),
        )
        if common.sum() > 2:
            row.update(
                common_prior=ncc(ref[jj], before[jj]),
                common_final=ncc(ref[jj], after[jj]),
                common_tracks=ncc(ref[jj], sample(target, p2[:n][common])),
                endpoint_vs_tracks_median=float(
                    np.median(np.linalg.norm(field[jj] - p2[:n][common], axis=-1))
                ),
            )
        # Evaluate the six-parameter window states separately from the dense map.
        dy, dx = np.mgrid[:12, :12] - 5.5
        offsets = np.stack((dx, dy), -1)
        fractions = []
        for j in np.flatnonzero(use & z["accepted"][eligible]):
            source = xy[j] + offsets
            active = sample(mask, source) > 0.5
            coeff = z["parameters"][eligible][j]
            linear = np.eye(2) + coeff[[1, 2, 4, 5]].reshape(2, 2)
            mapped = xy[j] + coeff[[0, 3]] + offsets @ linear.T
            inside = sample(target_mask, mapped)
            fractions.append(float((inside[active] < 0.5).mean()))
        row["accepted_states_over_quarter_background"] = int((np.array(fractions) > 0.25).sum())
        row["accepted_states_median_background_fraction"] = float(np.median(fractions))
        scores.append(row)
    im = Image.new("RGB", (880, 490), "white")
    im.paste(Image.fromarray(overlay).resize((880, 450), Image.Resampling.NEAREST), (0, 0))
    ImageDraw.Draw(im).text(
        (10, 460),
        "Rouge : pixel de matiere a moins de 8 px du bord, remappe majoritairement sur le fond cible.",
        fill="black",
    )
    im.save(out / "matiere_vers_fond.png")
    (out / "sides.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/hudge_dense_flow_20261118"))
    parser.add_argument("--folder", type=Path, default=Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.folder, args.output)

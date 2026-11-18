"""Render recorded affine retries, including proposals rejected by propagation."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from hudge_affine_restart import rectify
from PIL import Image
from scipy.ndimage import map_coordinates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("results/hudge_remap_diagnostic_20261115")
DATA = Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA")
BASE = Path("results/hudge_strains_20261114/silhouette_32/FEA")
TRACE = ROOT / "recorded/rectified_native"
FIG = ROOT / "figures"
IMAGES = ROOT / "images"


def sample(image, centre, parameters):
    y, x = np.mgrid[:32, :32].astype(float) - 15.5
    p = parameters
    qx = centre[0] + p[0] + (1 + p[1]) * x + p[2] * y
    qy = centre[1] + p[3] + p[4] * x + (1 + p[5]) * y
    return map_coordinates(image.astype(float), [qy, qx], order=1,
                           mode="constant", cval=np.nan, prefilter=False)


def correlation(a, b):
    if not np.isfinite(b).all():
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    return float(np.sum(a * b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    reference = np.asarray(Image.open(DATA / "Img_F.tif"))
    report = []
    for level in (50, 150, 191):
        target = np.asarray(Image.open(DATA / f"Img_wa_g{level:03d}.tif"))
        base = np.load(BASE / f"level_{level:03d}.npz")
        final = np.load(TRACE / f"level_{level:03d}.npz")
        archive = np.load(Path("results/hudge_affine_restart_20261115/all_levels/")
                          / f"rectified_native/level_{level:03d}.npz")
        assert np.array_equal(final["accepted"], archive["accepted"])
        assert np.array_equal(final["parameters"], archive["parameters"])
        centres, valid = final["centres"], final["valid_roi"]
        folder = TRACE / f"trace_{level:03d}"
        rounds = [dict(np.load(p)) for p in sorted(folder.glob("round_*_before.npz"))]
        after = dict(np.load(max(folder.glob("round_*_after.npz"))))
        # The silhouette initial field is globally affine; recover its intercept.
        p = base["initial_parameters"][np.flatnonzero(valid)[0]]
        matrix = np.eye(2) + p[[1, 2, 4, 5]].reshape(2, 2)
        centre = centres[np.flatnonzero(valid)[0]]
        shift = p[[0, 3]] - (matrix - np.eye(2)) @ centre
        stages = [("Référence", reference),
                  ("Proposition silhouette", rectify(target, matrix, shift))]
        stages += [(f"Affine ajustée — tour {i+1}", rectify(target, r["matrix"], r["translation"]))
                   for i, r in enumerate(rounds)]
        fig, axes = plt.subplots(2, len(stages), figsize=(4.5 * len(stages), 5.8))
        for col, (label, image) in enumerate(stages):
            Image.fromarray(image).save(IMAGES / f"{level:03d}_{col:02d}.png")
            axes[0, col].imshow(image[:225, :440], cmap="gray", vmin=0, vmax=255)
            axes[0, col].set_title(label)
            axes[1, col].imshow(np.abs(image[:225, :440].astype(float)
                                      - reference[:225, :440]), cmap="magma", vmin=0, vmax=128)
            axes[1, col].set_title("Écart absolu à la référence (0–128 / 255)")
            for ax in axes[:, col]:
                ax.set_xlim(0, 440)
                ax.set_ylim(225, 0)
                ax.set_xlabel("x référence [px]")
        axes[0, 0].set_ylabel("y référence [px]")
        fig.suptitle(f"Niveau {level} — chaque remappage repart de la cible originale")
        fig.tight_layout()
        save(fig, f"sequence_{level}")

        events = {int(i): [] for i in np.flatnonzero(valid)}
        # Wrapper outputs are already composed and checked on native pixels.
        paths = sorted(folder.glob("wrapper_*.npz"))
        paths += [p for p in sorted(folder.glob("call_*.npz"))
                  if str(np.load(p)["phase"]) == "merged_propagation"]
        for path in paths:
            d = np.load(path)
            for j, index in enumerate(d["indices"]):
                if int(index) not in events:
                    continue
                inp, out = d["input_parameters"][j], d["output_parameters"][j]
                events[int(index)].append({
                    "file": str(path), "phase": str(d["phase"]),
                    "iteration": int(d["iteration"]),
                    "input": inp.tolist(), "output": out.tolist(),
                    "accepted_by_solver": bool(d["raw_status"][j] == 1),
                    "du": float(np.linalg.norm((out-inp)[[0, 3]])),
                    "dg": float(np.linalg.norm((out-inp)[[1, 2, 4, 5]])),
                })
        rows = []
        for index, attempts in events.items():
            attempts.sort(key=lambda e: (e["iteration"], int(Path(e["file"]).stem.split("_")[-1])))
            rows.append({"index": index, "centre": centres[index].tolist(),
                         "baseline": bool(base["accepted"][index]),
                         "final": bool(final["accepted"][index]),
                         "solver_accepted_attempts": sum(e["accepted_by_solver"] for e in attempts),
                         "last_merged_reason": int(after["merged_rejection"][index]),
                         "last_restart_reason": int(after["restart_rejection"][index]),
                         "events": attempts})
        (ROOT / f"windows_{level}.json").write_text(json.dumps(rows, indent=2))
        missing = valid & ~final["accepted"]
        counts = {str(i): int(np.sum(missing & (after["merged_rejection"] == i)))
                  for i in range(4)}
        report.append({"level": level, "baseline": int(base["accepted"].sum()),
                       "final": int(final["accepted"].sum()), "exact_archive_parity": True,
                       "unresolved": int(missing.sum()), "last_merged_reasons": counts,
                       "unresolved_with_solver_success": sum(not r["final"] and
                                                             r["solver_accepted_attempts"] > 0
                                                             for r in rows)})
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(reference[:225, :440], cmap="gray", vmin=0, vmax=255)
        for row in rows:
            x, y = row["centre"]
            color = "#32c5ff" if row["baseline"] else ("#41ff50" if row["final"] else "#ff672e")
            ax.add_patch(Rectangle((x-16, y-16), 32, 32, fill=False, edgecolor=color, linewidth=1.6))
            ax.text(x, y, str(row["index"]), color=color, ha="center", va="center", fontsize=8,
                    bbox={"facecolor": "black", "alpha": .65, "pad": .5})
        ax.set_title(f"Niveau {level} : bleu = déjà retenu ; vert = récupéré ; orange = non retenu")
        ax.set_xlabel("x référence [px]")
        ax.set_ylabel("y référence [px]")
        save(fig, f"map_{level}")

        # Deterministic spatial examples: recovery if available, then unresolved
        # with/without successful native solve, and a retained control.
        chosen = []
        groups = [[r for r in rows if r["final"] and not r["baseline"]],
                  [r for r in rows if not r["final"] and r["solver_accepted_attempts"]],
                  [r for r in rows if not r["final"] and not r["solver_accepted_attempts"]],
                  [r for r in rows if r["baseline"]]]
        for group in groups:
            if group:
                chosen.append(min(group, key=lambda r: np.linalg.norm(np.array(r["centre"])-[175, 110])))
        details = []
        for row in chosen:
            index = row["index"]
            c = centres[index]
            refpatch = sample(reference, c, np.zeros(6))
            attempts = row["events"]
            # Last recorded native propagation attempt is displayed, not an
            # analytically selected best result. Some windows were never tried.
            event = attempts[-1] if attempts else None
            states = [("Silhouette", base["initial_parameters"][index]),
                      ("Affine globale, tour 1", rounds[0]["proposal"][index])]
            if event:
                states += [("Entrée du dernier essai", np.array(event["input"])),
                           ("Sortie du dernier essai", np.array(event["output"]))]
            if row["final"]:
                states.append(("État finalement retenu", final["parameters"][index]))
            fig, axes = plt.subplots(2, len(states)+1, figsize=(3*(len(states)+1), 5.9))
            axes[0, 0].imshow(refpatch, cmap="gray", vmin=0, vmax=255)
            axes[0, 0].set_title("Mouchetis de référence")
            axes[1, 0].axis("off")
            values = []
            for col, (label, state) in enumerate(states, 1):
                patch = sample(target, c, state)
                score = correlation(refpatch, patch)
                values.append({"stage": label, "correlation": score})
                axes[0, col].imshow(patch, cmap="gray", vmin=0, vmax=255)
                axes[0, col].set_title(label + f"\ncorrélation = {score:.3f}")
                axes[1, col].imshow(np.abs(patch-refpatch), cmap="magma", vmin=0, vmax=128)
                axes[1, col].set_title("Résidu absolu (0–128 / 255)")
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            subtitle = "retenue" if row["final"] else "NON retenue"
            if event:
                subtitle += (f" ; dernier essai : statut accepté={event['accepted_by_solver']}"
                             f" ; correction d={event['du']:.2f} px, G={event['dg']:.3f}")
            fig.suptitle(f"Niveau {level} — fenêtre {index}, centre ({c[0]}, {c[1]}) — {subtitle}")
            fig.tight_layout()
            save(fig, f"patch_{level}_{index}")
            details.append({"index": index, "final": row["final"], "values": values,
                            "last_event": event, "merged_reason": row["last_merged_reason"]})
        report[-1]["details"] = details
    (ROOT / "diagnostic.json").write_text(json.dumps(report, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != "details"} for r in report], indent=2))


if __name__ == "__main__":
    main()

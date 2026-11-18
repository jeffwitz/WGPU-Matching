"""Report the frozen direct-pair restart experiment from saved native states."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from summarize_hudge_strains import photometric_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("results/hudge_affine_restart_20261115"))
    root = parser.parse_args().root
    rows = json.loads((root/"all_levels/summary.json").read_text())
    assert len(rows)==573, "the full 191-pair, three-variant campaign must finish"
    baseline = Path("results/hudge_strains_20261114/silhouette_32/FEA")
    figures = root/"figures";figures.mkdir(exist_ok=True)
    labels={"native_retry":"Affine native", "native_two_pass":"Deux passes natives", "rectified_native":"Rectification + validation native"}
    metrics=[]
    for method in labels:
        group=sorted([r for r in rows if r["method"]==method],key=lambda r:r["level"])
        eligible=lambda threshold, group=group:[r["level"] for r in group if r["final"]/r["valid"]>=threshold]
        metrics.append({"method":method,"baseline_total":sum(r["baseline"] for r in group),
                        "final_total":sum(r["final"] for r in group),"improved_pairs":sum(r["final"]>r["baseline"] for r in group),
                        "full_coverage_max_level":max(eligible(1),default=None),"coverage90_max_level":max(eligible(.9),default=None),
                        "median_restart_s":float(np.median([r["restart_wall_s"] for r in group])),
                        "bad_photometry":sum(r["accepted_below_065"] for r in group),
                        "out_of_bounds":sum(r["accepted_out_of_bounds"] for r in group)})
    agreement=[]
    for level in range(1,192):
        with np.load(root/f"all_levels/native_retry/level_{level:03d}.npz") as z:a=z["accepted"]
        with np.load(root/f"all_levels/rectified_native/level_{level:03d}.npz") as z:b=z["accepted"]
        agreement.append({"level":level,"only_native":int((a & ~b).sum()),"only_rectified":int((b & ~a).sum())})
    (root/"comparison.json").write_text(json.dumps({"metrics":metrics,"paired_masks":agreement},indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,ax=plt.subplots(figsize=(9,4.4),constrained_layout=True)
    group=sorted([r for r in rows if r["method"]=="native_retry"],key=lambda r:r["level"])
    ax.plot([r["level"] for r in group],[r["baseline"] for r in group],label="Baseline silhouette",c="black",lw=1.5)
    for method,label in labels.items():
        group=sorted([r for r in rows if r["method"]==method],key=lambda r:r["level"])
        ax.plot([r["level"] for r in group],[r["final"] for r in group],label=label,lw=1,alpha=.8)
    ax.set(xlabel="Indice cible (sans séquence temporelle)",ylabel="Fenêtres acceptées / 36",ylim=(0,38),
           title="Plaque trouée : reprises affines, confirmation et propagation")
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    for ext in ["png","pdf","svg"]:fig.savefig(figures/f"coverage.{ext}",dpi=220)
    plt.close(fig)
    data=Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA")
    reference=np.asarray(Image.open(data/"Img_F.tif"),dtype=float)/255
    for level in [50,100,150,191]:
        target=np.asarray(Image.open(data/f"Img_wa_g{level:03d}.tif"),dtype=float)/255
        with np.load(baseline/f"level_{level:03d}.npz") as z:
            points,old,valid,p0=[z[k] for k in ["centres","accepted","valid_roi","parameters"]]
        with np.load(root/f"all_levels/rectified_native/level_{level:03d}.npz") as z:
            new,p1=z["accepted"],z["parameters"]
        _,_,before=photometric_check(reference,target,points,p0,old,32)
        _,_,after=photometric_check(reference,target,points,p1,new,32)
        fig,axes=plt.subplots(2,2,figsize=(9,5),constrained_layout=True)
        cmap=plt.get_cmap("gray").copy();cmap.set_bad("#e3a1a1")
        axes[0,0].imshow(reference[:225,:440],cmap=cmap,vmin=0,vmax=1)
        axes[0,0].set_title("Référence")
        axes[0,1].imshow(reference[:225,:440],cmap=cmap,vmin=0,vmax=1)
        for mask,color,marker,label in [(old,"#2469a0","o","Conservées"),(new & ~old,"#087e64","o","Nouvelles"),(valid & ~new,"#d47300","x","Non résolues")]:
            axes[0,1].scatter(*points[mask].T,c=color,marker=marker,s=20,label=label)
        axes[0,1].legend(fontsize=7,loc="upper left",bbox_to_anchor=(1,1))
        axes[0,1].set(xlim=(-.5,439.5),ylim=(224.5,-.5),title="Statuts sur les mêmes centres")
        axes[1,0].imshow(before[:225,:440],cmap=cmap,vmin=0,vmax=1)
        axes[1,0].set_title(f"Avant : {old.sum()}/36")
        axes[1,1].imshow(after[:225,:440],cmap=cmap,vmin=0,vmax=1)
        axes[1,1].set_title(f"Après : {new.sum()}/36")
        for ax in axes.ravel():ax.set(xlabel="x (pixels)",ylabel="y (pixels)")
        fig.suptitle(f"Plaque {level}: états natifs acceptés, exactitude non certifiée")
        for ext in ["png","pdf","svg"]:fig.savefig(figures/f"plate_{level:03d}.{ext}",dpi=220,bbox_inches="tight")
        plt.close(fig)
        frames=[]
        for title,array in [("Reference",reference),("Corrected after affine restarts",after)]:
            crop=array[:225,:440];rgb=np.repeat(np.rint(np.nan_to_num(crop)*255).astype(np.uint8)[...,None],3,axis=2)
            rgb[~np.isfinite(crop)]=[227,161,161]
            canvas=Image.new("RGB",(440,255),"white");canvas.paste(Image.fromarray(rgb),(0,30));ImageDraw.Draw(canvas).text((5,5),title,fill="black");frames.append(canvas)
        frames[0].save(figures/f"plate_{level:03d}.gif",save_all=True,append_images=frames[1:],duration=900,loop=0)
    text="""# Reprise affine itérative de la plaque trouée

Chaque cible est comparée directement à Img_F, sans état d'une autre cible. Les 191 niveaux sont évalués avec les mêmes 36 fenêtres de support/pas 32 pixels. Les états acceptés du contrôle silhouette du 14 novembre sont le départ commun et restent inchangés. Ce sont des acceptations numériques, pas des déplacements certifiés : la vérité associée manque toujours.

## Méthode exécutée

À chaque tour, on ajuste y=A x+t aux correspondances centrales déjà acceptées. L'ajustement est robuste : coordonnées centrées/réduites, dix résolutions de moindres carrés pondérés, poids de Huber min(1,c/r) avec c=max(1 pixel,2,5 fois le résidu médian). Les gradients locaux ne sont pas utilisés dans cet ajustement. Moins de trois centres non colinéaires, un déterminant non positif ou un conditionnement supérieur à 100 arrêtent la reprise.

Trois variantes sont comparées : affine réestimée et résolution native; deux résolutions natives successives (contrôle du travail supplémentaire); rectification puis résolution et validation native. La dernière construit J(z)=I₁(Az+t), par interpolation bilinéaire et arrondi uint8, à partir de la cible originale à chaque tour. Aucune image déjà rectifiée n'est rééchantillonnée à nouveau.

Pour une solution résiduelle ψ(x)=x+r(x), le retour natif est φ(x)=Aψ(x)+t. Le déplacement vaut u=A(x+r)+t−x et le gradient de transformation vaut F=A(I+∇r). Ces règles sont testées sur les positions et les dérivées, notamment par différences finies. Les états sont ensuite résolus de nouveau sur les pixels natifs; une acceptation dans l'image rectifiée seule ne suffit pas.

La sélection éparse, confirmation et propagation existantes utilisent toujours les coordonnées natives et les seuils 8 pixels / 0,25 en gradient. Solveur de Gauss–Newton à composition inverse (IC-GN) : affine, interpolation bilinéaire, 30 itérations par appel, tolérance 10⁻³, corrélation normalisée à moyenne nulle (ZNCC) minimale 0,65. Les anciennes fenêtres sont conservées; les nouvelles sont réunies puis la propagation native est relancée. Arrêt après un tour sans nouvelle fenêtre, ou au maximum cinq tours. Aucun seuil n'est ajusté par niveau.

L'ajustement et la rectification sont sur le processeur central; les résolutions utilisent WebGPU. Ce prototype n'est pas une chaîne entièrement résidente sur le processeur graphique.

## Résultats

| Variante | Total avant | Total après | Paires améliorées /191 | Dernier indice à 100 % | Dernier indice ≥90 % | Surcoût médian (s) |
|---|---:|---:|---:|---:|---:|---:|
"""
    for m in metrics:text+=f"| {labels[m['method']]} | {m['baseline_total']} | {m['final_total']} | {m['improved_pairs']} | {m['full_coverage_max_level']} | {m['coverage90_max_level']} | {m['median_restart_s']:.3f} |\n"
    text+="\nLes totaux additionnent des cibles de la même texture : ce ne sont pas des observations indépendantes. Les maxima ne garantissent pas un succès monotone aux niveaux inférieurs. Les temps sont des surcoûts de reprise, hors initialisation historique et préparation de référence.\n\n![Couverture](figures/coverage.png)\n"
    only_native=sum(r['only_native'] for r in agreement);only_rect=sum(r['only_rectified'] for r in agreement)
    text+=f"\nComparaison fenêtre par fenêtre : {only_native} acceptations exclusives à la reprise native et {only_rect} exclusives à la rectification. Les comptes identiques ne sont donc pas supposés impliquer des états identiques.\n"
    text+=f"\nLa rectification récupère {metrics[2]['final_total']-metrics[2]['baseline_total']} acceptations de plus que le départ; la reprise native en récupère déjà {metrics[0]['final_total']-metrics[0]['baseline_total']}. Sur ce jeu, conserver la reprise affine native comme premier essai est donc justifié par le coût; la rectification reste une reprise supplémentaire, sans promesse de couverture complète. Les comparaisons portent sur une seule série de texture, pas une validation générale.\n"
    text+=f"\nContrôles finaux : {sum(m['bad_photometry'] for m in metrics)} acceptations avec corrélation recalculée <0,65 et {sum(m['out_of_bounds'] for m in metrics)} supports acceptés hors image. Aucun cas n'atteint la limite de cinq tours dans cette campagne; les reprises s'arrêtent au plateau.\n"
    for level in [50,100,150,191]:text+=f"\n### Niveau {level}\n\n![Cartes et remappages](figures/plate_{level:03d}.png)\n\n[Animation référence/corrigée](figures/plate_{level:03d}.gif)\n"
    text+="\n## Limites et décision\n\nLa reprise peut enrichir les graines sans changer les seuils, mais ne garantit pas la résolution de toutes les fenêtres. Au niveau 191, examiner les diagnostics de l'affine et la distribution spatiale des graines avant de conclure à un manque d'itérations : une affine globale ne décrit pas nécessairement les zones encore non résolues. Les anciennes solutions restent immuables; une fausse acceptation initiale éventuelle n'est donc pas corrigée par cette expérience.\n\nLes contrôles photométriques finaux sont enregistrés par variante; ils ne remplacent pas une vérité de déplacement. `protocol.json`, `comparison.json`, `all_levels/` et les tests de composition rendent l'expérience reproductible. Aucun dépôt externe, poids appris ou seuil de production n'a été modifié.\n"
    (root/"README.md").write_text(text)


if __name__=="__main__":
    main()

# Strain manuscript draft

Read **main.pdf**. Editable source: main.tex and references.tex. Figures: figures/*.pdf. Regenerate with `python3 make_figures.py` from this directory, then `latexmk -pdf main.tex`. Python needs NumPy and Matplotlib; TeX needs a standard TeX Live installation.

This is an English research draft, not a submission-ready or peer-reviewed article. It preserves negative results and makes no universal-hardware or subsecond full-pipeline claim. No new solver campaign was executed for this draft. Numerical figures are regenerated from archived reports; the spatial map uses a per-window archive.

The manuscript uses Strain's explicitly permitted Free Format initial-submission route, with numbered references, unstructured abstract, five keywords, processing settings, declarations and separately supplied figures. It does not impersonate the publisher's final typesetting. See SUBMISSION.md for the official template download and submission checklist.

Evidence: data/sources.json records paths and SHA-256 hashes; data contains frozen report copies. The full source tree has uncommitted work beyond public commit 4dce5d0: archive a fixed release before submission. In particular, do not cite that commit as containing all experiments.

Scientific focus: image-derived affine preconditioning and local correlation, solver-confirmed seeds, first-order propagation, optional same-centre smaller-support solving. CPM-HOG is one optional proposal source, not the proven winner at large strain. The coarse learned matcher and rejected neural residual must not be conflated.

Revision 2: see REVISION_2.md. Mathematical derivations and controlled component impacts are now in mathematical_methods.tex and causal_results.tex. The original draft is preserved in versions/v1/.

## Image-based examples (revision 3)

Figures 5–7 now show actual reference/deformed speckles, archived solve outcomes, and native-pixel local affine pullbacks for a heterogeneous synthetic pair, a discontinuous pair and the masked Adrien experimental pair. `make_figures.py` also runs `make_image_examples.py`. Image and result fingerprints, chosen centres and the exact six parameters are in `data/image_examples.json`. Experimental TIFF inputs remain external at their recorded local paths. Missing inputs cause an explicit error; no synthetic substitute is used. The full-field images are downsampled only for print; the enlarged patches are sampled at native resolution. Remapping uses archived final solver parameters, not analytical truth, and does not claim initializer-only accuracy. The previous map-only manuscript is preserved in `versions/v2/main.pdf`.

## Refonte scientifique (CdC du 14 novembre 2026)

Entrées : `main.tex`, `sections/`, `references.tex`. Sorties : `main.pdf` et sa copie `manuscript_revised.pdf`, plus `supplement.pdf`. La version de 19 pages est dans `versions/v3/`.

Construction depuis la racine du dépôt :

```sh
rtk proxy sh paper/build_revision.sh
```

Le script recomptabilise les archives, régénère les figures, vérifie la géométrie avec les helpers existants dans `.venv` (PyTorch expérimental), puis compile les deux PDF et vérifie les logs. Il ne lance ni entraînement ni nouvelle qualification. Dépendances de figures : Python 3, NumPy, Matplotlib, Pillow, SciPy; compilation : latexmk et TeX Live. Les images expérimentales externes doivent être présentes à leurs chemins documentés. L'absence d'une dépendance provoque un échec explicite.

Traçabilité : `method_audit.md`, `evidence_matrix.csv`, `configurations/`, `audit/source_manifest.json`, `audit/recount.json`, `data/causal_recount.csv`. Les empreintes présentes ne prouvent pas une version historique non enregistrée; voir `open_issues.md`. Les diagrammes supplémentaires ont des sources SVG éditables et un générateur `make_method_figures.py`.

Contrôle visuel : rendre les PDF avec `pdftoppm`, puis inspecter les pages et les figures à taille de publication. Le détail des déplacements et corrections est dans `revision_map.md`. Aucun dépôt externe ni publication n'est effectué.

## Supplied large-deformation cases (14 November 2026)

`hudge_strains_addendum.tex` / `.pdf` report direct-pair tests on the supplied HudgeStrains sets. See `response_review_claude.md` and `../results/hudge_strains_20261114/README.md` for the operational scope, source fingerprints and unresolved displacement provenance. These results do not silently replace the original manuscript's accuracy evidence.

La méthode de silhouette est détaillée dans `sections/silhouette_initialization.tex`, intégrée à l’addendum, et en français dans `../results/hudge_strains_20261114/SILHOUETTE_METHOD.md`. Le schéma des boîtes et remappages est reproductible avec `make_silhouette_method_figure.py`.

Les quatre planches explicatives supplémentaires (masque, boîtes, support affine, propagation enregistrée) sont produites par `make_silhouette_explanation_figures.py` et incluses à côté de leurs équations dans la section 3.

Expérience complémentaire du 15 novembre : `../results/hudge_affine_restart_20261115/README.md` compare sur les 191 images de plaque les reprises affines natives et la rectification itérative suivie de validation native. Ces nouveaux résultats restent séparés des mesures de l’addendum du 14 novembre.

# Pistes de développement WGPU-Matching

## Objectif

Implémenter autant que possible le calcul des deux initialiseurs, DeepMatching et HORG-DIC, en WebGPU/WGSL. Garder les sources INRIA hors du dépôt. Toute expérimentation avec les jeux de données d’un autre dépôt reste en lecture seule. Conserver les mesures existantes et comparer toute modification à la vérité analytique Live-DIC.

## Ordre proposé pour HORG-DIC

1. **Améliorer la localisation des points-clés sur GPU.** Interpoler les extrema DoG dans les trois dimensions `(x, y, échelle)` par ajustement quadratique local WGSL, transmettre les coordonnées et l’échelle sous-pixel aux descripteurs, puis conserver les déplacements affines fractionnaires. Mesurer l’effet sur le nombre de points, la couverture, la précision à 2 px, la RMSE et la durée totale.
2. **Garder la préparation des correspondances sur GPU.** Porter le ratio Lowe, la construction des listes candidates par seed et leur compaction en WGSL. Réduire le retour CPU des descripteurs et des k plus proches voisins aux seuls cas nécessaires au diagnostic.
3. **Ajouter des contraintes de mouvement par cellules.** Évaluer un filtrage inspiré de GMS, adapté à l’affine locale des déformations DIC. Comparer qualité et coût avec le RANSAC GPU, sans supprimer les correspondances cohérentes avec un gradient spatial réel.
4. **Mieux couvrir les zones difficiles.** Si le consensus local rejette un POI, évaluer une recherche guidée par l’affine voisine ou par un descripteur multi-échelle. La garder conditionnelle pour limiter son coût.
5. **Optimiser après validation.** Profilage GPU par passe, parallélisation des hypothèses RANSAC, réduction des copies et fusions de noyaux ; comparer qualité et durée sur les mêmes frames et seeds.

## Références de départ

- Lowe, *Distinctive Image Features from Scale-Invariant Keypoints* (2004), localisation subpixel par développement de Taylor 3D du DoG : https://www-inst.eecs.berkeley.edu/~ee225b/sp14/readings/sift-lowe.pdf
- Bian et al., *GMS: Grid-based Motion Statistics for Fast, Ultra-Robust Feature Correspondence* (CVPR 2017) : https://openaccess.thecvf.com/content_cvpr_2017/html/Bian_GMS_Grid-based_Motion_CVPR_2017_paper.html
- *Chirality-aware initialization for reliable digital image correlation under large rotation and heterogeneous deformation* (HORG-DIC) : https://www.sciencedirect.com/science/article/abs/pii/S0263224126028435

Ces méthodes sont des pistes à adapter, pas des gains présumés. La contrainte d’acceptation reste la mesure sur les jeux et seeds déjà documentés.

## État

- **Mesuré :** interpolation quadratique 3D des extrema DoG dans `horg-detect.wgsl`, gradients et coordonnées sous-pixel dans les descripteurs HORG, déplacements affines flottants.
- **Mesuré :** consensus RANSAC affine dans `horg-ransac.wgsl` : tirages reproductibles selon le seed, évaluation des inliers, réajustement affine et résidu sur GPU. La compilation WGSL passe dans Chromium 156.
- **Mesuré :** ratio Lowe et préparation des candidats locaux migrés sur GPU (`horg-prepare.wgsl`). Les descripteurs et voisins kNN ne sont plus recopiés au CPU ; seul le petit tableau des ratios est lu pour les statistiques d’interface. Le repli quand moins de trois candidats tombent dans le rayon garde l’ordre et la logique antérieurs.
- **Mesuré :** cohérence affine leave-one-out à huit POI acceptés calculée par WGSL (`horg-confidence.wgsl`). L’écart et le nombre de voisins figurent dans le champ et le CSV exporté. C’est un diagnostic seulement : aucun POI n’est éliminé par cet indicateur.
- **Couche Python native :** `python/src/wgpu_matching` expose HORG, DeepMatching hiérarchique INRIA et la corrélation directe par `wgpu-py`. Les noyaux sont embarqués via `importlib.resources`; la wheel est pure Python (`py3-none-any`) et n’ajoute aucune extension à compiler. L’installation tire `wgpu` depuis sa wheel binaire.
- **Parité native validée sur Live-DIC :** HORG sous `wgpu-py`/Vulkan retrouve 693/685 points-clés, 387 ratios acceptés et 32/32 POI sur frame 040, avec les mêmes inliers/candidats et un écart maximal de déplacement de 0,0061 px par rapport au navigateur. DeepMatching hiérarchique, avec sa grille par défaut de 8 × 4, retrouve les 32 déplacements du CSV de comparaison (40 racines évaluées, 32 réciproques; écart maximal 0 px). Les premiers appels complets mesurés sont de 0,335 s en HORG et 1,927 s en DeepMatching avec seeds explicites, puis 2,114 s avec génération automatique de grille, sur l’adaptateur Intel Vulkan de cette machine; ces temps froids incluent la création des pipelines et ne sont pas à comparer aux médianes navigateur.
- **Packaging vérifié :** `pip wheel --no-deps .` produit `wgpu_matching-0.1.0-py3-none-any.whl`; installation dans une cible temporaire, chargement des WGSL depuis la wheel et exécution HORG + DeepMatching direct validés. Les fixtures `tests_gpuio` ont été seulement lues.
- **Benchmark Python natif :** `python/benchmarks/benchmark_live_dic.py` compare les deux méthodes sur les frames UHD 040/100/199; les POI hors champ sont exclus des métriques de qualité. Mesures et protocole dans `results/uniaxial/python_runtime_comparison.md` et `.csv`. Cinq appels répétés par méthode/frame donnent environ 0,09 s pour HORG et 1,84 s pour DeepMatching INRIA sur Intel Vulkan, avec RMSE analytique calculée sur les points acceptés visibles.
- **Déformation étendue :** `python/benchmarks/generate_tensile_60pct.py` adapte localement le générateur du dépôt DIC_CODEC jusqu’à 60 % d’allongement d’ingénieur; les six images 4K, 200 états analytiques et comparaison des détecteurs sont sous `results/uniaxial/tensile_60pct/`. Les deux détecteurs reçoivent uniquement les seeds visibles; HORG conserve leur identifiant RANSAC de la grille complète pour préserver les tirages déterministes. La source externe reste inchangée. Le matériau au-delà de 10 % est une extrapolation synthétique, pas une validation expérimentale.
- **Préconditionnement DeepMatching, calcul déplacé sur WGPU :** tirages déterministes réglables (`--seed`), hypothèses et scoring RANSAC, sélection du meilleur modèle et réajustement aux moindres carrés sont dans `ransac-affine*.wgsl`. ECC affine sur la demi-ROI centrale accumule/réduit les équations photométriques et résout le système 8 × 8 dans `ecc-affine-*.wgsl`; Python pilote les itérations et met à jour les paramètres depuis les huit incréments lus. Le remappage affine est WGSL dans `affine-warp.wgsl`. Sur l'essai synthétique 60 %, 16/16 POI visibles sont à ≤2 px, RMSE 0,00085 px; RANSAC GPU prend 6,2 ms, ECC 0,323 s (6 itérations), pipeline complet 4,107 s contre 4,112 s pour l'ancienne voie OpenCV sur un seul passage, sans gain de temps démontré. Les six niveaux donnent 24/28, 24/24, 24/24, 20/24, 20/20, 16/16 POI acceptés, tous à ≤2 px; les rejets à 10 % et 40 % persistent. La précision subpixel dépend ici de l'affine globale du jeu synthétique. À valider sur répétitions, champs hétérogènes et occlusions avant usage général.
- **Mesuré sur Live-DIC, trois appels après échauffement par frame :** HORG retient 87/96 POI, dont 85 à ≤2 px, RMSE combinée 0,557 px. Temps médian moyen 0,151 s ; le pipeline RANSAC + diagnostic LOO prend 2,7 à 7,6 ms selon la frame. Les comptes d’acceptation restent 32/32, 32/32 et 23/32. Détails et comparatif temporel : `results/uniaxial/deepmatching_horg_current_comparison.md`.
- **Indicateurs de confiance, étude exploratoire :** le leave-one-out affine à huit voisins réduit les faux positifs de 17 à 7 sur 18 frames de contrôle, mais rate encore sept erreurs et écarte six points corrects. Le seuil ≥10 inliers ne laisse passer aucune erreur dans ce contrôle, au prix d’une perte de couverture. Garder ces indicateurs comme diagnostics tant qu’ils ne sont pas validés sur DIC hétérogène et images expérimentales. Détails : `results/uniaxial/horg_quality_indicators.md`.
- **À faire ensuite :** évaluer une contrainte spatiale inspirée de GMS, adaptée aux gradients de déplacement DIC, comme score de diagnostic avant tout rejet. Puis paralléliser les hypothèses RANSAC par POI et profiler chaque noyau/transfert. Garder la recherche guidée par affine ou multi-échelle conditionnelle aux régions difficiles. Les sources INRIA et les fixtures externes restent hors modification.

## Sparse DeepMatching study (November 2026)

- **Finding:** explicit `seed_points` currently limit only returned POIs. The default INRIA-style path still evaluates the complete image root grid (40 roots on the tested UHD images).
- **Prototype:** `DeepMatchingOptions(mode="inria_sparse", seed_points=...)` evaluates only the distinct hierarchy roots containing the requested atomic source leaves. It reports `evaluated_root_count`, `global_root_count`, estimated descriptor comparisons, and `reciprocal_scope`. The default `mode="inria"` is unchanged.
- **Measured on Live-DIC frames 040, 100, and 199:** one seed evaluated 1/40 roots and took 0.110–0.127 s, versus 1.586–2.394 s for the full grid. Eight seeds evaluated 8/40 roots and took 0.312–0.351 s. On the tested subsets, accepted/rejected decisions and displacement vectors matched full-grid results exactly. Data and protocol: `results/uniaxial/sparse_deepmatching_study.md` and `.csv`; reproducible benchmark: `python/benchmarks/benchmark_sparse_deepmatching.py`.
- **Limit:** sparse reciprocal competition excludes unselected roots, so parity is empirical for these inputs, not guaranteed generally. The measured seeds are prefixes of the visible regular grid, a controlled workload proxy; automatic identification of problematic windows and an end-to-end fallback path are not implemented.
- **Next:** try sparse fallback on POIs rejected by HORG or local IC-GN, compare on held-out frames and heterogeneous/occluded fields, and add a caller-side policy only after validation. Retain full-grid INRIA mode for parity-sensitive comparisons.

# CPM-WGPU implementation contract

## 1. Context and objective

Repository: `https://github.com/jeffwitz/WGPU-Matching`.

The repository currently contains two DIC initializers:

- **DeepMatching WebGPU:** a hierarchical INRIA-style reimplementation whose outputs have been checked against the reference CPU implementation.
- **HORG-DIC WebGPU:** SIFT/DoG keypoint detection, HORG descriptors and GPU matching, followed by robust local affine fitting.

Add a third method, **CPM-WGPU**, inspired by Hu, Song, and Li, *Efficient Coarse-to-Fine PatchMatch for Large Displacement Optical Flow*, CVPR 2016:

- Paper: `https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Hu_Efficient_Coarse-To-Fine_PatchMatch_CVPR_2016_paper.html`
- Original implementation: `https://github.com/YinlinHu/CPM`

The intended use is initialization of translation or affine Lucas–Kanade IC-GN on experimental speckle images. Optimize the number of correctly initialized DIC windows per unit compute, not the quality of a dense optical-flow field in isolation.

Constraints:

- JavaScript ES modules and WGSL shaders; native browser WebGPU without CUDA or a GPU-specific compute framework.
- Support compatible integrated Intel GPUs and other WebGPU adapters.
- Handle images up to 3840 × 2160 and, where adapter limits permit, about 8000 × 6000.
- Do not require neural networks or machine-learning frameworks.
- Preserve DeepMatching and HORG behavior and results.

Implement, test, and integrate CPM; a proposal or theoretical study alone does not satisfy this contract.

## 2. Required preparation

Before CPM code changes:

1. Inspect `deepmatching.js`, `deepmatching-descriptor.wgsl`, `deepmatching-hierarchy.wgsl`, `horg.js`, `app.js`, and `index.html` for reusable GPU stages, data structures, and coordinate conventions.
2. Inspect current results in `results/uniaxial/`, especially the DeepMatching, HORG, and INRIA comparisons.
3. Consult the paper and, where useful, the original CPM source to verify its core operations.
4. Check the original source license before reusing any code. Keep the implementation self-contained and state what is inspired by CPM versus adapted for WebGPU.

## 3. Algorithm requirements

Implement coarse-to-fine correspondence search based on PatchMatch.

### 3.1 Pyramids

- Build Gaussian or otherwise suitably filtered image pyramids with factor-two scale changes.
- Make pyramid depth configurable and reduce it when speckle texture becomes unresolved.
- Preserve pixel-center and scale-coordinate conventions.
- Do not force one arbitrary pyramid depth for every image.

### 3.2 Descriptors

- Initially reuse the repository's validated HOG descriptors.
- Do not initially add CPM's dense 128-component SIFT descriptor.
- Define a normalized patch-descriptor similarity cost, with a future path to grayscale ZNCC/ZNSSD.
- Run descriptor computation on the GPU and avoid unnecessary GPU-to-CPU transfers.
- Document that CPM-HOG differs from original CPM; do not claim strict numerical equivalence when descriptors or propagation differ.

### 3.3 PatchMatch search

Maintain a displacement hypothesis `u(x,y) = (dx,dy)` at each internal matching-grid location.

**Initialization**

- Generate allowed displacement candidates at coarse levels.
- At finer levels, rescale and reuse the coarser hypotheses.
- Interpret the search radius in original-image pixels.
- Do not hard-code a maximum displacement such as 400 px.

**Spatial propagation**

- Test hypotheses proposed by neighbors, retain the lowest-cost match, and repeat for configurable iterations.
- Run propagation on the GPU with defined memory synchronization.
- Prefer deterministic passes with ping-pong buffers; consider checkerboard propagation only if it materially improves performance.
- Do not rely on undefined concurrent read/write behavior.

**Random search**

- Test candidates around the current best displacement using decreasing perturbation radii.
- Generate perturbations from a deterministic hash of coordinates, level, iteration, and configurable seed.
- Keep the best candidate and make runs reproducible for fixed parameters and adapter. Measure any GPU-dependent numerical variation.

**Coarse-to-fine traversal**

- Upscale displacement hypotheses between levels, then propagate and random-search at each finer level.
- Do not run an exhaustive global search at every level.

### 3.4 Internal matching grid

- Do not require a dense per-pixel flow field when only a few DIC windows are needed.
- Provide a configurable internal grid stride, for example 8, 16, or 32 px in original-image coordinates. It may be denser than the DIC grid to help propagation.
- Support manually requested seeds.
- Distinguish measured internal-grid centers, requested DIC centers, and any interpolated or fitted estimates. Never label an interpolated displacement as a directly measured correspondence.

## 4. Correspondence verification and confidence

False matches matter as much as missing matches.

### 4.1 Forward-backward consistency

- Match reference to target and target to reference.
- For source point `x` and its match `x'`, compute `e_FB = ||u_AB(x) + u_BA(x')||`.
- Interpolation of the reverse displacement is allowed when `x'` is off-grid, provided interpolation does not cross an invalid region.
- Make the reciprocity threshold configurable and document its default.

### 4.2 Match diagnostics

Expose, per match, the matching cost, forward-backward error, validity, measured source coordinate, and estimated displacement. Expose an ambiguity score only if the computed quantity truly measures ambiguity; do not invent a second-best score. Document acceptance criteria and retain rejected matches in CSV exports.

### 4.3 Invalid regions and masks

- Avoid evaluating patches that extend outside the image.
- Design the method so a full-image ROI mask can be added later.
- If an application mask exists when implementing CPM, respect it; otherwise do not require creating one in the first iteration.
- Future mask handling must reject windows crossing the mask boundary without imposing mechanical smoothness across that boundary.
- Do not artificially smooth displacement across separate regions.

## 5. IC-GN initialization interface

Provide an exported function compatible with the current app structure, for example:

```javascript
computeCPM(device, referencePixels, targetPixels, width, height, settings, onProgress)
```

Return a visualization/export-compatible structure with `algorithm: "CPM-WebGPU"`, image dimensions, patch size, stride, scale, and matches containing `x`, `y`, `dx`, `dy`, `score`, `reciprocal`, `usable`, plus required diagnostics and counters.

**Translation:** return a floating-point translation initialization at each DIC-window center; do not round displacements before initialization.

**Affine:** allow a local affine estimate from nearby correspondences using

```text
x' = a0 + a1 (x-xc) + a2 (y-yc)
y' = b0 + b1 (x-xc) + b2 (y-yc)
```

Expose all six parameters for an affine IC-GN initializer. A robust local fit is allowed when enough valid, non-degenerate correspondences exist. Do not force affine fitting when support is insufficient; return translation-only or an unusable status. Matching must not mechanically regularize the physical displacement field. This repository does not need a new IC-GN solver; prepare a compatible initialization interface only.

## 6. WebGPU performance constraints

- Pyramids, descriptors, matching, and intermediate hypotheses/scores stay on GPU.
- Read back only results needed by JavaScript.
- Organize work into GPU passes; avoid CPU synchronization on every PatchMatch iteration.
- Reuse buffers where appropriate, respect device buffer limits, and release temporary resources.
- Do not build a dense matrix of every source-target correspondence.
- Aim to beat the observed ~1.3 s DeepMatching WebGPU time on UHD tensile images on the same GPU and a comparable timing protocol. This is a performance target, never permission to sacrifice reliability. Do not claim performance without measurement.

## 7. UI integration

Add a third algorithm choice: DeepMatching WebGPU, HORG-DIC WebGPU, and CPM-WGPU. Preserve both existing methods' behavior and defaults. Expose only useful CPM parameters with documented defaults: pyramid levels, matching-grid stride, propagation iterations, random-search trials, search radius, reciprocity threshold, and any confidence controls actually used.

Keep manual seeds and current accepted/rejected vector visualization. Explicitly adapt `renderFieldSummary`, `exportCsv`, and `updateAlgorithmControls` in `app.js`; DeepMatching/HORG-specific branches cannot be reused unchanged. Add a CPM CSV export with coordinates, displacement, scores, reciprocity, validity, and principal parameters.

## 8. Mandatory scientific validation

Validation is a required deliverable.

### 8.1 Reproducible synthetic cases

Use realistic speckle patterns, not only independent white noise. Vary speckle size and blur. Cover identical images; translations of 8, 32, 64, 128 px and larger when image size allows; increasing rigid rotations; homogeneous tensile strain of 1%, 5%, 10%, and if possible 20%; homogeneous and heterogeneous shear; noise, contrast and illumination changes; and low-texture or ambiguous regions.

Check coordinate transforms and scale transitions for factor-of-two and half-pixel errors. As an initial regression target, a broad majority of simple synthetic-translation matches should be within 2 px. If not, diagnose the failure rather than changing rejection thresholds to hide it.

### 8.2 Live-DIC comparison

If accessible, use `tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile` read-only and compare frames 040, 100, and 199 against the reference. Use identical evaluation coordinates for DeepMatching, HORG, and CPM even when internal grids differ.

Report requested seeds, accepted matches, accepted matches within 2 px of analytic truth, accepted RMSE, median, P95, maximum error, and runtime. Report conditional precision among accepted matches separately from accurate coverage over all requested seeds. Preserve existing method parameters.

### 8.3 Performance protocol

Run repeated measurements on the same adapter and images. Document browser/version, adapter, per-method settings, preprocessing, matching, total time, and medians. Separate image read/decode from matching. Clearly flag asymmetric timing boundaries.

### 8.4 Method comparison

Publish a common DeepMatching/HORG/CPM table for accurate coverage, precision of accepted matches, accepted RMSE, and time. Do not tune existing methods merely to make CPM look better. If CPM loses, retain the result and analyze its limits.

### 8.5 Optional later IC-GN convergence validation

If the IC-GN solver in `tests_gpuio` is accessible without substantial modifications, optionally compare convergence rates using each initializer. Otherwise document exported data needed for future evaluation. This is not a blocker for CPM delivery.

## 9. Code organization

Prefer modular code such as `cpm.js`, `cpm-pyramid.wgsl`, and `cpm-match.wgsl`, with a `cpm-validation` or equivalent test area and reproducible results under `results/cpm/`. Adapt names and reuse existing modules where appropriate. Avoid unnecessary refactors of DeepMatching/HORG and avoid adding a large testing framework for a small need.

## 10. Acceptance criteria

The work is complete when:

1. CPM is a runnable third UI method for two images.
2. It genuinely implements spatial propagation, random search, and multi-scale refinement.
3. Costly matching stages run on WebGPU.
4. Results include displacement, score, confidence diagnostics, and reciprocity.
5. Visualization and CSV export work.
6. Coordinates and scale changes are verified analytically.
7. Reproducible tests compare CPM with both existing methods.
8. Real performance is measured and documented.
9. DeepMatching and HORG have no regressions.
10. README documents method, parameters, limitations, and results.

Many inaccurate matches are not a success. A smaller set of reliable matches may be preferable to a dense field of false matches, but accurate coverage must always be reported.

## 11. Out of scope

Do not implement Flow Fields/Flow Fields+, EpicFlow, RicFlow, neural networks, a new IC-GN solver, global mechanical regularization, or a complete dense-flow reconstruction system in this first iteration. Do not alter historical DeepMatching or HORG results. Avoid premature optimization that obstructs algorithm validation.

## 12. Deliverables and completion report

Deliver working integrated code; automated tests and their actual results; a DeepMatching/HORG/CPM comparison; a concise GPU performance report; analysis of speckle size, large displacement, rotation, and heterogeneous deformation limits; explicit differences from original CPM; and improvement proposals ranked by expected impact.

Run available tests, fix observed failures, and document anything not run. Never report an unexecuted test as successful. Once implementation and verification are complete, create a clearly named commit following repository conventions.

**Top priority:** deliver a third WebGPU initializer that is scientifically comparable with the existing methods; do not optimize only for benchmark time.

## CPM implementation log (November 2026)

- Added an experimental browser UI option and an independent `cpm.js`, `cpm-pyramid.wgsl`, and `cpm-match.wgsl` path. The prototype reuses the repository's INRIA-style GPU HOG descriptor builder, pools those descriptors into a factor-two pyramid, evaluates a configurable sparse work grid, and runs deterministic ping-pong propagation plus bounded random search on WebGPU in both directions.
- Added a forward/backward displacement error diagnostic, seed-to-work-grid distance, directly evaluated source coordinates, and a CPM-specific CSV schema. Manual requested centers are retained separately from measured work-grid centers.
- Initial implementation status (superseded by the audit follow-up below): the method pools a single base HOG descriptor map instead of recomputing descriptors at each image-pyramid level. The follow-up adds local affine output projection and comparative studies, but does not close the full scientific validation contract.
- The original CPM GitHub repository declares GPL-3.0. The implementation uses an independent WGSL design and does not copy its source.
- The CVPR paper describes a true top-down image pyramid (factor 2), transferring flow by factor 2 between levels, interleaved neighbor propagation and exponentially shrinking random search, a dense 128D SIFT L1 patch cost, and forward/backward outlier checking. The current implementation uses once-computed HOG descriptors pooled into a pyramid, cosine patch cost, and synchronous four-neighbor ping-pong passes; these are deliberate WebGPU/DIC adaptations, not numerical CPM equivalence. The paper reports `{d,r,k,n}={3,4,5,6}` for its own evaluation; these are not adopted as DIC defaults without validation.
- Static JavaScript syntax checks pass for `app.js`, `cpm.js`, and `deepmatching.js`. With the `tests_gpuio` Chromium Vulkan flags, the deterministic synthetic translation harness returns 3/3 accepted matches with 0 px error for the tested `(8,-4)` translation. The first browser launch with an incomplete flag set had exited with code 139; the corrected launch now executes GPU tests.
- **Audit follow-up: CPM support and diagnostics (November 2026):** matching support is now separately configurable from the DIC window. The matching support controls the rigid HOG cost patch, valid image borders, and effective pyramid depth. A deterministic stratified initial hypothesis sampler, per-level analytic-oracle HOG scoring, strict bilinear inverse-field interpolation, and robust local affine projection at requested seeds have been added. The latter returns requested centers separately from the directly measured grid positions and marks unsupported fits unusable. Current browser results: `results/cpm/progressive_tensile_040_100_199.md` and `results/cpm/oracle_diagnostics_60.md`, with raw JSON alongside them.
- **Progressive comparison:** using the same visible seeds and three repeated calls, CPM accepted 32/32 at 0.11% (RMSE 1.078 px; 0.565 s), 2/28 at 3.84% (accepted RMSE 1.554 px; 0.550 s), and 0/28 at 10% (0.526 s). At 10%, 13/28 unfiltered vectors are within 5 px and 10/28 within 2 px, but all are rejected; the all-proposal RMSE is 168.039 px. DeepMatching accepts 32/32, 28/28, 28/28; HORG accepts 32/32, 28/28, 23/28 at those frames. These are synthetic affine image warps, not heterogeneous experimental strain fields.
- **Oracle diagnosis and 60% update:** at 60%, updated CPM accepts 0/16; proposal RMSE is 757.235 px, and work-grid reciprocal count is zero. With 32 px support, the known displacement has lower HOG cost than the best found candidate for 9/16 seeds at scale ×4 and 11/16 at ×8; this implicates search coverage. With 64 px support it has lower cost for only 0/16, 1/16, and 4/16 seeds at ×4, ×8, ×16, consistent with a rigid-patch objective problem under affine deformation. Thus the audit's separation of search and cost failure is supported but not conclusive. The current 60% browser medians are 1.307 s DeepMatching, 0.146 s HORG, and 0.517 s CPM. Exact GPU adapter identity is unavailable from Chromium, and CPM is not qualified for these larger strains. Updated report/raw files: `results/uniaxial/tensile_60pct/three_method_browser_60.md`, `.csv`, `.json`.
- **Validation status:** the simple `(8,-4)` translation browser test passes 3/3 with zero vector error both through the local harness and the new CI runner. The Python unit suite passes (8 tests) in the project test virtual environment; plain system `pytest` without the package installed fails collection. `sphinx-build -b html -W --keep-going` succeeds. The progressive/60% studies remain separate browser experiments requiring the neighboring read-only `tests_gpuio` assets; they are not CI tests.

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

# Learned detector and speckle-specific detector contract

## Objective

Evaluate DaD (official 2025 detector), DeDoDe v2-L, and a lightweight DIC-specific detector for initializing DIC on speckle images with large displacement and affine or heterogeneous deformation. Compare detector quality separately from descriptors, matching, geometric verification, and IC-GN convergence. Preserve DeepMatching, HORG, CPM, XFeat, and EfficientLoFTR behavior and all historical measurements.

## Required sequence

1. **P0 reference adapters:** add a shared floating-point keypoint result contract; use official DaD weights and the official DeDoDe v2-L detector checkpoint. Keep PyTorch, model code, and weights optional and outside the core install and Git history. Record upstream revision, checkpoint URL, and SHA-256 for every run. Verify model output and coordinate conversion against upstream conventions. Use DeDoDe-B as the common descriptor; permit sampling it at points from DaD, DeDoDe v2, HORG/SIFT, and DIC-Detector.
2. **P1 detector and matcher comparison:** measure repeatability at 1, 2, and 5 px, point coverage, spatial distribution, points per DIC window, match precision, false positives, local-affine coverage, and extraction/matching time. Include budgets 512, 1024, 2048, 4096, and 8192 and compare global top-K with spatial quotas. Use shared seeds, common descriptors, common reciprocal matching, and the same WGSL affine RANSAC criteria. Keep full original pipelines as separate baselines.
3. **P2 WGSL non-learned baseline:** implement local structure-tensor scoring, non-maximum suppression, ROI-aware point selection, and spatial quotas in WGSL. Report conditioning metrics as local IC-GN information indicators, not as evidence of correspondence uniqueness.
4. **P3/P4 DIC-Detector:** provide a small trainable convolutional model and deterministic speckle/warp data-generation pipeline. Separate train/validation/test textures and deformation fields. Begin with repeatability and descriptor-discrimination supervision; add an IC-GN-success reliability target only when a compatible solver is available. Never claim that an untrained model is validated.
5. **P5 WebGPU:** use PyTorch as an optional oracle. Test ONNX/WebGPU where graph operators allow it, and port DIC-Detector inference and the WGSL baseline to WebGPU. Measure CPU fallbacks, transfers, memory, UHD tiling, reference caching, and paired versus cached-reference latency. Do not label execution GPU-resident without profile evidence.
6. **P6 descriptor:** train a compact affine-aware DIC descriptor only if common DeDoDe-B results show that descriptor discrimination limits the detector comparison.
7. **P7 qualification:** compare HORG, DeepMatching, XFeat, EfficientLoFTR, DaD + DeDoDe-B, DeDoDe v2 + DeDoDe-B, DIC-Detector + DeDoDe-B, and any later DIC descriptor using the same criteria. Include progressive Live-DIC frames 040/100/199, synthetic 10–60% tension, and held-out heterogeneous fields/rotations. Report accepted coverage and correct coverage over all visible seeds separately, including error thresholds ≤2 px and <5 px, timing scope, memory, and model resolution.

## License and reproducibility constraints

- Do not copy third-party implementations into the repository. Depend on or adapt through the official projects and keep their license notices.
- The DaD repository and the DeDoDe detector/DeDoDe-B code and weights state MIT licensing. DeDoDe's DINOv2-based descriptor-G has a separate Apache-2.0 license; it is outside the initial protocol.
- Do not commit learned weights. Download them to an external cache or explicit user path and write their SHA-256 to raw benchmark results.
- PyTorch and upstream model packages remain optional; `pip install wgpu-matching` must not install them.
- Treat the external `tests_gpuio` and `DIC_CODEC` trees as read-only. Do not copy their implementation or alter their files.

## Acceptance and scientific limitations

- A detector-only repeatability gain is not a matching or DIC-initialization gain.
- Report measured keypoints separately from interpolated or fitted DIC seed estimates.
- Do not tune thresholds on the held-out final test cases or change baseline parameters to make a new detector look better.
- Validate seed coordinates, pixel-centre conventions, ROI filtering, chunked matching determinism, and descriptor parity.
- The final scientific target is correctly converged DIC windows per second. If IC-GN is not available in this repository without invasive work, explicitly report that convergence remains unmeasured and retain error-to-analytic-truth metrics.
- Retain negative results and explain cross-domain limits: DaD/DeDoDe pretraining targets natural-image local matching, not speckle textures.

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

## Shared WGSL sources and Python/browser parity (November 2026)

- **Single shader source:** canonical WGSL files now live only in `python/src/wgpu_matching/shaders/`. Python continues to load them with `importlib.resources`; browser DeepMatching, HORG, and CPM modules fetch those same files from the static repository tree. The duplicate root-level files were removed, including placing the two CPM shaders in the packaged shader directory. Setuptools package data already includes `shaders/*.wgsl`, so the Python wheel keeps the full runtime shader set.
- **Shared option defaults:** `python/src/wgpu_matching/matcher-contracts.json` defines DeepMatching, HORG, and CPM defaults, constraints, coordinate conventions, and Python/JavaScript option-key mappings. Python option dataclasses derive defaults from the packaged JSON; browser matcher functions merge these defaults, and the UI initializes its controls from them. The common DIC defaults are a 128 px window and 512 px stride. CPM's matching support and work grid are independently 32 px and 16 px.
- **Shared option validation:** the same contract declares integer/numeric bounds, accepted DeepMatching modes, and a 512-point manual-seed limit. Browser defaults application and Python public matcher methods validate these rules before GPU work; both reject non-finite or out-of-image manual coordinates. Keep the Python contract tests and browser parity rejection checks in sync when changing constraints.
- **Validation checks:** `python/src/wgpu_matching/contracts.py` applies manifest constraints at the Python API boundary, while `matcher-contracts.js` applies them to browser calls. Python tests exercise invalid values and schema completeness; the Chromium parity runner checks rejection of an excessive downscale level, zero HORG feature count, and 513 manual seeds. Empty seed lists consistently mean “use the regular grid.”
- **Layout contract:** `python/tests/test_shader_source_layout.py` checks all WGSL filenames referenced by browser modules exist in the package shader directory, rejects root-level duplicate WGSL files, and checks the wheel package-data declaration.
- **Frontend parity harness:** `tests/run_matcher_parity.mjs` runs all three frontends in Chromium and Python on deterministic 512×384 speckle fixtures with known translation, mild affine warp, and three irregular fractional manual seeds. CPM uses a 64 px DIC window and a separate 32 px matching support; parity includes work-grid counts, local affine diagnostics, reciprocal status, and oracle costs. The harness checks each frontend's defaults and accepted displacement error. Sources are `tests/generate_matcher_parity_fixture.py` and `tests/matcher_parity_python.py`.
- **Measured locally:** DeepMatching matched grid and manual-seed results exactly, all within 2 px. HORG accepted all grid and manual seeds; maximum browser/Python displacement difference was 0.000367 px. CPM accepted all three manual seeds within 2 px; displacement differences were below 1e-14 px and oracle costs matched exactly. Python reported Intel Graphics LNL Vulkan; Chromium did not expose its adapter name. This translation fixture does not qualify CPM on large deformation.
- **Native CPM:** `python/src/wgpu_matching/cpm.py` uses a shared HOG descriptor builder, creates the pooled descriptor pyramid, and runs the same ping-pong PatchMatch/oracle WGSL as the browser. Reciprocal interpolation and weighted local affine fitting follow the browser path. Work-grid dimensions derive from matching support, independently of DIC window size; parity exercises that distinction.
- **CI:** the software-Vulkan browser job runs parity for all three methods before the CPM smoke test, verifies visible app defaults, and checks rejection of invalid options. Python unit tests cover the shared constraints. CPM browser smoke passes 3/3 with 0 px error.
- **Latest validation:** 25 Python tests pass, Ruff is clean, Sphinx strict build succeeds, and the rebuilt wheel installs with `cpm.py`, the contract JSON, and all 20 WGSL files. Cross-frontend parity, CPM smoke, and the all-method Python quickstart pass. Historical 60% results predate the work-grid correction and need rerunning before they are treated as current CPM measurements.
- **Remaining architecture work:** algorithm orchestration and some result assembly are still duplicated between JavaScript and Python; the option-key mapping is declared once, but frontend-specific key names remain. The parity fixture is an initial regression guard, not proof of full algorithm equivalence across datasets or adapters. Avoid expanding it by loosening thresholds; add additional controlled transforms and validated method contracts incrementally.


# Active implementation contract: shared affine / Gaussian-RBF initialization

**Requested by the user on 2026-11-06. Status: Python/WGPU affine and Gaussian-RBF projection prototype implemented; scientific qualification and browser integration remain in progress.**
The designated implementation agent is Luna. This section defines its work; it does
not authorize claiming that any of the implementation or validation below already exists.
For this feature, this contract supersedes earlier proposals requiring crack-aware
segmentation or native-resolution photometric refinement before initialization.
Historical matcher contracts and results remain applicable to their own methods.

## 1. Objective and scope

Provide one method-independent layer that converts measured image correspondences
into a smooth displacement initializer and then into the six affine parameters of
each requested DIC patch. Offer two user-selectable bases:

- `affine`: a robust global affine displacement field;
- `gaussian_rbf`: a robust regularized Gaussian-RBF displacement field with an
  unpenalized affine component and automatic hyperparameter selection.

Support every existing correspondence-producing method: DeepMatching (including
sparse modes), HORG, CPM, XFeat, RaCo/ALIKED/LightGlue variants, and EfficientLoFTR.
A detector alone does not provide displacement observations: its corresponding
matcher must supply source/target pairs. Optional external models remain optional.

The initializer is not the final mechanical measurement. IC-GN receives the patch
parameters and remains free to converge independently on original image pixels.
Do not warp entire images as part of the production API. Do not add an IC-GN solver,
photometric refinement, fracture detection, discontinuity segmentation, separate
motion branches, or physical constitutive constraints in this first version.
Continuous heterogeneous deformation and localization are in scope. A localization
may be under-resolved by the initializer; report its consequences rather than
claiming the reconstructed field is the true deformation. Crack handling is deferred.

Python / wgpu-py is the primary interface. Perform the expensive projection and
patch evaluation on WebGPU; retain a small independent float64 CPU reference for
validation. Preserve all existing matcher behavior unless projection is explicitly
requested. Do not change historical results or external datasets/repositories.

## 2. Inspect and reuse before implementation

Read `api.py`, `horg.py`, `deepmatching.py`, `cpm.py`, their browser counterparts,
`matcher-contracts.json`, the feature benchmark adapters and shared WGSL directory.
Use the codebase graph tools for discovery if available; otherwise document that
fallback and inspect sources. Identify where each backend still has actual measured
pairs before local affine fitting, interpolation, or seed association.

The current `MatchField`/`LocalAffineField` APIs must remain backward compatible.
In particular, do not feed a field already reconstructed from a handful of seed
fits back into projection and represent it as thousands of independent matches.
Prefer an opt-in correspondence export at the actual measurement stage.

## 3. Common correspondence contract and provenance

Define a documented `CorrespondenceSet` (name may follow repository conventions):

- reference and target coordinates, floating point, shape `(N,2)` each;
- stable match identifiers, extractor/matcher identity and run metadata;
- native image dimensions and complete resize/padding/crop coordinate mappings;
- raw method score, reciprocal diagnostics and original acceptance/rejection flags;
- `measurement_kind`, distinguishing directly measured from previously inferred points;
- optional explicitly calibrated confidence or covariance; absence must be supported.

All projection coordinates are native original-image pixels, with the repository's
pixel-center convention. Respect separate x/y resize factors and padding offsets.
Never assume that multiplication by a single downscale factor is sufficient.
Validate these mappings on controlled coordinates, including fractional centers.

Default fitting input consists of direct, finite, in-image correspondences passing
the backend's documented descriptor/reciprocity checks. Preserve excluded rows and
reasons. Do not introduce a mandatory global-affine RANSAC gate: real non-affine
motion must remain eligible for the RBF fit. Legacy affine rejection may be recorded
as a diagnostic without becoming an irreversible input filter for the new layer.

Do not compare raw confidence scales across matchers or interpret arbitrary scores
as probabilities. Initial score weighting is uniform unless a documented calibration
is available. Deduplicate identical source/target pairs deterministically; retain
provenance and identify incompatible duplicate-source observations. Use bounded
spatial-density weights so one dense feature cluster cannot dominate the field;
publish the weighting rule and its effect. Do not fuse different matchers by default.

Allow host-array inputs and same-device GPU-buffer inputs with explicit ownership,
layout and lifetime. ONNX-to-wgpu interoperability is not assumed; record any actual
host round trip. The common adapter must not label externally computed outputs as
resident in wgpu memory without evidence.

## 4. Coordinate system and mathematical model

Fit displacement `d_i = y_i - x_i`, not absolute target positions. Choose one
reference-coordinate normalization `z=(x-o)/s` with isotropic scalar `s>0` derived
from the reference ROI/image extent. Store it with the fitted model. Normalize
displacements by the same `s`. Do not independently normalize source and target
without explicitly restoring the induced affine transformation.

Affine model in normalized units:

`v(z) = b + A z`, with `u(x) = s v((x-o)/s)`.

Gaussian-RBF model:

`v(z) = b + A z + sum_k c_k exp(-||z-q_k||^2 / (2 ell^2))`.

The affine component is unpenalized. RBF centers `q_k` are a bounded, spatially
representative subset of training source locations, selected deterministically
(e.g. farthest-point sampling over occupied spatial cells). Center selection must
not depend on analytic truth or validation target displacements. Avoid an N-by-N
kernel solve over all feature points. The center count is a tunable capacity limit.

Use an explicitly normalized robust regression objective, for example:

`sum_i w_i rho(||v(z_i)-d_i/s|| / sigma) / sum_i w_i + lambda ||C||_F^2`.

Here sigma is a robust positive residual scale with a documented lower floor,
rho is a vector-residual Huber loss, and lambda penalizes only RBF coefficients.
Specify exact Huber tuning, scale update and stopping conditions in the shared
contract. Freeze the selected loss settings across benchmark methods.
This finite-basis ridge model is an adaptation inspired by RKHS approaches;
it is not an exact FG-GMM implementation or an assertion that coefficient ridge
is identical to the paper's RKHS norm penalty.

Resolve affine/RBF collinearity explicitly through a stable weighted projection,
QR/SVD in the CPU reference, or an equivalent constrained/regularized formulation.
The prediction must retain unpenalized affine reproduction. Document how any
basis orthogonalization is carried into evaluation at new points. Never stabilize
by silently penalizing the affine coefficients or rounding displacements.

Use IRLS with bounded iteration count and monitored convergence. Severe outlier
contamination can defeat this estimator; include that failure mode in validation.
Existing RANSAC may provide an optional starting hypothesis, but must not permanently
remove non-affine observations. Preserve final robust weights and residuals per pair.

Gaussian RBFs are the selected first implementation because of their analytic
values/derivatives and controllable length scale. Do not add an assortment of
kernels without measured evidence of a Gaussian failure. A later kernel substitution
requires a separately reported comparison using the same data and evaluation seeds.

## 5. Automatic selection: bounded and reproducible, not an optimality claim

`gaussian_rbf` supports fixed and automatic `(center_count, ell, lambda)` settings.
Automatic is the normal user path. Record the full candidate table, validation
errors, failures and selection decision. "Optimal" means selected within this
finite search, never a guarantee of the globally best function or kernel.

Start with a bounded candidate family: center counts 16/32/64 (capped by available
training information), widths 1/2/4 times the typical center spacing, and normalized
ridge strengths 1e-6/1e-4/1e-2. These are initial engineering search bounds, not
validated defaults. Allow explicit overrides; expansions must have a time/memory
budget and be reported. Include affine as a baseline candidate and expose an
optional automatic simplicity fallback. If RBF mode returns an affine fallback,
return `effective_basis='affine'` and a reason, never hide the fallback.

Use deterministic spatial block cross-validation (initially 3 folds), balancing
coverage and checking training rank. Select centers, robust weights and residual
scale using training rows only; no geometric filtering fitted on validation data.
Ensure block size reflects the intended interpolation distance; an entire missing
half-image tests extrapolation and must not be confused with local interpolation.

Score every eligible held-out observation using a fixed robust criterion with a
validation residual scale shared across candidates, derived from training data.
Do not let each candidate discard its own hard validation points or inflate its
own evaluation scale. Also publish uncapped median/P95 errors and counts. Select
the least complex candidate within the estimated fold-score uncertainty of the
best; define the tie-break rule deterministically. If folds cannot support a
meaningful comparison, report `selection_unavailable` and use a documented fixed
configuration or affine fallback with explicit status.

Cross-validation residuals include feature localization noise and outliers. They
are not analytic field error and are not calibrated confidence intervals. Use
synthetic truth only to evaluate this selection procedure, never to choose the
model during a production call.

## 6. Derive each patch's affine initialization from the fitted field

Separate fitting the common field from producing parameters for requested patches.
Keep requested fractional centers exactly. Default to `patch_projection='support_fit'`:
fit the local affine displacement approximation to samples of the predicted field
throughout each patch, keeping its center displacement exact.

For center `xc`, offsets `r_j` covering the patch, solve the small weighted problem:

`u_c = u(xc)`

`G_c = argmin_G sum_j omega_j ||u(xc+r_j) - u_c - G r_j||^2`.

Use a fixed symmetric tensor grid initially (e.g. 5x5 quadrature/sample points),
with coordinates based on `(patch_width-1)/2` and `(patch_height-1)/2`; specify
weights and handling of non-square patches. This is projection of the predicted
field, with no image interpolation and no patch-specific rematching/RANSAC.

Return the following explicit parameter order:

`[u_c.x, u_c.y, G_c[0,0], G_c[0,1], G_c[1,0], G_c[1,1]]`

corresponding to `[u,v,du/dx,du/dy,dv/dx,dv/dy]`. The coordinate map is
`x_target = xc + u_c + (I+G_c)(x-xc)`.

Also offer `patch_projection='taylor'` using the analytic gradient at the center,
as a documented alternative and validation reference. For a Gaussian basis,
`grad_z phi = -(z-q)/ell^2 * phi`. Because both positions and displacements use
the same scalar normalization, `grad_x u = grad_z v`; do not add an extra scale.

Report patch-support approximation RMS/max residual, center prediction,
support/extrapolation diagnostics and status. For an exactly affine input field,
both projection modes must reproduce the same six parameters. For curved fields,
the two modes can differ; benchmark their initialization errors separately.

Respect an existing ROI mask for requested patches and observations using its
explicit coordinate frame. A reference mask is not automatically a target mask.
Reject or flag patches crossing known invalid support according to a documented
policy. Do not invent crack detection or automatically split motion fields.

## 7. API, frontend integration and result semantics

Provide a Python-first API equivalent to:

```python
pairs = ...  # CorrespondenceSet supplied by any supported matcher adapter
model = matcher.fit_initialization_field(pairs, basis='gaussian_rbf', options=options)
initialization = model.initialize_patches(seed_points, patch_size=128,
                                        projection='support_fit')
```

Names may follow existing conventions. Keep fitting reusable for multiple seed
sets/patch sizes without rerunning feature extraction. Do not silently cache the
same field for a different target image. Expose evaluation at arbitrary points,
optional analytic gradients, coefficient serialization and safe resource cleanup.

For each requested patch return its original ID/center, six parameters, finite
status, support diagnostics, approximation residual and acceptance reason.
Always mark these as `estimated_from_field`, not directly measured matches.
Return failed patches too, with explicit unavailable parameters rather than a
fake zero displacement. Separate `computable` from validated IC-GN convergence;
field-fit success alone must not claim a patch converged or is correct.

Model metadata includes requested/effective basis, centers/normalization,
hyperparameters and search bounds, robust fit status, condition/solver diagnostics,
input counts, full timings, execution backend and any CPU transfers/fallbacks.
Preserve matcher scores separately from projection residuals.

Browser native matchers must use the same projection contract and shared WGSL.
Expose basis selection independently of extraction method and export all six
parameters in CSV/JSON. External ONNX matchers currently available only in Python
need Python adapters; this task does not require porting their inference to the
browser. Document actual frontend support rather than pretending it is identical.
Store defaults, parameter ordering and option mappings in the shared contract.

## 8. WebGPU execution and numerical requirements

Implement basis evaluation, residuals/IRLS weights, batched reductions, bounded
linear solves, field evaluation and per-patch projection on GPU where feasible.
Keep intermediate N-by-K arrays/weights resident and reuse buffers and pipelines.
Batch model candidates when it improves measured performance within a bounded
memory budget. Respect device limits and release temporary resources.

Avoid an all-source/all-target correspondence matrix and a mandatory N-by-N RBF
system. Report the memory complexity and measured/estimated allocation sizes.
Use normalized coordinates and FP32 GPU accumulation with stable reduction order;
compare with independent float64 CPU fitting. If normal equations are used,
acknowledge squared conditioning and monitor solve residual/rank. A Cholesky failure,
non-finite coefficient or condition failure must produce an explicit diagnostic
and documented fallback, not silently clamped coefficients.

An initial CPU solve/orchestration fallback may be used while validating the GPU
path, but cannot be described as the final GPU implementation. Report how many
host synchronizations occur per fit/IRLS iteration. Small candidate selection and
metadata decisions may stay on CPU. Do not introduce required CUDA, PyTorch,
TensorRT, Rust or compiled Python extensions. Keep pip installation viable.

Canonical shaders remain in `python/src/wgpu_matching/shaders/`; no manual copies.
Python and browser parity must cover the same fixed parameters and correspondence
fixtures. Keep feature inference time separate from projection time and total time.

## 9. Required validation and acceptance experiments

First test the projection independently of detectors using known point pairs:

- exact translations, rotation, shear and affine stretches including 60%;
- smooth non-affine fields and continuous localized deformation;
- varying density, clustered features, missing regions and boundary extrapolation;
- noise and isolated/clustered outliers, including coherent false displacement groups;
- empty, insufficient, duplicate and collinear inputs; deterministic repeatability;
- coordinate scaling, crop/padding transforms, gradients and patch parameter ordering.

Validate analytical derivatives against finite differences and independent formulas.
Check pure affine reproduction even with RBFs enabled and regularization nonzero.
As initial numerical targets on well-conditioned exact-affine fixtures, require
native-coordinate prediction error <0.01 px and gradient error <1e-4. These are
solver correctness targets, not promises about measured image features. Document
and investigate any unmet target instead of silently loosening it.

Then compare all available matchers on Live-DIC frames 040/100/199 and tensile 60%,
using identical visible seeds, with projection disabled, affine, and selected RBF.
Include synthetic continuous heterogeneous fields so affine-only cases do not
artificially favor a global affine model. Report for all requested patches:
coverage, accepted counts, error <=2 px and strictly <5 px, median/RMSE/P95/max,
false acceptances, gradient errors, and error of each affine initializer across
its patch support. Separate accepted-only precision from total correct coverage.

Reuse the AdrienVariational TIFF pair and existing mask read-only as an additional
real-data case. Match residuals and visual alignment are diagnostics, not truth.
Do not transfer its coordinates or fit settings silently from 640x512 network
inputs to the original 7920x6004 images. Retain raw observations for reproducibility.

Measure warmed repeated timings on Intel LNL (or explicitly identify a substitute):
feature extraction/matching, correspondence conversion/transfers, selection,
final robust fit, patch projection, total; report median and dispersion. Include
fixed-parameter and auto-selection costs. Sweep N and center counts with bounded
memory. GPU acceleration is an empirical result, not an assumed deliverable number.

If an existing IC-GN solver is accessible without modifying another project,
compare convergence, final analytic error and converged windows per second. If
not accessible, deliver the complete six-parameter export and mark convergence
validation as pending. Projection acceptance alone does not establish DIC reliability.

## 10. Implementation order, documentation and completion

1. Audit raw-correspondence access and define/test adapters and shared conventions.
2. Implement independent CPU reference and robust affine projection.
3. Implement Gaussian RBF + affine component, fixed settings and patch projections.
4. Implement bounded deterministic selection and preserve its diagnostics.
5. Execute GPU projection with shared shaders and validate Python/browser parity.
6. Integrate selectors/exports and benchmark all actual method adapters.
7. Publish an English Sphinx/Diataxis explanation, examples and dated reports under
   `results/projection/`, with raw correspondence fixtures or reproducible extraction.

Proposed modules: Python `projection.py`, matcher adapters, shared projection WGSL,
a browser projection module, and focused tests/benchmarks. Adjust names to the
existing architecture; do not duplicate the projection algorithm for each matcher.

Completion requires both bases, automatic RBF selection, all six patch parameters,
explicit model/failure provenance, real GPU measurements, analytical validation,
existing matcher non-regression and documented frontend coverage. Maintain a status
checklist naming what remains unimplemented/unmeasured. Do not claim to have solved
cracks, to have reproduced FG-GMM exactly, or to have globally optimal RBFs.

## 11. Scientific source and deliberate adaptations

Primary source: Zhang, Yan and Li, *Improved initial guess with semi-subpixel level
accuracy in digital image correlation by feature-based method*, Optics and Lasers
in Engineering 104 (2018), 149–158, DOI 10.1016/j.optlaseng.2017.05.014 (online 2017).
Local PDF: `1-s2.0-S0143816617303111-main.pdf`.

Retain its useful separation: feature correspondences -> continuous initializer
and derivatives -> independent IC-GN. This contract deliberately substitutes modern
matcher adapters, a bounded Gaussian basis with affine component, robust regression
and selected hyperparameters for its ORB/FG-GMM full soft-assignment EM pipeline.
It also adds finite-support patch projection. These are adaptations to validate,
not evidence of numerical or algorithmic equivalence with the paper.

The paper's synthetic final error of 0.021 px is after IC-GN, not initialization
alone. Its larger experimental example reaches about 11.5% elongation. It does
not validate our 60% cases or cracks. Do not copy its heuristic rejecting large
absolute displacements (about one tenth of image size).

## Implementation progress (November 2026)

- `python/src/wgpu_matching/projection.py` now contains a float64 CPU reference and
  an opt-in `WGPUMatching.fit_initialization_field` path. Affine/RBF IRLS normal
  systems, basis construction, residual/robust-weight updates, field evaluation,
  and support-fit/Taylor patch projection run in canonical WGSL shaders. Pipelines
  are cached per live device and released when the matcher closes.
- `CorrespondenceSet` retains matcher identity, unique IDs, native coordinate
  mapping metadata, scores, acceptance/rejection reasons and measurement kind.
  It accepts pairs from any method; `from_match_field` consumes direct targets
  only. HORG optionally exposes its actual nearest-descriptor pairs, including
  Lowe-ratio rejections. Do not turn HORG's already fitted local affine estimates
  into synthetic training matches.
- `CorrespondenceSet.from_preprocessed_pairs` applies separate x/y source and
  target scale, padding and crop mappings. The public projection API filters to
  accepted, direct, finite, in-image observations; keeps excluded rows and
  incompatible duplicate sources in diagnostics; and returns six parameters,
  support error, robust weights, residuals and original correspondence IDs.
- The Gaussian-RBF candidate family and affine baseline use deterministic spatial
  cross-validation in NumPy float64. The final robust solve and patch projection
  are GPU-resident after host-array upload. Current GPU IRLS uses a fixed scale
  estimated by a float64 pilot, FP32 normal equations, and an explicit convergence
  flag/delta; non-convergence is reported. Standard IRLS (zero damping) is the
  default; optional update damping is shared by the CPU reference and WGSL.
  Same-device GPU-buffer inputs, GPU-side cross-validation/scale estimation, and
  a condition estimate are not implemented yet.
- `matcher-contracts.json` stores projection defaults and scalar bounds. The
  Sphinx how-to and API reference describe adaptation for HORG, direct
  MatchFields, and external resized-image matchers. Browser matchers still do not
  expose the projection operation or its six-parameter CSV output.
- Analytic tests cover affine/RBF selection, patch parameter ordering, measured
  versus inferred provenance, anisotropic resize/padding/crop restoration, and
  hardware WebGPU affine and Gaussian-RBF fits. Python test suite: 39 passed on
  Intel Graphics LNL. Browser/Python DeepMatching-HORG-CPM parity also passes.
  The GPU affine smoke recovered patch gradients within 5e-5. The synthetic
  heterogeneous benchmark is in `results/projection/gpu_projection_synthetic_20261107.md`.
  Its RBF fit reached 432/432 points under 5 px, but the 20-iteration IRLS did not
  meet its convergence tolerance; report this limitation with the accuracy.
- On 2026-11-07, Ruff passed, Sphinx built with warnings treated as errors,
  browser/Python matcher parity passed, and a wheel was built and installed into
  an isolated target to confirm both projection WGSL shaders are packaged.
- Still pending: move spatial RBF cross-validation and scale estimation from
  NumPy to WGSL; test projection of HORG and other extracted correspondences on
  experimental frames; compare all available matchers on frames 040/100/199 and
  tensile 60%; measure downstream IC-GN convergence; complete browser projection
  and CSV parity; and characterize allocation, conditioning and failure behavior
  across adapters. Do not call the scientific qualification complete.

## Native ONNX/WebGPU performance work (November 2026)

- The 2026-11-06 audit measured EfficientLoFTR at 640x360 on Intel Graphics LNL.
  Its original profile had eight CPU `ScatterND` border-mask updates, each
  materializing a 14.7 MB boolean tensor. Never interpret ORT event durations
  as GPU hardware timestamps or claim a model is GPU-resident from its selected
  provider alone.
- An opt-in rewrite now lives in
  `python/benchmarks/optimize_onnx_webgpu.py`; the comparison harness is
  `python/benchmarks/compare_efficientloftr_border_mask.py`. It replaces those
  updates with one same-shape WebGPU `And`, specializes the graph to a padded
  640x384 NCHW input, and changes the binary-mask cast from int8 to float32 so
  `ArgMax`/`ReduceMax` can run on the native WebGPU EP. The rewrite is not part
  of the default API and must be regenerated for any other tensor size. The
  14.7 MB constant is a deliberate memory trade-off: chained broadcast `And`
  operations gave incorrect outputs on the installed provider and were rejected.
- **Measured 2026-11-07:** original and rewritten output coordinates/confidences
  are bit-identical on Live-DIC frames 040/100/199 and synthetic 60% tensile at
  640x360. Three alternating warmed repetitions gave 1.06x-1.18x speedup. The
  frame-100 diagnostic profile reduced CPU node events from 1,256 to 92 and
  `MemcpyToHost` volume from 33.45 MB to 63 KB; the event duration remained
  about 1.05 s, so synchronization still dominates and that event duration is
  not copy bandwidth. This is not a 4K network run or a fully GPU-resident
  pipeline. Report and data:
  `results/features/efficientloftr_border_mask_optimization_20261107.md` and
  `.json`.
- The RaCo/ALIKED profile found 37 zero-padded bilinear `GridSample` nodes on
  CPU and two border-padded nodes. `decompose_raco_grid_sample` in
  `python/benchmarks/optimize_onnx_webgpu.py` rewrites only the zero-padded
  nodes as bilinear coordinate/weight arithmetic plus GPU `GatherND`; the
  dynamic input shapes are retained. It is opt-in with
  `--decompose-grid-sample` and composes with the existing `--decompose-selu`
  transform in `experiment_aliked_lightglue.py`.
- **Measured 2026-11-07:** on the 640x360 network input, three unprofiled
  repetitions gave 1.34x–1.43x speedup on frames 040/100/199 and tensile 60%.
  Every per-seed affine initialization and analytic error was identical to the
  SELU-only baseline. Across eight profiled inferences, CPU `GridSample` time
  fell 457.5→1.4 ms, summed CPU node duration 622.1→176.9 ms, estimated
  `MemcpyToHost` bytes 2.092 GB→86.8 MB and `MemcpyFromHost` bytes
  941.0→613.9 MB. The CPU event count increased because one operator expands
  to many primitive nodes. ORT trace durations are not hardware GPU timestamps;
  these reductions do not establish a GPU-resident pipeline. The optional
  stable `LogSoftmax` decomposition was benchmarked and removed because it
  slowed all four production cases despite lowering estimated transfer volume.
  Full report and data:
  `results/features/raco_gridsample_webgpu_optimization_20261107.md` and `.json`.
- Remaining P1 work: investigate CPU `Expand`, the two border-padded
  `GridSample` nodes, and residual ORT synchronization/transfers. Keep rewrites
  opt-in until per-seed DIC quality and unprofiled performance are measured.
  Do not start native-4K local refinement or report the <1 s end-to-end target
  as met until those stages have a bounded-memory design and a measured
  comparison. No shared allocation between ONNX Runtime's WebGPU provider and
  `wgpu-py` has been demonstrated.

## DaD, DeDoDe v2 and speckle-specific detector progress (November 2026)

- Optional Python reference adapters now expose official DaD, DeDoDe-L-v2,
  and DeDoDe-B models through a common floating-point `FeatureSet` contract.
  The experiment records checkpoint SHA-256 and upstream revisions. PyTorch
  and the upstream model packages are optional; neither is a core package
  dependency. DaD + DeDoDe-B is an experimental combination, not an official
  end-to-end model.
- **ONNX/WebGPU detector path validated:** `export_learned_detector_onnx.py`
  exports keypoint-only DaD and DeDoDe-v2 graphs with dynamic image dimensions
  and a fixed top-K capacity. `OnnxWGPUKeypointDetector` runs them through
  ONNX Runtime WebGPU without importing PyTorch at inference. The custom
  keypoint-selection export was checked against the official PyTorch detector;
  DeDoDe-v2 coordinates match within 4.4e-5 px on the tested random tensor,
  while DaD's selected set has near-tie ordering differences (1012/1024
  reference points within 1 px; scores differ by <5e-7). A generated speckle
  smoke pair yielded 1019/1024 DaD and 1024/1024 DeDoDe-v2 points within 1 px
  under one-way nearest-set comparison. This metric is not a one-to-one
  assignment and is only an export parity check, not a matching-quality result.
- On Intel Graphics LNL, keypoint-only ONNX inference at 640x360 with K=1024
  measured 0.120 s for DaD and 0.405 s for DeDoDe-v2 (five warm runs; ONNX
  `session.run` wall time). ONNX Runtime profiles attributed 120.153 ms and
  409.067 ms respectively to WebGPU nodes, versus 0.516 ms and 1.034 ms to CPU
  node kernels. The returned keypoint and score arrays total 12 KiB. Profile
  node durations do not establish buffer sharing with `wgpu-py`; ONNX Runtime
  returns CPU NumPy arrays. Peak memory was not measured. These timings cover
  detector inference only, not preprocessing, DeDoDe-B, matching or RANSAC.
- `experiment_learned_detectors.py` accepts `--dad-onnx` and `--dedode-onnx`
  to compare those detectors with the common DeDoDe-B descriptor and WGSL
  affine RANSAC. Its per-detector timings separate preprocessing, ONNX run,
  coordinate processing and total detection. ONNX graphs have a fixed K, so
  the candidate pool must not exceed the exported capacity. This comparison
  still uses a reduced model input and CPU reference descriptor/matcher stages;
  it is not native 4K inference or an all-GPU pipeline.
- **60% matching-path comparison:** on the same 3840x2160 affine synthetic
  pair, both PyTorch and ONNX paths use 640x360 detector inputs, K=1024,
  budget=512, DeDoDe-B on CPU, mutual cosine matching on CPU, and WGSL affine
  RANSAC. Across three measured repeats after one warm-up, DaD gives 6/16
  accepted seeds under 5 px through both paths. Detector pair time changes
  from 1.447 s (PyTorch) to 0.391 s (ONNX/WebGPU), and total pair time from
  3.645 s to 2.969 s. DeDoDe v2 gives 0/16 under both paths; detector pair time
  changes from 3.709 s to 1.056 s, while total pair time changes from 6.302 s
  to 3.708 s. The raw repeated report is
  `results/features/detector_onnx_webgpu_20261107/common_descriptor_60pct_repeats3.json`;
  earlier single-repeat raw files are also preserved. The summary is
  `results/features/detector_onnx_common_descriptor_60pct_20261107.md`. The
  paths do not demonstrate an improvement in detector/matcher quality; ONNX
  preserves these selected outcomes while speeding detector inference.
- **Higher point budgets:** K=8192 ONNX graphs were exported and checked
  against official PyTorch selection. DeDoDe v2 matches 8192/8192 keypoints
  within 1 px and scores within 6.55e-11; DaD's near-tied ONNX order yields
  8130/8192 within 1 px and 8187/8192 within 5 px, while custom selection vs
  official PyTorch is within 0.1 px for all 8192. On the 60% affine pair, with
  candidate pool K=8192, budget 4096 yields 15/16 correct <5 px for DaD and
  14/16 for DeDoDe v2 through both execution paths. At budget 8192 with a
  1x candidate pool, both methods reach 16/16 correct <5 px through both
  paths. The high-budget run uses one measured repeat after warm-up; detector
  input is still 640x360 and descriptor/mutual matching remain CPU. Raw data
  are in `results/features/detector_onnx_webgpu_20261107/` and summarized by
  `results/features/detector_onnx_high_budget_20261107.md`.
- **60% cross-method context:** on the common 16 analytic seed centres, the
  high-budget ONNX runs give DaD 15/16 correct <=2 px and 16/16 <5 px; DeDoDe
  v2 gives 16/16 at both thresholds. This is compared contextually with
  DeepMatching (8/16 <5 px at 16.405 s), XFeat (8/16 at 0.553 s), EfficientLoFTR
  (14/16 at 1.501 s), full HORG (0/16 at 0.146 s), and the learned DIC
  detector (16/16 at 7.006 s). Timing scopes and input resolutions differ, and
  this remains a globally affine synthetic stress test, not IC-GN convergence
  or heterogeneous-motion validation. Summary:
  `results/features/detector_modern_matchers_60pct_20261107.md`.
- **Progressive frames:** on frames 040/100/199 with three repeats, K=1024,
  and budget=512, the ONNX path preserves DaD's correct <5 px affine seed
  counts (21/32, 19/28, 17/28 under global top-K) and DeDoDe v2's counts
  (12/32, 1/28, 0/28). Full-pair times improve from 3.406/4.015/4.169 s to
  3.051/2.884/2.960 s for DaD, and from 6.719/6.197/6.391 s to
  3.522/3.510/3.604 s for DeDoDe v2. This is reduced-resolution detector
  inference with CPU descriptor/matching stages, not native UHD inference.
  Details and raw per-seed results are in
  `results/features/detector_onnx_progressive_20261107.md` and
  `results/features/detector_onnx_webgpu_20261107/common_descriptor_progressive_repeats3.json`.
- `experiment_learned_detectors.py` compares the official detectors, HORG/SIFT,
  and both DIC detector variants with the same DeDoDe-B descriptor, mutual
  cosine reference matching, and WGSL local-affine RANSAC. On its measured
  3840x2160 60% affine stress case, DaD + DeDoDe-B accepted 16/16 affine seeds,
  all 16 under 5 px, at about 11.17 s/pair. DeDoDe v2-L + B accepted 0/16;
  DIC tensor + B 0/16; HORG/SIFT + B 2/16; learned DIC + B 5/16 in the selected
  global-top-K candidate-pool variant. DaD inference, DeDoDe-B description and
  mutual matching in this result ran on CPU; WGSL RANSAC is not evidence of a
  GPU-resident feature pipeline. The separate ONNX/WebGPU path preserves these
  seed-level results at a 512 point budget on the same affine stress pair,
  while reducing detector and total-pair time; this does not qualify it on the
  progressive or heterogeneous datasets.
- The DIC structure-tensor baseline, ROI-support checks, subpixel peak
  localization, NMS and spatial quota selection run in WGSL. A 169-parameter
  score CNN also runs in WGSL; its training and PyTorch oracle are optional.
  At 3840x2160, the detector pair call takes about 0.13 s with 8192 selected
  points on Intel Graphics LNL/Vulkan. This is detector-only and excludes a
  descriptor, matching and RANSAC. Learned WGSL score parity with PyTorch has
  maximum absolute error 1.788e-6 in the recorded 4K check.
- The synthetic generator now covers translation, rigid rotation, affine,
  smooth heterogeneous and piecewise-discontinuous motion. Discontinuous pairs
  include an explicit invalid gap; source points near its seam and target
  points in the gap are excluded from repeatability and equivariance losses.
  A test checks determinism, the analytic displacement jump and gap masks.
- A separate 10-epoch model run used 640 training pairs, including 111
  discontinuous pairs. On 24 held-out 256x256 pairs, its WGSL detector achieved
  1403/4756 repeatable points at <=1 px, 2711/4756 at <=2 px and 3715/4756 at
  <=5 px; pair latency was 4.05 ms (about 247 detector pairs/s) on Intel LNL.
  Re-evaluating the previous checkpoint on the same generated pairs yielded
  2608/4816 at <=2 px and 3686/4816 at <=5 px. On the five discontinuous cases,
  <=2 px changed only from 613/954 (64.3%) to 600/924 (64.9%). The added
  training therefore has only a small exploratory gain for discontinuities;
  varying visible-point denominators are shown in the report. No descriptor,
  matching or IC-GN convergence was measured in this detector-only study.
- The same discontinuity-aware score checkpoint was checked against PyTorch
  at 3840x2160 on Intel Graphics LNL/Vulkan with 8192 selected points: maximum
  absolute score difference 2.414e-6, median dispatch/readback 64.28 ms and
  256 KiB returned. Full timing and parity data:
  `results/features/dic_detector_discontinuous_wgsl_parity_4k_20261107.json`.
- A broader shared-descriptor report covers progressive frames 040/100/199 and
  60% tensile. It records spatial quota and candidate-pool scope, per-seed
  affine analytic errors, detector/descriptor/match/RANSAC timing, raw results,
  and the external model revisions. The DIC detector's “global top-K” acts on
  its WGSL-generated 16x9 per-tile candidate pool; it is not a global top-K over
  the full UHD response map. Do not conflate detector repeatability, raw
  descriptor matches, accepted affine seeds, correct coverage, or IC-GN
  convergence.
- A separate 60% budget sweep now covers 512, 1024, 2048, 4096 and 8192 points
  using the discontinuity-aware DIC checkpoint and a 2x candidate pool. With
  all 16 analytic seeds visible, DaD + DeDoDe-B rises from 9/16 correct accepted
  seeds at budget 512 to 16/16 by 1024; it reaches 0.46 px RMSE / 0.81 px P95
  at 8192 in 9.92 s. DeDoDe v2-L + DeDoDe-B reaches 0/16 at budgets 512/1024,
  3/16 at 2048, 15/16 at 4096 and 16/16 at 8192 (17.14 s). DIC learned WGSL +
  DeDoDe-B reaches 8/16 at 1024, 15/16 at 2048, then 16/16 correct at <=2 px
  and <5 px from 4096; at 8192 its accepted RMSE/P95 is 0.38/0.61 px in
  7.01 s. DIC tensor-only reaches 7/16 accepted at 8192 (5/16 correct <5 px),
  while HORG/SIFT + B plateaus at 2/16 from 1024 onward. Detector/descriptor
  timing, matching and RANSAC are reported by stage; the descriptor and mutual
  matching remain CPU references. This is one synthetic, globally affine
  stress case, not heterogeneous or experimental validation and not IC-GN
  convergence. Summary and raw data:
  `results/features/detector_budget_sweep_60pct_20261107.md`,
  `results/features/detector_common_descriptor_60pct_budgets_low_20261107.json`,
  and `results/features/detector_common_descriptor_60pct_budgets_20261107.json`.
- Current docs and reports:
  `results/features/detector_method_comparison_20261107.md`,
  `results/features/detector_common_descriptor_20261107.md`,
  `results/features/dic_detector_native_4k_20261107.md`,
  `results/features/dic_detector_discontinuous_training_20261107.md`, and
  their adjacent JSON data. The trained `.npz` weights remain ignored and must
  be regenerated or supplied explicitly.
- Still required before scientific qualification: test high point budgets on
  progressive frames 040/100/199 and on heterogeneous motion; test
  detector/descriptors on rotated and spatially localized finite-element
  fields; implement a matching
  path for DIC-Detector candidates if they prove useful; expose the methods in
  the browser UI; and measure IC-GN convergence. The present learned model's
  score target is a structure-tensor proxy plus score equivariance, not
  descriptor uniqueness or IC-GN success supervision. Do not claim these
  methods are qualified for experimental cracks or Live-DIC initialization.
- **Verification on 2026-11-07:** full Python suite 55 passed;
  `ruff check python` passed; Ruff format checks pass for the Python files
  changed in this detector/ONNX work (the repository-wide format check still
  flags 19 other existing Python files). Sphinx HTML built from a clean output
  directory with `-E -W` under the RTD theme, including the ONNX result pages.
  A pure-Python wheel was built and installed to a clean target earlier; the
  feature adapter and canonical learned WGSL loaded from it without importing
  PyTorch. The K=8192 ONNX adapter was separately smoke-tested in an environment
  without PyTorch: it returned 8192 points and 96 KiB of coordinate/score data
  for each detector.

## Scientific documentation review (2026-11-07)

The explanation path now starts at `docs/explanation/detector-principles.rst`
and covers DaD, DeDoDe v2/B, tensor and learned DIC scores, XFeat, RaCo, ALIKED,
LightGlue and EfficientLoFTR, with primary-source links and repository scope.
Keep training objectives distinct from inference scores and proposed features.
The DIC score model has one saliency output, not a calibrated IC-GN reliability
head. How-to commands moved out of the API reference; a weight-free detector
tutorial provides the first practical outcome. Benchmark definitions now state
that common-detector totals sum stage medians; do not call them measured
end-to-end medians. Historical browser/Python DeepMatching statistics must not
be combined across search-radius protocols.

## Abbreviations in documentation

Every technical abbreviation must be defined independently on each rendered
page, including generated reports and API prose. The canonical definitions
are in `docs/acronyms.json`; `docs/_ext/define_acronyms.py` expands first prose
uses and adds definitions for abbreviations used in headings. Later uses have
HTML tooltips. Preserve literal code, mathematical symbols and file identifiers.
The glossary is in `docs/reference/glossary.rst`; it supplements first-use
definitions and never substitutes for them. Do not invent expansions of
published model names; give their actual paper title or explain the name.

## Complete benchmark refresh before the PDF edition

Re-run every current initializer and detector/descriptor pipeline on shared
progressive frames, the full synthetic 10–60% series, and the masked Adrien
pair. Preserve historical measurements. Use source pixel-centre coordinates,
actual measured pairs (not CPM's fitted seed endpoints), and identical
projection settings for affine versus automatically selected affine/Gaussian
RBF fields. Record each effective basis and fit convergence. Ground truth may
select visible evaluation centres and compute errors, but must never select
correspondences, tune projection coefficients or decide the model basis.
A 60% accuracy pass requires all visible seeds to have finite estimated
displacements with error strictly below 5 px. Show coverage, false accepted
points and <=2 px accuracy independently. Adrien has no analytic truth and
cannot receive a synthetic accuracy verdict. Produce reference/corrected GIFs
for every method/case/basis, including failures. Any visualization obtained
from rejected pairs must be explicitly diagnostic and excluded from pass
counts. Record model hashes, input dimensions, stage timings and full pipeline
wall-clock trials. Refresh the English online report before generating its PDF.

The initial raw-pair projection diagnostic showed that IRLS alone can converge
to grossly wrong fields under descriptor outliers. The primary refreshed
protocol therefore preserves the pre-existing geometric-verification stage:
local WGSL affine RANSAC (radius256, residual3, minimum6/8 supports, fraction0.2),
then WGSL filtering of ORIGINAL observed pairs by those local models. Never
substitute predicted or interpolated endpoints as new observations. Preserve
raw failures as an explicitly separate ablation. A deterministic source-space
capacity reduction caps the existing GPU projector at8192 pairs, with all
pre-reduction pairs and chosen identifiers archived. This is not an intrinsic
matching failure when a dense detector exceeds the solver's input limit.


### Refreshed qualification protocol and evidence

The current campaign is `results/qualification/all_methods_20261107/`:
11 pipelines, 10 cases, two projection requests, and three timed complete
affine calls after warm-up per method/case. It includes current DaD,
DeDoDe v2-L, tensor and learned DIC detection with a common DeDoDe-B descriptor.
The independent audit checks 220 result rows, 264 two-frame animations,
all analytic centre errors, raw-pair counts, repeat digests and runtime-source
fingerprints. The 330 timed affine calls use decoded images, both images
processed per call, no prepared-reference cache. RBF selection is a separate
single fit on the final observed pair set, not another complete FPS trial.

The controlled raw-projection ablation uses the EXACT archived upstream pairs
and centres from the main run: `results/qualification/raw_projection_ablation_20261107/`.
It contains 132 records and 176 animations; detector extraction is not rerun.
Across the 99 analytic affine comparisons, common verification improves the
correct-below-5 count in 31 cases, preserves it in 68 and reduces it in none.
At 60%, verified affine fields pass 16/16 below 5 for eight pipelines; native
HORG and CPM do not provide a field, and HORG/SIFT+B recovers only 5/16.
DeepMatching, RaCo, DaD+B, DeDoDe v2+B and learned DIC+B give16/16 at most 2;
tensor DIC+B gives14/16, EfficientLoFTR10/16 and XFeat0/16 at that stricter bound.
These are inferred affine-field initializations, NOT direct matches at every
seed or IC-GN convergence. Do not merge them with old local-initializer counts.
The automatic affine/Gaussian request improves correct coverage in zero of 99
analytic comparisons, preserves it in 97 and reduces it in 2. Model selection
predicts observed residuals, not physical truth. Preserve non-convergence flags
independently of analytic accuracy; do not relax iteration thresholds to make
accepted fields appear numerically converged. Adrien has no analytic truth.

DeepMatching's extended 1024-original-pixel search radius gives about 17.13 s
on the 60% UHD case and 108.21 s on the 7920x6004 Adrien pair. This is a different
workload from the historical default-radius 1.3 s comparison or sparse-root modes.
Native matcher defaults were not changed. Current common-B description is a
CPU reference and dominates the new detectors' 7.5–8.5 s UHD pair times; do not call this a
pipeline resident on GPU. Tensor/learned scoring and final solves are WGSL,
with host pilots and cross-validation as documented.

Opt-in native observation export is now supported by hierarchical DeepMatching
and CPM through `include_correspondences=True`. Ordinary CPM seed endpoints
are fitted and cannot be treated as independent observations. Optional
`fit_local_affine(..., include_affine_models=True)` exports target-map rows
`a1,a2,a0,1,b1,b2,b0,1`; consume only usable models. The canonical optional
model export and measured-pair filter shaders live in the Python package.
Default native outputs and Python/browser parity were checked unchanged.

Sphinx now groups report navigation by study and places raw arrays in a
separate catalogue. JSON previews are limited to 160 lines and CSV previews
to 50 rows, with complete downloads. Compressed arrays/models are downloads,
not raw Markdown browser views. Keep the raw pages marked orphan so thousands
of model records do not fill every page's sidebar. The print root is
`docs/pdf/index.rst`; `tools/build_handbook.py` produces the curated handbook
under ignored `output/pdf/`. Generate its evidence from the validated campaign
with `python/benchmarks/prepare_handbook_evidence.py`. Do not print profiler
traces or huge raw arrays as chapters. Preserve the English Diataxis paths,
first-use abbreviation definitions, a separate convergence caveat, static
native-pixel crop pairs in print and every animated outcome in the web edition.

### Affine-normalization descriptor diagnostic (8 November 2026)

Do not continue contrastive training of the compact descriptor until the
geometry-normalization diagnostic is understood. The experiment
`python/benchmarks/benchmark_oracle_affine_descriptor_retrieval.py` and its
report `results/descriptors/oracle_affine_compensation_20261108/README.md`
compare zero-mean normalized cross-correlation, HORG-64, the learned compact
DIC convolutional neural network descriptor, and DeDoDe-B on identical source
queries and target candidate pools. An analytic global or query-local affine
map rectifies the target image only in the oracle arm; it never chooses or
pre-ranks a target candidate. This is an upper-bound diagnosis, not a practical
initializer.

On 3840 × 2160 synthetic images with 8,192 target candidates, oracle
compensation raised top-1 retrieval within 5 pixels from 0/486 to 474/486 for
the compact descriptor under rigid motion, and from 5/431 to 290/419 for the
affine field. DeDoDe-B changed from 0/486 to 485/486 under rigid motion and
from 218/431 to 413/419 under affine motion. The heterogeneous local-oracle
sample returned 28/30 compact-descriptor and 27/30 DeDoDe-B top-1 matches
within 5 pixels, versus 0/29 before compensation; this sample has only 32
queries. Baseline and oracle repeatability denominators differ because
rectification changes valid support coverage. DeDoDe-B uses a 1024 × 576
input; HORG-64 and the compact descriptor use native-resolution pixels. Its
runtime reports both the WebGPU and CPU providers; this is not a claim of a
fully GPU-resident path.

Interpretation: local geometric mismatch is a major contributor, not the only
cause. HORG-64 and zero-mean normalized cross-correlation improve only
modestly, while the learned descriptors recover much of their ranking ability
with correct geometry. The oracle uses synthetic truth and does not establish
performance on unknown motion or convergence of inverse compositional
Gauss–Newton. The follow-up translation-free support experiment and its
Jacobian-error sweep are recorded below. Do not implement a dense affine bank
from the oracle result alone: combined Jacobian errors, heterogeneous fields,
and image-derived hypotheses remain untested.

Reproduce after exporting the official DeDoDe-B model and providing the
recorded compact-descriptor weights:

```bash
PYTHONPATH=python/benchmarks:python/src python3 \
  python/benchmarks/benchmark_oracle_affine_descriptor_retrieval.py \
  --cnn-weights results/descriptors/dic_descriptor_hardneg_training_20261108/dic_patch_cnn_descriptor.npz \
  --dedode-onnx /path/to/dedode_descriptor_1024x576_k8192.onnx \
  --output results/descriptors/oracle_affine_compensation_20261108/oracle_retrieval_4k.json \
  --width 3840 --height 2160 --candidate-budget 8192 \
  --global-queries 512 --local-queries 32 \
  --families rigid affine heterogeneous --seed 20261108
```

The official model graph and weights are external artifacts, not committed.
Their SHA-256 fingerprints and execution providers are recorded in the JSON
result. See the dated report for sources on affine-equivariant descriptors,
deformation-conditioned descriptor transforms, and feature-guided DIC
initialization.

### Translation-free affine candidate-patch sensitivity (8 November 2026)

The follow-up is implemented by
`python/benchmarks/benchmark_affine_candidate_patch_sensitivity.py` and
summarized in
`results/descriptors/affine_candidate_sensitivity_20261108/README.md`.
For each target candidate `y`, it samples `I1(y + F_hat xi)`: the analytic
translation is never used to construct or select candidate patches. The
analytic Jacobian is still an oracle input, and analytic displacement is used
only to score retrieval and the post-fit error. Candidate detector positions
remain fixed across each Jacobian sweep. No analytic visibility or occlusion
mask is consulted; image-border-invalid transformed supports receive a score
floor and are excluded from ranking.

On three 3840 × 2160 synthetic global-affine pairs, all with 512 source queries
and exactly 8,192 target candidates, 1,002 of 1,536 queries had a detector
candidate within 5 pixels. With the exact Jacobian and no translation oracle,
the compact DIC convolutional neural network (CNN) descriptor rose from
196/1,002 to 823/1,002 correct top-1 retrievals and from 366/1,002 to 901/1,002
top-five retrievals. The corresponding WebGPU random sample consensus
(RANSAC) retained 775 estimates, all within 5 pixels, with zero false
acceptances in these globally affine cases. This is a support-normalization
diagnostic, not a deployable initializer or an inverse compositional
Gauss–Newton convergence result.

One-axis Jacobian perturbations show roughly 80% top-1 retrieval within ±5°
rotation, ±10% independent axial stretch, and ±0.10 shear. At ±10°,
±20% stretch, or ±0.20 shear, top-1 drops to roughly 54–68%; top-five remains
higher. These sweeps do not establish a bank spacing because simultaneous
parameter errors and non-affine fields were not tested. Descriptor recompute
on the current candidate atlas takes a median 1.87 seconds per hypothesis for
this 8,192-candidate workload; its CPU sampler and atlas are diagnostic
prototypes, not production kernels. Do not build a dense bank on this path.

Reproduce the raw benchmark and report with:

```bash
PYTHONPATH=python/benchmarks:python/src python3 \
  python/benchmarks/benchmark_affine_candidate_patch_sensitivity.py \
  --cnn-weights results/descriptors/dic_descriptor_hardneg_training_20261108/dic_patch_cnn_descriptor.npz \
  --output results/descriptors/affine_candidate_sensitivity_20261108/oracle_sensitivity_4k_8192.json \
  --width 3840 --height 2160 --candidate-budget 8192 \
  --global-queries 512 --textures 3 --families affine --sensitivity
PYTHONPATH=python/benchmarks:python/src python3 \
  python/benchmarks/summarize_affine_candidate_patch_sensitivity.py \
  --input results/descriptors/affine_candidate_sensitivity_20261108/oracle_sensitivity_4k_8192.json \
  --output results/descriptors/affine_candidate_sensitivity_20261108/README.md
```

The measurement used Vulkan on Intel Graphics (LNL). Candidate patch sampling
and similarity ranking in this first sweep are CPU-side; HORG/CNN description
and affine RANSAC run through WebGPU. DeDoDe-B is not part of this support
sweep because its current fixed-shape dense-image descriptor API does not
accept candidate-centred affine patches. The preceding full-image oracle
report retains its DeDoDe-B comparison.

### Combined Jacobian errors and image-only rotation hypotheses (8 November 2026)

The next diagnostic is implemented in
`python/benchmarks/benchmark_affine_hypothesis_search.py`; the raw 4K data and
summary are in
`results/descriptors/affine_hypothesis_wgpu_20261108/`. Unlike the CPU atlas
prototype above, the compact descriptor's candidate-centred affine support is
now sampled directly from the native target image in the canonical WGSL
shader. This path is exposed by `describe_dic_cnn_keypoints(...,
support_jacobian=...)` and is checked against the PyTorch sampling grid.

The combined-error experiment uses three independent global-affine 4K pairs,
512 fixed queries and 8,192 fixed candidates per pair. An 81-point grid
combines rotation errors (-10°, 0°, +10°), independent axial stretch errors
(-20%, 0%, +20%) and shear errors (-0.20, 0, +0.20). The repeatable denominator
is fixed at 751/1,536 queries across every hypothesis. Top-1 retrieval falls
from 80.4% among repeatable queries for the exact Jacobian to 11.9% when all
four parameters are perturbed; the pooled rate falls as more parameters are
wrong. At the exact Jacobian, the direct consensus retains 551 pairs, including
6 false acceptances under the 5-pixel truth check. The fitted affine predicts
all query centres correctly only because these test fields are globally
affine; this is not a result for heterogeneous fields.

The image-only rigid-rotation bank tests 18, 36 and 72 angles over a full turn.
Its hypothesis selection uses only reciprocal matching and WebGPU affine
consensus, not the true angle or translation. Across three independent rigid
4K pairs, top-1 retrieval among repeatable queries is 58.8%, 78.7% and 81.3%,
respectively, versus 88.1% for exact-Jacobian oracle scoring and 0% without
support compensation. All retained direct matches in this sample are within
5 pixels, while fitted rigid fields predict all 1,536 query centres correctly.
Again, reconstructed centres are not direct correspondences. Median bank time
per pair is 2.36 s, 4.72 s and 9.36 s, so a 72-angle exhaustive pass is too
slow for the target pipeline. The 36-to-72 increase provides only 24 additional
top-1 matches among 941 repeatable queries in this sample.

The detector/visibility split is also recorded after the fact for the affine
cases: 603/1,536 mapped centres are outside the image, 182 are inside but have
no detected target candidate within 5 pixels, and 751 have such a candidate.
Analytic visibility is never used to filter hypotheses or candidates.

This remains a diagnostic on three globally affine and three globally rigid
synthetic pairs, with no IC-GN qualification and no heterogeneous field test.
The exhaustive rotation-bank diagnostic still reads descriptor arrays to NumPy
for per-query max-bank ranking, then uploads them to the ordinary WGSL matcher.
The separate coarse-to-fine path below keeps descriptor vectors resident
through reciprocal matching. Do not describe that path as fully device-only:
compact matches are read back before geometric consensus is selected.

Reproduce the campaign with:

```bash
PYTHONPATH=python/benchmarks:python/src python3 \\
  python/benchmarks/benchmark_affine_hypothesis_search.py \\
  --mode both \\
  --cnn-weights results/descriptors/dic_descriptor_hardneg_training_20261108/dic_patch_cnn_descriptor.npz \\
  --output results/descriptors/affine_hypothesis_wgpu_20261108/combined_and_rotation_4k.json \\
  --width 3840 --height 2160 --candidate-budget 8192 --query-budget 512 --textures 3
PYTHONPATH=python/benchmarks:python/src python3 \\
  python/benchmarks/summarize_affine_hypothesis_search.py \\
  results/descriptors/affine_hypothesis_wgpu_20261108/combined_and_rotation_4k.json \\
  --output-dir results/descriptors/affine_hypothesis_wgpu_20261108
```

### Resident coarse-to-fine rotation search (8 November 2026)

`python/benchmarks/benchmark_affine_hypothesis_search.py` now includes a
resident coarse-to-fine path. It uploads each native image and the compact CNN
weights once, reuses the per-point convolution workspace across support
hypotheses, keeps source and target descriptor vectors in GPU buffers, and
performs reciprocal matching with the WGSL masked nearest-neighbour kernel.
Only compact reciprocal indices and similarity scores are transferred before
the existing affine consensus stage. The exhaustive diagnostic remains
unchanged and still reads descriptors for its per-query max-score analysis.

On three fixed 3840 × 2160 rigid speckle pairs, with 512 source queries and
8,192 target candidates, the resident search uses 18 coarse angles, local
5-degree refinement, then full verification of two hypotheses. It retained
800/800 direct matches correctly at the 5-pixel criterion, with no false
acceptances, out of 941 repeatable source queries. Median search time across
the three pairs was 0.887 s; median total pair time, including feature
detection and source descriptor preparation, was 1.373 s. Each pair was run
once in this pass, so these medians are across three different pairs rather
than repeated calls on one pair. The earlier
coarse-to-fine path took 1.077 s for search and 1.606 s total on those same
seeds and settings, with exactly the same accepted direct matches. This is a
17.6% search-time reduction and 14.5% complete-pair reduction, not a quality
change. The exhaustive 36-angle bank is a different workload and must not be
used as an apples-to-apples timing baseline.

The global rigid fit predicts all 1,536 evaluated centres within 5 pixels in
these three globally rigid synthetic pairs. This is extrapolated field output,
not 1,536 direct observations. The experiment does not establish performance
on heterogeneous deformation, discontinuities, or IC-GN convergence. Next,
evaluate multiple local models on held-out heterogeneous fields, report direct
matches separately from local interpolation/extrapolation, and do not transfer
a model across a physical discontinuity without independent evidence.

Raw resident measurements and report:
`results/descriptors/affine_hypothesis_resident_20261108/`.

Reproduce the resident comparison with:

```bash
PYTHONPATH=python/benchmarks:python/src python3 \\
  python/benchmarks/benchmark_affine_hypothesis_search.py \\
  --mode coarse --coarse-queries 128 --coarse-candidates 2048 \\
  --cnn-weights results/descriptors/dic_descriptor_hardneg_training_20261108/dic_patch_cnn_descriptor.npz \\
  --output results/descriptors/affine_hypothesis_wgpu_20261108/coarse_to_fine_4k_resident.json \\
  --width 3840 --height 2160 --candidate-budget 8192 --query-budget 512 --textures 3
PYTHONPATH=python/benchmarks:python/src python3 \\
  python/benchmarks/summarize_affine_hypothesis_search.py \\
  results/descriptors/affine_hypothesis_wgpu_20261108/affine_search_4k_optimized.json \\
  --coarse-to-fine-resident results/descriptors/affine_hypothesis_wgpu_20261108/coarse_to_fine_4k_resident.json \\
  --output-dir results/descriptors/affine_hypothesis_resident_20261108
```

import { computeDeepMatching } from "./deepmatching.js";
import { computeHorg } from "./horg.js";
import { computeCPM } from "./cpm.js";

const $ = id => document.getElementById(id);
const state = {
  device: null,
  reference: null,
  target: null,
  field: null,
  settings: null,
  busy: false,
};

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function updateButtons() {
  $("run").disabled = !state.device || !state.reference || !state.target || state.busy;
  $("export").disabled = !state.field || state.busy;
}

async function decodeGray(file) {
  const bitmap = await createImageBitmap(file);
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(bitmap, 0, 0);
    const rgba = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    const pixels = new Uint8Array(bitmap.width * bitmap.height);
    for (let index = 0; index < pixels.length; index++) {
      const r = rgba[4 * index];
      const g = rgba[4 * index + 1];
      const b = rgba[4 * index + 2];
      pixels[index] = Math.min(255,
        Math.floor((2126 * r + 7152 * g + 722 * b + 5000) / 10000));
    }
    return { width: bitmap.width, height: bitmap.height, pixels, name: file.name };
  } finally {
    bitmap.close();
  }
}

function showImage() {
  const image = state.target || state.reference;
  if (!image) return;
  const canvas = $("preview");
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext("2d");
  const rgba = new Uint8ClampedArray(image.width * image.height * 4);
  for (let index = 0; index < image.pixels.length; index++) {
    const value = image.pixels[index];
    rgba[4 * index] = value;
    rgba[4 * index + 1] = value;
    rgba[4 * index + 2] = value;
    rgba[4 * index + 3] = 255;
  }
  context.putImageData(new ImageData(rgba, image.width, image.height), 0, 0);
  if (!state.field) return;

  const acceptedMotions = state.field.matches.filter(match => match.usable)
    .map(match => Math.hypot(match.dx, match.dy)).sort((a, b) => a - b);
  const allMotions = state.field.matches.map(match => Math.hypot(match.dx, match.dy)).sort((a, b) => a - b);
  const motions = acceptedMotions.length ? acceptedMotions : allMotions;
  const referenceMotion = motions[Math.floor(0.9 * Math.max(0, motions.length - 1))] || 0;
  const arrowScale = referenceMotion > 100 ? 100 / referenceMotion : 1;
  const maximumArrowLength = 120;
  context.lineWidth = Math.max(2, Math.min(8, image.width / 500));
  for (const match of state.field.matches) {
    let dx = match.dx * arrowScale;
    let dy = match.dy * arrowScale;
    const displayLength = Math.hypot(dx, dy);
    if (displayLength > maximumArrowLength) {
      dx *= maximumArrowLength / displayLength;
      dy *= maximumArrowLength / displayLength;
    }
    const endX = match.x + dx;
    const endY = match.y + dy;
    context.strokeStyle = match.usable ? "#00a978" : "#ef6949";
    context.fillStyle = context.strokeStyle;
    context.beginPath();
    context.moveTo(match.x, match.y);
    context.lineTo(endX, endY);
    context.stroke();
    const angle = Math.atan2(dy, dx);
    const head = Math.max(4, Math.min(12, image.width / 900));
    context.beginPath();
    context.moveTo(endX, endY);
    context.lineTo(endX - head * Math.cos(angle - 0.45), endY - head * Math.sin(angle - 0.45));
    context.lineTo(endX - head * Math.cos(angle + 0.45), endY - head * Math.sin(angle + 0.45));
    context.closePath();
    context.fill();
    context.beginPath();
    context.arc(match.x, match.y, Math.max(2, Math.min(5, image.width / 1500)), 0, 2 * Math.PI);
    context.fill();
  }
}

function selectedSettings() {
  const algorithm = $("algorithm").value;
  const patchSize = Number($("patch-size").value);
  const rootPatchSize = Number($("root-patch-size").value);
  const stride = Number($("grid-stride").value);
  const downscaleLevels = Number($("downscale-levels").value);
  const searchRadiusPx = Number($("search-radius").value);
  const seedText = $("seed-points").value.trim();
  const horgReduction = Number($("horg-reduction").value);
  const horgMaxFeatures = Number($("horg-max-features").value);
  const cpmGridStride = Number($("cpm-grid-stride").value);
  const cpmLevels = Number($("cpm-levels").value);
  const cpmPropagationIterations = Number($("cpm-propagation-iterations").value);
  const cpmRandomTrials = Number($("cpm-random-trials").value);
  const cpmLocalSearchRadiusPx = Number($("cpm-local-radius").value);
  const cpmSeed = Number($("cpm-seed").value);
  const cpmReciprocalThresholdPx = Number($("cpm-reciprocal-threshold").value);
  const factor = 2 ** downscaleLevels;
  if (!Number.isInteger(patchSize) || patchSize < 8 || patchSize > 128) {
    throw new Error("La fenêtre doit mesurer de 8 à 128 px.");
  }
  const scaledRootPatch = rootPatchSize / factor;
  const rootCellsSide = scaledRootPatch / 4;
  if (algorithm === "deepmatching" && (!Number.isInteger(rootPatchSize)
      || rootPatchSize < 32 || rootPatchSize > 2048 || !Number.isInteger(scaledRootPatch)
      || !Number.isInteger(rootCellsSide) || rootCellsSide < 1
      || (rootCellsSide & (rootCellsSide - 1)) !== 0)) {
    throw new Error("L’échelle racine doit être un multiple du facteur réduit, avec 4, 8, 16… cellules atomiques par côté.");
  }
  if (!Number.isInteger(stride) || stride < 4 || stride > 2048) {
    throw new Error("Le pas de grille doit être un entier entre 4 et 2048 px.");
  }
  if (!Number.isInteger(searchRadiusPx) || searchRadiusPx < 0 || searchRadiusPx > 8192) {
    throw new Error("Le rayon de recherche doit être un entier entre 0 et 8192 px.");
  }
  let seedPoints = null;
  if (seedText) {
    const lines = seedText.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    if (lines.length > 512) throw new Error("Saisissez au plus 512 seeds manuelles.");
    seedPoints = lines.map((line, index) => {
      let coordinates;
      if (line.includes(";")) coordinates = line.split(";");
      else if (/\s/.test(line)) coordinates = line.split(/\s+/);
      else coordinates = line.split(",");
      if (coordinates.length !== 2) {
        throw new Error("Seed ligne " + (index + 1) + " : entrez x ; y en pixels.");
      }
      const parseCoordinate = value => Number(value.trim().replace(",", "."));
      const x = parseCoordinate(coordinates[0]);
      const y = parseCoordinate(coordinates[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        throw new Error("Seed ligne " + (index + 1) + " : coordonnées numériques attendues.");
      }
      return { x, y };
    });
  }
  return { algorithm, patchSize, rootPatchSize, stride, downscaleLevels, searchRadiusPx, seedPoints,
    horgReduction, horgMaxFeatures, cpmGridStride, cpmLevels, cpmPropagationIterations,
    cpmRandomTrials, cpmLocalSearchRadiusPx, cpmSeed, cpmReciprocalThresholdPx };
}

function renderFieldSummary(field, elapsedMs) {
  const accepted = field.matches.reduce((sum, match) => sum + Number(match.usable), 0);
  const fraction = field.count ? 100 * accepted / field.count : 0;
  if (field.algorithm === "CPM-WebGPU") {
    const reciprocals = field.matches.filter(match => match.reciprocal).length;
    const meanGridDistance = field.matches.reduce((sum, match) => sum + match.gridDistancePx, 0) /
      Math.max(1, field.count);
    $("field-summary").textContent = field.width + " × " + field.height + " px · CPM-HOG · grille interne " +
      field.gridWidth + " × " + field.gridHeight + " au pas " + field.workGridStridePx +
      " px · " + field.workGridPointCount + " positions évaluées · rayon " + field.searchRadiusPx +
      " px · pyramide " + field.levels + "/" + field.requestedLevels + " niveaux · " + accepted + "/" + field.count +
      " graines utilisables (" + fraction.toFixed(1) + " %), " + reciprocals + " réciproques · distance moyenne seed/grille " +
      meanGridDistance.toFixed(2) + " px · HOG " + (field.descriptorMs / 1000).toFixed(2) +
      " s, pyramide " + (field.pyramidMs / 1000).toFixed(3) + " s, PatchMatch bidirectionnel " + (field.matchingMs / 1000).toFixed(2) +
      " s, total " + (elapsedMs / 1000).toFixed(2) + " s. Déplacements quantifiés au pas ×" + field.scale + ".";
    return;
  }
  if (field.algorithm === "HORG-WebGPU") {
    const looValid = field.matches.filter(match => match.usable && Number.isFinite(match.localConsistencyPx));
    const looWithin = looValid.filter(match => match.localConsistencyPx <= 1.5).length;
    $("field-summary").textContent = field.width + " × " + field.height + " px · grille " +
      field.gridWidth + " × " + field.gridHeight + " · HORG/SIFT ×" + field.scale +
      " · " + field.keypointsReference + "/" + field.keypointsTarget + " points-clés détectés · " +
      field.descriptorsReference + "/" + field.descriptorsTarget + " descripteurs comparés · " +
      field.ratioMatches + " paires au ratio test · " + accepted + "/" + field.count +
      " affines locales validées par RANSAC (" + fraction.toFixed(1) + " %) · extraction " +
      (field.extractionMs / 1000).toFixed(2) + " s, appariement + ratio GPU " +
      (field.matchComputeMs / 1000).toFixed(2) + " s, RANSAC + LOO GPU " +
      (field.ransacGpuMs / 1000).toFixed(3) + " s, cohérence LOO ≤1,5 px " + looWithin + "/" +
      looValid.length + " (diagnostic), ratios min/médiane/p90 " +
      field.ratioStats.min.toFixed(3) + "/" + field.ratioStats.median.toFixed(3) + "/" +
      field.ratioStats.p90.toFixed(3) + ", total " + (elapsedMs / 1000).toFixed(2) + " s.";
    return;
  }
  $("field-summary").textContent =
    field.width + " × " + field.height + " px · " + (field.seedMode === "manual"
      ? field.count + " seeds manuelles" : "grille " + field.gridWidth + " × " + field.gridHeight) +
    " · fenêtre " + field.patchSize + " px · pas " + field.stride + " px · réduction ×" + field.scale +
    " · rayon " + (field.searchRadiusPx ? field.searchRadiusPx + " px" : "illimité") +
    " · pyramide " + field.levels + " niveaux · racines INRIA " + field.evaluatedRootCount +
    " à " + field.rootPatchSize + " px · " +
    accepted + " / " + field.count + " graines réciproques (" + fraction.toFixed(1) + " %) · " +
    "descripteurs " + (field.descriptorMs / 1000).toFixed(2) + " s · hiérarchie " +
    (field.hierarchyMs / 1000).toFixed(2) + " s · total " + (elapsedMs / 1000).toFixed(2) + " s.";
}

function exportCsv() {
  if (!state.field) return;
  const clean = value => String(value == null ? "" : value).replace(/[\r\n,]/g, " ");
  const rows = [
    "# algorithm=" + clean(state.field.algorithm || "DeepMatching-WebGPU-INRIA-dense-grid"),
    "# reference=" + clean(state.reference && state.reference.name),
    "# target=" + clean(state.target && state.target.name),
    "# patch_size_px=" + state.field.patchSize,
    ...(state.field.algorithm === "DeepMatching-WebGPU-INRIA-dense-grid" ? [
      "# root_patch_size_px=" + state.field.rootPatchSize,
      "# atomic_patch_size_px=" + state.field.atomicPatchSize,
      "# hierarchy_levels=" + state.field.levels,
      "# evaluated_root_count=" + state.field.evaluatedRootCount,
    ] : []),
    "# grid_stride_px=" + state.field.stride,
    "# image_reduction_factor=" + state.field.scale,
    ...(state.field.algorithm === "HORG-WebGPU" ? [
      "# detected_keypoints_reference=" + state.field.keypointsReference,
      "# detected_keypoints_target=" + state.field.keypointsTarget,
      "# retained_descriptors_reference=" + state.field.descriptorsReference,
      "# retained_descriptors_target=" + state.field.descriptorsTarget,
      "# horg_ratio_test=" + state.field.ratioLimit,
      "# ransac_threshold_px=" + state.field.ransacThreshold,
      "# maximum_features_per_image=" + state.field.maxFeatures,
      "# local_affine_leave_one_out_neighbors=8 (diagnostic only; does not filter matches)",
      "x_ref_px,y_ref_px,dx_px,dy_px,inliers,local_candidates,inlier_fraction,mean_residual_px,loo_affine_error_8_neighbors_px,loo_neighbor_count,usable",
    ] : state.field.algorithm === "CPM-WebGPU" ? [
      "# search_radius_px=" + state.field.searchRadiusPx,
      "# work_grid_stride_px=" + state.field.workGridStridePx,
      "# pyramid_levels_requested=" + state.field.requestedLevels,
      "# pyramid_levels_effective=" + state.field.levels,
      "# random_trials=" + state.field.randomTrials,
      "# propagation_iterations_per_phase=" + state.field.propagationIterations,
      "# local_search_radius_px=" + state.field.localSearchRadiusPx,
      "# seed=" + state.field.seed,
      "# method_variant=" + clean(state.field.methodVariant),
      "requested_x_px,requested_y_px,measured_x_px,measured_y_px,dx_px,dy_px,target_x_px,target_y_px,similarity,reciprocal_error_px,grid_distance_px,directly_evaluated,reciprocal,usable",
    ] : [
      "# matcher_mode=" + clean(state.field.matcherMode),
      "# search_radius_px=" + state.field.searchRadiusPx,
      "# seed_mode=" + state.field.seedMode,
      "# reciprocal_only=true",
      "seed_x_requested_px,seed_y_requested_px,x_ref_px,y_ref_px,root_x_px,root_y_px,dx_px,dy_px,deepmatching_score,reciprocal,usable",
    ]),
  ];
  for (const match of state.field.matches) {
    rows.push(state.field.algorithm === "HORG-WebGPU" ? [
      match.x.toFixed(2),
      match.y.toFixed(2),
      match.dx.toFixed(2),
      match.dy.toFixed(2),
      match.inliers,
      match.candidates,
      match.inlierFraction.toFixed(6),
      Number.isFinite(match.residual) ? match.residual.toFixed(4) : "",
      Number.isFinite(match.localConsistencyPx) ? match.localConsistencyPx.toFixed(4) : "",
      match.localConsistencyNeighbors,
      match.usable ? "1" : "0",
    ].join(",") : state.field.algorithm === "CPM-WebGPU" ? [
      match.requestedX.toFixed(3), match.requestedY.toFixed(3), match.x.toFixed(3), match.y.toFixed(3),
      match.dx.toFixed(3), match.dy.toFixed(3),
      Number.isFinite(match.targetX) ? match.targetX.toFixed(3) : "",
      Number.isFinite(match.targetY) ? match.targetY.toFixed(3) : "",
      match.score.toFixed(6), match.reciprocalErrorPx.toFixed(4), match.gridDistancePx.toFixed(4),
      match.directlyEvaluated ? "1" : "0", match.reciprocal ? "1" : "0", match.usable ? "1" : "0",
    ].join(",") : [
      match.seedX.toFixed(2),
      match.seedY.toFixed(2),
      match.x.toFixed(2),
      match.y.toFixed(2),
      match.rootX.toFixed(2),
      match.rootY.toFixed(2),
      match.dx.toFixed(2),
      match.dy.toFixed(2),
      match.score.toFixed(6),
      match.reciprocal ? "1" : "0",
      match.usable ? "1" : "0",
    ].join(","));
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = state.field.algorithm === "HORG-WebGPU" ? "horg-dic-initializations.csv"
    : state.field.algorithm === "CPM-WebGPU" ? "cpm-wgpu-initializations.csv" : "deepmatching-dic-seeds.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function initializeGpu() {
  if (!("gpu" in navigator)) {
    status("WebGPU indisponible. Ouvrez cette page via localhost ou HTTPS dans un navigateur compatible.", true);
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" })
      || await navigator.gpu.requestAdapter({ powerPreference: "low-power" });
    if (!adapter) throw new Error("Aucun adaptateur WebGPU trouvé.");
    const requiredFeatures = adapter.features.has("texture-formats-tier1")
      ? ["texture-formats-tier1"] : [];
    state.device = await adapter.requestDevice({ requiredFeatures });
    state.device.lost.then(info => {
      state.device = null;
      updateButtons();
      status("Le périphérique WebGPU a été perdu : " + (info.message || info.reason) +
        ". Rechargez la page.", true);
    });
    const adapterInfo = adapter.info && (adapter.info.description || adapter.info.device);
    status("WebGPU prêt" + (adapterInfo ? " · " + adapterInfo : "") + ".");
    updateButtons();
  } catch (error) {
    status(error.message || String(error), true);
  }
}

function updateAlgorithmControls() {
  const horg = $("algorithm").value === "horg";
  const cpm = $("algorithm").value === "cpm";
  $("horg-settings").hidden = !horg;
  $("horg-hint").hidden = !horg;
  $("cpm-settings").hidden = !cpm;
  $("cpm-hint").hidden = !cpm;
  $("deepmatching-hint").hidden = horg || cpm;
  $("root-patch-control").hidden = horg || cpm;
  $("run").textContent = horg ? "Lancer HORG WebGPU"
    : cpm ? "Lancer CPM-WGPU" : "Lancer DeepMatching INRIA · WebGPU";
  $("usable-legend").textContent = horg ? "Affine locale validée par RANSAC"
    : cpm ? "Match CPM réciproque dans le seuil" : "Correspondance INRIA réciproque";
  $("reject-legend").textContent = horg ? "Pas assez de correspondances cohérentes"
    : cpm ? "Match invalide ou non réciproque" : "Pas de correspondance réciproque";
}

for (const [id, key] of [["reference-file", "reference"], ["target-file", "target"]]) {
  $(id).addEventListener("change", async event => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    state.field = null;
    state.settings = null;
    updateButtons();
    $("field-summary").textContent = "Aucun champ calculé.";
    try {
      status("Lecture de " + file.name + "…");
      const image = await decodeGray(file);
      const other = state[key === "reference" ? "target" : "reference"];
      if (other && (other.width !== image.width || other.height !== image.height)) {
        throw new Error("Les deux images doivent avoir les mêmes dimensions.");
      }
      state[key] = image;
      showImage();
      updateButtons();
      status(state.reference && state.target
        ? "Deux images chargées. Réglez la recherche puis lancez " +
          ($("algorithm").value === "horg" ? "HORG." : $("algorithm").value === "cpm" ? "CPM." : "DeepMatching.")
        : file.name + " chargé.");
    } catch (error) {
      state[key] = null;
      updateButtons();
      status(error.message || String(error), true);
    }
  });
}

$("run").addEventListener("click", async () => {
  if (!state.device || !state.reference || !state.target || state.busy) return;
  state.busy = true;
  state.field = null;
  updateButtons();
  try {
    if (state.reference.width !== state.target.width || state.reference.height !== state.target.height) {
      throw new Error("Les deux images doivent avoir les mêmes dimensions.");
    }
    const settings = selectedSettings();
    state.settings = settings;
    const start = performance.now();
    const field = settings.algorithm === "horg"
      ? await computeHorg(state.device, state.reference.pixels, state.target.pixels,
        state.reference.width, state.reference.height, settings, status)
      : settings.algorithm === "cpm"
        ? await computeCPM(state.device, state.reference.pixels, state.target.pixels,
          state.reference.width, state.reference.height, settings, status)
        : await computeDeepMatching(state.device, state.reference.pixels, state.target.pixels,
          state.reference.width, state.reference.height, settings, status);
    field.algorithm = settings.algorithm === "horg" ? "HORG-WebGPU"
      : settings.algorithm === "cpm" ? "CPM-WebGPU" : "DeepMatching-WebGPU-INRIA-dense-grid";
    const elapsedMs = performance.now() - start;
    state.field = field;
    showImage();
    renderFieldSummary(field, elapsedMs);
    status("Calcul terminé. Examinez les graines, puis exportez le CSV.");
  } catch (error) {
    status(error.message || String(error), true);
  } finally {
    state.busy = false;
    updateButtons();
  }
});

$("export").addEventListener("click", exportCsv);
for (const id of ["patch-size", "root-patch-size", "grid-stride", "search-radius", "seed-points", "downscale-levels",
  "horg-reduction", "horg-max-features", "cpm-grid-stride", "cpm-levels",
  "cpm-propagation-iterations", "cpm-random-trials", "cpm-local-radius", "cpm-seed", "cpm-reciprocal-threshold"]) {
  $(id).addEventListener("change", () => {
    state.field = null;
    updateButtons();
    $("field-summary").textContent = "Paramètres modifiés · relancez le calcul pour actualiser les graines.";
  });
}

$("algorithm").addEventListener("change", () => {
  state.field = null;
  updateButtons();
  $("field-summary").textContent = "Méthode modifiée · relancez le calcul pour actualiser le champ.";
  updateAlgorithmControls();
});

updateAlgorithmControls();

initializeGpu();

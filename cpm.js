import { createDeepMatchingHogDescriptors } from "./deepmatching.js";

const BU = globalThis.GPUBufferUsage;
const MM = globalThis.GPUMapMode;

async function loadMatcher(device) {
  const response = await fetch(new URL("./cpm-match.wgsl", import.meta.url));
  if (!response.ok) throw new Error("Could not load CPM WGSL shader (" + response.status + ").");
  const module = device.createShaderModule({ label: "CPM PatchMatch HOG", code: await response.text() });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  if (errors.length) {
    throw new Error(errors.map(message => "cpm-match.wgsl:" + message.lineNum + ":" +
      message.linePos + " " + message.message).join("\n"));
  }
  return device.createComputePipelineAsync({
    label: "CPM sparse HOG PatchMatch",
    layout: "auto",
    compute: { module, entryPoint: "patchmatch_step" },
  });
}

async function buildDescriptorPyramid(device, hog, levelCount) {
  const response = await fetch(new URL("./cpm-pyramid.wgsl", import.meta.url));
  if (!response.ok) throw new Error("Could not load CPM pyramid WGSL shader (" + response.status + ").");
  const module = device.createShaderModule({ label: "CPM pooled-HOG pyramid", code: await response.text() });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  if (errors.length) {
    throw new Error(errors.map(message => "cpm-pyramid.wgsl:" + message.lineNum + ":" +
      message.linePos + " " + message.message).join("\n"));
  }
  const pipeline = await device.createComputePipelineAsync({
    label: "CPM HOG pyramid downsample",
    layout: "auto",
    compute: { module, entryPoint: "downsample_hog" },
  });
  const usage = globalThis.GPUTextureUsage.STORAGE_BINDING | globalThis.GPUTextureUsage.TEXTURE_BINDING;
  const levels = [hog];
  const ownedTextures = [];
  try {
  for (let level = 1; level < levelCount; level++) {
    const source = levels[level - 1];
    const width = Math.floor(source.width / 2);
    const height = Math.floor(source.height / 2);
    if (width < 1 || height < 1) break;
    const createSet = label => ({
      width, height,
      first: device.createTexture({ label: label + " HOG 1-4", size: [width, height],
        format: "rgba16float", usage }),
      second: device.createTexture({ label: label + " HOG 5-8", size: [width, height],
        format: "rgba16float", usage }),
    });
    const reference = createSet("CPM level " + level + " reference");
    const target = createSet("CPM level " + level + " target");
    ownedTextures.push(reference.first, reference.second, target.first, target.second);
    const params = storage(device, 16, BU.UNIFORM | BU.COPY_DST, "CPM pyramid dimensions " + level);
    device.queue.writeBuffer(params, 0, new Uint32Array([source.width, source.height, width, height]));
    const bind = (input, output) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: input.first.createView() },
        { binding: 1, resource: input.second.createView() },
        { binding: 2, resource: output.first.createView() },
        { binding: 3, resource: output.second.createView() },
        { binding: 4, resource: { buffer: params } },
      ],
    });
    const encoder = device.createCommandEncoder({ label: "CPM HOG pyramid level " + level });
    for (const [imageLabel, input, output] of [
      ["reference", source.reference, reference], ["target", source.target, target],
    ]) {
      const pass = encoder.beginComputePass({ label: "CPM HOG pyramid " + imageLabel });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind(input, output));
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
    params.destroy();
    levels.push({ width, height, reference, target });
  }
  return { levels, ownedTextures };
  } catch (error) {
    try { await device.queue.onSubmittedWorkDone(); } catch {}
    for (const texture of ownedTextures) texture.destroy();
    throw error;
  }
}

function storage(device, size, usage, label) {
  return device.createBuffer({ label, size: Math.max(16, size), usage });
}

function makeParams(device, values, label) {
  const data = new Uint32Array(16);
  data.set(values.map(value => value >>> 0));
  const buffer = storage(device, data.byteLength, BU.UNIFORM | BU.COPY_DST, label);
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function bindStep(device, pipeline, sourceSet, targetSet, points, previous, next, params) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: sourceSet.first.createView() },
      { binding: 1, resource: sourceSet.second.createView() },
      { binding: 2, resource: targetSet.first.createView() },
      { binding: 3, resource: targetSet.second.createView() },
      { binding: 4, resource: { buffer: points } },
      { binding: 5, resource: { buffer: previous } },
      { binding: 6, resource: { buffer: next } },
      { binding: 7, resource: { buffer: params } },
    ],
  });
}

async function runDirection(device, pipeline, levels, pointBuffers, gridWidth, gridHeight,
  patchSizePx, baseScale, radiusPx, seed, settings, label) {
  const count = gridWidth * gridHeight;
  const stateBytes = count * 16;
  const previous = storage(device, stateBytes, BU.STORAGE | BU.COPY_SRC, label + " states A");
  const next = storage(device, stateBytes, BU.STORAGE | BU.COPY_SRC, label + " states B");
  const readback = storage(device, stateBytes, BU.COPY_DST | BU.MAP_READ, label + " readback");
  const paramsToDestroy = [];
  const groups = [];
  try {
    let readState = previous;
    let writeState = next;
    let iteration = 0;
    for (let levelIndex = levels.length - 1; levelIndex >= 0; levelIndex--) {
      const level = levels[levelIndex];
      const levelScale = baseScale * (2 ** levelIndex);
      const patchSize = Math.max(2, Math.round(patchSizePx / levelScale));
      const searchRadius = radiusPx > 0
        ? Math.floor(radiusPx / levelScale) : Math.max(level.width, level.height);
      const randomRadiusAtLevel = levelIndex === levels.length - 1
        ? searchRadius
        : Math.max(1, Math.min(searchRadius,
          Math.ceil(settings.cpmLocalSearchRadiusPx * (2 ** levelIndex) / levelScale)));
      const append = (initialize, scalePrevious, randomRadius) => {
        const params = makeParams(device, [
          gridWidth, gridHeight, level.width, level.height, patchSize,
          searchRadius, seed >>> 0, iteration, randomRadius, settings.cpmRandomTrials,
          initialize ? 1 : 0, scalePrevious ? 1 : 0, 0, 0, 0, 0,
        ], label + " level " + levelIndex + " pass " + iteration);
        paramsToDestroy.push(params);
        groups.push(bindStep(device, pipeline, level.reference, level.target,
          pointBuffers[levelIndex], readState, writeState, params));
        [readState, writeState] = [writeState, readState];
        iteration++;
      };
      if (levelIndex === levels.length - 1) {
        append(true, false, searchRadius);
        for (let pass = 0; pass < settings.cpmPropagationIterations; pass++) {
          append(false, false, Math.max(1, Math.ceil(searchRadius / (2 ** pass))));
        }
      } else {
        for (let pass = 0; pass < settings.cpmPropagationIterations; pass++) {
          const radius = Math.max(1, Math.min(searchRadius,
            Math.ceil(randomRadiusAtLevel / (2 ** pass))));
          append(false, pass === 0, radius);
        }
      }
    }

    const encoder = device.createCommandEncoder({ label: label + " coarse-to-fine ping-pong passes" });
    for (const group of groups) {
      const pass = encoder.beginComputePass({ label: label + " PatchMatch pass" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(count / 64));
      pass.end();
    }
    encoder.copyBufferToBuffer(readState, 0, readback, 0, stateBytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(MM.READ);
    const raw = new ArrayBuffer(stateBytes);
    new Uint8Array(raw).set(new Uint8Array(readback.getMappedRange()));
    readback.unmap();
    return { integers: new Int32Array(raw), floats: new Float32Array(raw), count };
  } finally {
    for (const resource of [previous, next, readback, ...paramsToDestroy]) resource.destroy();
  }
}

function regularCenters(width, height, patchSize, stride) {
  const half = (patchSize - 1) / 2;
  const centers = [];
  for (let y = half; y <= height - 1 - half; y += stride) {
    for (let x = half; x <= width - 1 - half; x += stride) centers.push({ x, y });
  }
  return centers;
}

function nearestIndex(centers, x, y, stride) {
  const estimate = Math.max(0, Math.min(centers.length - 1,
    Math.round((y - centers[0].y) / stride) * Math.max(1, Math.round((centers.at(-1).x - centers[0].x) / stride + 1)) +
    Math.round((x - centers[0].x) / stride)));
  let best = estimate;
  let bestDistance = Infinity;
  for (let index = Math.max(0, estimate - 2); index <= Math.min(centers.length - 1, estimate + 2); index++) {
    const distance = (centers[index].x - x) ** 2 + (centers[index].y - y) ** 2;
    if (distance < bestDistance) { bestDistance = distance; best = index; }
  }
  return { index: best, distance: Math.sqrt(bestDistance) };
}

/**
 * CPM-HOG prototype: deterministic sparse-grid PatchMatch with reciprocal
 * verification. The output points are directly evaluated work-grid locations;
 * arbitrary requested seeds are mapped to their nearest measured grid point.
 */
export async function computeCPM(device, referencePixels, targetPixels,
  width, height, settings, onProgress = () => {}) {
  if (!BU || !MM) throw new Error("WebGPU buffer constants are unavailable.");
  const scale = 2 ** (settings.downscaleLevels ?? 2);
  const patchSizePx = settings.patchSize || 32;
  const patchSize = Math.max(2, Math.round(patchSizePx / scale));
  const workStride = settings.cpmGridStride || 16;
  const radiusPx = settings.searchRadiusPx || 0;
  if (!Number.isInteger(workStride) || workStride < 4 || workStride > 256) {
    throw new Error("CPM work-grid stride must be an integer from 4 to 256 px.");
  }
  if (!Number.isInteger(settings.cpmLevels) || settings.cpmLevels < 1 || settings.cpmLevels > 6) {
    throw new Error("CPM pyramid levels must be from 1 to 6.");
  }
  if (!Number.isInteger(settings.cpmPropagationIterations) || settings.cpmPropagationIterations < 1
      || settings.cpmPropagationIterations > 16 || !Number.isInteger(settings.cpmRandomTrials)
      || settings.cpmRandomTrials < 1 || settings.cpmRandomTrials > 16) {
    throw new Error("CPM propagation iterations and random trials must each be from 1 to 16.");
  }
  if (!Number.isInteger(radiusPx) || radiusPx < 0) throw new Error("CPM search radius must be non-negative.");
  const localSearchRadiusPx = settings.cpmLocalSearchRadiusPx ?? 16;
  if (!Number.isFinite(localSearchRadiusPx) || localSearchRadiusPx < 1 || localSearchRadiusPx > 512) {
    throw new Error("CPM local refinement radius must be from 1 to 512 px.");
  }
  const seeds = settings.seedPoints?.length ? settings.seedPoints : regularCenters(width, height,
    patchSizePx, settings.stride || workStride);
  if (!seeds.length) throw new Error("There are no requested DIC seed centers.");
  if (seeds.length > 512) throw new Error("CPM currently limits requested DIC seeds to 512.");
  for (let index = 0; index < seeds.length; index++) {
    const seed = seeds[index];
    if (!Number.isFinite(seed.x) || !Number.isFinite(seed.y)
        || seed.x < (patchSizePx - 1) / 2 || seed.y < (patchSizePx - 1) / 2
        || seed.x > width - 1 - (patchSizePx - 1) / 2
        || seed.y > height - 1 - (patchSizePx - 1) / 2) {
      throw new Error("CPM seed " + (index + 1) + " does not fit a full DIC patch in the reference image.");
    }
  }
  const centers = regularCenters(width, height, patchSizePx, workStride);
  if (!centers.length) throw new Error("The CPM work grid does not fit the DIC patch in the image.");
  if (centers.length > 200000) throw new Error("CPM work grid exceeds the current 200000-point safety limit.");

  let hog = null;
  let descriptorPyramid = null;
  const pointBuffers = [];
  let pipeline = null;
  try {
    onProgress("CPM-HOG · building shared INRIA-style GPU descriptors…");
    const descriptorStart = performance.now();
    hog = await createDeepMatchingHogDescriptors(device, referencePixels, targetPixels,
      width, height, Math.log2(scale), onProgress);
    const descriptorMs = performance.now() - descriptorStart;
    if (patchSize > hog.width || patchSize > hog.height) {
      throw new Error("CPM patch is larger than the reduced descriptor image.");
    }
    const maxLevelsFromPatch = Math.max(1, Math.floor(Math.log2(patchSize / 4)) + 1);
    const requestedLevels = settings.cpmLevels;
    const effectiveLevels = Math.min(requestedLevels, maxLevelsFromPatch);
    const pyramidStart = performance.now();
    descriptorPyramid = await buildDescriptorPyramid(device, hog, effectiveLevels);
    await device.queue.onSubmittedWorkDone();
    const pyramidMs = performance.now() - pyramidStart;
    const levels = descriptorPyramid.levels;
    if (levels.length < effectiveLevels) {
      onProgress("CPM · pyramid depth reduced because the descriptor image became too small.");
    }
    pipeline = await loadMatcher(device);
    if (!Number.isSafeInteger(centers.length * 16)
        || centers.length * 16 > device.limits.maxBufferSize
        || centers.length * 16 > device.limits.maxStorageBufferBindingSize
        || centers.length * 8 > device.limits.maxStorageBufferBindingSize
        || Math.ceil(centers.length / 64) > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new Error("CPM state buffers exceed this adapter's WebGPU buffer limit.");
    }
    const gridWidth = 1 + Math.floor((width - patchSizePx) / workStride);
    const gridHeight = Math.ceil(centers.length / Math.max(1, gridWidth));
    const perLevelBytes = centers.length * 8;
    for (let levelIndex = 0; levelIndex < levels.length; levelIndex++) {
      const levelScale = scale * (2 ** levelIndex);
      const coordinates = new Int32Array(centers.length * 2);
      centers.forEach((point, index) => {
        coordinates[2 * index] = Math.round(point.x / levelScale);
        coordinates[2 * index + 1] = Math.round(point.y / levelScale);
      });
      const pointBuffer = storage(device, perLevelBytes, BU.STORAGE | BU.COPY_DST,
        "CPM source centers at pyramid level " + levelIndex);
      device.queue.writeBuffer(pointBuffer, 0, coordinates);
      pointBuffers.push(pointBuffer);
    }
    const patchmatchStart = performance.now();
    const runSettings = { ...settings, cpmLocalSearchRadiusPx: localSearchRadiusPx };
    const forward = await runDirection(device, pipeline, levels, pointBuffers,
      gridWidth, gridHeight, patchSizePx, scale, radiusPx, settings.cpmSeed >>> 0, runSettings,
      "CPM reference to target");
    const backward = await runDirection(device, pipeline, levels.map(level => ({
      width: level.width, height: level.height,
      reference: level.target, target: level.reference,
    })), pointBuffers, gridWidth, gridHeight, patchSizePx, scale, radiusPx,
      settings.cpmSeed >>> 0, runSettings,
      "CPM target to reference");
    const matchingMs = performance.now() - patchmatchStart;

    const matches = seeds.map(requested => {
      const nearest = nearestIndex(centers, requested.x, requested.y, workStride);
      const index = nearest.index;
      const intIndex = index * 4;
      const forwardDx = forward.integers[intIndex];
      const forwardDy = forward.integers[intIndex + 1];
      const score = 1 - Math.max(0, Math.min(1, forward.floats[intIndex + 2]));
      const targetX = centers[index].x + forwardDx * scale;
      const targetY = centers[index].y + forwardDy * scale;
      const reverse = nearestIndex(centers, targetX, targetY, workStride);
      const reverseOffset = reverse.index * 4;
      const fbX = forwardDx * scale + backward.integers[reverseOffset] * scale;
      const fbY = forwardDy * scale + backward.integers[reverseOffset + 1] * scale;
      const reciprocalError = Math.hypot(fbX, fbY);
      const reverseValid = backward.integers[reverseOffset + 3] !== 0;
      const reciprocal = nearest.distance <= workStride * Math.SQRT2
        && reverse.distance <= workStride * Math.SQRT2
        && reverseValid
        && reciprocalError <= (settings.cpmReciprocalThresholdPx ?? 8);
      const valid = forward.integers[intIndex + 3] !== 0;
      const usable = valid && reciprocal && score >= (settings.cpmMinimumSimilarity ?? 0.0);
      return {
        x: centers[index].x,
        y: centers[index].y,
        requestedX: requested.x,
        requestedY: requested.y,
        dx: valid ? forwardDx * scale : 0,
        dy: valid ? forwardDy * scale : 0,
        targetX: valid ? targetX : null,
        targetY: valid ? targetY : null,
        score,
        reciprocal,
        reciprocalErrorPx: reciprocalError,
        gridDistancePx: nearest.distance,
        directlyEvaluated: true,
        usable,
      };
    });
    return {
      algorithm: "CPM-WebGPU",
      width, height, patchSize: patchSizePx, stride: settings.stride || workStride,
      scale, levels: levels.length, requestedLevels, count: matches.length, matches,
      seedMode: settings.seedPoints?.length ? "manual" : "grid",
      gridWidth, gridHeight, workGridStridePx: workStride,
      workGridPointCount: centers.length, searchRadiusPx: radiusPx,
      randomTrials: settings.cpmRandomTrials,
      propagationIterations: settings.cpmPropagationIterations,
      localSearchRadiusPx,
      seed: settings.cpmSeed >>> 0,
      descriptorMs, pyramidMs, matchingMs,
      methodVariant: "CPM-inspired sparse-grid PatchMatch using mean-pooled INRIA-style HOG, cosine patch cost, and synchronous four-neighbor propagation",
    };
  } finally {
    for (const pointBuffer of pointBuffers) pointBuffer.destroy();
    if (descriptorPyramid) {
      for (const texture of descriptorPyramid.ownedTextures) texture.destroy();
    }
    if (hog) hog.destroy();
  }
}

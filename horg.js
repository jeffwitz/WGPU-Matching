import { applyBrowserDefaults } from "./matcher-contracts.js";

const bufferUsage = globalThis.GPUBufferUsage;
const textureUsage = globalThis.GPUTextureUsage;
const mapMode = globalThis.GPUMapMode;

const DESCRIPTOR_SIZE = 64;
const FEATURE_STRIDE = 68;
const MAX_DETECTIONS_PER_OCTAVE = 60000;
const RATIO_LIMIT = 0.75;
const RANSAC_THRESHOLD = 1.5;
const MIN_AFFINE_INLIERS = 4;
const MIN_LOCAL_CANDIDATES = 8;
const MIN_LOCAL_INLIER_FRACTION = 0.2;
const MAX_RANSAC_CANDIDATES = 96;
const MAX_RANSAC_ITERATIONS = 256;

async function loadShader(device, filename) {
  const response = await fetch(new URL(`./python/src/wgpu_matching/shaders/${filename}`, import.meta.url), { cache: "no-store" });
  if (!response.ok) throw new Error("Chargement du noyau " + filename + " impossible (" + response.status + ").");
  const module = device.createShaderModule({ label: filename, code: await response.text() });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  if (errors.length) {
    throw new Error(errors.map(message => filename + ":" + message.lineNum + ":" + message.linePos + " " + message.message).join("\n"));
  }
  return module;
}

async function makePipeline(device, module, entryPoint, label) {
  return device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint },
  });
}

function ownBuffer(device, resources, size, usage, label) {
  if (!Number.isSafeInteger(size) || size < 4) throw new Error("Taille de tampon invalide pour " + label + ".");
  const buffer = device.createBuffer({ label, size, usage });
  resources.add(buffer);
  return buffer;
}

function ownTexture(device, resources, width, height, label, format = "r32float",
  usage = textureUsage.STORAGE_BINDING | textureUsage.TEXTURE_BINDING | textureUsage.COPY_DST) {
  if (width < 1 || height < 1) throw new Error("Niveau de pyramide HORG trop petit.");
  const texture = device.createTexture({
    label,
    size: { width, height, depthOrArrayLayers: 1 },
    format,
    usage,
  });
  resources.add(texture);
  return texture;
}

function release(resources, resource) {
  if (!resource || !resources.has(resource)) return;
  resource.destroy();
  resources.delete(resource);
}

function bindGroup(device, pipeline, entries) {
  return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
}

function ownPyramidParams(device, resources, dims, config, weights = null, label = "HORG pyramid parameters") {
  const data = new ArrayBuffer(96);
  new Uint32Array(data, 0, 8).set([...dims, ...config]);
  if (weights) new Float32Array(data, 32, 16).set(weights);
  const buffer = ownBuffer(device, resources, data.byteLength,
    bufferUsage.UNIFORM | bufferUsage.COPY_DST, label);
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function gaussianWeights(sigma) {
  const radius = Math.min(7, Math.max(1, Math.ceil(3 * sigma)));
  const values = new Float32Array(16);
  let sum = 0;
  for (let offset = -radius; offset <= radius; offset++) {
    const weight = Math.exp(-(offset * offset) / (2 * sigma * sigma));
    values[offset + radius] = weight;
    sum += weight;
  }
  for (let index = 0; index < 2 * radius + 1; index++) values[index] /= sum;
  return { radius, values };
}

function encodePass(encoder, pipeline, group, groupsX, groupsY = 1, groupsZ = 1, label = "HORG pass") {
  const pass = encoder.beginComputePass({ label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(groupsX, groupsY, groupsZ);
  pass.end();
}

function encodeBlur(device, resources, encoder, pipeline, source, destination, width, height, sigma, axis, temporaryUniforms) {
  const { radius, values } = gaussianWeights(sigma);
  const params = ownPyramidParams(device, resources,
    [width, height, 0, 0], [radius, axis, 0, 0], values, "HORG Gaussian blur parameters");
  temporaryUniforms.push(params);
  const group = bindGroup(device, pipeline, [
    { binding: 0, resource: source.createView() },
    { binding: 1, resource: destination.createView() },
    { binding: 2, resource: { buffer: params } },
  ]);
  encodePass(encoder, pipeline, group, Math.ceil(width / 8), Math.ceil(height / 8), 1,
    "HORG Gaussian blur " + (axis === 0 ? "horizontal" : "vertical"));
}

function encodeSubtract(device, resources, encoder, pipeline, lower, upper, output, width, height, temporaryUniforms) {
  const params = ownPyramidParams(device, resources, [width, height, 0, 0], [0, 0, 0, 0], null,
    "HORG DoG parameters");
  temporaryUniforms.push(params);
  const group = bindGroup(device, pipeline, [
    { binding: 0, resource: lower.createView() },
    { binding: 2, resource: { buffer: params } },
    { binding: 3, resource: upper.createView() },
    { binding: 4, resource: output.createView() },
  ]);
  encodePass(encoder, pipeline, group, Math.ceil(width / 8), Math.ceil(height / 8), 1, "HORG Difference of Gaussians");
}

async function readCount(device, resources, encoder, counter) {
  const result = ownBuffer(device, resources, 4, bufferUsage.COPY_DST | bufferUsage.MAP_READ,
    "HORG keypoint count readback");
  encoder.copyBufferToBuffer(counter, 0, result, 0, 4);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await result.mapAsync(mapMode.READ);
  const count = new Uint32Array(result.getMappedRange())[0];
  result.unmap();
  release(resources, result);
  return count;
}

function makeUniform(device, resources, words, label) {
  const data = new Uint32Array(Math.max(4, words.length));
  data.set(words);
  const buffer = ownBuffer(device, resources, data.byteLength,
    bufferUsage.UNIFORM | bufferUsage.COPY_DST, label);
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function makeRansacUniform(device, resources, counts, settings, label) {
  const data = new ArrayBuffer(32);
  new Uint32Array(data, 0, 4).set(counts);
  new Float32Array(data, 16, 4).set(settings);
  const buffer = ownBuffer(device, resources, data.byteLength,
    bufferUsage.UNIFORM | bufferUsage.COPY_DST, label);
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

async function imageFeatures(device, resources, pipelines, pixels, width, height, reduction,
  onProgress, imageName) {
  const limits = device.limits;
  if (width > limits.maxTextureDimension2D || height > limits.maxTextureDimension2D) {
    throw new Error("Les images dépassent la limite de texture WebGPU (" + limits.maxTextureDimension2D + " px).");
  }
  const input = ownTexture(device, resources, width, height, imageName + " input", "r8unorm",
    textureUsage.TEXTURE_BINDING | textureUsage.COPY_DST);
  device.queue.writeTexture(
    { texture: input }, pixels,
    { bytesPerRow: width, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );

  let baseWidth = Math.max(1, Math.floor(width / reduction));
  let baseHeight = Math.max(1, Math.floor(height / reduction));
  if (baseWidth > limits.maxTextureDimension2D || baseHeight > limits.maxTextureDimension2D) {
    throw new Error("La base de la pyramide HORG dépasse la limite de texture WebGPU.");
  }
  const segments = [];
  let totalDetected = 0;
  let overflow = 0;
  let octaveBase = null;
  let octaveCount = Math.min(6, Math.max(1, Math.floor(Math.log2(Math.min(baseWidth, baseHeight))) - 2));
  const sigmaBase = 1.6;
  const contrastThreshold = 0.04 / 3;
  const edgeLimit = (10 + 1) * (10 + 1) / 10;

  for (let octave = 0; octave < octaveCount; octave++) {
    if (Math.min(baseWidth, baseHeight) < 28) break;
    const gaussian = [];
    const dogs = [];
    const scratchTextures = [];
    const temporaryUniforms = [];
    let rawBase = null;
    let blurScratch = null;
    const encoder = device.createCommandEncoder({ label: imageName + " HORG octave " + octave });

    if (octave === 0) {
      rawBase = ownTexture(device, resources, baseWidth, baseHeight, imageName + " initial downsample");
      blurScratch = ownTexture(device, resources, baseWidth, baseHeight, imageName + " initial blur scratch");
      const downParams = ownPyramidParams(device, resources,
        [width, height, baseWidth, baseHeight], [reduction, 0, 0, 0], null,
        "HORG initial downsample parameters");
      temporaryUniforms.push(downParams);
      const downGroup = bindGroup(device, pipelines.downsampleBase, [
        { binding: 0, resource: input.createView() },
        { binding: 1, resource: rawBase.createView() },
        { binding: 2, resource: { buffer: downParams } },
      ]);
      encodePass(encoder, pipelines.downsampleBase, downGroup,
        Math.ceil(baseWidth / 8), Math.ceil(baseHeight / 8), 1, "HORG initial image reduction");
      const first = ownTexture(device, resources, baseWidth, baseHeight, imageName + " Gaussian level 0");
      gaussian.push(first);
      encodeBlur(device, resources, encoder, pipelines.blur, rawBase, blurScratch,
        baseWidth, baseHeight, sigmaBase, 0, temporaryUniforms);
      encodeBlur(device, resources, encoder, pipelines.blur, blurScratch, first,
        baseWidth, baseHeight, sigmaBase, 1, temporaryUniforms);
    } else {
      gaussian.push(octaveBase);
      octaveBase = null;
    }

    for (let level = 1; level <= 5; level++) {
      const previousSigma = sigmaBase * Math.pow(2, (level - 1) / 3);
      const currentSigma = sigmaBase * Math.pow(2, level / 3);
      const deltaSigma = Math.sqrt(currentSigma * currentSigma - previousSigma * previousSigma);
      const scratch = ownTexture(device, resources, baseWidth, baseHeight,
        imageName + " octave " + octave + " blur scratch " + level);
      scratchTextures.push(scratch);
      const next = ownTexture(device, resources, baseWidth, baseHeight,
        imageName + " octave " + octave + " Gaussian " + level);
      gaussian.push(next);
      encodeBlur(device, resources, encoder, pipelines.blur, gaussian[level - 1], scratch,
        baseWidth, baseHeight, deltaSigma, 0, temporaryUniforms);
      encodeBlur(device, resources, encoder, pipelines.blur, scratch, next,
        baseWidth, baseHeight, deltaSigma, 1, temporaryUniforms);
      // Scratch textures are released after the encoder has completed below.
    }

    for (let level = 0; level < 5; level++) {
      const dog = ownTexture(device, resources, baseWidth, baseHeight,
        imageName + " octave " + octave + " DoG " + level);
      dogs.push(dog);
      encodeSubtract(device, resources, encoder, pipelines.subtract,
        gaussian[level], gaussian[level + 1], dog, baseWidth, baseHeight, temporaryUniforms);
    }

    const counter = ownBuffer(device, resources, 4,
      bufferUsage.STORAGE | bufferUsage.COPY_SRC | bufferUsage.COPY_DST,
      imageName + " octave " + octave + " keypoint counter");
    device.queue.writeBuffer(counter, 0, new Uint32Array([0]));
    const keypoints = ownBuffer(device, resources, MAX_DETECTIONS_PER_OCTAVE * 16,
      bufferUsage.STORAGE | bufferUsage.COPY_SRC, imageName + " octave " + octave + " keypoints");
    const detectParamsData = new ArrayBuffer(48);
    new Uint32Array(detectParamsData, 0, 4).set([baseWidth, baseHeight, 0, 0]);
    new Float32Array(detectParamsData, 16, 4).set([contrastThreshold, edgeLimit, 0, 0]);
    new Uint32Array(detectParamsData, 32, 4).set([MAX_DETECTIONS_PER_OCTAVE, 0, 0, 0]);
    const detectParams = ownBuffer(device, resources, 48,
      bufferUsage.UNIFORM | bufferUsage.COPY_DST, "HORG keypoint detector parameters");
    device.queue.writeBuffer(detectParams, 0, detectParamsData);
    temporaryUniforms.push(detectParams);
    const detectGroup = bindGroup(device, pipelines.detect, [
      ...dogs.map((texture, index) => ({ binding: index, resource: texture.createView() })),
      { binding: 5, resource: { buffer: counter } },
      { binding: 6, resource: { buffer: keypoints } },
      { binding: 7, resource: { buffer: detectParams } },
    ]);
    encodePass(encoder, pipelines.detect, detectGroup,
      Math.ceil(baseWidth / 8), Math.ceil(baseHeight / 8), 3, "HORG SIFT scale-space extrema");
    const count = await readCount(device, resources, encoder, counter);
    const featureCount = Math.min(count, MAX_DETECTIONS_PER_OCTAVE);
    totalDetected += count;
    overflow += Math.max(0, count - MAX_DETECTIONS_PER_OCTAVE);
    onProgress("HORG " + imageName + " · octave " + (octave + 1) + "/" + octaveCount + " · " + featureCount + " points-clés.");

    // The pipeline above also creates one scratch texture for each Gaussian level.
    // Locate and release all scratch images once their command encoder is complete.
    for (const resource of temporaryUniforms.splice(0)) release(resources, resource);
    release(resources, rawBase);
    release(resources, blurScratch);
    for (const texture of scratchTextures) release(resources, texture);

    if (featureCount > 0) {
      const featureBytes = featureCount * FEATURE_STRIDE * 4;
      if (featureBytes > limits.maxStorageBufferBindingSize || featureBytes > limits.maxBufferSize) {
        throw new Error("Les descripteurs HORG d’un octave dépassent la mémoire de stockage WebGPU.");
      }
      const output = ownBuffer(device, resources, featureBytes,
        bufferUsage.STORAGE | bufferUsage.COPY_SRC, imageName + " octave " + octave + " HORG descriptors");
      const descriptorParams = makeUniform(device, resources,
        [baseWidth, baseHeight, featureCount, 0, reduction * (1 << octave), 0, 0, baseWidth],
        "HORG descriptor dimensions and scale");
      const descriptorGroup = bindGroup(device, pipelines.descriptor, [
        { binding: 0, resource: gaussian[2].createView() },
        { binding: 1, resource: gaussian[3].createView() },
        { binding: 2, resource: gaussian[4].createView() },
        { binding: 3, resource: { buffer: keypoints } },
        { binding: 4, resource: { buffer: output } },
        { binding: 5, resource: { buffer: descriptorParams } },
      ]);
      const descriptorEncoder = device.createCommandEncoder({ label: imageName + " HORG descriptor construction" });
      encodePass(descriptorEncoder, pipelines.descriptor, descriptorGroup,
        featureCount, 1, 1, "HORG orientation and 64D descriptors");
      const readback = ownBuffer(device, resources, featureBytes,
        bufferUsage.COPY_DST | bufferUsage.MAP_READ, imageName + " HORG descriptor readback");
      descriptorEncoder.copyBufferToBuffer(output, 0, readback, 0, featureBytes);
      device.queue.submit([descriptorEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await readback.mapAsync(mapMode.READ);
      segments.push(new Float32Array(readback.getMappedRange()).slice());
      readback.unmap();
      release(resources, readback);
      release(resources, output);
      release(resources, descriptorParams);
    } else {
      // readCount already waited for pyramid and detection work to finish.
    }

    release(resources, counter);
    release(resources, keypoints);
    const hasNextOctave = octave < octaveCount - 1 && Math.min(baseWidth, baseHeight) >= 56;
    if (hasNextOctave) {
      const nextWidth = Math.floor(baseWidth / 2);
      const nextHeight = Math.floor(baseHeight / 2);
      const nextBase = ownTexture(device, resources, nextWidth, nextHeight,
        imageName + " octave " + (octave + 1) + " base");
      const downParams = ownPyramidParams(device, resources,
        [baseWidth, baseHeight, nextWidth, nextHeight], [2, 0, 0, 0], null,
        "HORG octave downsample parameters");
      const downGroup = bindGroup(device, pipelines.downsampleHalf, [
        { binding: 0, resource: gaussian[3].createView() },
        { binding: 1, resource: nextBase.createView() },
        { binding: 2, resource: { buffer: downParams } },
      ]);
      const downEncoder = device.createCommandEncoder({ label: imageName + " HORG octave reduction" });
      encodePass(downEncoder, pipelines.downsampleHalf, downGroup,
        Math.ceil(nextWidth / 8), Math.ceil(nextHeight / 8), 1, "HORG next SIFT octave");
      device.queue.submit([downEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      release(resources, downParams);
      octaveBase = nextBase;
    } else {
      octaveBase = null;
    }

    for (const texture of dogs) release(resources, texture);
    for (const texture of gaussian) {
      if (texture !== octaveBase) release(resources, texture);
    }
    // All remaining octave textures are blur scratch images. Their labels are
    // intentionally opaque to the algorithm, so the local resource list is kept
    // explicitly below instead of relying on GPU-side garbage collection.
    if (!hasNextOctave) break;
    baseWidth = Math.floor(baseWidth / 2);
    baseHeight = Math.floor(baseHeight / 2);
  }

  release(resources, octaveBase);
  release(resources, input);
  const total = segments.reduce((sum, part) => sum + part.length / FEATURE_STRIDE, 0);
  const all = new Float32Array(total * FEATURE_STRIDE);
  let offset = 0;
  for (const part of segments) {
    all.set(part, offset);
    offset += part.length;
  }
  return { data: all, count: total, detected: totalDetected, overflow };
}

function selectFeatures(features, maxCount, width, height) {
  const { data, count } = features;
  if (!count) return new Float32Array(0);
  const order = [];
  for (let index = 0; index < count; index++) {
    const offset = index * FEATURE_STRIDE;
    const x = data[offset];
    const y = data[offset + 1];
    if (Number.isFinite(x) && Number.isFinite(y) && x >= 0 && y >= 0 && x < width && y < height) {
      order.push(index);
    }
  }
  order.sort((a, b) => data[b * FEATURE_STRIDE + 3] - data[a * FEATURE_STRIDE + 3]);
  const limit = Math.min(maxCount, order.length);
  const selected = [];
  const used = new Uint8Array(count);
  const perCell = new Map();
  for (const index of order) {
    if (selected.length >= limit) break;
    const offset = index * FEATURE_STRIDE;
    const cell = Math.floor(data[offset + 1] / 128) * Math.ceil(width / 128)
      + Math.floor(data[offset] / 128);
    const hits = perCell.get(cell) || 0;
    if (hits >= 4) continue;
    selected.push(index);
    used[index] = 1;
    perCell.set(cell, hits + 1);
  }
  if (selected.length < limit) {
    for (const index of order) {
      if (selected.length >= limit) break;
      if (!used[index]) selected.push(index);
    }
  }
  const result = new Float32Array(selected.length * FEATURE_STRIDE);
  for (let row = 0; row < selected.length; row++) {
    const source = selected[row] * FEATURE_STRIDE;
    result.set(data.subarray(source, source + FEATURE_STRIDE), row * FEATURE_STRIDE);
  }
  return result;
}

function featureCoordinates(features) {
  const count = features.length / FEATURE_STRIDE;
  const coordinates = new Float32Array(count * 2);
  for (let index = 0; index < count; index++) {
    coordinates[index * 2] = features[index * FEATURE_STRIDE];
    coordinates[index * 2 + 1] = features[index * FEATURE_STRIDE + 1];
  }
  return coordinates;
}

function ratioDiagnostics(ratioValues) {
  const ratios = [];
  let ratioMatches = 0;
  for (const ratio of ratioValues) {
    if (!(ratio >= 0 && ratio <= 1)) continue;
    ratios.push(ratio);
    if (ratio < RATIO_LIMIT) ratioMatches++;
  }
  ratios.sort((a, b) => a - b);
  const quantile = q => ratios.length
    ? ratios[Math.floor(q * Math.max(0, ratios.length - 1))]
    : Number.NaN;
  return {
    ratioMatches,
    ratioStats: { min: quantile(0), p25: quantile(0.25), median: quantile(0.5), p90: quantile(0.9) },
  };
}

function buildLocalProblem(width, height, settings) {
  const defaultGridWidth = 1 + Math.floor((width - settings.patchSize) / settings.stride);
  const defaultGridHeight = 1 + Math.floor((height - settings.patchSize) / settings.stride);
  if (defaultGridWidth < 1 || defaultGridHeight < 1) {
    throw new Error("La fenêtre DIC est plus grande que l’image.");
  }
  const manual = Array.isArray(settings.seedPoints) && settings.seedPoints.length > 0;
  let centers;
  let gridWidth;
  let gridHeight;
  if (manual) {
    if (settings.seedPoints.length > 512) throw new Error("HORG accepts at most 512 manual seeds.");
    centers = settings.seedPoints.map((point, index) => {
      const x = Number(point.x);
      const y = Number(point.y);
      if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x >= width || y >= height) {
        throw new Error(`HORG seed ${index + 1} must contain finite coordinates inside the image.`);
      }
      return { x, y };
    });
    gridWidth = centers.length;
    gridHeight = 1;
  } else {
    gridWidth = defaultGridWidth;
    gridHeight = defaultGridHeight;
    centers = Array.from({ length: gridWidth * gridHeight }, (_, seed) => ({
      x: (seed % gridWidth) * settings.stride + (settings.patchSize - 1) / 2,
      y: Math.floor(seed / gridWidth) * settings.stride + (settings.patchSize - 1) / 2,
    }));
  }
  const seeds = new Float32Array(centers.length * 4);
  const half = (settings.patchSize - 1) / 2;
  for (let seed = 0; seed < centers.length; seed++) {
    const center = centers[seed];
    let randomStream = seed;
    if (manual) {
      const gx = Math.round((center.x - half) / settings.stride);
      const gy = Math.round((center.y - half) / settings.stride);
      const onGrid = gx >= 0 && gx < defaultGridWidth
        && Math.abs(center.x - (gx * settings.stride + half)) < 1e-3
        && Math.abs(center.y - (gy * settings.stride + half)) < 1e-3;
      if (onGrid) randomStream = gy * defaultGridWidth + gx;
    }
    seeds.set([center.x, center.y, 0, randomStream], seed * 4);
  }
  return { gridWidth, gridHeight, centers, seeds };
}

function makeLocalField(ratioMatchCount, ratioStats, problem, ransacResults, consistencyResults, settings) {
  const matches = [];
  for (let seed = 0; seed < problem.centers.length; seed++) {
      const center = problem.centers[seed];
      const result = seed * 8;
      const consistency = seed * 4;
      const candidateCount = Math.round(ransacResults[result + 3]);
      const inliers = Math.round(ransacResults[result + 2]);
      const fit = ransacResults[result + 6] > 0.5 ? {
        dx: ransacResults[result],
        dy: ransacResults[result + 1],
        inliers,
        candidates: candidateCount,
        inlierFraction: ransacResults[result + 4],
        residual: ransacResults[result + 5],
      } : null;
      const usable = !!fit && fit.inliers >= MIN_AFFINE_INLIERS
        && fit.candidates >= MIN_LOCAL_CANDIDATES
        && fit.inlierFraction >= MIN_LOCAL_INLIER_FRACTION
        && fit.residual <= RANSAC_THRESHOLD;
      matches.push({
        x: center.x,
        y: center.y,
        dx: fit ? fit.dx : 0,
        dy: fit ? fit.dy : 0,
        score: fit ? fit.inlierFraction : 0,
        secondScore: fit ? fit.residual : Number.NaN,
        margin: fit ? fit.inlierFraction : 0,
        usable,
        inliers: fit ? fit.inliers : 0,
        candidates: fit ? fit.candidates : 0,
        inlierFraction: fit ? fit.inlierFraction : 0,
        residual: fit ? fit.residual : Number.NaN,
        localConsistencyPx: consistencyResults[consistency] >= 0
          ? consistencyResults[consistency] : Number.NaN,
        localConsistencyNeighbors: Math.round(consistencyResults[consistency + 1]),
      });
  }
  return {
    ratioMatches: ratioMatchCount,
    matches,
    gridWidth: problem.gridWidth,
    gridHeight: problem.gridHeight,
    ratioStats,
  };
}

export async function computeHorg(device, referencePixels, targetPixels,
  width, height, settings, onProgress = () => {}) {
  settings = await applyBrowserDefaults(settings ?? {}, "horg");
  if (!bufferUsage || !textureUsage || !mapMode) {
    throw new Error("Les constantes WebGPU requises ne sont pas disponibles dans ce navigateur.");
  }
  if (referencePixels.length !== width * height || targetPixels.length !== width * height) {
    throw new Error("Les pixels reçus ne correspondent pas aux dimensions des images.");
  }
  const reduction = settings.horgReduction || 4;
  const maxFeatures = settings.horgMaxFeatures || 4096;
  const resources = new Set();
  try {
    const [pyramidModule, detectModule, descriptorModule, matcherModule, prepareModule,
      ransacModule, confidenceModule] = await Promise.all([
      loadShader(device, "horg-pyramid.wgsl"),
      loadShader(device, "horg-detect.wgsl"),
      loadShader(device, "horg-descriptor.wgsl"),
      loadShader(device, "horg-match.wgsl"),
      loadShader(device, "horg-prepare.wgsl"),
      loadShader(device, "horg-ransac.wgsl"),
      loadShader(device, "horg-confidence.wgsl"),
    ]);
    const pipelines = await Promise.all([
      makePipeline(device, pyramidModule, "downsample_base", "HORG initial image reduction"),
      makePipeline(device, pyramidModule, "downsample_half", "HORG octave reduction"),
      makePipeline(device, pyramidModule, "gaussian_blur", "HORG Gaussian blur"),
      makePipeline(device, pyramidModule, "subtract_gaussian", "HORG DoG"),
      makePipeline(device, detectModule, "detect_extrema", "HORG SIFT keypoint detection"),
      makePipeline(device, descriptorModule, "build_horg", "HORG 64D descriptor"),
      makePipeline(device, matcherModule, "nearest_two", "HORG descriptor nearest neighbours"),
      makePipeline(device, prepareModule, "filter_lowe_ratio", "HORG GPU Lowe ratio filter"),
      makePipeline(device, prepareModule, "build_local_candidates", "HORG GPU local candidate preparation"),
      makePipeline(device, ransacModule, "ransac_local", "HORG local affine RANSAC"),
      makePipeline(device, confidenceModule, "local_affine_leave_one_out", "HORG GPU local confidence diagnostic"),
    ]);
    const [downsampleBase, downsampleHalf, blur, subtract, detect, descriptor,
      nearest, filterRatios, buildCandidates, ransac, localConsistency] = pipelines;

    const extractionStart = performance.now();
    const referenceAll = await imageFeatures(device, resources,
      { downsampleBase, downsampleHalf, blur, subtract, detect, descriptor },
      referencePixels, width, height, reduction, onProgress, "référence");
    const targetAll = await imageFeatures(device, resources,
      { downsampleBase, downsampleHalf, blur, subtract, detect, descriptor },
      targetPixels, width, height, reduction, onProgress, "déformée");
    const referenceFeatures = selectFeatures(referenceAll, maxFeatures, width, height);
    const targetFeatures = selectFeatures(targetAll, maxFeatures, width, height);
    const extractionMs = performance.now() - extractionStart;
    onProgress("HORG · " + (referenceFeatures.length / FEATURE_STRIDE) + " / "
      + (targetFeatures.length / FEATURE_STRIDE) + " descripteurs retenus.");
    if (referenceFeatures.length < FEATURE_STRIDE || targetFeatures.length < FEATURE_STRIDE * 2) {
      throw new Error("HORG : points-clés détectés " + referenceAll.detected + "/" + targetAll.detected +
        ", descripteurs retenus " + (referenceFeatures.length / FEATURE_STRIDE) + "/" +
        (targetFeatures.length / FEATURE_STRIDE) + "; il en faut au moins deux par image.");
    }

    const referenceCount = referenceFeatures.length / FEATURE_STRIDE;
    const targetCount = targetFeatures.length / FEATURE_STRIDE;
    const referenceCoordinates = featureCoordinates(referenceFeatures);
    const targetCoordinates = featureCoordinates(targetFeatures);
    const localProblem = buildLocalProblem(width, height, settings);
    const seedCount = localProblem.centers.length;
    const referenceDescriptors = new Float32Array(referenceCount * DESCRIPTOR_SIZE);
    const targetDescriptors = new Float32Array(targetCount * DESCRIPTOR_SIZE);
    for (let feature = 0; feature < referenceCount; feature++) {
      referenceDescriptors.set(referenceFeatures.subarray(feature * FEATURE_STRIDE + 4,
        feature * FEATURE_STRIDE + FEATURE_STRIDE), feature * DESCRIPTOR_SIZE);
    }
    for (let feature = 0; feature < targetCount; feature++) {
      targetDescriptors.set(targetFeatures.subarray(feature * FEATURE_STRIDE + 4,
        feature * FEATURE_STRIDE + FEATURE_STRIDE), feature * DESCRIPTOR_SIZE);
    }

    const descriptorBytes = Math.max(referenceDescriptors.byteLength, targetDescriptors.byteLength);
    if (descriptorBytes > device.limits.maxStorageBufferBindingSize) {
      throw new Error("Les descripteurs dépassent la limite de tampon WebGPU de ce périphérique.");
    }
    const referenceBuffer = ownBuffer(device, resources, referenceDescriptors.byteLength,
      bufferUsage.STORAGE | bufferUsage.COPY_DST, "HORG reference descriptor matrix");
    const targetBuffer = ownBuffer(device, resources, targetDescriptors.byteLength,
      bufferUsage.STORAGE | bufferUsage.COPY_DST, "HORG target descriptor matrix");
    device.queue.writeBuffer(referenceBuffer, 0, referenceDescriptors);
    device.queue.writeBuffer(targetBuffer, 0, targetDescriptors);
    const coordinateBufferSize = referenceCoordinates.byteLength + targetCoordinates.byteLength;
    const resultBytes = referenceCount * 16;
    const ratioBytes = referenceCount * 4;
    const candidateBytes = seedCount * MAX_RANSAC_CANDIDATES * 4;
    const ransacResultBytes = seedCount * 32;
    if (Math.max(coordinateBufferSize, resultBytes, ratioBytes, candidateBytes, ransacResultBytes)
        > device.limits.maxStorageBufferBindingSize) {
      throw new Error("Les buffers HORG de correspondance dépassent la limite de stockage WebGPU.");
    }
    const referenceCoordinateBuffer = ownBuffer(device, resources, referenceCoordinates.byteLength,
      bufferUsage.STORAGE | bufferUsage.COPY_DST, "HORG reference feature coordinates");
    const targetCoordinateBuffer = ownBuffer(device, resources, targetCoordinates.byteLength,
      bufferUsage.STORAGE | bufferUsage.COPY_DST, "HORG target feature coordinates");
    const resultBuffer = ownBuffer(device, resources, resultBytes,
      bufferUsage.STORAGE | bufferUsage.COPY_SRC, "HORG top two descriptor matches");
    const ratioMatchesBuffer = ownBuffer(device, resources, resultBytes,
      bufferUsage.STORAGE, "HORG GPU ratio-filtered matches");
    const ratioBuffer = ownBuffer(device, resources, ratioBytes,
      bufferUsage.STORAGE | bufferUsage.COPY_SRC, "HORG GPU ratio diagnostics");
    const ratioReadback = ownBuffer(device, resources, ratioBytes,
      bufferUsage.COPY_DST | bufferUsage.MAP_READ, "HORG ratio diagnostics readback");
    const ransacSeeds = ownBuffer(device, resources, localProblem.seeds.byteLength,
      bufferUsage.STORAGE | bufferUsage.COPY_DST, "HORG RANSAC POI seeds");
    const ransacCandidates = ownBuffer(device, resources, candidateBytes,
      bufferUsage.STORAGE, "HORG GPU local candidate ids");
    const ransacResultsBuffer = ownBuffer(device, resources, ransacResultBytes,
      bufferUsage.STORAGE | bufferUsage.COPY_SRC, "HORG RANSAC affine results");
    const ransacReadback = ownBuffer(device, resources, ransacResultBytes,
      bufferUsage.COPY_DST | bufferUsage.MAP_READ, "HORG RANSAC result readback");
    const consistencyBytes = seedCount * 16;
    const consistencyBuffer = ownBuffer(device, resources, consistencyBytes,
      bufferUsage.STORAGE | bufferUsage.COPY_SRC, "HORG GPU leave-one-out confidence diagnostic");
    const consistencyReadback = ownBuffer(device, resources, consistencyBytes,
      bufferUsage.COPY_DST | bufferUsage.MAP_READ, "HORG confidence diagnostic readback");
    device.queue.writeBuffer(referenceCoordinateBuffer, 0, referenceCoordinates);
    device.queue.writeBuffer(targetCoordinateBuffer, 0, targetCoordinates);
    device.queue.writeBuffer(ransacSeeds, 0, localProblem.seeds);

    const matchParams = makeUniform(device, resources,
      [referenceCount, targetCount, 0, 0], "HORG nearest-neighbour parameters");
    const matchGroup = bindGroup(device, nearest, [
      { binding: 0, resource: { buffer: referenceBuffer } },
      { binding: 1, resource: { buffer: targetBuffer } },
      { binding: 2, resource: { buffer: resultBuffer } },
      { binding: 3, resource: { buffer: matchParams } },
    ]);
    const ratioParams = makeRansacUniform(device, resources,
      [referenceCount, targetCount, 0, 0], [RATIO_LIMIT, 0, 0, 0], "HORG GPU ratio parameters");
    const ratioGroup = bindGroup(device, filterRatios, [
      { binding: 0, resource: { buffer: resultBuffer } },
      { binding: 1, resource: { buffer: referenceCoordinateBuffer } },
      { binding: 2, resource: { buffer: targetCoordinateBuffer } },
      { binding: 3, resource: { buffer: ratioMatchesBuffer } },
      { binding: 4, resource: { buffer: ratioBuffer } },
      { binding: 7, resource: { buffer: ratioParams } },
    ]);
    const candidateParams = makeRansacUniform(device, resources,
      [seedCount, referenceCount, MAX_RANSAC_CANDIDATES, 0],
      [2 * settings.stride, 0, 0, 0], "HORG local candidate parameters");
    const candidateGroup = bindGroup(device, buildCandidates, [
      { binding: 3, resource: { buffer: ratioMatchesBuffer } },
      { binding: 5, resource: { buffer: ransacSeeds } },
      { binding: 6, resource: { buffer: ransacCandidates } },
      { binding: 7, resource: { buffer: candidateParams } },
    ]);

    const matchStart = performance.now();
    const encoder = device.createCommandEncoder({ label: "HORG descriptor matching and local preparation" });
    encodePass(encoder, nearest, matchGroup, referenceCount, 1, 1, "HORG exact k=2 descriptor matching");
    encodePass(encoder, filterRatios, ratioGroup, Math.ceil(referenceCount / 64), 1, 1,
      "HORG GPU Lowe ratio filtering");
    encodePass(encoder, buildCandidates, candidateGroup, seedCount, 1, 1,
      "HORG GPU spatial candidate preparation");
    encoder.copyBufferToBuffer(ratioBuffer, 0, ratioReadback, 0, ratioBytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await ratioReadback.mapAsync(mapMode.READ);
    const ratioValues = new Float32Array(ratioReadback.getMappedRange()).slice();
    ratioReadback.unmap();
    const matchComputeMs = performance.now() - matchStart;
    const ratioData = ratioDiagnostics(ratioValues);

    const ransacParams = makeRansacUniform(device, resources,
      [seedCount, referenceCount, MAX_RANSAC_CANDIDATES, MAX_RANSAC_ITERATIONS],
      [RANSAC_THRESHOLD * reduction, 0, 0, 0], "HORG GPU RANSAC parameters");
    const ransacGroup = bindGroup(device, ransac, [
      { binding: 0, resource: { buffer: ratioMatchesBuffer } },
      { binding: 1, resource: { buffer: ransacSeeds } },
      { binding: 2, resource: { buffer: ransacCandidates } },
      { binding: 3, resource: { buffer: ransacResultsBuffer } },
      { binding: 4, resource: { buffer: ransacParams } },
    ]);
    const consistencyParams = makeRansacUniform(device, resources,
      [seedCount, 0, 0, 0],
      [RANSAC_THRESHOLD, MIN_AFFINE_INLIERS, MIN_LOCAL_CANDIDATES,
        MIN_LOCAL_INLIER_FRACTION], "HORG local confidence parameters");
    const consistencyGroup = bindGroup(device, localConsistency, [
      { binding: 0, resource: { buffer: ransacSeeds } },
      { binding: 1, resource: { buffer: ransacResultsBuffer } },
      { binding: 2, resource: { buffer: consistencyBuffer } },
      { binding: 3, resource: { buffer: consistencyParams } },
    ]);
    const ransacStart = performance.now();
    const ransacEncoder = device.createCommandEncoder({ label: "HORG local affine RANSAC" });
    encodePass(ransacEncoder, ransac, ransacGroup, seedCount, 1, 1, "HORG deterministic GPU RANSAC");
    encodePass(ransacEncoder, localConsistency, consistencyGroup, seedCount, 1, 1,
      "HORG GPU local leave-one-out confidence diagnostic");
    ransacEncoder.copyBufferToBuffer(ransacResultsBuffer, 0, ransacReadback, 0, ransacResultBytes);
    ransacEncoder.copyBufferToBuffer(consistencyBuffer, 0, consistencyReadback, 0, consistencyBytes);
    device.queue.submit([ransacEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await Promise.all([ransacReadback.mapAsync(mapMode.READ), consistencyReadback.mapAsync(mapMode.READ)]);
    const ransacResults = new Float32Array(ransacReadback.getMappedRange()).slice();
    const consistencyResults = new Float32Array(consistencyReadback.getMappedRange()).slice();
    ransacReadback.unmap();
    consistencyReadback.unmap();
    const ransacGpuMs = performance.now() - ransacStart;

    const local = makeLocalField(ratioData.ratioMatches, ratioData.ratioStats,
      localProblem, ransacResults, consistencyResults, settings);
    const accepted = local.matches.reduce((count, match) => count + Number(match.usable), 0);
    return {
      algorithm: "HORG-WebGPU",
      width,
      height,
      patchSize: settings.patchSize,
      stride: settings.stride,
      scale: reduction,
      hierarchyDepth: 0,
      gridWidth: local.gridWidth,
      gridHeight: local.gridHeight,
      count: local.matches.length,
      matches: local.matches,
      keypointsReference: referenceAll.detected,
      keypointsTarget: targetAll.detected,
      keypointOverflow: referenceAll.overflow + targetAll.overflow,
      descriptorsReference: referenceCount,
      descriptorsTarget: targetCount,
      ratioMatches: local.ratioMatches,
      ratioStats: local.ratioStats,
      acceptedInitializations: accepted,
      extractionMs,
      matchComputeMs,
      ransacGpuMs,
      ransacThreshold: RANSAC_THRESHOLD * reduction,
      ratioLimit: RATIO_LIMIT,
      maxFeatures,
    };
  } finally {
    try { await device.queue.onSubmittedWorkDone(); } catch {}
    for (const resource of resources) release(resources, resource);
  }
}

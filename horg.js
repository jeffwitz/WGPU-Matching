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

async function loadShader(device, filename) {
  const response = await fetch(new URL(filename, import.meta.url), { cache: "no-store" });
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

    const counter = ownBuffer(device, resources, 4, bufferUsage.STORAGE | bufferUsage.COPY_SRC,
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

function solve3(matrix, vector) {
  const a = [
    [matrix[0], matrix[1], matrix[2], vector[0]],
    [matrix[3], matrix[4], matrix[5], vector[1]],
    [matrix[6], matrix[7], matrix[8], vector[2]],
  ];
  for (let column = 0; column < 3; column++) {
    let pivot = column;
    for (let row = column + 1; row < 3; row++) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    if (Math.abs(a[pivot][column]) < 1e-10) return null;
    [a[pivot], a[column]] = [a[column], a[pivot]];
    const divisor = a[column][column];
    for (let k = column; k < 4; k++) a[column][k] /= divisor;
    for (let row = 0; row < 3; row++) {
      if (row === column) continue;
      const scale = a[row][column];
      for (let k = column; k < 4; k++) a[row][k] -= scale * a[column][k];
    }
  }
  return [a[0][3], a[1][3], a[2][3]];
}

function fitAffine(matches, indices, center) {
  if (indices.length < 3) return null;
  const normal = new Float64Array(9);
  const rhsX = new Float64Array(3);
  const rhsY = new Float64Array(3);
  for (const index of indices) {
    const match = matches[index];
    const x = match.x - center.x;
    const y = match.y - center.y;
    const terms = [x, y, 1];
    for (let row = 0; row < 3; row++) {
      rhsX[row] += terms[row] * match.targetX;
      rhsY[row] += terms[row] * match.targetY;
      for (let column = 0; column < 3; column++) normal[row * 3 + column] += terms[row] * terms[column];
    }
  }
  const xFit = solve3(normal, rhsX);
  const yFit = solve3(normal, rhsY);
  if (!xFit || !yFit) return null;
  return { x: xFit, y: yFit };
}

function affineError(model, match, center) {
  const x = match.x - center.x;
  const y = match.y - center.y;
  const dx = model.x[0] * x + model.x[1] * y + model.x[2] - match.targetX;
  const dy = model.y[0] * x + model.y[1] * y + model.y[2] - match.targetY;
  return Math.hypot(dx, dy);
}

function ransacLocal(matches, center, stride, seed, threshold) {
  const radius = 2 * stride;
  let candidates = [];
  for (let index = 0; index < matches.length; index++) {
    const match = matches[index];
    const distance = Math.hypot(match.x - center.x, match.y - center.y);
    if (distance <= radius) candidates.push({ index, distance });
  }
  candidates.sort((a, b) => a.distance - b.distance);
  if (candidates.length < 3) {
    const present = new Set(candidates.map(value => value.index));
    const nearest = matches.map((match, index) => ({
      index,
      distance: Math.hypot(match.x - center.x, match.y - center.y),
    })).sort((a, b) => a.distance - b.distance);
    for (const item of nearest) {
      if (!present.has(item.index)) candidates.push(item);
      if (candidates.length >= 3) break;
    }
  }
  const candidateIds = candidates.slice(0, Math.min(96, candidates.length)).map(value => value.index);
  if (candidateIds.length < 3) return null;

  let state = (seed + 1) >>> 0;
  const randomIndex = maximum => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state % maximum;
  };
  let best = [];
  const iterations = Math.min(256, Math.max(64, candidateIds.length * 4));
  for (let iteration = 0; iteration < iterations; iteration++) {
    const a = randomIndex(candidateIds.length);
    let b = randomIndex(candidateIds.length);
    let c = randomIndex(candidateIds.length);
    if (a === b || a === c || b === c) continue;
    const sample = [candidateIds[a], candidateIds[b], candidateIds[c]];
    const model = fitAffine(matches, sample, center);
    if (!model) continue;
    const inliers = [];
    for (const index of candidateIds) {
      if (affineError(model, matches[index], center) <= threshold) inliers.push(index);
    }
    if (inliers.length > best.length) best = inliers;
  }
  if (best.length < 3) return null;
  let model = fitAffine(matches, best, center);
  if (!model) return null;
  best = candidateIds.filter(index => affineError(model, matches[index], center) <= threshold);
  if (best.length < 3) return null;
  model = fitAffine(matches, best, center) || model;
  const residual = best.reduce((sum, index) => sum + affineError(model, matches[index], center), 0) / best.length;
  return {
    dx: model.x[2] - center.x,
    dy: model.y[2] - center.y,
    inliers: best.length,
    candidates: candidateIds.length,
    inlierFraction: best.length / candidateIds.length,
    residual,
  };
}

function makeLocalField(referenceFeatures, targetFeatures, knnResults, width, height, settings) {
  const ratioMatches = [];
  const ratioValues = [];
  for (let index = 0; index < referenceFeatures.length / FEATURE_STRIDE; index++) {
    const firstIndex = Math.round(knnResults[index * 4]);
    const secondIndex = Math.round(knnResults[index * 4 + 1]);
    const firstDistance2 = knnResults[index * 4 + 2];
    const secondDistance2 = knnResults[index * 4 + 3];
    if (firstIndex < 0 || firstIndex >= targetFeatures.length / FEATURE_STRIDE
        || secondIndex < 0 || secondIndex >= targetFeatures.length / FEATURE_STRIDE
        || !(secondDistance2 > 0)) continue;
    const ratio = Math.sqrt(Math.max(0, firstDistance2 / secondDistance2));
    if (Number.isFinite(ratio)) ratioValues.push(ratio);
    if (!(ratio < RATIO_LIMIT)) continue;
    const source = index * FEATURE_STRIDE;
    const target = firstIndex * FEATURE_STRIDE;
    ratioMatches.push({
      x: referenceFeatures[source],
      y: referenceFeatures[source + 1],
      targetX: targetFeatures[target],
      targetY: targetFeatures[target + 1],
      ratio,
    });
  }

  ratioValues.sort((a, b) => a - b);
  const quantile = q => ratioValues.length
    ? ratioValues[Math.floor(q * Math.max(0, ratioValues.length - 1))]
    : Number.NaN;

  const gridWidth = 1 + Math.floor((width - settings.patchSize) / settings.stride);
  const gridHeight = 1 + Math.floor((height - settings.patchSize) / settings.stride);
  if (gridWidth < 1 || gridHeight < 1) throw new Error("La fenêtre DIC est plus grande que l’image.");
  const matches = [];
  for (let gy = 0; gy < gridHeight; gy++) {
    for (let gx = 0; gx < gridWidth; gx++) {
      const center = {
        x: gx * settings.stride + (settings.patchSize - 1) / 2,
        y: gy * settings.stride + (settings.patchSize - 1) / 2,
      };
      // Keypoint coordinates are integer positions in the reduced pyramid. Scale
      // the paper's 1.5-pixel cutoff back to the original image coordinates.
      const fit = ransacLocal(ratioMatches, center, settings.stride,
        gy * gridWidth + gx, RANSAC_THRESHOLD * settings.horgReduction);
      const usable = !!fit && fit.inliers >= MIN_AFFINE_INLIERS
        && fit.candidates >= MIN_LOCAL_CANDIDATES
        && fit.inlierFraction >= MIN_LOCAL_INLIER_FRACTION
        && fit.residual <= RANSAC_THRESHOLD;
      matches.push({
        x: center.x,
        y: center.y,
        dx: fit ? Math.round(fit.dx) : 0,
        dy: fit ? Math.round(fit.dy) : 0,
        score: fit ? fit.inlierFraction : 0,
        secondScore: fit ? fit.residual : Number.NaN,
        margin: fit ? fit.inlierFraction : 0,
        usable,
        inliers: fit ? fit.inliers : 0,
        candidates: fit ? fit.candidates : 0,
        inlierFraction: fit ? fit.inlierFraction : 0,
        residual: fit ? fit.residual : Number.NaN,
      });
    }
  }
  return {
    ratioMatches,
    matches,
    gridWidth,
    gridHeight,
    ratioStats: { min: quantile(0), p25: quantile(0.25), median: quantile(0.5), p90: quantile(0.9) },
  };
}

export async function computeHorg(device, referencePixels, targetPixels,
  width, height, settings, onProgress = () => {}) {
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
    const [pyramidModule, detectModule, descriptorModule, matcherModule] = await Promise.all([
      loadShader(device, "horg-pyramid.wgsl"),
      loadShader(device, "horg-detect.wgsl"),
      loadShader(device, "horg-descriptor.wgsl"),
      loadShader(device, "horg-match.wgsl"),
    ]);
    const pipelines = await Promise.all([
      makePipeline(device, pyramidModule, "downsample_base", "HORG initial image reduction"),
      makePipeline(device, pyramidModule, "downsample_half", "HORG octave reduction"),
      makePipeline(device, pyramidModule, "gaussian_blur", "HORG Gaussian blur"),
      makePipeline(device, pyramidModule, "subtract_gaussian", "HORG DoG"),
      makePipeline(device, detectModule, "detect_extrema", "HORG SIFT keypoint detection"),
      makePipeline(device, descriptorModule, "build_horg", "HORG 64D descriptor"),
      makePipeline(device, matcherModule, "nearest_two", "HORG descriptor nearest neighbours"),
    ]);
    const [downsampleBase, downsampleHalf, blur, subtract, detect, descriptor, nearest] = pipelines;

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
    const resultBytes = referenceCount * 16;
    const resultBuffer = ownBuffer(device, resources, resultBytes,
      bufferUsage.STORAGE | bufferUsage.COPY_SRC, "HORG top two descriptor matches");
    const readback = ownBuffer(device, resources, resultBytes,
      bufferUsage.COPY_DST | bufferUsage.MAP_READ, "HORG match readback");
    const matchParams = makeUniform(device, resources,
      [referenceCount, targetCount, 0, 0], "HORG nearest-neighbour parameters");
    const matchGroup = bindGroup(device, nearest, [
      { binding: 0, resource: { buffer: referenceBuffer } },
      { binding: 1, resource: { buffer: targetBuffer } },
      { binding: 2, resource: { buffer: resultBuffer } },
      { binding: 3, resource: { buffer: matchParams } },
    ]);
    const matchStart = performance.now();
    const encoder = device.createCommandEncoder({ label: "HORG exhaustive GPU descriptor search" });
    encodePass(encoder, nearest, matchGroup, referenceCount, 1, 1, "HORG exact k=2 descriptor matching");
    encoder.copyBufferToBuffer(resultBuffer, 0, readback, 0, resultBytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(mapMode.READ);
    const knnResults = new Float32Array(readback.getMappedRange()).slice();
    readback.unmap();
    const matchComputeMs = performance.now() - matchStart;

    const local = makeLocalField(referenceFeatures, targetFeatures, knnResults,
      width, height, settings);
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
      ratioMatches: local.ratioMatches.length,
      ratioStats: local.ratioStats,
      acceptedInitializations: accepted,
      extractionMs,
      matchComputeMs,
      ransacThreshold: RANSAC_THRESHOLD * reduction,
      ratioLimit: RATIO_LIMIT,
      maxFeatures,
    };
  } finally {
    try { await device.queue.onSubmittedWorkDone(); } catch {}
    for (const resource of resources) release(resources, resource);
  }
}

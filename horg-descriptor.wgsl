struct Params {
    dims: vec4<u32>,
    options: vec4<u32>,
}

struct KeypointBuffer { values: array<vec4<u32>> }
struct FloatBuffer { values: array<f32> }

@group(0) @binding(0) var gaussian2: texture_2d<f32>;
@group(0) @binding(1) var gaussian3: texture_2d<f32>;
@group(0) @binding(2) var gaussian4: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> keypoints: KeypointBuffer;
@group(0) @binding(4) var<storage, read_write> features: FloatBuffer;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> orientation_histogram: array<atomic<u32>, 36>;
var<workgroup> main_orientation: f32;
var<workgroup> descriptor_values: array<f32, 64>;

fn image_value(layer: u32, point: vec2<i32>) -> f32 {
    switch layer {
        case 2u: { return textureLoad(gaussian2, point, 0).x; }
        case 3u: { return textureLoad(gaussian3, point, 0).x; }
        default: { return textureLoad(gaussian4, point, 0).x; }
    }
}

fn sample_gradient(layer: u32, point: vec2<i32>) -> vec2<f32> {
    let width = i32(params.options.w);
    let height = i32(params.dims.y);
    let p = clamp(point, vec2<i32>(1), vec2<i32>(width - 2, height - 2));
    let dx = 0.5 * (image_value(layer, p + vec2<i32>(1, 0)) - image_value(layer, p - vec2<i32>(1, 0)));
    let dy = 0.5 * (image_value(layer, p + vec2<i32>(0, 1)) - image_value(layer, p - vec2<i32>(0, 1)));
    return vec2<f32>(dx, dy);
}

@compute @workgroup_size(64)
fn build_horg(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let feature_index = group.x;
    if feature_index >= params.dims.z { return; }
    let kp = keypoints.values[feature_index];
    let center = vec2<i32>(kp.xy);
    let layer = kp.z;
    let octave_scale = 1.6 * pow(2.0, f32(layer - 2u) / 3.0);
    let orientation_radius = i32(clamp(round(4.5 * octave_scale), 1.0, 32.0));
    let orientation_sigma = 1.5 * octave_scale;
    let diameter = orientation_radius * 2 + 1;

    if lane < 36u { atomicStore(&orientation_histogram[lane], 0u); }
    workgroupBarrier();
    for (var sample = lane; sample < u32(diameter * diameter); sample = sample + 64u) {
        let sx = i32(sample % u32(diameter)) - orientation_radius;
        let sy = i32(sample / u32(diameter)) - orientation_radius;
        if sx * sx + sy * sy > orientation_radius * orientation_radius { continue; }
        let gradient = sample_gradient(layer, center + vec2<i32>(sx, sy));
        let magnitude = length(gradient);
        if magnitude <= 0.0 { continue; }
        var angle = atan2(gradient.y, gradient.x);
        if angle < 0.0 { angle = angle + 6.28318530718; }
        let bin = u32(round(angle * (36.0 / 6.28318530718))) % 36u;
        let radius2 = f32(sx * sx + sy * sy);
        let weight = exp(-radius2 / (2.0 * orientation_sigma * orientation_sigma));
        let quantized = u32(max(1.0, round(magnitude * weight * 100000.0)));
        atomicAdd(&orientation_histogram[bin], quantized);
    }
    workgroupBarrier();

    if lane == 0u {
        var best_bin = 0u;
        var best_value = -1.0;
        var best_left = 0.0;
        var best_right = 0.0;
        for (var bin = 0u; bin < 36u; bin = bin + 1u) {
            let prev2 = f32(atomicLoad(&orientation_histogram[(bin + 34u) % 36u]));
            let prev1 = f32(atomicLoad(&orientation_histogram[(bin + 35u) % 36u]));
            let current = f32(atomicLoad(&orientation_histogram[bin]));
            let next1 = f32(atomicLoad(&orientation_histogram[(bin + 1u) % 36u]));
            let next2 = f32(atomicLoad(&orientation_histogram[(bin + 2u) % 36u]));
            let smoothed = (prev2 + 4.0 * prev1 + 6.0 * current + 4.0 * next1 + next2) / 16.0;
            if smoothed > best_value {
                best_bin = bin;
                best_value = smoothed;
                best_left = (prev2 + 4.0 * prev1 + 6.0 * current + 4.0 * next1 + next2) / 16.0;
                let pprev2 = f32(atomicLoad(&orientation_histogram[(bin + 33u) % 36u]));
                let pprev1 = f32(atomicLoad(&orientation_histogram[(bin + 34u) % 36u]));
                let pnext1 = f32(atomicLoad(&orientation_histogram[(bin + 2u) % 36u]));
                let pnext2 = f32(atomicLoad(&orientation_histogram[(bin + 3u) % 36u]));
                best_left = (pprev2 + 4.0 * pprev1 + 6.0 * f32(atomicLoad(&orientation_histogram[(bin + 35u) % 36u])) + 4.0 * current + next1) / 16.0;
                best_right = (prev1 + 4.0 * current + 6.0 * next1 + 4.0 * pnext1 + pnext2) / 16.0;
            }
        }
        let denom = best_left + best_right - 2.0 * best_value;
        var offset = 0.0;
        if abs(denom) > 1e-8 { offset = clamp(0.5 * (best_left - best_right) / denom, -0.5, 0.5); }
        main_orientation = (f32(best_bin) + offset) * (6.28318530718 / 36.0);
    }
    workgroupBarrier();

    if lane < 16u {
        let row = i32(lane / 4u);
        let column = i32(lane % 4u);
        let cell_width = 3.0 * octave_scale;
        let cell_x = (f32(column) - 1.5) * cell_width;
        let cell_y = (f32(row) - 1.5) * cell_width;
        let radial = normalize(vec2<f32>(-cell_x, -cell_y));
        let branch1 = radial;
        let branch2 = vec2<f32>(-0.5 * radial.x - 0.86602540378 * radial.y,
                                0.86602540378 * radial.x - 0.5 * radial.y);
        let branch3 = vec2<f32>(-0.5 * radial.x + 0.86602540378 * radial.y,
                               -0.86602540378 * radial.x - 0.5 * radial.y);
        let cosine = cos(main_orientation);
        let sine = sin(main_orientation);
        let half = i32(ceil(cell_width * 0.5));
        var magnitude1 = 0.0;
        var magnitude2 = 0.0;
        var magnitude3 = 0.0;
        for (var dy = -half; dy <= half; dy = dy + 1) {
            for (var dx = -half; dx <= half; dx = dx + 1) {
                let u = cell_x + f32(dx);
                let v = cell_y + f32(dy);
                if abs(u - cell_x) > cell_width * 0.5 || abs(v - cell_y) > cell_width * 0.5 { continue; }
                let rotated = vec2<f32>(cosine * u - sine * v, sine * u + cosine * v);
                let sample_point = center + vec2<i32>(round(rotated));
                let gradient_image = sample_gradient(layer, sample_point);
                let gradient = vec2<f32>(cosine * gradient_image.x + sine * gradient_image.y,
                                         -sine * gradient_image.x + cosine * gradient_image.y);
                let weight = exp(-(u * u + v * v) / (8.0 * cell_width * cell_width));
                magnitude1 += weight * max(0.0, dot(gradient, branch1));
                magnitude2 += weight * max(0.0, dot(gradient, branch2));
                magnitude3 += weight * max(0.0, dot(gradient, branch3));
            }
        }

        var mags = array<f32, 3>(magnitude1, magnitude2, magnitude3);
        var dirs = array<vec2<f32>, 3>(branch1, branch2, branch3);
        for (var i = 0u; i < 3u; i = i + 1u) {
            for (var j = i + 1u; j < 3u; j = j + 1u) {
                if mags[j] > mags[i] {
                    let temp_mag = mags[i]; mags[i] = mags[j]; mags[j] = temp_mag;
                    let temp_dir = dirs[i]; dirs[i] = dirs[j]; dirs[j] = temp_dir;
                }
            }
        }
        let vector_l = mags[0] * dirs[0];
        let vector_m = mags[1] * dirs[1];
        let vector_s = mags[2] * dirs[2];
        let d1 = vector_l.x * vector_m.y - vector_l.y * vector_m.x;
        let d2 = vector_m.x * vector_s.y - vector_m.y * vector_s.x;
        let d3 = vector_s.x * vector_l.y - vector_s.y * vector_l.x;
        var chirality = 0.0;
        if d1 > 1e-6 && d2 > 1e-6 && d3 > 1e-6 { chirality = 1.0; }
        if d1 < -1e-6 && d2 < -1e-6 && d3 < -1e-6 { chirality = -1.0; }
        let base = lane * 4u;
        descriptor_values[base] = mags[0];
        descriptor_values[base + 1u] = mags[1];
        descriptor_values[base + 2u] = mags[2];
        descriptor_values[base + 3u] = chirality;
    }
    workgroupBarrier();

    if lane == 0u {
        var norm2 = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) { norm2 += descriptor_values[i] * descriptor_values[i]; }
        let norm = sqrt(norm2);
        var clipped_norm2 = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            var value = 0.0;
            if norm > 1e-12 { value = descriptor_values[i] / norm; }
            if abs(value) > 0.2 { value = sign(value) * 0.2; }
            descriptor_values[i] = value;
            clipped_norm2 += value * value;
        }
        let clipped_norm = sqrt(clipped_norm2);
        if clipped_norm > 1e-12 {
            for (var i = 0u; i < 64u; i = i + 1u) { descriptor_values[i] /= clipped_norm; }
        }
        let output_index = feature_index * 68u;
        let scale_to_input = f32(params.options.x);
        let local_scale = 1.6 * pow(2.0, f32(layer - 2u) / 3.0);
        features.values[output_index] = f32(kp.x) * scale_to_input;
        features.values[output_index + 1u] = f32(kp.y) * scale_to_input;
        features.values[output_index + 2u] = local_scale * scale_to_input;
        features.values[output_index + 3u] = bitcast<f32>(kp.w);
    }
    workgroupBarrier();
    if lane < 64u {
        features.values[feature_index * 68u + 4u + lane] = descriptor_values[lane];
    }
}

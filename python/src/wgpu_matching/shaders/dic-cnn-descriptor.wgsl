struct Params {
    image: vec4<u32>, // width, height, keypoint count, support size
    reserved: vec4<u32>,
}
struct FloatBuffer { values: array<f32> }

@group(0) @binding(0) var<storage, read> image_pixels: FloatBuffer;
@group(0) @binding(1) var<storage, read> keypoints: FloatBuffer;
@group(0) @binding(2) var<storage, read> weights: FloatBuffer;
@group(0) @binding(3) var<storage, read_write> convolution: FloatBuffer;
@group(0) @binding(4) var<storage, read_write> descriptors: FloatBuffer;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> feature_values: array<f32, 2048>;
var<workgroup> pooled_values: array<f32, 128>;
var<workgroup> output_values: array<f32, 64>;

fn pixel(x: i32, y: i32) -> f32 {
    let width = i32(params.image.x);
    let height = i32(params.image.y);
    let px = clamp(x, 0, width - 1);
    let py = clamp(y, 0, height - 1);
    return image_pixels.values[u32(py * width + px)];
}

fn bilinear(x: f32, y: f32) -> f32 {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let wx = x - f32(x0);
    let wy = y - f32(y0);
    return pixel(x0, y0) * (1.0 - wx) * (1.0 - wy)
        + pixel(x0 + 1, y0) * wx * (1.0 - wy)
        + pixel(x0, y0 + 1) * (1.0 - wx) * wy
        + pixel(x0 + 1, y0 + 1) * wx * wy;
}

@compute @workgroup_size(64)
fn first_convolution(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let point_index = group.y;
    if point_index >= params.image.z { return; }
    let feature_index = group.x * 64u + lane;
    if feature_index >= 2048u { return; }
    let channel = feature_index / 256u;
    let spatial = feature_index % 256u;
    let oy = spatial / 16u;
    let ox = spatial % 16u;
    let center = vec2<f32>(keypoints.values[point_index * 2u], keypoints.values[point_index * 2u + 1u]);
    let support = f32(params.image.w);
    let step = support / 16.0;
    let start = 1.0 - support * 0.5;
    var value = weights.values[72u + channel];
    for (var ky = 0u; ky < 3u; ky++) {
        for (var kx = 0u; kx < 3u; kx++) {
            let patch_x = i32(ox * u32(step) + kx) - 1;
            let patch_y = i32(oy * u32(step) + ky) - 1;
            var sample = 0.0;
            if patch_x >= 0 && patch_x < i32(support) && patch_y >= 0 && patch_y < i32(support) {
                let offset = vec2<f32>(start + f32(ox) * step + f32(kx) - 1.0,
                                       start + f32(oy) * step + f32(ky) - 1.0);
                sample = bilinear(center.x + offset.x, center.y + offset.y);
            }
            value += weights.values[channel * 9u + ky * 3u + kx] * sample;
        }
    }
    convolution.values[point_index * 2048u + feature_index] = max(value, 0.0);
}

@compute @workgroup_size(64)
fn pool_project(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let point_index = group.x;
    if point_index >= params.image.z { return; }
    let width = params.image.x;
    let height = params.image.y;
    let center = vec2<f32>(keypoints.values[point_index * 2u], keypoints.values[point_index * 2u + 1u]);
    let support = f32(params.image.w);
    let half_support = support * 0.5;
    if center.x < half_support - 1.0 || center.y < half_support - 1.0
        || center.x + half_support >= f32(width) || center.y + half_support >= f32(height) {
        if lane < 64u {
            for (var d = lane; d < 64u; d += 64u) {
                descriptors.values[point_index * 64u + d] = 0.0;
            }
        }
        return;
    }
    let base = point_index * 2048u;
    for (var i = lane; i < 2048u; i += 64u) {
        feature_values[i] = convolution.values[base + i];
    }
    workgroupBarrier();

    if lane < 64u {
        for (var i = lane; i < 128u; i += 64u) {
            let channel = i / 16u;
            let cell = i % 16u;
            let cell_y = cell / 4u;
            let cell_x = cell % 4u;
            var sum = 0.0;
            for (var py = 0u; py < 4u; py++) {
                for (var px = 0u; px < 4u; px++) {
                    let oy = i32(cell_y * 4u + py);
                    let ox = i32(cell_x * 4u + px);
                    var response = weights.values[152u + channel];
                    for (var ky = 0u; ky < 3u; ky++) {
                        for (var kx = 0u; kx < 3u; kx++) {
                            let sy = oy + i32(ky) - 1;
                            let sx = ox + i32(kx) - 1;
                            if sy >= 0 && sy < 16 && sx >= 0 && sx < 16 {
                                let feature = feature_values[channel * 256u + u32(sy) * 16u + u32(sx)];
                                response += weights.values[80u + channel * 9u + ky * 3u + kx] * feature;
                            }
                        }
                    }
                    sum += max(response, 0.0);
                }
            }
            pooled_values[i] = sum / 16.0;
        }
    }
    workgroupBarrier();

    if lane < 64u {
        var value = weights.values[8352u + lane];
        for (var i = 0u; i < 128u; i++) {
            value += weights.values[160u + lane * 128u + i] * pooled_values[i];
        }
        output_values[lane] = value;
    }
    workgroupBarrier();
    if lane == 0u {
        var norm2 = 0.0;
        for (var i = 0u; i < 64u; i++) { norm2 += output_values[i] * output_values[i]; }
        let norm = max(sqrt(norm2), 1e-12);
        for (var i = 0u; i < 64u; i++) { output_values[i] /= norm; }
    }
    workgroupBarrier();
    if lane < 64u { descriptors.values[point_index * 64u + lane] = output_values[lane]; }
}

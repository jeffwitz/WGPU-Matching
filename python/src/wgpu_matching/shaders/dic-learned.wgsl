struct Params {
    image_width: u32,
    image_height: u32,
    grid_width: u32,
    grid_height: u32,
    stride: u32,
    radius: u32,
    tile_columns: u32,
    tile_rows: u32,
    max_keypoints: u32,
    roi_enabled: u32,
    minimum_score: f32,
    learned_enabled: u32,
}

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> roi_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> response: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read> weights: array<f32>;

struct FeatureVector {
    values: array<f32, 8>,
}

fn pixel_is_valid(x: i32, y: i32) -> bool {
    if x < 0 || y < 0 || x >= i32(params.image_width) || y >= i32(params.image_height) {
        return false;
    }
    if params.roi_enabled == 0u {
        return true;
    }
    let index = u32(y) * params.image_width + u32(x);
    return ((roi_mask[index >> 5u] >> (index & 31u)) & 1u) != 0u;
}

fn silu(value: f32) -> f32 {
    let bounded = clamp(value, -20.0, 20.0);
    return value / (1.0 + exp(-bounded));
}

fn conv1_vector(x: i32, y: i32) -> FeatureVector {
    var value: array<f32, 8>;
    for (var channel = 0u; channel < 8u; channel = channel + 1u) {
        value[channel] = weights[72u + channel];
    }
    for (var ky = -1; ky <= 1; ky = ky + 1) {
        for (var kx = -1; kx <= 1; kx = kx + 1) {
            let px = x + kx;
            let py = y + ky;
            if px >= 0 && py >= 0
                && px < i32(params.image_width)
                && py < i32(params.image_height) {
                let intensity = textureLoad(image, vec2<i32>(px, py), 0).x;
                let kernel_index = u32((ky + 1) * 3 + kx + 1);
                for (var channel = 0u; channel < 8u; channel = channel + 1u) {
                    value[channel] += intensity * weights[channel * 9u + kernel_index];
                }
            }
        }
    }
    var result: array<f32, 8>;
    for (var channel = 0u; channel < 8u; channel = channel + 1u) {
        result[channel] = silu(value[channel]);
    }
    return FeatureVector(result);
}

@compute @workgroup_size(8, 8, 1)
fn score_keypoints(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.grid_width || gid.y >= params.grid_height {
        return;
    }
    let index = gid.y * params.grid_width + gid.x;
    let center_x = i32(gid.x * params.stride + params.stride / 2u);
    let center_y = i32(gid.y * params.stride + params.stride / 2u);
    if response[index].w == 0.0 || !pixel_is_valid(center_x, center_y) {
        response[index].x = 0.0;
        return;
    }

    var conv1_features: array<f32, 72>;
    for (var location = 0u; location < 9u; location = location + 1u) {
        let dx = i32(location % 3u) - 1;
        let dy = i32(location / 3u) - 1;
        let feature = conv1_vector(center_x + dx, center_y + dy);
        for (var channel = 0u; channel < 8u; channel = channel + 1u) {
            conv1_features[location * 8u + channel] = feature.values[channel];
        }
    }

    var logit = weights[168u];
    for (var channel = 0u; channel < 8u; channel = channel + 1u) {
        var hidden = weights[152u + channel];
        for (var location = 0u; location < 9u; location = location + 1u) {
            hidden += conv1_features[location * 8u + channel]
                * weights[80u + channel * 9u + location];
        }
        logit += silu(hidden) * weights[160u + channel];
    }
    logit = clamp(logit, -20.0, 20.0);
    response[index].x = 1.0 / (1.0 + exp(-logit));
}

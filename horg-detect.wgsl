struct Params {
    dims: vec4<u32>,
    settings: vec4<f32>,
    options: vec4<u32>,
}

struct Counter { value: atomic<u32> }
struct KeypointBuffer { values: array<vec4<u32>> }

@group(0) @binding(0) var dog0: texture_2d<f32>;
@group(0) @binding(1) var dog1: texture_2d<f32>;
@group(0) @binding(2) var dog2: texture_2d<f32>;
@group(0) @binding(3) var dog3: texture_2d<f32>;
@group(0) @binding(4) var dog4: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> counter: Counter;
@group(0) @binding(6) var<storage, read_write> keypoints: KeypointBuffer;
@group(0) @binding(7) var<uniform> params: Params;

fn dog_value(layer: u32, point: vec2<i32>) -> f32 {
    switch layer {
        case 0u: { return textureLoad(dog0, point, 0).x; }
        case 1u: { return textureLoad(dog1, point, 0).x; }
        case 2u: { return textureLoad(dog2, point, 0).x; }
        case 3u: { return textureLoad(dog3, point, 0).x; }
        default: { return textureLoad(dog4, point, 0).x; }
    }
}

@compute @workgroup_size(8, 8)
fn detect_extrema(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = params.dims.x;
    let height = params.dims.y;
    if id.x < 2u || id.y < 2u || id.x + 2u >= width || id.y + 2u >= height || id.z >= 3u {
        return;
    }

    let layer = id.z + 1u;
    let point = vec2<i32>(id.xy);
    let center = dog_value(layer, point);
    if abs(center) < params.settings.x { return; }

    var is_maximum = true;
    var is_minimum = true;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                if dx == 0 && dy == 0 && dz == 0 { continue; }
                let other = dog_value(u32(i32(layer) + dz), point + vec2<i32>(dx, dy));
                if center <= other { is_maximum = false; }
                if center >= other { is_minimum = false; }
            }
        }
    }
    if !is_maximum && !is_minimum { return; }

    let left = dog_value(layer, point + vec2<i32>(-1, 0));
    let right = dog_value(layer, point + vec2<i32>(1, 0));
    let top = dog_value(layer, point + vec2<i32>(0, -1));
    let bottom = dog_value(layer, point + vec2<i32>(0, 1));
    let upper_left = dog_value(layer, point + vec2<i32>(-1, -1));
    let upper_right = dog_value(layer, point + vec2<i32>(1, -1));
    let lower_left = dog_value(layer, point + vec2<i32>(-1, 1));
    let lower_right = dog_value(layer, point + vec2<i32>(1, 1));
    let dxx = left + right - 2.0 * center;
    let dyy = top + bottom - 2.0 * center;
    let dxy = 0.25 * (upper_left - upper_right - lower_left + lower_right);
    let determinant = dxx * dyy - dxy * dxy;
    if determinant <= 0.0 { return; }
    let trace = dxx + dyy;
    if trace * trace / determinant >= params.settings.y { return; }

    let index = atomicAdd(&counter.value, 1u);
    if index < params.options.x {
        keypoints.values[index] = vec4<u32>(id.x, id.y, layer + 1u, bitcast<u32>(abs(center)));
    }
}

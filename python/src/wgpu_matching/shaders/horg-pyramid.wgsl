struct Params {
    dims: vec4<u32>,
    config: vec4<u32>,
    weights0: vec4<f32>,
    weights1: vec4<f32>,
    weights2: vec4<f32>,
    weights3: vec4<f32>,
}

@group(0) @binding(0) var source_image: texture_2d<f32>;
@group(0) @binding(1) var output_image: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

fn gaussian_weight(index: u32) -> f32 {
    switch index {
        case 0u, 1u, 2u, 3u: { return params.weights0[index]; }
        case 4u, 5u, 6u, 7u: { return params.weights1[index - 4u]; }
        case 8u, 9u, 10u, 11u: { return params.weights2[index - 8u]; }
        default: { return params.weights3[index - 12u]; }
    }
}

@compute @workgroup_size(8, 8)
fn downsample_base(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_width = params.dims.z;
    let out_height = params.dims.w;
    if id.x >= out_width || id.y >= out_height { return; }
    let factor = params.config.x;
    var sum = 0.0;
    var samples = 0.0;
    for (var dy = 0u; dy < factor; dy = dy + 1u) {
        for (var dx = 0u; dx < factor; dx = dx + 1u) {
            let x = id.x * factor + dx;
            let y = id.y * factor + dy;
            if x < params.dims.x && y < params.dims.y {
                sum += textureLoad(source_image, vec2<i32>(i32(x), i32(y)), 0).x;
                samples += 1.0;
            }
        }
    }
    textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(sum / max(samples, 1.0), 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn downsample_half(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_width = params.dims.z;
    let out_height = params.dims.w;
    if id.x >= out_width || id.y >= out_height { return; }
    let origin = vec2<i32>(id.xy * 2u);
    var sum = 0.0;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            sum += textureLoad(source_image, origin + vec2<i32>(dx, dy), 0).x;
        }
    }
    textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(sum * 0.25, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn gaussian_blur(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = params.dims.x;
    let height = params.dims.y;
    if id.x >= width || id.y >= height { return; }
    let radius = i32(params.config.x);
    let axis = params.config.y;
    var sum = 0.0;
    for (var offset = -radius; offset <= radius; offset = offset + 1) {
        var point = vec2<i32>(id.xy);
        if axis == 0u {
            point.x += offset;
        } else {
            point.y += offset;
        }
        point = clamp(point, vec2<i32>(0), vec2<i32>(i32(width) - 1, i32(height) - 1));
        let weight_index = u32(offset + radius);
        sum += textureLoad(source_image, point, 0).x * gaussian_weight(weight_index);
    }
    textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(sum, 0.0, 0.0, 1.0));
}

@group(0) @binding(3) var second_image: texture_2d<f32>;
@group(0) @binding(4) var difference_image: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn subtract_gaussian(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = params.dims.x;
    let height = params.dims.y;
    if id.x >= width || id.y >= height { return; }
    let point = vec2<i32>(id.xy);
    let difference = textureLoad(second_image, point, 0).x - textureLoad(source_image, point, 0).x;
    textureStore(difference_image, point, vec4<f32>(difference, 0.0, 0.0, 1.0));
}

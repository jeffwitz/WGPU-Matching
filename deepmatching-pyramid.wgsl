struct Config {
    width: u32,
    height: u32,
    padding0: u32,
    padding1: u32,
}

@group(0) @binding(0) var source_image: texture_2d<f32>;
@group(0) @binding(1) var reduced_image: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> config: Config;

fn mirror(value: i32, extent: i32) -> i32 {
    if value < 0 { return -value; }
    if value >= extent { return 2 * extent - 2 - value; }
    return value;
}

@compute @workgroup_size(16, 16)
fn reduce_half(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= config.width || id.y >= config.height { return; }
    let source_size = vec2<i32>(textureDimensions(source_image));
    let center = vec2<i32>(id.xy) * 2;
    let weights = array<f32, 5>(
        1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0
    );
    var value = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let x = mirror(center.x + dx, source_size.x);
            let y = mirror(center.y + dy, source_size.y);
            value += weights[u32(dx + 2)] * weights[u32(dy + 2)]
                * textureLoad(source_image, vec2<i32>(x, y), 0).x;
        }
    }
    textureStore(reduced_image, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}

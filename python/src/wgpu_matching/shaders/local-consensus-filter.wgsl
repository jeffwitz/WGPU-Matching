// Retain original observed pairs supported by at least one usable local affine.
// Models map source offsets around each seed into absolute target coordinates.
struct Params { count: u32, model_count: u32, radius: f32, threshold: f32 }
@group(0) @binding(0) var<storage, read> pairs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> models: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> retained: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn retain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count { return; }
    let point = pairs[i];
    retained[i] = 0u;
    for (var m = 0u; m < params.model_count; m = m + 1u) {
        let seed = models[3u*m];
        if seed.z < 0.5 { continue; }
        let delta = point.xy - seed.xy;
        if dot(delta, delta) > params.radius * params.radius { continue; }
        let x = models[3u*m+1u].xyz;
        let y = models[3u*m+2u].xyz;
        let predicted = vec2<f32>(dot(x, vec3<f32>(delta, 1.0)),
                                  dot(y, vec3<f32>(delta, 1.0)));
        if distance(predicted, point.zw) <= params.threshold {
            retained[i] = 1u;
            return;
        }
    }
}

struct States { values: array<vec4<f32>> }
struct Uints { values: array<u32> }
struct Config { count: u32, hypotheses: u32, index: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> trial: States;
@group(0) @binding(1) var<storage, read> active_mask: Uints;
@group(0) @binding(2) var<storage, read_write> history: States;
@group(0) @binding(3) var<storage, read_write> counts: Uints;
@group(0) @binding(4) var<uniform> config: Config;
@group(0) @binding(5) var<storage, read_write> selected: States;

@compute @workgroup_size(256)
fn store(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= config.count { return; }
    let a = trial.values[2u * i];
    var b = trial.values[2u * i + 1u];
    let finite = all(abs(a) <= vec4<f32>(3.402823e38)) && all(abs(b.xy) <= vec2<f32>(3.402823e38));
    if active_mask.values[i] == 0u || !finite { b.z = 0.0; }
    let dst = 2u * (config.index * config.count + i);
    history.values[dst] = a;
    history.values[dst + 1u] = b;
}

@compute @workgroup_size(1)
fn select_best() {
    var best = 0u;
    var best_count = 0u;
    for (var h = 0u; h < config.hypotheses; h++) {
        var accepted = 0u;
        for (var i = 0u; i < config.count; i++) {
            if history.values[2u * (h * config.count + i) + 1u].z > 0.5 { accepted++; }
        }
        counts.values[h] = accepted;
        if accepted > best_count { best_count = accepted; best = h; }
    }
    counts.values[config.hypotheses] = best;
}

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= config.count { return; }
    let src = 2u * (counts.values[config.hypotheses] * config.count + i);
    selected.values[2u * i] = history.values[src];
    selected.values[2u * i + 1u] = history.values[src + 1u];
}

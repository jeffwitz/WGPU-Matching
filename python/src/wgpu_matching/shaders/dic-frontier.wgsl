// Jacobi frontier with affine state transport; solver execution is separate.
struct States { values: array<vec4<f32>> }
struct Uints { values: array<u32> }
struct Metadata {
    parent: i32, origin: i32, generation: i32, attempts: u32,
    reason: u32, pad0: u32, pad1: u32, pad2: u32,
}
struct MetaBuffer { values: array<Metadata> }
struct Counts { accepted: atomic<u32>, active_count: atomic<u32> }
struct Config {
    width: u32, height: u32, stride: u32, pass_number: u32,
    tau_u: f32, tau_g: f32, pad0: u32, pad1: u32,
}
@group(0) @binding(0) var<storage, read> snapshot: States;
@group(0) @binding(1) var<storage, read_write> current: States;
@group(0) @binding(2) var<storage, read_write> seeds: States;
// The solver treats any nonzero value as active. We store parent + 1 here.
@group(0) @binding(3) var<storage, read_write> active_mask: Uints;
@group(0) @binding(4) var<storage, read> trial: States;
@group(0) @binding(5) var<storage, read_write> metadata: MetaBuffer;
@group(0) @binding(6) var<storage, read_write> counts: Counts;
@group(0) @binding(7) var<uniform> config: Config;
@group(0) @binding(8) var<storage, read> valid: Uints;

fn propose(index: u32, rank: u32) {
    if index >= config.width * config.height { return; }
    active_mask.values[index] = 0u;
    if valid.values[index] == 0u || current.values[2u * index + 1u].z >= 0.5 { return; }
    let x = i32(index % config.width);
    let y = i32(index / config.width);
    let dx = array<i32, 8>(0, -1, 1, 0, -1, 1, -1, 1);
    let dy = array<i32, 8>(-1, 0, 0, 1, -1, -1, 1, 1);
    var seen = 0u;
    for (var k = 0u; k < 8u; k++) {
        let xx = x + dx[k];
        let yy = y + dy[k];
        if xx < 0 || yy < 0 || xx >= i32(config.width) || yy >= i32(config.height) { continue; }
        let parent = u32(yy) * config.width + u32(xx);
        if snapshot.values[2u * parent + 1u].z < 0.5 { continue; }
        if seen != rank { seen++; continue; }
        let a = snapshot.values[2u * parent];
        let b = snapshot.values[2u * parent + 1u];
        let delta = vec2<f32>(f32(-dx[k] * i32(config.stride)), f32(-dy[k] * i32(config.stride)));
        seeds.values[2u * index] = vec4<f32>(a.x + dot(a.yz, delta), a.yz,
                                               a.w + dot(b.xy, delta));
        seeds.values[2u * index + 1u] = vec4<f32>(b.xy, 1.0, 0.0);
        active_mask.values[index] = parent + 1u;
        atomicAdd(&counts.active_count, 1u);
        return;
    }
}

@compute @workgroup_size(256)
fn propose_first(@builtin(global_invocation_id) id: vec3<u32>) { propose(id.x, 0u); }
@compute @workgroup_size(256)
fn propose_second(@builtin(global_invocation_id) id: vec3<u32>) { propose(id.x, 1u); }

@compute @workgroup_size(256)
fn accept(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= config.width * config.height || active_mask.values[index] == 0u { return; }
    let parent = active_mask.values[index] - 1u;
    let a = trial.values[2u * index];
    let b = trial.values[2u * index + 1u];
    let pa = seeds.values[2u * index];
    let pb = seeds.values[2u * index + 1u];
    metadata.values[index].attempts += 1u;
    let finite = all(abs(a) <= vec4<f32>(3.402823e38)) && all(abs(b.xy) <= vec2<f32>(3.402823e38));
    let du = length(a.xw - pa.xw);
    let dg = length(vec4<f32>(a.yz - pa.yz, b.xy - pb.xy));
    if !finite || !(b.z >= 0.5) { metadata.values[index].reason = 1u; return; }
    if du > config.tau_u { metadata.values[index].reason = 2u; return; }
    if dg > config.tau_g { metadata.values[index].reason = 3u; return; }
    current.values[2u * index] = a;
    current.values[2u * index + 1u] = b;
    metadata.values[index].parent = i32(parent);
    metadata.values[index].origin = metadata.values[parent].origin;
    metadata.values[index].generation = i32(config.pass_number);
    metadata.values[index].reason = 0u;
    atomicAdd(&counts.accepted, 1u);
}

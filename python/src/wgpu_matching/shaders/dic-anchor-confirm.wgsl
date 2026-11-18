// Speculative neighbour solves are replayed in the original parent order.
struct States { values: array<vec4<f32>> }
struct Uints { values: array<u32> }
struct Metadata {
    parent: i32, origin: i32, generation: i32, attempts: u32,
    reason: u32, verified: u32, confirmed: u32, candidate: u32,
}
struct MetaBuffer { values: array<Metadata> }
struct Config {
    width: u32, height: u32, stride: u32, pass_number: u32,
    tau_u: f32, tau_g: f32, direction: u32, pad: u32,
}
@group(0) @binding(0) var<storage, read_write> current: States;
@group(0) @binding(1) var<storage, read_write> seeds: States;
@group(0) @binding(2) var<storage, read_write> active_mask: Uints;
@group(0) @binding(3) var<storage, read> trial: States;
@group(0) @binding(4) var<storage, read_write> history: States;
@group(0) @binding(5) var<storage, read_write> metadata: MetaBuffer;
@group(0) @binding(6) var<storage, read> valid: Uints;
@group(0) @binding(7) var<uniform> config: Config;

const DX = array<i32, 8>(0, -1, 1, 0, -1, 1, -1, 1);
const DY = array<i32, 8>(-1, 0, 0, 1, -1, -1, 1, 1);

// The selected solver result stays on this device. Only accepted and valid
// candidates become anchors; all other state is cleared, as on the host path.
@compute @workgroup_size(256)
fn prepare_selected(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= config.width * config.height { return; }
    let accepted = valid.values[i] != 0u && current.values[2u * i + 1u].z > 0.5;
    metadata.values[i].verified = select(0u, 1u, accepted);
    metadata.values[i].candidate = select(0u, 1u, accepted);
    if !accepted {
        current.values[2u * i] = vec4<f32>(0.0);
        current.values[2u * i + 1u] = vec4<f32>(0.0);
    }
}

fn prediction(parent: u32, direction: u32) -> vec4<f32> {
    let a = current.values[2u * parent];
    let b = current.values[2u * parent + 1u];
    let d = vec2<f32>(f32(DX[direction] * i32(config.stride)), f32(DY[direction] * i32(config.stride)));
    return vec4<f32>(a.x + dot(a.yz, d), a.yz, a.w + dot(b.xy, d));
}

@compute @workgroup_size(256)
fn propose(@builtin(global_invocation_id) id: vec3<u32>) {
    let child = id.x;
    if child >= config.width * config.height { return; }
    active_mask.values[child] = 0u;
    if valid.values[child] == 0u || metadata.values[child].candidate != 0u { return; }
    let x = i32(child % config.width) - DX[config.direction];
    let y = i32(child / config.width) - DY[config.direction];
    if x < 0 || y < 0 || x >= i32(config.width) || y >= i32(config.height) { return; }
    let parent = u32(y) * config.width + u32(x);
    if metadata.values[parent].candidate == 0u { return; }
    seeds.values[2u * child] = prediction(parent, config.direction);
    seeds.values[2u * child + 1u] = vec4<f32>(current.values[2u * parent + 1u].xy, 1.0, 0.0);
    active_mask.values[child] = parent + 1u;
}

@compute @workgroup_size(256)
fn store(@builtin(global_invocation_id) id: vec3<u32>) {
    let child = id.x;
    if child >= config.width * config.height || active_mask.values[child] == 0u { return; }
    let parent = active_mask.values[child] - 1u;
    let index = 2u * (parent * 8u + config.direction);
    history.values[index] = trial.values[2u * child];
    history.values[index + 1u] = trial.values[2u * child + 1u];
}

fn is_consistent(a: vec4<f32>, b: vec4<f32>, pa: vec4<f32>, pb: vec4<f32>) -> bool {
    let finite = all(abs(a) <= vec4<f32>(3.402823e38)) && all(abs(b.xy) <= vec2<f32>(3.402823e38));
    return finite && b.z >= 0.5 && length(a.xw - pa.xw) <= config.tau_u
        && length(vec4<f32>(a.yz - pa.yz, b.xy - pb.xy)) <= config.tau_g;
}

// One invocation preserves the sequential decisions of the existing host
// algorithm. The expensive photometric solves above remain parallel.
@compute @workgroup_size(1)
fn replay() {
    let count = config.width * config.height;
    for (var parent = 0u; parent < count; parent++) {
        if metadata.values[parent].candidate == 0u { continue; }
        let x = i32(parent % config.width);
        let y = i32(parent / config.width);
        let pb = current.values[2u * parent + 1u];
        for (var k = 0u; k < 8u; k++) {
            let xx = x + DX[k]; let yy = y + DY[k];
            if xx < 0 || yy < 0 || xx >= i32(config.width) || yy >= i32(config.height) { continue; }
            let child = u32(yy) * config.width + u32(xx);
            if metadata.values[child].verified == 0u { continue; }
            let a = current.values[2u * child]; let b = current.values[2u * child + 1u];
            // A verified neighbour's status is irrelevant to the coherence test.
            if is_consistent(a, vec4<f32>(b.xy, 1.0, b.w), prediction(parent, k), pb) {
                metadata.values[parent].confirmed = 1u;
                metadata.values[child].confirmed = 1u;
            }
        }
        var good_count = 0u;
        for (var k = 0u; k < 8u; k++) {
            let xx = x + DX[k]; let yy = y + DY[k];
            if xx < 0 || yy < 0 || xx >= i32(config.width) || yy >= i32(config.height) { continue; }
            let child = u32(yy) * config.width + u32(xx);
            if valid.values[child] == 0u || metadata.values[child].verified != 0u { continue; }
            let hi = 2u * (parent * 8u + k);
            if is_consistent(history.values[hi], history.values[hi + 1u], prediction(parent, k), pb) {
                good_count++;
            }
        }
        if good_count < 2u { continue; }
        metadata.values[parent].confirmed = 1u;
        for (var k = 0u; k < 8u; k++) {
            let xx = x + DX[k]; let yy = y + DY[k];
            if xx < 0 || yy < 0 || xx >= i32(config.width) || yy >= i32(config.height) { continue; }
            let child = u32(yy) * config.width + u32(xx);
            if valid.values[child] == 0u || metadata.values[child].verified != 0u { continue; }
            let hi = 2u * (parent * 8u + k);
            let a = history.values[hi]; let b = history.values[hi + 1u];
            if is_consistent(a, b, prediction(parent, k), pb) {
                current.values[2u * child] = a;
                current.values[2u * child + 1u] = b;
                metadata.values[child].verified = 1u;
                metadata.values[child].confirmed = 1u;
            }
        }
    }
    for (var i = 0u; i < count; i++) {
        let confirmed = metadata.values[i].confirmed != 0u;
        current.values[2u * i + 1u].z = select(0.0, 1.0, confirmed);
        metadata.values[i].parent = -1;
        metadata.values[i].origin = select(-1, i32(i), confirmed);
        metadata.values[i].generation = select(-1, 0, confirmed);
        metadata.values[i].attempts = 0u;
        metadata.values[i].reason = 0u;
    }
}

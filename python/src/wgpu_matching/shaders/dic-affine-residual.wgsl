// Global and quadrant zero-mean normalized cross-correlation for affine states.
struct Floats { values: array<f32> }
struct States { values: array<vec4<f32>> }
struct Scores { values: array<vec4<f32>> }
struct Config { width: u32, height: u32, grid_width: u32, count: u32, subset: u32, stride: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> reference: Floats;
@group(0) @binding(1) var<storage, read> target_image: Floats;
@group(0) @binding(2) var<storage, read> states: States;
@group(0) @binding(3) var<storage, read_write> scores: Scores;
@group(0) @binding(4) var<uniform> config: Config;
var<workgroup> sums0: array<vec4<f32>, 320>;
var<workgroup> sums1: array<vec4<f32>, 320>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let id = group.x;
    if id >= config.count { return; }
    let p = states.values[2u * id];
    let q = states.values[2u * id + 1u];
    if q.z < 0.5 {
        if local.x == 0u {
            scores.values[2u * id] = vec4<f32>(-2.0);
            scores.values[2u * id + 1u] = vec4<f32>(-2.0, 0.0, 0.0, 0.0);
        }
        return;
    }
    let gx = id % config.grid_width;
    let gy = id / config.grid_width;
    let origin = vec2<f32>(f32(gx * config.stride), f32(gy * config.stride));
    let half = (f32(config.subset) - 1.0) * 0.5;
    var a: array<vec4<f32>, 5>;
    var b: array<vec4<f32>, 5>;
    for (var k = local.x; k < config.subset * config.subset; k += 64u) {
        let xx = k % config.subset;
        let yy = k / config.subset;
        let refxy = origin + vec2<f32>(f32(xx), f32(yy));
        let d = vec2<f32>(f32(xx) - half, f32(yy) - half);
        let t = refxy + vec2<f32>(p.x + dot(p.yz, d), p.w + dot(q.xy, d));
        let valid = refxy.x >= 0.0 && refxy.y >= 0.0 && refxy.x < f32(config.width)
            && refxy.y < f32(config.height) && t.x >= 0.0 && t.y >= 0.0
            && t.x <= f32(config.width - 1u) && t.y <= f32(config.height - 1u);
        var first = 0.0;
        var second = 0.0;
        if valid {
            first = reference.values[u32(refxy.y) * config.width + u32(refxy.x)];
            let x0 = u32(floor(t.x)); let y0 = u32(floor(t.y));
            let x1 = min(x0 + 1u, config.width - 1u); let y1 = min(y0 + 1u, config.height - 1u);
            let fx = t.x - f32(x0); let fy = t.y - f32(y0);
            let upper = mix(target_image.values[y0 * config.width + x0], target_image.values[y0 * config.width + x1], fx);
            let lower = mix(target_image.values[y1 * config.width + x0], target_image.values[y1 * config.width + x1], fx);
            second = mix(upper, lower, fy);
        }
        let v = select(0.0, 1.0, valid);
        let av = vec4<f32>(first, second, first * first, second * second);
        let bv = vec4<f32>(first * second, v, 1.0, 0.0);
        let quadrant = 1u + select(0u, 1u, xx >= config.subset / 2u) + select(0u, 2u, yy >= config.subset / 2u);
        a[0] += av; b[0] += bv;
        a[quadrant] += av; b[quadrant] += bv;
    }
    for (var r = 0u; r < 5u; r++) {
        sums0[r * 64u + local.x] = a[r];
        sums1[r * 64u + local.x] = b[r];
    }
    workgroupBarrier();
    var offset = 32u;
    loop {
        if offset == 0u { break; }
        if local.x < offset {
            for (var r = 0u; r < 5u; r++) {
                let i = r * 64u + local.x;
                sums0[i] += sums0[i + offset];
                sums1[i] += sums1[i + offset];
            }
        }
        workgroupBarrier();
        offset /= 2u;
    }
    if local.x == 0u {
        var rho: array<f32, 5>;
        var all_valid = true;
        for (var r = 0u; r < 5u; r++) {
            let s = sums0[r * 64u]; let t = sums1[r * 64u];
            let n = t.y;
            let vf = s.z - s.x * s.x / max(n, 1.0);
            let vg = s.w - s.y * s.y / max(n, 1.0);
            let ok = t.y == t.z && n > 0.0 && vf > 1e-8 && vg > 1e-8;
            all_valid = all_valid && ok;
            rho[r] = select(-2.0, clamp((t.x - s.x * s.y / max(n, 1.0)) / sqrt(max(vf * vg, 1e-20)), -1.0, 1.0), ok);
        }
        scores.values[2u * id] = vec4<f32>(rho[0], rho[1], rho[2], rho[3]);
        scores.values[2u * id + 1u] = vec4<f32>(rho[4], select(0.0, 1.0, all_valid), 0.0, 0.0);
    }
}

struct Params {
    counts: vec4<u32>,
    settings: vec4<f32>,
}
struct MatchBuffer { values: array<vec4<f32>> }
struct SeedBuffer { values: array<vec4<f32>> }
struct CandidateBuffer { values: array<u32> }
struct ResultBuffer { values: array<vec4<f32>> }
struct Model { x: vec3<f32>, y: vec3<f32>, valid: u32 }
struct Solution { coefficients: vec3<f32>, valid: u32 }

@group(0) @binding(0) var<storage, read> matches: MatchBuffer;
@group(0) @binding(1) var<storage, read> seeds: SeedBuffer;
@group(0) @binding(2) var<storage, read> candidate_ids: CandidateBuffer;
@group(0) @binding(3) var<storage, read_write> results: ResultBuffer;
@group(0) @binding(4) var<uniform> params: Params;

fn solve3(row0: vec3<f32>, row1: vec3<f32>, row2: vec3<f32>, rhs: vec3<f32>) -> Solution {
    let determinant = row0.x * (row1.y * row2.z - row1.z * row2.y)
        - row0.y * (row1.x * row2.z - row1.z * row2.x)
        + row0.z * (row1.x * row2.y - row1.y * row2.x);
    if abs(determinant) < 1e-12 {
        return Solution(vec3<f32>(0.0), 0u);
    }
    let x = (rhs.x * (row1.y * row2.z - row1.z * row2.y)
        - row0.y * (rhs.y * row2.z - row1.z * rhs.z)
        + row0.z * (rhs.y * row2.y - row1.y * rhs.z)) / determinant;
    let y = (row0.x * (rhs.y * row2.z - row1.z * rhs.z)
        - rhs.x * (row1.x * row2.z - row1.z * row2.x)
        + row0.z * (row1.x * rhs.z - rhs.y * row2.x)) / determinant;
    let z = (row0.x * (row1.y * rhs.z - rhs.y * row2.y)
        - row0.y * (row1.x * rhs.z - rhs.y * row2.x)
        + rhs.x * (row1.x * row2.y - row1.y * row2.x)) / determinant;
    return Solution(vec3<f32>(x, y, z), 1u);
}

fn affine_error(model: Model, point: vec4<f32>, center: vec2<f32>) -> f32 {
    let source = point.xy - center;
    let dx = dot(model.x, vec3<f32>(source, 1.0)) - point.z;
    let dy = dot(model.y, vec3<f32>(source, 1.0)) - point.w;
    return length(vec2<f32>(dx, dy));
}

fn fit_affine(center: vec2<f32>, candidate_start: u32, candidate_count: u32,
    sample_ids: vec3<u32>, mode: u32, input_model: Model, threshold: f32) -> Model {
    var xx = 0.0;
    var xy = 0.0;
    var xsum = 0.0;
    var yy = 0.0;
    var ysum = 0.0;
    var count = 0u;
    var rhs_x = vec3<f32>(0.0);
    var rhs_y = vec3<f32>(0.0);

    for (var k = 0u; k < candidate_count; k = k + 1u) {
        var index = candidate_ids.values[candidate_start + k];
        if mode == 0u {
            if k >= 3u { break; }
            index = sample_ids[k];
        }
        let point = matches.values[index];
        if mode != 0u && affine_error(input_model, point, center) > threshold { continue; }
        let source = point.xy - center;
        xx = xx + source.x * source.x;
        xy = xy + source.x * source.y;
        xsum = xsum + source.x;
        yy = yy + source.y * source.y;
        ysum = ysum + source.y;
        rhs_x = rhs_x + vec3<f32>(source.x * point.z, source.y * point.z, point.z);
        rhs_y = rhs_y + vec3<f32>(source.x * point.w, source.y * point.w, point.w);
        count = count + 1u;
    }
    if count < 3u { return Model(vec3<f32>(0.0), vec3<f32>(0.0), 0u); }
    let row0 = vec3<f32>(xx, xy, xsum);
    let row1 = vec3<f32>(xy, yy, ysum);
    let row2 = vec3<f32>(xsum, ysum, f32(count));
    let x_fit = solve3(row0, row1, row2, rhs_x);
    let y_fit = solve3(row0, row1, row2, rhs_y);
    if x_fit.valid == 0u || y_fit.valid == 0u {
        return Model(vec3<f32>(0.0), vec3<f32>(0.0), 0u);
    }
    return Model(x_fit.coefficients, y_fit.coefficients, 1u);
}

fn random_index(state: ptr<function, u32>, maximum: u32) -> u32 {
    *state = (*state) * 1664525u + 1013904223u;
    return *state % maximum;
}

@compute @workgroup_size(1)
fn ransac_local(@builtin(global_invocation_id) id: vec3<u32>) {
    let seed_index = id.x;
    let result_stride = select(2u, 4u, params.settings.y > 0.5);
    if seed_index >= params.counts.x { return; }
    let metadata = seeds.values[seed_index];
    let center = metadata.xy;
    let candidate_count = u32(metadata.z);
    if candidate_count < 3u {
        results.values[seed_index * result_stride] = vec4<f32>(0.0, 0.0, 0.0, f32(candidate_count));
        results.values[seed_index * result_stride + 1u] = vec4<f32>(0.0, 1e30, 0.0, 0.0);
        return;
    }

    let candidate_start = seed_index * params.counts.z;
    let iterations = min(params.counts.w, max(64u, candidate_count * 4u));
    var state = u32(metadata.w) + 1u;
    var best_count = 0u;
    var best_model = Model(vec3<f32>(0.0), vec3<f32>(0.0), 0u);
    let threshold = params.settings.x;
    for (var iteration = 0u; iteration < iterations; iteration = iteration + 1u) {
        let a = random_index(&state, candidate_count);
        let b = random_index(&state, candidate_count);
        let c = random_index(&state, candidate_count);
        if a == b || a == c || b == c { continue; }
        let sample_ids = vec3<u32>(
            candidate_ids.values[candidate_start + a],
            candidate_ids.values[candidate_start + b],
            candidate_ids.values[candidate_start + c]);
        let model = fit_affine(center, candidate_start, candidate_count,
            sample_ids, 0u, Model(vec3<f32>(0.0), vec3<f32>(0.0), 0u), threshold);
        if model.valid == 0u { continue; }
        var inlier_count = 0u;
        for (var k = 0u; k < candidate_count; k = k + 1u) {
            let index = candidate_ids.values[candidate_start + k];
            if affine_error(model, matches.values[index], center) <= threshold {
                inlier_count = inlier_count + 1u;
            }
        }
        if inlier_count > best_count {
            best_count = inlier_count;
            best_model = model;
        }
    }
    if best_count < 3u || best_model.valid == 0u {
        results.values[seed_index * result_stride] = vec4<f32>(0.0, 0.0, f32(best_count), f32(candidate_count));
        results.values[seed_index * result_stride + 1u] = vec4<f32>(0.0, 1e30, 0.0, 0.0);
        return;
    }

    var model = fit_affine(center, candidate_start, candidate_count, vec3<u32>(0u),
        1u, best_model, threshold);
    if model.valid == 0u {
        results.values[seed_index * result_stride] = vec4<f32>(0.0, 0.0, f32(best_count), f32(candidate_count));
        results.values[seed_index * result_stride + 1u] = vec4<f32>(0.0, 1e30, 0.0, 0.0);
        return;
    }
    var inlier_count = 0u;
    for (var k = 0u; k < candidate_count; k = k + 1u) {
        let index = candidate_ids.values[candidate_start + k];
        if affine_error(model, matches.values[index], center) <= threshold {
            inlier_count = inlier_count + 1u;
        }
    }
    if inlier_count < 3u {
        results.values[seed_index * result_stride] = vec4<f32>(0.0, 0.0, f32(inlier_count), f32(candidate_count));
        results.values[seed_index * result_stride + 1u] = vec4<f32>(0.0, 1e30, 0.0, 0.0);
        return;
    }
    let refined = fit_affine(center, candidate_start, candidate_count, vec3<u32>(0u), 1u, model, threshold);
    if refined.valid != 0u { model = refined; }
    var residual = 0.0;
    inlier_count = 0u;
    for (var k = 0u; k < candidate_count; k = k + 1u) {
        let index = candidate_ids.values[candidate_start + k];
        let error = affine_error(model, matches.values[index], center);
        if error <= threshold {
            residual = residual + error;
            inlier_count = inlier_count + 1u;
        }
    }
    residual = residual / f32(max(inlier_count, 1u));
    results.values[seed_index * result_stride] = vec4<f32>(
        model.x.z - center.x, model.y.z - center.y, f32(inlier_count), f32(candidate_count));
    results.values[seed_index * result_stride + 1u] = vec4<f32>(
        f32(inlier_count) / f32(candidate_count), residual, f32(model.valid), 0.0);
    if params.settings.y > 0.5 {
        results.values[seed_index * result_stride + 2u] = vec4<f32>(model.x, 1.0);
        results.values[seed_index * result_stride + 3u] = vec4<f32>(model.y, 1.0);
    }
}

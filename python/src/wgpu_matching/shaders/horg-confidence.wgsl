struct Params {
    counts: vec4<u32>,
    settings: vec4<f32>,
}
struct SeedBuffer { values: array<vec4<f32>> }
struct ResultBuffer { values: array<vec4<f32>> }
struct DiagnosticBuffer { values: array<vec4<f32>> }
struct Solution { coefficients: vec3<f32>, valid: u32 }

@group(0) @binding(0) var<storage, read> seeds: SeedBuffer;
@group(0) @binding(1) var<storage, read> ransac_results: ResultBuffer;
@group(0) @binding(2) var<storage, read_write> diagnostics: DiagnosticBuffer;
@group(0) @binding(3) var<uniform> params: Params;

fn solve3(row0: vec3<f32>, row1: vec3<f32>, row2: vec3<f32>, rhs: vec3<f32>) -> Solution {
    let determinant = row0.x * (row1.y * row2.z - row1.z * row2.y)
        - row0.y * (row1.x * row2.z - row1.z * row2.x)
        + row0.z * (row1.x * row2.y - row1.y * row2.x);
    if abs(determinant) < 1e-12 { return Solution(vec3<f32>(0.0), 0u); }
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

fn accepted(index: u32) -> bool {
    let primary = ransac_results.values[index * 2u];
    let secondary = ransac_results.values[index * 2u + 1u];
    return secondary.z > 0.5
        && primary.z >= params.settings.y
        && primary.w >= params.settings.z
        && secondary.x >= params.settings.w
        && secondary.y <= params.settings.x;
}

@compute @workgroup_size(1)
fn local_affine_leave_one_out(@builtin(global_invocation_id) id: vec3<u32>) {
    let poi = id.x;
    if poi >= params.counts.x { return; }
    if !accepted(poi) {
        diagnostics.values[poi] = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
        return;
    }

    let center = seeds.values[poi].xy;
    var nearest_distances: array<f32, 8>;
    var nearest_ids: array<u32, 8>;
    var nearest_count = 0u;
    for (var other = 0u; other < params.counts.x; other = other + 1u) {
        if other == poi || !accepted(other) { continue; }
        let delta = seeds.values[other].xy - center;
        let distance_squared = dot(delta, delta);
        var position = nearest_count;
        for (var k = 0u; k < nearest_count; k = k + 1u) {
            if distance_squared < nearest_distances[k] {
                position = k;
                break;
            }
        }
        if position < 8u {
            var end = nearest_count;
            if nearest_count < 8u {
                nearest_count = nearest_count + 1u;
                end = nearest_count - 1u;
            } else {
                end = 7u;
            }
            var shift = end;
            loop {
                if shift <= position { break; }
                nearest_distances[shift] = nearest_distances[shift - 1u];
                nearest_ids[shift] = nearest_ids[shift - 1u];
                shift = shift - 1u;
            }
            nearest_distances[position] = distance_squared;
            nearest_ids[position] = other;
        }
    }
    if nearest_count < 8u {
        diagnostics.values[poi] = vec4<f32>(-1.0, f32(nearest_count), 0.0, 0.0);
        return;
    }

    var xx = 0.0;
    var xy = 0.0;
    var xsum = 0.0;
    var yy = 0.0;
    var ysum = 0.0;
    var rhs_x = vec3<f32>(0.0);
    var rhs_y = vec3<f32>(0.0);
    for (var k = 0u; k < 8u; k = k + 1u) {
        let neighbor = nearest_ids[k];
        let source = seeds.values[neighbor].xy - center;
        let movement = ransac_results.values[neighbor * 2u].xy;
        let destination = seeds.values[neighbor].xy + movement;
        xx = xx + source.x * source.x;
        xy = xy + source.x * source.y;
        xsum = xsum + source.x;
        yy = yy + source.y * source.y;
        ysum = ysum + source.y;
        rhs_x = rhs_x + vec3<f32>(source.x * destination.x, source.y * destination.x, destination.x);
        rhs_y = rhs_y + vec3<f32>(source.x * destination.y, source.y * destination.y, destination.y);
    }
    let row0 = vec3<f32>(xx, xy, xsum);
    let row1 = vec3<f32>(xy, yy, ysum);
    let row2 = vec3<f32>(xsum, ysum, 8.0);
    let fit_x = solve3(row0, row1, row2, rhs_x);
    let fit_y = solve3(row0, row1, row2, rhs_y);
    if fit_x.valid == 0u || fit_y.valid == 0u {
        diagnostics.values[poi] = vec4<f32>(-1.0, 8.0, 0.0, 0.0);
        return;
    }

    let observed = center + ransac_results.values[poi * 2u].xy;
    let predicted = vec2<f32>(fit_x.coefficients.z, fit_y.coefficients.z);
    diagnostics.values[poi] = vec4<f32>(distance(observed, predicted), 8.0, 1.0, 0.0);
}

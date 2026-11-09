const MAX_PATCH_PIXELS: u32 = 225u;

struct Params {
    width: u32,
    height: u32,
    field_width: u32,
    field_height: u32,
    search_radius: u32,
    patch_radius: u32,
    candidate_count: u32,
    patch_count: u32,
}

@group(0) @binding(0) var<storage, read> reference: array<f32>;
@group(0) @binding(1) var<storage, read> target_image: array<f32>;
@group(0) @binding(2) var<storage, read> initial_flow: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> source_values: array<f32>;
@group(0) @binding(4) var<storage, read_write> source_stats: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> candidate_scores: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_flow: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> output_scores: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> output_valid: array<u32>;
@group(0) @binding(9) var<uniform> params: Params;
@group(0) @binding(10) var<storage, read> initial_jacobian: array<vec4<f32>>;

fn sample_reference(x: f32, y: f32) -> f32 {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let tx = x - f32(x0);
    let ty = y - f32(y0);
    let xa = u32(clamp(x0, 0, i32(params.width) - 1));
    let xb = u32(clamp(x0 + 1, 0, i32(params.width) - 1));
    let ya = u32(clamp(y0, 0, i32(params.height) - 1));
    let yb = u32(clamp(y0 + 1, 0, i32(params.height) - 1));
    let a = reference[ya * params.width + xa] * (1.0 - tx)
        + reference[ya * params.width + xb] * tx;
    let b = reference[yb * params.width + xa] * (1.0 - tx)
        + reference[yb * params.width + xb] * tx;
    return a * (1.0 - ty) + b * ty;
}

fn sample_target(x: f32, y: f32) -> f32 {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let tx = x - f32(x0);
    let ty = y - f32(y0);
    let xa = u32(clamp(x0, 0, i32(params.width) - 1));
    let xb = u32(clamp(x0 + 1, 0, i32(params.width) - 1));
    let ya = u32(clamp(y0, 0, i32(params.height) - 1));
    let yb = u32(clamp(y0 + 1, 0, i32(params.height) - 1));
    let a = target_image[ya * params.width + xa] * (1.0 - tx)
        + target_image[ya * params.width + xb] * tx;
    let b = target_image[yb * params.width + xa] * (1.0 - tx)
        + target_image[yb * params.width + xb] * tx;
    return a * (1.0 - ty) + b * ty;
}

fn source_centre(node: u32) -> vec2<f32> {
    let x = node % params.field_width;
    let y = node / params.field_width;
    let sx = f32(params.width) / f32(params.field_width);
    let sy = f32(params.height) / f32(params.field_height);
    return vec2<f32>((f32(x) + 0.5) * sx - 0.5, (f32(y) + 0.5) * sy - 0.5);
}

fn is_inside_patch(centre: vec2<f32>) -> bool {
    let radius = f32(params.patch_radius);
    return centre.x >= radius && centre.y >= radius
        && centre.x < f32(params.width) - radius
        && centre.y < f32(params.height) - radius;
}

fn is_inside_warped_patch(centre: vec2<f32>, jacobian: vec4<f32>) -> bool {
    let extent = f32(params.patch_radius)
        * vec2<f32>(abs(jacobian.x) + abs(jacobian.y), abs(jacobian.z) + abs(jacobian.w));
    return centre.x >= extent.x && centre.y >= extent.y
        && centre.x < f32(params.width) - extent.x
        && centre.y < f32(params.height) - extent.y;
}

fn jacobian_for_mode(node: u32, mode: u32, geometry_count: u32) -> vec4<f32> {
    if geometry_count > 1u && mode == 0u {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    return initial_jacobian[node];
}

@compute @workgroup_size(64)
fn prepare_source(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    let node_count = params.field_width * params.field_height;
    if node >= node_count {
        return;
    }
    let centre = source_centre(node);
    let patch_width = 2u * params.patch_radius + 1u;
    var sum = 0.0;
    var sum2 = 0.0;
    for (var index = 0u; index < params.patch_count; index += 1u) {
        let px = i32(index % patch_width) - i32(params.patch_radius);
        let py = i32(index / patch_width) - i32(params.patch_radius);
        let value = sample_reference(centre.x + f32(px), centre.y + f32(py));
        source_values[node * MAX_PATCH_PIXELS + index] = value;
        sum += value;
        sum2 += value * value;
    }
    let variance = max(sum2 - sum * sum / f32(params.patch_count), 1e-12);
    source_stats[node] = vec2<f32>(sum, variance);
}

@compute @workgroup_size(64)
fn score_candidates(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_index = gid.x;
    let node_count = params.field_width * params.field_height;
    let work_count = node_count * params.candidate_count;
    if flat_index >= work_count {
        return;
    }
    let node = flat_index / params.candidate_count;
    let candidate_id = flat_index % params.candidate_count;
    let side = 2u * params.search_radius + 1u;
    let offsets_per_geometry = side * side;
    let geometry_count = params.candidate_count / offsets_per_geometry;
    let geometry_mode = candidate_id / offsets_per_geometry;
    let offset_id = candidate_id % offsets_per_geometry;
    let dx = i32(offset_id % side) - i32(params.search_radius);
    let dy = i32(offset_id / side) - i32(params.search_radius);
    let base = source_centre(node) + initial_flow[node];
    let candidate = base + vec2<f32>(f32(dx), f32(dy));
    let jacobian = jacobian_for_mode(node, geometry_mode, geometry_count);
    if !is_inside_warped_patch(candidate, jacobian) || !is_inside_patch(source_centre(node)) {
        candidate_scores[flat_index] = -2.0;
        return;
    }
    let patch_width = 2u * params.patch_radius + 1u;
    let stats = source_stats[node];
    var target_sum = 0.0;
    var target_sum2 = 0.0;
    var dot = 0.0;
    for (var index = 0u; index < params.patch_count; index += 1u) {
        let px = i32(index % patch_width) - i32(params.patch_radius);
        let py = i32(index / patch_width) - i32(params.patch_radius);
        let offset = vec2<f32>(f32(px), f32(py));
        let target_offset = vec2<f32>(
            jacobian.x * offset.x + jacobian.y * offset.y,
            jacobian.z * offset.x + jacobian.w * offset.y,
        );
        let value = sample_target(candidate.x + target_offset.x, candidate.y + target_offset.y);
        let source_value = source_values[node * MAX_PATCH_PIXELS + index];
        target_sum += value;
        target_sum2 += value * value;
        dot += source_value * value;
    }
    let inverse_count = 1.0 / f32(params.patch_count);
    let covariance = dot - stats.x * target_sum * inverse_count;
    let target_variance = max(target_sum2 - target_sum * target_sum * inverse_count, 1e-12);
    candidate_scores[flat_index] = covariance / sqrt(max(stats.y * target_variance, 1e-12));
}

fn score_at(node: u32, dx: i32, dy: i32, geometry_mode: u32, geometry_count: u32) -> f32 {
    let radius = i32(params.search_radius);
    if abs(dx) > radius || abs(dy) > radius {
        return -2.0;
    }
    let side = 2u * params.search_radius + 1u;
    let offsets_per_geometry = side * side;
    let candidate_id = geometry_mode * offsets_per_geometry
        + u32(dy + radius) * side + u32(dx + radius);
    return candidate_scores[node * params.candidate_count + candidate_id];
}

fn parabolic_offset(minus: f32, centre: f32, plus: f32) -> f32 {
    let denominator = minus - 2.0 * centre + plus;
    if denominator >= -1e-7 {
        return 0.0;
    }
    return clamp(0.5 * (minus - plus) / denominator, -0.5, 0.5);
}

@compute @workgroup_size(64)
fn reduce_best(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    let node_count = params.field_width * params.field_height;
    if node >= node_count {
        return;
    }
    var best_score = -2.0;
    var second_score = -2.0;
    var best_x = 0;
    var best_y = 0;
    var best_mode = 0u;
    let radius = i32(params.search_radius);
    let side = 2u * params.search_radius + 1u;
    let geometry_count = params.candidate_count / (side * side);
    for (var mode = 0u; mode < geometry_count; mode += 1u) {
        for (var dy = -radius; dy <= radius; dy += 1) {
            for (var dx = -radius; dx <= radius; dx += 1) {
                let score = score_at(node, dx, dy, mode, geometry_count);
                if score > best_score {
                    second_score = best_score;
                    best_score = score;
                    best_x = dx;
                    best_y = dy;
                    best_mode = mode;
                } else if score > second_score {
                    second_score = score;
                }
            }
        }
    }
    let centre = source_centre(node);
    let source_valid = is_inside_patch(centre);
    let jacobian = jacobian_for_mode(node, best_mode, geometry_count);
    let base = centre + initial_flow[node];
    let best_target = base + vec2<f32>(f32(best_x), f32(best_y));
    var subpixel = vec2<f32>(0.0, 0.0);
    if best_x > -radius && best_x < radius {
        subpixel.x = parabolic_offset(
            score_at(node, best_x - 1, best_y, best_mode, geometry_count),
            best_score,
            score_at(node, best_x + 1, best_y, best_mode, geometry_count),
        );
    }
    if best_y > -radius && best_y < radius {
        subpixel.y = parabolic_offset(
            score_at(node, best_x, best_y - 1, best_mode, geometry_count),
            best_score,
            score_at(node, best_x, best_y + 1, best_mode, geometry_count),
        );
    }
    let valid = source_valid && is_inside_warped_patch(best_target, jacobian) && best_score > -1.5;
    output_flow[node] = initial_flow[node] + vec2<f32>(f32(best_x), f32(best_y)) + subpixel;
    output_scores[node] = vec2<f32>(best_score, max(0.0, best_score - second_score));
    output_valid[node] = select(0u, 1u, valid);
}

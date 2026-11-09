struct Params {
    width: u32,
    height: u32,
    field_width: u32,
    field_height: u32,
    search_radius: u32,
    patch_radius: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> reference: array<f32>;
@group(0) @binding(1) var<storage, read> target_image: array<f32>;
@group(0) @binding(2) var<storage, read> initial_flow: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output_flow: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> output_scores: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> output_valid: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read> initial_jacobian: array<vec4<f32>>;

fn sample_bilinear(which_image: u32, x: f32, y: f32) -> f32 {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let tx = x - f32(x0);
    let ty = y - f32(y0);
    let xa = u32(clamp(x0, 0, i32(params.width) - 1));
    let xb = u32(clamp(x0 + 1, 0, i32(params.width) - 1));
    let ya = u32(clamp(y0, 0, i32(params.height) - 1));
    let yb = u32(clamp(y0 + 1, 0, i32(params.height) - 1));
    var a = 0.0;
    var b = 0.0;
    if which_image == 0u {
        a = reference[ya * params.width + xa] * (1.0 - tx)
            + reference[ya * params.width + xb] * tx;
        b = reference[yb * params.width + xa] * (1.0 - tx)
            + reference[yb * params.width + xb] * tx;
    } else {
        a = target_image[ya * params.width + xa] * (1.0 - tx)
            + target_image[ya * params.width + xb] * tx;
        b = target_image[yb * params.width + xa] * (1.0 - tx)
            + target_image[yb * params.width + xb] * tx;
    }
    return a * (1.0 - ty) + b * ty;
}

fn score_candidate(
    centre: vec2<f32>,
    source_values: ptr<function, array<f32, 256>>,
    patch_count: u32,
    source_sum: f32,
    source_sum2: f32,
    source_norm: f32,
    jacobian: vec4<f32>,
) -> f32 {
    let extent = f32(params.patch_radius)
        * vec2<f32>(abs(jacobian.x) + abs(jacobian.y), abs(jacobian.z) + abs(jacobian.w));
    if centre.x < extent.x || centre.y < extent.y
        || centre.x >= f32(params.width) - extent.x
        || centre.y >= f32(params.height) - extent.y {
        return -2.0;
    }
    var target_sum = 0.0;
    var target_sum2 = 0.0;
    var dot = 0.0;
    for (var index = 0u; index < patch_count; index += 1u) {
        let patch_width = 2u * params.patch_radius + 1u;
        let px = i32(index % patch_width) - i32(params.patch_radius);
        let py = i32(index / patch_width) - i32(params.patch_radius);
        let target_offset = vec2<f32>(
            jacobian.x * f32(px) + jacobian.y * f32(py),
            jacobian.z * f32(px) + jacobian.w * f32(py),
        );
        let value = sample_bilinear(1u, centre.x + target_offset.x, centre.y + target_offset.y);
        let source_value = (*source_values)[index];
        target_sum += value;
        target_sum2 += value * value;
        dot += source_value * value;
    }
    let inverse_count = 1.0 / f32(patch_count);
    let covariance = dot - source_sum * target_sum * inverse_count;
    let target_variance = max(target_sum2 - target_sum * target_sum * inverse_count, 1e-12);
    return covariance / sqrt(max(source_norm * target_variance, 1e-12));
}

fn parabolic_offset(minus: f32, centre: f32, plus: f32) -> f32 {
    let denominator = minus - 2.0 * centre + plus;
    if denominator >= -1e-7 {
        return 0.0;
    }
    return clamp(0.5 * (minus - plus) / denominator, -0.5, 0.5);
}

@compute @workgroup_size(64)
fn refine(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let count = params.field_width * params.field_height;
    if index >= count {
        return;
    }
    let feature_x = index % params.field_width;
    let feature_y = index / params.field_width;
    let stride_x = f32(params.width) / f32(params.field_width);
    let stride_y = f32(params.height) / f32(params.field_height);
    let source_centre = vec2<f32>(
        (f32(feature_x) + 0.5) * stride_x - 0.5,
        (f32(feature_y) + 0.5) * stride_y - 0.5,
    );
    let base_centre = source_centre + initial_flow[index];
    let jacobian = initial_jacobian[index];
    let patch_width = 2u * params.patch_radius + 1u;
    let patch_count = patch_width * patch_width;
    var source_values: array<f32, 256>;
    var source_sum = 0.0;
    var source_sum2 = 0.0;
    var patch_index = 0u;
    for (var py = -i32(params.patch_radius); py <= i32(params.patch_radius); py += 1) {
        for (var px = -i32(params.patch_radius); px <= i32(params.patch_radius); px += 1) {
            let value = sample_bilinear(0u, source_centre.x + f32(px), source_centre.y + f32(py));
            source_values[patch_index] = value;
            source_sum += value;
            source_sum2 += value * value;
            patch_index += 1u;
        }
    }
    let source_norm = max(source_sum2 - source_sum * source_sum / f32(patch_count), 1e-12);
    let radius = i32(params.search_radius);
    var best_score = -2.0;
    var second_score = -2.0;
    var best_x = 0;
    var best_y = 0;
    for (var dy = -radius; dy <= radius; dy += 1) {
        for (var dx = -radius; dx <= radius; dx += 1) {
            let candidate = base_centre + vec2<f32>(f32(dx), f32(dy));
            let score = score_candidate(candidate, &source_values, patch_count, source_sum, source_sum2, source_norm, jacobian);
            if score > best_score {
                second_score = best_score;
                best_score = score;
                best_x = dx;
                best_y = dy;
            } else if score > second_score {
                second_score = score;
            }
        }
    }
    let best_centre = base_centre + vec2<f32>(f32(best_x), f32(best_y));
    var subpixel = vec2<f32>(0.0, 0.0);
    if best_x > -radius && best_x < radius {
        let minus = score_candidate(best_centre + vec2<f32>(-1.0, 0.0), &source_values, patch_count, source_sum, source_sum2, source_norm, jacobian);
        let plus = score_candidate(best_centre + vec2<f32>(1.0, 0.0), &source_values, patch_count, source_sum, source_sum2, source_norm, jacobian);
        subpixel.x = parabolic_offset(minus, best_score, plus);
    }
    if best_y > -radius && best_y < radius {
        let minus = score_candidate(best_centre + vec2<f32>(0.0, -1.0), &source_values, patch_count, source_sum, source_sum2, source_norm, jacobian);
        let plus = score_candidate(best_centre + vec2<f32>(0.0, 1.0), &source_values, patch_count, source_sum, source_sum2, source_norm, jacobian);
        subpixel.y = parabolic_offset(minus, best_score, plus);
    }
    let patch_radius = f32(params.patch_radius);
    let source_valid = source_centre.x >= patch_radius && source_centre.y >= patch_radius
        && source_centre.x < f32(params.width) - patch_radius
        && source_centre.y < f32(params.height) - patch_radius;
    let is_valid = source_valid && best_score > -1.5;
    output_flow[index] = initial_flow[index] + vec2<f32>(f32(best_x), f32(best_y)) + subpixel;
    output_scores[index] = vec2<f32>(best_score, max(0.0, best_score - second_score));
    output_valid[index] = select(0u, 1u, is_valid);
}

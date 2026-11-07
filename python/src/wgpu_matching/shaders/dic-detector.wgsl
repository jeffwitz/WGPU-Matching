struct Params {
    image_width: u32,
    image_height: u32,
    grid_width: u32,
    grid_height: u32,
    stride: u32,
    radius: u32,
    tile_columns: u32,
    tile_rows: u32,
    max_keypoints: u32,
    roi_enabled: u32,
    minimum_eigenvalue: f32,
    learned_enabled: u32,
}

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> roi_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> response: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> peaks: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> selected_flags: array<u32>;
@group(0) @binding(5) var<storage, read_write> selected_points: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> params: Params;

fn pixel_is_valid(x: i32, y: i32) -> bool {
    if x < 0 || y < 0 || x >= i32(params.image_width) || y >= i32(params.image_height) {
        return false;
    }
    if params.roi_enabled == 0u {
        return true;
    }
    let pixel_index = u32(y) * params.image_width + u32(x);
    let word = roi_mask[pixel_index >> 5u];
    let bit = (word >> (pixel_index & 31u)) & 1u;
    return bit != 0u;
}

fn intensity(x: i32, y: i32) -> f32 {
    return textureLoad(image, vec2<i32>(x, y), 0).x;
}

fn has_valid_support(center_x: i32, center_y: i32) -> bool {
    let r = i32(params.radius) + i32(params.stride / 2u) + 1;
    if center_x < r || center_y < r
        || center_x + r >= i32(params.image_width)
        || center_y + r >= i32(params.image_height) {
        return false;
    }
    if params.roi_enabled == 0u {
        return true;
    }

    // Check the complete finite-difference support, including the ROI boundary.
    for (var y = center_y - r; y <= center_y + r; y = y + 1) {
        for (var x = center_x - r; x <= center_x + r; x = x + 1) {
            if !pixel_is_valid(x, y) {
                return false;
            }
        }
    }
    return true;
}

@compute @workgroup_size(8, 8, 1)
fn compute_tensor_response(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.grid_width || gid.y >= params.grid_height {
        return;
    }
    let index = gid.y * params.grid_width + gid.x;
    selected_flags[index] = 0u;

    let center_x = i32(gid.x * params.stride + params.stride / 2u);
    let center_y = i32(gid.y * params.stride + params.stride / 2u);
    if !has_valid_support(center_x, center_y) || !pixel_is_valid(center_x, center_y) {
        response[index] = vec4<f32>(0.0);
        return;
    }

    let radius = i32(params.radius);
    var h_xx = 0.0;
    var h_xy = 0.0;
    var h_yy = 0.0;
    var sample_count = 0.0;
    for (var dy = -radius; dy <= radius; dy = dy + 2) {
        for (var dx = -radius; dx <= radius; dx = dx + 2) {
            let x = center_x + dx;
            let y = center_y + dy;
            let gx = 0.5 * (intensity(x + 1, y) - intensity(x - 1, y));
            let gy = 0.5 * (intensity(x, y + 1) - intensity(x, y - 1));
            h_xx = h_xx + gx * gx;
            h_xy = h_xy + gx * gy;
            h_yy = h_yy + gy * gy;
            sample_count = sample_count + 1.0;
        }
    }

    h_xx = h_xx / sample_count;
    h_xy = h_xy / sample_count;
    h_yy = h_yy / sample_count;
    let trace = h_xx + h_yy;
    let discriminant = sqrt(max(0.0, (h_xx - h_yy) * (h_xx - h_yy) + 4.0 * h_xy * h_xy));
    let lambda_min = max(0.0, 0.5 * (trace - discriminant));
    let lambda_max = max(lambda_min, 0.5 * (trace + discriminant));
    let condition = lambda_max / max(lambda_min, 1e-12);
    response[index] = vec4<f32>(lambda_min, lambda_max, condition, 1.0);
}

@compute @workgroup_size(64, 1, 1)
fn suppress_nonmaxima(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let grid_count = params.grid_width * params.grid_height;
    if index >= grid_count {
        return;
    }
    let value = response[index];
    if value.w == 0.0 || value.x < params.minimum_eigenvalue {
        peaks[index] = vec4<f32>(0.0);
        return;
    }

    let x = i32(index % params.grid_width);
    let y = i32(index / params.grid_width);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= i32(params.grid_width) || ny >= i32(params.grid_height) {
                continue;
            }
            let neighbor_index = u32(ny) * params.grid_width + u32(nx);
            let neighbor = response[neighbor_index];
            if neighbor.w > 0.0
                && (neighbor.x > value.x || (neighbor.x == value.x && neighbor_index < index)) {
                peaks[index] = vec4<f32>(0.0);
                return;
            }
        }
    }
    peaks[index] = value;
}

fn parabolic_offset(left: f32, center: f32, right: f32) -> f32 {
    let denominator = left - 2.0 * center + right;
    if abs(denominator) < 1e-12 {
        return 0.0;
    }
    return clamp(0.5 * (left - right) / denominator, -0.5, 0.5);
}

@compute @workgroup_size(1, 1, 1)
fn select_spatial_quotas(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_count = params.tile_columns * params.tile_rows;
    let tile_index = gid.x;
    if tile_index >= tile_count {
        return;
    }
    let base_quota = params.max_keypoints / tile_count;
    let remainder = params.max_keypoints % tile_count;
    let quota = base_quota + select(0u, 1u, tile_index < remainder);
    let output_start = tile_index * base_quota + min(tile_index, remainder);

    let tile_x = tile_index % params.tile_columns;
    let tile_y = tile_index / params.tile_columns;
    let x0 = tile_x * params.grid_width / params.tile_columns;
    let x1 = (tile_x + 1u) * params.grid_width / params.tile_columns;
    let y0 = tile_y * params.grid_height / params.tile_rows;
    let y1 = (tile_y + 1u) * params.grid_height / params.tile_rows;

    for (var rank = 0u; rank < quota; rank = rank + 1u) {
        var best_index = 0xffffffffu;
        var best_score = params.minimum_eigenvalue;
        for (var y = y0; y < y1; y = y + 1u) {
            for (var x = x0; x < x1; x = x + 1u) {
                let index = y * params.grid_width + x;
                let candidate = peaks[index];
                if candidate.w == 0.0 || selected_flags[index] != 0u {
                    continue;
                }
                if candidate.x > best_score || (candidate.x == best_score && index < best_index) {
                    best_score = candidate.x;
                    best_index = index;
                }
            }
        }

        let output_index = output_start + rank;
        if best_index == 0xffffffffu {
            selected_points[output_index * 2u] = vec4<f32>(-1.0, -1.0, -1.0, 0.0);
            selected_points[output_index * 2u + 1u] = vec4<f32>(0.0);
            continue;
        }
        selected_flags[best_index] = 1u;
        let grid_x = best_index % params.grid_width;
        let grid_y = best_index / params.grid_width;
        let center_score = peaks[best_index].x;
        var offset_x = 0.0;
        var offset_y = 0.0;
        if grid_x > 0u && grid_x + 1u < params.grid_width {
            offset_x = parabolic_offset(
                response[best_index - 1u].x,
                center_score,
                response[best_index + 1u].x,
            );
        }
        if grid_y > 0u && grid_y + 1u < params.grid_height {
            offset_y = parabolic_offset(
                response[best_index - params.grid_width].x,
                center_score,
                response[best_index + params.grid_width].x,
            );
        }
        let x_px = f32(grid_x * params.stride + params.stride / 2u) + offset_x * f32(params.stride);
        let y_px = f32(grid_y * params.stride + params.stride / 2u) + offset_y * f32(params.stride);
        let condition = peaks[best_index].z;
        let lambda_max = peaks[best_index].y;
        let lambda_min = select(
            center_score,
            lambda_max / max(condition, 1e-12),
            params.learned_enabled != 0u,
        );
        selected_points[output_index * 2u] = vec4<f32>(x_px, y_px, center_score, condition);
        selected_points[output_index * 2u + 1u] = vec4<f32>(lambda_min, lambda_max, condition, 0.0);
    }
}

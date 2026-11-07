struct Params {
    counts: vec4<u32>,
    settings: vec4<f32>,
}
struct NeighborBuffer { values: array<vec4<f32>> }
struct CoordinateBuffer { values: array<vec2<f32>> }
struct MatchBuffer { values: array<vec4<f32>> }
struct RatioBuffer { values: array<f32> }
struct SeedBuffer { values: array<vec4<f32>> }
struct CandidateBuffer { values: array<u32> }

@group(0) @binding(0) var<storage, read> neighbors: NeighborBuffer;
@group(0) @binding(1) var<storage, read> reference_coordinates: CoordinateBuffer;
@group(0) @binding(2) var<storage, read> target_coordinates: CoordinateBuffer;
@group(0) @binding(3) var<storage, read_write> ratio_matches: MatchBuffer;
@group(0) @binding(4) var<storage, read_write> ratio_values: RatioBuffer;
@group(0) @binding(5) var<storage, read_write> seeds: SeedBuffer;
@group(0) @binding(6) var<storage, read_write> candidate_ids: CandidateBuffer;
@group(0) @binding(7) var<uniform> params: Params;
@group(0) @binding(8) var<storage, read_write> raw_matches: MatchBuffer;

@compute @workgroup_size(64)
fn filter_lowe_ratio(@builtin(global_invocation_id) id: vec3<u32>) {
    let query = id.x;
    if query >= params.counts.x { return; }

    let result = neighbors.values[query];
    let first_index = round(result.x);
    let second_index = round(result.y);
    let valid_indices = first_index >= 0.0 && second_index >= 0.0
        && first_index < f32(params.counts.y) && second_index < f32(params.counts.y);
    if !valid_indices || !(result.w > 0.0) {
        ratio_matches.values[query] = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
        raw_matches.values[query] = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
        ratio_values.values[query] = -1.0;
        return;
    }

    let ratio = sqrt(max(0.0, result.z / result.w));
    if !(ratio >= 0.0) || ratio > 1.0 {
        ratio_matches.values[query] = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
        raw_matches.values[query] = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
        ratio_values.values[query] = -1.0;
        return;
    }
    ratio_values.values[query] = ratio;
    let source = reference_coordinates.values[query];
    let target_coordinate = target_coordinates.values[u32(first_index)];
    raw_matches.values[query] = vec4<f32>(source, target_coordinate);
    if ratio < params.settings.x {
        ratio_matches.values[query] = vec4<f32>(source, target_coordinate);
    } else {
        ratio_matches.values[query] = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
}

@compute @workgroup_size(1)
fn build_local_candidates(@builtin(global_invocation_id) id: vec3<u32>) {
    let seed_index = id.x;
    if seed_index >= params.counts.x { return; }
    let center = seeds.values[seed_index].xy;
    let candidate_capacity = params.counts.z;
    let radius = params.settings.x;
    let radius_squared = radius * radius;

    var local_distances: array<f32, 96>;
    var local_ids: array<u32, 96>;
    var local_count = 0u;
    var in_radius_count = 0u;
    var nearest_distances: array<f32, 8>;
    var nearest_ids: array<u32, 8>;
    var nearest_count = 0u;
    var valid_count = 0u;

    for (var query = 0u; query < params.counts.y; query = query + 1u) {
        let match_value = ratio_matches.values[query];
        if match_value.x < 0.0 { continue; }
        valid_count = valid_count + 1u;
        let delta = match_value.xy - center;
        let distance_squared = dot(delta, delta);

        var nearest_position = nearest_count;
        for (var i = 0u; i < nearest_count; i = i + 1u) {
            if distance_squared < nearest_distances[i] {
                nearest_position = i;
                break;
            }
        }
        if nearest_position < 8u {
            var end = nearest_count;
            if nearest_count < 8u {
                nearest_count = nearest_count + 1u;
                end = nearest_count - 1u;
            } else {
                end = 7u;
            }
            var shift = end;
            loop {
                if shift <= nearest_position { break; }
                nearest_distances[shift] = nearest_distances[shift - 1u];
                nearest_ids[shift] = nearest_ids[shift - 1u];
                shift = shift - 1u;
            }
            nearest_distances[nearest_position] = distance_squared;
            nearest_ids[nearest_position] = query;
        }

        if distance_squared <= radius_squared {
            in_radius_count = in_radius_count + 1u;
            var local_position = local_count;
            for (var i = 0u; i < local_count; i = i + 1u) {
                if distance_squared < local_distances[i] {
                    local_position = i;
                    break;
                }
            }
            if local_position < candidate_capacity {
                var end = local_count;
                if local_count < candidate_capacity {
                    local_count = local_count + 1u;
                    end = local_count - 1u;
                } else {
                    end = candidate_capacity - 1u;
                }
                var shift = end;
                loop {
                    if shift <= local_position { break; }
                    local_distances[shift] = local_distances[shift - 1u];
                    local_ids[shift] = local_ids[shift - 1u];
                    shift = shift - 1u;
                }
                local_distances[local_position] = distance_squared;
                local_ids[local_position] = query;
            }
        }
    }

    let start = seed_index * candidate_capacity;
    for (var i = 0u; i < local_count; i = i + 1u) {
        candidate_ids.values[start + i] = local_ids[i];
    }
    // Match the reference CPU behavior: use global nearest points only when
    // fewer than three points fall inside the local radius, appending them
    // after the radius-sorted points and avoiding duplicates.
    if in_radius_count < 3u {
        for (var nearest = 0u; nearest < nearest_count; nearest = nearest + 1u) {
            if local_count >= 3u || local_count >= candidate_capacity { break; }
            let candidate = nearest_ids[nearest];
            var already_present = false;
            for (var local = 0u; local < local_count; local = local + 1u) {
                if local_ids[local] == candidate { already_present = true; break; }
            }
            if !already_present {
                candidate_ids.values[start + local_count] = candidate;
                local_ids[local_count] = candidate;
                local_count = local_count + 1u;
            }
        }
    }
    seeds.values[seed_index].z = f32(local_count);
}

struct Params { counts: vec4<u32> }
struct DescriptorBuffer { values: array<f32> }
struct ResultBuffer { values: array<vec4<f32>> }

@group(0) @binding(0) var<storage, read> reference_descriptors: DescriptorBuffer;
@group(0) @binding(1) var<storage, read> target_descriptors: DescriptorBuffer;
@group(0) @binding(2) var<storage, read_write> results: ResultBuffer;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> lane_best: array<f32, 64>;
var<workgroup> lane_second: array<f32, 64>;
var<workgroup> lane_best_index: array<u32, 64>;
var<workgroup> lane_second_index: array<u32, 64>;

@compute @workgroup_size(64)
fn nearest_two(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let query = group.x;
    if query >= params.counts.x { return; }
    var best = 1e30;
    var second = 1e30;
    var best_index = 0xffffffffu;
    var second_index = 0xffffffffu;
    for (var candidate = lane; candidate < params.counts.y; candidate = candidate + 64u) {
        var distance2 = 0.0;
        for (var component = 0u; component < 64u; component = component + 1u) {
            let delta = reference_descriptors.values[query * 64u + component]
                - target_descriptors.values[candidate * 64u + component];
            distance2 += delta * delta;
        }
        if distance2 < best {
            second = best;
            second_index = best_index;
            best = distance2;
            best_index = candidate;
        } else if distance2 < second {
            second = distance2;
            second_index = candidate;
        }
    }
    lane_best[lane] = best;
    lane_second[lane] = second;
    lane_best_index[lane] = best_index;
    lane_second_index[lane] = second_index;
    workgroupBarrier();

    if lane == 0u {
        var first_distance = 1e30;
        var second_distance = 1e30;
        var first_index = 0xffffffffu;
        var second_index = 0xffffffffu;
        for (var i = 0u; i < 64u; i = i + 1u) {
            let candidate_distance = lane_best[i];
            if candidate_distance < first_distance {
                second_distance = first_distance;
                second_index = first_index;
                first_distance = candidate_distance;
                first_index = lane_best_index[i];
            } else if candidate_distance < second_distance {
                second_distance = candidate_distance;
                second_index = lane_best_index[i];
            }
            let local_second = lane_second[i];
            if local_second < first_distance {
                second_distance = first_distance;
                second_index = first_index;
                first_distance = local_second;
                first_index = lane_second_index[i];
            } else if local_second < second_distance {
                second_distance = local_second;
                second_index = lane_second_index[i];
            }
        }
        results.values[query] = vec4<f32>(f32(first_index), f32(second_index), first_distance, second_distance);
    }
}

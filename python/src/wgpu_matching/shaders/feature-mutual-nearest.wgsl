struct Params { counts: vec4<u32> }
struct Values { values: array<f32> }
struct Best { values: array<vec2<f32>> }

@group(0) @binding(0) var<storage, read> queries: Values;
@group(0) @binding(1) var<storage, read> candidates: Values;
@group(0) @binding(2) var<storage, read_write> best: Best;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> lane_scores: array<f32, 64>;
var<workgroup> lane_indices: array<u32, 64>;

@compute @workgroup_size(64)
fn nearest(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let query = group.x;
    let dimensions = params.counts.z;
    var best_score = -3.402823e38;
    var best_index = 0xffffffffu;
    if query < params.counts.x {
      for (var candidate = lane; candidate < params.counts.y; candidate += 64u) {
        var score = 0.0;
        for (var d = 0u; d < dimensions; d += 1u) {
            score += queries.values[query * dimensions + d]
                * candidates.values[candidate * dimensions + d];
        }
        if score > best_score {
            best_score = score;
            best_index = candidate;
        }
      }
    }
    lane_scores[lane] = best_score;
    lane_indices[lane] = best_index;
    workgroupBarrier();
    if lane == 0u && query < params.counts.x {
        var result_score = -3.402823e38;
        var result_index = 0xffffffffu;
        for (var i = 0u; i < 64u; i += 1u) {
            if lane_scores[i] > result_score {
                result_score = lane_scores[i];
                result_index = lane_indices[i];
            }
        }
        best.values[query] = vec2<f32>(f32(result_index), result_score);
    }
}

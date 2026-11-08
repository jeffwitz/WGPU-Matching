struct Params {
    dims: vec4<u32>, // count, input dimension, hidden dimension, mode
    options: vec4<u32>, // output dimension, reserved
}
struct FloatBuffer { values: array<f32> }

@group(0) @binding(0) var<storage, read> inputs: FloatBuffer;
@group(0) @binding(1) var<storage, read> weights: FloatBuffer;
@group(0) @binding(2) var<storage, read_write> outputs: FloatBuffer;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> input_values: array<f32, 192>;
var<workgroup> hidden_values: array<f32, 128>;
var<workgroup> output_values: array<f32, 64>;

// The hidden layer has 96 units. A 64-lane group silently left units 64-95
// uninitialized; use 128 lanes so every hidden activation is written.
@compute @workgroup_size(128)
fn project(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let feature = group.x;
    if feature >= params.dims.x { return; }
    let input_dim = params.dims.y;
    let hidden_dim = params.dims.z;
    let mode = params.dims.w;
    let input_offset = feature * input_dim;
    for (var i = lane; i < input_dim; i = i + 64u) {
        input_values[i] = inputs.values[input_offset + i];
    }
    workgroupBarrier();

    if mode == 0u {
        let bias_offset = 64u * input_dim;
        if lane < 64u {
            var value = weights.values[bias_offset + lane];
            for (var i = 0u; i < input_dim; i = i + 1u) {
                value += weights.values[lane * input_dim + i] * input_values[i];
            }
            output_values[lane] = value;
        }
    } else {
        let first_bias = hidden_dim * input_dim;
        if lane < hidden_dim {
            var value = weights.values[first_bias + lane];
            for (var i = 0u; i < input_dim; i = i + 1u) {
                value += weights.values[lane * input_dim + i] * input_values[i];
            }
            hidden_values[lane] = max(value, 0.0);
        }
        workgroupBarrier();
        let second_weight = first_bias + hidden_dim;
        let second_bias = second_weight + 64u * hidden_dim;
        if lane < 64u {
            var value = weights.values[second_bias + lane];
            for (var i = 0u; i < hidden_dim; i = i + 1u) {
                value += weights.values[second_weight + lane * hidden_dim + i] * hidden_values[i];
            }
            if mode == 1u { value += 0.25 * input_values[lane]; }
            output_values[lane] = value;
        }
    }
    workgroupBarrier();

    if lane == 0u {
        var norm2 = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) { norm2 += output_values[i] * output_values[i]; }
        let norm = max(sqrt(norm2), 1e-12);
        for (var i = 0u; i < 64u; i = i + 1u) { output_values[i] /= norm; }
    }
    workgroupBarrier();
    if lane < 64u {
        outputs.values[feature * 64u + lane] = output_values[lane];
    }
}

struct Hypotheses { values: array<vec4<f32>>, }
struct Output { values: array<vec4<f32>>, }
struct Params { counts: vec4<u32>, }
@group(0) @binding(0) var<storage,read> candidates: Hypotheses;
@group(0) @binding(1) var<storage,read_write> best: Output;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(1)
fn select_best() {
    var best_index=0u; var best_count=-1.0;
    for (var i=0u;i<p.counts.x;i=i+1u) {
        let count=candidates.values[i*2u+1u].z;
        if (count>best_count) { best_count=count; best_index=i; }
    }
    best.values[0]=candidates.values[best_index*2u];
    best.values[1]=candidates.values[best_index*2u+1u];
    best.values[2]=vec4<f32>(f32(best_index),best_count,0.0,0.0);
}

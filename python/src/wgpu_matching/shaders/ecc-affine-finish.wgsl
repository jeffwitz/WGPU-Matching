struct Partial { values: array<f32>, }
struct Params { tiles: vec4<u32>, }
@group(0) @binding(0) var<storage,read> partial: Partial;
@group(0) @binding(1) var<storage,read_write> result: Partial;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) gid:vec3<u32>) {
    let component=gid.x;
    if (component>=44u) { return; }
    var value=0.0;
    for (var tile=0u;tile<p.tiles.x;tile=tile+1u) { value=value+partial.values[tile*44u+component]; }
    result.values[component]=value;
}

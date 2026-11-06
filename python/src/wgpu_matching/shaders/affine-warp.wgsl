struct Image { values: array<u32>, }
struct Output { values: array<u32>, }
struct Params { dims:vec4<u32>, a:vec4<f32>, b:vec4<f32>, }
@group(0) @binding(0) var<storage,read> src:Image;
@group(0) @binding(1) var<storage,read_write> dst:Output;
@group(0) @binding(2) var<uniform> p:Params;
fn sample(x:i32,y:i32)->f32 {
    if (x<0 || y<0 || x>=i32(p.dims.x) || y>=i32(p.dims.y)) { return 0.0; }
    return f32(src.values[u32(y)*p.dims.x+u32(x)] & 255u);
}
@compute @workgroup_size(16,16)
fn warp(@builtin(global_invocation_id) id:vec3<u32>) {
    if (id.x>=p.dims.x || id.y>=p.dims.y) { return; }
    let x=f32(id.x); let y=f32(id.y);
    let sx=p.a.x*x+p.a.y*y+p.a.z; let sy=p.a.w*x+p.b.x*y+p.b.y;
    let x0=i32(floor(sx)); let y0=i32(floor(sy)); let fx=sx-f32(x0); let fy=sy-f32(y0);
    let a=sample(x0,y0)*(1.0-fx)+sample(x0+1,y0)*fx;
    let b=sample(x0,y0+1)*(1.0-fx)+sample(x0+1,y0+1)*fx;
    dst.values[id.y*p.dims.x+id.x]=u32(clamp(floor(a*(1.0-fy)+b*fy+0.5),0.0,255.0));
}

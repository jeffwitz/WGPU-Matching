struct PairBuffer { values: array<vec4<f32>>, }
struct ResultBuffer { values: array<vec4<f32>>, }
struct Params { counts: vec4<u32>, settings: vec4<f32>, }
@group(0) @binding(0) var<storage,read> pairs: PairBuffer;
@group(0) @binding(1) var<storage,read_write> output: ResultBuffer;
@group(0) @binding(2) var<uniform> p: Params;

fn next_random(s: ptr<function,u32>) -> u32 {
    *s = (*s) * 1664525u + 1013904223u;
    return *s;
}

@compute @workgroup_size(64)
fn hypotheses(@builtin(global_invocation_id) gid:vec3<u32>) {
    let h=gid.x;
    if (h>=p.counts.y) { return; }
    let n=p.counts.x;
    if (n<3u) { output.values[h*2u]=vec4<f32>(0.0); output.values[h*2u+1u]=vec4<f32>(0.0); return; }
    var rng=(h+1u)*747796405u+2891336453u+p.counts.z*277803737u;
    let i0=next_random(&rng)%n;
    var i1=next_random(&rng)%(n-1u);
    if (i1>=i0) { i1=i1+1u; }
    var i2=next_random(&rng)%(n-2u);
    let lo=min(i0,i1); let hi=max(i0,i1);
    if (i2>=lo) { i2=i2+1u; }
    if (i2>=hi) { i2=i2+1u; }
    let q0=pairs.values[i0]; let q1=pairs.values[i1]; let q2=pairs.values[i2];
    let x0=q0.x; let y0=q0.y; let x1=q1.x; let y1=q1.y; let x2=q2.x; let y2=q2.y;
    let d=x0*(y1-y2)+x1*(y2-y0)+x2*(y0-y1);
    if (abs(d)<1e-5) { output.values[h*2u]=vec4<f32>(0.0); output.values[h*2u+1u]=vec4<f32>(0.0); return; }
    let u0=q0.z; let u1=q1.z; let u2=q2.z; let v0=q0.w; let v1=q1.w; let v2=q2.w;
    let a=(u0*(y1-y2)+u1*(y2-y0)+u2*(y0-y1))/d;
    let b=(u0*(x2-x1)+u1*(x0-x2)+u2*(x1-x0))/d;
    let c=(u0*(x1*y2-x2*y1)+u1*(x2*y0-x0*y2)+u2*(x0*y1-x1*y0))/d;
    let e=(v0*(y1-y2)+v1*(y2-y0)+v2*(y0-y1))/d;
    let f=(v0*(x2-x1)+v1*(x0-x2)+v2*(x1-x0))/d;
    let g=(v0*(x1*y2-x2*y1)+v1*(x2*y0-x0*y2)+v2*(x0*y1-x1*y0))/d;
    var count=0u;
    for (var j=0u;j<n;j=j+1u) {
        let q=pairs.values[j];
        let du=a*q.x+b*q.y+c-q.z; let dv=e*q.x+f*q.y+g-q.w;
        if (du*du+dv*dv<=p.settings.x*p.settings.x) { count=count+1u; }
    }
    output.values[h*2u]=vec4<f32>(a,b,c,e);
    output.values[h*2u+1u]=vec4<f32>(f,g,f32(count),0.0);
}

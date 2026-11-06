struct PairBuffer { values: array<vec4<f32>>, }
struct Output { values: array<vec4<f32>>, }
struct Params { counts: vec4<u32>, affine0: vec4<f32>, affine1: vec4<f32>, settings: vec4<f32>, }
@group(0) @binding(0) var<storage,read> pairs: PairBuffer;
@group(0) @binding(1) var<storage,read_write> output: Output;
@group(0) @binding(2) var<uniform> p: Params;

fn solve3(a0:vec3<f32>,a1:vec3<f32>,a2:vec3<f32>,b:vec3<f32>)->vec3<f32> {
    let d=dot(a0,cross(a1,a2));
    if (abs(d)<1e-12) { return vec3<f32>(0.0); }
    return vec3<f32>(dot(b,cross(a1,a2)),dot(a0,cross(b,a2)),dot(a0,cross(a1,b)))/d;
}

@compute @workgroup_size(1)
fn refit() {
    let n=p.counts.x; let threshold2=p.settings.x*p.settings.x;
    var xx=0.0; var xy=0.0; var x=0.0; var yy=0.0; var y=0.0; var count=0u;
    var xu=0.0; var yu=0.0; var u=0.0; var xv=0.0; var yv=0.0; var v=0.0;
    for (var i=0u;i<n;i=i+1u) {
        let q=pairs.values[i]; let du=p.affine0.x*q.x+p.affine0.y*q.y+p.affine0.z-q.z;
        let dv=p.affine1.x*q.x+p.affine1.y*q.y+p.affine1.z-q.w;
        if (du*du+dv*dv<=threshold2) {
            xx=xx+q.x*q.x; xy=xy+q.x*q.y; x=x+q.x;
            yy=yy+q.y*q.y; y=y+q.y; count=count+1u;
            xu=xu+q.x*q.z; yu=yu+q.y*q.z; u=u+q.z;
            xv=xv+q.x*q.w; yv=yv+q.y*q.w; v=v+q.w;
        }
    }
    let m0=vec3<f32>(xx,xy,x); let m1=vec3<f32>(xy,yy,y); let m2=vec3<f32>(x,y,f32(count));
    let au=solve3(m0,m1,m2,vec3<f32>(xu,yu,u));
    let av=solve3(m0,m1,m2,vec3<f32>(xv,yv,v));
    output.values[0]=vec4<f32>(au,count_f32(count));
    output.values[1]=vec4<f32>(av,0.0);
}
fn count_f32(v:u32)->f32 { return f32(v); }

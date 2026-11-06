struct Image { values: array<u32>, }
struct Partial { values: array<f32>, }
struct Params { dims: vec4<u32>, roi: vec4<u32>, affine0: vec4<f32>, affine1: vec4<f32>, photo: vec4<f32>, }
@group(0) @binding(0) var<storage,read> reference: Image;
@group(0) @binding(1) var<storage,read> moving: Image;
@group(0) @binding(2) var<storage,read_write> partial: Partial;
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> sums: array<array<f32,64>,44>;

fn sample_target(x:f32,y:f32)->f32 {
    if (x<0.0 || y<0.0 || x>=f32(p.dims.x-1u) || y>=f32(p.dims.y-1u)) { return -1.0; }
    let x0=u32(floor(x)); let y0=u32(floor(y)); let fx=x-f32(x0); let fy=y-f32(y0);
    let i=y0*p.dims.x+x0;
    let a=f32(moving.values[i]&255u)*(1.0-fx)+f32(moving.values[i+1u]&255u)*fx;
    let b=f32(moving.values[i+p.dims.x]&255u)*(1.0-fx)+f32(moving.values[i+p.dims.x+1u]&255u)*fx;
    return a*(1.0-fy)+b*fy;
}

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(local_invocation_index) lane:u32, @builtin(workgroup_id) group:vec3<u32>) {
    let width=p.roi.z-p.roi.x;
    let px=group.x*8u+(lane&7u); let py=group.y*8u+(lane>>3u);
    var jac:array<f32,8>; var residual=0.0;
    for (var i=0u;i<8u;i=i+1u) { jac[i]=0.0; }
    if (px<width && py<p.roi.w-p.roi.y) {
        let x=p.roi.x+px; let y=p.roi.y+py;
        let xf=f32(x)-0.5*f32(p.roi.x+p.roi.z); let yf=f32(y)-0.5*f32(p.roi.y+p.roi.w);
        let sx=p.affine0.x*xf+p.affine0.y*yf+p.affine0.z;
        let sy=p.affine0.w*xf+p.affine1.x*yf+p.affine1.y;
        let value=sample_target(sx,sy);
        if (value>=0.0 && sx>=1.0 && sy>=1.0 && sx<f32(p.dims.x-2u) && sy<f32(p.dims.y-2u)) {
            let gx=(sample_target(sx+1.0,sy)-sample_target(sx-1.0,sy))*0.5;
            let gy=(sample_target(sx,sy+1.0)-sample_target(sx,sy-1.0))*0.5;
            let t=f32(reference.values[y*p.dims.x+x]&255u);
            let gain=p.photo.x; let bias=p.photo.y;
            residual=value-gain*t-bias;
            jac[0]=gx*xf; jac[1]=gx*yf; jac[2]=gx;
            jac[3]=gy*xf; jac[4]=gy*yf; jac[5]=gy;
            jac[6]=-t; jac[7]=-1.0;
        }
    }
    var k=0u;
    for (var row=0u;row<8u;row=row+1u) {
        for (var col=row;col<8u;col=col+1u) {
            sums[k][lane]=jac[row]*jac[col]; k=k+1u;
        }
    }
    for (var row=0u;row<8u;row=row+1u) { sums[36u+row][lane]=-jac[row]*residual; }
    workgroupBarrier();
    for (var stride=32u;stride>0u;stride=stride>>1u) {
        if (lane<stride) {
            for (var c=0u;c<44u;c=c+1u) { sums[c][lane]=sums[c][lane]+sums[c][lane+stride]; }
        }
        workgroupBarrier();
    }
    if (lane==0u) {
        let tiles_x=(width+7u)/8u;
        let tile_index=group.y*tiles_x+group.x;
        for (var c=0u;c<44u;c=c+1u) { partial.values[tile_index*44u+c]=sums[c][0]; }
    }
}

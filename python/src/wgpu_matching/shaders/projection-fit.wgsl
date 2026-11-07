struct Params { count: u32, centers: u32, ell: f32, ridge: f32, huber: f32, tolerance:f32, _b:f32, _c:f32 };
@group(0) @binding(0) var<storage,read> obs: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> ctr: array<vec2<f32>>;
@group(0) @binding(2) var<storage,read> density: array<f32>;
@group(0) @binding(3) var<storage,read_write> robust: array<f32>;
@group(0) @binding(4) var<storage,read_write> normal: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;
@group(0) @binding(6) var<storage,read_write> design: array<f32>;
@group(0) @binding(7) var<storage,read_write> status: array<u32>;
@group(0) @binding(8) var<storage,read_write> change: array<atomic<u32>>;
@group(0) @binding(9) var<storage,read_write> residuals: array<f32>;
fn nparam()->u32 { return 3u+p.centers; }
fn stride()->u32 { return nparam()+2u; }
fn design_at(row:u32,col:u32)->f32 { return design[row*nparam()+col]; }
@compute @workgroup_size(1)
fn clear_status(@builtin(global_invocation_id) id:vec3<u32>) {
 if(id.x==0u){status[0]=0u;status[1]=0u;status[2]=0u;}
}
@compute @workgroup_size(1)
fn clear_change(@builtin(global_invocation_id) id:vec3<u32>) {
 if(id.x==0u){atomicStore(&change[0],0u);}
}
@compute @workgroup_size(64)
fn build_design(@builtin(global_invocation_id) id:vec3<u32>) {
 let index=id.x; let n=nparam(); let total=p.count*n;
 if(index>=total){return;} let row=index/n; let col=index%n;
 if(col==0u){design[index]=1.0;return;}
 if(col==1u){design[index]=obs[row].x;return;}
 if(col==2u){design[index]=obs[row].y;return;}
 let delta=obs[row].xy-ctr[col-3u];
 design[index]=exp(-dot(delta,delta)/(2.0*p.ell*p.ell));
}
@compute @workgroup_size(8,8)
fn accumulate(@builtin(global_invocation_id) id:vec3<u32>) {
 if(status[0]!=0u||status[1]!=0u){return;}
 let rows=nparam(); let cols=stride(); let i=id.x; let j=id.y;
 if(i>=rows||j>=cols){return;}
 var sum=0.0;var total_weight=0.0;
 for(var k=0u;k<p.count;k++){
  let w=density[k]*robust[k]; let bi=design_at(k,i);
  total_weight+=w;
  var bj=0.0;
  if(j<rows){bj=design_at(k,j);} else if(j==rows){bj=obs[k].z;} else {bj=obs[k].w;}
  sum+=w*bi*bj;
 }
 if(j==i && i>=3u){sum+=p.ridge*total_weight;}
 normal[i*cols+j]=sum;
}
@compute @workgroup_size(128)
fn solve(@builtin(local_invocation_id) lane:vec3<u32>) {
 if(status[0]!=0u||status[1]!=0u){return;} let n=nparam(); let s=stride();
 for(var k=0u;k<n;k++){
  let diagonal=normal[k*s+k];
  if(!(diagonal>1e-12)){
   if(lane.x==0u){status[0]=1u;}
   storageBarrier();
   workgroupBarrier();
   break;
  }
  if(lane.x==0u){normal[k*s+k]=sqrt(diagonal);}
  storageBarrier();
  workgroupBarrier();
  if(lane.x>k&&lane.x<n){normal[lane.x*s+k]/=normal[k*s+k];}
  storageBarrier();
  workgroupBarrier();
  let tail=n-k-1u;let cells=tail*tail;
  for(var cell=lane.x;cell<cells;cell+=128u){
   let row=k+1u+cell/tail;let col=k+1u+cell%tail;
   normal[row*s+col]-=normal[row*s+k]*normal[col*s+k];
  }
  storageBarrier();
  workgroupBarrier();
 }
 storageBarrier();
 workgroupBarrier();
 if(status[0]!=0u){return;}
 if(lane.x==0u){
  for(var rhs=0u;rhs<2u;rhs++){
   var solution:array<f32,67>;
   for(var i=0u;i<n;i++){
    var value=normal[i*s+n+rhs];
    for(var j=0u;j<i;j++){value-=normal[i*s+j]*solution[j];}
    solution[i]=value/normal[i*s+i];
   }
   for(var step=0u;step<n;step++){
    let i=n-1u-step;var value=solution[i];
    for(var j=i+1u;j<n;j++){value-=normal[j*s+i]*solution[j];}
    solution[i]=value/normal[i*s+i];
   }
   for(var i=0u;i<n;i++){normal[i*s+n+rhs]=solution[i];}
  }
 }
}
@compute @workgroup_size(64)
fn update_weights(@builtin(global_invocation_id) id:vec3<u32>) {
 let k=id.x;if(k>=p.count||status[0]!=0u||status[1]!=0u){return;}let n=nparam();let s=stride();var prediction=vec2<f32>(0.0);
 for(var i=0u;i<n;i++){prediction+=design_at(k,i)*vec2<f32>(normal[i*s+n],normal[i*s+n+1u]);}
 let err=length(prediction-obs[k].zw); residuals[k]=err;
 let target_weight=min(1.0,p.huber/max(err,1e-12));
 let updated=(1.0-p._b)*target_weight+p._b*robust[k];
 let delta=abs(updated-robust[k]);robust[k]=updated;
 atomicMax(&change[0],bitcast<u32>(delta));
}
@compute @workgroup_size(1)
fn check_convergence(@builtin(global_invocation_id) id:vec3<u32>) {
 if(id.x==0u&&status[0]==0u&&status[1]==0u){
  let delta=atomicLoad(&change[0]);status[2]=delta;
  if(bitcast<f32>(delta)<p.tolerance){status[1]=1u;}
 }
}

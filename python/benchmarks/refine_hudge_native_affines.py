"""Native photometric refinement with target-image derivatives and trial steps."""
import json
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from qualify_hudge_rectified_windows import OUTPUT, ROOT, SIZE, HALF, ICGNWGPU, unpack_support, material_mask, Path


def main():
    info = json.loads(Path('results/hudge_hole_moments_20261117/proposals/summary.json').read_text())
    ref,target = [np.asarray(Image.open(p)) for p in info['input_sha256']]
    z = np.load(OUTPUT/'windows.npz')
    x = z['centres']
    # Run on all windows, including previously rejected residual solves. Fall
    # back to the original proposal wherever the rectified fit was rejected.
    baseline = np.load(ROOT/'composition_v2/original_image_verification_04.npz')
    params = np.where(z['accepted'][:,None],z['parameters'],baseline['proposal'][z['indices']]).copy()
    solver = ICGNWGPU(device_type='gpu',subset_size=12,subset_stride=4,interpolation='bspline',motion_model='affine',outlier_policy='disabled')
    solver.set_reference(ref,roi_mask=material_mask(ref),roi_policy='partial',roi_truncation='visibility',iterations=30,tolerance=.001,zncc_min=.65)
    roi = solver.roi
    pick = {int(i):unpack_support(w,SIZE).reshape(-1) for i,w in zip(roi.partial_indices,roi.support_words,strict=True)}
    solver.close()
    s = np.array([pick.get(i,np.ones(144,bool)) for i in z['indices']])
    y0,x0 = np.mgrid[:12,:12]-HALF
    B = np.c_[np.ones(144),x0.ravel(),y0.ravel()]
    offset = B[:,1:]
    xy = (x[:,None]+offset).astype(int)
    a = ref[xy[:,:,1],xy[:,:,0]].astype(float)
    target = target.astype(float)
    def normalize(a,s):
        v=(a-(a*s).sum(1)[:,None]/s.sum(1)[:,None])*s
        norm=np.maximum(np.linalg.norm(v,axis=1),1e-10)
        return v/norm[:,None],norm
    a,_=normalize(a,s)
    def measure(ids,p,derivatives=False):
        q = x[ids,None]+offset + np.einsum('nkj,pj->npk',p.reshape(-1,2,3),B)
        valid = (((q>=0)&(q<np.array(target.shape[::-1])-1)).all(-1)|~s[ids]).all(1)
        q = np.clip(q,0,np.array(target.shape[::-1])-1.000001)
        ix,iy = q.astype(int).transpose(2,0,1)
        fx,fy = (q-q.astype(int)).transpose(2,0,1)
        v00,v01,v10,v11=target[iy,ix],target[iy,ix+1],target[iy+1,ix],target[iy+1,ix+1]
        low=v00+(v01-v00)*fx
        high=v10+(v11-v10)*fx
        b,norm=normalize(low+(high-low)*fy,s[ids])
        score=(a[ids]*b).sum(1)
        if not derivatives:
            return score,valid
        gx=(1-fy)*(v01-v00)+fy*(v11-v10)
        J=np.concatenate([gx[...,None]*B,(high-low)[...,None]*B],axis=-1)
        w=s[ids,...,None]
        J=(J-(J*w).sum(1)[:,None]/w.sum(1)[:,None])*w
        J=(J-b[...,None]*(J*b[...,None]).sum(1)[:,None])/norm[:,None,None]
        return score,valid,J,a[ids]-b
    conv=np.zeros(len(x),bool)
    score=np.zeros(len(x))
    for start in range(0,len(x),256):
        ids=np.arange(start,min(start+256,len(x)))
        for _ in range(40):
            if not len(ids): break
            sc,ok,J,r=measure(ids,params[ids],True)
            H=J.transpose(0,2,1)@J
            dp=np.linalg.solve(H+1e-5*np.eye(6),(J.transpose(0,2,1)@r[...,None]))[:,:,0]
            # Convergence requires a small FULL proposed change across the
            # subset, never just a failed line search or clipped step.
            change=np.max(np.linalg.norm(np.einsum('nkj,pj->npk',dp.reshape(-1,2,3),B),axis=-1),axis=1)
            done=(change<.001)&ok
            conv[ids[done]]=True
            waiting=~done
            moved=np.zeros(len(ids),bool)
            for alpha in (1,.5,.25,.125,.0625):
                loc=np.flatnonzero(waiting)
                if not len(loc): break
                trial=params[ids[loc]]+alpha*dp[loc]
                val,bounds=measure(ids[loc],trial)
                accept=bounds&(val>sc[loc])
                params[ids[loc[accept]]]=trial[accept]
                moved[loc[accept]]=True
                waiting[loc[accept]]=False
            ids=ids[moved]
    for start in range(0,len(x),256):
        ids=np.arange(start,min(start+256,len(x)))
        score[ids],valid=measure(ids,params[ids])
        conv[ids]&=valid
    det=np.linalg.det(np.eye(2)+params[:,[1,2,4,5]].reshape(-1,2,2))
    accepted=conv&(det>0)&(score>=.65)
    radius=np.min([np.linalg.norm(x-h['centre'],axis=1)/np.sqrt(h['area']/np.pi) for h in info['source_holes']],axis=0)
    d={}
    for name,region in [('all',np.ones(len(x),bool)),('crown',(radius>=1)&(radius<1.25))]:
        prior=baseline['accepted'][z['indices']]
        d[name]={k:int((region&v).sum()) for k,v in {'eligible':np.ones(len(x),bool),'baseline':prior,'native_forward_converged':accepted,'retained_union':accepted|prior,'added':accepted&~prior}.items()}
    np.savez_compressed(OUTPUT/'native_forward.npz',centres=x,parameters=params,accepted=accepted,score=score,converged=conv)
    (OUTPUT/'native_forward.json').write_text(json.dumps(d,indent=2))
    print(d,flush=True)

if __name__=='__main__':
    main()

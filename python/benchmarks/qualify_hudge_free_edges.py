"""Material-side matching along the four observed outer edges of plate 191."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, map_coordinates
from scipy.spatial import cKDTree

from hudge_strains_data import material_mask, profiles
from qualify_hudge_rectified_windows import ROOT, OUTPUT, ICGNWGPU, unpack_support, samples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--without-normal',action='store_true')
    args=parser.parse_args()
    folder = ROOT/('free_edges_control_191' if args.without_normal else 'free_edges_191')
    folder.mkdir(exist_ok=True)
    info = json.loads(Path('results/hudge_hole_moments_20261117/proposals/summary.json').read_text())
    ref,target = [np.asarray(Image.open(p)) for p in info['input_sha256']]
    m0,m1 = material_mask(ref),material_mask(target)
    p0,p1 = profiles(binary_fill_holes(m0)),profiles(binary_fill_holes(m1))
    baseline = np.load(OUTPUT/'native_restart_verification.npz')
    measured = baseline['accepted']
    centres = baseline['centres']
    gradients = baseline['parameters'][:,[1,2,4,5]].reshape(-1,2,2)+np.eye(2)
    tree = cKDTree(centres[measured])
    measured_ids = np.flatnonzero(measured)
    candidate = baseline['proposal'].copy()
    size,stride = 12,4
    solver = ICGNWGPU(device_type='gpu',subset_size=size,subset_stride=stride,
                      interpolation='bspline',motion_model='affine',outlier_policy='disabled')
    solver.set_reference(ref,iterations=30,tolerance=.001,zncc_min=.65,
                         roi_mask=m0,roi_policy='partial',roi_truncation='visibility')
    roi=solver.roi
    eligible=roi.selected.ravel()
    band=np.zeros(len(centres),bool)
    proposed=np.zeros_like(band)
    side_id=np.full(len(centres),-1)
    y,x=np.mgrid[:12,:12]-5.5
    offsets=np.stack((x,y),-1).reshape(-1,2)
    supports={int(i):unpack_support(w,12).ravel() for i,w in zip(roi.partial_indices,roi.support_words,strict=True)}
    targetf=target.astype(float)
    materialf=m1.astype(float)
    for i in np.flatnonzero(eligible):
        c=centres[i]
        distance=np.array([abs(c[n]-b[int(round(c[1-n]))]) for n,b in p0])
        distance[~np.isfinite(distance)]=np.inf
        side=int(np.argmin(distance))
        if distance[side]>8: continue
        side_id[i]=side
        band[i]=True
        if measured[i]: continue
        n,src=p0[side]
        t=1-n
        edge=c.copy()
        edge[n]=src[int(round(c[t]))]
        _,near=tree.query(c,k=3)
        basis=[(candidate[i,[0,3]]+c,c,candidate[i,[1,2,4,5]].reshape(2,2)+np.eye(2))]
        for j in measured_ids[near]:
            basis.append((centres[j]+baseline['parameters'][j,[0,3]],centres[j],gradients[j]))
        support=supports.get(i,np.ones(144,bool))
        xy=c+offsets[support]
        signal=ref[xy[:,1].astype(int),xy[:,0].astype(int)].astype(float)
        signal-=signal.mean()
        best=-np.inf
        for mapped,origin,F in basis:
            if np.linalg.det(F)<=0: continue
            # Search along the observed boundary. For every tangent trial the
            # corresponding normal coordinate is read from the target silhouette.
            tangent=(mapped+F@(edge-origin))[t]+np.arange(-36.,37.,1.)
            dst=p1[side][1]
            normal=np.interp(tangent,np.arange(len(dst)),dst,left=np.nan,right=np.nan)
            if args.without_normal:
                normal[np.isfinite(normal)]=(mapped+F@(edge-origin))[n]
            keep=np.isfinite(normal)
            if not keep.any(): continue
            anchor=np.empty((keep.sum(),2))
            anchor[:,t]=tangent[keep]
            anchor[:,n]=normal[keep]
            q=anchor[:,None,:]+(xy-edge)@F.T
            val=samples(targetf,q)
            val-=val.mean(axis=1,keepdims=True)
            scores=(val@signal)/np.maximum(np.linalg.norm(val,axis=1)*np.linalg.norm(signal),1e-12)
            scores[(samples(materialf,q)>.5).mean(1)<.95]=-np.inf
            k=np.argmax(scores)
            if scores[k]>best:
                best=scores[k]
                candidate[i,[0,3]]=anchor[k]+F@(c-edge)-c
                candidate[i,[1,2,4,5]]=(F-np.eye(2)).ravel()
        proposed[i]=np.isfinite(best)
    indices=np.flatnonzero(proposed)
    params=candidate[indices].copy()
    status=np.zeros(len(indices))
    for _ in range(4):
        pending=np.flatnonzero((status==0)&np.isfinite(params).all(1))
        if not len(pending): break
        packed=np.zeros((len(centres),8),np.float32)
        active=np.zeros(len(centres),np.uint32)
        ids=indices[pending]
        packed[ids,:6]=params[pending]
        active[ids]=1
        out=solver._register_seeded_frame(target,packed,active.reshape(roi.selected.shape))
        status[pending]=out.status.ravel()[ids]
        params[pending]=out.parameters.reshape(-1,6)[ids]
    solver.close()
    added=np.zeros_like(band)
    states=baseline['parameters'].copy()
    scores=np.full(len(centres),np.nan)
    for k in np.flatnonzero((status==1)&np.isfinite(params).all(1)):
        i=indices[k]
        c=centres[i]
        s=supports.get(i,np.ones(144,bool))
        F=np.eye(2)+params[k,[1,2,4,5]].reshape(2,2)
        q=c+params[k,[0,3]]+offsets[s]@F.T
        if np.linalg.det(F)<=0 or np.any(q<0) or np.any(q>np.array(target.shape[::-1])-1): continue
        v=samples(targetf,q)
        xy=(c+offsets[s]).astype(int)
        a=ref[xy[:,1],xy[:,0]].astype(float)
        scores[i]=np.corrcoef(a,v)[0,1]
        if scores[i]>=.65:
            added[i]=True
            states[i]=params[k]
    merged=measured|added
    result={}
    for name,part in [('outer',band),('all',eligible)]+[(name,side_id==j) for j,name in enumerate(['left','right','top','bottom'])]:
        result[name]=dict(eligible=int(part.sum()),before=int((part&measured).sum()),attempted=int((part&proposed).sum()),added=int((part&added).sum()),after=int((part&merged).sum()))
    np.savez_compressed(folder/'initialisation.npz',centres=centres,parameters=states,accepted=merged,added=added,score=scores,side=side_id,trial_indices=indices,proposal=candidate[indices],status=status)
    (folder/'counts.json').write_text(json.dumps(result,indent=2))
    img=Image.fromarray(ref[:225,:440]).convert('RGB').resize((880,450))
    draw=ImageDraw.Draw(img)
    for i in np.flatnonzero(band):
        x,y=2*centres[i]
        draw.ellipse((x-2,y-2,x+2,y+2),fill='lime' if added[i] else ('cyan' if measured[i] else 'red'))
    img.save(folder/'outer_centres.png')
    for name,accept in [('before',measured),('after',merged)]:
        total=np.zeros((225,440)); disp=np.zeros((*total.shape,2))
        win=np.exp(-np.sum(offsets**2,axis=1)/18)
        for i in np.flatnonzero(accept):
            x,y=(centres[i]-5.5).astype(int)
            box=np.s_[y:y+12,x:x+12]
            w=(win*supports.get(i,1)).reshape(12,12)
            u=states[i,[0,3]]+offsets@states[i,[1,2,4,5]].reshape(2,2).T
            total[box]+=w
            disp[box]+=w[...,None]*u.reshape(12,12,2)
        y,x=np.indices(total.shape)
        im=samples(target,np.stack((x,y),-1)+disp/np.maximum(total[...,None],1e-12))
        im[total==0]=ref[-1,-1]
        Image.fromarray(im).save(folder/f'{name}.png')
    Image.fromarray(ref[:225,:440]).save(folder/'reference.png')
    Image.open(folder/'reference.png').save(folder/'reference_after.gif',save_all=True,append_images=[Image.open(folder/'after.png')],duration=1000,loop=0)
    print(result)

if __name__=='__main__':
    main()

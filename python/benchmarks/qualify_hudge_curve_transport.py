"""Arc-length transport on matched open sides and closed hole boundaries, plate 191."""
import json
import sys
from pathlib import Path

import numpy as np
from contourpy import contour_generator
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, gaussian_filter1d, label
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import cKDTree

from qualify_hudge_free_edges import profiles
from qualify_hudge_rectified_windows import ROOT, material_mask, samples
from verify_hudge_composed_seeds import main as verify

OUT = ROOT/'curve_transport_191'


class Curve:
    def __init__(self, vertices, closed):
        self.closed = closed
        if closed:
            vertices = vertices[:-1]
            signed = np.sum(vertices[:,0]*np.roll(vertices[:,1],-1)-vertices[:,1]*np.roll(vertices[:,0],-1))
            if signed < 0:
                vertices = vertices[::-1]
            # Reproducible seam on the right, near the mid-height.
            centre = (vertices.max(0)+vertices.min(0))/2
            index = np.lexsort((abs(vertices[:,1]-centre[1]),-vertices[:,0]))[0]
            vertices = np.roll(vertices,-index,axis=0)
        pts = gaussian_filter1d(vertices.astype(float),1,axis=0,mode='wrap' if closed else 'nearest')
        if closed:
            pts=np.vstack((pts,pts[0]))
        else:
            pts[[0,-1]]=vertices[[0,-1]]
        self.dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
        self.length=self.dist[-1]
        self.s=self.dist/self.length
        self.pts=pts
        self.grid=np.linspace(0,1,int(np.ceil(4*self.length))+1,endpoint=not closed)
        self.tree=cKDTree(self.at(self.grid))

    def at(self,s):
        s=np.asarray(s)%1 if self.closed else np.clip(s,0,1)
        return np.stack([np.interp(s,self.s,c) for c in self.pts.T],axis=-1)

    def nearest(self,x):
        d,i=self.tree.query(x)
        return d,self.grid[i]


def contours(mask,holes):
    labels,_=label(~mask)
    curves=[]
    for n,profile in profiles(binary_fill_holes(mask)):
        t=np.flatnonzero(np.isfinite(profile))
        v=np.empty((len(t),2))
        v[:,1-n]=t
        v[:,n]=profile[t]
        curves.append(Curve(v,False))
    for h in holes:
        lines=contour_generator(z=(labels==h['component']).astype(float)).lines(.5)
        curves.append(Curve(max(lines,key=len),True))
    return curves


def correspondence(c0,c1,centres,params,measured,dist,s,part,weighted):
    # Fit unwrapped target arc fraction at knots. Periodic curves may change
    # phase freely; open curves pair the endpoints of each observed side.
    knots=np.linspace(0,1,max(6,int(c0.length/14))+1)
    use=measured & part & (dist<=24)
    x=centres[use]
    anchor=c0.at(s[use])
    F=np.eye(2)+params[use][:,[1,2,4,5]].reshape(-1,2,2)
    predicted=x+params[use][:,[0,3]]+np.einsum('nij,nj->ni',F,anchor-x)
    miss,obs=c1.nearest(predicted)
    if c0.closed:
        obs=s[use]+(obs-s[use]+.5)%1-.5
    kval=knots.copy()
    if weighted and len(obs):
        A=np.stack([np.interp(s[use],knots,col) for col in np.eye(len(knots))],axis=1)
        w=1/(1+(dist[use]/8)**2+(miss/8)**2)
        D=np.diff(np.eye(len(knots)),axis=0)
        smooth=np.diff(D,axis=0)
        if c0.closed:
            smooth=np.vstack((smooth,D[0]-D[-1]))
        def loss(y):
            error=(A@y-obs)*c1.length
            rho=np.sqrt(1+(error/3)**2)
            reg=smooth@y*c1.length
            return (((w*9*(rho-1)).sum()+.5*(reg@reg))/c1.length**2,
                    (A.T@(w*error/rho)+smooth.T@reg)/c1.length)
        bounds=[(None,None)]*len(knots)
        cons=[LinearConstraint(D,.02*np.diff(knots),np.inf)]
        if c0.closed:
            cons.append(LinearConstraint(D.sum(0),1,1))
        else:
            bounds[0]=(0,0)
            bounds[-1]=(1,1)
        res=minimize(loss,knots,method='SLSQP',jac=True,bounds=bounds,constraints=cons,
                     options={'maxiter':300,'ftol':1e-12})
        if not res.success:
            raise RuntimeError(res.message)
        kval=res.x
    return knots,kval


def main():
    OUT.mkdir(exist_ok=True)
    info=json.loads(Path('results/hudge_hole_moments_20261117/proposals/summary.json').read_text())
    ref,target=[np.asarray(Image.open(p)) for p in info['input_sha256']]
    cs=contours(material_mask(ref),[info['source_holes'][i] for i,j,*_ in info['matches']])
    ct=contours(material_mask(target),[info['target_candidates'][j] for i,j,*_ in info['matches']])
    prev=np.load(ROOT/'free_edges_191/initialisation.npz')
    centres,params,accepted=[prev[k] for k in ('centres','parameters','accepted')]
    roi=np.load('results/hudge_boundary_frontier_20261115/partial12/seeds.npz')['selected']
    idx=np.flatnonzero(roi)
    centres,params,accepted=centres[idx],params[idx],accepted[idx]
    ds,ss=np.array([c.nearest(centres) for c in cs]).transpose(1,0,2)
    closest=np.argmin(ds,axis=0)
    dist=ds.min(0)
    band=dist<=8
    near=np.flatnonzero(accepted)
    grad=np.eye(2)+params[near][:,[1,2,4,5]].reshape(-1,2,2)
    _,ix=cKDTree(centres[near]).query(centres)
    F=grad[ix]
    metrics={}
    for mode in ('arc_only','speckle'):
        folder=OUT/mode
        folder.mkdir(exist_ok=True)
        proposals=prev['parameters'].copy()
        traces=[]
        for k,(c0,c1) in enumerate(zip(cs,ct,strict=True)):
            knots,values=correspondence(c0,c1,centres,params,accepted,dist,ss[k],closest==k,mode=='speckle')
            use=(closest==k)&band&~accepted
            s=ss[k,use]
            mapped=np.interp(s,knots,values)
            edge=c0.at(s)
            # Replace the tangential column by the derivative of the fitted
            # boundary correspondence. Retain the neighbour's normal column.
            def tangent(curve,pos):
                v=curve.at(pos+.5/curve.length)-curve.at(pos-.5/curve.length)
                return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9)
            t0,t1=tangent(c0,s),tangent(c1,mapped)
            stretch=(np.diff(values)/np.diff(knots))[np.clip(np.searchsorted(knots,s)-1,0,len(knots)-2)]*c1.length/c0.length
            f=F[use]+(stretch[:,None]*t1-np.einsum('nij,nj->ni',F[use],t0))[:,:,None]*t0[:,None,:]
            q=c1.at(mapped)+np.einsum('nij,nj->ni',f,centres[use]-edge)
            proposals[idx[use],0]=q[:,0]-centres[use,0]
            proposals[idx[use],3]=q[:,1]-centres[use,1]
            proposals[idx[use,None],[1,2,4,5]]=(f-np.eye(2)).reshape(-1,4)
            traces.append(dict(source=c0.at(knots).tolist(),destination=c1.at(values).tolist(),s=knots.tolist(),sigma=values.tolist()))
        (folder/'curves.json').write_text(json.dumps(traces))
        active=prev['accepted'].copy()
        active[idx[band]]=True
        np.savez_compressed(folder/'input.npz',centres=prev['centres'],accepted=active,parameters=proposals)
        sys.argv=['verify',str(ROOT/'composition_v2'),'--seed-archive',str(folder/'input.npz'),'--output',str(folder)]
        verify()
        z=np.load(folder/'native_restart_verification.npz')
        added=z['accepted'][idx]&band&~accepted
        metrics[mode]={n:[int((reg&accepted).sum()),int((reg&added).sum()),int(reg.sum())] for n,reg in [('exterior',band&(closest<4)),('holes',band&(closest>=4))]}
        print(mode,metrics[mode],flush=True)
        if mode=='speckle':
            merged=prev['accepted'].copy()
            merged[idx[added]]=True
            states=prev['parameters'].copy()
            states[idx[added]]=z['parameters'][idx[added]]
            np.savez_compressed(OUT/'initialisation.npz',centres=prev['centres'],accepted=merged,parameters=states)
            im=Image.fromarray(ref[:225,:440]).convert('RGB').resize((880,450))
            draw=ImageDraw.Draw(im)
            for j in np.flatnonzero(band):
                x,y=2*centres[j]
                draw.ellipse((x-2,y-2,x+2,y+2),fill='lime' if added[j] else 'cyan' if accepted[j] else 'red')
            im.save(OUT/'centres.png')
            y,x=np.mgrid[:12,:12]-5.5
            off=np.stack((x,y),-1)
            w=np.exp(-(x*x+y*y)/18)
            total=np.zeros((225,440))
            disp=np.zeros((*total.shape,2))
            material=material_mask(ref)
            for j in np.flatnonzero(merged):
                x,y=(prev['centres'][j]-5.5).astype(int)
                box=np.s_[y:y+12,x:x+12]
                p=states[j]
                weight=w*material[box]
                total[box]+=weight
                disp[box]+=weight[...,None]*(p[[0,3]]+off@p[[1,2,4,5]].reshape(2,2).T)
            y,x=np.indices(total.shape)
            warp=samples(target,np.stack((x,y),-1)+disp/np.maximum(total[...,None],1e-12))
            warp[total==0]=ref[-1,-1]
            before=Image.fromarray(ref[:225,:440])
            after=Image.fromarray(warp)
            before.save(OUT/'reference.png')
            after.save(OUT/'after.png')
            before.save(OUT/'reference_after.gif',save_all=True,append_images=[after],duration=1000,loop=0)
    (OUT/'counts.json').write_text(json.dumps(metrics,indent=2))

if __name__=='__main__':
    main()

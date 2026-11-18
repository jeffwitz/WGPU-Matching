"""Rectify each affine subset, solve the residual and compose on original coordinates.

Run from the repository root. Pixel centres are (x,y), and the six coefficients
are (u,ux,uy,v,vx,vy). Each tile has a 12-pixel guard around its 12-pixel subset.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from hudge_strains_data import material_mask

sys.path.insert(0, '/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src')
from live_dic import ICGNWGPU
from live_dic.roi import unpack_support

ROOT = Path('results/hudge_hole_rbf_20261117')
OUTPUT = ROOT/'rectified_windows_191'
SIZE, TILE, HALF = 12, 36, 5.5


def samples(image, xy):
    return map_coordinates(image, [xy[..., 1], xy[..., 0]], order=1,
                           mode='constant', cval=0, prefilter=False)


def pack(tiles, rows, cols):
    return tiles.reshape(rows, cols, TILE, TILE).transpose(0, 2, 1, 3).reshape(rows*TILE, cols*TILE)


def main():
    OUTPUT.mkdir(exist_ok=True)
    info = json.loads(Path('results/hudge_hole_moments_20261117/proposals/summary.json').read_text())
    ref, target = [np.asarray(Image.open(p)) for p in info['input_sha256']]
    original = np.load(ROOT/'composition_v2/original_image_verification_04.npz')
    archive = np.load('results/hudge_boundary_frontier_20261115/partial12/seeds.npz')
    # Use the exact same material supports as the ordinary native solve.
    settings = dict(device_type='gpu', prefer_subgroups=False, subset_size=SIZE,
                    interpolation='bspline', motion_model='affine', outlier_policy='disabled')
    config = dict(iterations=30, tolerance=.001, zncc_min=.65, roi_policy='partial',
                  roi_truncation='visibility')
    native = ICGNWGPU(subset_stride=4, **settings)
    native.set_reference(ref, roi_mask=material_mask(ref), **config)
    roi = native.roi
    indices = np.flatnonzero(roi.selected.ravel())
    support = np.ones((len(roi.selected.ravel()), SIZE, SIZE), bool)
    for i, words in zip(roi.partial_indices, roi.support_words, strict=True):
        support[i] = unpack_support(words, SIZE).reshape(SIZE, SIZE)
    centres = archive['centres'][indices]
    seed = original['proposal'][indices].astype(float)
    F = np.eye(2)+seed[:, [1, 2, 4, 5]].reshape(-1, 2, 2)
    centre1 = centres+seed[:, [0, 3]]
    n = len(indices)
    cols, rows = 32, 16
    count = cols*rows
    y,x = np.mgrid[:TILE, :TILE]
    offset = np.stack((x,y), -1)-17.5
    local = offset[12:24, 12:24]
    params = seed.copy()
    status = np.zeros(n)
    accepted = np.zeros(n, bool)
    score = np.zeros(n)
    engine = ICGNWGPU(subset_stride=SIZE, **settings)
    tile_indices = (3*np.arange(rows)[:,None]+1)*cols*3+3*np.arange(cols)+1
    for start in range(0,n,count):
        sl = slice(start,min(start+count,n))
        m = len(seed[sl])
        a = np.zeros((count,TILE,TILE), np.uint8)
        b = a.copy()
        mask = np.zeros_like(a, bool)
        a[:m] = samples(ref, centres[sl,None,None]+offset)
        xy = centre1[sl,None,None]+np.einsum('nij,hwj->nhwi',F[sl],offset)
        b[:m] = samples(target, xy)
        mask[:m,12:24,12:24] = support[indices[sl]]
        engine.set_reference(pack(a,rows,cols), roi_mask=pack(mask,rows,cols), **config)
        enable = np.zeros(engine.roi.selected.shape,np.uint32)
        tid = tile_indices.ravel()[:m]
        enable.ravel()[tid] = 1
        packed = np.zeros((enable.size,8),np.float32)
        result_params = np.zeros((m,6))
        tile_target = np.ascontiguousarray(pack(b,rows,cols))
        for _ in range(4):
            pending = np.flatnonzero(enable.ravel()[tid])
            if not len(pending):
                break
            out = engine._register_seeded_frame(tile_target,packed,enable)
            j = tid[pending]
            result_params[pending] = out.parameters.reshape(-1,6)[j]
            status[start+pending] = out.status.ravel()[j]
            packed[j,:6] = result_params[pending]
            enable.ravel()[j] = (status[start+pending]==0)&np.isfinite(result_params[pending]).all(1)
        R = np.eye(2)+result_params[:,[1,2,4,5]].reshape(-1,2,2)
        shift = result_params[:,[0,3]]
        affine = F[sl]@R
        moved = centre1[sl]+np.einsum('nij,nj->ni',F[sl],shift)
        params[sl,0] = moved[:,0]-centres[sl,0]
        params[sl,3] = moved[:,1]-centres[sl,1]
        params[start:start+m, [1,2,4,5]] = (affine-np.eye(2)).reshape(-1,4)
        tile_q = shift[:,None,None]+np.einsum('nij,hwj->nhwi',R,local)
        q = moved[:,None,None]+np.einsum('nij,hwj->nhwi',affine,local)
        s = support[indices[sl]]
        valid = (((q>=0)&(q<=np.array(target.shape[::-1])-1)&(np.abs(tile_q)<=15.5)).all(-1)|~s).all((1,2))
        v = samples(target.astype(float),q)
        u = a[:m,12:24,12:24].astype(float)
        for arr in [u,v]:
            arr -= ((arr*s).sum((1,2))/s.sum((1,2)))[:,None,None]
            arr *= s
        score[sl] = (u*v).sum((1,2))/np.maximum(np.sqrt((u*u).sum((1,2))*(v*v).sum((1,2))),1e-12)
        accepted[sl] = (status[sl]==1)&valid&(score[sl]>=.65)&(np.linalg.det(affine)>0)
    engine.close()
    # Ordinary target-frame IC-GN started from the rectified solutions.
    ids = indices[accepted]
    values = params[accepted].copy()
    raw = np.zeros(len(ids))
    for _ in range(4):
        pending = np.flatnonzero((raw==0)&np.isfinite(values).all(1))
        if not len(pending):
            break
        pack_seed = np.zeros((roi.selected.size,8),np.float32)
        flags = np.zeros(roi.selected.shape,np.uint32)
        flags.ravel()[ids[pending]] = 1
        pack_seed[ids[pending],:6] = values[pending]
        out = native._register_seeded_frame(target,pack_seed,flags)
        values[pending] = out.parameters.reshape(-1,6)[ids[pending]]
        raw[pending] = out.status.ravel()[ids[pending]]
    native.close()
    orig_f = np.eye(2)+values[:,[1,2,4,5]].reshape(-1,2,2)
    q = (archive['centres'][ids]+values[:,[0,3]])[:,None,None]+np.einsum('nij,hwj->nhwi',orig_f,local)
    s = support[ids]
    valid = (((q>=0)&(q<=np.array(target.shape[::-1])-1)).all(-1)|~s).all((1,2))
    a = samples(ref.astype(float),archive['centres'][ids,None,None]+local)
    b = samples(target.astype(float),q)
    for v in [a,b]:
        v -= ((v*s).sum((1,2))/s.sum((1,2)))[:,None,None]
        v *= s
    ncc = (a*b).sum((1,2))/np.maximum(np.sqrt((a*a).sum((1,2))*(b*b).sum((1,2))),1e-12)
    direct = np.zeros(n,bool)
    direct[accepted] = (raw==1)&valid&(ncc>=.65)&(np.linalg.det(orig_f)>0)
    prior = original['accepted'][indices]
    rad = np.min([np.linalg.norm(centres-h['centre'],axis=1)/np.sqrt(h['area']/np.pi) for h in info['source_holes']],axis=0)
    metrics = {}
    for name,region in [('all',np.ones(n,bool)),('crown',(rad>=1)&(rad<1.25))]:
        metrics[name] = {k:int((v&region).sum()) for k,v in dict(eligible=np.ones(n,bool),baseline=prior,rectified=accepted,native_retry=direct,union_native=prior|direct,new_native=direct&~prior).items()}
    print(json.dumps(metrics),flush=True)
    np.savez_compressed(OUTPUT/'windows.npz',indices=indices,centres=centres,parameters=params,score=score,status=status,accepted=accepted,native_ids=ids,native_parameters=values,native_status=raw,native_scores=ncc,native_accepted=direct)
    (OUTPUT/'counts.json').write_text(json.dumps(metrics,indent=2))
    # Remap only pixels covered by accepted supports, leaving gaps explicit.
    weight = np.zeros(ref.shape)
    displacement = np.zeros((*ref.shape,2))
    for k in np.flatnonzero(accepted):
        x,y = (centres[k]-HALF).astype(int)
        window = np.s_[y:y+SIZE,x:x+SIZE]
        w = support[indices[k]]*np.exp(-np.sum(local**2,axis=2)/18)
        displacement[window] += w[...,None]*(params[k,[0,3]]+local@params[k,[1,2,4,5]].reshape(2,2).T)
        weight[window] += w
    y,x = np.mgrid[:225,:440]
    w = weight[:225,:440]
    xy = np.stack((x,y),-1)+displacement[:225,:440]/np.maximum(w[...,None],1e-12)
    im = samples(target,xy)
    im[w==0] = ref[-1,-1]
    after = Image.fromarray(im)
    before = Image.fromarray(ref[:225,:440])
    before.save(OUTPUT/'image1_ref.png')
    after.save(OUTPUT/'image1_def_cor.png')
    before.save(OUTPUT/'remap.gif',save_all=True,append_images=[after],duration=1100,loop=0)
    Image.fromarray(np.uint8((w>0)*255)).save(OUTPUT/'coverage.png')
    (OUTPUT/'index.html').write_text('<meta charset=utf-8><h1>Plaque 191 — fenêtres redressées individuellement</h1><img width=880 src="remap.gif"><p>Référence et cible remappée sur les supports acceptés. Les lacunes sont remplies avec le niveau du fond.</p><img width=440 src="coverage.png"><p>En blanc, domaine couvert par une fenêtre acceptée après redressement.</p><a href="counts.json">Comptages et contrôle sur les images originales</a>')
if __name__ == '__main__':
    main()

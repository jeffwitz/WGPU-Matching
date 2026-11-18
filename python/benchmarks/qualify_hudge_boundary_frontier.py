"""Propagate measured affine seeds into partial material supports on WebGPU."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from hudge_strains_data import material_mask
from PIL import Image
from scipy.ndimage import distance_transform_edt, label, map_coordinates


def transport(p, delta):
    result = p.copy()
    result[[0, 3]] += p[[1, 2, 4, 5]].reshape(2, 2) @ delta
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--seed-archive', type=Path)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--outer-band', type=float, default=0)
    parser.add_argument('--resume-exact', action='store_true',
                        help='Preserve accepted states from an identical grid without re-solving')
    parser.add_argument('--extra-seeds', type=Path,
                        help='Experimental image-derived candidates; each must pass the GPU solver')
    parser.add_argument('--data', type=Path, default=Path('/home/jeff/CNRS/DIC/HudgeStrains/FEA'))
    parser.add_argument('--level', type=int, default=191)
    parser.add_argument('--sector-validation', type=Path,
                        help='Fixed wider sector supports used to verify each final extra seed')
    parser.add_argument('--max-rings', type=int, default=40)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, '/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src')
    from live_dic import ICGNWGPU
    from live_dic.roi import unpack_support
    data = args.data
    ref = np.asarray(Image.open(data/'Img_F.tif'))
    target = np.asarray(Image.open(data/f'Img_wa_g{args.level:03d}.tif'))
    mask = material_mask(ref)
    original_shape = ref.shape
    padding = args.padding
    if padding < 0 or (padding % args.stride and not args.seed_archive):
        parser.error('nonnegative padding required; shifted grid needs explicit seed archive')
    if padding:
        ref = np.pad(ref, padding, constant_values=int(ref[-1,-1]))
        target = np.pad(target, padding, constant_values=int(target[-1,-1]))
        mask = np.pad(mask, padding, constant_values=False)
    native_crop = (slice(padding, padding+original_shape[0]), slice(padding, padding+original_shape[1]))
    size, stride = args.size, args.stride
    half = (size-1)/2
    engine = ICGNWGPU(device_type='gpu', prefer_subgroups=False, subset_size=size,
                      subset_stride=stride, interpolation='bspline', motion_model='affine', outlier_policy='disabled')
    engine.set_reference(ref, iterations=30, tolerance=.001, zncc_min=.65,
                         roi_mask=mask, roi_policy='partial', roi_truncation='visibility')
    roi = engine.roi
    y, x = np.indices(roi.selected.shape)
    centres = np.column_stack((stride*x.ravel()+half, stride*y.ravel()+half))
    n, width = len(centres), roi.selected.shape[1]
    eligible = roi.selected.ravel().copy()
    if args.outer_band:
        labels, _ = label(~mask)
        exterior = np.isin(labels, np.unique(np.r_[labels[0], labels[-1], labels[:,0], labels[:,-1]]))
        edge_distance = distance_transform_edt(~exterior)
        ij = np.rint(centres).astype(int)
        eligible &= edge_distance[ij[:,1],ij[:,0]] <= args.outer_band
    supports = {int(i): unpack_support(w, size).reshape(size, size) for i, w in zip(roi.partial_indices, roi.support_words, strict=True)}
    default_support = np.ones((size, size), bool)
    source_path = args.seed_archive or Path('results/hudge_dense_remap_20261115/bspline/level_191/affine16/seeds.npz')
    archive = dict(np.load(source_path))
    archive['centres'] = archive['centres']+padding
    parameters = np.zeros((n, 6)); accepted = np.zeros(n, bool); scores = np.zeros(n)
    if args.resume_exact and not args.seed_archive:
        parser.error('--resume-exact requires --seed-archive')
    if args.resume_exact:
        np.testing.assert_allclose(archive['centres'], centres, rtol=0, atol=1e-10)
        parameters[:] = archive['parameters']
        accepted[:] = archive['accepted'] & eligible
        scores[:] = archive['scores'] if 'scores' in archive else archive['correlation']
    elif not args.seed_archive:
        indices = ((archive['centres'][:, 1]-half)/stride).astype(int)*width+((archive['centres'][:, 0]-half)/stride).astype(int)
        parameters[indices] = archive['parameters']; accepted[indices] = archive['accepted']; scores[indices] = archive['correlation']
        accepted &= eligible
    attempts = np.zeros(n, int); rings = np.full(n, -1, int); rings[accepted] = 0
    tried = [set() for _ in range(n)]
    history = []
    target_float = target.astype(float)
    yy_patch, xx_patch = np.mgrid[:size, :size].astype(float)-half
    def check(i, p):
        x0, y0 = np.rint(centres[i]-half).astype(int)
        a = ref[y0:y0+size, x0:x0+size].astype(float)
        support = supports.get(i, default_support)
        qx = centres[i, 0]+xx_patch+p[0]+p[1]*xx_patch+p[2]*yy_patch
        qy = centres[i, 1]+yy_patch+p[3]+p[4]*xx_patch+p[5]*yy_patch
        aa = a[support]
        bb = map_coordinates(target_float, [qy[support], qx[support]], order=1, mode='constant', cval=0., prefilter=False)
        aa, bb = aa-aa.mean(), bb-bb.mean()
        corr = np.sum(aa*bb)/max(np.linalg.norm(aa)*np.linalg.norm(bb), 1e-12)
        inside = np.all((qx[support]>=padding)&(qx[support]<=padding+original_shape[1]-1)&(qy[support]>=padding)&(qy[support]<=padding+original_shape[0]-1))
        return corr, inside, np.linalg.det(np.eye(2)+p[[1,2,4,5]].reshape(2,2))
    if args.seed_archive and not args.resume_exact:
        from scipy.spatial import cKDTree
        previous = np.flatnonzero(archive['accepted'])
        distances, nearest = cKDTree(archive['centres'][previous]).query(centres)
        candidates = np.flatnonzero(eligible & (distances <= 8))
        permitted = []
        for i in candidates:
            a, b = centres[i], archive['centres'][previous[nearest[i]]]
            line = np.rint(a+np.linspace(0, 1, 33)[:, None]*(b-a)).astype(int)
            if mask[line[:, 1], line[:, 0]].all(): permitted.append(i)
        active = np.asarray(permitted, int)
        seeds = np.array([transport(archive['parameters'][previous[nearest[i]]], centres[i]-archive['centres'][previous[nearest[i]]]) for i in active])
        status = np.zeros(len(active)); proposal = seeds.copy()
        for block in range(4):
            pending = np.flatnonzero((status==0)&np.isfinite(seeds).all(axis=1))
            if not len(pending): break
            packed = np.zeros((n,8),np.float32); packed[active[pending],:6]=seeds[pending]
            enabled = np.zeros(n,np.uint32); enabled[active[pending]]=1
            result=engine._register_seeded_frame(target,packed,enabled.reshape(roi.selected.shape))
            seeds[pending]=np.asarray(result.parameters).reshape(-1,6)[active[pending]]
            status[pending]=np.asarray(result.status).ravel()[active[pending]]
        checks=np.array([check(i,p) for i,p in zip(active,seeds,strict=True)])
        keep=(status==1)&np.isfinite(seeds).all(axis=1)&(checks[:,0]>=.65)&(checks[:,1]>0)&(checks[:,2]>0)
        parameters[active[keep]]=seeds[keep]; scores[active[keep]]=checks[keep,0]
        accepted[active[keep]]=True; rings[active[keep]]=0
        np.savez_compressed(args.output/'bootstrap.npz',indices=active,proposal=proposal,output=seeds,status=status,checks=checks,retained=keep)
        print('bootstrap',int(keep.sum()),'/',len(active),flush=True)
    initial = accepted.copy()
    prepare_seconds = time.perf_counter() - started
    extra_started = time.perf_counter()
    extra_history = []
    if args.extra_seeds:
        extra = np.load(args.extra_seeds)
        if args.sector_validation:
            from hudge_hole_moments import (
                quadratic_mapping,
                sample,
                score_affine,
                score_coordinates,
            )
            sector_support = np.load(args.sector_validation)
            target_material = material_mask(target).astype(float)
        positions = (extra['centres'] + padding - half) / stride
        if not np.allclose(positions, np.rint(positions), rtol=0, atol=1e-8):
            raise ValueError('extra hypotheses must use exact solver grid centres')
        positions = np.rint(positions).astype(int)
        if np.any(positions < 0) or np.any(positions >= np.array(roi.selected.shape[::-1])):
            raise ValueError('extra hypothesis outside grid')
        indices = positions[:, 1]*width + positions[:, 0]
        for rank in np.unique(extra['rank']):
            use = np.flatnonzero((extra['rank'] == rank) & eligible[indices] & ~accepted[indices])
            active = indices[use]
            if not len(active):
                continue
            if len(np.unique(active)) != len(active):
                raise ValueError('one hypothesis per centre and rank required')
            seeds = extra['parameters'][use].copy()
            proposal = seeds.copy()
            status = np.zeros(len(active))
            for block in range(4):
                pending = np.flatnonzero((status == 0) & np.isfinite(seeds).all(axis=1))
                if not len(pending):
                    break
                packed = np.zeros((n, 8), np.float32)
                packed[active[pending], :6] = seeds[pending]
                enabled = np.zeros(n, np.uint32)
                enabled[active[pending]] = 1
                result = engine._register_seeded_frame(target, packed, enabled.reshape(roi.selected.shape))
                seeds[pending] = np.asarray(result.parameters).reshape(-1, 6)[active[pending]]
                status[pending] = np.asarray(result.status).ravel()[active[pending]]
            checks = np.array([check(i, p) for i, p in zip(active, seeds, strict=True)])
            sector_scores = np.full(len(active), np.nan)
            if args.sector_validation:
                for k, (i, p, source_row) in enumerate(zip(active, seeds, use, strict=True)):
                    if not np.isfinite(p).all():
                        sector_scores[k] = -1
                        continue
                    sector = extra['sector'][source_row]
                    start, stop = sector_support['offsets'][sector:sector+2]
                    points = sector_support['points'][start:stop] + padding
                    f = np.eye(2) + p[[1, 2, 4, 5]].reshape(2, 2)
                    t = centres[i] + p[[0, 3]] - f @ centres[i]
                    if 'curvature' in extra:
                        q, jac = quadratic_mapping(points, centres[i], f, t, extra['curvature'][source_row])
                        sector_scores[k], _ = score_coordinates(sample(ref, points), q, target_float, target_material)
                        if np.linalg.det(jac).min() <= 0:
                            sector_scores[k] = -1
                    else:
                        sector_scores[k], _ = score_affine(sample(ref, points), points, target_float,
                                                           target_material, f, t)
            keep = ((status == 1) & np.isfinite(seeds).all(axis=1) & (checks[:, 0] >= .65)
                    & (checks[:, 1] > 0) & (checks[:, 2] > 0))
            if args.sector_validation:
                keep &= sector_scores >= .65
            parameters[active[keep]] = seeds[keep]
            scores[active[keep]] = checks[keep, 0]
            accepted[active[keep]] = True
            rings[active[keep]] = 0
            np.savez_compressed(args.output/f'extra_{int(rank):02d}.npz', indices=active,
                                candidate_rows=use, proposal=proposal, output=seeds,
                                status=status, checks=checks, sector_scores=sector_scores, retained=keep)
            row = {'rank': int(rank), 'attempted': len(active), 'new': int(keep.sum())}
            extra_history.append(row)
            print('extra', row, flush=True)
    extra_seconds = time.perf_counter() - extra_started
    propagation_started = time.perf_counter()
    for ring in range(1, args.max_rings + 1):
        active, parents = [], []
        for i in np.flatnonzero(eligible & ~accepted & (attempts < 3)):
            candidates = []
            for bit in range(9):
                if bit == 4 or not (int(roi.neighbour_bits.ravel()[i]) & (1 << bit)):
                    continue
                dy, dx = divmod(bit, 3); j = i+(dy-1)*width+(dx-1)
                if accepted[j] and j not in tried[i]: candidates.append(j)
            if candidates:
                parent = max(candidates, key=lambda j: scores[j])
                active.append(i); parents.append(parent); tried[i].add(parent)
        active, parents = np.array(active, int), np.array(parents, int)
        if not len(active): break
        seeds = np.array([transport(parameters[j], centres[i]-centres[j]) for i,j in zip(active,parents,strict=True)])
        status = np.zeros(len(active)); proposal = seeds.copy()
        for block in range(4):
            pending = np.flatnonzero((status==0)&np.isfinite(seeds).all(axis=1))
            if not len(pending): break
            packed = np.zeros((n,8),np.float32); packed[active[pending],:6]=seeds[pending]
            enabled = np.zeros(n,np.uint32); enabled[active[pending]]=1
            result=engine._register_seeded_frame(target,packed,enabled.reshape(roi.selected.shape))
            seeds[pending]=np.asarray(result.parameters).reshape(-1,6)[active[pending]]
            status[pending]=np.asarray(result.status).ravel()[active[pending]]
        checks=np.array([check(i,p) for i,p in zip(active,seeds,strict=True)])
        keep=(status==1)&np.isfinite(seeds).all(axis=1)&(checks[:,0]>=.65)&(checks[:,1]>0)&(checks[:,2]>0)
        attempts[active]+=1
        parameters[active[keep]]=seeds[keep]; scores[active[keep]]=checks[keep,0]
        accepted[active[keep]]=True; rings[active[keep]]=ring
        np.savez_compressed(args.output/f'ring_{ring:02d}.npz',indices=active,parents=parents,proposal=proposal,output=seeds,status=status,checks=checks,retained=keep)
        row={'ring':ring,'attempted':len(active),'new':int(keep.sum()),'total':int(accepted.sum())};history.append(row);print(row,flush=True)
    propagation_seconds = time.perf_counter() - propagation_started
    np.savez_compressed(args.output/'seeds.npz',centres=centres-padding,parameters=parameters,accepted=accepted,initial=initial,selected=eligible,scores=scores,rings=rings,attempts=attempts)
    # A field only where a solved support contributes; no extrapolation into gaps.
    weights=np.zeros(ref.shape); displacement=np.zeros((*ref.shape,2))
    yy,xx=np.mgrid[:size,:size].astype(float)-half
    spatial=np.exp(-(xx*xx+yy*yy)/(2*(size/4)**2))
    for i in np.flatnonzero(accepted):
        x0,y0=np.rint(centres[i]-half).astype(int);p=parameters[i]
        support=supports.get(int(i),default_support)
        w=spatial*support
        displacement[y0:y0+size,x0:x0+size,0]+=w*(p[0]+p[1]*xx+p[2]*yy)
        displacement[y0:y0+size,x0:x0+size,1]+=w*(p[3]+p[4]*xx+p[5]*yy)
        weights[y0:y0+size,x0:x0+size]+=w
    covered=weights>0
    displacement/=np.maximum(weights[...,None],1e-30)
    yy,xx=np.indices(ref.shape); q=np.stack((xx,yy),-1)+displacement
    remap=map_coordinates(target.astype(float),q[...,::-1].reshape(-1,2).T,order=1,mode='constant',cval=float(target[-1,-1])).reshape(ref.shape)
    remap[~covered]=ref[-1,-1]
    Image.fromarray(np.uint8(np.clip(remap[native_crop],0,255))).save(args.output/'def_cor_supported.png')
    marked=np.repeat(np.uint8(np.clip(remap,0,255))[...,None],3,axis=2);marked[mask&~covered]=[255,0,180]
    Image.fromarray(marked[native_crop]).save(args.output/'def_cor_missing_magenta.png')
    Image.fromarray(ref[native_crop]).save(args.output/'reference.png')
    np.savez_compressed(args.output/'field.npz',displacement=displacement[native_crop].astype(np.float32),covered=covered[native_crop])
    lab,_=label(~mask);outside=np.unique(np.r_[lab[0],lab[-1],lab[:,0],lab[:,-1]])
    holes=(lab>0)&~np.isin(lab,outside);distance=distance_transform_edt(~holes)
    bands=[]
    for lo,hi in [(0,4),(4,8),(8,16),(16,32)]:
        region=mask&(distance>lo)&(distance<=hi); valid=region&covered
        ci=np.rint(centres).astype(int);d=distance[ci[:,1],ci[:,0]]; centre_band=(d>lo)&(d<=hi)&roi.selected.ravel()
        bands.append({'band':[lo,hi],'pixels':int(region.sum()),'covered':int(valid.sum()),'correlation_covered':float(np.corrcoef(ref[valid],remap[valid])[0,1]),'initial_centres':int((centre_band&initial).sum()),'accepted_centres':int((centre_band&accepted).sum()),'selected_centres':int(centre_band.sum())})
    report={'outer_band':args.outer_band,'padding':padding,'level':191,'size':size,'stride':stride,'policy':'partial','truncation':'visibility','initial':int(initial.sum()),'accepted':int(accepted.sum()),'selected':int(roi.selected.sum()),'history':history,'bands':bands,'source':str(source_path),'limits':'status and masked photometry only; no displacement truth; CPU orchestration and remap; GPU solves; uncovered pixels are not initialized'}
    report['extra_seeds'] = str(args.extra_seeds) if args.extra_seeds else None
    report['extra_history'] = extra_history
    report['resume_exact'] = args.resume_exact
    report['sector_validation'] = str(args.sector_validation) if args.sector_validation else None
    report['level'] = args.level
    report['data'] = str(data)
    report['times'] = {'prepare_seconds': prepare_seconds, 'extra_seconds': extra_seconds,
                       'propagation_seconds': propagation_seconds,
                       'total_including_exports_seconds': time.perf_counter() - started}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2));print(bands,flush=True)
    engine.close()


if __name__=='__main__':main()

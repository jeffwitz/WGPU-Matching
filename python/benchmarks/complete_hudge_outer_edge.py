"""Fill only unresolved exterior material using solved partial supports."""
import json
from pathlib import Path

import numpy as np
from hudge_strains_data import material_mask
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
    map_coordinates,
    uniform_filter,
)


def fill_missing(displacement, covered, candidate, candidate_covered, allowed):
    """Never change an existing observation or fill outside the permitted region."""
    fill = allowed & ~covered & candidate_covered & np.isfinite(candidate).all(axis=-1)
    result, valid = displacement.copy(), covered.copy()
    result[fill] = candidate[fill]
    valid[fill] = True
    return result, valid, fill


def main():
    root = Path('results/hudge_boundary_frontier_20261115')
    out = root/'outer_completed'
    out.mkdir(exist_ok=True)
    ref = np.asarray(Image.open(root/'partial12/reference.png'))
    target = np.asarray(Image.open('/home/jeff/CNRS/DIC/HudgeStrains/FEA/Img_wa_g191.tif'))
    mask = material_mask(ref)
    labels, _ = label(~mask)
    outside = np.isin(labels, np.unique(np.r_[labels[0], labels[-1], labels[:,0], labels[:,-1]]))
    holes = (labels > 0) & ~outside
    edge_distance = distance_transform_edt(~outside)
    hole_distance = distance_transform_edt(~holes)
    allowed = mask & (edge_distance <= 20) & (hole_distance > 16)
    base = np.load(root/'partial12/field.npz')
    u, covered = base['displacement'].copy(), base['covered'].copy()
    source = np.full(mask.shape, -1, np.int16); source[covered] = 0
    steps = []
    stages = ['outer_control','outer_padding32','outer16','outer24','outer32','outer32_phase2']
    for index, stage in enumerate(stages, 1):
        candidate = np.load(root/stage/'field.npz')
        u, covered, filled = fill_missing(u, covered, candidate['displacement'], candidate['covered'], allowed)
        source[filled] = index
        steps.append({'stage':stage,'new_pixels':int(filled.sum()),'remaining_outer12':int(np.sum(mask & (edge_distance<=12) & ~covered))})
    np.testing.assert_array_equal(u[base['covered']],base['displacement'][base['covered']])
    np.testing.assert_array_equal(covered[hole_distance<=16],base['covered'][hole_distance<=16])
    np.testing.assert_array_equal(u[hole_distance<=16],base['displacement'][hole_distance<=16])
    yy, xx = np.indices(mask.shape)
    q = np.stack((xx, yy), -1)+u
    inside=(q[...,0]>=0)&(q[...,0]<=1023)&(q[...,1]>=0)&(q[...,1]<=1023)
    assert np.all(inside[covered])
    image = map_coordinates(target.astype(float), q[...,::-1].reshape(-1,2).T, order=1, mode='constant', cval=float(target[-1,-1])).reshape(mask.shape)
    # Additional image-based screening of NEW pixels, never of archived observations.
    raw_covered = covered.copy()
    valid = mask & covered
    mass = uniform_filter(valid.astype(float), 9, mode='constant')
    def average(a):
        return uniform_filter(a*valid, 9, mode='constant')/np.maximum(mass,1e-12)
    rf = ref.astype(float)
    ma, mb = average(rf), average(image)
    local_corr = (average(rf*image)-ma*mb)/np.sqrt(np.maximum((average(rf*rf)-ma*ma)*(average(image*image)-mb*mb),1e-12))
    uy, ux = np.gradient(u,axis=(0,1))
    determinant = (1+ux[...,0])*(1+uy[...,1])-uy[...,0]*ux[...,1]
    newly_added = covered & ~base['covered']
    reject = newly_added & ((local_corr < .65) | (mass*81 < 25) | (binary_erosion(valid) & (determinant <= 0)))
    np.savez_compressed(out/'screening.npz',raw_covered=raw_covered,local_correlation=local_corr.astype(np.float32),rejected=reject)
    covered[reject] = False
    u[reject] = base['displacement'][reject]
    source[reject] = -1
    for _ in range(8):
        uy, ux = np.gradient(u, axis=(0,1))
        determinant = (1+ux[...,0])*(1+uy[...,1])-uy[...,0]*ux[...,1]
        bad = binary_erosion(covered) & (determinant <= 0)
        extra = binary_dilation(bad) & newly_added & covered
        if not extra.any():
            break
        reject |= extra
        covered[extra] = False
        u[extra] = base['displacement'][extra]
        source[extra] = -1
    q = np.stack((xx,yy),-1)+u
    image = map_coordinates(target.astype(float),q[...,::-1].reshape(-1,2).T,order=1,mode='constant',cval=float(target[-1,-1])).reshape(mask.shape)
    np.savez_compressed(out/'screening.npz',raw_covered=raw_covered,local_correlation=local_corr.astype(np.float32),rejected=reject)
    old_image = np.asarray(Image.open(root/'partial12/def_cor_supported.png'))
    measurements=[]
    for lo,hi in [(0,4),(4,8),(8,12),(12,20)]:
        region=mask&(edge_distance>lo)&(edge_distance<=hi)
        old=region&base['covered']; new=region&covered; added=new&~old
        measurements.append({'band':[lo,hi],'pixels':int(region.sum()),'before':int(old.sum()),'after':int(new.sum()),
                             'correlation_after_covered':float(np.corrcoef(ref[new],image[new])[0,1]),
                             'correlation_new_pixels':float(np.corrcoef(ref[added],image[added])[0,1]) if added.any() else None})
    uy,ux=np.gradient(u,axis=(0,1));det=(1+ux[...,0])*(1+uy[...,1])-uy[...,0]*ux[...,1]
    report={'screening':{'new_before':int(newly_added.sum()),'rejected':int(reject.sum()),'retained':int((newly_added&~reject).sum()),'window':9,'minimum_material_samples':25,'minimum_correlation':.65},'level':191,'steps':steps,'bands':measurements,'holes_16px_unchanged':True,'old_covered_pixels_unchanged':True,
            'nonpositive_fd_jacobian_inside_covered':int(np.sum(binary_erosion(covered)&(det<=0))),
            'limits':'Coverage of solved supports is not displacement truth. Composite remap is diagnostic, not a qualified global deformation. No contour fit or extrapolation.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'field.npz',displacement=u,covered=covered,source=source,allowed=allowed)
    # Export actual local initializers with their support sizes, not blended derivatives.
    entries=[]
    for stage,size in [('partial12',12),('outer_control',12),('outer_padding32',12),('outer16',16),('outer24',24),('outer32',32),('outer32_phase2',32)]:
        a=np.load(root/stage/'seeds.npz'); ids=np.flatnonzero(a['accepted'])
        entries.append((a['centres'][ids],a['parameters'][ids],np.full(len(ids),size),np.full(len(ids),stage)))
    np.savez_compressed(out/'local_initializers.npz',centres=np.concatenate([e[0] for e in entries]),parameters=np.concatenate([e[1] for e in entries]),support_size=np.concatenate([e[2] for e in entries]),source=np.concatenate([e[3] for e in entries]))
    hatch=np.where(((xx+yy)//4)%2,55,75).astype(np.uint8)
    def display(raw,valid):
        rgb=np.repeat(np.uint8(np.clip(raw,0,255))[...,None],3,axis=2)
        rgb[~mask]=np.repeat(hatch[~mask,None],3,axis=1)
        rgb[mask&~valid]=[255,0,180]
        return Image.fromarray(rgb).crop((0,0,440,225)).resize((880,450),Image.Resampling.NEAREST)
    before=display(old_image,base['covered']);after=display(image,covered);reference=display(ref,covered)
    before.save(out/'before.png');after.save(out/'after.png');reference.save(out/'reference.png')
    before.save(out/'before_after.gif',save_all=True,append_images=[after],duration=1100,loop=0)
    reference.save(out/'reference_corrected.gif',save_all=True,append_images=[after],duration=1100,loop=0)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

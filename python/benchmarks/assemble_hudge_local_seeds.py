"""Preserve archived measurements and add locally verified native affine states."""
import json
import numpy as np
from PIL import Image
from qualify_hudge_rectified_windows import OUTPUT, ROOT, SIZE, HALF, ICGNWGPU, unpack_support, material_mask, Path, samples


def main():
    info=json.loads(Path('results/hudge_hole_moments_20261117/proposals/summary.json').read_text())
    ref,target=[np.asarray(Image.open(p)) for p in info['input_sha256']]
    orig=np.load(ROOT/'composition_v2/original_image_verification_04.npz')
    forward=np.load(OUTPUT/'native_forward.npz')
    patch=np.load(OUTPUT/'windows.npz')
    ids=patch['indices']
    centres=orig['centres']
    parameters=orig['merged_parameters'].copy()
    source=np.zeros(len(centres),np.uint8)
    source[orig['prior_accepted']]=1
    source[orig['merged_accepted']&~orig['prior_accepted']]=2
    native_set=set(ids[patch['native_accepted']])
    for i,p in zip(patch['native_ids'],patch['native_parameters'],strict=True):
        if source[i]==0 and i in native_set:
            parameters[i]=p
            source[i]=3
    new=forward['accepted']&(source[ids]==0)
    parameters[ids[new]]=forward['parameters'][new]
    source[ids[new]]=4
    np.savez_compressed(OUTPUT/'initialisation.npz',centres=centres,parameters=parameters,accepted=source>0,source=source)
    r=np.min([np.linalg.norm(centres-h['centre'],axis=1)/np.sqrt(h['area']/np.pi) for h in info['source_holes']],axis=0)
    solver=ICGNWGPU(device_type='gpu',subset_size=SIZE,subset_stride=4,interpolation='bspline',motion_model='affine',outlier_policy='disabled')
    solver.set_reference(ref,iterations=30,tolerance=.001,zncc_min=.65,roi_mask=material_mask(ref),roi_policy='partial',roi_truncation='visibility')
    roi=solver.roi
    partial={int(i):unpack_support(w,SIZE).reshape(SIZE,SIZE) for i,w in zip(roi.partial_indices,roi.support_words,strict=True)}
    solver.close()
    crown=(r>=1)&(r<1.25)
    counts={name:[int((m&v).sum()) for v in [source==1,(source>0)&(source<=2),source>0]] for name,m in [('all',roi.selected.ravel()),('crown',roi.selected.ravel()&crown)]}
    counts['source_codes']={'0':'missing','1':'archived accepted','2':'composed original IC-GN','3':'rectified then original IC-GN','4':'native forward Gauss-Newton'}
    (OUTPUT/'union.json').write_text(json.dumps(counts,indent=2))
    print(counts)
    y,x=np.mgrid[:SIZE,:SIZE]-HALF
    offsets=np.stack((x,y),-1)
    win=np.exp(-np.sum(offsets**2,axis=-1)/18)
    frames=[]
    for label,keep in [('before',(source>0)&(source<=2)),('after',source>0)]:
        total=np.zeros(ref.shape)
        disp=np.zeros((*ref.shape,2))
        for i in np.flatnonzero(keep):
            x0,y0=(centres[i]-HALF).astype(int)
            p=parameters[i]
            box=np.s_[y0:y0+SIZE,x0:x0+SIZE]
            w=win*partial.get(i,1)
            disp[box]+=w[...,None]*(p[[0,3]]+offsets@p[[1,2,4,5]].reshape(2,2).T)
            total[box]+=w
        y,x=np.mgrid[:225,:440]
        weight=total[:225,:440]
        xy=np.stack((x,y),-1)+disp[:225,:440]/np.maximum(weight[...,None],1e-12)
        im=samples(target,xy)
        im[weight==0]=ref[-1,-1]
        frame=Image.fromarray(im)
        frame.save(OUTPUT/f'native_{label}.png')
        frames.append(frame)
        Image.fromarray(np.uint8((weight>0)*255)).save(OUTPUT/f'coverage_{label}.png')
    Image.fromarray(ref[:225,:440]).save(OUTPUT/'native_final.gif',save_all=True,append_images=[frames[-1]],duration=950,loop=0)
    page=(OUTPUT/'index.html').read_text().split('<!-- native -->')[0]
    (OUTPUT/'index.html').write_text(page+'<!-- native --><h2>Initialisations vérifiées dans la cible originale</h2><p>Les anciennes fenêtres sont conservées. Les lacunes reçoivent une affine vérifiée sur les pixels originaux. Les fenêtres ajoutées par Gauss–Newton direct utilisent les dérivées de la cible et un pas décroissant.</p>'+f'<p>Plaque entière {counts["all"][1]} → {counts["all"][2]} / 5126. Première couronne {counts["crown"][1]} → {counts["crown"][2]} / 246.</p><img width=880 src="native_final.gif"><p>Zones sans support représentées par le niveau du fond.</p><img width=440 src="coverage_before.png"><img width=440 src="coverage_after.png"><p>Couverture avant et après. <a href="initialisation.npz">Paramètres par fenêtre et provenance</a> · <a href="union.json">Comptages</a> · <a href="native_after.png">Image corrigée</a></p>')
if __name__=='__main__':
    main()

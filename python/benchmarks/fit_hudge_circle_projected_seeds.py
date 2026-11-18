"""Fit a fresh RBF using eight affine-projected displacement seeds per hole."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label
from experiment_hudge_hole_rbf import MixedField, basis, corr, evaluate, fit
from hudge_affine_restart import fit_global
from hudge_hole_moments import sample
from hudge_rbf_restart import SeedField
from hudge_strains_data import material_mask

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--count',type=int,default=8); p.add_argument('--weight',type=float,default=1.); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 root=Path('results/hudge_hole_rbf_20261117'); comp=root/'composition_v2'; info=json.loads((root/'../hudge_hole_moments_20261117/proposals/summary.json').read_text()) if False else json.loads(Path('results/hudge_hole_moments_20261117/proposals/summary.json').read_text())
 ref,target=[np.asarray(Image.open(f),float) for f in info['input_sha256']]; mask=material_mask(ref); labels,_=label(~mask)
 old=np.load('results/hudge_boundary_frontier_20261115/partial12/seeds.npz'); centres=old['centres'][old['accepted']]; states=old['parameters'][old['accepted']]
 prev=np.load(comp/'residual_04.npz'); sites,lengths=prev['sites'],prev['length']
 moment=info; aff=json.loads((root/'identified_hole_affines.json').read_text())
 anchors=[]; targets=[]; detail=[]
 for k,h in enumerate(moment['source_holes']):
  c=np.asarray(h['centre']); radius=np.sqrt(h['area']/np.pi); theta=np.arange(a.count)*2*np.pi/a.count
  circle=c+radius*np.c_[np.cos(theta),np.sin(theta)]
  dest=circle@np.asarray(aff[k]['F']).T+np.asarray(aff[k]['t'])
  anchors.extend(circle); targets.extend(dest-circle)
  detail.append({'hole':k+1,'centre':c.tolist(),'radius_px':radius,'source_points':circle.tolist(),'projected_points':dest.tolist(),'displacements':(dest-circle).tolist(),'F_used':aff[k]['F'],'interpretation':'Affine estimated from matched hole moments and local texture; virtual constraints, not observed material displacement.'})
 anchors=np.asarray(anchors); targets=np.asarray(targets)
 f,t,_=fit_global(centres,states,np.ones(len(centres),bool)); base=SeedField(f,t,np.empty((0,2)),np.empty((0,2)),16.)
 holes=np.asarray([h['centre'] for h in moment['source_holes']]); grads=np.zeros((len(holes),2,2))
 yy,xx=np.mgrid[:225,:440]; pix=np.c_[xx.ravel(),yy.ravel()]; mat=mask[:225,:440]
 variants=[]
 for w in [.1, a.weight, 10.]:
  model=fit(centres,states,sites,lengths,base,anchors,targets,holes,grads,w,False)
  vals=evaluate(model,pix); q=pix+vals[:,[0,3]]; warp=sample(target,q).reshape(mat.shape)
  det=np.linalg.det(np.eye(2)+vals[:,[1,2,4,5]].reshape(-1,2,2)).reshape(mat.shape)
  radius=np.stack([np.linalg.norm(pix-np.asarray(h['centre']),axis=1)/np.sqrt(h['area']/np.pi) for h in moment['source_holes']]).min(0).reshape(mat.shape)
  b=mat&(radius>=1)&(radius<1.25); e=evaluate(model,centres)-states
  row={'weight':w,'material_correlation':corr(ref[:225,:440][mat],warp[mat]),'crown_correlation':corr(ref[:225,:440][b],warp[b]),'seed_position_rms_px':float(np.sqrt(np.mean(np.sum(e[:,[0,3]]**2,axis=1)))),'nonpositive_material_jacobians':int((mat&(det<=0)).sum())}
  variants.append(row); Image.fromarray(np.uint8(np.clip(warp,0,255))).save(a.output/f'weight_{w:g}.png'); np.savez_compressed(a.output/f'weight_{w:g}.npz',matrix=model.matrix,translation=model.translation,sites=sites,coefficients=model.coefficients,length=lengths,points=pix,parameters=vals)
 summary={'experiment':'Eight equally spaced points on each reference hole, transported by the identified hole affine. Refit one RBF to the same distant measured seed states plus these virtual circle displacements.','count_per_hole':a.count,'projected_constraints':detail,'variants':variants,'limits':'The projected displacements are derived from hole geometry; not independent measurements. No displacement truth. Photometric scores describe remapping only.'}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)); (a.output/'index.html').write_text('<meta charset="utf-8"><h1>RBF recalee sur huit projections par trou</h1><p>Chaque projection est issue de la transformation affine identifiée pour le trou. Ce sont des contraintes virtuelles. Les anciennes graines mesurees restent dans la fonction objectif.</p>'+''.join(f'<h2>Poids {r["weight"]}</h2><p>{r}</p><img src="weight_{r["weight"]:g}.png">' for r in variants)+'<p><a href="summary.json">Points et chiffres</a></p>')
 print(json.dumps(variants,indent=2))
if __name__=='__main__':
 import sys; main()

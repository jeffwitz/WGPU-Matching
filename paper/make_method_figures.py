"""Pedagogical diagrams and archive-only stage/provenance figures."""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'paper'; R=ROOT/'results/seed_propagation_20261111'
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','font.family':'DejaVu Sans'})
provenance=[]
def record(path):
    provenance.append({'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight',dpi=300)
    fig.savefig(OUT/'figures'/f'{name}.svg',bbox_inches='tight',dpi=300)
    fig.savefig(OUT/'review'/f'{name}.png',bbox_inches='tight',dpi=130);plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4.9));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
boxes=[(0.1,8,'Two native images\n+ requested centres / supplied mask'),(5.2,8,'Area reduction + learned encoder\n192 × 108 → 48 × 27 features'),(5.2,5.8,'Orientation ranking + affine fit\nCoordinate pairs → geometry'),(0.1,5.8,'Rectification + local ZNCC\nResidual proposals → six-parameter states'),(0.1,3.5,'Sparse native IC-GN solves\nHypothesis selection + confirmation'),(5.2,3.5,'Affine frontier + new native solves\nAccepted states / rejects / provenance')]
for x,y,t in boxes:
 ax.add_patch(FancyBboxPatch((x,y),4.6,1.3,boxstyle='round,pad=.08',facecolor='#eaf1f5',edgecolor='#333333'))
 ax.text(x+2.3,y+.65,t,ha='center',va='center',fontsize=8)
for a,b in [((4.8,8.65),(5.1,8.65)),((7.5,7.9),(7.5,7.2)),((5.1,6.45),(4.8,6.45)),((2.4,5.7),(2.4,4.9)),((4.8,4.15),(5.1,4.15))]:
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.2})
ax.annotate('',xy=(9.9,4.4),xytext=(9.9,3.8),arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=1.5'})
ax.text(5,2.85,'If anchor coverage fails: regional retries from existing image hypotheses',ha='center',fontsize=7)
ax.axhline(2.3,color='#777777',ls='--',lw=.7)
ax.text(5,1.55,'Evaluation only: archived outputs + analytical truth / visibility\nCorrect recovery, false acceptances, errors; no feedback to selection',ha='center',va='center',bbox={'facecolor':'#fff2d6','edgecolor':'#777777','pad':7})
ax.annotate('',xy=(7.5,2),xytext=(7.5,3.3),arrowprops={'arrowstyle':'->','ls':'--'})
save(fig,'pipeline')

fig,axs=plt.subplots(1,4,figsize=(7,2.4)); F=np.array([[1.6,.35],[0,1.]])
for ax,title,A in zip(axs,['Reference\nmaterial square','Its material image\nunder deformation','Fixed target square\n≠ same material','Compensated samples\nin reference frame'],[np.eye(2),F,np.eye(2),np.eye(2)]):
 t=np.linspace(-1,1,7)
 for q in t:
  for v in [np.array([t,np.full_like(t,q)]),np.array([np.full_like(t,q),t])]:
   z=A@v;ax.plot(*z,lw=.65,color='#0072B2')
 if ax==axs[2]:
  contour=F@np.array([[-1,1,1,-1,-1],[-1,-1,1,1,-1]])
  ax.plot(*contour,color='#D55E00',ls='--',lw=1.3)
 ax.set(aspect='equal',xlim=(-2,2),ylim=(-1.5,1.5),title=title);ax.axis('off')
save(fig,'material_support')

# Complete recorded chain: no intermediate network output is recreated.
pair='heterogeneous_texture62051005_warp62091005'
base=R/'p3_fine_transfer_multifamily_independent/inputs/affine_ncc'
archive=base/'windows_seed62000000'/f'{pair}.npz';a=np.load(archive);record(archive)
images=[]
for kind in ['reference','target']:
 p=base/'solver_images_seed62000000'/f'{pair}_{kind}.png';record(p)
 images.append(np.asarray(Image.open(p).convert('L'),float)/255)
ref,target=images;xy=a['centres_xy_native'];ids=np.flatnonzero(a['icgn_status']>.5)
i=int(ids[np.argmin(np.sum((xy[ids]-[1920,1080])**2,axis=1))]);c=xy[i]
y,x=np.meshgrid(np.arange(64)-31.5,np.arange(64)-31.5,indexing='ij');offset=np.stack([x,y],-1)
def patch(im,points):return map_coordinates(im,[points[...,1],points[...,0]],order=1,mode='constant',cval=np.nan,prefilter=False)
def warp(p):return patch(target,c+p[[0,3]]+offset@np.array([[1+p[1],p[2]],[p[4],1+p[5]]]).T)
rpatch=patch(ref,c+offset); pre=warp(a['pre_ncc_initial_parameters'][i]);ncc=warp(a['initial_parameters'][i]);final=warp(a['icgn_parameters'][i])
fig,axs=plt.subplots(2,3,figsize=(7,5.9),layout='constrained')
axs[0,0].imshow(ref[::4,::4],cmap='gray',vmin=0,vmax=1,extent=(0,3840,2160,0))
pairs=a['hard_match_pairs'];indices=np.linspace(0,len(pairs)-1,32).astype(int);m=pairs[indices]
axs[0,0].quiver(m[:,0],m[:,1],m[:,2]-m[:,0],m[:,3]-m[:,1],angles='xy',scale_units='xy',scale=1,color='#D55E00',width=.003)
axs[0,0].set(title='(a) 32 of 1296 hard candidates',xlim=(0,3840),ylim=(2160,0),xlabel='Native x (pixels)',ylabel='Native y (pixels)')
for ax,im,title in zip(axs.ravel()[1:],[rpatch,pre,ncc,final,np.abs(rpatch-final)],['(b) Native reference support','(c) Affine-only proposal','(d) Affine + local-correlation proposal','(e) After native IC-GN','(f) Absolute final residual']):
 h=ax.imshow(im,cmap='magma' if ax==axs[1,2] else 'gray',vmin=0,vmax=.25 if ax==axs[1,2] else 1,extent=(-32,32,32,-32))
 ax.set(title=title,xlabel='Local x (pixels)',ylabel='Local y (pixels)');ax.title.set_fontsize(7.3)
 if ax==axs[1,2]:fig.colorbar(h,ax=ax,shrink=.7,label='Intensity difference (0–1)')
save(fig,'recorded_chain')
provenance.append({'selected_window':i,'pair':pair,'selection':'solver-accepted centre nearest image centre; no error/residual selection','proposal_source':'archived pre_ncc_initial_parameters and initial_parameters; no raw peak reconstructed'})

# Two explicit examples: small seed population and extreme-shear false states.
fig,axs=plt.subplots(2,3,figsize=(7,4.9),layout='constrained')
for row,(directory,pair,exact) in enumerate([
 (R/'p2_confirmatory/evaluation','heterogeneous_texture24051003_warp24091003',False),
 (R/'p3_shear_band_frozen_protocol/g2.0/evaluation','shear_band_texture59051005_warp59091005',True)]):
 p=directory/f'{pair}.npz';a=np.load(p);record(p);xy=a['centres'];ac=a['accepted'].astype(bool);v=a['visible'].astype(bool);e=a['error']
 if exact:
  p=R/'p3_shear_band_frozen_protocol/exact_center_analysis'/f'{pair}.npz';b=np.load(p);record(p);v=b['visible'].astype(bool);e=b['final_error_px']
 seeds=a['confirmed_seeds'].astype(bool)
 ax=axs[row,0];ax.scatter(*xy.T,s=.4,c='#dddddd');ax.scatter(*xy[seeds].T,s=4,c='#0072B2',marker='o');ax.set_title(f'{"Heterogeneous" if row==0 else "Severe shear"}: {seeds.sum()} anchors',fontsize=8)
 ax=axs[row,1];h=ax.scatter(*xy[ac].T,c=a['generation'][ac],s=3,cmap='viridis',marker='s',rasterized=True);ax.set_title('Accepted generation (0 = anchors)',fontsize=7.5);fig.colorbar(h,ax=ax,shrink=.65)
 ax=axs[row,2];good=v&ac&(e<2);bad=v&ac&(e>=2);unresolved=v&~ac
 ax.scatter(*xy[good].T,s=2,c='#0072B2',marker='s',label='Correct');ax.scatter(*xy[unresolved].T,s=5,c='#777777',marker='x',label='Rejected')
 ax.scatter(*xy[bad].T,s=12,c='#D55E00',marker='x',label='False accepted');ax.set_title(f'{good.sum()} correct; {bad.sum()} false',fontsize=8)
 for ax in axs[row]:ax.set(xlim=(0,3840),ylim=(2160,0),aspect='equal',xlabel='x (pixels)',ylabel='y (pixels)');ax.tick_params(labelsize=6)
axs[1,2].legend(fontsize=6,loc='upper center',bbox_to_anchor=(.5,-.3),ncol=2,frameon=False)
save(fig,'propagation_limits')
(OUT/'data/method_figures.json').write_text(json.dumps(provenance,indent=2)+'\n')

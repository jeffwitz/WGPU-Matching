from pathlib import Path
import re, json, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"paper"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"savefig.bbox":"tight"})
sources=["results/seed_propagation_20261111/p2_confirmatory/evaluation/README.md","results/seed_propagation_20261111/p3_fine_transfer_multifamily_independent/fine/README.md","results/seed_propagation_20261111/p3_decision_20261111.md","results/seed_propagation_20261111/p2_cached_pipeline_paired_isolated/README.md"]
manifest=[]
for s in sources:
 p=ROOT/s; b=p.read_bytes(); (OUT/"data"/ (p.parent.name+"_"+p.name)).write_bytes(b)
 manifest.append({"path":s,"sha256":hashlib.sha256(b).hexdigest()})
(OUT/"data/sources.json").write_text(json.dumps(manifest,indent=2))
rows=[]
for line in (ROOT/sources[0]).read_text().splitlines():
 if line.startswith("| heterogeneous_texture"):
  c=[x.strip() for x in line.split("|")[1:-1]]; good,vis=map(int,c[3].split("/")); rows.append((c[0],good,vis,int(c[4]),int(c[5])))
a=np.array([r[1:] for r in rows]); order=np.argsort(a[:,0]/a[:,1]-a[:,2]/a[:,1]); fig,ax=plt.subplots(figsize=(7,4.8))
for k,i in enumerate(order):
 y=a[i,0]/a[i,1]*100; x=a[i,2]/a[i,1]*100; ax.plot([x,y],[k,k],color="#999999",lw=1); ax.scatter(x,k,color="#D55E00",s=20,label="One-pass baseline" if k==0 else None); ax.scatter(y,k,color="#0072B2",s=20,label="Confirmed seeds + propagation" if k==0 else None)
ax.set(xlabel="Correctly converged / visible windows (%)",ylabel="Independent image pair (sorted by gain)",xlim=(0,101),yticks=range(20),yticklabels=[str(i+1) for i in range(20)]); ax.legend(loc="lower left",frameon=False); ax.grid(axis="x",alpha=.2); fig.savefig(OUT/"figures/paired_coverage.pdf"); plt.close(fig)
rows=[]
for line in (ROOT/sources[1]).read_text().splitlines():
 if re.match(r"\| (translation|rigid|affine|heterogeneous|discontinuous) \| \d+/",line):
  c=[x.strip() for x in line.split("|")[1:-1]]; n,d=map(int,c[1].split("/")); f,_=map(int,c[2].split("/")); rows.append((c[0],n,f,d))
fig,ax=plt.subplots(figsize=(7,3.7)); x=np.arange(len(rows)); ax.bar(x-.17,[100*r[1]/r[3] for r in rows],.34,color="#0072B2",label="64-pixel subsets"); ax.bar(x+.17,[100*r[2]/r[3] for r in rows],.34,color="#009E73",label="32-pixel subsets, same centres"); ax.set(xticks=x,xticklabels=[r[0].capitalize() for r in rows],ylabel="Correctly converged / visible (%)",ylim=(0,105)); ax.legend(frameon=False,loc="lower left"); fig.savefig(OUT/"figures/multifamily.pdf"); plt.close(fig)
# Geometrical explanatory figure, explicitly not experimental data.
fig,axes=plt.subplots(1,3,figsize=(7,2.8)); t=np.linspace(-1,1,9); F=np.array([[1.6,.3],[0,1.]])
for ax,title,A in zip(axes,["Reference material support","Deformed material support","Fixed target support"],[np.eye(2),F,np.eye(2)]):
 for q in t:
  for v in [np.array([t,np.full_like(t,q)]),np.array([np.full_like(t,q),t])]:
   z=A@v; ax.plot(*z,color="#0072B2",lw=.6)
 if title.startswith("Fixed"):
  for q in [-1,1]:
   for v in [np.array([t,np.full_like(t,q)]),np.array([np.full_like(t,q),t])]:
    z=F@v; ax.plot(*z,color="#D55E00",lw=1.5,ls="--")
 ax.set(aspect="equal",xlim=(-2,2),ylim=(-1.4,1.4),title=title.replace(" material support"," material\nsupport").replace(" target support"," target\nsupport")); ax.axis("off")
fig.savefig(OUT/"figures/support_geometry.pdf"); plt.close(fig)
print("Generated three vector figures from archived reports and an analytical schematic")
# Controlled ablation: parse the archived propagation table directly.
src="results/ndf_dic_20261109/causal_ablation_frontier_corrected_20261109/README.md"
p=ROOT/src; rows=[]
for line in p.read_text().splitlines():
 c=[x.strip() for x in line.split("|")[1:-1]]
 if len(c)==10 and c[0].startswith(("Image-derived affine only","Affine +")):
  rows.append((c[0],int(c[1].replace(",","")),int(c[2].replace(",",""))))
assert len(rows)==4,rows
assert [r[2] for r in rows]==[19915,20651,19070,19917]
fig,ax=plt.subplots(figsize=(7,4.1));x=np.arange(4)
ax.bar(x-.18,[100*r[1]/22416 for r in rows],.36,color="#E69F00",label="One pass")
ax.bar(x+.18,[100*r[2]/22416 for r in rows],.36,color="#0072B2",label="After propagation")
ax.axhline(100*21367/22416,ls="--",color="#555555",lw=1,label="Analytical local-affine initialization")
ax.set(xticks=x,xticklabels=["Affine","Affine + NCC","Affine + NDF","Affine + NDF\n+ NCC"],ylabel="Correctly converged / visible (%)",ylim=(0,105))
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.15),frameon=False,ncol=2,fontsize=8)
fig.savefig(OUT/"figures/causal_components.pdf");plt.close(fig)
(OUT/"data/causal_frontier_README.md").write_bytes(p.read_bytes())
for src in [src,"results/descriptors/oracle_affine_compensation_20261108/README.md","results/seed_propagation_20261111/p2_gpu_integrated_four_v2/README.md"]:
 p=ROOT/src; manifest.append({"path":src,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/"data/sources.json").write_text(json.dumps(manifest,indent=2))

# Image-based examples supersede the old map-only spatial panel.
import subprocess, sys
subprocess.run([sys.executable, str(OUT / "make_image_examples.py")], check=True)

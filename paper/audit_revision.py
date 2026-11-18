"""Read-only audit of archived evidence; never executes or retunes inference."""
from pathlib import Path
import csv, hashlib, json, shutil, subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"paper"; R=ROOT/"results/seed_propagation_20261111"
records=[]
def fingerprint(path):
 p=Path(path);p=p if p.is_absolute() else ROOT/p
 rec={"path":str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size}
 records.append(rec);return rec
sources=["python/benchmarks/qualify_ndf_pair_icgn.py","python/benchmarks/ndf_pair_model.py","python/benchmarks/ndf_rotation_search.py","python/benchmarks/ndf_field.py","python/benchmarks/qualify_image_seed_pilot.py","python/benchmarks/qualify_fine_same_centres.py","python/src/wgpu_matching/seed_initialization.py","python/src/wgpu_matching/seed_propagation.py","python/src/wgpu_matching/shaders/ndf-local-ncc-parallel.wgsl","python/src/wgpu_matching/shaders/cpm-match.wgsl","python/benchmarks/train_ndf_pair_matcher.py"]
for src in sources:
 fingerprint(src);dest=OUT/"audit/source_snapshot"/src;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/src,dest)
solver=Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src/live_dic")
for src in [solver/"shaders/icgn.wgsl",solver/"shaders/reference.wgsl"]:
 fingerprint(src);d=OUT/"audit/source_snapshot/external_live_dic"/src.name;d.parent.mkdir(exist_ok=True);shutil.copyfile(src,d)
weight=ROOT/"results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt"
fingerprint(weight);fingerprint(weight.with_name("training.json"))
# Preserve exact protocol dictionaries, never infer values from another campaign.
campaigns={
 "causal":ROOT/"results/ndf_dic_20261109/causal_ablation_frontier_corrected_20261109",
 "confirmation":R/"p2_confirmatory", "confirmation_inputs":R/"p2_confirmatory_inputs",
 "support_transfer":R/"p3_fine_transfer_multifamily_independent", "shear":R/"p3_shear_band_frozen_protocol",
 "experimental":R/"p3_external_experimental/adrien_masked", "gpu_parity":R/"p2_gpu_integrated_four_v2",
 "cache_timing":R/"p2_cached_pipeline_paired_isolated"}
for name,root in campaigns.items():
 protocols=[]
 for p in sorted(root.rglob("*.json")):
  obj=json.loads(p.read_text())
  if isinstance(obj,dict) and "protocol" in obj:
   protocols.append({"source":fingerprint(p),"protocol":obj["protocol"]})
 (OUT/"configurations"/(name+".json")).write_text(json.dumps({"campaign":name,"protocols":protocols,"missing_values":"not inferred"},indent=2)+"\n")
rows=[];false=[]
causal=campaigns["causal"]
for method in ["affine_only","affine_ncc","affine_ndf","affine_ndf_ncc"]:
 for p in sorted((causal/method).glob("windows_seed*/*.npz")):
  fingerprint(p);a=np.load(p);v=a["visible"].astype(bool);status=a["propagated_status"]>.5
  err=np.linalg.norm((a["propagated_parameters"]-a["truth_parameters"])[:,[0,3]],axis=1)
  correct=v&status&np.isfinite(err)&(err<2)
  assert np.array_equal(correct,a["propagated_correct_lt2px"])
  one=v&(a["icgn_status"]>.5)&(np.linalg.norm((a["icgn_parameters"]-a["truth_parameters"])[:,[0,3]],axis=1)<2)
  bad=v&status&(err>=2)
  for i in np.flatnonzero(bad):false.append({"method":method,"pair":p.stem,"index":int(i),"error":float(err[i]),"initially_accepted":bool(a["icgn_status"][i]>.5)})
  rows.append({"method":method,"pair":p.stem,"visible":int(v.sum()),"one_pass":int(one.sum()),"propagated":int(correct.sum()),"seeded":int((a["icgn_status"]>.5).any()),"false":int(bad.sum()),"invisible":int((status&~v).sum()),"nonfinite_accepted":int((status&~np.isfinite(a["propagated_parameters"]).all(1)).sum())})
with (OUT/"data/causal_recount.csv").open("w") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
totals={m:{k:sum(r[k] for r in rows if r["method"]==m) for k in list(rows[0])[2:]} for m in {r["method"] for r in rows}}
assert [totals[m]["propagated"] for m in ["affine_only","affine_ncc","affine_ndf","affine_ndf_ncc"]]==[19915,20651,19070,19917]
# Independent campaign counts from window files, labels preserved.
confirmation={}
for family in ["heterogeneous","translation"]:
 vals=[]
 for p in sorted((R/"p2_confirmatory/evaluation").glob(family+"*.npz")):
  fingerprint(p);a=np.load(p);v=a["visible"].astype(bool);ac=a["accepted"].astype(bool);e=a["error"]
  assert np.array_equal(a["correct"],v&ac&(e<2))
  vals.append([int(v.sum()),int(a["correct"].sum()),int((v&ac&(e>=2)).sum())])
 confirmation[family]=dict(zip(["visible","correct","false"],np.array(vals).sum(0).tolist()))
assert confirmation["heterogeneous"]["correct"]==23570
assert confirmation["translation"]["correct"]==28641
# Support study counts reconstructed from states and exact truth.
fine={}
for p in sorted((R/"p3_fine_transfer_multifamily_independent/fine").glob("*.npz")):
 fingerprint(p);a=np.load(p);v=a["visible"].astype(bool);ac=a["accepted"].astype(bool);err=np.linalg.norm((a["parameters"]-a["truth"])[:,[0,3]],axis=1)
 assert np.array_equal(a["correct"],v&ac&(err<2))
 key=p.stem.split("_texture")[0];fine.setdefault(key,np.zeros(4,dtype=int));fine[key]+=np.array([v.sum(),(v&a["prior_correct"]).sum(),a["correct"].sum(),(v&ac&(err>=2)).sum()])
summary={"causal":totals,"confirmation":confirmation,"support_transfer":{k:v.tolist() for k,v in fine.items()},"false_causal":false,"audit":"archive reanalysis; not an independent experiment; no inference executed"}
(OUT/"audit/recount.json").write_text(json.dumps(summary,indent=2)+"\n")
(OUT/"audit/source_manifest.json").write_text(json.dumps(records,indent=2)+"\n")
print(json.dumps(summary,indent=2))

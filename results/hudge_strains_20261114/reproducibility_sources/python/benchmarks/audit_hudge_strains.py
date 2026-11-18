"""Reproduce data inventory and coarse-mask diagnostics, without running inference."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from hudge_strains_data import material_mask, read_ascii_vtu
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/home/jeff/CNRS/DIC/HudgeStrains"))
    parser.add_argument("--output", type=Path, default=Path("results/hudge_strains_20261114"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory, backgrounds, guard = [], [], []
    for family in ["FEA", "ClosedFormExpression"]:
        directory = args.data/family
        reference = np.asarray(Image.open(directory/"Img_F.tif"))
        mask = material_mask(reference)
        expected_mask = args.output/f"{family}_reference_mask.png"
        if expected_mask.exists():
            assert np.array_equal(mask, np.asarray(Image.open(expected_mask)) > 0)
        else:
            Image.fromarray((255*mask).astype(np.uint8)).save(expected_mask)
        for path in sorted(directory.glob("*.tif")):
            a = np.asarray(Image.open(path))
            yy, xx = np.where(a != reference[-1,-1])
            inventory.append({"family": family, "file": str(path),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "shape": list(a.shape), "background": int(reference[-1,-1]),
                              "nonbackground_box": [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]})
        for marker in sorted((args.output/family).glob("*/completed.json")):
            record = json.loads(marker.read_text())
            target = np.asarray(Image.open(record["target"]))
            target_mask = material_mask(target)
            archive = next((marker.parent/"inputs/affine_ncc").rglob("*.npz"))
            with np.load(archive) as z:
                pairs = z["hard_match_pairs"]
            xy = np.rint(pairs).astype(int)
            flags = []
            for start, domain in [(0,mask),(2,target_mask)]:
                x,y = xy[:,start:start+2].T
                inside = (x>=0)&(x<domain.shape[1])&(y>=0)&(y<domain.shape[0])
                valid = np.zeros(len(x),bool)
                valid[inside] = domain[y[inside],x[inside]]
                flags.append(valid)
            a,b = flags
            backgrounds.append({"family": family, "level": record["level"], "matches":len(a),
                                "source_on_background":int((~a).sum()), "both_on_material":int((a&b).sum()),
                                "source_material_to_background":int((a&~b).sum()), "source_material":int(a.sum())})
            if record["level"] == 1:
                bins = np.clip((pairs[a,:2]/np.array(reference.shape[::-1])*3).astype(int),0,2)
                guard.append({"family":family,"material_query_count":int(a.sum()),"total_queries":len(a),
                              "maximum_full_image_bins_for_material_queries":len(np.unique(bins[:,0]+3*bins[:,1])),
                              "required_bins":4,"source":"qualify_ndf_pair_icgn.py valid_affine spatial_bins>=4",
                              "interpretation":"Material-only queries cannot satisfy this full-raster coverage guard."})
    fields = []
    for path in sorted((args.data/"DEPL/DEPL").glob("*.vtu")):
        points, u = read_ascii_vtu(path)
        fields.append({"file":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                       "nodes":len(points),"coordinates_min":points.min(0).tolist(),"coordinates_max":points.max(0).tolist(),
                       "displacement_min":u.min(0).tolist(),"displacement_max":u.max(0).tolist()})
    tree = ET.parse(args.data/"DEPL/DEPL/depl.pvd")
    outputs = {"inventory.json":inventory,"coarse_background_diagnostic.json":backgrounds,
               "roi_coverage_guard.json":guard,"displacement_audit.json":{
                   "files":fields,"time_map":[e.attrib for e in tree.findall(".//DataSet")],
                   "association":"unresolved; need field/image mapping, units, origin, axes and analytical formula"}}
    for name, value in outputs.items():
        (args.output/name).write_text(json.dumps(value,indent=2)+"\n")
    print(len(inventory),"images;",len(fields),"displacement files;",len(backgrounds),"coarse diagnostics")


if __name__ == "__main__":
    main()

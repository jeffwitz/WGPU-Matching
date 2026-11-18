"""Score saved states only after explicit image/mesh association is provided.

The association manifest must supply reference, target, archive, vtu,
physical_to_pixel (2x2), pixel_origin (2), and total_from_image_reference=true.
This script never participates in inference or estimates that coordinate map.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from hudge_strains_data import evaluate_vtu_truth
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--associations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.associations.read_text())
    rows = []
    args.output.mkdir(parents=True, exist_ok=False)
    for i, entry in enumerate(entries):
        if entry.get("total_from_image_reference") is not True:
            raise ValueError("DEPL must be confirmed as total displacement from the image reference")
        if entry.get("physical_to_pixel") is None or entry.get("pixel_origin") is None:
            raise ValueError("coordinate map must be supplied, never fitted from the computed states")
        with Image.open(entry["reference"]) as image:
            ref_size = image.size
        with Image.open(entry["target"]) as image:
            width, height = image.size
        if ref_size != (width, height):
            raise ValueError("this scorer expects the same pixel frame in both images")
        with np.load(entry["archive"]) as data:
            centres, parameters, accepted, valid = (data[k] for k in ["centres", "parameters", "accepted", "valid_roi"])
        truth, material = evaluate_vtu_truth(entry["vtu"], centres, entry["physical_to_pixel"], entry["pixel_origin"])
        target_xy = centres + truth[:, [0, 3]]
        visible = material & valid & (target_xy[:,0]>=0) & (target_xy[:,0]<=width-1) & (target_xy[:,1]>=0) & (target_xy[:,1]<=height-1)
        finite = np.isfinite(parameters).all(1)
        error = np.linalg.norm(parameters[:, [0, 3]]-truth[:, [0, 3]],axis=1)
        correct = visible & accepted & finite & (error<2)
        false = visible & accepted & finite & (error>=2)
        scored = visible & accepted & finite
        row = {"association":entry,"visible_centres":int(visible.sum()),"correct_lt2px":int(correct.sum()),
               "false_ge2px":int(false.sum()),"accepted_nonfinite":int((accepted & ~finite).sum()),
               "accepted_outside_truth":int((accepted & ~visible).sum()),
               "displacement_rmse_px":float(np.sqrt(np.mean(error[scored]**2))) if scored.any() else None,
               "gradient_component_rmse":float(np.sqrt(np.mean((parameters[scored][:,[1,2,4,5]]-truth[scored][:,[1,2,4,5]])**2))) if scored.any() else None}
        np.savez_compressed(args.output/f"pair_{i:03d}.npz",truth=truth,visible=visible,error=error,correct=correct,false=false)
        rows.append(row)
    (args.output/"scores.json").write_text(json.dumps(rows,indent=2)+"\n")


if __name__ == "__main__":
    main()

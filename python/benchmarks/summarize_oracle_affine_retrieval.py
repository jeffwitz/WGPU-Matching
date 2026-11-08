"""Create a readable report from the oracle affine retrieval JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHOD_LABELS = {
    "zncc": "Zero-mean normalized cross-correlation",
    "horg64": "Histogram of oriented robust gradients (64 components)",
    "dic_cnn": "Compact DIC convolutional descriptor",
    "dedode_b": "DeDoDe-B learned descriptor",
}


def count_rate(count: int, denominator: int) -> str:
    rate = 100.0 * count / denominator if denominator else 0.0
    return f"{count}/{denominator} ({rate:.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    output = args.output or args.input.with_name("README.md")
    protocol = result["protocol"]
    lines = [
        "# Oracle affine compensation: descriptor retrieval diagnostic",
        "",
        "## Purpose and scope",
        "",
        "This is a diagnostic ceiling experiment for digital image correlation (DIC), not a usable initializer. Synthetic analytic motion is used to warp the deformed image back into a reference material frame and to identify correct matches. No analytic correspondence is used to choose, filter, or pre-rank the target candidates. The same target detection pool is presented to each descriptor before and after compensation.",
        "",
        "The tested methods are zero-mean normalized cross-correlation (ZNCC), HORG-64 (a 64-component histogram of oriented robust gradients), a compact convolutional neural network (CNN) descriptor trained for speckle, and the official DeDoDe-B descriptor (Detect, Don't Describe—Describe, Don't Detect, B variant). HORG-64 and the compact CNN descriptor are extracted with WebGPU Shading Language (WGSL) kernels. DeDoDe-B is sampled through an Open Neural Network Exchange (ONNX) graph; the runtime reports both WebGPU and central processing unit (CPU) execution providers, so this experiment does not claim that every graph operator ran on the graphics processing unit (GPU). Candidate ranking is computed from returned descriptor arrays on the host.",
        "",
        f"Source images: {result['results'][0]['image_size_px'][0]} × {result['results'][0]['image_size_px'][1]} pixels. Candidate budget: {protocol['candidate_budget']:,}. The detector, HORG-64 and compact CNN use native-resolution image pixels. DeDoDe-B receives {protocol['dedode_runtime_metadata']['model_size_px'][0]} × {protocol['dedode_runtime_metadata']['model_size_px'][1]} images; its descriptor map is sampled at the selected keypoints. ZNCC samples a 32 × 32 lattice across a 64 × 64 native-pixel support. Detector and WGSL descriptor device: {protocol.get('webgpu_adapter', 'not recorded')}.",
        "",
        f"Compact CNN weights SHA-256: `{protocol['cnn_weights_sha256']}`. DeDoDe-B graph SHA-256: `{protocol['dedode_graph_sha256']}`. DeDoDe execution providers: `{', '.join(protocol['dedode_runtime_metadata'].get('providers', []))}`.",
        "",
        "## Results: global rigid and affine cases",
        "",
            "`Repeatable baseline / oracle` reports the number of queries with a target candidate within 5 pixels before and after compensation, each out of 512 source queries. The counts can differ because the affine pullback changes which candidate supports remain inside valid target pixels. Every retrieval rate uses the repeatable count from its own column as denominator. Top-1 ≤2 px and top-1 ≤5 px are separate accuracy criteria. Rows use the same 512 source queries and initial 8,192 target candidates before support-validity filtering.",
        "",
        "| Motion | Descriptor | Repeatable baseline / oracle | Baseline top-1 ≤2 px | Baseline top-1 ≤5 px | Baseline top-5 | Oracle top-1 ≤2 px | Oracle top-1 ≤5 px | Oracle top-5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["results"]:
        if case["oracle_type"] != "global_affine":
            continue
        for method, label in METHOD_LABELS.items():
            values = case["retrieval"][method]
            base, oracle = values["baseline"], values["oracle"]
            n = base["repeatable_le5px"]
            lines.append(
                f"| {case['family']} | {label} | {n}/{oracle['repeatable_le5px']} of 512 | "
                f"{count_rate(base['top1_correct_le2px'], n)} | "
                f"{count_rate(base['top1_correct_le5px'], n)} | "
                f"{count_rate(base['top5_retrieval'], n)} | "
                f"{count_rate(oracle['top1_correct_le2px'], oracle['repeatable_le5px'])} | "
                f"{count_rate(oracle['top1_correct_le5px'], oracle['repeatable_le5px'])} | "
                f"{count_rate(oracle['top5_retrieval'], oracle['repeatable_le5px'])} |"
            )
    lines.extend(
        [
            "",
            "## Results: heterogeneous field with query-local affine oracle",
            "",
            "For each of 32 source queries, the local Jacobian is estimated from the analytic displacement field using central differences over a 4-pixel baseline. The corresponding affine hypothesis is then applied identically to all 8,192 target candidates for that query. Candidate support validity is checked after mapping; no candidate is selected using its true match identity. The solver discussed in the final validation criterion is inverse compositional Gauss–Newton (IC-GN).",
            "",
            "| Descriptor | Repeatable baseline / oracle queries | Baseline top-1 ≤2 px | Baseline top-1 ≤5 px | Baseline top-5 | Oracle top-1 ≤2 px | Oracle top-1 ≤5 px | Oracle top-5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    local = next(case for case in result["results"] if case["oracle_type"] == "query_local_affine")
    for method, label in METHOD_LABELS.items():
        values = local["retrieval"][method]
        base = values["baseline"]
        oracle_rows = values["oracle"] or []
        repeatable = sum(row["repeatable_le5px"] for row in oracle_rows)
        lines.append(
        f"| {label} | {base['repeatable_le5px']}/{repeatable} of 32 | "
            f"{count_rate(base['top1_correct_le2px'], base['repeatable_le5px'])} | "
            f"{count_rate(base['top1_correct_le5px'], base['repeatable_le5px'])} | "
            f"{count_rate(base['top5_retrieval'], base['repeatable_le5px'])} | "
            f"{count_rate(sum(row['top1_correct_le2px'] for row in oracle_rows), repeatable)} | "
            f"{count_rate(sum(row['top1_correct_le5px'] for row in oracle_rows), repeatable)} | "
            f"{count_rate(sum(row['top5_retrieval'] for row in oracle_rows), repeatable)} |"
        )
    lines.extend(
        [
            "",
            f"Query-local target image warping took {local['oracle_target_warp_s']:.2f} s, and target descriptor extraction plus patch sampling took {local['oracle_target_description_s']:.2f} s for 32 queries. This naïve query-conditioned cost is diagnostic only; the same full image is warped repeatedly and these timings are not production pipeline performance.",
            "",
            "## Interpretation",
            "",
            "For the rigid case, oracle affine compensation raises the compact CNN descriptor from 0/486 top-1 correct within 5 pixels to 474/486, and DeDoDe-B from 0/486 to 485/486. For the affine case, the compact CNN rises from 5/431 to 290/419, while DeDoDe-B rises from 218/431 to 413/419. In the query-local heterogeneous diagnostic, the compact CNN returns 28/30 repeatable queries at top-1 within 5 pixels, compared with 0/29 before compensation; DeDoDe-B returns 27/30 compared with 0/29. Baseline and oracle denominators differ because affine warping changes target support validity.",
            "",
            "This supports the hypothesis that geometric mismatch between fixed square supports is a major part of the failure, and that the current descriptors retain useful identity information after local affine normalization. It does not prove that geometry is the only cause: synthetic images provide exact analytic motion, and this procedure uses it as an oracle. Detection repeatability also limits the maximum coverage (486/512 rigid and 419/512 affine). ZNCC improves less than learned descriptors, and HORG-64 improves only partially; compensation does not make every descriptor equally discriminative.",
            "",
            "The diagnostic does not establish a practical method. The next experiment must replace the oracle with a finite set of image-derived affine hypotheses, retain several candidates per source query, and use a common geometric consensus stage. In the heterogeneous case, the 32-query local-oracle run is an initial diagnostic sample rather than a full statistical qualification. The final evaluation should include independent seeds, IC-GN convergence, and runtime including image warping and descriptor recomputation.",
            "",
            "## Protocol and reproducibility",
            "",
            f"WebGPU adapter: {protocol.get('webgpu_adapter', 'not recorded')}. DeDoDe-B providers: `{', '.join(protocol['dedode_runtime_metadata'].get('providers', []))}`. The provider list includes CPU fallback; no operator-level profile was collected in this experiment.",
            "",
            "Pillow bicubic resampling pulls target intensities into the analytic material frame; the validity mask uses nearest-neighbour resampling. All points and candidate identities are kept in floating-point coordinates. Each family uses the same seed pair as the earlier full-image descriptor retrieval experiment. Exact parameters, image dimensions, candidate counts, local Jacobian matrices, and timings are in the adjacent JSON file.",
            "",
            "The DeDoDe-B ONNX graph was exported from the official Parskatt/DeDoDe implementation at a temporary checkout and used the locally cached official descriptor weights. Its 1024 × 576 input is downsampled from the 3840 × 2160 source; this differs from the native-pixel support used by HORG-64, the compact CNN, and ZNCC. Neither third-party source code nor model weights are included in this repository. The graph SHA-256 and weight SHA-256 are recorded in the JSON and model metadata.",
            "",
            "## Scientific references",
            "",
            "* Mishchuk et al., *Working Hard to Know Your Neighbor’s Margins* (NeurIPS 2017), trains local descriptors with hard negatives. Its setup does not by itself make a fixed raw patch affine invariant: [paper](https://papers.neurips.cc/paper_files/paper/2017/hash/831caa1b600f852b7844499430ecac17-Abstract.html).",
            "* Mishkin, Radenovic and Matas, *Repeatability Is Not Enough: Learning Affine Regions via Discriminability* (ECCV 2018), learns affine-covariant support regions as well as matchability: [paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Dmytro_Mishkin_Repeatability_Is_Not_ECCV_2018_paper.html).",
            "* Pautrat et al., *Online Invariance Selection for Local Feature Descriptors* (ECCV 2020), selects among representations with different invariance from image-pair context: [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470698.pdf).",
            "* Li, Li and Liang, *Enhanced feature-based path-independent initial value estimation for robust point-wise digital image correlation* (2019), uses dense gradient descriptors, a PatchMatch variant, forward-backward consistency, and IC-GN: [article](https://doi.org/10.1016/j.optlaseng.2019.04.016).",
            "* Bökman et al., *Affine steerers for structured keypoint description* (ECCV 2024), learns approximately equivariant local descriptors for affine transformations: [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11742_ECCV_2024_paper.php).",
            "* von Witzleben and Haouchine, *Stretcher: a learning-based framework for deformation-robust keypoint descriptors* (2026), learns local affine-conditioned transformations in descriptor space. Its experiments concern surgical images, not speckle DIC: [article](https://link.springer.com/article/10.1007/s11548-026-03698-w).",
            "* Zhou, Pan and Chen, *Large deformation measurement using digital image correlation: a fully automated approach* (2012), combines robust feature matches, local affine initialization, and reliability-guided propagation: [article](https://doi.org/10.1364/AO.51.007674).",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

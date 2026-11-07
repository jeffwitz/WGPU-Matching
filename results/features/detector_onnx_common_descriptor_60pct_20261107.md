# DaD/DeDoDe v2 ONNX-WebGPU versus PyTorch at 60% traction

## Protocol

The synthetic tensile pair is 3840 × 2160 px with analytic 60% axial
engineering strain. Both detector paths use the same official weights and
640 × 360 model input, and return a candidate pool of 1024 points. The
comparison selects 512 points, extracts the same DeDoDe-B descriptor on CPU,
performs mutual cosine matching on CPU, and uses identical WGSL local-affine
RANSAC settings. Sixteen analytic DIC seeds are visible.

The table reports three measured repeats after one warm-up. The separate
single-repeat raw files are preserved for provenance. The ONNX detector path
runs through ONNX Runtime's WebGPU provider and returns small point/score arrays
to NumPy. RANSAC uses WGPU-Matching's WebGPU context. Descriptor extraction and
mutual matching are still CPU reference stages.

## Results, global top-K selection

| Detector | Extraction path | Repeatable ≤2 px | Raw matches (<5 px) | Affine accepted / correct <5 px | Affine RMSE / P95 (px) | Detect pair (s) | Describe pair (s) | Full pair (s) | Full pair FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DaD | PyTorch CPU | 71/345 | 252 (169) | 6/16 (6) | 2.24 / 3.55 | 1.447 | 2.191 | 3.645 | 0.274 |
| DaD | ONNX Runtime WebGPU | 71/346 | 253 (168) | 6/16 (6) | 2.14 / 3.52 | 0.391 | 2.569 | 2.969 | 0.337 |
| DeDoDe v2-L | PyTorch CPU | 15/331 | 206 (27) | 0/16 (0) | — | 3.709 | 2.583 | 6.302 | 0.159 |
| DeDoDe v2-L | ONNX Runtime WebGPU | 15/331 | 206 (27) | 0/16 (0) | — | 1.056 | 2.644 | 3.708 | 0.270 |

For DaD, detector-pair time fell by 73% (3.7×), while full-pair time fell by
19% (1.23×). For DeDoDe v2-L, detector-pair time fell by 72% (3.5×), and
full-pair time fell by 41% (1.70×). The descriptor/matcher cost limits the
end-to-end benefit, particularly for DaD in this run.

Affine quality is preserved at this budget: DaD gives 6/16 accepted seeds
under 5 px through both paths, with nearly identical error statistics;
DeDoDe v2 gives 0/16 through both. DaD has a one-count difference in
repeatability and raw candidates due to near-tied keypoint ordering. The
spatial-quota selection also gives 4/16 correct affine seeds for DaD and 0/16
for DeDoDe v2 in both execution paths.

## ONNX detector-stage timings

For the 4K source pair, the ONNX adapter resizes each image to 640 × 360. Median
times are collected from the three repeated reference and target calls:

| Detector | Preprocessing pair (s) | ONNX `session.run` pair (s) | Coordinate conversion pair (s) | Total detect pair (s) |
|---|---:|---:|---:|---:|
| DaD | 0.130 | 0.262 | <0.001 | 0.391 |
| DeDoDe v2-L | 0.145 | 0.914 | <0.001 | 1.056 |

The separate five-repeat 640 × 360 random-input audit measured 119.63 ms for
DaD and 405.26 ms for DeDoDe v2-L per ONNX `session.run`. Its adapter and
provider profile details are in
`detector_onnx_webgpu_20261107.md` and the adjacent raw JSON/profile files.

## Interpretation and limits

- This comparison demonstrates that keypoint-only ONNX/WebGPU export preserves
  the tested common-descriptor matching result while reducing detector time
  relative to CPU PyTorch reference inference.
- It does **not** show that DaD or DeDoDe v2 outperform DeepMatching. The
  comparison isolates inference backend for these two detectors.
- The neural detectors see a 640 × 360 image, not native 4K. Their point
  coordinates are mapped back to the 3840 × 2160 source for descriptor and
  affine analysis.
- DeDoDe-B descriptor extraction and mutual matching run on CPU. Complete
  pair throughput is 0.337 FPS for DaD and 0.270 FPS for DeDoDe v2 in this
  protocol; this is not a GPU-resident pipeline.
- The dataset contains one globally affine synthetic stress pair. This does
  not validate heterogeneous deformation, experimental images, IC-GN
  convergence, peak memory, or timing variance across runs.

Repeated raw data:
`detector_onnx_webgpu_20261107/common_descriptor_60pct_repeats3.json`. Earlier
single-repeat raw runs remain preserved as
`common_descriptor_60pct.json` and
`common_descriptor_60pct_torch_reference.json` in the same directory.

Timing qualification: the reported full-pair total is a sum of per-stage
medians, not the measured median of complete end-to-end calls. The raw stage
repetitions remain the evidence for its construction.

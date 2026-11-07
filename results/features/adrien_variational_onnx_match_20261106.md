# ONNX feature matching on AdrienVariational images — 2026-11-06

## Input and protocol

The pair was read as image 000000 (reference) → image 000075 (target). Both TIFFs are 7920 × 6004 grayscale (about 47.6 MP each). Models received 640 × 485 content, padded at the bottom to 640 × 512, a downscale of about 12.4× per axis. `Mask.tiff` channel 0 was used only to count correspondences whose endpoints lie inside its ROI; the model images were not masked. No files in the external image directory were changed.

The models ran through ONNX Runtime 1.31.1 with the native WebGPU EP 0.5.0 on Intel Graphics LNL. `session.run` timings include the runtime’s transfers. Input decoding/resizing, session startup, and the separately timed CPU affine RANSAC are excluded. Each timing is one warm-up plus five measured `session.run` calls. The RaCo Selu rewrite uses a temporary model copy.

## Results

| Method | ROI matches | Global affine inliers (30 px threshold) | Affine residual median / RMSE | Inference time median |
|---|---:|---:|---:|---:|
| RaCo original | 680 | 677/680 (99.56%) | 7.65 / 10.10 px | 2.399 s |
| RaCo SELU→WebGPU | 680 | 677/680 (99.56%) | 7.65 / 10.10 px | 1.390 s |
| EfficientLoFTR | 2675 | 2673/2675 (99.93%) | 6.88 / 8.87 px | 1.579 s |

RaCo original measured 2.399 s median (MAD 0.042 s); the SELU-decomposed graph measured 1.390 s (MAD 0.010 s), a 42.1% time reduction. Their ROI match counts and affine inlier counts were identical in the measurement run. The estimated affine transforms differed by less than 0.00001 in the reported coefficients.

EfficientLoFTR returned 3,272 raw correspondences; 2,675 had both endpoints in the mask ROI, with 2,673 affine inliers. The global transform estimates from both model families were close: about 1.092 horizontal scale, 0.955 vertical scale, and translation (-368, +92) px.

## Interpretation and limits

Both methods produced dense, mutually consistent matches for this pair at 640 × 512 network input. This is evidence that the pair can be matched after substantial downsampling. The affine RANSAC residual measures agreement with one fitted global affine model, not error against measured displacement or DIC ground truth. It does not establish subpixel accuracy or convergence of IC-GN. The ROI mask was not applied to model inference.

The model outputs were returned to host memory for geometric filtering in this experiment, and CPU RANSAC took about 0.10–0.17 s. This is not a GPU-resident 4K pipeline. No native-resolution refinement was performed.

Detailed run metadata, per-model counts, timing samples, transform coefficients, model hashes, and input hashes are in [`adrien_variational_onnx_match_20261106.json`](adrien_variational_onnx_match_20261106.json).

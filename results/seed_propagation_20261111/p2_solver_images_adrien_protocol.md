# Solver-image cache check on the masked Adrien pair

Fixed before running: use the original `img_000000_0.09422.tiff`,
`img_000075_1608.88508.tiff` and `Mask.tiff`, native 7920 × 6004 pixels, model
raster 192 × 146, seed group 34000000 and frozen P2 thresholds. Produce one
shared-prefit archive with exact 8-bit solver-image exports. Run the same
image-only initializer once by reading the original image pair and once by
reading the fingerprint-verified solver images. Keep thresholds, windows,
lazy-rotation policy, mask and solver settings identical.

Compare accepted-window masks, parameters and provenance exactly. Source-image
file hashes must match the cache metadata. The pair has no analytical truth:
report acceptance and parity only, not accuracy, convergence correctness or a
claim about the mechanical validity of accepted windows. Do not tune parameters
on this comparison. These diagnostics are not production latency measurements.

# Masked Adrien image pair

The image-only initializer was run on `img_000000_0.09422.tiff` and `img_000075_1608.88508.tiff` at their native 7920 × 6004 resolution, using the supplied `Mask.tiff` for inverse-compositional Gauss–Newton (IC-GN) window validity. The coarse neural image raster was 192 × 146. Hypothesis generation did not use the region mask; the mask was used for the window solver and propagation.

Of 7,280 region-valid windows, 6,321 were confirmed as seeds and 776 additional windows were accepted through propagation: **7,097 accepted windows** in total. The [accepted map](adrien_masked/accepted_map.png) shows confirmed seeds in blue, other accepted windows in green, unaccepted valid windows in gray and outside-region windows in black.

There is no analytical ground truth for this pair. These are acceptance and provenance counts, not accuracy or measurement-validity claims. The local solver and region mask still determine whether a window is eligible; no claim is made about unseen material boundaries. Details are in `summary.json` and `adrien_masked/qualification/evaluation/`.

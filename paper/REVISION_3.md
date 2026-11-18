# Revision 3: images behind the spatial maps

- Replaced map-only Figure 5 with the actual heterogeneous image pair, outcome overlay and native 32-pixel support before/after the archived affine correction.
- Added Figure 6 for a synthetic discontinuity, with the same display convention.
- Added Figure 7 for native 7920 x 6004 experimental Adrien frames 000000/000075: 7097 accepted of 7280 mask-valid centres; no accuracy claim.
- The local examples use the accepted/correct non-crossing centre nearest the image centre, not the smallest residual. The transformation comes only from archived final solver parameters. No new qualification or initializer-only result is claimed.
- Full-frame views are reduced for print. Patches use native samples and bilinear interpolation. Reference and target grayscale is fixed to [0,1]; absolute differences use [0,0.25] with overflow indicated and no photometric correction.
- `data/image_examples.json` records image/result hashes, centre indices and exact affine parameters. `make_figures.py` invokes the reproducible image-example script.
- Earlier manuscript preserved in `versions/v2/`. PDF pages 14–16 inspected after rendering; no undefined references or overfull boxes.

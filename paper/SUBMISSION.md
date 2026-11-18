# Strain requirements and remaining author decisions

Checked 14 November 2026:
https://onlinelibrary.wiley.com/page/journal/14751305/homepage/forauthors.html
https://authorservices.wiley.com/author-resources/Journal-Authors/Prepare/new-journal-design.html

Strain permits Free Format initial submissions. It requests an informative unabbreviated title, an unstructured abstract <=300 words, <=5 keywords, English, tables/captions, references with identifiers, and DIC processing settings following the International Digital Image Correlation Society guide. Typical main-text length is around 8,000 words, not an asserted hard minimum. This draft is shorter. LaTeX submissions require a compiled review PDF and supporting sources. Review is single anonymized. Author contribution roles are required at revision.

Official archive downloaded from https://authorservices.wiley.com/asset/WileyNJDv5_Template.zip and preserved under vendor/. The current download contains Optimal-Design-layout/USG.cls and an unrelated journal sample; it does not contain WileyNJDv5.cls. Therefore the compiled draft uses the explicitly allowed Free Format route rather than assigning an unverified journal class/style. The official archive is retained for subsequent production adaptation. No third-party imitation class is used.

Before submission:
- Confirm full authors, affiliations, corresponding author, contribution roles, funding, conflicts and acknowledgements.
- Complete a single executable release and persistent data deposit, including source hashes and actual solver settings for each campaign. Current source hashes do not by themselves distribute missing data.
- Verify bibliography metadata and replace preprints with final journal versions where appropriate; expand the comparison literature if claims are broadened.
- Add a documented physical experiment with independent truth or explicitly retain experimental images as qualitative only. Confirm image publication permissions.
- Complete end-to-end timing on representative hardware; do not infer full throughput from kernel measurements.
- Disclose assisted drafting/coding under Wiley policy after author review.
- Provide separate figure files at revision. Vector graphics supplied; rasterized spatial points export at 600 dpi. All figures are original; no publisher artwork reproduced.
- Resolve the severe-shear failure as a limitation, not by hiding its data or retuning test thresholds.

Novelty claim is deliberately scoped: a portable solver-confirmed initialization workflow, controlled ablations, and explicit propagation/support provenance. Neither reliability-guided propagation nor affine shape correction is claimed as newly invented.

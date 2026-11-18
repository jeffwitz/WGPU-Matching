# Fixed protocol: batched anchor confirmation on new synthetic pairs

Recorded before generation or inspection. Use frozen P2 settings, seed groups
55000000 and 56000000, and one heterogeneous and one rigid case per group:
four synthetic 3840 × 2160 image pairs. Generate shared-prefit archives and
fingerprinted solver-image caches once. Run serial and batched anchor
confirmation on exactly the same cached pixels and hypotheses, with the
image-only deferred-rotation policy and the same inverse-compositional
Gauss–Newton (IC-GN) settings. Truth is used only to score after inference.

Compare all window parameters, acceptance masks, anchor masks and propagation
provenance, and separately report correct and falsely accepted visible windows
for both modes. Record solver-call and active-window counts. Retain differences
and negative outcomes. Do not change thresholds or select cases after reading
results. This small sample does not establish general robustness or production
latency. The graphics processing unit (GPU) is used by the existing solver;
the confirmation orchestration is not GPU-resident.

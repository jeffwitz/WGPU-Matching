# Verification-stage timing breakdown

One previously inspected synthetic 3840 × 2160 pair, one warm-up and three
measured process runs. The timer starts inside the Python entry point and does
not include interpreter startup or imports. Times are host-observed durations,
not graphics processing unit (GPU) timestamps. The median was:

| Stage | Median |
|---|---:|
| Synthetic image-pair regeneration and photometric perturbation | 4.846 s |
| Reference upload/preparation and target quantization | 0.851 s |
| Image-only hypothesis verification and propagation | 0.738 s |
| Whole case before output writes | 6.437 s |

The reference and target are regenerated because this diagnostic starts a
second process; synthetic generation must not be mistaken for the cost of
reading a real camera image. The stages do not include interpreter startup and
must not be added to an already measured full-process time. Per-run timing and
solver results are in each `run*/summary.json`.

The corresponding fingerprinted image-reuse test is in
`../p2_solver_images_benchmark/README.md`. It explicitly includes image-cache
creation in the paired full-script comparison, and does not claim that image
buffers are GPU-resident.

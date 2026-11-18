"""Verify composed proposals with the existing GPU solver on ORIGINAL images."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from experiment_hudge_hole_rbf import corr, evaluate
from hudge_hole_moments import sample
from hudge_strains_data import material_mask
from iterate_hudge_hole_rbf import Composition, load_field
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--iteration', type=int, help='Verify a prefix; zero is the initial RBF without composition')
    parser.add_argument('--seed-archive', type=Path, help='Verify exported native states on their original grid')
    parser.add_argument('--output', type=Path, help='Directory for verification results')
    parser.add_argument('--images', nargs=2, type=Path, help='Alternative rendering of the same reference plate for a paired control')
    parser.add_argument('--prior-archive', type=Path, default=Path('results/hudge_boundary_frontier_20261115/partial12/seeds.npz'))
    args = parser.parse_args()
    root = args.folder
    report = json.loads((root/'summary.json').read_text())
    info = json.loads(Path('results/hudge_hole_moments_20261117/proposals/summary.json').read_text())
    reference, target = [np.asarray(Image.open(p)) for p in (args.images or info['input_sha256'])]
    field = Composition(load_field(report['initial']))
    effective_iteration = 0
    for step in report['steps']:
        if step['chosen_alpha'] > 0 and (args.iteration is None or step['iteration'] <= args.iteration):
            residual = load_field(root/f"residual_{step['iteration']:02d}.npz")
            residual.coefficients *= step['chosen_alpha']
            field.residuals.append(residual)
            effective_iteration = step['iteration']
    tag = f'original_image_verification_{effective_iteration:02d}'
    sys.path.insert(0, '/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src')
    from live_dic import ICGNWGPU
    from live_dic.roi import unpack_support
    start = time.perf_counter()
    size, stride, half = 12, 4, 5.5
    solver = ICGNWGPU(device_type='gpu', prefer_subgroups=False, subset_size=size,
                     subset_stride=stride, interpolation='bspline', motion_model='affine', outlier_policy='disabled')
    solver.set_reference(reference, iterations=30, tolerance=.001, zncc_min=.65,
                         roi_mask=material_mask(reference), roi_policy='partial', roi_truncation='visibility')
    roi = solver.roi
    yy, xx = np.indices(roi.selected.shape)
    centres = np.column_stack((xx.ravel()*stride+half, yy.ravel()*stride+half))
    active = np.flatnonzero(roi.selected.ravel())
    parameters = np.zeros((len(centres), 6), dtype=np.float32)
    parameters[active] = evaluate(field, centres[active])
    if args.seed_archive:
        archive = np.load(args.seed_archive)
        np.testing.assert_array_equal(archive['centres'], centres)
        use = archive['accepted'] & roi.selected.ravel()
        parameters[use] = archive['parameters'][use]
        tag = 'native_restart_verification'
    if args.output:
        root = args.output
        root.mkdir(parents=True, exist_ok=True)
    proposal = parameters.copy()
    status = np.full(len(centres), -99, np.float32)
    status[active] = 0
    blocks = []
    for block in range(4):
        pending = np.flatnonzero((status == 0) & np.isfinite(parameters).all(1))
        if not len(pending):
            break
        packed = np.zeros((len(centres), 8), np.float32)
        packed[pending, :6] = parameters[pending]
        enabled = np.zeros(len(centres), np.uint32)
        enabled[pending] = 1
        result = solver._register_seeded_frame(target, packed, enabled.reshape(roi.selected.shape))
        parameters[pending] = np.asarray(result.parameters).reshape(-1, 6)[pending]
        status[pending] = np.asarray(result.status).ravel()[pending]
        blocks.append({'block': block+1, 'attempted': len(pending), 'accepted_total': int((status == 1).sum())})
    supports = {int(i): unpack_support(w, size).reshape(size, size) for i, w in zip(roi.partial_indices, roi.support_words, strict=True)}
    dy, dx = np.mgrid[:size, :size].astype(float)-half
    offsets = np.column_stack((dx.ravel(), dy.ravel()))
    accepted = (status == 1) & roi.selected.ravel() & np.isfinite(parameters).all(1)
    scores = np.full(len(centres), np.nan)
    for i in np.flatnonzero(accepted):
        support = supports.get(int(i), np.ones((size, size), bool)).ravel()
        p = centres[i]+offsets[support]
        f = np.eye(2)+parameters[i, [1, 2, 4, 5]].reshape(2, 2)
        q = centres[i]+parameters[i, [0, 3]]+offsets[support]@f.T
        scores[i] = corr(sample(reference, p), sample(target, q))
        accepted[i] &= scores[i] >= .65 and np.linalg.det(f) > 0 and ((q >= 0) & (q <= np.array(target.shape[::-1])-1)).all()
    old = np.load(args.prior_archive)
    np.testing.assert_allclose(centres, old['centres'], atol=1e-10, rtol=0)
    prior = old['accepted']
    # Keep old observations immutable; new measurements are separate and can
    # subsequently seed propagation without replacing a previously accepted state.
    merged = old['parameters'].copy()
    new = accepted & ~prior
    merged[new] = parameters[new]
    np.savez_compressed(root/f'{tag}.npz', centres=centres, proposal=proposal,
                        parameters=parameters, status=status, accepted=accepted, scores=scores,
                        prior_accepted=prior, merged_accepted=prior | accepted, merged_parameters=merged)
    radius = np.stack([np.linalg.norm(centres-np.array(h['centre']), axis=1)/np.sqrt(h['area']/np.pi)
                       for h in info['source_holes']]).min(0)
    bands = []
    for lo, hi in [(1, 1.25), (1.25, 1.5), (1.5, 2), (2, 3)]:
        use = roi.selected.ravel() & (radius >= lo) & (radius < hi)
        bands.append({'radius': [lo, hi], 'eligible': int(use.sum()), 'prior_accepted': int((use & prior).sum()),
                      'composed_accepted': int((use & accepted).sum()), 'new': int((use & new).sum())})
    result = {'iteration': effective_iteration, 'eligible': len(active), 'prior_accepted': int(prior.sum()), 'composed_accepted': int(accepted.sum()),
              'new_original_image_solves': int(new.sum()), 'old_not_recovered': int((prior & ~accepted).sum()),
              'union_preserving_old_seeds': int((prior | accepted).sum()), 'bands': bands, 'blocks': blocks,
              'seconds': time.perf_counter()-start,
              'note': 'Solver status plus original-image masked correlation >=.65, finite positive affine determinant and in-bounds support. '
                      'No displacement truth, no correctness claim. Same 12px/4px grid and partial mask; at most four blocks of 30 iterations. '
                      'Merged archive preserves every old accepted state. No frontier propagation performed in this verification.'}
    (root/f'{tag}.json').write_text(json.dumps(result, indent=2))
    solver.close()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

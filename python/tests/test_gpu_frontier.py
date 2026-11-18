"""Integration check for the experimental GPU-resident IC-GN frontier."""

import numpy as np
import pytest

live_dic = pytest.importorskip("live_dic", reason="optional live-dic solver is not installed")

from wgpu_matching.gpu_frontier import GPUFrontier
from wgpu_matching.seed_initialization import initialize
from wgpu_matching.seed_propagation import propagate


def test_gpu_frontier_matches_cpu_frontier_with_real_solver():
    image = np.random.default_rng(4).integers(0, 256, (128, 128), dtype=np.uint8)
    solver = live_dic.ICGNWGPU(device_type="auto", prefer_subgroups=False, subset_size=16, subset_stride=16,
                              interpolation="linear", motion_model="affine", outlier_policy="disabled")
    try:
        solver.set_reference(image, iterations=30, tolerance=1e-3, zncc_min=.65)
        shape = (solver.grid_height, solver.grid_width)
        count = int(np.prod(shape))
        centres = np.array([((x + .5) * 16 - .5, (y + .5) * 16 - .5)
                            for y in range(shape[0]) for x in range(shape[1])])
        valid = np.ones(count, bool)
        seeds = np.zeros(count, bool)
        seeds[shape[1] * 3 + 3] = True
        initial = np.zeros((count, 6))

        def solve(indices, candidates):
            state = np.zeros((count, 8), np.float32)
            state[indices, :6] = candidates
            state[indices, 6] = 1
            active = np.zeros(count, np.uint32)
            active[indices] = 1
            result = solver._register_seeded_frame(image, state, active)
            return (np.asarray(result.parameters).reshape(-1, 6)[indices],
                    np.asarray(result.status).reshape(-1)[indices] > .5)

        cpu = propagate(centres, shape, valid, initial, seeds, solve)
        frontier = GPUFrontier(solver)
        gpu = frontier.propagate(image, valid, initial, seeds)
        for key in ("accepted", "parameters", "parent", "origin", "generation", "attempts", "rejection_reason"):
            np.testing.assert_array_equal(cpu[key], gpu[key])
        assert gpu["converged"]
        assert cpu["passes"] == gpu["passes"]
        isolated = np.zeros(count, bool)
        isolated[shape[1] * 3 + 3] = True
        empty_cpu = propagate(centres, shape, isolated, initial, seeds, solve)
        empty_gpu = frontier.propagate(image, isolated, initial, seeds)
        assert empty_cpu["passes"] == empty_gpu["passes"] == 0
        np.testing.assert_array_equal(empty_cpu["accepted"], empty_gpu["accepted"])
        with pytest.raises(ValueError, match="thresholds"):
            frontier.propagate(image, valid, initial, seeds, tau_u=float("nan"))
        with pytest.raises(ValueError, match="seed mask"):
            frontier.propagate(image, np.zeros(count, bool), initial, seeds)
        hypotheses = [("identity", initial)]
        cpu_init = initialize(centres, shape, valid, hypotheses, solve, sparse_stride=3,
                              batch_confirmation=True)

        def gpu_anchor(_centres, _shape, mask, parameters, candidates, _solve, **kwargs):
            selected_buffer = kwargs.pop("selected_buffer", None)
            return frontier.propagate(image, mask, parameters, candidates,
                                      confirm_candidates=True, selected_buffer=selected_buffer,
                                      target_uploaded=selected_buffer is not None, **kwargs)

        gpu_init = initialize(centres, shape, valid, hypotheses, solve, sparse_stride=3,
                              anchor_propagator=gpu_anchor)
        for key in ("confirmed_seeds", "accepted", "parameters", "parent", "origin", "generation", "attempts"):
            np.testing.assert_allclose(cpu_init[key], gpu_init[key], atol=1e-5)
        resident_init = initialize(centres, shape, valid, hypotheses, solve, sparse_stride=3,
                                   anchor_propagator=gpu_anchor,
                                   hypothesis_selector=lambda sparse, hs: frontier.select_hypotheses(
                                       image, sparse, hs, resident=True))
        assert cpu_init["hypothesis_acceptances"] == resident_init["hypothesis_acceptances"]
        for key in ("confirmed_seeds", "accepted", "parameters", "parent", "origin", "generation", "attempts"):
            np.testing.assert_allclose(cpu_init[key], resident_init[key], atol=1e-5)
        bad = initial.copy()
        bad[:, 0] = 40
        mask = np.arange(count) % 4 == 0
        hypotheses = [("bad", bad), ("masked", initial, mask), ("identity", initial), ("tie", initial)]
        cpu_init = initialize(centres, shape, valid, hypotheses, solve, sparse_stride=2,
                              batch_confirmation=True)
        gpu_init = initialize(centres, shape, valid, hypotheses, solve, sparse_stride=2,
                              anchor_propagator=gpu_anchor,
                              hypothesis_selector=lambda sparse, hs: frontier.select_hypotheses(image, sparse, hs))
        assert cpu_init["hypothesis_acceptances"] == gpu_init["hypothesis_acceptances"]
        assert cpu_init["selected_hypothesis"] == gpu_init["selected_hypothesis"] == "identity"
        for key in ("confirmed_seeds", "accepted", "parameters", "parent", "origin", "generation", "attempts"):
            np.testing.assert_allclose(cpu_init[key], gpu_init[key], atol=1e-5)
        resident_readbacks = []

        def select_resident(sparse, hs):
            selection = frontier.select_hypotheses(image, sparse, hs, resident=True)
            resident_readbacks.append(selection["readback_bytes"])
            return selection

        resident_init = initialize(centres, shape, valid, hypotheses, solve, sparse_stride=2,
                                   anchor_propagator=gpu_anchor,
                                   hypothesis_selector=select_resident)
        assert resident_readbacks == [4 * (len(hypotheses) + 1)]
        assert cpu_init["hypothesis_acceptances"] == resident_init["hypothesis_acceptances"]
        assert resident_init["selected_hypothesis"] == "identity"
        for key in ("confirmed_seeds", "accepted", "parameters", "parent", "origin", "generation", "attempts"):
            np.testing.assert_allclose(cpu_init[key], resident_init[key], atol=1e-5)
    finally:
        solver.close()

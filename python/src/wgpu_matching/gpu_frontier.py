"""Experimental WebGPU-resident affine frontier for an existing live-dic solver.

The adapter can rank sparse hypotheses, confirm anchors and propagate them on
the solver's device. In resident mode the selected field remains on the GPU;
only the small array of hypothesis counts is read back. Frontier
termination reads an eight-byte counter per pass, followed by final readback.
This adapter uses live-dic's private seeded-solver interface. Feature matching
and the complete pipeline are not GPU-resident.
"""

import struct
import time
from importlib.resources import files

import numpy as np
import wgpu


class GPUFrontier:
    """Keep one Jacobi affine-propagation field on the solver's WebGPU device."""

    def __init__(self, solver):
        if solver.motion_model != "affine" or solver.interpolation != "linear" or solver.outlier_policy != "disabled":
            raise ValueError("GPU frontier currently requires affine motion, linear interpolation and no outlier filter")
        if solver._partial_solver is not None:
            raise ValueError("GPU frontier does not support partial-subset solvers")
        self.solver = solver
        self.device = solver.device
        source = files("wgpu_matching").joinpath("shaders", "dic-frontier.wgsl").read_text(encoding="utf-8")
        module = self.device.create_shader_module(label="DIC affine frontier", code=source)
        types = ("read-only-storage", "storage", "storage", "storage", "read-only-storage",
                 "storage", "storage", "uniform", "read-only-storage")
        entries = [{"binding": index, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": kind}}
                   for index, kind in enumerate(types)]
        self.layout = self.device.create_bind_group_layout(entries=entries)
        layout = self.device.create_pipeline_layout(bind_group_layouts=[self.layout])
        self.pipelines = {name: self.device.create_compute_pipeline(
            label=f"DIC frontier {name}", layout=layout, compute={"module": module, "entry_point": name})
            for name in ("propose_first", "propose_second", "accept")}

    def select_hypotheses(self, target, sparse, hypotheses, *, resident=False):
        """Solve and rank sparse hypotheses on the solver device.

        In resident mode, the caller owns and must destroy the returned
        ``selected_buffer``. Hypotheses and target currently originate on CPU.
        """
        solver, device = self.solver, self.device
        shape = (solver.grid_height, solver.grid_width)
        count, number = shape[0] * shape[1], len(hypotheses)
        sparse = np.asarray(sparse, dtype=np.int64).reshape(-1)
        target = np.asarray(target)
        if not number or np.any((sparse < 0) | (sparse >= count)):
            raise ValueError("invalid sparse grid or empty hypotheses")
        if target.dtype != np.uint8 or target.shape != (solver.height, solver.width):
            raise ValueError("incompatible target image")
        history_bytes = number * count * 32
        history_limit = min(256 * 1024 * 1024, device.limits["max-storage-buffer-binding-size"],
                            device.limits["max-buffer-size"])
        if history_bytes > history_limit:
            raise ValueError("hypothesis history exceeds the bounded GPU storage limit")
        slot = solver._pipeline_slots[0]
        solver._ensure_seed_resources(slot)
        module = device.create_shader_module(code=files("wgpu_matching").joinpath(
            "shaders", "dic-hypothesis-select.wgsl").read_text(encoding="utf-8"))
        kinds = ("read-only-storage", "read-only-storage", "storage", "storage", "uniform", "storage")
        layout = device.create_bind_group_layout(entries=[
            {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": kind}}
            for i, kind in enumerate(kinds)])
        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[layout])
        pipelines = {name: device.create_compute_pipeline(layout=pipeline_layout,
                     compute={"module": module, "entry_point": name}) for name in ("store", "select_best", "gather")}
        usage = wgpu.BufferUsage
        buffers = []

        def create(size, flags):
            buf = device.create_buffer(size=size, usage=flags)
            buffers.append(buf)
            return buf

        try:
            state_size = count * 32
            history = create(history_bytes, usage.STORAGE)
            counts = create((number + 1) * 4, usage.STORAGE | usage.COPY_SRC)
            selected_buffer = create(state_size, usage.STORAGE | usage.COPY_SRC)
            solver._upload_frame(slot, np.ascontiguousarray(target))
            names, indices_by_hypothesis = [], []
            started = time.perf_counter()
            for index, hypothesis in enumerate(hypotheses):
                if len(hypothesis) not in (2, 3):
                    raise ValueError("hypothesis must contain a name, states and optional validity mask")
                name, parameters = hypothesis[:2]
                parameters = np.asarray(parameters, dtype=np.float32).reshape(count, 6)
                eligible = sparse if len(hypothesis) == 2 else sparse[np.asarray(hypothesis[2], bool).reshape(count)[sparse]]
                names.append(name)
                indices_by_hypothesis.append(eligible)
                seeds = np.zeros((count, 8), np.float32)
                seeds[eligible, :6] = parameters[eligible]
                seeds[eligible, 6] = 1
                active = np.zeros(count, np.uint32)
                active[eligible] = 1
                device.queue.write_buffer(slot.buffers["seed_state"], 0, seeds)
                device.queue.write_buffer(slot.buffers["active_mask"], 0, active)
                cfg = create(16, usage.UNIFORM | usage.COPY_DST)
                device.queue.write_buffer(cfg, 0, struct.pack("<4I", count, number, index, 0))
                items = (slot.buffers["state"], slot.buffers["active_mask"], history, counts, cfg, selected_buffer)
                group = device.create_bind_group(layout=layout, entries=[
                    {"binding": i, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
                    for i, buf in enumerate(items)])
                encoder = device.create_command_encoder(label="DIC sparse hypothesis")
                phase = encoder.begin_compute_pass()
                phase.set_pipeline(solver._seeded_solver)
                phase.set_bind_group(0, slot.seeded_bind_group)
                phase.dispatch_workgroups(shape[1], shape[0])
                phase.end()
                phase = encoder.begin_compute_pass()
                phase.set_pipeline(pipelines["store"])
                phase.set_bind_group(0, group)
                phase.dispatch_workgroups((count + 255) // 256)
                phase.end()
                device.queue.submit([encoder.finish()])
            encoder = device.create_command_encoder(label="DIC rank sparse hypotheses")
            for name in ("select_best", "gather"):
                phase = encoder.begin_compute_pass()
                phase.set_pipeline(pipelines[name])
                phase.set_bind_group(0, group)
                phase.dispatch_workgroups(1 if name == "select_best" else (count + 255) // 256)
                phase.end()
            device.queue.submit([encoder.finish()])
            scores = np.frombuffer(device.queue.read_buffer(counts), np.uint32).copy()
            chosen = int(scores[-1])
            indices = indices_by_hypothesis[chosen]
            trials = [{"name": name, "accepted": int(scores[i])} for i, name in enumerate(names)]
            if resident:
                buffers.remove(selected_buffer)
                return {"selected_index": chosen, "trials": trials, "selected_buffer": selected_buffer,
                        "readback_bytes": (number + 1) * 4, "solver_dispatches": number,
                        "wall_s": time.perf_counter() - started}
            chosen_states = np.frombuffer(device.queue.read_buffer(selected_buffer), np.float32).reshape(count, 8).copy()
            return {"selected_index": chosen, "trials": trials, "indices": indices,
                    "parameters": chosen_states[indices, :6], "status": chosen_states[indices, 6] > .5,
                    "readback_bytes": state_size + (number + 1) * 4,
                    "solver_dispatches": number, "wall_s": time.perf_counter() - started}
        finally:
            for buffer in buffers:
                buffer.destroy()

    def propagate(self, target, valid, parameters, seeds, *, tau_u=8., tau_g=.25, max_passes=None,
                  confirm_candidates=False, selected_buffer=None, target_uploaded=False):
        """Propagate accepted states; return arrays and transfer/synchronization metadata.

        The caller must have prepared the solver reference and supply the same
        8-bit target as used to confirm the seeds. No truth or visibility is read.
        """
        solver, device = self.solver, self.device
        shape = (solver.grid_height, solver.grid_width)
        count = shape[0] * shape[1]
        valid = np.asarray(valid, dtype=bool).reshape(-1)
        seeds = np.asarray(seeds, dtype=bool).reshape(-1)
        parameters = np.asarray(parameters, dtype=np.float32).reshape(-1, 6)
        target = np.asarray(target)
        if (len(valid) != count or len(seeds) != count or len(parameters) != count or np.any(seeds & ~valid)
                or target.dtype != np.uint8 or target.shape != (solver.height, solver.width)):
            raise ValueError("incompatible grid, seed mask or target image")
        if (not np.isfinite(parameters[seeds]).all() or not np.isfinite((tau_u, tau_g)).all()
                or tau_u < 0 or tau_g < 0):
            raise ValueError("seed states must be finite and thresholds nonnegative")
        if selected_buffer is not None and (not confirm_candidates or selected_buffer.size != count * 32):
            raise ValueError("selected GPU state requires anchor confirmation and a matching grid")
        if target_uploaded and selected_buffer is None:
            raise ValueError("the uploaded target may only be reused with resident selection")
        if solver.roi is not None and np.any(valid & ~np.asarray(solver.roi.selected).reshape(-1)):
            raise ValueError("valid windows must respect the solver region of interest")
        limit = count + 1 if max_passes is None else int(max_passes)
        if limit < 1:
            raise ValueError("max_passes must be positive")
        slot = solver._pipeline_slots[0]
        solver._ensure_seed_resources(slot)
        if solver._partial_solver is not None:
            raise ValueError("GPU frontier does not support partial-subset solvers")
        usage = wgpu.BufferUsage
        buffers = []

        def create(size, flags):
            buf = device.create_buffer(size=size, usage=flags)
            buffers.append(buf)
            return buf

        try:
            state_size = count * 8 * 4
            initial = np.zeros((count, 8), dtype=np.float32)
            initial[:, :6] = parameters
            initial[:, 6] = seeds.astype(np.float32)
            meta = np.zeros((count, 8), dtype=np.int32)
            meta[:, :3] = -1
            meta[seeds, 1] = np.flatnonzero(seeds)
            meta[seeds, 2] = 0
            if confirm_candidates:
                meta[seeds, 5] = 1
                meta[seeds, 7] = 1
            snapshot = create(state_size, usage.STORAGE | usage.COPY_DST)
            current = create(state_size, usage.STORAGE | usage.COPY_SRC | usage.COPY_DST)
            metadata = create(count * 32, usage.STORAGE | usage.COPY_SRC | usage.COPY_DST)
            counters = create(8, usage.STORAGE | usage.COPY_SRC | usage.COPY_DST)
            config = create(32, usage.UNIFORM | usage.COPY_DST)
            valid_buffer = create(count * 4, usage.STORAGE | usage.COPY_DST)
            if selected_buffer is None:
                device.queue.write_buffer(current, 0, initial)
            else:
                encoder = device.create_command_encoder(label="DIC resident selected state")
                encoder.copy_buffer_to_buffer(selected_buffer, 0, current, 0, state_size)
                device.queue.submit([encoder.finish()])
            device.queue.write_buffer(metadata, 0, meta)
            device.queue.write_buffer(valid_buffer, 0, valid.astype(np.uint32))
            if not target_uploaded:
                solver._upload_frame(slot, np.ascontiguousarray(target))
            items = (snapshot, current, slot.buffers["seed_state"], slot.buffers["active_mask"],
                     slot.buffers["state"], metadata, counters, config, valid_buffer)
            group = device.create_bind_group(layout=self.layout, entries=[
                {"binding": i, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
                for i, buf in enumerate(items)])
            active_total = 0
            converged = False
            completed_passes = 0
            started = time.perf_counter()
            anchor_dispatches = 0
            if confirm_candidates:
                source = files("wgpu_matching").joinpath("shaders", "dic-anchor-confirm.wgsl").read_text(encoding="utf-8")
                module = device.create_shader_module(label="DIC anchor confirmation", code=source)
                kinds = ("storage", "storage", "storage", "read-only-storage", "storage", "storage",
                         "read-only-storage", "uniform")
                entries = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": kind}}
                           for i, kind in enumerate(kinds)]
                anchor_layout = device.create_bind_group_layout(entries=entries)
                pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[anchor_layout])
                pipelines = {name: device.create_compute_pipeline(layout=pipeline_layout,
                             compute={"module": module, "entry_point": name})
                             for name in ("prepare_selected", "propose", "store", "replay")}
                history = create(count * 8 * 8 * 4, usage.STORAGE)
                encoder = device.create_command_encoder(label="DIC anchor confirmation")
                for direction in range(8):
                    cfg = create(32, usage.UNIFORM | usage.COPY_DST)
                    device.queue.write_buffer(cfg, 0, struct.pack("<4I2f2I", shape[1], shape[0],
                                               solver.subset_stride, 0, tau_u, tau_g, direction, 0))
                    anchor_items = (current, slot.buffers["seed_state"], slot.buffers["active_mask"],
                                    slot.buffers["state"], history, metadata, valid_buffer, cfg)
                    anchor_group = device.create_bind_group(layout=anchor_layout, entries=[
                        {"binding": i, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
                        for i, buf in enumerate(anchor_items)])
                    if direction == 0 and selected_buffer is not None:
                        phase = encoder.begin_compute_pass()
                        phase.set_pipeline(pipelines["prepare_selected"])
                        phase.set_bind_group(0, anchor_group)
                        phase.dispatch_workgroups((count + 255) // 256)
                        phase.end()
                    for name in ("propose", "solve", "store"):
                        phase = encoder.begin_compute_pass()
                        phase.set_pipeline(solver._seeded_solver if name == "solve" else pipelines[name])
                        phase.set_bind_group(0, slot.seeded_bind_group if name == "solve" else anchor_group)
                        if name == "solve":
                            phase.dispatch_workgroups(shape[1], shape[0])
                        else:
                            phase.dispatch_workgroups((count + 255) // 256)
                        phase.end()
                    anchor_dispatches += 1
                phase = encoder.begin_compute_pass()
                phase.set_pipeline(pipelines["replay"])
                phase.set_bind_group(0, anchor_group)
                phase.dispatch_workgroups(1)
                phase.end()
                device.queue.submit([encoder.finish()])
            for pass_number in range(1, limit + 1):
                device.queue.write_buffer(config, 0, struct.pack("<4I2f2I", shape[1], shape[0], solver.subset_stride,
                                                                 pass_number, tau_u, tau_g, 0, 0))
                encoder = device.create_command_encoder(label="DIC affine frontier pass")
                encoder.copy_buffer_to_buffer(current, 0, snapshot, 0, state_size)
                encoder.clear_buffer(counters)
                for entry in ("propose_first", "propose_second"):
                    phase = encoder.begin_compute_pass()
                    phase.set_pipeline(self.pipelines[entry])
                    phase.set_bind_group(0, group)
                    phase.dispatch_workgroups((count + 255) // 256)
                    phase.end()
                    phase = encoder.begin_compute_pass()
                    phase.set_pipeline(solver._seeded_solver)
                    phase.set_bind_group(0, slot.seeded_bind_group)
                    phase.dispatch_workgroups(shape[1], shape[0])
                    phase.end()
                    phase = encoder.begin_compute_pass()
                    phase.set_pipeline(self.pipelines["accept"])
                    phase.set_bind_group(0, group)
                    phase.dispatch_workgroups((count + 255) // 256)
                    phase.end()
                device.queue.submit([encoder.finish()])
                counts = np.frombuffer(device.queue.read_buffer(counters, 0, 8), dtype=np.uint32)
                active_total += int(counts[1])
                if counts[1] > 0:
                    completed_passes += 1
                if counts[0] == 0:
                    converged = True
                    break
            result = np.frombuffer(device.queue.read_buffer(current, 0, state_size), dtype=np.float32).reshape(count, 8).copy()
            provenance = np.frombuffer(device.queue.read_buffer(metadata, 0, count * 32), dtype=np.int32).reshape(count, 8).copy()
            return {"parameters": result[:, :6], "accepted": result[:, 6] >= .5,
                    "parent": provenance[:, 0], "origin": provenance[:, 1], "generation": provenance[:, 2],
                    "attempts": provenance[:, 3], "rejection_reason": provenance[:, 4],
                    "confirmed_seeds": provenance[:, 6].astype(bool) if confirm_candidates else seeds.copy(),
                    "passes": completed_passes, "active_windows": active_total, "converged": converged,
                    "solver_calls": 2 * pass_number + anchor_dispatches,
                    "anchor_solver_dispatches": anchor_dispatches,
                    "counter_readbacks": pass_number, "counter_readback_bytes": 8 * pass_number,
                    "final_readback_bytes": state_size + count * 32,
                    "wall_s": time.perf_counter() - started}
        finally:
            for buffer in buffers:
                buffer.destroy()

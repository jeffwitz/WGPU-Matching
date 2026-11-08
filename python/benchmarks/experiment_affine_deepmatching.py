"""Image-only affine preconditioning of native DeepMatching; analytic truth is evaluation only."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import wgpu
from wgpu_matching import DeepMatchingOptions, WGPUMatching, load_gray
from wgpu_matching._gpu import (
    ResourceScope,
    dispatch,
    make_bind_group,
    make_pipeline,
    shader_source,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/uniaxial/tensile_60pct"


def warp_gpu(device, image, affine):
    h, w = image.shape
    scope = ResourceScope()
    try:
        src = scope.keep(
            device.create_buffer_with_data(
                data=np.ascontiguousarray(image, dtype=np.uint32), usage=wgpu.BufferUsage.STORAGE
            )
        )
        dst = scope.keep(
            device.create_buffer(
                size=w * h * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        matrix = np.zeros(8, dtype=np.float32)
        matrix[:6] = np.asarray(affine, dtype=np.float32).reshape(-1)
        params = np.zeros(12, dtype=np.uint32)
        params[:2] = [w, h]
        params[4:12].view(np.float32)[:6] = matrix[:6]
        uniform = scope.keep(
            device.create_buffer_with_data(data=params, usage=wgpu.BufferUsage.UNIFORM)
        )
        module = device.create_shader_module(code=shader_source("affine-warp.wgsl"))
        pipeline = make_pipeline(device, module, "warp", "Affine image warp")
        group = make_bind_group(
            device,
            pipeline,
            [
                {"binding": 0, "resource": {"buffer": src}},
                {"binding": 1, "resource": {"buffer": dst}},
                {"binding": 2, "resource": {"buffer": uniform}},
            ],
        )
        encoder = device.create_command_encoder()
        dispatch(encoder, pipeline, group, (w + 15) // 16, (h + 15) // 16)
        device.queue.submit([encoder.finish()])
        result = np.frombuffer(device.queue.read_buffer(dst), dtype=np.uint32).copy().reshape(h, w)
        return result.astype(np.uint8)
    finally:
        scope.close()


def gpu_ransac_affine(device, source, destination, threshold=4.0, hypotheses=10_000, seed=42):
    """Affine RANSAC sampling/scoring and consensus refit on WebGPU."""
    if len(source) < 3:
        raise RuntimeError("At least three matches are required for affine RANSAC")
    scope = ResourceScope()
    try:
        pairs_data = np.column_stack([source, destination]).astype(np.float32)
        pairs = scope.keep(
            device.create_buffer_with_data(
                data=np.ascontiguousarray(pairs_data), usage=wgpu.BufferUsage.STORAGE
            )
        )
        output = scope.keep(
            device.create_buffer(
                size=hypotheses * 2 * 16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        params = np.zeros(8, dtype=np.uint32)
        params[:4] = [len(source), hypotheses, int(seed) & 0xFFFFFFFF, 0]
        params[4:8].view(np.float32)[:] = [threshold, 0, 0, 0]
        uniform = scope.keep(
            device.create_buffer_with_data(data=params, usage=wgpu.BufferUsage.UNIFORM)
        )
        module = device.create_shader_module(code=shader_source("ransac-affine.wgsl"))
        pipeline = make_pipeline(
            device, module, "hypotheses", "Affine RANSAC hypotheses and consensus"
        )
        group = make_bind_group(
            device,
            pipeline,
            [
                {"binding": 0, "resource": {"buffer": pairs}},
                {"binding": 1, "resource": {"buffer": output}},
                {"binding": 2, "resource": {"buffer": uniform}},
            ],
        )
        best_output = scope.keep(
            device.create_buffer(
                size=48, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        best_params = scope.keep(
            device.create_buffer_with_data(
                data=np.array([hypotheses, 0, 0, 0], dtype=np.uint32),
                usage=wgpu.BufferUsage.UNIFORM,
            )
        )
        best_module = device.create_shader_module(code=shader_source("ransac-affine-best.wgsl"))
        best_pipeline = make_pipeline(
            device, best_module, "select_best", "Affine RANSAC GPU model selection"
        )
        best_group = make_bind_group(
            device,
            best_pipeline,
            [
                {"binding": 0, "resource": {"buffer": output}},
                {"binding": 1, "resource": {"buffer": best_output}},
                {"binding": 2, "resource": {"buffer": best_params}},
            ],
        )
        encoder = device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups((hypotheses + 63) // 64)
        compute.set_pipeline(best_pipeline)
        compute.set_bind_group(0, best_group)
        compute.dispatch_workgroups(1)
        compute.end()
        device.queue.submit([encoder.finish()])
        best = (
            np.frombuffer(device.queue.read_buffer(best_output), dtype=np.float32)
            .copy()
            .reshape(3, 4)
        )
        winner = round(float(best[2, 0]))
        if best[2, 1] < 3:
            raise RuntimeError("GPU RANSAC found no affine consensus")

        # Refit the winning consensus on GPU to avoid moving matches back for a host-side fit.
        refit_params = np.zeros(16, dtype=np.uint32)
        refit_params[:4] = [len(source), 0, 0, 0]
        refit_params[4:8].view(np.float32)[:] = best[0]
        refit_params[8:12].view(np.float32)[:] = [best[0, 3], best[1, 0], best[1, 1], 0.0]
        refit_params[12:16].view(np.float32)[:] = [threshold, 0, 0, 0]
        refit_uniform = scope.keep(
            device.create_buffer_with_data(data=refit_params, usage=wgpu.BufferUsage.UNIFORM)
        )
        refit_output = scope.keep(
            device.create_buffer(
                size=32, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        refit_module = device.create_shader_module(code=shader_source("ransac-affine-refit.wgsl"))
        refit_pipeline = make_pipeline(
            device, refit_module, "refit", "Affine RANSAC GPU consensus refit"
        )
        refit_group = make_bind_group(
            device,
            refit_pipeline,
            [
                {"binding": 0, "resource": {"buffer": pairs}},
                {"binding": 1, "resource": {"buffer": refit_output}},
                {"binding": 2, "resource": {"buffer": refit_uniform}},
            ],
        )
        encoder = device.create_command_encoder()
        dispatch(encoder, refit_pipeline, refit_group, 1)
        device.queue.submit([encoder.finish()])
        fitted = (
            np.frombuffer(device.queue.read_buffer(refit_output), dtype=np.float32)
            .copy()
            .reshape(2, 4)
        )
        affine = np.array(
            [
                [fitted[0, 0], fitted[0, 1], fitted[0, 2]],
                [fitted[1, 0], fitted[1, 1], fitted[1, 2]],
            ],
            dtype=np.float32,
        )
        return affine, round(float(fitted[0, 3])), winner
    finally:
        scope.close()


def gpu_ecc_affine(device, reference, target, affine, iterations=30):
    """Maximize photometric correlation by affine warp plus gain/bias on WebGPU."""
    h, w = reference.shape
    x0, y0, x1, y1 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    a = np.asarray(affine, dtype=np.float64).copy()
    # Center translations to keep the six geometric parameters well-conditioned.
    transform = np.array(
        [
            a[0, 0],
            a[0, 1],
            a[0, 0] * cx + a[0, 1] * cy + a[0, 2],
            a[1, 0],
            a[1, 1],
            a[1, 0] * cx + a[1, 1] * cy + a[1, 2],
            1.0,
            0.0,
        ],
        dtype=np.float64,
    )
    tiles_x = (x1 - x0 + 7) // 8
    tiles_y = (y1 - y0 + 7) // 8
    tile_count = tiles_x * tiles_y
    scope = ResourceScope()
    try:
        ref = scope.keep(
            device.create_buffer_with_data(
                data=np.ascontiguousarray(reference, dtype=np.uint32),
                usage=wgpu.BufferUsage.STORAGE,
            )
        )
        tgt = scope.keep(
            device.create_buffer_with_data(
                data=np.ascontiguousarray(target, dtype=np.uint32), usage=wgpu.BufferUsage.STORAGE
            )
        )
        partial = scope.keep(
            device.create_buffer(size=tile_count * 44 * 4, usage=wgpu.BufferUsage.STORAGE)
        )
        result = scope.keep(
            device.create_buffer(
                size=44 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        params = np.zeros(20, dtype=np.uint32)
        params[:8] = [w, h, 0, 0, x0, y0, x1, y1]
        uniform = scope.keep(
            device.create_buffer_with_data(
                data=params, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
            )
        )
        module = device.create_shader_module(code=shader_source("ecc-affine-reduce.wgsl"))
        accumulate = make_pipeline(device, module, "accumulate", "ECC affine GPU normal equations")
        group = make_bind_group(
            device,
            accumulate,
            [
                {"binding": 0, "resource": {"buffer": ref}},
                {"binding": 1, "resource": {"buffer": tgt}},
                {"binding": 2, "resource": {"buffer": partial}},
                {"binding": 3, "resource": {"buffer": uniform}},
            ],
        )
        finish_module = device.create_shader_module(code=shader_source("ecc-affine-finish.wgsl"))
        finish = make_pipeline(device, finish_module, "finish", "ECC affine GPU reduction")
        finish_params = scope.keep(
            device.create_buffer_with_data(
                data=np.array([tile_count, 0, 0, 0], dtype=np.uint32),
                usage=wgpu.BufferUsage.UNIFORM,
            )
        )
        finish_group = make_bind_group(
            device,
            finish,
            [
                {"binding": 0, "resource": {"buffer": partial}},
                {"binding": 1, "resource": {"buffer": result}},
                {"binding": 2, "resource": {"buffer": finish_params}},
            ],
        )
        delta_buffer = scope.keep(
            device.create_buffer(
                size=8 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        solve_module = device.create_shader_module(code=shader_source("ecc-affine-solve.wgsl"))
        solve = make_pipeline(device, solve_module, "solve", "ECC affine GPU normal equation solve")
        solve_group = make_bind_group(
            device,
            solve,
            [
                {"binding": 0, "resource": {"buffer": result}},
                {"binding": 1, "resource": {"buffer": delta_buffer}},
            ],
        )
        used_iterations = 0
        for iteration in range(iterations):
            used_iterations = iteration + 1
            payload = np.zeros(20, dtype=np.uint32)
            payload[:8] = [w, h, 0, 0, x0, y0, x1, y1]
            payload[8:12].view(np.float32)[:] = transform[:4]
            payload[12:16].view(np.float32)[:] = [transform[4], transform[5], 0.0, 0.0]
            payload[16:20].view(np.float32)[:] = [transform[6], transform[7], 0.0, 0.0]
            device.queue.write_buffer(uniform, 0, payload)
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(accumulate)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups(tiles_x, tiles_y)
            compute.set_pipeline(finish)
            compute.set_bind_group(0, finish_group)
            compute.dispatch_workgroups(1)
            compute.set_pipeline(solve)
            compute.set_bind_group(0, solve_group)
            compute.dispatch_workgroups(1)
            compute.end()
            device.queue.submit([encoder.finish()])
            delta = np.frombuffer(device.queue.read_buffer(delta_buffer), dtype=np.float32).astype(
                np.float64
            )
            transform += delta
            if not np.all(np.isfinite(transform)):
                raise RuntimeError("GPU ECC produced a non-finite affine update")
            if (
                np.linalg.norm(delta[:2]) < 1e-6
                and np.linalg.norm(delta[3:5]) < 1e-6
                and np.linalg.norm(delta[[2, 5]]) < 1e-2
                and np.linalg.norm(delta[6:]) < 1e-3
            ):
                break
        output = np.array(
            [
                [transform[0], transform[1], transform[2] - transform[0] * cx - transform[1] * cy],
                [transform[3], transform[4], transform[5] - transform[3] * cx - transform[4] * cy],
            ],
            dtype=np.float32,
        )
        return output, used_iterations
    finally:
        scope.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecc", action="store_true", help="refine the affine warp by GPU ECC")
    parser.add_argument("--seed", type=int, default=42, help="deterministic GPU RANSAC seed")
    args = parser.parse_args()
    reference = load_gray(DATA / "reference.png")
    manifest = json.loads((DATA / "manifest.json").read_text())
    results = []
    with WGPUMatching() as matcher:
        for snapshot in manifest["snapshots"]:
            target = load_gray(DATA / snapshot["file"])
            started = time.perf_counter()
            phase = time.perf_counter()
            dense = matcher.match_deepmatching(
                reference, target, options=DeepMatchingOptions(stride=128)
            )
            dense_seconds = time.perf_counter() - phase
            matches = dense.accepted
            source = np.array([(m.x, m.y) for m in matches], np.float32)
            destination = np.array([(m.x + m.dx, m.y + m.dy) for m in matches], np.float32)
            phase = time.perf_counter()
            affine, inlier_count, ransac_hypothesis = gpu_ransac_affine(
                matcher.device, source, destination, seed=args.seed
            )
            ransac_seconds = time.perf_counter() - phase
            phase = time.perf_counter()
            aligned = warp_gpu(matcher.device, target, affine)
            warp_seconds = time.perf_counter() - phase
            ecc_seconds = 0.0
            ecc_iterations = 0
            if args.ecc:
                phase = time.perf_counter()
                affine, ecc_iterations = gpu_ecc_affine(matcher.device, reference, target, affine)
                ecc_seconds = time.perf_counter() - phase
                phase = time.perf_counter()
                aligned = warp_gpu(matcher.device, target, affine)
                warp_seconds += time.perf_counter() - phase
            # All grid queries are submitted; ground truth only selects the evaluation population.
            phase = time.perf_counter()
            refined = matcher.match_deepmatching(reference, aligned, options=DeepMatchingOptions())
            refine_seconds = time.perf_counter() - phase
            elapsed = time.perf_counter() - started
            errors = []
            visible = 0
            lx, ly = snapshot["stretch_x"], snapshot["stretch_y"]
            for m in refined.matches:
                sx, sy = m.seed_x, m.seed_y
                if not (
                    0 <= 1920 + lx * (sx - 1920) < 3840 and 0 <= 1080 + ly * (sy - 1080) < 2160
                ):
                    continue
                visible += 1
                if m.usable:
                    predicted = affine @ np.array([m.x + m.dx, m.y + m.dy, 1.0])
                    truth = np.array([1920 + lx * (m.x - 1920), 1080 + ly * (m.y - 1080)])
                    errors.append(float(np.linalg.norm(predicted - truth)))
            row = {
                "strain": snapshot["actual_engineering_strain"],
                "visible": visible,
                "accepted": len(errors),
                "within_2px": sum(e <= 2 for e in errors),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                "seconds": elapsed,
                "dense_matches": len(matches),
                "affine_inliers": inlier_count,
                "dense_seconds": dense_seconds,
                "ransac_seconds": ransac_seconds,
                "ecc_seconds": ecc_seconds,
                "warp_seconds": warp_seconds,
                "ecc_iterations": ecc_iterations,
                "refine_seconds": refine_seconds,
                "ransac_hypothesis": ransac_hypothesis,
                "affine": affine.tolist(),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    name = (
        "affine_ecc_deepmatching_experiment.json"
        if args.ecc
        else "affine_deepmatching_experiment.json"
    )
    (DATA / name).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

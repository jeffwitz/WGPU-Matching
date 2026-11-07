"""Project ONNX matcher correspondences onto affine/RBF fields on DIC datasets.

This is a diagnostic experiment: ONNX Runtime extracts direct matches, then
the package's wgpu-py implementation fits affine and cross-validated Gaussian
RBF fields and initializes patch-sized affine IC-GN parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import wgpu
from PIL import Image
from wgpu_matching import (
    CorrespondenceSet,
    ProjectionOptions,
    WGPUMatching,
)

ROOT = Path(__file__).resolve().parents[2]
STRESS_DIR = ROOT / "results/uniaxial/tensile_60pct"
ADRIEN_DIR = Path("/home/jeff/CNRS/DIC/AdrienVariational")
DEFAULT_RACO = Path("/tmp/wgpu-matching-aliked-study/raco_aliked_lightglue_pipeline_k1024.onnx")
DEFAULT_LOFTR = Path("/tmp/wgpu-matching-aliked-study/eloftr_outdoor_opt.onnx")
NETWORK_WIDTH = 640
PATCH_SIZE = 128
SEED_STRIDE = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tensor(
    path: Path, content_size: tuple[int, int], channels: int
) -> tuple[np.ndarray, tuple[int, int]]:
    width, height = content_size
    image = Image.open(path).convert("RGB" if channels == 3 else "L")
    resized = np.asarray(image.resize((width, height)), dtype=np.float32) / 255.0
    padded_width = ((width + 31) // 32) * 32
    padded_height = ((height + 31) // 32) * 32
    if channels == 3:
        tensor = np.pad(
            resized,
            ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        )
    else:
        tensor = np.pad(resized, ((0, padded_height - height), (0, padded_width - width)))
    return np.ascontiguousarray(tensor, dtype=np.float32), (padded_width, padded_height)


def read_truth(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["frame"]): {key: float(value) for key, value in row.items() if key != "frame"}
            for row in csv.DictReader(stream)
        }


def visible_tensile_seeds(
    width: int, height: int, stretch_x: float, stretch_y: float
) -> np.ndarray:
    offset = (PATCH_SIZE - 1) / 2
    points = []
    center_x, center_y = width / 2.0, height / 2.0
    for row in range(1 + (height - PATCH_SIZE) // SEED_STRIDE):
        for column in range(1 + (width - PATCH_SIZE) // SEED_STRIDE):
            x, y = column * SEED_STRIDE + offset, row * SEED_STRIDE + offset
            tx = center_x + stretch_x * (x - center_x)
            ty = center_y + stretch_y * (y - center_y)
            if 0 <= tx < width and 0 <= ty < height:
                points.append((x, y))
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def roi_seeds(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    valid = (mask > 0).astype(np.int64)
    integral = np.pad(valid.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    offset = (PATCH_SIZE - 1) / 2
    points = []
    for y0 in range(0, height - PATCH_SIZE + 1, SEED_STRIDE):
        for x0 in range(0, width - PATCH_SIZE + 1, SEED_STRIDE):
            total = (
                integral[y0 + PATCH_SIZE, x0 + PATCH_SIZE]
                - integral[y0, x0 + PATCH_SIZE]
                - integral[y0 + PATCH_SIZE, x0]
                + integral[y0, x0]
            )
            if total >= 0.95 * PATCH_SIZE * PATCH_SIZE:
                points.append((x0 + offset, y0 + offset))
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def mask_points(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    px = np.rint(points[:, 0]).astype(np.int64)
    py = np.rint(points[:, 1]).astype(np.int64)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    result = np.zeros(len(points), dtype=bool)
    result[inside] = mask[py[inside], px[inside]] > 0
    return result


def create_webgpu_sessions(raco_path: Path, loftr_path: Path, intra_op_threads: int):
    import onnxruntime as ort
    import onnxruntime_ep_webgpu as webgpu_ep

    ort.register_execution_provider_library(
        "wgpu_matching_projection", webgpu_ep.get_library_path()
    )
    devices = [device for device in ort.get_ep_devices() if device.ep_name == webgpu_ep.get_ep_name()]
    if not devices:
        raise RuntimeError("ONNX Runtime found no WebGPU execution device")

    sessions = {}
    for name, path in (("raco_aliked_lightglue", raco_path), ("efficientloftr", loftr_path)):
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.add_provider_for_devices(devices, {"preferredLayout": "NHWC"})
        sessions[name] = ort.InferenceSession(str(path), sess_options=options, providers=None)
    return sessions, devices


def infer_pairs(
    name: str,
    session: Any,
    reference: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if name == "efficientloftr":
        names = {item.name for item in session.get_inputs()}
        if names != {"image0", "image1"}:
            raise ValueError(f"EfficientLoFTR inputs differ from image0/image1: {sorted(names)}")
        source, destination, confidence = session.run(
            None,
            {
                "image0": reference[None, None],
                "image1": target[None, None],
            },
        )
        return (
            np.asarray(source, dtype=np.float64).reshape(-1, 2),
            np.asarray(destination, dtype=np.float64).reshape(-1, 2),
            np.asarray(confidence, dtype=np.float64).reshape(-1),
        )

    if name != "raco_aliked_lightglue":
        raise ValueError(f"unsupported model: {name}")
    input_name = session.get_inputs()[0].name
    pair_tensor = np.stack((reference, target)).transpose(0, 3, 1, 2)
    keypoints, matches, scores = session.run(None, {input_name: pair_tensor})
    edge_rows = np.flatnonzero(matches[:, 0] == 0)
    selected = matches[edge_rows]
    source = np.asarray(keypoints[0, selected[:, 1].astype(np.int64)], dtype=np.float64)
    destination = np.asarray(keypoints[1, selected[:, 2].astype(np.int64)], dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    score_values = score_array[edge_rows] if len(score_array) == len(matches) else None
    return source.reshape(-1, 2), destination.reshape(-1, 2), score_values


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median_px": None, "p95_px": None, "rmse_px": None, "max_px": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_px": float(np.median(array)),
        "p95_px": float(np.percentile(array, 95)),
        "rmse_px": float(np.sqrt(np.mean(array * array))),
        "max_px": float(np.max(array)),
    }


def render_projected_pair(
    model: Any,
    case: dict[str, Any],
    output_dir: Path,
    sequence_dir: Path,
    sequence_index: int,
    model_name: str,
    basis: str,
) -> tuple[list[Path], int]:
    """Warp the target by the fitted forward field for quick EOG inspection."""
    native_width, native_height = case["native_size"]
    with Image.open(case["reference"]) as image:
        reference = image.convert("L")
    with Image.open(case["target"]) as image:
        target = np.asarray(image.convert("L"), dtype=np.uint8)
    if reference.size != (native_width, native_height) or target.shape != (
        native_height,
        native_width,
    ):
        raise ValueError(f"Image dimensions changed since case setup: {case['name']}")

    output_width = min(native_width, 1920)
    output_height = round(native_height * output_width / native_width)
    preview_reference = reference.resize(
        (output_width, output_height), Image.Resampling.BILINEAR
    )
    corrected = np.zeros((output_height, output_width), dtype=np.uint8)
    x_out = np.arange(output_width, dtype=np.float32)
    output_x_scale = native_width / output_width
    output_y_scale = native_height / output_height
    chunk_rows = 256

    for y0 in range(0, output_height, chunk_rows):
        y1 = min(output_height, y0 + chunk_rows)
        yy, xx = np.meshgrid(np.arange(y0, y1, dtype=np.float32), x_out, indexing="ij")
        native_x = (xx + 0.5) * output_x_scale - 0.5
        native_y = (yy + 0.5) * output_y_scale - 0.5
        points = np.column_stack((native_x.ravel(), native_y.ravel()))
        displacement = model.evaluate(points)
        sample_x = native_x + displacement[:, 0].reshape(native_x.shape)
        sample_y = native_y + displacement[:, 1].reshape(native_y.shape)

        inside = (
            (sample_x >= 0.0)
            & (sample_x <= native_width - 1)
            & (sample_y >= 0.0)
            & (sample_y <= native_height - 1)
        )
        x0 = np.floor(np.clip(sample_x, 0, native_width - 1)).astype(np.int32)
        y0s = np.floor(np.clip(sample_y, 0, native_height - 1)).astype(np.int32)
        x1s = np.minimum(x0 + 1, native_width - 1)
        y1s = np.minimum(y0s + 1, native_height - 1)
        dx = sample_x - x0
        dy = sample_y - y0s
        values = (
            target[y0s, x0] * (1.0 - dx) * (1.0 - dy)
            + target[y0s, x1s] * dx * (1.0 - dy)
            + target[y1s, x0] * (1.0 - dx) * dy
            + target[y1s, x1s] * dx * dy
        )
        values[~inside] = 0
        corrected[y0:y1] = np.clip(np.rint(values), 0, 255).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "image1_ref.png"
    corrected_path = output_dir / "image1_def_cor.png"
    preview_reference.save(reference_path, format="PNG", compress_level=2)
    Image.fromarray(corrected, mode="L").save(corrected_path, format="PNG", compress_level=2)

    linked_paths = []
    for offset, source_path in enumerate((reference_path, corrected_path)):
        sequence_name = (
            f"{sequence_index + offset:03d}_{case['name']}_{model_name}_{basis}_"
            f"{source_path.name}"
        )
        destination = sequence_dir / sequence_name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        try:
            destination.hardlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, destination)
        linked_paths.append(destination)
    return [reference_path, corrected_path, *linked_paths], sequence_index + 2


def analytic_target(source: np.ndarray, truth: dict[str, float], width: int, height: int) -> np.ndarray:
    return np.column_stack(
        (
            width / 2.0 + truth["stretch_x"] * (source[:, 0] - width / 2.0),
            height / 2.0 + truth["stretch_y"] * (source[:, 1] - height / 2.0),
        )
    )


def build_cases(stress_dir: Path, adrien_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    manifest = json.loads((stress_dir / "manifest.json").read_text(encoding="utf-8"))
    truth = read_truth(stress_dir / manifest["truth_file"])
    cases = []
    for snapshot in manifest["snapshots"]:
        frame = int(snapshot["frame"])
        state = truth[frame]
        cases.append(
            {
                "name": f"tensile_{round(100 * state['engineering_axial']):02d}pct",
                "kind": "analytic_tensile",
                "reference": stress_dir / "reference.png",
                "target": stress_dir / snapshot["file"],
                "native_size": (int(manifest["width"]), int(manifest["height"])),
                "content_size": (640, 360),
                "truth": state,
                "frame": frame,
                "mask": None,
            }
        )

    reference = adrien_dir / "img_000000_0.09422.tiff"
    target = adrien_dir / "img_000075_1608.88508.tiff"
    mask_path = adrien_dir / "Mask.tiff"
    with Image.open(reference) as image:
        native_size = image.size
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if mask.shape != (native_size[1], native_size[0]):
        raise ValueError("Adrien ROI mask dimensions do not match its images")
    cases.append(
        {
            "name": "adrien_000000_to_000075",
            "kind": "experimental_no_analytic_truth",
            "reference": reference,
            "target": target,
            "native_size": native_size,
            "content_size": (640, 485),
            "truth": None,
            "frame": None,
            "mask": mask,
            "roi_seed_points": roi_seeds(mask),
        }
    )
    return cases, mask


def project_case(
    matcher: WGPUMatching,
    name: str,
    case: dict[str, Any],
    pairs: CorrespondenceSet,
    seeds: np.ndarray,
    output_dir: Path,
    sequence_dir: Path,
    sequence_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    summary_rows = []
    patch_rows = []
    width, height = case["native_size"]
    truth = case["truth"]
    if truth is not None and len(pairs.source_xy):
        measured_error = np.linalg.norm(
            pairs.target_xy - analytic_target(pairs.source_xy, truth, width, height), axis=1
        )
        raw_quality = {
            "direct_median_px": float(np.median(measured_error)),
            "direct_rmse_px": float(np.sqrt(np.mean(measured_error**2))),
            "direct_le2_count": int(np.count_nonzero(measured_error <= 2.0)),
            "direct_lt5_count": int(np.count_nonzero(measured_error < 5.0)),
        }
    else:
        raw_quality = {}

    for basis in ("affine", "gaussian_rbf"):
        model_start = time.perf_counter()
        try:
            model = matcher.fit_initialization_field(
                pairs,
                case["native_size"],
                options=ProjectionOptions(basis=basis),
            )
        except (RuntimeError, ValueError) as error:
            summary_rows.append(
                {
                    "dataset": case["name"],
                    "model": name,
                    "basis_requested": basis,
                    "status": "fit_failed",
                    "error": str(error),
                    "direct_pair_input_count": len(pairs.source_xy),
                    "seed_count": len(seeds),
                }
            )
            continue
        fit_ms = (time.perf_counter() - model_start) * 1000.0
        patch_start = time.perf_counter()
        patches = model.initialize_patches(
            seeds,
            patch_size=PATCH_SIZE,
            projection="support_fit",
            seed_ids=tuple(range(len(seeds))),
        )
        patch_projection_ms = (time.perf_counter() - patch_start) * 1000.0

        errors = []
        supports = []
        for seed, patch in zip(seeds, patches, strict=True):
            truth_error = None
            if truth is not None and patch.parameters is not None:
                expected = analytic_target(seed[None, :], truth, width, height)[0] - seed
                truth_error = float(np.linalg.norm(np.asarray(patch.parameters[:2]) - expected))
                errors.append(truth_error)
            if patch.support_rms_px is not None:
                supports.append(patch.support_rms_px)
            patch_rows.append(
                {
                    "dataset": case["name"],
                    "model": name,
                    "basis_requested": basis,
                    "basis_effective": model.effective_basis,
                    "seed_id": patch.seed_id,
                    "x_px": patch.x,
                    "y_px": patch.y,
                    "status": patch.status,
                    "u_px": None if patch.parameters is None else patch.parameters[0],
                    "v_px": None if patch.parameters is None else patch.parameters[1],
                    "du_dx": None if patch.parameters is None else patch.parameters[2],
                    "du_dy": None if patch.parameters is None else patch.parameters[3],
                    "dv_dx": None if patch.parameters is None else patch.parameters[4],
                    "dv_dy": None if patch.parameters is None else patch.parameters[5],
                    "analytic_error_px": truth_error,
                    "support_rms_px": patch.support_rms_px,
                    "support_max_px": patch.support_max_px,
                }
            )

        image_paths, sequence_index = render_projected_pair(
            model,
            case,
            output_dir / "visualizations" / case["name"] / name / basis,
            sequence_dir,
            sequence_index,
            name,
            basis,
        )
        diagnostics = model.diagnostics
        summary_rows.append(
            {
                "dataset": case["name"],
                "kind": case["kind"],
                "frame": case["frame"],
                "model": name,
                "basis_requested": basis,
                "basis_effective": model.effective_basis,
                "selection_reason": diagnostics["selection_reason"],
                "status": "ok",
                "input_pairs": diagnostics["input_count"],
                "retained_pairs": diagnostics["retained_count"],
                "seed_count": len(seeds),
                "projected_seed_count": sum(patch.parameters is not None for patch in patches),
                "selected_center_count": diagnostics["selected_center_count"],
                "selected_ell_normalized": diagnostics["selected_ell_normalized"],
                "selected_ridge": diagnostics["selected_ridge"],
                "irls_converged": diagnostics["irls_converged"],
                "irls_final_max_weight_delta": diagnostics["irls_final_max_weight_delta"],
                "fit_wall_ms": fit_ms,
                "cpu_selection_and_scale_pilot_ms": diagnostics["cpu_selection_and_scale_pilot_ms"],
                "gpu_final_fit_wall_ms": diagnostics["gpu_final_fit_wall_ms"],
                "gpu_patch_projection_wall_ms": diagnostics["last_field_evaluation_wall_ms"],
                "patch_api_wall_ms": patch_projection_ms,
                "field_residual_median_px": float(np.median(model.residuals_px)),
                "field_residual_p95_px": float(np.percentile(model.residuals_px, 95)),
                "seed_error_median_px": float(np.median(errors)) if errors else None,
                "seed_error_rmse_px": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                "seed_error_p95_px": float(np.percentile(errors, 95)) if errors else None,
                "seed_error_le2_count": sum(error <= 2.0 for error in errors) if errors else None,
                "seed_error_lt5_count": sum(error < 5.0 for error in errors) if errors else None,
                "seed_error_total_count": len(errors) if truth is not None else None,
                "seed_support_rms_median_px": float(np.median(supports)) if supports else None,
                "seed_support_rms_p95_px": float(np.percentile(supports, 95)) if supports else None,
                "candidate_scores": diagnostics["candidate_scores"],
                "visualization_files": [str(path.relative_to(output_dir)) for path in image_paths[:2]],
                **raw_quality,
            }
        )
    return summary_rows, patch_rows, sequence_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raco-model", type=Path, default=DEFAULT_RACO)
    parser.add_argument("--loftr-model", type=Path, default=DEFAULT_LOFTR)
    parser.add_argument("--stress-dir", type=Path, default=STRESS_DIR)
    parser.add_argument("--adrien-dir", type=Path, default=ADRIEN_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--intra-op-threads", type=int, default=8)
    parser.add_argument("--skip-adrien", action="store_true")
    parser.add_argument("--only-adrien", action="store_true")
    args = parser.parse_args()
    for path in (args.raco_model, args.loftr_model):
        if not path.is_file():
            parser.error(f"ONNX model does not exist: {path}")
    if args.intra_op_threads < 1:
        parser.error("--intra-op-threads must be positive")
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROOT / "results/projection" / f"dataset_run_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases, _ = build_cases(args.stress_dir, args.adrien_dir)
    if args.skip_adrien:
        cases = [case for case in cases if case["truth"] is not None]
    if args.only_adrien:
        cases = [case for case in cases if case["truth"] is None]
    sessions, ort_devices = create_webgpu_sessions(
        args.raco_model, args.loftr_model, args.intra_op_threads
    )
    summary_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    projection_adapter = None
    sequence_dir = output_dir / "eog_sequence"
    sequence_index = 1
    raw_path = output_dir / "raw_correspondences.csv"
    raw_fields = (
        "dataset", "model", "match_id", "source_x_px", "source_y_px",
        "target_x_px", "target_y_px", "score", "accepted_roi", "analytic_error_px",
    )
    tensor_cache: dict[tuple[str, int, int, int], np.ndarray] = {}
    cache_dims: dict[tuple[str, int, int, int], tuple[int, int]] = {}
    model_specs = (
        ("raco_aliked_lightglue", args.raco_model, 3),
        ("efficientloftr", args.loftr_model, 1),
    )

    with WGPUMatching(device_type="gpu") as matcher, raw_path.open(
        "w", newline="", encoding="utf-8"
    ) as raw_stream:
        raw_writer = csv.DictWriter(raw_stream, fieldnames=raw_fields)
        raw_writer.writeheader()
        projection_adapter = matcher.adapter_info
        for model_name, model_path, channels in model_specs:
            session = sessions[model_name]
            first_case = cases[0]

            def cached_tensor(
                path: Path, case: dict[str, Any], channels: int = channels
            ) -> tuple[np.ndarray, tuple[int, int]]:
                w, h = case["content_size"]
                key = (str(path), w, h, channels)
                if key not in tensor_cache:
                    tensor_cache[key], cache_dims[key] = load_tensor(path, (w, h), channels)
                return tensor_cache[key], cache_dims[key]

            # Exclude model compilation and first-call costs from per-pair timings.
            warm_ref, _ = cached_tensor(first_case["reference"], first_case)
            warm_tar, _ = cached_tensor(first_case["target"], first_case)
            infer_pairs(model_name, session, warm_ref, warm_tar)

            for case in cases:
                native_width, native_height = case["native_size"]
                content_width, content_height = case["content_size"]
                reference, padded_size = cached_tensor(case["reference"], case)
                target, _ = cached_tensor(case["target"], case)
                inference_start = time.perf_counter()
                source, destination, scores = infer_pairs(model_name, session, reference, target)
                inference_ms = (time.perf_counter() - inference_start) * 1000.0
                if len(source) != len(destination):
                    raise RuntimeError(f"{model_name} returned mismatched source/target pair counts")
                if scores is not None and len(scores) != len(source):
                    scores = None

                source_native = source * np.asarray(
                    [native_width / content_width, native_height / content_height]
                )
                target_native = destination * np.asarray(
                    [native_width / content_width, native_height / content_height]
                )
                accepted = np.ones(len(source), dtype=bool)
                mask = case["mask"]
                if mask is not None:
                    accepted = mask_points(mask, source_native) & mask_points(mask, target_native)
                score_values = None if scores is None else scores
                pairs = CorrespondenceSet.from_preprocessed_pairs(
                    source,
                    destination,
                    image_size=case["native_size"],
                    matcher_id=model_name,
                    source_scale_xy=(content_width / native_width, content_height / native_height),
                    target_scale_xy=(content_width / native_width, content_height / native_height),
                    raw_scores=score_values,
                    accepted=accepted,
                    rejection_reasons=tuple(
                        None if value else "outside_adrien_mask_roi" for value in accepted
                    ) if mask is not None else None,
                    coordinate_mapping={
                        "content_size_px": [content_width, content_height],
                        "padded_size_px": list(padded_size),
                        "resize_backend": "Pillow Image.resize default",
                        "padding": "bottom/right zero padding to 32-pixel network stride",
                    },
                )
                if case["truth"] is not None:
                    seed_points = visible_tensile_seeds(
                        native_width,
                        native_height,
                        case["truth"]["stretch_x"],
                        case["truth"]["stretch_y"],
                    )
                else:
                    seed_points = case["roi_seed_points"]

                raw_error_values = None
                if case["truth"] is not None:
                    target_truth = analytic_target(
                        pairs.source_xy, case["truth"], native_width, native_height
                    )
                    raw_error_values = np.linalg.norm(pairs.target_xy - target_truth, axis=1)
                for index, (source_point, target_point) in enumerate(
                    zip(pairs.source_xy, pairs.target_xy, strict=True)
                ):
                    raw_writer.writerow(
                        {
                            "dataset": case["name"],
                            "model": model_name,
                            "match_id": index,
                            "source_x_px": source_point[0],
                            "source_y_px": source_point[1],
                            "target_x_px": target_point[0],
                            "target_y_px": target_point[1],
                            "score": "" if score_values is None else score_values[index],
                            "accepted_roi": bool(accepted[index]),
                            "analytic_error_px": "" if raw_error_values is None else raw_error_values[index],
                        }
                    )
                direct_kept = raw_error_values[accepted] if raw_error_values is not None else None
                print(
                    f"{case['name']} / {model_name}: {len(source)} raw pairs, "
                    f"{int(accepted.sum())} ROI pairs, inference {inference_ms:.1f} ms",
                    flush=True,
                )
                rows, patches, sequence_index = project_case(
                    matcher,
                    model_name,
                    case,
                    pairs,
                    seed_points,
                    output_dir,
                    sequence_dir,
                    sequence_index,
                )
                for row in rows:
                    row["inference_wall_ms"] = inference_ms
                    row["raw_match_count"] = len(source)
                    row["mask_roi_match_count"] = int(accepted.sum())
                    row["raw_direct_error_median_px"] = (
                        float(np.median(direct_kept)) if direct_kept is not None and len(direct_kept) else None
                    )
                    row["raw_direct_error_rmse_px"] = (
                        float(np.sqrt(np.mean(direct_kept**2))) if direct_kept is not None and len(direct_kept) else None
                    )
                summary_rows.extend(rows)
                patch_rows.extend(patches)
                del reference, target, source, destination, pairs

    sessions.clear()
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(UTC).isoformat(),
                "protocol": {
                    "analytic_cases": [case["name"] for case in cases if case["truth"] is not None],
                    "adrien_case": None if args.skip_adrien else "adrien_000000_to_000075",
                    "analytic_reference_size_px": [3840, 2160],
                    "analytic_network_content_size_px": [640, 360],
                    "adrien_reference_size_px": [7920, 6004],
                    "adrien_network_content_size_px": [640, 485],
                    "model_weights": {
                        "RaCo-ALIKED-LightGlue+": {
                            "path": str(args.raco_model),
                            "sha256": sha256_file(args.raco_model),
                        },
                        "EfficientLoFTR": {
                            "path": str(args.loftr_model),
                            "sha256": sha256_file(args.loftr_model),
                        },
                    },
                    "onnx_execution_provider": "WebGpuExecutionProvider",
                    "onnx_devices": [
                        {"ep_name": device.ep_name, "ep_vendor": device.ep_vendor}
                        for device in ort_devices
                    ],
                    "projection_device": projection_adapter,
                    "projection_backend": "wgpu-py WebGPU/WGSL; input pairs staged from ONNX host outputs",
                    "projection_variants": ["affine", "automatic Gaussian-RBF"],
                    "adrien_roi": "Mask.tiff; both endpoints must fall inside the nonzero mask; images themselves are unmasked",
                    "analytic_error": "exact analytic stretch map at native reference coordinates",
                    "timing_scope": "single measured inference and fit; one inference warmup per model; projection CV and GPU solve both included in fit wall time",
                },
                "software": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "wgpu_py": wgpu.__version__,
                },
                "results": summary_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    patch_path = output_dir / "patch_predictions.csv"
    patch_fields = (
        "dataset", "model", "basis_requested", "basis_effective", "seed_id", "x_px", "y_px",
        "status", "u_px", "v_px", "du_dx", "du_dy", "dv_dx", "dv_dy",
        "analytic_error_px", "support_rms_px", "support_max_px",
    )
    with patch_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=patch_fields)
        writer.writeheader()
        writer.writerows(patch_rows)
    (output_dir / "README.md").write_text(
        "# Projected matcher fields and visual checks\n\n"
        "Each `visualizations/<case>/<model>/<basis>/` directory contains "
        "`image1_ref.png` and `image1_def_cor.png`. The corrected target is "
        "bilinearly sampled at `x + u(x,y), y + v(x,y)` from the fitted field; "
        "it is a visual check, not an additional matcher or a DIC solve. Preview "
        "images are capped at 1920 pixels wide. The ordered hard-linked sequence "
        "is convenient for EOG:\n\n"
        f"```bash\neog {sequence_dir}/*.png\n```\n\n"
        "The affine/RBF error metrics and exact seeds are in `summary.json` and "
        "`patch_predictions.csv`; direct extracted pairs are in `raw_correspondences.csv`.\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path} and {patch_path}")


if __name__ == "__main__":
    main()

"""Re-run current initializers with shared projection, exact seeds and failure GIFs.

Truth is used only by the evaluator and visible-seed construction. The fitting
functions never receive analytic displacement, strain, or seed correctness.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_projection_datasets import (
    analytic_target,
    build_cases,
    infer_pairs,
    mask_points,
    read_truth,
    visible_tensile_seeds,
)
from geometric_consensus import LocalConsensus, bounded_pairs
from PIL import Image
from qualification_visuals import (
    assess_parameters,
    render_pair,
    restore_centres,
)
from qualification_visuals import (
    statistics as error_statistics,
)
from wgpu_matching import (
    CorrespondenceSet,
    CPMOptions,
    DeDoDeBDescriptor,
    DeepMatchingOptions,
    DICDetectorOptions,
    HORGOptions,
    OnnxWGPUKeypointDetector,
    ProjectionOptions,
    WGPUMatching,
    mutual_nearest_cosine,
)
from wgpu_matching.images import load_gray

ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "deepmatching",
    "horg",
    "cpm",
    "xfeat",
    "efficientloftr",
    "raco",
    "dad_b",
    "dedode_b",
    "dic_tensor_b",
    "dic_learned_b",
    "horg_b",
)
LABELS = {
    "deepmatching": "DeepMatching INRIA hierarchy",
    "horg": "HORG",
    "cpm": "CPM-HOG",
    "xfeat": "XFeat",
    "efficientloftr": "EfficientLoFTR",
    "raco": "RaCo-ALIKED-LightGlue+",
    "dad_b": "DaD + DeDoDe-B",
    "dedode_b": "DeDoDe v2-L + DeDoDe-B",
    "dic_tensor_b": "DIC tensor + DeDoDe-B",
    "dic_learned_b": "Learned DIC + DeDoDe-B",
    "horg_b": "HORG/SIFT detector + DeDoDe-B",
}
DEFAULT_MODELS = {
    "dad_b": "/tmp/wgpu_matching_detector_exports/dad_keypoints_k8192.onnx",
    "dedode_b": "/tmp/wgpu_matching_detector_exports/dedodev2_keypoints_k8192.onnx",
    "xfeat": "/tmp/xfeat-eval/xfeat_dense_e2e.onnx",
    "efficientloftr": "/tmp/wgpu-matching-aliked-study/eloftr_border_mask_640x384.onnx",
    "efficientloftr_dynamic": "/tmp/wgpu-matching-aliked-study/eloftr_outdoor_opt.onnx",
    "raco": "/tmp/wgpu-matching-aliked-study/raco_selu_gathernd_i32_v1.onnx",
    "dic_learned_b": str(
        ROOT
        / "results/features/dic_detector_training_discontinuous_20261107/dic_detector_weights.npz"
    ),
}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def archive_model(directory: Path, model, name="model"):
    """Keep full provenance compressed, with a small inspectable coefficient record."""
    directory.mkdir(parents=True, exist_ok=True)
    data = model.to_dict()
    with gzip.open(directory / f"{name}.json.gz", "wt") as stream:
        json.dump(clean(data), stream, allow_nan=False)
    write_json(
        directory / f"{name}_coefficients.json",
        {
            key: value
            for key, value in data.items()
            if key
            not in {
                "diagnostics",
                "correspondence_ids",
                "input_row_indices",
                "robust_weights",
                "residuals_px",
            }
        },
    )


def compact_diagnostics(model):
    if model is None:
        return None
    return {
        key: value
        for key, value in model.diagnostics.items()
        if key
        not in {
            "excluded_rows",
            "retained_raw_scores",
            "retained_reciprocal_errors_px",
            "coordinate_mapping",
            "retained_input_rows",
            "retained_match_ids",
        }
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def cases():
    result, _ = build_cases(
        ROOT / "results/uniaxial/tensile_60pct",
        Path("/home/jeff/CNRS/DIC/AdrienVariational"),
    )
    progressive = Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile")
    truth = read_truth(progressive / "machine_truth.csv")
    extra = [
        {
            "name": f"progressive_{frame:03d}",
            "kind": "analytic_tensile",
            "reference": progressive / "reference.png",
            "target": progressive / f"frame_{frame:03d}.png",
            "native_size": (3840, 2160),
            "content_size": (640, 360),
            "truth": truth[frame],
            "frame": frame,
            "mask": None,
        }
        for frame in (40, 100, 199)
    ]
    return extra + result


def prepare_network(gray: np.ndarray, content: tuple[int, int], channels: int):
    mode = "RGB" if channels == 3 else "L"
    array = (
        np.asarray(
            Image.fromarray(gray).convert(mode).resize(content, Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        / 255
    )
    width, height = content
    pad = ((0, (-height) % 32), (0, (-width) % 32))
    if channels == 3:
        pad += ((0, 0),)
    return np.ascontiguousarray(np.pad(array, pad))


def finalize_pairs(
    source,
    target,
    size,
    method,
    *,
    accepted=None,
    scores=None,
    reasons=None,
    mask=None,
    mapping=None,
):
    source, target = np.asarray(source, float), np.asarray(target, float)
    n = len(source)
    flags = np.ones(n, bool) if accepted is None else np.asarray(accepted, bool).copy()
    why = [None] * n if reasons is None else list(reasons)
    width, height = size
    finite = np.isfinite(source).all(1) & np.isfinite(target).all(1)
    inside = finite.copy()
    for points in (source, target):
        inside &= (points[:, 0] >= 0) & (points[:, 0] < width)
        inside &= (points[:, 1] >= 0) & (points[:, 1] < height)
    roi = np.ones(n, bool)
    if mask is not None:
        roi = mask_points(mask, source) & mask_points(mask, target)
    for i in range(n):
        if not finite[i]:
            why[i] = "non_finite_coordinate"
        elif not inside[i]:
            why[i] = "outside_native_image_or_padding"
        elif not flags[i]:
            why[i] = why[i] or "matcher_rejected"
        elif not roi[i]:
            why[i] = "outside_mask"
    flags &= inside & roi
    return CorrespondenceSet(
        source,
        target,
        tuple(size),
        method,
        accepted=flags,
        raw_scores=scores,
        rejection_reasons=tuple(why),
        coordinate_mapping=mapping or {"frame": "native pixels"},
    )


class Extractor:
    """Optional neural runtimes and native WebGPU engines; no analytic truth."""

    def __init__(
        self, matcher, methods, models, keypoints, descriptor_side, geometry, feature_matcher
    ):
        self.matcher, self.models = matcher, models
        self.keypoints, self.descriptor_side = keypoints, descriptor_side
        self.feature_matcher = feature_matcher
        self.detectors, self.sessions = {}, {}
        self.consensus = LocalConsensus(matcher) if geometry == "local_consensus" else None
        common = set(methods) & {"dad_b", "dedode_b", "dic_tensor_b", "dic_learned_b", "horg_b"}
        self.descriptor = (
            DeDoDeBDescriptor(max_side=descriptor_side, device="cpu") if common else None
        )
        for method in ("dad_b", "dedode_b"):
            if method in methods:
                self.detectors[method] = OnnxWGPUKeypointDetector(models[method], max_side=640)
                if self.detectors[method].keypoint_capacity < keypoints:
                    raise ValueError(
                        f"{method}: graph capacity is smaller than the common point budget"
                    )
        if set(methods) & {"xfeat", "efficientloftr", "raco"}:
            import onnxruntime as ort
            import onnxruntime_ep_webgpu as ep

            devices = [d for d in ort.get_ep_devices() if d.ep_name == ep.get_ep_name()]
            if not devices:
                ort.register_execution_provider_library("qualification", ep.get_library_path())
                devices = [d for d in ort.get_ep_devices() if d.ep_name == ep.get_ep_name()]
            if not devices:
                raise RuntimeError("No ONNX Runtime WebGPU device")
            wanted = set(methods) & {"xfeat", "efficientloftr", "raco"}
            if "efficientloftr" in wanted:
                wanted.add("efficientloftr_dynamic")
            for method in wanted:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 8
                options.inter_op_num_threads = 1
                options.log_severity_level = 3
                options.add_provider_for_devices(devices, {"preferredLayout": "NHWC"})
                self.sessions[method] = ort.InferenceSession(
                    models[method], sess_options=options, providers=None
                )

    def extract(self, method, ref, tar, seeds, *, mask=None):
        size = (ref.shape[1], ref.shape[0])
        seed_tuple = tuple(tuple(float(v) for v in point) for point in seeds)
        metadata, stage = {}, {}
        started = time.perf_counter()
        if method in ("deepmatching", "horg", "cpm"):
            if method == "deepmatching":
                options = DeepMatchingOptions(search_radius_px=1024, seed_points=seed_tuple)
                field = self.matcher.match_deepmatching(
                    ref, tar, options=options, include_correspondences=True
                )
            elif method == "horg":
                options = HORGOptions(seed_points=seed_tuple)
                field = self.matcher.match_horg(
                    ref, tar, options=options, include_correspondences=True
                )
            else:
                options = CPMOptions(search_radius_px=1024, seed_points=seed_tuple)
                field = self.matcher.match_cpm(
                    ref, tar, options=options, include_correspondences=True
                )
            pairs = CorrespondenceSet.from_match_field(field)
            metadata.update(
                options=asdict(options),
                algorithm=field.algorithm,
                native_seed_accepted=len(field.accepted),
                export_kind=field.metadata.get("raw_correspondence_kind"),
                seed_output_kind="native estimates; only exported direct pairs are projected",
            )
            result = finalize_pairs(
                pairs.source_xy,
                pairs.target_xy,
                size,
                method,
                accepted=pairs.accepted,
                scores=pairs.raw_scores,
                reasons=pairs.rejection_reasons,
                mask=mask,
                mapping=pairs.coordinate_mapping,
            )
        elif method.endswith("_b"):
            detected = []
            t = time.perf_counter()
            detector_meta = []
            for image in (ref, tar):
                if method in self.detectors:
                    features = self.detectors[method].detect(image, max_keypoints=self.keypoints)
                elif method == "horg_b":
                    features = self.matcher.detect_horg(
                        image, max_keypoints=self.keypoints, reduction=4
                    )
                else:
                    features = self.matcher.detect_dic(
                        image,
                        options=DICDetectorOptions(max_keypoints=self.keypoints),
                        learned_weights_path=(
                            self.models["dic_learned_b"] if method == "dic_learned_b" else None
                        ),
                    ).features
                detected.append(features)
                detector_meta.append(features.metadata)
            stage["detection_pair_s"] = time.perf_counter() - t
            t = time.perf_counter()
            described = [
                self.descriptor.describe(image, f)
                for image, f in zip((ref, tar), detected, strict=True)
            ]
            stage["description_pair_s"] = time.perf_counter() - t
            t = time.perf_counter()
            matched = (
                self.matcher.match_features(
                    *described,
                    seeds,
                    min_cosine_similarity=0.0,
                    fit_local_models=False,
                )
                if self.feature_matcher == "wgpu"
                else None
            )
            matches = (
                matched.correspondences
                if matched is not None
                else mutual_nearest_cosine(*described, min_cosine_similarity=0.0)
            )
            stage["mutual_matching_s"] = time.perf_counter() - t
            values = matches.points
            result = finalize_pairs(values[:, :2], values[:, 2:], size, method, mask=mask)
            metadata.update(
                detectors=detector_meta,
                descriptor=self.descriptor.metadata,
                point_budget=self.keypoints,
                detected_counts=[len(f.xy) for f in detected],
                common_matcher="mutual nearest cosine; minimum similarity 0",
                feature_matcher=self.feature_matcher,
                gpu_local_initializations=(
                    [asdict(m) for m in matched.initializations.matches]
                    if matched is not None
                    else None
                ),
            )
        else:
            if method == "xfeat":
                width = min(3840, size[0])
                content = (width, round(size[1] * width / size[0]))
                inputs = []
                for image in (ref, tar):
                    array = (
                        np.asarray(
                            Image.fromarray(image).resize(content, Image.Resampling.BICUBIC),
                            dtype=np.float32,
                        )
                        / 255
                    )
                    inputs.append(np.ascontiguousarray(np.repeat(array[None, None], 3, 1)))
                session = self.sessions[method]
                source, target = session.run(
                    None, dict(zip(("image0", "image1"), inputs, strict=True))
                )
                source, target = source.reshape(-1, 2), target.reshape(-1, 2)
                scores, session_name = None, method
            else:
                content = (640, round(size[1] * 640 / size[0]))
                channels = 3 if method == "raco" else 1
                inputs = [prepare_network(image, content, channels) for image in (ref, tar)]
                session_name = method
                if method == "efficientloftr" and inputs[0].shape != (384, 640):
                    session_name = "efficientloftr_dynamic"
                session = self.sessions[session_name]
                source, target, scores = infer_pairs(
                    "raco_aliked_lightglue" if method == "raco" else method,
                    session,
                    *inputs,
                )
            result = finalize_pairs(
                restore_centres(source, size, content),
                restore_centres(target, size, content),
                size,
                method,
                scores=scores,
                mask=mask,
                mapping={
                    "resize": "bicubic",
                    "inverse_mapping": "(x+0.5)*native/content-0.5",
                    "content_size": content,
                    "padded_size": inputs[0].shape,
                    "model_path": self.models[session_name],
                },
            )
            metadata.update(
                model_path=self.models[session_name],
                providers=session.get_providers(),
                content_size=content,
                network_preprocessing="bicubic; grayscale replicated when RGB required",
            )
        if self.consensus is not None:
            result, verification = self.consensus.filter(result, seeds)
            metadata["geometric_verification"] = verification
            stage["geometric_verification_s"] = verification["wall_s"]
        stage["matching_pipeline_s"] = time.perf_counter() - started
        metadata["stages_s"] = stage
        return result, metadata


def fit(matcher, pairs, seeds, basis):
    """No ground-truth parameter can enter this numerical path."""
    t = time.perf_counter()
    selected, capacity = bounded_pairs(pairs)
    model = matcher.fit_initialization_field(
        selected, pairs.image_size, options=ProjectionOptions(basis=basis)
    )
    model.diagnostics["projection_capacity_selection"] = capacity
    fitted = time.perf_counter()
    patches = model.initialize_patches(seeds, patch_size=128, projection="support_fit")
    return (
        model,
        patches,
        {
            "field_fit_s": fitted - t,
            "patch_projection_s": time.perf_counter() - fitted,
            "fit_and_patch_s": time.perf_counter() - t,
        },
    )


def save_pairs(path, pairs, truth):
    expected = (
        analytic_target(pairs.source_xy, truth, *pairs.image_size) if truth is not None else None
    )
    errors = np.linalg.norm(pairs.target_xy - expected, axis=1) if expected is not None else None
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "id",
                "x",
                "y",
                "target_x",
                "target_y",
                "upstream_accepted",
                "accepted",
                "reason",
                "score",
                "analytic_error_px",
            ]
        )
        for i, (source, target) in enumerate(zip(pairs.source_xy, pairs.target_xy, strict=True)):
            upstream = (
                bool(pairs.accepted[i]) or pairs.rejection_reasons[i] == "local_consensus_rejected"
            )
            writer.writerow(
                [
                    i,
                    *source,
                    *target,
                    upstream,
                    bool(pairs.accepted[i]),
                    pairs.rejection_reasons[i],
                    "" if pairs.raw_scores is None else pairs.raw_scores[i],
                    "" if errors is None else errors[i],
                ]
            )
    selected = np.asarray(pairs.accepted, bool)
    return {
        "raw_count": len(selected),
        "retained_count": int(selected.sum()),
        "upstream_retained_count": int(
            sum(
                bool(flag) or reason == "local_consensus_rejected"
                for flag, reason in zip(pairs.accepted, pairs.rejection_reasons, strict=True)
            )
        ),
        "raw_errors_px": error_statistics(errors) if errors is not None else None,
        "retained_errors_px": error_statistics(errors[selected]) if errors is not None else None,
        "retained_correct_le2": int(np.count_nonzero(errors[selected] <= 2))
        if errors is not None
        else None,
        "retained_correct_lt5": int(np.count_nonzero(errors[selected] < 5))
        if errors is not None
        else None,
    }


def qualify_one(extractor, matcher, method, case, ref, tar, seeds, output, repeats, render_width):
    directory = output / case["name"] / method
    directory.mkdir(parents=True, exist_ok=True)
    print(f"START {case['name']} / {method}", flush=True)
    warm_start = time.perf_counter()
    warm_pairs, _ = extractor.extract(method, ref, tar, seeds, mask=case["mask"])
    try:
        fit(matcher, warm_pairs, seeds, "affine")
    except (ValueError, RuntimeError):
        pass
    warm_s = time.perf_counter() - warm_start
    trials, result = [], None
    for trial in range(repeats):
        cycle_start = time.perf_counter()
        pairs, extraction = extractor.extract(method, ref, tar, seeds, mask=case["mask"])
        try:
            model, patches, stages = fit(matcher, pairs, seeds, "affine")
            fit_error = None
        except (ValueError, RuntimeError) as exc:
            model, patches, stages = None, [], {}
            fit_error = str(exc)
        cycle_s = time.perf_counter() - cycle_start
        trials.append(
            {
                "trial": trial,
                "full_affine_pipeline_s": cycle_s,
                "extraction_s": extraction["stages_s"],
                "projection_s": stages,
                "fit_error": fit_error,
                "raw_pairs": len(pairs.source_xy),
                "retained_pairs": int(pairs.accepted.sum()),
                "pair_digest": hashlib.sha256(pairs.pairs_xyxy.tobytes()).hexdigest(),
                "acceptance_digest": hashlib.sha256(pairs.accepted.tobytes()).hexdigest(),
            }
        )
        result = (pairs, extraction, model, patches, fit_error)
        print(
            f"  trial {trial + 1}/{repeats}: {cycle_s:.3f}s, "
            f"{int(pairs.accepted.sum())}/{len(pairs.source_xy)} pairs",
            flush=True,
        )
    pairs, extraction, affine, affine_patches, fit_error = result
    raw_stats = save_pairs(directory / "correspondences.csv.gz", pairs, case["truth"])
    np.savez_compressed(
        directory / "pair_cache.npz",
        source=pairs.source_xy,
        target=pairs.target_xy,
        accepted=pairs.accepted,
        upstream_accepted=np.asarray(
            [
                bool(flag) or reason == "local_consensus_rejected"
                for flag, reason in zip(pairs.accepted, pairs.rejection_reasons, strict=True)
            ]
        ),
    )
    rows = []
    for basis in ("affine", "gaussian_rbf"):
        error = fit_error
        model, patches, stages = affine, affine_patches, {}
        if basis != "affine":
            try:
                model, patches, stages = fit(matcher, pairs, seeds, basis)
                error = None
            except (ValueError, RuntimeError) as exc:
                model, patches, error = None, [], str(exc)
        parameters = [p.parameters for p in patches] if model is not None else [None] * len(seeds)
        assessment = assess_parameters(parameters, seeds, case["truth"], pairs.image_size)
        verdict = (
            "NO ANALYTIC TRUTH"
            if case["truth"] is None
            else "PASS"
            if assessment["pass_below_5px"]
            else "FAIL"
        )
        if model is None:
            verdict += " - NO ACCEPTED FIELD"
        detail = (
            f"{assessment['correct_lt5']}/{len(seeds)} windows below 5 px"
            if case["truth"] is not None
            else "Residuals and appearance do not establish physical accuracy"
        )
        bdir = directory / basis
        visual_model, diagnostic = model, False
        diagnostic_error = None
        if model is None and len(pairs.source_xy) >= 3:
            # Optional failure illustration: never changes assessment or acceptance.
            try:
                proposals = CorrespondenceSet(
                    pairs.source_xy,
                    pairs.target_xy,
                    pairs.image_size,
                    method + "_rejected_diagnostic",
                )
                visual_model, _, _ = fit(matcher, proposals, seeds, basis)
                diagnostic = True
                archive_model(bdir, visual_model, "rejected_diagnostic_model")
            except (RuntimeError, ValueError) as exc:
                diagnostic_error = str(exc)
        effective = model.effective_basis if model is not None else None
        visuals = render_pair(
            ref,
            tar,
            visual_model,
            bdir,
            label=f"{case['name']} | {LABELS[method]} | request {basis}, effective {effective or 'none'}",
            verdict=verdict,
            detail=detail,
            width=render_width,
            mask=case["mask"],
            diagnostic_only=diagnostic,
            zoom=case["name"] == "tensile_60pct" or case["name"].startswith("adrien"),
        )
        if model is not None:
            archive_model(bdir, model)
        row = {
            "case": case["name"],
            "method": method,
            "label": LABELS[method],
            "basis_requested": basis,
            "basis_effective": effective,
            "status": "estimated" if model is not None else "no_field",
            "verdict": verdict,
            "fit_error": error,
            "assessment": assessment,
            "direct_pairs": raw_stats,
            "extraction": extraction,
            "warmup_s": warm_s,
            "trials": trials,
            "affine_pipeline_wall_s": [t["full_affine_pipeline_s"] for t in trials],
            "affine_pipeline_median_s": statistics.median(
                t["full_affine_pipeline_s"] for t in trials
            ),
            "rbf_fit_single_run_stages_s": stages if basis != "affine" else None,
            "diagnostics": compact_diagnostics(model),
            "visualizations": visuals,
            "rejected_diagnostic_error": diagnostic_error,
            "truth_access": "evaluation and visibility only; never fitting or basis selection",
            "image_size": pairs.image_size,
        }
        write_json(bdir / "result.json", row)
        rows.append(row)
    write_json(directory / "complete.json", rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--keypoints", type=int, default=8192)
    parser.add_argument("--descriptor-side", type=int, default=1024)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--geometry", choices=("local_consensus", "raw"), default="local_consensus")
    parser.add_argument("--feature-matcher", choices=("wgpu", "cpu"), default="wgpu")
    parser.add_argument("--models", type=Path, help="Optional JSON overriding external model paths")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.keypoints < 1 or args.render_width < 64:
        parser.error("repeats, keypoints and render width must be positive")
    import torch

    torch.set_num_threads(8)
    models = dict(DEFAULT_MODELS)
    if args.models:
        models.update(json.loads(args.models.read_text()))
    selected = cases()
    if args.cases:
        selected = [c for c in selected if c["name"] in args.cases]
        missing = set(args.cases) - {c["name"] for c in selected}
        if missing:
            parser.error(f"Unknown cases: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "methods": args.methods,
        "cases": [c["name"] for c in selected],
        "repeats": args.repeats,
        "keypoints": args.keypoints,
        "descriptor_side": args.descriptor_side,
        "render_width": args.render_width,
        "projection_defaults": asdict(ProjectionOptions()),
        "geometric_verification": args.geometry,
        "feature_matcher": args.feature_matcher,
        "local_consensus_settings": {
            "radius_px": 256,
            "threshold_px": 3,
            "iterations": 256,
            "minimum_inliers": 6,
            "minimum_candidates": 8,
            "minimum_inlier_fraction": 0.2,
        },
        "maximum_projection_pairs": 8192,
        "models": {
            k: {"path": v, "sha256": sha256(Path(v)) if Path(v).is_file() else None}
            for k, v in models.items()
        },
        "source_files": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted(
                list((ROOT / "python/src/wgpu_matching").rglob("*.py"))
                + list((ROOT / "python/src/wgpu_matching/shaders").glob("*.wgsl"))
                + [
                    Path(__file__).resolve(),
                    Path(__file__).with_name("qualification_visuals.py").resolve(),
                ]
                + [Path(__file__).with_name("geometric_consensus.py").resolve()]
            )
        },
        "images": [
            {
                "case": c["name"],
                "reference": str(c["reference"]),
                "target": str(c["target"]),
                "reference_sha256": sha256(c["reference"]),
                "target_sha256": sha256(c["target"]),
                "size": c["native_size"],
                "analytic_truth": c["truth"],
                "mask_sha256": (
                    sha256(Path("/home/jeff/CNRS/DIC/AdrienVariational/Mask.tiff"))
                    if c["mask"] is not None
                    else None
                ),
            }
            for c in selected
        ],
        "pass_rule": "all visible requested windows have finite displacement error strictly <5 px",
        "primary_projection": "affine, predeclared from prior tensile study",
        "secondary_projection": "spatial cross-validation between affine and Gaussian RBF",
        "timing_scope": "affine: measured complete extraction+fit+patch call; decoding, model load, warmup, rendering excluded",
        "rbf_timing_scope": "single fit+patch call on final measured correspondence set; no end-to-end FPS claim",
        "coordinate_mapping": "integer source pixel centres; inverse resize (x+0.5)*scale-0.5",
        "oracle_policy": "truth used only for seed visibility and evaluation",
        "peak_gpu_memory": "not measured",
        "peak_host_memory": "not measured",
        "reference_caching": "none: each timed call processes two decoded images",
        "numerical_threads": "PyTorch8, ORT8; launch with OpenBLAS/OMP/MKL8",
    }
    fingerprint = hashlib.sha256(json.dumps(clean(protocol), sort_keys=True).encode()).hexdigest()
    manifest_path = args.output / "manifest.json"
    previous = None
    if args.resume and manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        previous = old
        if old["protocol_sha256"] != fingerprint:
            raise ValueError(
                "Resume protocol/model/source fingerprint differs; use a new output directory"
            )
    elif manifest_path.exists():
        raise ValueError("Output run already exists; use --resume or a new directory")
    manifest = {
        "created_utc": previous["created_utc"] if previous else datetime.now(UTC).isoformat(),
        "resumed_utc": (
            []
            if previous is None
            else previous.get("resumed_utc", []) + [datetime.now(UTC).isoformat()]
        ),
        "protocol_sha256": fingerprint,
        "protocol": protocol,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "decoded_images": {},
    }
    rows = []
    with WGPUMatching(device_type="gpu") as matcher:
        manifest["adapter"] = matcher.adapter_info
        write_json(manifest_path, manifest)
        extractor = Extractor(
            matcher,
            args.methods,
            models,
            args.keypoints,
            args.descriptor_side,
            args.geometry,
            args.feature_matcher,
        )
        for case in selected:
            ref, tar = load_gray(case["reference"]), load_gray(case["target"])
            if ref.shape != tar.shape or ref.shape[::-1] != tuple(case["native_size"]):
                raise ValueError("Dataset dimensions changed")
            manifest["decoded_images"][case["name"]] = {
                "reference_sha256": hashlib.sha256(ref.tobytes()).hexdigest(),
                "target_sha256": hashlib.sha256(tar.tobytes()).hexdigest(),
                "dtype": str(ref.dtype),
            }
            write_json(manifest_path, manifest)
            seeds = (
                visible_tensile_seeds(
                    *case["native_size"], case["truth"]["stretch_x"], case["truth"]["stretch_y"]
                )
                if case["truth"]
                else case["roi_seed_points"]
            )
            for method in args.methods:
                completed = args.output / case["name"] / method / "complete.json"
                if args.resume and completed.exists():
                    rows.extend(json.loads(completed.read_text()))
                    print(f"RESUME {case['name']} / {method}", flush=True)
                    continue
                try:
                    rows.extend(
                        qualify_one(
                            extractor,
                            matcher,
                            method,
                            case,
                            ref,
                            tar,
                            seeds,
                            args.output,
                            args.repeats,
                            args.render_width,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - record per-method failures and continue campaign
                    failure = {
                        "case": case["name"],
                        "method": method,
                        "status": "execution_error",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    write_json(
                        args.output / case["name"] / method / "execution_error.json", failure
                    )
                    for basis in ("affine", "gaussian_rbf"):
                        assessment = assess_parameters(
                            [None] * len(seeds), seeds, case["truth"], case["native_size"]
                        )
                        visual = render_pair(
                            ref,
                            tar,
                            None,
                            args.output / case["name"] / method / basis,
                            label=f"{case['name']} | {LABELS[method]} | {basis}",
                            verdict="FAIL - EXECUTION ERROR",
                            detail=str(exc)[:140],
                            width=args.render_width,
                            mask=case["mask"],
                            zoom=case["name"] == "tensile_60pct"
                            or case["name"].startswith("adrien"),
                        )
                        rows.append(
                            {
                                **failure,
                                "basis_requested": basis,
                                "basis_effective": None,
                                "assessment": assessment,
                                "visualizations": visual,
                                "verdict": "FAIL - EXECUTION ERROR",
                            }
                        )
                    print(f"ERROR {case['name']} / {method}: {exc}", flush=True)
                write_json(
                    args.output / "summary.json", {"manifest": "manifest.json", "rows": rows}
                )
    write_json(args.output / "summary.json", {"manifest": "manifest.json", "rows": rows})
    print(f"Completed {len(rows)} result records in {args.output}", flush=True)


if __name__ == "__main__":
    main()

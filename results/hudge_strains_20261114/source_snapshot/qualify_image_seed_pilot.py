"""Measure image-only sparse seed selection on archived development pairs."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from wgpu_matching.dic_synthetic import generate_speckle_pair, perturb_target
from wgpu_matching.seed_initialization import initialize, initialize_with_escalation


def rotations_for_pair(rotations, texture, warp, mode):
    """Match seeds and family; distinct families may share the same seeds."""
    def family_from_path(path):
        name = Path(path).name
        if name.startswith("shear_band_"):
            return "shear_band"
        return name.split("_", 1)[0]

    return [r for r in rotations if r["texture_seed"] == texture and r["warp_seed"] == warp
            and r.get("family", family_from_path(r["windows"])) == mode]


def file_sha256(path):
    """Hash input files without loading full-resolution images into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_lazy_cache(directory, fingerprint):
    """Refuse unverified or incompatible rotation-cache reuse."""
    marker = directory / "input_fingerprint.json"
    if marker.exists():
        if json.loads(marker.read_text()) != fingerprint:
            raise ValueError("lazy rotation cache has incompatible inputs")
    elif directory.exists() and any(directory.rglob("report.json")):
        raise ValueError("lazy rotation cache has no input fingerprint")
    return marker


def source_model_size(protocol):
    """Recover the original model raster, including non-16:9 real images."""
    size = protocol.get("model_input_px", [192, 108])
    if len(size) != 2 or any(not isinstance(x, int) or x <= 0 for x in size):
        raise ValueError("invalid model image dimensions in source protocol")
    return size


def select_archive_windows(files, families=None, seed_offsets=None, textures=None):
    """Select only the explicitly requested synthetic families and seed groups."""
    if families:
        files = [p for p in files if p.stem.split("_texture", 1)[0] in families]
    if seed_offsets:
        files = [p for p in files if int(p.parent.name.removeprefix("windows_seed")) in seed_offsets]
    if textures:
        files = [p for p in files if int(p.stem.split("texture")[1].split("_")[0]) in textures]
    return files


def main():
    main_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sparse-stride", type=int, default=4)
    parser.add_argument("--tau-u", type=float, default=8)
    parser.add_argument("--tau-g", type=float, default=.25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rotation-manifest", type=Path)
    parser.add_argument("--lazy-rotation-weights", type=Path,
                        help="experimentally generate rotation hypotheses only when initial seeds are insufficient")
    parser.add_argument("--lazy-rotation-output", type=Path)
    parser.add_argument("--ndf-onnx-webgpu-cache", type=Path,
                        help="use ONNX Runtime WebGPU for lazily generated rotation hypotheses")
    parser.add_argument("--textures", nargs="+", type=int)
    parser.add_argument("--families", nargs="+", choices=("heterogeneous", "translation", "rigid", "affine", "discontinuous", "shear_band"))
    parser.add_argument("--shear-band-peak-gradient", type=float, default=.67)
    parser.add_argument("--seed-offsets", nargs="+", type=int)
    parser.add_argument("--escalation", action="store_true")
    parser.add_argument("--batch-anchor-confirmation", action="store_true",
                        help="experimentally batch independent neighbour solves and replay serial confirmation")
    parser.add_argument("--gpu-frontier", action="store_true",
                        help="experimentally retain frontier states and IC-GN trials on the solver's GPU")
    parser.add_argument("--gpu-resident-selection", action="store_true",
                        help="keep the selected sparse field on GPU through anchor confirmation")
    parser.add_argument("--gpu-hypothesis-selection", action="store_true",
                        help="rank sparse IC-GN hypotheses on GPU")
    parser.add_argument("--gpu-anchor-frontier", action="store_true",
                        help="confirm anchors on GPU and feed the GPU frontier without intermediate readback")
    parser.add_argument("--minimum-anchors", type=int, default=3)
    parser.add_argument("--local-affine-escalation", action="store_true")
    parser.add_argument("--escalate-unresolved", action="store_true",
                        help="experimentally retry large unresolved components even with enough anchors")
    parser.add_argument("--local-affine-validation", action="store_true",
                        help="evaluate local affine hypotheses on every pair as an ablation")
    parser.add_argument("--same-budget-control", action="store_true")
    parser.add_argument("--frozen-parameters", type=Path)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--subset-stride", type=int, default=64)
    parser.add_argument("--discontinuity-boundary-fraction", type=float, default=.5)
    parser.add_argument("--speckle-noise-std", type=float, default=.025)
    parser.add_argument("--target-gain", type=float, default=1.)
    parser.add_argument("--target-offset", type=float, default=0.)
    parser.add_argument("--target-blur-sigma", type=float, default=0.)
    parser.add_argument("--target-noise-std", type=float, default=0.)
    parser.add_argument("--periodic-strength", type=float, default=0.)
    parser.add_argument("--periodic-period-px", type=float, default=128.)
    parser.add_argument("--external-reference", type=Path)
    parser.add_argument("--external-target", type=Path)
    parser.add_argument("--roi-mask", type=Path)
    parser.add_argument("--use-cached-solver-images", action="store_true",
                        help="read verified exact solver pixels exported with the shared-prefit archives")
    args = parser.parse_args()
    if bool(args.lazy_rotation_weights) != bool(args.lazy_rotation_output):
        parser.error("lazy rotation requires both weights and an output directory")
    if args.lazy_rotation_weights and args.rotation_manifest:
        parser.error("lazy rotation and a precomputed rotation manifest are mutually exclusive")
    if (args.local_affine_escalation or args.local_affine_validation) and not (args.rotation_manifest or args.lazy_rotation_weights):
        parser.error("local affine escalation requires three rotation hypotheses with hard matches")
    if args.lazy_rotation_weights and args.local_affine_validation:
        parser.error("lazy rotation is incompatible with the unconditional local-affine ablation")
    if args.lazy_rotation_weights and args.escalate_unresolved:
        parser.error("lazy rotation currently uses the frozen minimum-anchor policy, not --escalate-unresolved")
    if args.escalate_unresolved and not args.local_affine_escalation:
        parser.error("--escalate-unresolved requires --local-affine-escalation")
    if args.gpu_resident_selection and not (args.gpu_frontier and args.gpu_anchor_frontier and args.gpu_hypothesis_selection):
        parser.error("--gpu-resident-selection requires GPU frontier, anchors and hypothesis selection")
    if args.gpu_hypothesis_selection and not args.gpu_frontier:
        parser.error("--gpu-hypothesis-selection requires --gpu-frontier")
    if args.gpu_anchor_frontier and not args.gpu_frontier:
        parser.error("--gpu-anchor-frontier requires --gpu-frontier")
    if args.gpu_frontier and args.same_budget_control:
        parser.error("--same-budget-control is not yet available for the GPU-resident frontier")
    if bool(args.external_reference) != bool(args.external_target):
        parser.error("external reference and target must be supplied together")
    if args.frozen_parameters:
        frozen = json.loads(args.frozen_parameters.read_text())["selected"]
        for key in ("tau_u", "tau_g", "sparse_stride", "minimum_anchors"):
            if getattr(args, key) != frozen[key]:
                parser.error(f"{key} differs from the frozen development value")
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.archive.glob("affine_ncc/windows_seed*/*.npz"))
    files = select_archive_windows(files, args.families, args.seed_offsets, args.textures)
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error("no archived pairs found")
    solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=args.subset_size,
                      subset_stride=args.subset_stride, interpolation="linear", motion_model="affine",
                      outlier_policy="disabled")
    startup_s = time.perf_counter() - main_start
    rows = []
    gpu_frontier = None
    rotations = json.loads(args.rotation_manifest.read_text()) if args.rotation_manifest else []
    for path in files:
        case_start = time.perf_counter()
        with np.load(path) as data:
            data = {k: data[k].copy() for k in data.files}
        affine_path = args.archive / "affine_only" / path.parent.name / path.name
        if affine_path.is_file():
            with np.load(affine_path) as affine_data:
                affine = affine_data["initial_parameters"].copy()
        elif "pre_ncc_initial_parameters" in data:
            affine = data["pre_ncc_initial_parameters"].copy()
        else:
            raise ValueError(f"missing affine-only input for {path.name}")
        name = path.stem
        mode = name.split("_texture", 1)[0]
        if mode not in {"heterogeneous", "translation", "rigid", "affine", "discontinuous", "shear_band"}:
            raise ValueError(f"unsupported motion family {mode}")
        texture = int(name.split("texture")[1].split("_")[0])
        warp = int(name.split("warp")[1])
        seed_offset = texture // 1000000 * 1000000
        reports = list((args.archive / "affine_ncc").glob(f"*seed{seed_offset}.json"))
        if not reports and (args.archive / "affine_ncc/report.json").is_file():
            reports = [args.archive / "affine_ncc/report.json"]
        if len(reports) != 1:
            raise ValueError(f"expected one source protocol for {name}")
        source_protocol = json.loads(reports[0].read_text())["protocol"]
        if (mode == "shear_band"
                and source_protocol.get("shear_band_peak_gradient") != args.shear_band_peak_gradient):
            raise ValueError(f"archive {name} has a different shear-band peak gradient")
        for option in ("external_reference", "external_target"):
            value = getattr(args, option)
            if source_protocol.get(option) != (str(value) if value else None):
                raise ValueError(f"archive {name} has different {option}")
        for option, default in (("discontinuity_boundary_fraction", .5), ("speckle_noise_std", .025),
                                ("target_gain", 1.), ("target_offset", 0.), ("target_blur_sigma", 0.),
                                ("target_noise_std", 0.), ("periodic_strength", 0.), ("periodic_period_px", 128.)):
            if source_protocol.get(option, default) != getattr(args, option):
                raise ValueError(f"archive {name} has different {option}")
        image_start = time.perf_counter()
        if args.use_cached_solver_images:
            from solver_image_cache import read_solver_images

            cache_dir = args.archive / "affine_ncc" / f"solver_images_seed{seed_offset}"
            sources = (args.external_reference, args.external_target) if args.external_reference else None
            reference_pixels, target = read_solver_images(cache_dir, path, source_files=sources)
        else:
            pair = generate_speckle_pair(texture, warp, 3840, 2160, motion_mode=mode,
                                         shear_band_peak_gradient=args.shear_band_peak_gradient if mode == "shear_band" else None,
                                         discontinuity_boundary_fraction=args.discontinuity_boundary_fraction,
                                         texture_options={"characteristic_size_px": 7., "coverage": .31,
                                                          "blur_sigma_px": .9, "softness": .612,
                                                          "noise_std": args.speckle_noise_std,
                                                          "periodic_strength": args.periodic_strength,
                                                          "periodic_period_px": args.periodic_period_px}) if not args.external_reference else None
            if args.external_reference:
                from types import SimpleNamespace

                from external_dic_input import load_gray_pair
                external_reference, external_target = load_gray_pair(args.external_reference, args.external_target)
                pair = SimpleNamespace(reference=external_reference, target=external_target)
            pair.target = perturb_target(pair.target, warp, gain=args.target_gain, offset=args.target_offset,
                                         blur_sigma=args.target_blur_sigma, noise_std=args.target_noise_std)
            reference_pixels = np.rint(pair.reference * 255).astype(np.uint8)
            target = np.rint(pair.target * 255).astype(np.uint8)
        image_prepare_s = time.perf_counter() - image_start
        roi_mask = None
        if args.roi_mask:
            from PIL import Image
            with Image.open(args.roi_mask) as image:
                roi_mask = np.asarray(image.convert("L")) > 0
            if roi_mask.shape != reference_pixels.shape:
                raise ValueError("region mask and image dimensions differ")
        reference_start = time.perf_counter()
        solver.set_reference(reference_pixels,
                             iterations=30, tolerance=1e-3, zncc_min=.65, roi_mask=roi_mask)
        solver_reference_and_target_s = time.perf_counter() - reference_start
        centres = data["centres_xy_native"]
        if not np.isclose(np.min(centres[:, 0]), (args.subset_size - 1) / 2):
            raise ValueError("archive grid does not match requested subset size")
        x_coordinates = np.unique(centres[:, 0])
        if len(x_coordinates) > 1 and not np.allclose(np.diff(x_coordinates), args.subset_stride):
            raise ValueError("archive grid does not match requested subset stride")
        count = len(centres)
        shape = (len(np.unique(centres[:, 1])), len(np.unique(centres[:, 0])))

        def raw_solve(indices, candidates, count=count, target=target, roi_mask=roi_mask):
            states = np.zeros((count, 8), np.float32)
            states[indices, :6] = candidates
            states[indices, 6] = 1
            active = np.zeros(count, np.uint32)
            active[indices] = 1
            if roi_mask is not None:
                active = active.reshape(solver.roi.selected.shape)
            result = solver._register_seeded_frame(target, states, active)
            return (np.asarray(result.parameters).reshape(-1, 6)[indices],
                    np.asarray(result.status).reshape(-1)[indices] > .5)

        schedule = []
        def solve(indices, candidates, schedule=schedule):
            schedule.append(indices.copy())
            return raw_solve(indices, candidates)

        frontier_stats = []
        selector_stats = []
        if args.gpu_frontier:
            from wgpu_matching.gpu_frontier import GPUFrontier

            if gpu_frontier is None:
                gpu_frontier = GPUFrontier(solver)

            def gpu_propagate(centres_arg, shape_arg, valid_arg, states_arg, seeds_arg, solve_arg,
                              *, tau_u, tau_g, selected_buffer=None, frontier=gpu_frontier, target_img=target, stats=frontier_stats):
                if shape_arg != (solver.grid_height, solver.grid_width) or len(centres_arg) != len(valid_arg):
                    raise ValueError("GPU frontier grid mismatch")
                result_gpu = frontier.propagate(target_img, valid_arg, states_arg, seeds_arg, tau_u=tau_u, tau_g=tau_g,
                                                confirm_candidates=args.gpu_anchor_frontier, selected_buffer=selected_buffer,
                                                target_uploaded=selected_buffer is not None)
                if not result_gpu["converged"]:
                    raise RuntimeError("GPU frontier stopped before convergence")
                stats.append({key: result_gpu[key] for key in
                                       ("passes", "active_windows", "counter_readbacks", "counter_readback_bytes",
                                        "final_readback_bytes", "anchor_solver_dispatches", "wall_s")})
                return result_gpu
            propagation = gpu_propagate

            def gpu_select(sparse, hypotheses_arg, frontier=gpu_frontier, target_img=target, stats=selector_stats):
                selection = frontier.select_hypotheses(target_img, sparse, hypotheses_arg,
                                                       resident=args.gpu_resident_selection)
                stats.append({key: selection[key] for key in
                              ("readback_bytes", "solver_dispatches", "wall_s")})
                return selection

        else:
            propagation = None
            gpu_select = None

        start = time.perf_counter()
        hypotheses = [("zero", np.zeros((count, 6)))]
        pair_rotations = rotations_for_pair(rotations, texture, warp, mode)
        if args.rotation_manifest and sorted(r["rank"] for r in pair_rotations) != [1, 2, 3]:
            raise ValueError(f"missing or duplicate rotation hypotheses for {name}")
        for rotation in sorted(pair_rotations, key=lambda r: r["rank"]):
            with np.load(rotation["windows"]) as rotation_data:
                if not np.array_equal(rotation_data["centres_xy_native"], centres):
                    raise ValueError("rotation hypothesis has a different window grid")
                hypotheses.append((f"rotation_rank{rotation['rank']}",
                                   rotation_data["initial_parameters"].copy()))
        hypotheses.extend([("affine", affine), ("affine_ncc", data["initial_parameters"])])
        source_hypothesis_names = [h[0] for h in hypotheses]
        operation = initialize_with_escalation if args.escalation else initialize
        valid = np.asarray(solver.roi.selected, dtype=bool).reshape(-1) if roi_mask is not None else np.ones(count, bool)
        result = operation(centres, shape, valid, hypotheses, solve,
                           sparse_stride=args.sparse_stride, tau_u=args.tau_u, tau_g=args.tau_g,
                           minimum_anchors=args.minimum_anchors, batch_confirmation=args.batch_anchor_confirmation,
                           propagator=None if args.gpu_anchor_frontier else propagation,
                           anchor_propagator=propagation if args.gpu_anchor_frontier else None,
                           hypothesis_selector=gpu_select if args.gpu_hypothesis_selection else None)
        lazy_rotation_wall_s = 0.0
        lazy_rotation_generated = False
        lazy_rotation_cache_hit = False
        if args.lazy_rotation_weights and not result["minimum_anchors_met"]:
            lazy_directory = args.lazy_rotation_output / f"{mode}_{texture}_{warp}"
            model_size = source_model_size(source_protocol)
            command = [sys.executable, str(Path(__file__).with_name("generate_rotation_seed_hypotheses.py")),
                       "--ranking-reports", str(reports[0]), "--family", mode, "--textures", str(texture),
                       "--weights", str(args.lazy_rotation_weights), "--live-dic-source", str(args.live_dic_source),
                       "--output-dir", str(lazy_directory),
                       "--model-width", str(model_size[0]), "--model-height", str(model_size[1])]
            if args.ndf_onnx_webgpu_cache:
                command.extend(("--ndf-onnx-webgpu-cache", str(args.ndf_onnx_webgpu_cache)))
            if mode == "shear_band":
                command.extend(("--shear-band-peak-gradient", str(args.shear_band_peak_gradient)))
            for option in ("discontinuity_boundary_fraction", "speckle_noise_std", "target_gain", "target_offset",
                           "target_blur_sigma", "target_noise_std", "periodic_strength", "periodic_period_px",
                           "external_reference", "external_target"):
                value = getattr(args, option)
                if value is not None:
                    command.extend(("--" + option.replace("_", "-"), str(value)))
            for option in ("external_stretch_x", "external_stretch_y"):
                if source_protocol.get(option) is not None:
                    command.extend(("--" + option.replace("_", "-"), str(source_protocol[option])))
            fingerprint = {"weights_sha256": file_sha256(args.lazy_rotation_weights),
                           "ranking_report_sha256": file_sha256(reports[0]),
                           "generator_sha256": file_sha256(Path(__file__).with_name("generate_rotation_seed_hypotheses.py")),
                           "qualifier_sha256": file_sha256(Path(__file__).with_name("qualify_ndf_pair_icgn.py")),
                           "arguments": command[2:]}
            original_weights = source_protocol.get("training_weights")
            if (original_weights and Path(original_weights).is_file()
                    and file_sha256(original_weights) != fingerprint["weights_sha256"]):
                raise ValueError("lazy rotation weights differ from the original affine inputs")
            for option in ("external_reference", "external_target"):
                value = getattr(args, option)
                if value is not None:
                    fingerprint[option + "_sha256"] = file_sha256(value)
            marker = check_lazy_cache(lazy_directory, fingerprint)
            before_rotation = time.perf_counter()
            if marker.exists():
                if not (lazy_directory / "manifest.json").is_file():
                    raise ValueError("verified lazy rotation cache has no manifest")
                lazy_rotation_cache_hit = True
            else:
                subprocess.run(command, check=True)
                marker.write_text(json.dumps(fingerprint, indent=2) + "\n")
            lazy_rotation_wall_s = time.perf_counter() - before_rotation
            pair_rotations = rotations_for_pair(json.loads((lazy_directory / "manifest.json").read_text()),
                                                texture, warp, mode)
            if sorted(r["rank"] for r in pair_rotations) != [1, 2, 3]:
                raise ValueError("lazy rotation did not produce exactly three hypotheses")
            lazy_rotation_generated = True
            for rotation in sorted(pair_rotations, key=lambda r: r["rank"]):
                with np.load(rotation["windows"]) as rotation_data:
                    if not np.array_equal(rotation_data["centres_xy_native"], centres):
                        raise ValueError("rotation hypothesis has a different window grid")
                    hypotheses.append((f"rotation_rank{rotation['rank']}", rotation_data["initial_parameters"].copy()))
            source_hypothesis_names = [h[0] for h in hypotheses]
            result = initialize_with_escalation(
                centres, shape, valid, hypotheses, solve, initial_result=result,
                sparse_stride=args.sparse_stride, tau_u=args.tau_u, tau_g=args.tau_g,
                minimum_anchors=args.minimum_anchors, batch_confirmation=args.batch_anchor_confirmation,
                propagator=None if args.gpu_anchor_frontier else propagation,
                           anchor_propagator=propagation if args.gpu_anchor_frontier else None,
                           hypothesis_selector=gpu_select if args.gpu_hypothesis_selection else None)
        local_fit_window_count = 0
        retry_unresolved = args.escalate_unresolved and np.count_nonzero(valid & ~result["accepted"]) >= 16
        if args.local_affine_validation or (args.local_affine_escalation and
                                            (not result["minimum_anchors_met"] or retry_unresolved)):
            from qualify_local_hypothesis_icgn import local_affine_seed_parameters
            from wgpu_matching import WGPUMatching

            matcher = WGPUMatching(device_type="gpu")
            try:
                local_indices = np.flatnonzero(valid if args.local_affine_validation else valid & ~result["accepted"])
                local_centres = centres[local_indices]
                local_fit_window_count = len(local_indices)
                for rotation in sorted(pair_rotations, key=lambda r: r["rank"]):
                    with np.load(rotation["windows"]) as rotation_data:
                        if "hard_match_pairs" not in rotation_data:
                            raise ValueError("local affine escalation requires freshly exported hard matches")
                        pairs = rotation_data["hard_match_pairs"].copy()
                    local_params = np.zeros((count, 6), dtype=np.float32)
                    local_valid = np.zeros(count, dtype=bool)
                    if len(local_indices):
                        field = matcher.fit_local_affine(
                            pairs, local_centres, radius_px=192., threshold_px=4., min_inliers=6,
                            min_candidates=8, min_inlier_fraction=.15, max_iterations=256,
                            include_affine_models=True,
                        )
                        subset_params, subset_valid = local_affine_seed_parameters(field, local_centres)
                        local_params[local_indices] = subset_params
                        local_valid[local_indices] = subset_valid
                    hypotheses.append((f"local_rotation_rank{rotation['rank']}", local_params, local_valid))
            finally:
                matcher.close()
            if args.local_affine_validation:
                candidates = [h for h in hypotheses if h[0].startswith("local_rotation_")]
                source_hypothesis_names = [h[0] for h in candidates]
                result = operation(centres, shape, valid, candidates, solve,
                                   sparse_stride=args.sparse_stride, tau_u=args.tau_u, tau_g=args.tau_g,
                                   minimum_anchors=args.minimum_anchors, batch_confirmation=args.batch_anchor_confirmation,
                                   propagator=None if args.gpu_anchor_frontier else propagation,
                           anchor_propagator=propagation if args.gpu_anchor_frontier else None,
                           hypothesis_selector=gpu_select if args.gpu_hypothesis_selection else None)
            else:
                source_hypothesis_names = [h[0] for h in hypotheses]
                result = initialize_with_escalation(
                    centres, shape, valid, hypotheses, solve, initial_result=result,
                    sparse_stride=args.sparse_stride, tau_u=args.tau_u, tau_g=args.tau_g,
                    minimum_anchors=args.minimum_anchors, fill_unresolved=args.escalate_unresolved,
                    batch_confirmation=args.batch_anchor_confirmation, propagator=None if args.gpu_anchor_frontier else propagation,
                           anchor_propagator=propagation if args.gpu_anchor_frontier else None,
                           hypothesis_selector=gpu_select if args.gpu_hypothesis_selection else None)
        elapsed = time.perf_counter() - start
        # Ground truth is accessed only after the hypotheses and propagation are fixed.
        truth_available = bool(data.get("truth_available", True))
        error = (np.linalg.norm(result["parameters"][:, [0, 3]]
                                - data["truth_parameters"][:, [0, 3]], axis=1)
                 if truth_available else np.full(count, np.nan))
        visible = data["visible"].astype(bool)
        accepted = result["accepted"]
        correct = accepted & visible & (error < 2)
        control_state = data["initial_parameters"].copy()
        control_status = np.zeros(count, dtype=bool)
        control_start = time.perf_counter()
        if args.same_budget_control:
            for indices in schedule:
                states, status = raw_solve(indices, control_state[indices])
                control_state[indices] = states
                control_status[indices] = status
        control_error = np.linalg.norm(control_state[:, [0, 3]]
                                       - data["truth_parameters"][:, [0, 3]], axis=1)
        rows.append({"pair": name, "selected": result["selected_hypothesis"],
                     "minimum_anchors_met": bool(result["minimum_anchors_met"]),
                     "lazy_rotation_requested": lazy_rotation_generated,
                     "lazy_rotation_cache_hit": lazy_rotation_cache_hit,
                     "lazy_rotation_hypotheses_generated": 3 if lazy_rotation_generated and not lazy_rotation_cache_hit else 0,
                     "lazy_rotation_wall_s": lazy_rotation_wall_s,
                     "source_hypothesis_names": source_hypothesis_names,
                     "local_affine_fit_window_count_per_rotation": local_fit_window_count,
                     "hypotheses": result["hypothesis_acceptances"],
                     "confirmed_seeds": int(result["confirmed_seeds"].sum()),
                     "truth_available": truth_available,
                     "visible": int(visible.sum()) if truth_available else None, "accepted": int(accepted.sum()),
                     "accepted_invisible": int((accepted & ~visible).sum()) if truth_available else None,
                     "correct": int(correct.sum()) if truth_available else None,
                     "false_accepted": int((accepted & visible & (error >= 2)).sum()) if truth_available else None,
                     "oracle": int(data["oracle_icgn_correct_lt2px"].sum()) if truth_available else None,
                     "baseline_correct": int(data["icgn_correct_lt2px"].sum()) if truth_available else None,
                     "passes": result["passes"], "propagation_solver_calls": result["solver_calls"],
                     "total_solver_calls_before_control": len(schedule) + sum(2 * s["counter_readbacks"] + s["anchor_solver_dispatches"] for s in frontier_stats)
                         + sum(s["solver_dispatches"] for s in selector_stats),
                     "total_active_windows_before_control": (None if args.gpu_anchor_frontier or args.gpu_hypothesis_selection else
                         sum(len(indices) for indices in schedule) + sum(s["active_windows"] for s in frontier_stats)),
                     "gpu_frontier_stages": frontier_stats,
                     "gpu_hypothesis_stages": selector_stats,
                     "control_correct": int((control_status & visible & (control_error < 2)).sum())
                     if args.same_budget_control and truth_available else None,
                     "control_solver_calls": len(schedule) if args.same_budget_control else 0,
                     "wall_s_excluding_hypothesis_generation": elapsed - lazy_rotation_wall_s,
                     "wall_s_including_lazy_rotation": elapsed,
                     "image_prepare_s": image_prepare_s,
                     "solver_reference_and_target_s": solver_reference_and_target_s,
                     "same_budget_control_s": time.perf_counter() - control_start,
                     "case_wall_s_before_output": time.perf_counter() - case_start,
                     "escalation_history": result.get("escalation_history", [])})
        np.savez_compressed(args.output_dir / f"{name}.npz", centres=centres, visible=visible,
                            truth_available=np.array(truth_available), valid_roi=valid,
                            error=error, correct=correct, **{k: result[k] for k in
                            ("parameters", "accepted", "parent", "origin", "generation",
                             "attempts", "rejection_reason", "confirmed_seeds")},
                            **({"control_parameters": control_state, "control_status": control_status}
                               if args.same_budget_control else {}),
                            **{k: result[k] for k in ("source_stage", "source_hypothesis") if k in result})
        (args.output_dir / f"{name}.json").write_text(json.dumps(rows[-1], indent=2) + "\n")
        print(f"Completed {name}", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"startup_from_main_s": startup_s, "protocol": vars(args) | {
        "archive": str(args.archive), "output_dir": str(args.output_dir),
        "live_dic_source": str(args.live_dic_source),
        "rotation_manifest": str(args.rotation_manifest) if args.rotation_manifest else None,
        "lazy_rotation_weights": str(args.lazy_rotation_weights) if args.lazy_rotation_weights else None,
        "lazy_rotation_output": str(args.lazy_rotation_output) if args.lazy_rotation_output else None,
        "ndf_onnx_webgpu_cache": str(args.ndf_onnx_webgpu_cache) if args.ndf_onnx_webgpu_cache else None,
        "frozen_parameters": str(args.frozen_parameters) if args.frozen_parameters else None,
        "external_reference": str(args.external_reference) if args.external_reference else None,
        "external_target": str(args.external_target) if args.external_target else None,
        "roi_mask": str(args.roi_mask) if args.roi_mask else None},
        "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()

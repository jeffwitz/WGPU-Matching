"""Native wgpu-py implementation of direct and hierarchical DeepMatching."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import wgpu

from ._gpu import (
    ResourceScope,
    create_buffer,
    create_texture,
    dispatch,
    make_bind_group,
    make_params,
    make_pipeline,
    read_array,
    shader_source,
)
from .api import DeepMatchingOptions

TILE_SIDE = 16
DESCRIPTOR_HALO = 4


class _NodeLevel(list[dict[str, Any]]):
    map_width = 0
    map_height = 0
    map_area = 0


class DeepMatchingBackend:
    """Run the INRIA-style dense hierarchy or direct full-patch correlation."""

    def __init__(self, device: Any) -> None:
        self.device = device

    def _shader(self, name: str, gray_format: str | None = None) -> Any:
        code = shader_source(name)
        if gray_format:
            code = code.replace("{{GRAY_STORAGE_FORMAT}}", gray_format)
        return self.device.create_shader_module(label=name, code=code)

    @staticmethod
    def _entry_buffer(binding: int, buffer: Any) -> dict[str, Any]:
        return {"binding": binding, "resource": {"buffer": buffer}}

    @staticmethod
    def _entry_view(binding: int, texture: Any) -> dict[str, Any]:
        return {"binding": binding, "resource": texture.create_view()}

    @staticmethod
    def _write_params(scope: ResourceScope, device: Any, values: list[int], label: str, length: int = 16) -> Any:
        return make_params(device, scope, values, label, length)

    def _texture(
        self,
        scope: ResourceScope,
        width: int,
        height: int,
        format: str,
        usage: int,
        label: str,
    ) -> Any:
        return create_texture(self.device, scope, width, height, format, usage, label)

    def _upload_gray(
        self,
        scope: ResourceScope,
        pixels_a: np.ndarray,
        pixels_b: np.ndarray,
    ) -> list[dict[str, Any]]:
        height, width = pixels_a.shape
        result = []
        for index, pixels in enumerate((pixels_a, pixels_b)):
            texture = self._texture(
                scope,
                width,
                height,
                "r8unorm",
                wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
                f"DeepMatching grayscale source {index}",
            )
            self.device.queue.write_texture(
                {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
                pixels,
                {"offset": 0, "bytes_per_row": width, "rows_per_image": height},
                (width, height, 1),
            )
            result.append({"width": width, "height": height, "texture": texture})
        return result

    def build_hog_descriptors(
        self,
        scope: ResourceScope,
        reference: np.ndarray,
        target: np.ndarray,
        downscale_levels: int,
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        """Build the shared INRIA-style HOG descriptor textures for a matcher backend."""
        height, width = reference.shape
        inputs = self._upload_gray(scope, reference, target)
        return self._descriptors(scope, inputs, width, height, downscale_levels, progress)

    def _descriptors(
        self,
        scope: ResourceScope,
        inputs: list[dict[str, Any]],
        full_width: int,
        full_height: int,
        downscale_levels: int,
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        scale = 2**downscale_levels
        width = full_width // scale
        height = full_height // scale
        if width < 1 or height < 1:
            raise ValueError("DeepMatching reduction exceeds the input image size")
        format = "r8unorm" if "texture-formats-tier1" in self.device.features else "rgba8unorm"
        storage_usage = wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING
        sets = []
        for label in ("Reference", "Target"):
            sets.append({
                "width": width,
                "height": height,
                "first": self._texture(scope, width, height, "rgba16float", storage_usage, label + " HOG 1-4"),
                "second": self._texture(scope, width, height, "rgba16float", storage_usage, label + " HOG 5-8"),
            })
        reference_set, target_set = sets

        descriptor_module = self._shader("deepmatching-descriptor.wgsl", format)
        descriptor_pipelines = {
            name: make_pipeline(self.device, descriptor_module, name, "DeepMatching " + name)
            for name in (
                "presmooth_horizontal", "presmooth_vertical", "build_hog", "smooth_horizontal",
                "smooth_vertical", "sigmoid_smooth_horizontal", "average_pool",
            )
        }
        image_config = self._write_params(scope, self.device, [full_width, full_height], "DeepMatching image config", 4)

        for image_index, source in enumerate(inputs):
            output = reference_set if image_index == 0 else target_set
            gray_horizontal = self._texture(scope, full_width, full_height, format, storage_usage,
                                            "DeepMatching pre-smooth horizontal")
            gray_smoothed = self._texture(scope, full_width, full_height, format, storage_usage,
                                          "DeepMatching pre-smoothed image")
            encoder = self.device.create_command_encoder(label="DeepMatching image presmoothing")
            horizontal = make_bind_group(self.device, descriptor_pipelines["presmooth_horizontal"], [
                self._entry_view(0, source["texture"]), self._entry_view(1, gray_horizontal),
                self._entry_buffer(2, image_config),
            ])
            vertical = make_bind_group(self.device, descriptor_pipelines["presmooth_vertical"], [
                self._entry_view(0, gray_horizontal), self._entry_view(1, gray_smoothed),
                self._entry_buffer(2, image_config),
            ])
            dispatch(encoder, descriptor_pipelines["presmooth_horizontal"], horizontal,
                     math.ceil(full_width / 16), math.ceil(full_height / 16), label="DeepMatching pre-smooth x")
            dispatch(encoder, descriptor_pipelines["presmooth_vertical"], vertical,
                     math.ceil(full_width / 16), math.ceil(full_height / 16), label="DeepMatching pre-smooth y")
            self.device.queue.submit([encoder.finish()])
            scope.release(gray_horizontal)

            work_side = TILE_SIDE * scale + DESCRIPTOR_HALO * 2
            tile_bytes = work_side * work_side * 8 * 4
            max_tiles_binding = self.device.limits["max-storage-buffer-binding-size"] // tile_bytes
            max_tiles_buffer = self.device.limits["max-buffer-size"] // tile_bytes
            max_groups = self.device.limits["max-compute-workgroups-per-dimension"]
            tiles_x, tiles_y = math.ceil(width / TILE_SIDE), math.ceil(height / TILE_SIDE)
            total_tiles = tiles_x * tiles_y
            groups_per_tile_axis = math.ceil(work_side / 8)
            if groups_per_tile_axis > max_groups:
                raise ValueError("DeepMatching HOG tile exceeds WebGPU dispatch limits")
            batch_size = min(32, total_tiles, max_tiles_binding, max_tiles_buffer, max_groups)
            if batch_size < 1:
                raise ValueError("DeepMatching HOG tile scratch buffer exceeds device limits")
            scratch_bytes = batch_size * tile_bytes
            scratch_a = create_buffer(self.device, scope, scratch_bytes, wgpu.BufferUsage.STORAGE,
                                      "DeepMatching HOG scratch A")
            scratch_b = create_buffer(self.device, scope, scratch_bytes, wgpu.BufferUsage.STORAGE,
                                      "DeepMatching HOG scratch B")

            for batch_start in range(0, total_tiles, batch_size):
                actual = min(batch_size, total_tiles - batch_start)
                config = self._write_params(scope, self.device, [
                    full_width, full_height, width, height, scale, TILE_SIDE, work_side,
                    DESCRIPTOR_HALO, batch_start, actual, tiles_x, total_tiles,
                ], "DeepMatching HOG tile parameters")
                encoder = self.device.create_command_encoder(label="DeepMatching HOG tile batch")
                build_group = make_bind_group(self.device, descriptor_pipelines["build_hog"], [
                    self._entry_view(3, gray_smoothed), self._entry_buffer(4, scratch_a),
                    self._entry_buffer(6, config),
                ])
                smooth_h = make_bind_group(self.device, descriptor_pipelines["smooth_horizontal"], [
                    self._entry_buffer(4, scratch_a), self._entry_buffer(5, scratch_b),
                    self._entry_buffer(6, config),
                ])
                smooth_v = make_bind_group(self.device, descriptor_pipelines["smooth_vertical"], [
                    self._entry_buffer(4, scratch_a), self._entry_buffer(5, scratch_b),
                    self._entry_buffer(6, config),
                ])
                sigmoid_h = make_bind_group(self.device, descriptor_pipelines["sigmoid_smooth_horizontal"], [
                    self._entry_buffer(4, scratch_a), self._entry_buffer(5, scratch_b),
                    self._entry_buffer(6, config),
                ])
                pool = make_bind_group(self.device, descriptor_pipelines["average_pool"], [
                    self._entry_buffer(4, scratch_a), self._entry_buffer(6, config),
                    self._entry_view(7, output["first"]), self._entry_view(8, output["second"]),
                ])
                tile_groups = (math.ceil(work_side / 8), math.ceil(work_side / 8), actual)
                dispatch(encoder, descriptor_pipelines["build_hog"], build_group, *tile_groups,
                         label="DeepMatching pixel HOG")
                dispatch(encoder, descriptor_pipelines["smooth_horizontal"], smooth_h, *tile_groups,
                         label="DeepMatching HOG smooth horizontal")
                dispatch(encoder, descriptor_pipelines["smooth_vertical"], smooth_v, *tile_groups,
                         label="DeepMatching HOG smooth vertical")
                dispatch(encoder, descriptor_pipelines["sigmoid_smooth_horizontal"], sigmoid_h, *tile_groups,
                         label="DeepMatching HOG sigmoid")
                dispatch(encoder, descriptor_pipelines["smooth_vertical"], smooth_v, *tile_groups,
                         label="DeepMatching HOG post smooth")
                dispatch(encoder, descriptor_pipelines["average_pool"], pool,
                         math.ceil(TILE_SIDE / 8), math.ceil(TILE_SIDE / 8), actual,
                         label="DeepMatching descriptor average pool")
                self.device.queue.submit([encoder.finish()])
                progress(f"DeepMatching HOG image {image_index + 1}/2: tiles {min(batch_start + actual, total_tiles)}/{total_tiles}")
                scope.release(config)

            scope.release(gray_smoothed)
            scope.release(scratch_a)
            scope.release(scratch_b)
            scope.release(source["texture"])
        scope.release(image_config)
        return {"width": width, "height": height, "reference": reference_set, "target": target_set}

    def _direct(
        self,
        scope: ResourceScope,
        descriptors: dict[str, Any],
        width: int,
        height: int,
        settings: DeepMatchingOptions,
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        scale = 2**settings.downscale_levels
        scaled_patch = settings.patch_size / scale
        cells_side = scaled_patch / 4
        if not scaled_patch.is_integer() or not cells_side.is_integer() or cells_side < 1 or int(cells_side) & (int(cells_side) - 1):
            raise ValueError("Reduced patch must contain 1, 2, 4, 8, 16 or 32 cells per side")
        if settings.patch_size % scale or (not settings.seed_points and settings.stride % scale):
            raise ValueError("Patch size and grid stride must be divisible by the image reduction")
        manual = bool(settings.seed_points)
        half_patch = (settings.patch_size - 1) / 2
        manual_tl = []
        if manual:
            for index, (x, y) in enumerate(settings.seed_points or ()):
                px = round(x - half_patch)
                py = round(y - half_patch)
                px = round(px / scale) * scale
                py = round(py / scale) * scale
                if px < 0 or py < 0 or px + settings.patch_size > width or py + settings.patch_size > height:
                    raise ValueError(f"Seed {index + 1}: patch falls outside the image")
                manual_tl.append((px, py))
        grid_width = len(manual_tl) if manual else 1 + (width - settings.patch_size) // settings.stride
        grid_height = 1 if manual else 1 + (height - settings.patch_size) // settings.stride
        if grid_width < 1 or grid_height < 1:
            raise ValueError("Patch is larger than the image")
        query_count = grid_width * grid_height
        if query_count > 5120:
            raise ValueError("DeepMatching supports at most 5120 requested seeds")
        sources = manual_tl if manual else [
            (column * settings.stride, row * settings.stride)
            for row in range(grid_height) for column in range(grid_width)
        ]
        desc_w, desc_h = descriptors["width"], descriptors["height"]
        max_x, max_y = desc_w - int(scaled_patch) + 1, desc_h - int(scaled_patch) + 1
        radius = math.floor(settings.search_radius_px / scale) if settings.search_radius_px > 0 else 0
        cand_w = min(max_x, 2 * radius + 1) if settings.search_radius_px > 0 else max_x
        cand_h = min(max_y, 2 * radius + 1) if settings.search_radius_px > 0 else max_y
        map_area = cand_w * cand_h
        estimate = query_count * map_area * int(scaled_patch) ** 2
        if estimate > 600_000_000:
            raise ValueError(f"Direct patch correlation would require {estimate // 1_000_000} million descriptor comparisons")
        max_batch = self.device.limits["max-storage-buffer-binding-size"] // (map_area * 4)
        batch_size = min(8, query_count, max_batch)
        if batch_size < 1:
            raise ValueError("Direct correlation map exceeds the device storage-buffer limit")
        maps = create_buffer(self.device, scope, map_area * batch_size * 4,
                             wgpu.BufferUsage.STORAGE, "DeepMatching direct response maps")
        queries = create_buffer(self.device, scope, batch_size * 16,
                                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
                                "DeepMatching direct queries")
        results = create_buffer(self.device, scope, batch_size * 16,
                                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                                "DeepMatching direct results")
        params = self._write_params(scope, self.device, [
            cand_w, cand_h, desc_w, desc_h, int(scaled_patch), batch_size, map_area, radius,
        ], "DeepMatching direct correlation parameters")
        matcher_module = self._shader("deepmatching-match.wgsl")
        correlation = make_pipeline(self.device, matcher_module, "correlate_patches", "DeepMatching patch correlation")
        selection = make_pipeline(self.device, matcher_module, "select_best_patch", "DeepMatching patch argmax")
        group_correlate = make_bind_group(self.device, correlation, [
            self._entry_view(0, descriptors["reference"]["first"]),
            self._entry_view(1, descriptors["target"]["first"]),
            self._entry_view(2, descriptors["reference"]["second"]),
            self._entry_view(3, descriptors["target"]["second"]),
            self._entry_buffer(6, queries), self._entry_buffer(7, maps),
            self._entry_buffer(8, params),
        ])
        selection_params = self._write_params(scope, self.device,
                                               [cand_w, cand_h, 0, map_area, batch_size, 4],
                                               "DeepMatching direct selection parameters")
        group_select = make_bind_group(self.device, selection, [
            self._entry_buffer(7, maps), self._entry_buffer(8, selection_params),
            self._entry_buffer(9, results),
        ])
        matches: list[dict[str, Any] | None] = [None] * query_count
        for batch, first in enumerate(range(0, query_count, batch_size)):
            actual = min(batch_size, query_count - first)
            query_values = np.zeros((batch_size, 4), dtype=np.uint32)
            starts = []
            for local in range(actual):
                sx, sy = sources[first + local]
                source_x, source_y = sx // scale, sy // scale
                start_x = max(0, min(max_x - cand_w, source_x - cand_w // 2))
                start_y = max(0, min(max_y - cand_h, source_y - cand_h // 2))
                query_values[local] = [source_x, source_y, start_x, start_y]
                starts.append((start_x, start_y))
            self.device.queue.write_buffer(queries, 0, query_values)
            encoder = self.device.create_command_encoder(label=f"DeepMatching direct batch {batch + 1}")
            dispatch(encoder, correlation, group_correlate, math.ceil(cand_w / 8),
                     math.ceil(cand_h / 8), batch_size, label="DeepMatching full patch correlations")
            dispatch(encoder, selection, group_select, batch_size, label="DeepMatching best and second patch")
            self.device.queue.submit([encoder.finish()])
            raw = read_array(self.device, results, np.dtype(np.float32), batch_size * 4).reshape(-1, 4)
            for local in range(actual):
                sx, sy = sources[first + local]
                map_x, map_y, score, second_score = map(float, raw[local])
                start_x, start_y = starts[local]
                tx, ty = (map_x + start_x) * scale, (map_y + start_y) * scale
                dx, dy = round(tx - sx, 2), round(ty - sy, 2)
                reduced_x, reduced_y = math.floor(sx / scale), math.floor(sy / scale)
                at_boundary = settings.search_radius_px > 0 and (
                    abs(map_x + start_x - reduced_x) >= radius - 0.51
                    or abs(map_y + start_y - reduced_y) >= radius - 0.51
                )
                margin = score - second_score if second_score >= 0 else 0
                support = tx >= 2 and ty >= 2 and tx + settings.patch_size <= width - 2 and ty + settings.patch_size <= height - 2
                usable = support and not at_boundary and second_score >= 0 and score >= settings.minimum_score and margin >= settings.minimum_margin
                matches[first + local] = {
                    "x": sx + (settings.patch_size - 1) / 2,
                    "y": sy + (settings.patch_size - 1) / 2,
                    "dx": dx, "dy": dy, "score": score, "second_score": second_score,
                    "margin": margin, "usable": bool(usable),
                    "seed_x": sx + (settings.patch_size - 1) / 2,
                    "seed_y": sy + (settings.patch_size - 1) / 2,
                    "reciprocal": None,
                }
            progress(f"DeepMatching direct: batch {batch + 1}/{math.ceil(query_count / batch_size)}")
        return {
            "algorithm": "DeepMatching-wgpu-py-direct",
            "width": width, "height": height, "patch_size": settings.patch_size,
            "stride": settings.stride, "scale": scale,
            "grid_width": grid_width, "grid_height": grid_height,
            "matches": matches, "estimated_descriptor_comparisons": estimate,
            "matcher_mode": "direct-patch-correlation",
            "seed_mode": "manual" if manual else "grid",
            "search_radius_px": settings.search_radius_px,
        }

    @staticmethod
    def _build_seed_pyramid(seed: tuple[float, float], source_width: int, source_height: int,
                            scale: int, patch_size: int, forced_root: tuple[int, int] | None = None) -> dict[str, Any]:
        atom_size = 4 * scale
        patch_reduced = patch_size / scale
        cells_side = patch_reduced / 4
        root_level = round(math.log2(cells_side))
        max_root_x = source_width // 4 * 4
        max_root_y = source_height // 4 * 4
        sx, sy = seed
        if forced_root is not None:
            root_x, root_y = forced_root
        elif root_level == 0:
            root_x = max(2, min(source_width - 2, 2 + round((sx / scale - 2) / 4) * 4))
            root_y = max(2, min(source_height - 2, 2 + round((sy / scale - 2) / 4) * 4))
        else:
            root_x = max(0, min(max_root_x, round(sx / atom_size) * 4))
            root_y = max(0, min(max_root_y, round(sy / atom_size) * 4))
        levels: list[_NodeLevel] = [_NodeLevel() for _ in range(root_level + 1)]

        def create_node(level: int, x: int, y: int) -> int:
            valid = (x >= 2 and y >= 2 and x + 2 <= source_width and y + 2 <= source_height
                     and x % 4 == 2 and y % 4 == 2) if level == 0 else (
                         x >= 0 and y >= 0 and x <= max_root_x and y <= max_root_y
                         and x % 4 == 0 and y % 4 == 0
                     )
            if not valid:
                return -1
            node = {"x": x, "y": y, "children": [-1, -1, -1, -1],
                    "norm": 1 if level == 0 else 0, "offset_x": 0, "offset_y": 0}
            index = len(levels[level])
            levels[level].append(node)
            if level:
                gap = 2**level
                slot = 0
                for dy in (-1, 1):
                    for dx in (-1, 1):
                        child = create_node(level - 1, x + dx * gap, y + dy * gap)
                        node["children"][slot] = child
                        slot += 1
                        if child >= 0:
                            node["norm"] += levels[level - 1][child]["norm"]
            return index

        if create_node(root_level, root_x, root_y) != 0:
            raise ValueError("INRIA seed pyramid has an invalid root")
        leaves = levels[0]
        closest_leaf = min(leaves, key=lambda node: (node["x"] * scale - sx) ** 2 + (node["y"] * scale - sy) ** 2,
                           default=None)
        return {"levels": levels, "root_x": root_x, "root_y": root_y,
                "root_level": root_level, "closest_leaf": closest_leaf, "atom_size": atom_size}

    @staticmethod
    def _root_lattice(root_level: int, source_width: int, source_height: int, root_spacing: int) -> list[tuple[int, int]]:
        max_x, max_y = source_width // 4 * 4, source_height // 4 * 4
        first, step = (2, 4) if root_level == 0 else (0, root_spacing)
        stop_x = max_x - 1 if root_level == 0 else max_x + 1
        stop_y = max_y - 1 if root_level == 0 else max_y + 1
        return [(x, y)
                for y in range(first, stop_y, step)
                for x in range(first, stop_x, step)]

    @staticmethod
    def _build_paths(levels: list[list[dict[str, Any]]]) -> np.ndarray:
        root_level = len(levels) - 1
        paths = np.zeros((len(levels[0]) * root_level, 4), dtype=np.int32)
        if root_level == 0 or not len(levels[0]):
            return paths.reshape(-1)
        path: list[list[int] | None] = [None] * root_level

        def visit(level: int, node_index: int) -> None:
            if level == 0:
                for step in range(root_level):
                    paths[node_index * root_level + step] = path[step]
                return
            parent = levels[level][node_index]
            for slot, child_index in enumerate(parent["children"]):
                if child_index < 0:
                    continue
                child = levels[level - 1][child_index]
                path[root_level - level] = [child_index, slot, child["offset_x"], child["offset_y"]]
                visit(level - 1, child_index)

        visit(root_level, 0)
        return paths.reshape(-1)

    def _storage_array(self, scope: ResourceScope, values: np.ndarray, label: str,
                       usage: int = wgpu.BufferUsage.STORAGE) -> Any:
        values = np.ascontiguousarray(values)
        size = max(16, values.nbytes)
        buffer = create_buffer(self.device, scope, size, usage | wgpu.BufferUsage.COPY_DST, label)
        if values.nbytes:
            self.device.queue.write_buffer(buffer, 0, values)
        return buffer

    def _parent_metadata(self, scope: ResourceScope, lower: list[dict[str, Any]],
                         current: list[dict[str, Any]], label: str) -> dict[str, Any]:
        parent_count, child_count = len(current), len(lower)
        children = np.zeros((parent_count, 4), dtype=np.int32)
        parent_info = np.zeros((parent_count, 4), dtype=np.int32)
        weights = np.zeros((parent_count, 4), dtype=np.float32)
        border_factors = np.zeros((parent_count, 12), dtype=np.float32)
        child_info = np.zeros((child_count, 4), dtype=np.int32)
        for index, child in enumerate(lower):
            child_info[index] = [child["offset_x"] // 2, child["offset_y"] // 2,
                                 child["offset_x"] % 2, child["offset_y"] % 2]
        for parent_index, parent in enumerate(current):
            children[parent_index] = parent["children"]
            available = []
            sum_x = sum_y = 0
            for slot, child_index in enumerate(parent["children"]):
                if child_index < 0:
                    continue
                dx, dy = (slot % 2) * 2 - 1, (slot // 2) * 2 - 1
                child = lower[child_index]
                ox, oy = child["offset_x"] // 2, child["offset_y"] // 2
                available.append((slot, dx, dy, ox, oy, child))
                sum_x += ox - dx
                sum_y += oy - dy
            if available:
                parent["offset_x"] = math.floor(0.5 + sum_x / len(available))
                parent["offset_y"] = math.floor(0.5 + sum_y / len(available))
            parent_info[parent_index, :2] = [parent["offset_x"], parent["offset_y"]]
            sums = np.zeros((3, 3), dtype=np.float64)
            for slot, dx, dy, ox, oy, child in available:
                tx = dx + parent["offset_x"] - ox
                ty = dy + parent["offset_y"] - oy
                weight = child["norm"] / parent["norm"] if parent["norm"] > 0 else 0.0
                weights[parent_index, slot] = weight
                for ry in (-1, 0, 1):
                    for rx in (-1, 0, 1):
                        if rx * tx <= 0 and ry * ty <= 0:
                            sums[ry + 1, rx + 1] += weight
            for ry in range(3):
                for rx in range(3):
                    total = sums[ry, rx]
                    border_factors[parent_index, ry * 4 + rx] = 1.0 / total**0.9 if total > 0 else 0.0
        return {
            "children": self._storage_array(scope, children, label + " children"),
            "parent_info": self._storage_array(scope, parent_info, label + " parent offsets"),
            "weights": self._storage_array(scope, weights, label + " child weights"),
            "border": self._storage_array(scope, border_factors, label + " border factors"),
            "child_info": self._storage_array(scope, child_info, label + " pooled child offsets"),
        }

    @staticmethod
    def _roots_for_sparse_seeds(
        seeds: list[tuple[float, float]],
        roots: list[tuple[int, int]],
        source_width: int,
        source_height: int,
        scale: int,
        root_patch: int,
        root_level: int,
    ) -> list[tuple[int, int]]:
        """Select hierarchy roots whose atomic leaves contain requested seeds."""
        if root_level == 0:
            spacing, half_support = 4, 0
        else:
            spacing = root_patch // scale
            # The hierarchy leaf offsets span one root tile minus one atom edge.
            half_support = spacing // 2 - 2

        available = set(roots)
        selected: set[tuple[int, int]] = set()
        for seed_x, seed_y in seeds:
            leaf_x = max(2, min(source_width - 2, 2 + round((seed_x / scale - 2) / 4) * 4))
            leaf_y = max(2, min(source_height - 2, 2 + round((seed_y / scale - 2) / 4) * 4))
            if root_level == 0:
                root = (leaf_x, leaf_y)
                if root not in available:
                    raise ValueError("Sparse DeepMatching seed has no matching atomic root")
                selected.add(root)
                continue

            root = (round(leaf_x / spacing) * spacing, round(leaf_y / spacing) * spacing)
            if (
                root not in available
                or abs(leaf_x - root[0]) > half_support
                or abs(leaf_y - root[1]) > half_support
            ):
                matches = [
                    candidate
                    for candidate in roots
                    if abs(leaf_x - candidate[0]) <= half_support
                    and abs(leaf_y - candidate[1]) <= half_support
                ]
                if not matches:
                    raise ValueError("Sparse DeepMatching seed is not covered by the root lattice")
                root = min(matches, key=lambda item: (item[0] - leaf_x) ** 2 + (item[1] - leaf_y) ** 2)
            selected.add(root)

        return [root for root in roots if root in selected]

    def _inria(
        self,
        scope: ResourceScope,
        descriptors: dict[str, Any],
        width: int,
        height: int,
        settings: DeepMatchingOptions,
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        scale = 2**settings.downscale_levels
        root_patch = settings.root_patch_size or settings.patch_size
        scaled_root = root_patch / scale
        cells_side = scaled_root / 4
        if not scaled_root.is_integer() or not cells_side.is_integer() or cells_side < 1 or int(cells_side) & (int(cells_side) - 1):
            raise ValueError("Reduced root size must contain 1, 2, 4, 8, 16 or 32 cells per side")
        if root_patch % scale:
            raise ValueError("Root patch size must be divisible by the reduction factor")
        sparse = settings.mode == "inria_sparse"
        if sparse and not settings.seed_points:
            raise ValueError("mode='inria_sparse' requires one or more explicit seed_points")
        manual = bool(settings.seed_points)
        if manual:
            seeds = list(settings.seed_points or ())
        else:
            grid_w = max(0, 1 + (width - settings.patch_size) // settings.stride)
            grid_h = max(0, 1 + (height - settings.patch_size) // settings.stride)
            seeds = [(column * settings.stride + (settings.patch_size - 1) / 2,
                      row * settings.stride + (settings.patch_size - 1) / 2)
                     for row in range(grid_h) for column in range(grid_w)]
        if not seeds:
            raise ValueError("No DeepMatching seed fits in the image")
        if len(seeds) > 512:
            raise ValueError("The INRIA-style mode supports at most 512 seeds")
        for index, (x, y) in enumerate(seeds):
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                raise ValueError(f"Seed {index + 1}: center must lie inside the image")
        descriptor_width, descriptor_height = descriptors["width"], descriptors["height"]
        source_width, source_height = descriptor_width // 4 * 4, descriptor_height // 4 * 4
        target_width, target_height = descriptor_width + 1, descriptor_height + 1
        reduced_radius = math.floor(settings.search_radius_px / scale) if settings.search_radius_px > 0 else 0
        map_width = min(target_width, 2 * reduced_radius) if reduced_radius > 0 else target_width
        map_height = min(target_height, 2 * reduced_radius) if reduced_radius > 0 else target_height
        map_area = map_width * map_height
        root_level = round(math.log2(cells_side))
        roots = self._root_lattice(root_level, source_width, source_height, root_patch // scale)
        global_root_count = len(roots)
        if sparse:
            roots = self._roots_for_sparse_seeds(
                seeds, roots, source_width, source_height, scale, root_patch, root_level
            )
        if not roots or global_root_count > 8192:
            raise ValueError(f"Invalid INRIA root lattice ({global_root_count} roots, limit 8192)")
        max_buf = min(self.device.limits["max-buffer-size"], self.device.limits["max-storage-buffer-binding-size"])

        hierarchy_module = self._shader("deepmatching-hierarchy.wgsl")
        atom_pipeline = make_pipeline(self.device, hierarchy_module, "correlate_atomic", "INRIA atom correlation")
        aggregate_pipeline = make_pipeline(self.device, hierarchy_module, "aggregate_parent", "INRIA hierarchy aggregation")
        init_pipeline = make_pipeline(self.device, hierarchy_module, "initialize_topdown", "INRIA top-down init")
        descend_pipeline = make_pipeline(self.device, hierarchy_module, "descend_topdown_one_level", "INRIA top-down descent")

        source_best: dict[str, dict[str, Any]] = {}
        target_best: dict[str, dict[str, Any]] = {}
        descriptor_comparisons = 0
        root_batch_size = 2
        for batch_start in range(0, len(roots), root_batch_size):
            batch_end = min(len(roots), batch_start + root_batch_size)
            batch_scope = ResourceScope()
            encoder = self.device.create_command_encoder(label=f"INRIA root batch {batch_start}:{batch_end}")
            jobs = []
            for root_index in range(batch_start, batch_end):
                root_center = roots[root_index]
                pyramid = self._build_seed_pyramid((root_center[0] * scale, root_center[1] * scale),
                                                   source_width, source_height, scale, root_patch, root_center)
                levels = pyramid["levels"]
                atom_count = len(levels[0])
                descriptor_comparisons += atom_count * map_area * 16
                levels[0].map_width, levels[0].map_height, levels[0].map_area = map_width, map_height, map_area
                for level in range(1, len(levels)):
                    lower = levels[level - 1]
                    current = levels[level]
                    current.map_width = max(5, (lower.map_width + 2) // 2)
                    current.map_height = max(5, (lower.map_height + 2) // 2)
                    current.map_area = current.map_width * current.map_height
                level_bytes = [max(1, len(level)) * level.map_area * 4 for level in levels]
                if any(size > max_buf for size in level_bytes):
                    raise ValueError("An INRIA response map exceeds the device buffer limit")

                queries = np.zeros((max(1, atom_count), 4), dtype=np.uint32)
                for atom_index, atom in enumerate(levels[0]):
                    valid = atom["x"] >= 2 and atom["y"] >= 2 and atom["x"] + 2 <= source_width and atom["y"] + 2 <= source_height
                    source_x = atom["x"] - 2 if valid else 0xFFFFFFFF
                    source_y = atom["y"] - 2 if valid else 0xFFFFFFFF
                    left = max(0, min(atom["x"] - reduced_radius, target_width - 2 * reduced_radius)) if reduced_radius else 0
                    top = max(0, min(atom["y"] - reduced_radius, target_height - 2 * reduced_radius)) if reduced_radius else 0
                    queries[atom_index] = [source_x, source_y, left, top]
                    atom["offset_x"], atom["offset_y"] = left, top
                atom_map = create_buffer(self.device, batch_scope, max(4, max(1, atom_count) * map_area * 4),
                                         wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                                         "INRIA atom response maps")
                query_buffer = self._storage_array(batch_scope, queries, "INRIA atom queries")
                atom_params = self._write_params(batch_scope, self.device,
                    [map_width, map_height, descriptor_width, descriptor_height, atom_count, map_area, reduced_radius, 0],
                    "INRIA atom parameters")
                atom_group = make_bind_group(self.device, atom_pipeline, [
                    self._entry_view(0, descriptors["reference"]["first"]),
                    self._entry_view(1, descriptors["target"]["first"]),
                    self._entry_view(2, descriptors["reference"]["second"]),
                    self._entry_view(3, descriptors["target"]["second"]),
                    self._entry_buffer(6, query_buffer), self._entry_buffer(7, atom_map),
                    self._entry_buffer(8, atom_params),
                ])
                if atom_count:
                    dispatch(encoder, atom_pipeline, atom_group, math.ceil(map_width / 8),
                             math.ceil(map_height / 8), atom_count, label="INRIA atom maps")
                maps = [atom_map]
                for level in range(1, len(levels)):
                    lower, current = levels[level - 1], levels[level]
                    parent_buffer = create_buffer(self.device, batch_scope, level_bytes[level],
                        wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC, "INRIA parent maps")
                    maps.append(parent_buffer)
                    metadata = self._parent_metadata(batch_scope, lower, current, f"INRIA level {level}")
                    params = self._write_params(batch_scope, self.device, [
                        lower.map_width, lower.map_height, (lower.map_width + 2) // 2,
                        (lower.map_height + 2) // 2, len(current), current.map_area,
                        lower.map_area, current.map_width, current.map_height,
                    ], "INRIA aggregate params")
                    group = make_bind_group(self.device, aggregate_pipeline, [
                        self._entry_buffer(8, params), self._entry_buffer(10, maps[level - 1]),
                        self._entry_buffer(11, parent_buffer), self._entry_buffer(12, metadata["children"]),
                        self._entry_buffer(13, metadata["child_info"]), self._entry_buffer(14, metadata["parent_info"]),
                        self._entry_buffer(15, metadata["weights"]), self._entry_buffer(16, metadata["border"]),
                    ])
                    if len(current):
                        dispatch(encoder, aggregate_pipeline, group, math.ceil(current.map_width / 8),
                                 math.ceil(current.map_height / 8), len(current), label=f"INRIA aggregate {level}")

                paths = self._build_paths(levels)
                paths_buffer = self._storage_array(batch_scope, paths, "INRIA atomic paths")
                root_area = levels[-1].map_area
                candidate_count = atom_count * root_area
                candidate_bytes = max(16, candidate_count * 16)
                if candidate_bytes > max_buf:
                    raise ValueError("INRIA top-down states exceed the device buffer limit")
                states = create_buffer(self.device, batch_scope, candidate_bytes,
                                       wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                                       "INRIA top-down states")
                root = levels[-1][0]
                init_params = self._write_params(batch_scope, self.device, [
                    levels[-1].map_width, levels[-1].map_height, atom_count, root_area,
                    candidate_count, root["offset_x"], root["offset_y"],
                ], "INRIA top-down init params")
                init_group = make_bind_group(self.device, init_pipeline, [
                    self._entry_buffer(8, init_params), self._entry_buffer(20, maps[-1]),
                    self._entry_buffer(22, states),
                ])
                if candidate_count:
                    dispatch(encoder, init_pipeline, init_group, math.ceil(candidate_count / 64),
                             label="INRIA top-down root init")
                for level in range(len(levels) - 1, 0, -1):
                    lower = levels[level - 1]
                    step = root_level - level
                    descend_params = self._write_params(batch_scope, self.device, [
                        lower.map_width, lower.map_height, lower.map_area, atom_count,
                        candidate_count, step, root_level,
                    ], "INRIA top-down descent params")
                    descend_group = make_bind_group(self.device, descend_pipeline, [
                        self._entry_buffer(8, descend_params), self._entry_buffer(21, paths_buffer),
                        self._entry_buffer(22, states), self._entry_buffer(24, maps[level - 1]),
                    ])
                    dispatch(encoder, descend_pipeline, descend_group, math.ceil(candidate_count / 64),
                             label=f"INRIA top-down descent {level}")
                jobs.append({"pyramid": pyramid, "levels": levels, "atom_count": atom_count,
                             "candidate_count": candidate_count, "states": states})

            self.device.queue.submit([encoder.finish()])
            for job in jobs:
                raw = self.device.queue.read_buffer(job["states"], 0, max(16, job["candidate_count"] * 16))
                values = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
                indices = np.frombuffer(raw, dtype=np.int32).reshape(-1, 4)
                pyramid, levels, atom_count = job["pyramid"], job["levels"], job["atom_count"]
                root_x, root_y = pyramid["root_x"] * scale, pyramid["root_y"] * scale
                for candidate_idx in range(job["candidate_count"]):
                    score = float(values[candidate_idx, 2])
                    if score <= 0:
                        continue
                    atom_idx = candidate_idx % atom_count
                    atom = levels[0][atom_idx]
                    x0, y0 = atom["x"] * scale, atom["y"] * scale
                    x1, y1 = int(indices[candidate_idx, 0]) * scale, int(indices[candidate_idx, 1]) * scale
                    source_key = f"{y0 // (4 * scale)}:{x0 // (4 * scale)}"
                    target_key = f"{y1 // (4 * scale)}:{x1 // (4 * scale)}"
                    previous_source, previous_target = source_best.get(source_key), target_best.get(target_key)
                    candidate = {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                 "score": score, "root_x": root_x, "root_y": root_y}
                    if previous_source is None or score > previous_source["score"]:
                        source_best[source_key] = candidate
                    if previous_target is None or score > previous_target["score"]:
                        target_best[target_key] = candidate
            batch_scope.close()
            progress(f"DeepMatching INRIA roots {batch_end}/{len(roots)}")

        matches = []
        for seed_x, seed_y in seeds:
            leaf_x = max(2, min(source_width - 2, 2 + round((seed_x / scale - 2) / 4) * 4))
            leaf_y = max(2, min(source_height - 2, 2 + round((seed_y / scale - 2) / 4) * 4))
            source_x, source_y = leaf_x * scale, leaf_y * scale
            source_key = f"{source_y // (4 * scale)}:{source_x // (4 * scale)}"
            candidate = source_best.get(source_key)
            target_key = (f"{candidate['y1'] // (4 * scale)}:{candidate['x1'] // (4 * scale)}"
                          if candidate else None)
            reciprocal = target_best.get(target_key) if target_key is not None else None
            is_reciprocal = bool(candidate and reciprocal and all(
                candidate[key] == reciprocal[key] for key in ("x0", "y0", "x1", "y1", "score")
            ))
            matches.append({
                "x": source_x, "y": source_y,
                "dx": candidate["x1"] - candidate["x0"] if is_reciprocal else 0,
                "dy": candidate["y1"] - candidate["y0"] if is_reciprocal else 0,
                "score": candidate["score"] if candidate else 0,
                "second_score": None, "margin": None, "usable": is_reciprocal,
                "reciprocal": is_reciprocal,
                "target_x": candidate["x1"] if is_reciprocal else None,
                "target_y": candidate["y1"] if is_reciprocal else None,
                "seed_x": seed_x, "seed_y": seed_y,
                "root_x": candidate["root_x"] if candidate else round(leaf_x / (root_patch / scale)) * root_patch,
                "root_y": candidate["root_y"] if candidate else round(leaf_y / (root_patch / scale)) * root_patch,
            })
        return {
            "algorithm": "DeepMatching-wgpu-py-INRIA-sparse" if sparse else "DeepMatching-wgpu-py-INRIA",
            "width": width, "height": height, "patch_size": settings.patch_size,
            "stride": settings.stride, "scale": scale,
            "grid_width": len(seeds) if manual else 1 + (width - settings.patch_size) // settings.stride,
            "grid_height": 1 if manual else 1 + (height - settings.patch_size) // settings.stride,
            "matches": matches,
            "root_patch_size": root_patch,
            "root_center_convention": "complete INRIA non-overlapping root grid at root patch stride",
            "evaluated_root_count": len(roots), "global_root_count": global_root_count,
            "reciprocal_scope": "selected-roots" if sparse else "full-root-grid",
            "sparse_mode": sparse, "atomic_patch_size": 4 * scale,
            "levels": root_level + 1,
            "estimated_descriptor_comparisons": descriptor_comparisons,
            "matcher_mode": (
                "inria-sparse-selected-roots-hierarchical-reciprocal"
                if sparse else "inria-dense-root-grid-hierarchical-reciprocal"
            ),
            "seed_mode": "manual" if manual else "grid",
            "search_radius_px": settings.search_radius_px,
        }

    def compute(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        settings: DeepMatchingOptions,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        progress = on_progress or (lambda _: None)
        reference = np.ascontiguousarray(reference, dtype=np.uint8)
        target = np.ascontiguousarray(target, dtype=np.uint8)
        if reference.ndim != 2 or target.ndim != 2 or reference.shape != target.shape:
            raise ValueError("DeepMatching expects same-sized two-dimensional grayscale uint8 images")
        if settings.mode not in {"inria", "inria_sparse", "direct"}:
            raise ValueError("DeepMatching mode must be 'inria', 'inria_sparse' or 'direct'")
        if settings.downscale_levels < 0 or settings.downscale_levels > 8:
            raise ValueError("downscale_levels must be between zero and eight")
        height, width = reference.shape
        if width > self.device.limits["max-texture-dimension-2d"] or height > self.device.limits["max-texture-dimension-2d"]:
            raise ValueError("Image exceeds the adapter's maximum texture dimension")
        scope = ResourceScope()
        try:
            inputs = self._upload_gray(scope, reference, target)
            descriptors = self._descriptors(scope, inputs, width, height, settings.downscale_levels, progress)
            if settings.mode == "direct":
                return self._direct(scope, descriptors, width, height, settings, progress)
            return self._inria(scope, descriptors, width, height, settings, progress)
        finally:
            scope.close()

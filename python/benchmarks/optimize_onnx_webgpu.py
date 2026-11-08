"""Small optional ONNX graph rewrites for the native WebGPU EP experiments."""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path
from typing import Any


def decompose_selu(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Replace ONNX Selu nodes by primitive ops supported by native WebGPU.

    The rewrite implements the ONNX definition
    ``gamma * where(x > 0, x, alpha * (exp(x) - 1))``. It is opt-in because
    different CPU/GPU floating-point evaluation can change near-tied raw
    feature matches even when the resulting DIC initializations are equivalent.
    The source model is read only.
    """
    try:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError("Install the optional 'onnx' package to rewrite this model.") from exc

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The optimized model must use a different path from the source model.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    graph = model.graph
    if any(initializer.name.startswith("wgpu_matching_selu_") for initializer in graph.initializer):
        raise ValueError("Model already contains WGPU-Matching SELU rewrite constants.")

    def add_scalar(name: str, value: float) -> str:
        tensor = numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)
        graph.initializer.append(tensor)
        return name

    zero = add_scalar("wgpu_matching_selu_zero", 0.0)
    one = add_scalar("wgpu_matching_selu_one", 1.0)
    rewritten = []
    count = 0

    for node in graph.node:
        if node.op_type != "Selu" or node.domain not in ("", "ai.onnx"):
            rewritten.append(node)
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            raise ValueError(f"Unsupported Selu node signature: {node.name!r}.")

        attributes = {
            attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute
        }
        alpha = float(attributes.get("alpha", 1.6732631921768188))
        gamma = float(attributes.get("gamma", 1.0507010221481323))
        count += 1
        prefix = f"wgpu_matching_selu_{count}"
        alpha_name = add_scalar(f"{prefix}_alpha", alpha)
        gamma_name = add_scalar(f"{prefix}_gamma", gamma)
        source = node.input[0]
        output = node.output[0]
        exponential = f"{prefix}_exp"
        exp_minus_one = f"{prefix}_exp_minus_one"
        negative_branch = f"{prefix}_negative_branch"
        positive_condition = f"{prefix}_positive_condition"
        selected = f"{prefix}_selected"
        rewritten.extend(
            [
                helper.make_node("Exp", [source], [exponential], name=f"{prefix}_Exp"),
                helper.make_node(
                    "Sub", [exponential, one], [exp_minus_one], name=f"{prefix}_Expm1"
                ),
                helper.make_node(
                    "Mul",
                    [exp_minus_one, alpha_name],
                    [negative_branch],
                    name=f"{prefix}_NegativeScale",
                ),
                helper.make_node(
                    "Greater",
                    [source, zero],
                    [positive_condition],
                    name=f"{prefix}_PositiveCondition",
                ),
                helper.make_node(
                    "Where",
                    [positive_condition, source, negative_branch],
                    [selected],
                    name=f"{prefix}_SelectBranch",
                ),
                helper.make_node(
                    "Mul", [selected, gamma_name], [output], name=f"{prefix}_GammaScale"
                ),
            ]
        )

    if count == 0:
        raise ValueError("No standard ONNX Selu nodes were found in the model graph.")
    graph.ClearField("node")
    graph.node.extend(rewritten)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return {
        "rewritten_operator": "Selu",
        "rewritten_node_count": count,
        "replacement_ops_per_node": ["Exp", "Sub", "Mul", "Greater", "Where", "Mul"],
        "output_path": str(output_path),
        "source_sha256": _sha256(input_path),
        "optimized_sha256": _sha256(output_path),
    }


def rewrite_efficientloftr_border_mask(
    input_path: Path,
    output_path: Path,
    *,
    tensor_width: int = 640,
    tensor_height: int = 384,
    coarse_stride: int = 8,
    border: int = 2,
) -> dict[str, Any]:
    """Replace EfficientLoFTR's eight full-tensor border ScatterND ops.

    The optimized graph is intentionally specialized to one padded input shape.
    A full border-validity constant lets the rewrite use one same-shape ``And``
    operation. This avoids a native WebGPU EP issue observed with two chained
    broadcast ``And`` operations on this large rank-5 tensor. The binary mask
    is also cast to float32 so ``ArgMax`` and ``ReduceMax`` stay on WebGPU rather
    than falling back to CPU for int8. The source model is not modified.
    """
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError("Install the optional 'onnx' package to rewrite this model.") from exc

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The optimized model must use a different path from the source model.")
    if min(tensor_width, tensor_height, coarse_stride) <= 0 or border < 0:
        raise ValueError(
            "Tensor dimensions and coarse stride must be positive; border nonnegative."
        )
    if tensor_width % coarse_stride or tensor_height % coarse_stride:
        raise ValueError("The specialized tensor dimensions must divide evenly by coarse_stride.")
    coarse_width = tensor_width // coarse_stride
    coarse_height = tensor_height // coarse_stride
    if 2 * border >= min(coarse_width, coarse_height):
        raise ValueError("The border must leave at least one valid coarse-grid row and column.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(input_path))
    graph = model.graph
    if any(
        initializer.name.startswith("wgpu_matching_loftr_border_")
        for initializer in graph.initializer
    ):
        raise ValueError("Model already contains WGPU-Matching LoFTR border rewrite constants.")

    image_inputs = {value.name: value for value in graph.input}
    if not {"image0", "image1"}.issubset(image_inputs):
        raise ValueError("Expected EfficientLoFTR image0 and image1 graph inputs.")
    for name in ("image0", "image1"):
        dimensions = image_inputs[name].type.tensor_type.shape.dim
        if len(dimensions) != 4:
            raise ValueError(f"Expected NCHW {name} input, got rank {len(dimensions)}.")
        dimensions[2].dim_value = tensor_height
        dimensions[3].dim_value = tensor_width

    scatter_nodes = [
        node
        for node in graph.node
        if node.op_type == "ScatterND" and node.name.startswith("/coarse_matching/ScatterND")
    ]
    scatter_nodes.sort(key=lambda node: node.name)
    if len(scatter_nodes) != 8:
        raise ValueError(
            f"Expected eight coarse-matching ScatterND nodes, found {len(scatter_nodes)}."
        )
    for previous, following in pairwise(scatter_nodes):
        if following.input[0] != previous.output[0]:
            raise ValueError(
                "The expected sequential coarse-matching ScatterND chain was not found."
            )

    final_scatter_value = scatter_nodes[-1].output[0]
    reshape_nodes = [node for node in graph.node if node.name == "/coarse_matching/Reshape_9"]
    if len(reshape_nodes) != 1 or reshape_nodes[0].input[0] != final_scatter_value:
        raise ValueError("The final border mask is not connected to the expected Reshape_9 node.")
    original_mask = scatter_nodes[0].input[0]

    coarse_cast = [node for node in graph.node if node.name == "/coarse_matching/Cast"]
    argmax = [node for node in graph.node if node.name == "/coarse_matching/ArgMax"]
    reduce_max = [node for node in graph.node if node.name == "/coarse_matching/ReduceMax_2"]
    if len(coarse_cast) != 1 or len(argmax) != 1 or len(reduce_max) != 1:
        raise ValueError("The expected coarse-matching cast/ArgMax/ReduceMax path was not found.")
    if (
        coarse_cast[0].output[0] not in argmax[0].input
        or coarse_cast[0].output[0] not in reduce_max[0].input
    ):
        raise ValueError("The coarse-matching cast does not feed the expected reductions.")
    cast_type = next(
        (attribute for attribute in coarse_cast[0].attribute if attribute.name == "to"), None
    )
    if cast_type is None or cast_type.i != TensorProto.INT8:
        raise ValueError("Expected the coarse-matching selection cast to produce int8.")
    # The WebGPU EP supports these reductions for float32 but assigns their
    # int8 form to CPU. The mask is binary, so float32 preserves ordering and
    # the max result exactly while allowing both reductions to stay on GPU.
    cast_type.i = TensorProto.FLOAT

    source_y = (np.arange(coarse_height) >= border) & (
        np.arange(coarse_height) < coarse_height - border
    )
    source_x = (np.arange(coarse_width) >= border) & (
        np.arange(coarse_width) < coarse_width - border
    )
    source_valid = (source_y[:, None] & source_x[None, :])[None, :, :, None, None]
    target_valid = source_valid.transpose(0, 3, 4, 1, 2)
    valid_name = "wgpu_matching_loftr_border_valid"
    full_shape = (1, coarse_height, coarse_width, coarse_height, coarse_width)
    full_valid = np.broadcast_to(source_valid, full_shape) & np.broadcast_to(
        target_valid, full_shape
    )
    graph.initializer.append(
        numpy_helper.from_array(full_valid.astype(np.bool_, copy=False), name=valid_name)
    )
    final_mask_name = "wgpu_matching_loftr_border_masked"
    replacement_nodes = [
        helper.make_node(
            "And",
            [original_mask, valid_name],
            [final_mask_name],
            name="WgpuMatching_EfficientLoFTR_BorderMask",
        ),
    ]

    insert_at = next(index for index, node in enumerate(graph.node) if node is scatter_nodes[0])
    nodes = list(graph.node)
    nodes[insert_at:insert_at] = replacement_nodes
    for node in reshape_nodes:
        node.input[0] = final_mask_name

    # Remove operators used only to materialize ScatterND slices and updates.
    producers = {value: node for node in nodes for value in node.output}
    required_nodes: set[int] = set()
    pending = [value.name for value in graph.output]
    while pending:
        value = pending.pop()
        node = producers.get(value)
        if node is None or id(node) in required_nodes:
            continue
        required_nodes.add(id(node))
        pending.extend(node.input)
    kept_nodes = [node for node in nodes if id(node) in required_nodes]
    removed_node_count = len(nodes) - len(kept_nodes)
    graph.ClearField("node")
    graph.node.extend(kept_nodes)

    used_values = {value for node in graph.node for value in node.input}
    used_values.update(value.name for value in graph.output)
    initializers = [
        initializer for initializer in graph.initializer if initializer.name in used_values
    ]
    graph.ClearField("initializer")
    graph.initializer.extend(initializers)

    metadata = {
        "rewrite": "efficientloftr-dense-border-mask-float-reductions-v1",
        "tensor_width": tensor_width,
        "tensor_height": tensor_height,
        "coarse_stride": coarse_stride,
        "coarse_width": coarse_width,
        "coarse_height": coarse_height,
        "border": border,
        "coarse_selection_cast": "int8 -> float32 for WebGPU ArgMax/ReduceMax",
    }
    property_map = {entry.key: entry for entry in model.metadata_props}
    key = "com.wgpu_matching.efficientloftr_border_mask"
    if key in property_map:
        model.metadata_props.remove(property_map[key])
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = __import__("json").dumps(metadata, sort_keys=True)

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return {
        **metadata,
        "rewritten_operator": "ScatterND border-mask chain",
        "rewritten_node_counts": {"ScatterND_removed": len(scatter_nodes), "Cast_dtype_changed": 1},
        "replacement_ops": ["And"],
        "border_mask_initializer_bytes": int(full_valid.nbytes),
        "coarse_selection_cast": "int8 -> float32",
        "removed_dead_node_count": removed_node_count,
        "specialized_model_inputs": ["image0", "image1"],
        "output_path": str(output_path),
        "source_sha256": _sha256(input_path),
        "optimized_sha256": _sha256(output_path),
    }


def decompose_raco_grid_sample(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Replace zero-padded bilinear GridSample nodes by GatherND primitives.

    This rewrite targets the RaCo/ALIKED export's opset-20 GridSample nodes
    with ``align_corners=1`` and ``padding_mode=zeros``. Border-padded nodes
    are retained. Shape handling remains dynamic; high-volume feature tensors
    are gathered on the WebGPU EP while scalar shapes and coordinate masks may
    be delegated to CPU by a particular runtime build.
    """
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError("Install the optional 'onnx' package to rewrite this model.") from exc

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The optimized model must use a different path from the source model.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(input_path))
    graph = model.graph
    if any(
        initializer.name.startswith("wgpu_matching_gridsample_")
        for initializer in graph.initializer
    ):
        raise ValueError("Model already contains WGPU-Matching GridSample rewrite nodes.")

    opset = next((item.version for item in model.opset_import if item.domain == ""), 0)
    if opset < 16:
        raise ValueError(f"Expected a standard GridSample model at opset 16 or newer; got {opset}.")

    rewritten: list[Any] = []
    rewritten_count = 0
    retained_count = 0
    for original in graph.node:
        if original.op_type != "GridSample" or original.domain not in ("", "ai.onnx"):
            rewritten.append(original)
            continue
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in original.attribute
        }
        mode = attributes.get("mode", b"linear")
        padding = attributes.get("padding_mode", b"zeros")
        align_corners = int(attributes.get("align_corners", 0))
        if mode not in (b"linear", "linear") or align_corners != 1:
            raise ValueError(
                f"Unsupported GridSample {original.name}: mode={mode!r}, align_corners={align_corners}."
            )
        if padding in (b"border", "border"):
            rewritten.append(original)
            retained_count += 1
            continue
        if padding not in (b"zeros", "zeros"):
            raise ValueError(
                f"Unsupported GridSample padding mode in {original.name}: {padding!r}."
            )

        rewritten_count += 1
        prefix = f"wgpu_matching_gridsample_{rewritten_count}"
        local_nodes = []
        local_initializers = []

        def add_initializer(
            label: str,
            value: Any,
            dtype: Any,
            *,
            prefix: str = prefix,
            local_initializers: list[Any] = local_initializers,
        ) -> str:
            name = f"{prefix}_{label}"
            local_initializers.append(
                numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)
            )
            return name

        def add_node(
            op: str,
            inputs: list[str],
            label: str,
            *,
            prefix: str = prefix,
            local_nodes: list[Any] = local_nodes,
            **attrs: Any,
        ) -> str:
            output = f"{prefix}_{label}"
            local_nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))
            return output

        def scalar_component(shape_name: str, position: int, label: str) -> str:
            index = add_initializer(f"{label}_index", position, np.int64)
            return add_node("Gather", [shape_name, index], label, axis=0)

        def shape_vector(values: list[str], label: str) -> str:
            axes = add_initializer(f"{label}_axis", [0], np.int64)
            vectors = [
                add_node("Unsqueeze", [value, axes], f"{label}_{i}")
                for i, value in enumerate(values)
            ]
            return add_node("Concat", vectors, label, axis=0)

        x_name, grid_name = original.input[:2]
        x_shape = add_node("Shape", [x_name], "input_shape")
        grid_shape = add_node("Shape", [grid_name], "grid_shape")
        batch = scalar_component(x_shape, 0, "batch")
        channels = scalar_component(x_shape, 1, "channels")
        height = scalar_component(x_shape, 2, "height")
        width = scalar_component(x_shape, 3, "width")
        out_height = scalar_component(grid_shape, 1, "out_height")
        out_width = scalar_component(grid_shape, 2, "out_width")
        spatial = add_node("Mul", [height, width], "input_spatial_size")
        output_spatial = add_node("Mul", [out_height, out_width], "output_spatial_size")
        one_i = add_initializer("one_i", 1, np.int64)
        one_f = add_initializer("one_f", 1.0, np.float32)
        half_f = add_initializer("half_f", 0.5, np.float32)
        zero_f = add_initializer("zero_f", 0.0, np.float32)
        height_float = add_node("Cast", [height], "height_f", to=TensorProto.FLOAT)
        width_float = add_node("Cast", [width], "width_f", to=TensorProto.FLOAT)
        height_minus_one = add_node("Sub", [height_float, one_f], "height_minus_one")
        width_minus_one = add_node("Sub", [width_float, one_f], "width_minus_one")
        height_scale = add_node("Mul", [height_minus_one, half_f], "height_scale")
        width_scale = add_node("Mul", [width_minus_one, half_f], "width_scale")

        grid_x_index = add_initializer("grid_x_index", 0, np.int64)
        grid_y_index = add_initializer("grid_y_index", 1, np.int64)
        grid_x = add_node("Gather", [grid_name, grid_x_index], "grid_x", axis=3)
        grid_y = add_node("Gather", [grid_name, grid_y_index], "grid_y", axis=3)
        grid_x = add_node("Add", [grid_x, one_f], "grid_x_plus_one")
        grid_y = add_node("Add", [grid_y, one_f], "grid_y_plus_one")
        pixel_x = add_node("Mul", [grid_x, width_scale], "pixel_x")
        pixel_y = add_node("Mul", [grid_y, height_scale], "pixel_y")
        floor_x = add_node("Floor", [pixel_x], "floor_x")
        floor_y = add_node("Floor", [pixel_y], "floor_y")
        upper_x = add_node("Add", [floor_x, one_f], "upper_x")
        upper_y = add_node("Add", [floor_y, one_f], "upper_y")
        delta_x = add_node("Sub", [pixel_x, floor_x], "delta_x")
        delta_y = add_node("Sub", [pixel_y, floor_y], "delta_y")
        inverse_x = add_node("Sub", [one_f, delta_x], "inverse_x")
        inverse_y = add_node("Sub", [one_f, delta_y], "inverse_y")
        nhwc = add_node("Transpose", [x_name], "input_nhwc", perm=[0, 2, 3, 1])
        xflat_shape = shape_vector([batch, spatial, channels], "xflat_shape")
        xflat = add_node("Reshape", [nhwc, xflat_shape], "input_flat")
        indices_shape = shape_vector([batch, output_spatial, one_i], "indices_shape")
        weights_shape = shape_vector([batch, output_spatial], "weights_shape")
        axes_two = add_initializer("axes_two", [2], np.int64)
        neighbors = [
            ("00", floor_x, floor_y, inverse_x, inverse_y),
            ("10", upper_x, floor_y, delta_x, inverse_y),
            ("01", floor_x, upper_y, inverse_x, delta_y),
            ("11", upper_x, upper_y, delta_x, delta_y),
        ]
        weighted_samples = []
        for label, x_index, y_index, weight_x, weight_y in neighbors:
            x_valid_low = add_node("GreaterOrEqual", [x_index, zero_f], f"x_valid_low_{label}")
            x_valid_high = add_node("Less", [x_index, width_float], f"x_valid_high_{label}")
            y_valid_low = add_node("GreaterOrEqual", [y_index, zero_f], f"y_valid_low_{label}")
            y_valid_high = add_node("Less", [y_index, height_float], f"y_valid_high_{label}")
            x_valid = add_node("And", [x_valid_low, x_valid_high], f"x_valid_{label}")
            y_valid = add_node("And", [y_valid_low, y_valid_high], f"y_valid_{label}")
            valid = add_node("And", [x_valid, y_valid], f"valid_{label}")
            safe_x = add_node("Where", [x_valid, x_index, zero_f], f"safe_x_{label}")
            safe_y = add_node("Where", [y_valid, y_index, zero_f], f"safe_y_{label}")
            y_offset = add_node("Mul", [safe_y, width_float], f"y_offset_{label}")
            linear_f = add_node("Add", [y_offset, safe_x], f"linear_float_{label}")
            linear = add_node("Cast", [linear_f], f"linear_index_{label}", to=TensorProto.INT64)
            gather_indices = add_node("Reshape", [linear, indices_shape], f"indices_{label}")
            sample = add_node("GatherND", [xflat, gather_indices], f"sample_{label}", batch_dims=1)
            weight = add_node("Mul", [weight_x, weight_y], f"weight_{label}")
            valid_weight = add_node("Where", [valid, weight, zero_f], f"valid_weight_{label}")
            flat_weight = add_node("Reshape", [valid_weight, weights_shape], f"flat_weight_{label}")
            weight_3d = add_node("Unsqueeze", [flat_weight, axes_two], f"weight_3d_{label}")
            weighted_samples.append(add_node("Mul", [sample, weight_3d], f"weighted_{label}"))

        accumulated = add_node("Add", weighted_samples[:2], "sample_sum_0")
        accumulated = add_node("Add", [accumulated, weighted_samples[2]], "sample_sum_1")
        accumulated = add_node("Add", [accumulated, weighted_samples[3]], "sample_sum_2")
        output_shape = shape_vector([batch, out_height, out_width, channels], "output_shape_nhwc")
        output_nhwc = add_node("Reshape", [accumulated, output_shape], "output_nhwc")
        final = helper.make_node(
            "Transpose",
            [output_nhwc],
            [original.output[0]],
            name=f"{prefix}_output_nchw",
            perm=[0, 3, 1, 2],
        )
        local_nodes.append(final)
        rewritten.extend(local_nodes)
        graph.initializer.extend(local_initializers)

    if rewritten_count == 0:
        raise ValueError("No compatible zero-padded linear GridSample nodes were found.")
    graph.ClearField("node")
    graph.node.extend(rewritten)
    key = "com.wgpu_matching.raco_grid_sample"
    metadata = {
        "rewrite": "raco-grid-sample-gathernd-v1",
        "mode": "linear",
        "padding_mode_rewritten": "zeros",
        "align_corners": 1,
        "rewritten_nodes": rewritten_count,
        "retained_border_nodes": retained_count,
        "gather": "GatherND(batch_dims=1) over NHWC-flattened features",
    }
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = __import__("json").dumps(metadata, sort_keys=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return {
        **metadata,
        "output_path": str(output_path),
        "source_sha256": _sha256(input_path),
        "optimized_sha256": _sha256(output_path),
    }


def rewrite_raco_gather_index_expands(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Keep RaCo's large ``GatherElements`` index expansions on WebGPU.

    The current exported RaCo graph expands int64 indices across feature
    channels, then casts the result to int64 again before ``GatherElements``.
    ORT's WebGPU EP assigns these large int64 ``Expand`` nodes to CPU. ONNX
    ``GatherElements`` accepts int32 indices as well as int64, so this rewrite
    casts the compact index tensor to int32 *before* expansion and removes the
    redundant post-expansion int64 cast. It is deliberately limited to the
    exported rank-3 pattern used by this RaCo model; source graphs are read only.
    """
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:
        raise RuntimeError("Install the optional 'onnx' package to rewrite this model.") from exc

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The optimized model must use a different path from the source model.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(input_path))
    graph = model.graph
    if any(
        entry.key == "com.wgpu_matching.raco_int32_gather_indices" for entry in model.metadata_props
    ):
        raise ValueError("Model already contains the WGPU-Matching int32-index rewrite.")

    inferred = onnx.shape_inference.infer_shapes(model)
    value_info = {
        item.name: item
        for item in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    }

    def tensor_shape(value_name: str) -> tuple[int | str, ...] | None:
        item = value_info.get(value_name)
        if item is None or not item.type.HasField("tensor_type"):
            return None
        dimensions = item.type.tensor_type.shape.dim
        return tuple(
            int(dimension.dim_value)
            if dimension.HasField("dim_value")
            else dimension.dim_param or "?"
            for dimension in dimensions
        )

    def tensor_dtype(value_name: str) -> int | None:
        item = value_info.get(value_name)
        if item is None or not item.type.HasField("tensor_type"):
            return None
        return int(item.type.tensor_type.elem_type)

    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    removals: set[int] = set()
    cast_before_expand: dict[int, Any] = {}
    rewrites: list[dict[str, Any]] = []
    int32_outputs: set[str] = set()

    for expand in graph.node:
        if expand.op_type != "Expand" or expand.domain not in ("", "ai.onnx"):
            continue
        expand_shape = tensor_shape(expand.output[0])
        source_shape = tensor_shape(expand.input[0])
        if (
            expand_shape is None
            or source_shape is None
            or tensor_dtype(expand.input[0]) not in (TensorProto.INT32, TensorProto.INT64)
            or len(expand_shape) != 3
            or len(source_shape) != 3
            or source_shape[1] != 1
            or expand_shape[1] <= 1
            or source_shape[0] != expand_shape[0]
            or source_shape[2] != expand_shape[2]
        ):
            continue

        # Restrict the pass to large index replication along the feature-channel
        # axis, with a redundant int64 cast immediately before GatherElements.
        if expand_shape[1] * expand_shape[2] < 1_000_000:
            continue
        expand_users = consumers.get(expand.output[0], [])
        if len(expand_users) != 1 or expand_users[0].op_type != "Cast":
            continue
        redundant_cast = expand_users[0]
        cast_target = next(
            (attribute.i for attribute in redundant_cast.attribute if attribute.name == "to"), None
        )
        if cast_target != TensorProto.INT64 or len(redundant_cast.output) != 1:
            continue
        gather_users = consumers.get(redundant_cast.output[0], [])
        if len(gather_users) != 1 or gather_users[0].op_type != "GatherElements":
            continue
        gather = gather_users[0]
        if len(gather.input) != 2 or gather.input[1] != redundant_cast.output[0]:
            continue
        axis = next((attribute.i for attribute in gather.attribute if attribute.name == "axis"), 0)
        if axis < 0:
            axis += 3
        gather_data_shape = tensor_shape(gather.input[0])
        gather_indices_shape = tensor_shape(gather.input[1])
        gather_axis_size = (
            gather_data_shape[2]
            if gather_data_shape is not None and len(gather_data_shape) == 3
            else None
        )
        known_dynamic_exports = {"node_expand_4", "node_expand_18"}
        axis_size_is_safe = (
            isinstance(gather_axis_size, int) and 0 < gather_axis_size < 2**31
        ) or (expand.name in known_dynamic_exports and isinstance(gather_axis_size, str))
        if (
            axis != 2
            or gather_data_shape is None
            or len(gather_data_shape) != 3
            or not axis_size_is_safe
            or gather_indices_shape != expand_shape
            or gather_data_shape[:2] != expand_shape[:2]
        ):
            continue

        if tensor_dtype(expand.input[0]) == TensorProto.INT64:
            cast_output = f"wgpu_matching_raco_i32_gather_{len(rewrites) + 1}"
            cast_node = helper.make_node(
                "Cast",
                [expand.input[0]],
                [cast_output],
                name=f"{cast_output}_Cast",
                to=TensorProto.INT32,
            )
            cast_before_expand[id(expand)] = cast_node
            expand.input[0] = cast_output
        gather.input[1] = expand.output[0]
        int32_outputs.add(expand.output[0])
        removals.add(id(redundant_cast))
        rewrites.append(
            {
                "expand_node": expand.name,
                "gather_node": gather.name,
                "input_shape": list(source_shape),
                "expanded_shape": list(expand_shape),
                "gather_axis_size": gather_axis_size,
                "expanded_int64_bytes_avoided_for_pair_batch": int(
                    2 * expand_shape[1] * expand_shape[2] * 8
                ),
            }
        )

    if not rewrites:
        raise ValueError("No compatible large RaCo GatherElements index expansions were found.")

    rewritten_nodes = []
    for node in graph.node:
        if id(node) in removals:
            continue
        cast = cast_before_expand.get(id(node))
        if cast is not None:
            rewritten_nodes.append(cast)
        rewritten_nodes.append(node)
    graph.ClearField("node")
    graph.node.extend(rewritten_nodes)
    for value in graph.value_info:
        if value.name in int32_outputs:
            value.type.tensor_type.elem_type = TensorProto.INT32
    metadata = {
        "rewrite": "raco-int32-gather-index-expands-v1",
        "rewritten_gathers": rewrites,
        "index_dtype": "int32",
        "assumption": (
            "Valid indices for the supported RaCo export fit signed int32 within the runtime "
            "feature-grid extent."
        ),
    }
    entry = model.metadata_props.add()
    entry.key = "com.wgpu_matching.raco_int32_gather_indices"
    entry.value = __import__("json").dumps(metadata, sort_keys=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return {
        **metadata,
        "output_path": str(output_path),
        "source_sha256": _sha256(input_path),
        "optimized_sha256": _sha256(output_path),
    }


def rewrite_raco_sparse_index_arithmetic_int32(
    input_path: Path, output_path: Path
) -> dict[str, Any]:
    """Move RaCo's large integer coordinate grid arithmetic off the CPU.

    The exported graph builds flattened feature indices with int64 ``Expand``,
    ``Add`` and ``Mul`` nodes. Those tensors are bounded by the feature-grid
    dimensions and are consumed as GatherElements indices. Converting the
    compact coordinate inputs to int32 lets the WebGPU EP execute the large
    broadcast arithmetic without first materializing int64 CPU tensors.
    This is intentionally tied to the named subgraph in the published RaCo
    export and validates that its expected operators are present.
    """
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:
        raise RuntimeError("Install the optional 'onnx' package to rewrite this model.") from exc

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The optimized model must use a different path from the source model.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(input_path))
    graph = model.graph
    if any(
        entry.key == "com.wgpu_matching.raco_int32_sparse_index_grid"
        for entry in model.metadata_props
    ):
        raise ValueError("Model already contains the WGPU-Matching int32 grid rewrite.")

    node_by_name = {node.name: node for node in graph.node}
    expected_ops = {
        "node_unsqueeze_3": "Unsqueeze",
        "node_unsqueeze_5": "Unsqueeze",
        "node_expand_2": "Expand",
        "node_expand_3": "Expand",
        "node_add_480": "Add",
        "node_add_495": "Add",
        "node_mul_378": "Mul",
        "node_add_506": "Add",
        "node_view_12": "Reshape",
        "node_slice_3": "Slice",
        "node_unsqueeze_6": "Unsqueeze",
        "node_expand_4": "Expand",
    }
    for name, op_type in expected_ops.items():
        if name not in node_by_name or node_by_name[name].op_type != op_type:
            raise ValueError(f"Expected RaCo index-grid node {name!r} ({op_type}).")

    casts_by_node: dict[int, list[Any]] = {}
    casted_values: list[str] = []

    def cast_input_before(node_name: str, input_index: int, suffix: str) -> None:
        node = node_by_name[node_name]
        if input_index >= len(node.input):
            raise ValueError(f"Unexpected input signature for {node_name}.")
        source = node.input[input_index]
        output = f"wgpu_matching_raco_i32_grid_{suffix}"
        casts_by_node.setdefault(id(node), []).append(
            helper.make_node(
                "Cast", [source], [output], name=f"{output}_Cast", to=TensorProto.INT32
            )
        )
        node.input[input_index] = output
        casted_values.append(output)

    # Cast the two compact row/column coordinate arrays and the stride scalar
    # before they fan out into the large broadcast Add/Mul tensors.
    cast_input_before("node_unsqueeze_3", 0, "row_coordinates")
    cast_input_before("node_unsqueeze_5", 0, "column_coordinates")
    cast_input_before("node_expand_2", 0, "row_grid")
    cast_input_before("node_expand_3", 0, "column_grid")
    cast_input_before("node_mul_378", 1, "grid_stride")

    int32_node_names = {
        "node_unsqueeze_3",
        "node_unsqueeze_5",
        "node_expand_2",
        "node_expand_3",
        "node_add_480",
        "node_add_495",
        "node_mul_378",
        "node_add_506",
        "node_view_12",
        "node_slice_3",
        "node_unsqueeze_6",
        "node_expand_4",
    }
    int32_values = {value for name in int32_node_names for value in node_by_name[name].output}
    rewritten_nodes = []
    for node in graph.node:
        rewritten_nodes.extend(casts_by_node.get(id(node), []))
        rewritten_nodes.append(node)
    graph.ClearField("node")
    graph.node.extend(rewritten_nodes)
    for value in graph.value_info:
        if value.name in int32_values or value.name in casted_values:
            value.type.tensor_type.elem_type = TensorProto.INT32

    metadata = {
        "rewrite": "raco-int32-sparse-index-grid-v1",
        "converted_nodes": sorted(int32_node_names),
        "inserted_int32_casts": 5,
        "purpose": "keep broadcast feature-index arithmetic in WebGPU",
        "assumption": "the exported 42x42 feature-grid coordinates and stride fit int32",
    }
    entry = model.metadata_props.add()
    entry.key = "com.wgpu_matching.raco_int32_sparse_index_grid"
    entry.value = __import__("json").dumps(metadata, sort_keys=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return {
        **metadata,
        "output_path": str(output_path),
        "source_sha256": _sha256(input_path),
        "optimized_sha256": _sha256(output_path),
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument(
        "--operation",
        choices=(
            "decompose-selu",
            "loftr-border-mask",
            "raco-grid-sample",
            "raco-gather-indices-int32",
            "raco-index-grid-int32",
        ),
        default="decompose-selu",
        help="Optional rewrite to apply (default: decompose-selu).",
    )
    parser.add_argument("--tensor-width", type=int, default=640)
    parser.add_argument("--tensor-height", type=int, default=384)
    parser.add_argument("--coarse-stride", type=int, default=8)
    parser.add_argument("--border", type=int, default=2)
    args = parser.parse_args()
    if args.operation == "loftr-border-mask":
        result = rewrite_efficientloftr_border_mask(
            args.input_model,
            args.output_model,
            tensor_width=args.tensor_width,
            tensor_height=args.tensor_height,
            coarse_stride=args.coarse_stride,
            border=args.border,
        )
    elif args.operation == "raco-grid-sample":
        result = decompose_raco_grid_sample(args.input_model, args.output_model)
    elif args.operation == "raco-gather-indices-int32":
        result = rewrite_raco_gather_index_expands(args.input_model, args.output_model)
    elif args.operation == "raco-index-grid-int32":
        result = rewrite_raco_sparse_index_arithmetic_int32(args.input_model, args.output_model)
    else:
        result = decompose_selu(args.input_model, args.output_model)
    print(f"Applied {args.operation}: wrote {result['output_path']}")


if __name__ == "__main__":
    main()

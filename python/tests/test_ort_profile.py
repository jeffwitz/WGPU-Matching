"""Regression checks for the compact ONNX Runtime trace summarizer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from ort_profile import summarize_profile


def test_summarize_profile_groups_provider_ops_and_tensor_sizes(tmp_path: Path) -> None:
    events = [
        {
            "cat": "Node",
            "ph": "X",
            "name": "copy_0_kernel_time",
            "dur": 1250,
            "args": {
                "op_name": "MemcpyToHost",
                "provider": "WebGpuExecutionProvider",
                "input_type_shape": [{"bool": [2, 4]}],
                "output_type_shape": [{"bool": [2, 4]}],
                "output_size": "8",
            },
        },
        {
            "cat": "Node",
            "ph": "X",
            "name": "scatter_0_kernel_time",
            "dur": 750,
            "args": {
                "op_name": "ScatterND",
                "provider": "CPUExecutionProvider",
                "input_type_shape": [{"float": [2, 3]}],
                "output_type_shape": [{"float": [2, 3]}],
            },
        },
    ]
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(events), encoding="utf-8")

    summary = summarize_profile(trace)

    assert summary["node_event_count"] == 2
    assert summary["providers"][0] == {
        "count": 1,
        "provider": "WebGpuExecutionProvider",
        "duration_ms": 1.25,
    }
    copy = next(row for row in summary["transfers"] if row["direction"] == "MemcpyToHost")
    assert copy["bytes_sum"] == 8
    assert copy["max_bytes"] == 8
    scatter = next(row for row in summary["operations_by_provider"] if row["op"] == "ScatterND")
    assert scatter["provider"] == "CPUExecutionProvider"
    assert scatter["max_output_bytes"] == 24


def test_selu_decomposition_keeps_source_and_emits_checked_onnx(tmp_path: Path) -> None:
    import pytest

    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    from optimize_onnx_webgpu import decompose_selu

    source = tmp_path / "source.onnx"
    output = tmp_path / "optimized.onnx"
    graph = helper.make_graph(
        [helper.make_node("Selu", ["x"], ["y"])],
        "selu_smoke",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.save(model, source)
    source_bytes = source.read_bytes()

    result = decompose_selu(source, output)
    optimized = onnx.load(output)

    assert result["rewritten_node_count"] == 1
    assert source.read_bytes() == source_bytes
    assert not any(node.op_type == "Selu" for node in optimized.graph.node)
    assert [node.op_type for node in optimized.graph.node] == [
        "Exp",
        "Sub",
        "Mul",
        "Greater",
        "Where",
        "Mul",
    ]
    onnx.checker.check_model(optimized)

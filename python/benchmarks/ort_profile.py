"""Summarize ONNX Runtime Chrome-trace profiles without retaining large traces."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

_TYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float16": 2,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
    "double": 8,
    "float64": 8,
}


def _shape_bytes(value: Any) -> int:
    """Estimate bytes represented by ORT's input_type_shape/output_type_shape."""
    if not isinstance(value, list):
        return 0
    total = 0
    for item in value:
        if not isinstance(item, dict):
            continue
        for dtype, shape in item.items():
            if dtype not in _TYPE_BYTES or not isinstance(shape, list):
                continue
            elements = 1
            for dimension in shape:
                if not isinstance(dimension, int) or dimension < 0:
                    elements = 0
                    break
                elements *= dimension
            total += elements * _TYPE_BYTES[dtype]
    return total


def summarize_profile(path: Path, *, top_nodes: int = 20) -> dict[str, Any]:
    """Aggregate provider/op timings, transfers, tensor sizes, and largest nodes.

    ORT event durations are reported as observed trace durations. In particular,
    WebGPU memcpy events can include queue waits and synchronization; they are
    not treated as isolated hardware copy-bandwidth measurements.
    """
    import json

    events = json.loads(path.read_text(encoding="utf-8"))
    operations: dict[tuple[str, str], dict[str, int | float]] = defaultdict(
        lambda: {
            "count": 0,
            "duration_us": 0.0,
            "input_bytes_sum": 0,
            "output_bytes_sum": 0,
            "max_input_bytes": 0,
            "max_output_bytes": 0,
        }
    )
    providers: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"count": 0, "duration_us": 0.0}
    )
    transfers: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"count": 0, "duration_us": 0.0, "bytes_sum": 0, "max_bytes": 0}
    )
    nodes: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    cpu_arena_peak_bytes = 0

    for event in events:
        args = event.get("args", {})
        if event.get("ph") != "X":
            continue
        duration = float(event.get("dur", 0) or 0)
        if event.get("cat") == "Session":
            sessions.append(
                {
                    "name": event.get("name"),
                    "duration_us": duration,
                    "timestamp_us": event.get("ts"),
                }
            )
            continue
        if event.get("cat") != "Node":
            continue

        op = str(args.get("op_name") or event.get("name") or "unknown")
        provider = str(args.get("provider") or "unknown")
        input_bytes = _shape_bytes(args.get("input_type_shape"))
        output_bytes = _shape_bytes(args.get("output_type_shape"))
        key = (op, provider)
        aggregate = operations[key]
        aggregate["count"] += 1
        aggregate["duration_us"] += duration
        aggregate["input_bytes_sum"] += input_bytes
        aggregate["output_bytes_sum"] += output_bytes
        aggregate["max_input_bytes"] = max(aggregate["max_input_bytes"], input_bytes)
        aggregate["max_output_bytes"] = max(aggregate["max_output_bytes"], output_bytes)
        providers[provider]["count"] += 1
        providers[provider]["duration_us"] += duration

        if op.startswith("Memcpy"):
            byte_count = max(input_bytes, output_bytes)
            try:
                byte_count = max(byte_count, int(args.get("output_size", 0) or 0))
            except (TypeError, ValueError):
                pass
            transfer = transfers[op]
            transfer["count"] += 1
            transfer["duration_us"] += duration
            transfer["bytes_sum"] += byte_count
            transfer["max_bytes"] = max(transfer["max_bytes"], byte_count)

        for name, value in args.items():
            if "cpu_memory_arena" in name.lower() and "peak" in name.lower():
                try:
                    cpu_arena_peak_bytes = max(cpu_arena_peak_bytes, int(value))
                except (TypeError, ValueError):
                    pass

        nodes.append(
            {
                "name": event.get("name"),
                "op": op,
                "provider": provider,
                "duration_us": duration,
                "input_bytes_estimate": input_bytes,
                "output_bytes_estimate": output_bytes,
                "input_type_shape": args.get("input_type_shape"),
                "output_type_shape": args.get("output_type_shape"),
            }
        )

    def sorted_rows(
        source: dict[Any, dict[str, int | float]], key_field: str
    ) -> list[dict[str, Any]]:
        rows = []
        for key, value in source.items():
            row = dict(value)
            if isinstance(key, tuple):
                row.update(op=key[0], provider=key[1])
            else:
                row[key_field] = key
            row["duration_ms"] = float(row.get("duration_us", 0)) / 1000
            row.pop("duration_us", None)
            rows.append(row)
        return sorted(rows, key=lambda row: row["duration_ms"], reverse=True)

    return {
        "trace_event_count": len(events),
        "node_event_count": len(nodes),
        "session_events": sessions,
        "providers": sorted_rows(providers, "provider"),
        "operations_by_provider": sorted_rows(operations, "op"),
        "transfers": sorted_rows(transfers, "direction"),
        "top_nodes": sorted(nodes, key=lambda node: node["duration_us"], reverse=True)[:top_nodes],
        "cpu_memory_arena_peak_bytes": cpu_arena_peak_bytes or None,
        "limitations": [
            "ORT trace durations are event durations, not hardware GPU timestamps.",
            "Memcpy event durations may include synchronization or queue waits.",
            "Tensor byte counts are estimates from static input/output shapes.",
            "The trace does not provide peak GPU allocation or live-buffer memory.",
        ],
    }

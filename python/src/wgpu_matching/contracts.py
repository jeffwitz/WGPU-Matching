"""Validation helpers driven by the shared browser/Python matcher contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from importlib.resources import files
from numbers import Integral, Real
from typing import Any

_CONTRACT = json.loads(
    files("wgpu_matching").joinpath("matcher-contracts.json").read_text(encoding="utf-8")
)


def validate_matcher_options(
    algorithm: str, options: Mapping[str, Any], width: int, height: int
) -> None:
    """Validate a Python options mapping against the shared contract constraints."""
    try:
        spec = _CONTRACT["algorithms"][algorithm]
    except KeyError as error:
        raise ValueError(f"No shared contract for matcher {algorithm}.") from error

    for parameter, rule in spec.get("constraints", {}).items():
        key = spec["option_keys"][parameter]["python"]
        value = options[key]
        kind = rule["type"]
        if kind == "integer":
            valid_type = isinstance(value, Integral) and not isinstance(value, bool)
            if not valid_type or value < rule["minimum"] or value > rule.get("maximum", math.inf):
                raise ValueError(f"{algorithm}.{parameter} must be an integer in the configured range")
        elif kind in {"number", "optional_number"}:
            if kind == "optional_number" and value is None:
                continue
            valid_type = isinstance(value, Real) and not isinstance(value, bool)
            if (
                not valid_type
                or not math.isfinite(value)
                or value < rule["minimum"]
                or value > rule.get("maximum", math.inf)
            ):
                raise ValueError(f"{algorithm}.{parameter} must be a finite number in the configured range")
        elif kind == "enum" and value not in rule["values"]:
            raise ValueError(f"{algorithm}.{parameter} must be one of {', '.join(rule['values'])}")
        elif kind == "points" and value is not None:
            try:
                points = tuple(tuple(point) for point in value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{algorithm}.seed_points must be a sequence of (x, y) pairs") from error
            if len(points) > rule["maximum_count"]:
                raise ValueError(
                    f"{algorithm}.seed_points must contain at most {rule['maximum_count']} points"
                )
            if points and any(len(point) != 2 for point in points):
                raise ValueError(f"{algorithm}.seed_points must be a sequence of (x, y) pairs")
            if points:
                for x, y in points:
                    if (
                        not isinstance(x, Real)
                        or isinstance(x, bool)
                        or not isinstance(y, Real)
                        or isinstance(y, bool)
                        or not math.isfinite(x)
                        or not math.isfinite(y)
                        or x < 0
                        or y < 0
                        or x >= width
                        or y >= height
                    ):
                        raise ValueError(
                            f"{algorithm}.seed_points must contain finite coordinates inside the image"
                        )

"""Source-declared radial station equations after eliminating a common axis.

Only source annotation values enter this affine construction. A fitted/pixel
datum coordinate is never a parameter. A relative equation does not certify
the absolute placement or completeness of its reference diameter.
"""
from __future__ import annotations

import math
import re

from .dimension_policy import source_dimension_target


def radial_station_relation(metadata, *, measurement_axis=None):
    required = {"type", "axis", "direction", "primary_kind", "reference_record_id", "reference_dimension"}
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("Radial station relation must use its exact source schema")
    if metadata["type"] != "half_section_common_axis_eliminated":
        raise ValueError("Unsupported radial station relation")
    if metadata["axis"] not in {"x", "y"} or (measurement_axis is not None and metadata["axis"] != measurement_axis):
        raise ValueError("Radial station relation axis mismatch")
    if type(metadata["direction"]) is not int or metadata["direction"] not in {-1, 1}:
        raise ValueError("Radial station relation requires a signed source direction")
    if metadata["primary_kind"] not in {"diameter", "length"}:
        raise ValueError("Radial station relation requires a diameter or radial length")
    reference = metadata["reference_record_id"]
    if not isinstance(reference, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", reference) is None:
        raise ValueError("Radial station relation needs a source reference record")
    parsed = metadata["reference_dimension"]
    if (not isinstance(parsed, dict) or set(parsed) != {"kind", "nominal", "lower_deviation", "upper_deviation"}
            or parsed["kind"] != "diameter"):
        raise ValueError("Radial station reference must be an OCR diameter")
    reference_target, policy = source_dimension_target(parsed)
    multiplier = 2. if metadata["primary_kind"] == "diameter" else 1.
    return {"coefficient": multiplier*metadata["direction"], "offset": multiplier*reference_target/2,
            "reference_target": reference_target, "reference_tolerance": policy,
            "reference_record_id": reference, "absolute_axis_position_verified": False,
            "reference_diameter_independently_solved": False}


def radial_station_value(delta, metadata, *, measurement_axis=None):
    relation = radial_station_relation(metadata, measurement_axis=measurement_axis)
    value = relation["coefficient"]*float(delta)+relation["offset"]
    if not math.isfinite(value):
        raise ValueError("Radial station measurement must be finite")
    return value

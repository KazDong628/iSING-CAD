"""Deterministic dimensional reconstruction and independent geometric checks.

Runtime never opens the dataset or ground truth.  The template supplies only
declared shape priors and topology, while supplied dimensions drive all points.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import ezdxf
import numpy as np

from .templates.solid293 import (
    LOWER_JOIN_DIRECTIONS_DEG, UPPER_JOIN_DIRECTIONS_DEG, template_schema,
)

LINEAR_TOLERANCE_MM = 1e-6
TANGENT_TOLERANCE_DEG = 1e-5
SHARP_JOINS = {0, 8, 9, 10, 11, 20}


def _v(angle):
    a = math.radians(angle)
    return np.array([math.cos(a), math.sin(a)], dtype=float)


def _normal(angle):
    a = _v(angle)
    return np.array([-a[1], a[0]])


def _point(value):
    return [float(value[0]), float(value[1])]


def _line(pid, start, end, *parameters):
    return {"id": pid, "type": "LINE", "start": _point(start), "end": _point(end),
            "driving_parameters": list(parameters)}


def _arc(pid, start, radius, angle_in, angle_out, sign, parameter):
    turn = (sign * (angle_out - angle_in)) % 360
    if not 1e-6 < turn < 180:
        raise ValueError(f"{pid}: arc branch changed ({turn:.3f} degrees); dimensions do not fit the template")
    center = np.asarray(start) + sign * radius * _normal(angle_in)
    end = center - sign * radius * _normal(angle_out)
    return {"id": pid, "type": "ARC", "start": _point(start), "end": _point(end),
            "center": _point(center), "radius": float(radius), "clockwise": sign < 0,
            "driving_parameters": [parameter], "shape_prior": "calibration tangent directions"}


def _parameters(values):
    schema = template_schema()
    required = {p["id"] for p in schema["parameters"]}
    if not isinstance(values, dict):
        raise ValueError("parameters must be an object")
    missing, unknown = required - values.keys(), values.keys() - required
    if missing:
        raise ValueError("Missing required dimensions: " + ", ".join(sorted(missing)))
    if unknown:
        raise ValueError("Unknown dimensions: " + ", ".join(sorted(unknown)))
    result = {}
    for field in schema["parameters"]:
        raw = values[field["id"]]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{field['id']}: a numeric value is required")
        value = float(raw)
        if not math.isfinite(value) or not field["min"] <= value <= field["max"]:
            raise ValueError(f"{field['id']}: expected finite value in [{field['min']}, {field['max']}]")
        result[field["id"]] = value
    if not result["d_left"] < result["d_shoulder"] < result["d_right"] < result["d_outer"]:
        raise ValueError("Radial stations must satisfy d_left < d_shoulder < d_right < d_outer")
    if result["left_base"] + result["left_height"] <= result["right_height"]:
        raise ValueError("Left upper face must be above the right upper face for this topology")
    return result


def _tangent_chain(start, end, theta_in, theta_out, arc_specs, joins, line_ids):
    """Solve two unknown straight lengths after freezing explicit shape priors.

    Arc translation = signed radius * (left_normal_in - left_normal_out).
    This is a direct 2x2 linear solve, with no fitted endpoint coordinates.
    """
    angles = [theta_in, *joins, theta_out]
    displacement = np.zeros(2)
    for i, (_, radius, sign, _) in enumerate(arc_specs):
        displacement += sign * radius * (_normal(angles[i]) - _normal(angles[i + 1]))
    matrix = np.column_stack((_v(theta_in), _v(theta_out)))
    if abs(float(np.linalg.det(matrix))) < 1e-5:
        raise ValueError("End support lines are parallel; template branch is underconstrained")
    lengths = np.linalg.solve(matrix, np.asarray(end) - np.asarray(start) - displacement)
    if any(float(length) <= LINEAR_TOLERANCE_MM for length in lengths):
        raise ValueError(f"Template branch requires reversed/zero straight segments: {lengths.tolist()}; review dimensions or use another template")
    cursor = np.asarray(start) + float(lengths[0]) * _v(theta_in)
    entities = [_line(line_ids[0], start, cursor)]
    for i, (pid, radius, sign, parameter) in enumerate(arc_specs):
        entity = _arc(pid, cursor, radius, angles[i], angles[i + 1], sign, parameter)
        entities.append(entity)
        cursor = np.asarray(entity["end"])
    entities.append(_line(line_ids[1], cursor, end))
    return entities, [float(v) for v in lengths]


def _construct(p):
    left, shoulder, right, outer = [p[key] / 2 for key in ("d_left", "d_shoulder", "d_right", "d_outer")]
    base, height = p["left_base"], p["left_base"] + p["left_height"]
    right_height, a, b, fillet = p["right_height"], p["angle_left"], p["angle_right"], p["r_fillet"]
    trim = fillet * math.tan(math.radians((90 - a) / 2))
    if shoulder - trim <= left:
        raise ValueError("R5 fillet consumes the left horizontal face; reduce r_fillet or review diameters")
    upper_fillet = _arc("P02_R5_UPPER", [shoulder - trim, height], fillet, 0, a - 90, -1, "r_fillet")
    lower_fillet_center = np.array([shoulder - trim, base + fillet])
    lower_fillet_start = lower_fillet_center + fillet * _v(-a)
    lower_fillet = _arc("P19_R5_LOWER", lower_fillet_start, fillet, 270 - a, 180, -1, "r_fillet")
    upper_specs = [
        ("P04_R62", p["r_upper_left"], 1, "r_upper_left"),
        ("P05_UPPER_TRANSITION", p["r_transition"], -1, "r_transition"),
        ("P06_R180", p["r_upper_bend"], 1, "r_upper_bend"),
        ("P07_R40_UPPER_RIGHT", p["r_upper_right"], 1, "r_upper_right"),
    ]
    lower_specs = [
        ("P13_R40_LOWER_RIGHT", p["r_lower_right"], 1, "r_lower_right"),
        ("P14_R199", p["r_lower_bend"], -1, "r_lower_bend"),
        ("P15_R110", p["r_lower_mid"], -1, "r_lower_mid"),
        ("P16_R130_LOWER", p["r_lower_left_bend"], 1, "r_lower_left_bend"),
        ("P17_R40_LOWER_LEFT", p["r_lower_left"], 1, "r_lower_left"),
    ]
    upper, upper_lengths = _tangent_chain(upper_fillet["end"], [right, right_height], a - 90, 90 - b,
        upper_specs, UPPER_JOIN_DIRECTIONS_DEG, ("P03_LINE_12_UPPER_LEFT", "P08_LINE_15_UPPER_RIGHT"))
    lower, lower_lengths = _tangent_chain([right, 0], lower_fillet_start, 90 + b, 270 - a,
        lower_specs, LOWER_JOIN_DIRECTIONS_DEG, ("P12_LINE_15_LOWER_RIGHT", "P18_LINE_12_LOWER_LEFT"))
    entities = [
        _line("P00_LEFT_VERTICAL", [left, base], [left, height], "d_left", "left_base", "left_height"),
        _line("P01_TOP_HORIZONTAL", [left, height], upper_fillet["start"], "d_shoulder", "r_fillet"),
        upper_fillet, *upper,
        _line("P09_RIGHT_TOP_HORIZONTAL", [right, right_height], [outer, right_height], "d_right", "d_outer", "right_height"),
        _line("P10_TREAD_SIMPLIFIED", [outer, right_height], [outer, 0], "d_outer", "right_height"),
        _line("P11_RIGHT_BOTTOM_HORIZONTAL", [outer, 0], [right, 0], "d_outer", "d_right"),
        *lower, lower_fillet,
        _line("P20_BOTTOM_HORIZONTAL", lower_fillet["end"], [left, base], "d_left", "left_base"),
    ]
    return entities, {"upper_end_straight_lengths_mm": upper_lengths, "lower_end_straight_lengths_mm": lower_lengths,
        "method": "analytic tangent-chain endpoint solve", "unmeasured_shape_dof_fixed_by_calibration": 7}


def _angle_at(entity, point):
    c = entity["center"]
    return math.atan2(point[1] - c[1], point[0] - c[0])


def _sweep(entity):
    a, b = _angle_at(entity, entity["start"]), _angle_at(entity, entity["end"])
    return ((a - b) if entity["clockwise"] else (b - a)) % (2 * math.pi)


def _on_entity(e, point):
    q, s, t = np.asarray(point), np.asarray(e["start"]), np.asarray(e["end"])
    if e["type"] == "LINE":
        d = t - s
        length_squared = float(np.dot(d, d))
        if length_squared <= LINEAR_TOLERANCE_MM ** 2:
            return float(np.linalg.norm(q - s)) <= LINEAR_TOLERANCE_MM
        u = float(np.dot(q - s, d) / length_squared)
        return -1e-8 <= u <= 1 + 1e-8 and float(np.linalg.norm(q - (s + u * d))) < LINEAR_TOLERANCE_MM
    if abs(float(np.linalg.norm(q - np.asarray(e["center"]))) - e["radius"]) > LINEAR_TOLERANCE_MM:
        return False
    start, theta = _angle_at(e, s), _angle_at(e, q)
    travelled = ((start - theta) if e["clockwise"] else (theta - start)) % (2 * math.pi)
    return travelled <= _sweep(e) + 1e-8 or float(np.linalg.norm(q - s)) < LINEAR_TOLERANCE_MM


def sample_entities(entities, step_mm=0.5):
    """Public preview/evaluation helper. Each primitive includes its two ends."""
    points = []
    for e in entities:
        if e["type"] == "LINE":
            n = max(2, math.ceil(math.dist(e["start"], e["end"]) / step_mm) + 1)
            points.extend(np.linspace(e["start"], e["end"], n).tolist())
        else:
            sweep = _sweep(e)
            n = max(3, math.ceil(e["radius"] * sweep / step_mm) + 1)
            sign = -1 if e["clockwise"] else 1
            angles = np.linspace(_angle_at(e, e["start"]), _angle_at(e, e["start"]) + sign * sweep, n)
            points.extend([[e["center"][0] + e["radius"] * math.cos(a), e["center"][1] + e["radius"] * math.sin(a)] for a in angles])
    return points


def _tangent(e, at_end):
    if e["type"] == "LINE":
        t = np.asarray(e["end"]) - np.asarray(e["start"])
    else:
        radial = np.asarray(e["end" if at_end else "start"]) - np.asarray(e["center"])
        t = np.array([-radial[1], radial[0]]) * (-1 if e["clockwise"] else 1)
    length = float(np.linalg.norm(t))
    return t / length if length > LINEAR_TOLERANCE_MM else np.zeros(2)


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _intersections(a, b):
    """Exact support intersections filtered by trimmed entity domains."""
    candidates = []
    if a["type"] == b["type"] == "LINE":
        s, d = np.asarray(a["start"]), np.asarray(a["end"]) - np.asarray(a["start"])
        t, v = np.asarray(b["start"]), np.asarray(b["end"]) - np.asarray(b["start"])
        det = _cross(d, v)
        if abs(det) > 1e-10:
            candidates = [s + _cross(t - s, v) / det * d]
        elif abs(_cross(t - s, d)) < LINEAR_TOLERANCE_MM:
            candidates = [s, s + d, t, t + v, (s + s + d) / 2, (t + t + v) / 2]
    elif a["type"] != b["type"]:
        line, arc = (a, b) if a["type"] == "LINE" else (b, a)
        start, d = np.asarray(line["start"]), np.asarray(line["end"]) - np.asarray(line["start"])
        offset = start - np.asarray(arc["center"])
        aa, bb = float(np.dot(d, d)), 2 * float(np.dot(offset, d))
        if aa <= LINEAR_TOLERANCE_MM ** 2:
            return [start] if _on_entity(arc, start) else []
        cc = float(np.dot(offset, offset)) - arc["radius"] ** 2
        discriminant = bb * bb - 4 * aa * cc
        if discriminant >= -1e-5:
            root = math.sqrt(max(0, discriminant))
            candidates = [start + t * d for t in ((-bb - root) / (2 * aa), (-bb + root) / (2 * aa))]
    else:
        c1, c2, r1, r2 = np.asarray(a["center"]), np.asarray(b["center"]), a["radius"], b["radius"]
        d = float(np.linalg.norm(c2 - c1))
        if d < 1e-10:
            if abs(r1 - r2) < LINEAR_TOLERANCE_MM:
                candidates = [a["start"], a["end"], b["start"], b["end"], *sample_entities([a], max(r1 / 8, 1))]
        elif abs(r1 - r2) - LINEAR_TOLERANCE_MM <= d <= r1 + r2 + LINEAR_TOLERANCE_MM:
            x = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
            h = math.sqrt(max(0, r1 * r1 - x * x))
            unit = (c2 - c1) / d
            base = c1 + x * unit
            perpendicular = np.array([-unit[1], unit[0]])
            candidates = [base + h * perpendicular, base - h * perpendicular]
    return [np.asarray(point) for point in candidates if _on_entity(a, point) and _on_entity(b, point)]


def _dimensions(entities, p):
    e = entities
    measured = {"d_left": 2 * e[0]["start"][0], "d_right": 2 * e[8]["end"][0],
        "d_outer": 2 * e[10]["start"][0], "left_base": e[0]["start"][1],
        "left_height": e[0]["end"][1] - e[0]["start"][1],
        "right_height": e[10]["start"][1] - e[10]["end"][1]}
    for key, index in (("angle_left", 3), ("angle_right", 8)):
        d = np.asarray(e[index]["end"]) - np.asarray(e[index]["start"])
        measured[key] = math.degrees(math.atan2(abs(d[0]), abs(d[1])))
    d = np.asarray(e[3]["end"]) - np.asarray(e[3]["start"])
    measured["d_shoulder"] = 2 * (e[3]["start"][0] + (e[1]["start"][1] - e[3]["start"][1]) * d[0] / d[1])
    for index, key in ((2, "r_fillet"), (4, "r_upper_left"), (5, "r_transition"),
            (6, "r_upper_bend"), (7, "r_upper_right"), (13, "r_lower_right"),
            (14, "r_lower_bend"), (15, "r_lower_mid"), (16, "r_lower_left_bend"), (17, "r_lower_left")):
        measured[key] = e[index]["radius"]
    return [{"id": key, "expected": p[key], "actual": float(measured[key]),
             "residual": float(abs(measured[key] - p[key])),
             "passed": bool(abs(measured[key] - p[key]) <= (TANGENT_TOLERANCE_DEG if key.startswith("angle_") else LINEAR_TOLERANCE_MM))} for key in p]


def validate_entities(entities, parameters=None):
    """Check closure, tangent direction, arc radii, and exact self intersections."""
    issues, joins, radial_errors = [], [], []
    if not entities:
        issues.append("Contour contains no entities")
    for i, entity in enumerate(entities):
        if math.dist(entity["start"], entity["end"]) <= LINEAR_TOLERANCE_MM:
            issues.append(f"{entity['id']}: zero-length primitive")
        if entity["type"] == "ARC":
            radial_errors.extend(abs(math.dist(entity["center"], entity[point]) - entity["radius"]) for point in ("start", "end"))
        nxt = entities[(i + 1) % len(entities)]
        gap = math.dist(entity["end"], nxt["start"])
        dot = float(np.clip(np.dot(_tangent(entity, True), _tangent(nxt, False)), -1, 1))
        angle = math.degrees(math.acos(dot))
        joins.append({"a": entity["id"], "b": nxt["id"], "gap_mm": gap,
                      "tangent_error_deg": angle, "smooth_expected": i not in SHARP_JOINS})
    max_gap = max((j["gap_mm"] for j in joins), default=0)
    max_tangent = max((j["tangent_error_deg"] for j in joins if j["smooth_expected"]), default=0)
    if max_gap > LINEAR_TOLERANCE_MM:
        issues.append("Contour is not closed/connected within 1e-6 mm")
    if max_tangent > TANGENT_TOLERANCE_DEG:
        issues.append("Expected smooth joins fail directed tangency within 1e-5 degrees")
    if max(radial_errors, default=0) > LINEAR_TOLERANCE_MM:
        issues.append("Arc endpoints are not on their claimed circles")
    for i, a in enumerate(entities):
        for j in range(i + 1, len(entities)):
            b = entities[j]
            common = a["end"] if j == i + 1 else (a["start"] if i == 0 and j == len(entities) - 1 else None)
            for point in _intersections(a, b):
                if common is None or math.dist(point, common) > 1e-5:
                    issues.append(f"Self intersection: {a['id']} / {b['id']}")
                    break
    dimensions = _dimensions(entities, parameters) if parameters else []
    if any(not dimension["passed"] for dimension in dimensions):
        issues.append("Reverse dimension check failed")
    return {"passed": not issues, "dimensions": dimensions, "max_gap_mm": float(max_gap),
            "max_tangent_error_deg": float(max_tangent),
            "max_radial_error_mm": float(max(radial_errors, default=0)), "issues": issues,
            "joins": joins, "tolerances": {"linear_mm": LINEAR_TOLERANCE_MM, "tangent_deg": TANGENT_TOLERANCE_DEG},
            "meaning": "Numerical validity only; not engineering acceptance or held-out generalization"}


def _bounds(entities):
    points = [p for e in entities for p in (e["start"], e["end"])]
    for e in entities:
        if e["type"] == "ARC":
            for angle in (0, 90, 180, 270):
                point = np.asarray(e["center"]) + e["radius"] * _v(angle)
                if _on_entity(e, point):
                    points.append(_point(point))
    return {"min_x": min(p[0] for p in points), "min_y": min(p[1] for p in points),
            "max_x": max(p[0] for p in points), "max_y": max(p[1] for p in points)}


def _export_dxf(entities, path):
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4
    doc.layers.new("MAIN_PROFILE", dxfattribs={"color": 4})
    model = doc.modelspace()
    for e in entities:
        if e["type"] == "LINE":
            model.add_line(e["start"], e["end"], dxfattribs={"layer": "MAIN_PROFILE"})
        else:
            start, end = _angle_at(e, e["start"]), _angle_at(e, e["end"])
            if e["clockwise"]:
                start, end = end, start
            model.add_arc(e["center"], e["radius"], math.degrees(start) % 360, math.degrees(end) % 360,
                          dxfattribs={"layer": "MAIN_PROFILE"})
    doc.saveas(path)


def _readback_dxf(path, expected, parameters):
    doc = ezdxf.readfile(path)
    records = list(doc.modelspace())
    errors = []
    if len(records) != len(expected):
        return {"passed": False, "issues": ["DXF entity count mismatch"]}
    reconstructed, errors = [], []
    for actual, prior in zip(records, expected):
        if actual.dxftype() != prior["type"]:
            return {"passed": False, "issues": ["DXF entity type mismatch"]}
        entity = dict(prior)
        if prior["type"] == "LINE":
            entity.update(start=_point(actual.dxf.start), end=_point(actual.dxf.end))
        else:
            start, end = actual.start_point, actual.end_point
            if prior["clockwise"]:
                start, end = end, start
            entity.update(start=_point(start), end=_point(end), center=_point(actual.dxf.center), radius=float(actual.dxf.radius))
            errors.append(abs(entity["radius"] - prior["radius"]))
            errors.append(math.dist(entity["center"], prior["center"]))
        errors.extend(math.dist(entity[key], prior[key]) for key in ("start", "end"))
        reconstructed.append(entity)
    report = validate_entities(reconstructed, parameters)
    report.update(entity_count=len(records), units_mm=doc.header.get("$INSUNITS") == 4,
                  max_roundtrip_error_mm=max(errors, default=0))
    report["passed"] = report["passed"] and report["units_mm"] and report["max_roundtrip_error_mm"] <= LINEAR_TOLERANCE_MM
    return report


def _export_svg(entities, bounds, path):
    def xy(point):
        return f"{point[0]:.9f},{-point[1]:.9f}"
    commands = ["M " + xy(entities[0]["start"])]
    for e in entities:
        if e["type"] == "LINE":
            commands.append("L " + xy(e["end"]))
        else:
            commands.append(f"A {e['radius']:.9f},{e['radius']:.9f} 0 {int(_sweep(e) > math.pi)} {int(e['clockwise'])} " + xy(e["end"]))
    commands.append("Z")
    width, height = bounds["max_x"] - bounds["min_x"], bounds["max_y"] - bounds["min_y"]
    padding = max(width, height) * .06
    view = f"{bounds['min_x']-padding} {-bounds['max_y']-padding} {width+2*padding} {height+2*padding}"
    content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" role="img" aria-label="293参数化主轮廓">
<title>尺寸驱动轮廓；含明确校准形状先验与简化踏面</title>
<rect x="{bounds['min_x']-padding}" y="{-bounds['max_y']-padding}" width="{width+2*padding}" height="{height+2*padding}" fill="#f7f5ef"/>
<path d="{' '.join(commands)}" fill="#dbe9e2" stroke="#087f72" stroke-width="1.6" vector-effect="non-scaling-stroke"/>
</svg>'''
    path.write_text(content, encoding="utf-8")


def solve_profile(parameters: dict, output_dir: Path) -> dict:
    """Rebuild a profile from explicit dimensions and persist inspectable artifacts.

    Missing/invalid parameters raise ValueError with an actionable explanation.
    Numerical validity does not confirm the template's engineering assumptions.
    """
    p = _parameters(parameters)
    entities, construction = _construct(p)
    validation = validate_entities(entities, p)
    bounds = _bounds(entities)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _export_dxf(entities, output_dir / "drawing.dxf")
    validation["dxf_readback"] = _readback_dxf(output_dir / "drawing.dxf", entities, p)
    if not validation["dxf_readback"]["passed"]:
        validation["passed"] = False
        validation["issues"].append("Exported DXF readback validation failed")
    _export_svg(entities, bounds, output_dir / "preview.svg")
    schema = template_schema()
    result = {"template_id": schema["id"], "template_version": schema["version"],
        "calibration_case": schema["calibration_case"], "entities": entities, "parameters": p,
        "validation": validation, "assumptions": schema["assumptions"], "shape_priors": schema["shape_priors"],
        "construction": construction, "bounds": bounds,
        "coordinate_system": {"units": "mm", "x": "radial", "y": "axial upward"},
        "artifacts": {"dxf": "drawing.dxf", "svg": "preview.svg", "validation": "validation.json", "model": "model.json"}}
    (output_dir / "model.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result

"""Independent DXF evaluation; never imported by prediction geometry/provider code.

Coordinates are compared in millimetres without registration, translation, scale
fitting, or reading references into a prediction. Entity count alone is not a
geometry score. The reference itself may contain reconstructed assumptions.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import ezdxf
import numpy as np

_NON_PROFILE = re.compile(r"(?:CONSTRUCTION|CONSTRUCT|DEBUG|MEASURE|DIMENSION|ANNOTATION|CENTERLINE|CENTRELINE|DATUM|GUIDE|HATCH|META)", re.I)
_ASSUMED = re.compile(r"(?:CLOSURE|SIMPLIFIED|FALLBACK|SOFT|TREAD_GOST.*REF)", re.I)


def _curves(path: Path, layers: set[str] | None = None) -> tuple[list[dict], dict]:
    doc = ezdxf.readfile(path)
    units = int(doc.units)
    factor = ezdxf.units.conversion_factor(units, ezdxf.units.MM) if units else 1.0
    curves, skipped, errors = [], Counter(), []
    for original in doc.modelspace():
        layer = str(original.dxf.layer)
        if _NON_PROFILE.search(layer) or (layers is not None and layer not in layers):
            skipped[f"layer:{layer}"] += 1
            continue
        if original.dxftype() in {"LWPOLYLINE", "POLYLINE"}:
            entities = list(original.virtual_entities())
        else:
            entities = [original]
        for entity in entities:
            kind = entity.dxftype()
            if kind not in {"LINE", "ARC", "CIRCLE"}:
                if kind in {"SPLINE", "ELLIPSE", "INSERT", "MESH", "3DFACE"}:
                    errors.append(f"Unsupported potentially geometric entity: {kind} on {layer}")
                skipped[f"type:{kind}"] += 1
                continue
            curve = {"kind": kind, "layer": layer, "assumed": bool(_ASSUMED.search(layer))}
            if kind == "LINE":
                start = np.asarray(entity.dxf.start, dtype=float) * factor
                end = np.asarray(entity.dxf.end, dtype=float) * factor
                if max(abs(start[2]), abs(end[2])) > 1e-7:
                    errors.append("Non-planar LINE cannot be evaluated as a 2D profile")
                curve.update(start=start[:2], end=end[:2])
                curve["length"] = float(np.linalg.norm(end[:2] - start[:2]))
            else:
                center = np.asarray(entity.dxf.center, dtype=float) * factor
                extrusion = np.asarray(entity.dxf.extrusion, dtype=float)
                if abs(center[2]) > 1e-7 or not np.allclose(extrusion, [0, 0, 1]):
                    errors.append("Non-default ARC/CIRCLE plane is not supported")
                radius = float(entity.dxf.radius) * factor
                start_angle = math.radians(float(entity.dxf.start_angle)) if kind == "ARC" else 0.0
                sweep = math.radians((float(entity.dxf.end_angle) - float(entity.dxf.start_angle)) % 360) if kind == "ARC" else 2 * math.pi
                curve.update(center=center[:2], radius=radius, angle=start_angle, sweep=sweep)
                curve["start"] = center[:2] + radius * np.array([math.cos(start_angle), math.sin(start_angle)])
                curve["end"] = center[:2] + radius * np.array([math.cos(start_angle + sweep), math.sin(start_angle + sweep)])
                curve["length"] = radius * sweep
            values = [curve["length"], *curve["start"], *curve["end"]]
            if not all(math.isfinite(float(v)) for v in values) or curve["length"] <= 0:
                errors.append(f"Non-finite or degenerate {kind} on {layer}")
                continue
            curves.append(curve)
            if len(curves) > 10_000:
                raise ValueError("DXF exceeds the 10000-primitive evaluation limit")
    return curves, {
        "entities": len(curves), "layers": sorted({c["layer"] for c in curves}),
        "types": dict(Counter(c["kind"] for c in curves)), "excluded": dict(skipped),
        "assumed_layers": sorted({c["layer"] for c in curves if c["assumed"]}),
        "source_units": units, "millimetre_conversion_factor": factor,
        "unit_assumption": "unspecified DXF units treated as mm" if not units else None,
        "issues": errors,
    }


def _samples(curves: list[dict], step: float) -> np.ndarray:
    counts = [max(2, math.ceil(c["length"] / step) + 1) for c in curves]
    if sum(counts) > 300_000:
        raise ValueError("Required sampling exceeds 300000 points; use a reviewed smaller scope")
    chunks = []
    for curve, count in zip(curves, counts):
        t = np.linspace(0, 1, count)
        if curve["kind"] == "LINE":
            chunks.append(curve["start"] + t[:, None] * (curve["end"] - curve["start"]))
        else:
            angle = curve["angle"] + t * curve["sweep"]
            chunks.append(curve["center"] + curve["radius"] * np.column_stack([np.cos(angle), np.sin(angle)]))
    return np.concatenate(chunks)


def _distances(points: np.ndarray, target: list[dict]) -> np.ndarray:
    """Distance to exact line segments and trimmed arcs, not to point clouds."""
    best = np.full(len(points), np.inf)
    for curve in target:
        if curve["kind"] == "LINE":
            direction = curve["end"] - curve["start"]
            t = np.clip(np.sum((points - curve["start"]) * direction, axis=1) / np.dot(direction, direction), 0, 1)
            distance = np.linalg.norm(points - (curve["start"] + t[:, None] * direction), axis=1)
        else:
            relative = points - curve["center"]
            angle = np.mod(np.arctan2(relative[:, 1], relative[:, 0]) - curve["angle"], 2 * np.pi)
            radial = np.abs(np.linalg.norm(relative, axis=1) - curve["radius"])
            endpoints = np.minimum(np.linalg.norm(points - curve["start"], axis=1), np.linalg.norm(points - curve["end"], axis=1))
            distance = np.where((angle <= curve["sweep"] + 1e-12) | (angle >= 2 * np.pi - 1e-12), radial, endpoints)
        best = np.minimum(best, distance)
    return best


def compare_dxf(
    prediction: Path, reference: Path, *, tolerance_mm: float = 0.1,
    sample_step_mm: float = 0.25, prediction_layers: set[str] | None = None,
    reference_layers: set[str] | None = None, partial_reference: bool = False,
) -> dict:
    """Evaluate a declared scope, with no best-fit alignment.

    By default include main-profile assumption/closure layers but report their
    scope separately; exclude construction and measurement layers. A partial
    reference cannot pass a complete-profile benchmark. Explicit layer scopes
    must be chosen before examining predictions. ``passed`` is geometric match
    only and never certifies engineering truth or held-out generalization.
    """
    if not math.isfinite(tolerance_mm) or tolerance_mm <= 0 or not math.isfinite(sample_step_mm) or sample_step_mm <= 0:
        raise ValueError("Evaluation tolerances must be finite and positive")
    actual_step = min(sample_step_mm, tolerance_mm / 2)
    result = {
        "passed": False, "status": "not_compared", "tolerance_mm": tolerance_mm,
        "method": "symmetric sampled-to-exact-curve distance in declared mm coordinates; no alignment",
        "sample_step_mm": actual_step, "sampling_upper_bound_mm": actual_step / 2,
        "engineering_certified": False, "notes": [],
    }
    try:
        predicted, pinfo = _curves(Path(prediction), prediction_layers)
        expected, rinfo = _curves(Path(reference), reference_layers)
        result["selection"] = {"prediction": pinfo, "reference": rinfo}
        partial = partial_reference or any(x in Path(reference).stem.lower() for x in ("scored_only", "strict_body"))
        result["scope"] = "partial_reference" if partial else "main_profile_including_reference_assumptions"
        if partial:
            result["notes"].append("Partial reference: reference coverage is diagnostic; complete-profile passed remains false.")
        if rinfo["assumed_layers"]:
            result["notes"].append("Reference has explicitly named assumption/closure layers; full-profile error includes them. Core directed error is reported separately.")
        if "293" in Path(reference).name:
            result["notes"].append("293 reference has an unlabeled fitted bridge and simplified tread on GT_MAIN. These cannot be separated by layer and are not dimension-certified ground truth.")
        if pinfo["issues"] or rinfo["issues"] or not predicted or not expected:
            result.update(status="invalid_geometry", issues=pinfo["issues"] + rinfo["issues"] + ([] if predicted and expected else ["Empty profile selection"]))
            return result
        ppoints, rpoints = _samples(predicted, actual_step), _samples(expected, actual_step)
        p_to_r = _distances(ppoints, expected)
        r_to_p = _distances(rpoints, predicted)
        maximum = max(float(p_to_r.max()), float(r_to_p.max()))
        length_p, length_r = sum(c["length"] for c in predicted), sum(c["length"] for c in expected)
        length_difference = abs(length_p - length_r)
        length_tolerance = max(tolerance_mm * 2, length_r * 0.001)
        match = maximum + actual_step / 2 <= tolerance_mm and length_difference <= length_tolerance
        result.update(
            status="compared", passed=bool(match and not partial),
            scope_geometry_match=bool(match),
            symmetric_max_error_mm=maximum,
            conservative_max_error_mm=maximum + actual_step / 2,
            rms_error_mm=float(np.sqrt(np.mean(np.concatenate([p_to_r, r_to_p]) ** 2))),
            prediction_to_reference_max_mm=float(p_to_r.max()),
            reference_to_prediction_max_mm=float(r_to_p.max()),
            prediction_length_mm=length_p, reference_length_mm=length_r,
            length_difference_mm=length_difference, length_tolerance_mm=length_tolerance,
            prediction_samples=len(ppoints), reference_samples=len(rpoints),
        )
        core = [c for c in expected if not c["assumed"]]
        if core:
            core_distances = _distances(_samples(core, actual_step), predicted)
            result["core_reference_coverage"] = {
                "max_error_mm": float(core_distances.max()),
                "rms_error_mm": float(np.sqrt(np.mean(core_distances ** 2))),
                "direction": "reference_to_prediction_only", "complete_profile_pass": False,
                "note": "Does not penalize extra predicted geometry and must not replace full-profile comparison.",
            }
        return result
    except (OSError, ValueError, ezdxf.DXFError, OverflowError) as exc:
        result.update(status="comparison_error", issues=[f"{type(exc).__name__}: {exc}"])
        return result


def summarize_online_trials(protocol_trials: list[dict], calibration_runs: list[dict], *, online_requested: bool) -> dict:
    """Aggregate every logical provider call without hiding retries/failed runs.

    Receipts are provider summaries, not raw HTTP bodies. Missing historical
    receipt fields remain unknown; they never become zero attempts or success.
    The no-key fallback fixture is intentionally outside this online measure.
    """
    receipts = []
    if online_requested:
        receipts.extend(("protocol", row) for row in protocol_trials)
        receipts.extend(("calibration", run.get("review_snapshot", {}).get("provider", {})) for run in calibration_runs)
    logical_count = len(receipts)
    http_success = sum(row.get("http_success") is True for _, row in receipts)
    http_unknown = sum(not isinstance(row.get("http_success"), bool) for _, row in receipts)
    schema_success = sum(row.get("schema_success") is True for _, row in receipts)
    schema_unknown = sum(not isinstance(row.get("schema_success"), bool) for _, row in receipts)
    known_attempts = [row["network_requests"] for _, row in receipts
                      if isinstance(row.get("network_requests"), int) and not isinstance(row["network_requests"], bool)
                      and row["network_requests"] >= 0]
    unknown_attempt_receipts = logical_count - len(known_attempts)
    attempts = sum(known_attempts) if not unknown_attempt_receipts else None
    return {
        "logical_calls": {
            "total": logical_count,
            "protocol_trials": sum(stage == "protocol" for stage, _ in receipts),
            "calibration_trials": sum(stage == "calibration" for stage, _ in receipts),
            "http_successful": http_success,
            "http_unknown": http_unknown,
            "http_success_rate": http_success / logical_count if logical_count and not http_unknown else None,
            "schema_successful": schema_success,
            "schema_unknown": schema_unknown,
            "schema_success_rate": schema_success / logical_count if logical_count and not schema_unknown else None,
        },
        "network_attempts": {
            "total": attempts,
            "known_total": sum(known_attempts),
            "receipts_missing_attempt_count": unknown_attempt_receipts,
            "logical_calls_with_attempts": sum(count > 0 for count in known_attempts),
            "retries": sum(max(0, count - 1) for count in known_attempts) if not unknown_attempt_receipts else None,
        },
        "scope": "Every protocol and calibration logical provider call across all repeats, including failed calls; excludes offline runs and the no-key fallback fixture.",
        "attempt_policy": "network_requests counts actual client.post attempts, including retries and attempts that fail before receiving a response. HTTP success rates are per logical call, not per network attempt; per-attempt HTTP status receipts are not retained.",
        "missing_receipt_policy": "Missing receipt fields remain unknown; aggregate attempt totals or rates requiring them are null.",
    }


def summarize_evaluation(catalog: dict, results: list[dict]) -> dict:
    """Keep all catalog cases in denominator; do not credit manual/unsupported runs.

    Result rows: case_id, status, provider.attempted/transport_success,
    parameter_accuracy.correct/total, validation.passed, comparison.passed,
    manual_confirmation (bool), automatic_success (bool). Missing booleans do not
    become success. Duplicate rows are rejected to prevent denominator inflation.
    """
    cases = catalog.get("cases", [])
    indexed = {}
    for row in results:
        case_id = row.get("case_id", row.get("id"))
        if case_id in indexed:
            raise ValueError(f"Duplicate evaluation row for {case_id}")
        indexed[case_id] = row
    known = {c["id"] for c in cases}
    if set(indexed) - known:
        raise ValueError("Evaluation contains a case outside the frozen catalog")

    def aggregate(subset):
        total = len(subset)
        counters = Counter()
        statuses = Counter()
        for case in subset:
            row = indexed.get(case["id"], {})
            status = row.get("status", "not_evaluated")
            statuses[status] += 1
            counters["evaluated"] += bool(row)
            counters["supported"] += case.get("supported_template") is not None
            provider = row.get("provider") or {}
            counters["online_attempted"] += provider.get("attempted") is True
            counters["online_transport_success"] += provider.get("transport_success") is True
            counters["geometry_valid"] += (row.get("validation") or {}).get("passed") is True
            counters["reference_geometry_pass"] += (row.get("comparison") or {}).get("passed") is True
            manual = row.get("manual_confirmation") is True or row.get("manually_confirmed") is True
            counters["manual_confirmation"] += manual
            automatic = row.get("automatic_success") is True and not manual and status == "completed" and case.get("supported_template") is not None and (row.get("validation") or {}).get("passed") is True and (row.get("comparison") or {}).get("passed") is True
            counters["automatic_success"] += automatic
            accuracy = row.get("parameter_accuracy") or {}
            if isinstance(accuracy.get("correct"), int) and isinstance(accuracy.get("total"), int) and 0 <= accuracy["correct"] <= accuracy["total"]:
                counters["parameter_correct"] += accuracy["correct"]
                counters["parameter_total"] += accuracy["total"]
        answer = {key: counters[key] for key in ("evaluated", "supported", "online_attempted", "online_transport_success", "geometry_valid", "reference_geometry_pass", "manual_confirmation", "automatic_success", "parameter_correct", "parameter_total")}
        answer.update(total=total, unsupported=total - counters["supported"], not_evaluated=total - counters["evaluated"], statuses=dict(statuses))
        answer.update(
            coverage=counters["supported"] / total if total else 0.0,
            automatic_success_rate=counters["automatic_success"] / total if total else 0.0,
            geometry_valid_rate=counters["geometry_valid"] / total if total else 0.0,
            reference_geometry_pass_rate=counters["reference_geometry_pass"] / total if total else 0.0,
            online_transport_success_rate=counters["online_transport_success"] / counters["online_attempted"] if counters["online_attempted"] else None,
            parameter_accuracy=counters["parameter_correct"] / counters["parameter_total"] if counters["parameter_total"] else None,
        )
        return answer

    return {
        "overall": aggregate(cases),
        "calibration": aggregate([c for c in cases if c.get("split") == "calibration"]),
        "holdout": aggregate([c for c in cases if c.get("split") == "holdout"]),
        "denominator_policy": "All catalog cases remain in their split, including unsupported, untested, missing reference and manual review cases.",
        "parameter_accuracy_policy": "Only explicitly scored labeled fields contribute; unscored/unknown fields are not assumed correct.",
        "transport_policy": "This case summary uses the last reported run per attempted case, not all logical calls or HTTP attempts. Use online_request_summary for all protocol/calibration trials and retries. Transport success is independent of geometry, coverage, and automation success.",
    }

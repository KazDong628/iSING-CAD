"""Independent, origin-free evaluation of autonomous image reconstructions.

Reference geometry is consumed here only AFTER prediction export. This module
does not repair predictions, call providers, or feed registered coordinates back
to a runtime agent. Registration NEVER estimates scale or a deformable warp.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import ezdxf
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .evaluation import _curves, _distances, _samples

STRICT_TOLERANCE_MM = 0.1
INITIAL_SMOKE_CASES = (
    "044-main", "solid-arrow-ping__img_000293",
    "solid-arrow-ping__img_000186-main", "solid-arrow-ping__img_000200-main",
    "solid-arrow-ping__img_000253", "solid-arrow-ping__img_000260",
    "solid-arrow-ping__img_000292",
)

D4 = (
    ("identity", ((1, 0), (0, 1))),
    ("quarter_turn", ((0, -1), (1, 0))),
    ("half_turn", ((-1, 0), (0, -1))),
    ("three_quarter_turn", ((0, 1), (-1, 0))),
    ("reflect_x", ((-1, 0), (0, 1))),
    ("reflect_y", ((1, 0), (0, -1))),
    ("swap_axes", ((0, 1), (1, 0))),
    ("swap_and_reflect", ((0, -1), (-1, 0))),
)


def _axis_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("Axis transform must be a finite 2x2 signed permutation")
    if not any(np.array_equal(matrix, candidate) for _, candidate in D4):
        raise ValueError("Only declared axis swaps/sign flips are allowed; scale and arbitrary rotations are forbidden")
    return matrix


def _transform(curves, matrix, translation=(0.0, 0.0)):
    offset = np.asarray(translation, dtype=float)
    reflection = np.linalg.det(matrix) < 0
    result = []
    for original in curves:
        curve = dict(original)
        curve["start"] = matrix @ original["start"] + offset
        curve["end"] = matrix @ original["end"] + offset
        if original["kind"] != "LINE":
            curve["center"] = matrix @ original["center"] + offset
            if reflection:
                curve["start"], curve["end"] = curve["end"], curve["start"]
            direction = curve["start"] - curve["center"]
            curve["angle"] = math.atan2(direction[1], direction[0])
        result.append(curve)
    return result


def _closest(points, curves):
    """Exact closest points on finite lines and trimmed circular arcs."""
    best = np.full(len(points), np.inf)
    closest = np.zeros_like(points)
    for curve in curves:
        if curve["kind"] == "LINE":
            direction = curve["end"] - curve["start"]
            t = np.clip(np.sum((points - curve["start"]) * direction, axis=1) / np.dot(direction, direction), 0, 1)
            candidate = curve["start"] + t[:, None] * direction
        else:
            relative = points - curve["center"]
            length = np.linalg.norm(relative, axis=1)
            unit = np.divide(relative, length[:, None], out=np.zeros_like(relative), where=length[:, None] > 1e-15)
            candidate = curve["center"] + curve["radius"] * unit
            angle = np.mod(np.arctan2(relative[:, 1], relative[:, 0]) - curve["angle"], 2 * np.pi)
            outside = ((angle > curve["sweep"] + 1e-12) & (angle < 2 * np.pi - 1e-12)) | (length <= 1e-15)
            start_distance = np.sum((points - curve["start"]) ** 2, axis=1)
            end_distance = np.sum((points - curve["end"]) ** 2, axis=1)
            endpoints = np.where((start_distance <= end_distance)[:, None], curve["start"], curve["end"])
            candidate = np.where(outside[:, None], endpoints, candidate)
        distance = np.sum((points - candidate) ** 2, axis=1)
        better = distance < best
        closest[better] = candidate[better]
        best[better] = distance[better]
    return closest


def _fit_translation(predicted, expected, ppoints, rpoints):
    """Two deterministic starts; optimize two translation variables only."""
    centers = (rpoints.min(axis=0) + rpoints.max(axis=0) - ppoints.min(axis=0) - ppoints.max(axis=0)) / 2
    centroid = rpoints.mean(axis=0) - ppoints.mean(axis=0)
    reference_tree, prediction_tree = cKDTree(rpoints), cKDTree(ppoints)

    def objective(offset):
        p = ppoints + offset
        r = rpoints - offset
        residual_p = p - rpoints[reference_tree.query(p)[1]]
        residual_r = r - ppoints[prediction_tree.query(r)[1]]
        loss = .5 * (np.mean(np.sum(residual_p ** 2, axis=1)) + np.mean(np.sum(residual_r ** 2, axis=1)))
        gradient = residual_p.mean(axis=0) - residual_r.mean(axis=0)
        return float(loss), gradient

    choices = []
    for start in (centers, centroid):
        fit = minimize(objective, start, method="L-BFGS-B", jac=True, options={"maxiter": 45, "maxfun": 100, "ftol": 1e-12, "gtol": 1e-8})
        if np.isfinite(fit.x).all() and math.isfinite(float(fit.fun)):
            choices.append((float(fit.fun), fit.x, bool(fit.success), int(fit.nit)))
    if not choices:
        raise ValueError("Translation registration did not return a finite estimate")
    return min(choices, key=lambda x: (x[0], tuple(x[1])))


def _refine_translation(predicted, expected, ppoints, rpoints, initial):
    """Refine only the selected orientation against exact native curves.

    Point-cloud search makes all eight alternatives affordable for dense raster
    polylines. This final curve-based step removes point-cloud snap bias without
    ever fitting scale or changing relative vertex positions.
    """
    def objective(offset):
        p, r = ppoints + offset, rpoints - offset
        residual_p = p - _closest(p, expected)
        residual_r = r - _closest(r, predicted)
        loss = .5 * (np.mean(np.sum(residual_p ** 2, axis=1)) + np.mean(np.sum(residual_r ** 2, axis=1)))
        return float(loss), residual_p.mean(axis=0) - residual_r.mean(axis=0)
    fit = minimize(objective, initial, method="L-BFGS-B", jac=True, options={"maxiter": 25, "maxfun": 60, "ftol": 1e-12, "gtol": 1e-8})
    if np.isfinite(fit.x).all() and math.isfinite(float(fit.fun)) and fit.fun <= objective(initial)[0]:
        return fit.x, bool(fit.success), int(fit.nit)
    return initial, False, 0


def _metrics(predicted, expected, ppoints, rpoints, step, tolerance):
    p_distance = _distances(ppoints, expected)
    r_distance = _distances(rpoints, predicted)
    combined = np.concatenate((p_distance, r_distance))
    length_p, length_r = sum(c["length"] for c in predicted), sum(c["length"] for c in expected)
    maximum = float(combined.max())
    # Conservative half-step bound prevents a sampled maximum from silently
    # claiming sub-sampling precision. Length check rejects duplicate curves.
    length_tolerance = max(2 * tolerance, .001 * length_r)
    return {
        "max_error_mm": maximum, "conservative_max_error_mm": maximum + step / 2,
        "p95_error_mm": float(np.quantile(combined, .95)),
        "rms_error_mm": float(np.sqrt(np.mean(combined ** 2))),
        "prediction_to_reference_max_mm": float(p_distance.max()),
        "reference_to_prediction_max_mm": float(r_distance.max()),
        "prediction_length_mm": float(length_p), "reference_length_mm": float(length_r),
        "length_error_mm": float(abs(length_p - length_r)),
        "length_error_fraction": float(abs(length_p - length_r) / length_r),
        "length_tolerance_mm": length_tolerance,
        "reference_match_at_tolerance": bool(maximum + step / 2 <= tolerance and abs(length_p - length_r) <= length_tolerance),
        "prediction_sample_count": len(ppoints), "reference_sample_count": len(rpoints),
        "sample_distribution": "approximately uniform arclength per primitive; endpoints retained",
    }


def _candidate_validity(curves, info):
    """Finite primitive/readback and closed endpoint degree audit, without GT."""
    if not curves or info["issues"]:
        return {"passed": False, "primitive_valid": False, "closed": False, "issues": info["issues"] or ["No contour primitives"]}
    points = np.asarray([point for curve in curves for point in (curve["start"], curve["end"])])
    neighbors = cKDTree(points).query_ball_point(points, 1e-5)
    invalid_degree = sum(len(group) != 2 for group in neighbors)
    return {
        "passed": invalid_degree == 0, "primitive_valid": True,
        "closed": invalid_degree == 0, "non_degree_two_endpoints": invalid_degree,
        "endpoint_tolerance_mm": 1e-5,
        "self_intersections_checked": False,
        "meaning": "Finite LINE/ARC/CIRCLE primitives and closed endpoint-degree audit only; self-intersections and engineering constraints require runtime validation.",
        "issues": [] if invalid_degree == 0 else ["Boundary is open, branched, or contains duplicate joins at the declared tolerance"],
    }


def evaluate_autonomous_artifact(
    dxf_path: Path, reference_path: Path | None, *,
    declared_axis_transform=None, declared_translation_mm=None,
    tolerance_mm: float = STRICT_TOLERANCE_MM, sample_step_mm: float = .05,
) -> dict:
    """Compare mm-scale geometry while disclosing all evaluation registration.

    Default: search the eight D4 axis conventions and fit translation, labeling
    the result ``shape_diagnostic``. No scale, shear, arbitrary rotation or
    prediction-coordinate changes are permitted. Supplying a signed permutation
    fixes the axis convention; supplying translation as well fixes registration.
    These declarations must be frozen before prediction results are inspected.
    Unitless predictions are never credited with physical/reference success.
    """
    if not math.isfinite(tolerance_mm) or tolerance_mm <= 0 or not math.isfinite(sample_step_mm) or sample_step_mm <= 0:
        raise ValueError("Positive finite tolerances required")
    step = min(sample_step_mm, tolerance_mm / 2)
    matrix = _axis_matrix(declared_axis_transform) if declared_axis_transform is not None else None
    fixed_translation = None
    if declared_translation_mm is not None:
        if matrix is None:
            raise ValueError("A declared translation requires a declared axis convention")
        fixed_translation = np.asarray(declared_translation_mm, dtype=float)
        if fixed_translation.shape != (2,) or not np.isfinite(fixed_translation).all():
            raise ValueError("Declared translation must be two finite millimetre coordinates")
    output = {
        "protocol": "autonomous-image-reference-v1", "status": "not_compared",
        "artifact_completed": False, "geometry_valid": False, "scaled_mm": False,
        "reference_compared": False, "reference_within_0_1mm": False,
        "reference_within_tolerance": False, "engineering_verified": False,
        "tolerance_mm": tolerance_mm, "sample_step_mm": step,
        "sampling_upper_bound_mm": step / 2, "scale_fit_permitted": False,
        "prediction_mutated": False, "notes": [],
    }
    try:
        predicted, pinfo = _curves(Path(dxf_path))
        output["artifact_completed"] = True
        output["prediction_info"] = pinfo
        output["candidate_validation"] = _candidate_validity(predicted, pinfo)
        output["geometry_valid"] = output["candidate_validation"]["passed"]
        output["scaled_mm"] = pinfo["source_units"] != 0
        if not predicted or pinfo["issues"]:
            output.update(status="invalid_prediction", issues=pinfo["issues"] or ["No prediction geometry"])
            return output
        if not output["scaled_mm"]:
            output.update(status="unscaled_prediction", issues=["Prediction has no DXF unit declaration; millimetre accuracy cannot be scored."])
            return output
        output["notes"].append("Unit metadata permits conversion to mm but does not independently prove that the source-derived scale was correct.")
        if reference_path is None or not Path(reference_path).is_file():
            output.update(status="missing_reference")
            return output
        expected, rinfo = _curves(Path(reference_path))
        output["reference_info"] = rinfo
        if not expected or rinfo["issues"] or rinfo["source_units"] == 0:
            output.update(status="invalid_reference", issues=rinfo["issues"] or ["Reference is empty or its physical units are unspecified"])
            return output
        partial = any(token in Path(reference_path).stem.lower() for token in ("scored_only", "strict_body"))
        output["reference_scope"] = "partial_profile" if partial else "complete_file_profile_including_reference_assumptions"
        output["notes"].append("Reference scope and nominal-versus-midpoint policy must be frozen separately. Default comparison includes reference closure/assumption layers; it does not certify true tread geometry.")
        if "293" in Path(reference_path).name:
            output["notes"].append("293 is an exposed calibration case with fitted and simplified reference geometry; it is not held-out generalization evidence.")
        if partial:
            output["notes"].append("Partial reference cannot establish full-profile reference success.")
        ppoints, rpoints = _samples(predicted, step), _samples(expected, step)
        output["direct_metrics"] = _metrics(predicted, expected, ppoints, rpoints, step, tolerance_mm)
        output["direct_metrics"]["coordinate_convention_verified"] = matrix is not None and fixed_translation is not None and np.array_equal(matrix, np.eye(2)) and np.array_equal(fixed_translation, [0., 0.])
        # Coarse registration is bounded independently of native line density.
        # Final scoring always uses dense samples to the exact native curves.
        total_length = max(sum(c["length"] for c in predicted), sum(c["length"] for c in expected))
        coarse_step = max(.25, total_length / 700)
        pc, rc = _samples(predicted, coarse_step), _samples(expected, coarse_step)
        if len(pc) > 12_000 or len(rc) > 12_000:
            raise ValueError("Too many native fragments for bounded registration")
        candidates = []
        alternatives = [("declared", matrix)] if matrix is not None else [(name, np.asarray(values, dtype=float)) for name, values in D4]
        for name, axis in alternatives:
            transformed = _transform(predicted, axis)
            transformed_points = pc @ axis.T
            if fixed_translation is None:
                loss, offset, converged, iterations = _fit_translation(transformed, expected, transformed_points, rc)
            else:
                offset, converged, iterations = fixed_translation, True, 0
                residual = _distances(transformed_points + offset, expected)
                loss = float(np.mean(residual ** 2))
            candidates.append((loss, name, axis, offset, converged, iterations))
        loss, name, axis, offset, converged, iterations = min(candidates, key=lambda item: item[0])
        if fixed_translation is None:
            offset, converged, refinement_iterations = _refine_translation(_transform(predicted, axis), expected, pc @ axis.T, rc, offset)
            iterations += refinement_iterations
        registered = _transform(predicted, axis, offset)
        metrics = _metrics(registered, expected, ppoints @ axis.T + offset, rpoints, step, tolerance_mm)
        alignment_kind = "shape_diagnostic" if matrix is None else "declared_axes_origin_free" if fixed_translation is None else "declared_alignment"
        output.update(
            status="compared", reference_compared=True, alignment_kind=alignment_kind,
            registered_metrics=metrics,
            transform={"axis_name": name, "matrix": axis.tolist(), "translation_mm": offset.tolist(), "scale": 1.0,
                       "axis_searched": matrix is None, "translation_fitted": fixed_translation is None,
                       "optimizer_converged": converged, "optimizer_iterations": iterations,
                       "coarse_registration_step_mm": coarse_step, "candidates_evaluated": len(candidates)},
        )
        if matrix is None:
            output["notes"].append("D4 reflection/axis selection used reference geometry. This is an origin-free shape diagnostic; orientation and engineering correctness are not verified.")
        elif fixed_translation is None:
            output["notes"].append("Axis convention was declared, but translation was fitted after prediction. Absolute origin placement is not evaluated.")
        numeric_match = metrics["reference_match_at_tolerance"] and output["geometry_valid"] and not partial
        output["reference_within_tolerance"] = bool(numeric_match)
        # Preserve the frozen 0.1 mm field even if a caller requests another
        # diagnostic tolerance. A looser tolerance cannot relabel a strict pass.
        strict_length_tolerance = max(.2, .001 * metrics["reference_length_mm"])
        output["reference_within_0_1mm"] = bool(
            output["geometry_valid"] and not partial
            and metrics["conservative_max_error_mm"] <= STRICT_TOLERANCE_MM
            and metrics["length_error_mm"] <= strict_length_tolerance
        )
        output["strict_match_meaning"] = "Reference shape agreement at 0.1 mm after disclosed registration; not engineering approval or verified annotation coverage."
        return output
    except (OSError, ValueError, ezdxf.DXFError, OverflowError) as exc:
        output.update(status="evaluation_error", issues=[f"{type(exc).__name__}: {exc}"])
        return output


def summarize_autonomous_evaluation(catalog: dict, results: list[dict]) -> dict:
    """Count every catalog case, without consulting legacy template support.

    Rows contain case_id, attempted, artifact_completed, geometry_valid,
    scaled_mm, manual_intervention, and comparison (the helper result). Manual
    confirmation objects also count as intervention. No result means unattempted.
    Metrics reported from a comparison can supply artifact/geometry/unit flags
    when the corresponding row field is absent.
    """
    cases = catalog.get("cases", [])
    known = {case["id"] for case in cases}
    if len(known) != len(cases):
        raise ValueError("Frozen catalog contains duplicate case identifiers")
    indexed = {}
    for row in results:
        key = row.get("case_id", row.get("id"))
        if key not in known:
            raise ValueError("Autonomous evaluation row is outside the frozen catalog")
        if key in indexed:
            raise ValueError(f"Duplicate autonomous evaluation row for {key}")
        indexed[key] = row

    def aggregate(subset):
        counts = Counter()
        statuses = Counter()
        for case in subset:
            row = indexed.get(case["id"], {})
            comparison = row.get("comparison") or {}
            attempted = row.get("attempted") is True
            artifact = row.get("artifact_completed", comparison.get("artifact_completed")) is True
            valid = row.get("geometry_valid", comparison.get("geometry_valid")) is True
            scaled = row.get("scaled_mm", comparison.get("scaled_mm")) is True
            manual = bool(row.get("manual_intervention") or row.get("manual_confirmation") or row.get("manual_interventions"))
            compared = comparison.get("reference_compared") is True
            within = comparison.get("reference_within_0_1mm") is True
            counts["attempted"] += attempted
            counts["artifact_completed"] += attempted and artifact
            counts["auto_generated"] += attempted and artifact and not manual
            counts["geometry_valid"] += attempted and valid
            counts["scaled_mm"] += attempted and artifact and scaled
            counts["reference_compared"] += attempted and compared
            counts["reference_within_0_1mm"] += attempted and compared and within and valid and scaled
            counts["autonomous_reference_within_0_1mm"] += attempted and compared and within and valid and scaled and not manual
            counts["manual_interventions"] += manual
            counts["searched_alignment_matches"] += attempted and within and comparison.get("alignment_kind") == "shape_diagnostic"
            statuses[row.get("status", "not_attempted")] += 1
        keys = ("attempted", "artifact_completed", "auto_generated", "geometry_valid", "scaled_mm", "reference_compared", "reference_within_0_1mm", "autonomous_reference_within_0_1mm", "manual_interventions", "searched_alignment_matches")
        total = len(subset)
        summary = {key: counts[key] for key in keys}
        summary.update(total=total, not_attempted=total - counts["attempted"], statuses=dict(statuses))
        for metric in ("auto_generated", "geometry_valid", "reference_within_0_1mm", "autonomous_reference_within_0_1mm"):
            summary[metric + "_rate"] = counts[metric] / total if total else 0.
        return summary

    overall = aggregate(cases)
    return {
        **overall, "overall": overall,
        "previous_calibration": aggregate([case for case in cases if case.get("split") == "calibration"]),
        "holdout": aggregate([case for case in cases if case.get("split") != "calibration"]),
        "mode": "autonomous_image", "template_gate_used": False,
        "denominator_policy": "All frozen catalog cases remain in denominator, including failed/unattempted/manual/missing-reference cases.",
        "auto_generated_meaning": "Artifact exported without manual intervention; separate from geometry validity and reference accuracy.",
        "strict_match_meaning": "0.1 mm reference shape agreement after disclosed unit conversion and registration; searched orientation is diagnostic only.",
        "engineering_verified": False,
    }

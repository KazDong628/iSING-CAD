"""Independently score a finished oracle-mask reconstruction against its DXF GT.

The reconstruction runner never imports this script. GT is opened only after
the exported prediction has passed a local DXF readback. An oracle mask was
derived from the same GT, so this is an exposed development diagnostic rather
than a held-out or independent reference test.

Usage: python scripts/evaluate_oracle_mask_run.py --run-dir runtime/oracle-mask/RUN
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contour_agent.autonomous_evaluation import STRICT_TOLERANCE_MM
from contour_agent.autonomous_qualification import _reference_for_scoring
from contour_agent.dataset import build_catalog
from contour_agent.dxf_comparison import audit_dxf, compare_dxf_entities


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _single_cycle_area(audit: dict) -> dict:
    """Exact Green integral for one closed LINE/ARC/CIRCLE endpoint cycle.

    Area is descriptive. This does not classify nested holes or prove that a
    degree-two drawing is free of self-intersections.
    """
    profile = audit["filtered_profile"]
    connections = audit["connections"]
    if profile["coordinate_unit"] != "mm":
        return {"status": "unknown_units", "area_mm2": None}
    if not connections["closed"] or len(connections["components"]) != 1:
        return {"status": "not_one_closed_cycle", "area_mm2": None}
    entities = {row["id"]: row for row in profile["entities"]}
    edges = {row["entity_id"]: row for row in connections["edges"]}
    adjacent: dict[str, list[str]] = {}
    for edge_id, edge in edges.items():
        adjacent.setdefault(edge["start_node"], []).append(edge_id)
        adjacent.setdefault(edge["end_node"], []).append(edge_id)
    if not edges:
        return {"status": "empty_profile", "area_mm2": None}
    first = edges[sorted(edges)[0]]
    origin = current = first["start_node"]
    visited: set[str] = set()
    twice_signed_area = 0.0
    while len(visited) < len(edges):
        available = sorted({edge_id for edge_id in adjacent[current] if edge_id not in visited})
        if not available or (visited and len(available) != 1):
            return {"status": "ambiguous_cycle", "area_mm2": None}
        edge_id = available[0]
        edge, entity = edges[edge_id], entities[edge_id]
        forward = edge["start_node"] == current
        if entity["type"] == "LINE":
            (x0, y0), (x1, y1) = entity["start"], entity["end"]
            integral = x0 * y1 - x1 * y0
        elif entity["type"] in {"ARC", "CIRCLE"}:
            cx, cy = entity["center"]
            radius = entity["radius"]
            angle = math.radians(entity["start_angle_deg"])
            sweep = math.radians(entity["sweep_deg"])
            end = angle + sweep
            integral = (cx * radius * (math.sin(end) - math.sin(angle))
                        + cy * radius * (math.cos(angle) - math.cos(end))
                        + radius * radius * sweep)
        else:
            return {"status": "unsupported_primitive", "area_mm2": None}
        twice_signed_area += integral if forward else -integral
        visited.add(edge_id)
        current = edge["end_node"] if forward else edge["start_node"]
    if current != origin:
        return {"status": "open_traversal", "area_mm2": None}
    return {"status": "single_closed_cycle", "area_mm2": abs(twice_signed_area) / 2,
            "method": "analytic Green integral over oriented native LINE/ARC/CIRCLE primitives",
            "self_intersections_checked": False, "holes_classified": False}


def _summarize(comparison: dict, *, case_id: str, prediction_sha256: str,
               reference_sha256: str, oracle_mask_sha256: str | None,
               mask_reference_hash_verified: bool) -> dict:
    prediction, reference = comparison["prediction"], comparison["reference"]
    score = comparison["physical_score"]
    registered = score.get("registered_metrics") or {}
    direct = score.get("direct_metrics") or {}
    correspondence = comparison["entity_correspondence"]
    rows = correspondence["rows"]
    accepted = [row for row in rows if row["status"] == "unique_full_parameter_agreement"]
    error_fields = ("endpoint_error_mm", "length_error_mm", "radius_error_mm", "center_error_mm",
                    "undirected_angle_error_deg", "sweep_error_deg")
    errors = {field: max((row["parameter_errors"][field] for row in accepted
                          if field in row["parameter_errors"]), default=None) for field in error_fields}
    pa, ra = _single_cycle_area(prediction), _single_cycle_area(reference)
    area_difference = (abs(pa["area_mm2"] - ra["area_mm2"])
                       if pa["area_mm2"] is not None and ra["area_mm2"] is not None else None)
    ptypes = prediction["filtered_profile"]["types"]
    rtypes = reference["filtered_profile"]["types"]
    type_delta = {kind: ptypes.get(kind, 0) - rtypes.get(kind, 0)
                  for kind in sorted(set(ptypes) | set(rtypes))}
    length_limit = max(.2, .001 * registered["reference_length_mm"]) if registered else None
    direct_length_limit = max(.2, .001 * direct["reference_length_mm"]) if direct else None
    native_frame_01 = bool(direct and score["geometry_valid"]
                           and score.get("reference_scope") != "partial_profile"
                           and direct["conservative_max_error_mm"] <= STRICT_TOLERANCE_MM
                           and direct["length_error_mm"] <= direct_length_limit)
    all_primitive_parameters = bool(comparison["physical_units_available"]
                                    and prediction["filtered_profile"]["count"] == reference["filtered_profile"]["count"]
                                    and len(accepted) == len(rows)
                                    and len(accepted) == reference["filtered_profile"]["count"])
    adjacency_match = None
    if all_primitive_parameters:
        mapping = {row["prediction_id"]: row["reference_id"] for row in accepted}
        prediction_joints = Counter(
            tuple(sorted(mapping[incident["entity_id"]] for incident in node["incidents"]))
            for node in prediction["connections"]["nodes"])
        reference_joints = Counter(
            tuple(sorted(incident["entity_id"] for incident in node["incidents"]))
            for node in reference["connections"]["nodes"])
        adjacency_match = prediction_joints == reference_joints
    return {
        "schema_version": "oracle-mask-dxf-evaluation-v1", "status": "compared" if comparison["physical_units_available"] else "compared_without_physical_units",
        "case_id": case_id, "condition": "GT-derived_oracle_material_mask", "held_out": False,
        "independent_reference_accuracy_verified": False,
        "exposure": "The input mask was derived from the scoring GT. Success here is conditional on an oracle mask and cannot establish source-image segmentation or held-out generalization.",
        "provenance": {"prediction_sha256": prediction_sha256, "reference_sha256": reference_sha256,
                       "oracle_mask_sha256": oracle_mask_sha256,
                       "mask_and_scoring_reference_same_bytes": mask_reference_hash_verified,
                       "online_provider_input_gt_audit": "outside_independent_comparator_scope"},
        "units": {"prediction_dxf_insunits": prediction["source_units"],
                  "reference_dxf_insunits": reference["source_units"],
                  "filtered_primitive_coordinates": "mm" if comparison["physical_units_available"] else "unknown_physical_units",
                  "raw_objects": "each file's original drawing units",
                  "display": comparison["alignment"]["coordinate_unit"]},
        "objects": {"prediction_raw_modelspace_count": prediction["raw_modelspace"]["count"],
                    "reference_raw_modelspace_count": reference["raw_modelspace"]["count"],
                    "prediction_raw_types": prediction["raw_modelspace"]["types"],
                    "reference_raw_types": reference["raw_modelspace"]["types"],
                    "prediction_filtered_count": prediction["filtered_profile"]["count"],
                    "reference_filtered_count": reference["filtered_profile"]["count"],
                    "prediction_filtered_types": ptypes, "reference_filtered_types": rtypes,
                    "filtered_type_count_delta_prediction_minus_reference": type_delta},
        "connectivity": {side: {"closed_endpoint_degree": comparison[side]["connections"]["closed"],
                                "component_count": len(comparison[side]["connections"]["components"]),
                                "non_degree_two_node_count": sum(node["degree"] != 2 for node in comparison[side]["connections"]["nodes"]),
                                "tolerance_drawing_units_or_mm": comparison[side]["connections"]["tolerance"],
                                "coordinate_unit": comparison[side]["connections"]["coordinate_unit"]}
                         for side in ("prediction", "reference")},
        "matched_primitive_adjacency_equal": adjacency_match,
        "geometry": {"registered_shape_diagnostic": registered or None,
                     "native_coordinate_frame": direct or None,
                     "registered_alignment": comparison["alignment"],
                     "symmetric_sampled_hausdorff_mm": registered.get("max_error_mm"),
                     "hausdorff_sampling_upper_bound_mm": score.get("sampling_upper_bound_mm") if registered else None,
                     "prediction_area": pa, "reference_area": ra,
                     "absolute_area_difference_mm2": area_difference,
                     "registered_length_tolerance_mm": length_limit,
                     "registered_shape_within_0_1mm": bool(score.get("reference_within_0_1mm")),
                     "native_coordinate_frame_within_0_1mm": native_frame_01},
        "primitive_matching": {"method": "mutually_unique_same_type_complete_parameter_match_after_disclosed_registration",
                               "tolerance_mm": correspondence.get("parameter_tolerance_mm"),
                               "status": correspondence["status"], "status_counts": correspondence["counts"],
                               "matched_count": len(accepted),
                               "all_filtered_primitives_match": all_primitive_parameters,
                               "unmatched_reference_count": len(correspondence["unmatched_reference_ids"]),
                               "maximum_errors_of_uniquely_matched_primitives": errors,
                               "ambiguous_or_fragmented_pairs_not_forced": True},
        "checks": {"raw_modelspace_type_counts_equal": prediction["raw_modelspace"]["types"] == reference["raw_modelspace"]["types"],
                   "filtered_line_arc_type_counts_equal": ptypes == rtypes,
                   "both_endpoint_closed": prediction["connections"]["closed"] and reference["connections"]["closed"],
                   "registered_shape_within_0_1mm": bool(score.get("reference_within_0_1mm")),
                   "same_native_coordinate_frame_within_0_1mm": native_frame_01,
                   "all_native_primitive_parameters_match_after_registration": all_primitive_parameters,
                   "matched_primitive_adjacency_equal": adjacency_match,
                   "byte_identical_dxf": prediction_sha256 == reference_sha256},
        "limits": ["D4 axis search and translation use GT only in this independent report; a registered shape match does not prove the output coordinate frame matches GT.",
                   "Endpoint closure does not prove absence of self-intersections or correct design tangency.",
                   "Area is reported only for one closed endpoint cycle; it does not classify holes or prove a valid material region.",
                   "GT may contain assumed/simplified or closure layers; inspect comparison.json before treating it as a fully specified engineering target."]
    }


def _svg(comparison: dict, case_id: str) -> str:
    visual = comparison["visualization"]
    curves = [(side, row) for side in ("reference", "prediction") for row in visual[side]]
    if len(curves) > 1000:
        return ""
    points = [point for _, curve in curves for point in curve["points"]]
    if not points:
        return ""
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    scale = min(1320 / max(xmax - xmin, 1e-9), 740 / max(ymax - ymin, 1e-9))
    body = []
    for side, curve in curves:
        coordinates = " ".join(f"{45 + (x - xmin) * scale:.2f},{790 - (y - ymin) * scale:.2f}" for x, y in curve["points"])
        color = "#b33d62" if side == "reference" else "#087f78"
        dash = ' stroke-dasharray="8 5"' if curve["assumption_layer"] else ""
        body.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
    unit = "mm after D4/translation registration" if comparison["physical_units_available"] else "separate dimensionless display normalization"
    title = html.escape(case_id)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="850" viewBox="0 0 1400 850">'
            '<rect width="1400" height="850" fill="#fff"/>'
            f'<text x="40" y="35" font-family="sans-serif" font-size="20">{title} — oracle-mask DXF comparison</text>'
            f'<text x="40" y="65" font-family="sans-serif" font-size="14">Reference: magenta; prediction: teal; dashed: GT assumption layer. {html.escape(unit)}.</text>'
            + "".join(body) + '</svg>')


def evaluate_oracle_mask_run(run_dir: Path, *, dataset_root: Path = ROOT / "__dataset",
                             case_id: str | None = None) -> dict:
    """Write immutable-after-export evaluation files; never invoke generation."""
    run_dir = Path(run_dir).resolve(strict=True)
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    declared_id = manifest.get("case_id")
    if case_id is not None and declared_id != case_id:
        raise ValueError("Requested case_id does not match the frozen run manifest")
    case_id = declared_id
    if not isinstance(case_id, str) or not case_id or manifest.get("oracle_mask_conditioned") is not True:
        raise ValueError("A frozen case_id and oracle_mask_conditioned=true are required")
    evaluation = run_dir / "evaluation"
    evaluation.mkdir(exist_ok=True)
    prediction = run_dir / "after" / "drawing.dxf"
    prefix = {"schema_version": "oracle-mask-dxf-evaluation-v1", "case_id": case_id,
              "condition": "GT-derived_oracle_material_mask", "held_out": False,
              "independent_reference_accuracy_verified": False,
              "ground_truth_use": "oracle_mask_input_and_independent_post_export_scoring_only"}

    def unavailable(status: str, reason: str) -> dict:
        report = {**prefix, "status": status, "reason": reason, "reference_opened": False,
                  "reference_compared": False}
        _write(evaluation / "summary.json", report)
        return report

    if not prediction.is_file() or prediction.stat().st_size == 0:
        return unavailable("missing_prediction", "No finished after/drawing.dxf; GT was not opened")
    if not prediction.resolve().is_relative_to(run_dir):
        return unavailable("invalid_prediction_location", "Prediction path escapes the run directory")
    try:
        prediction_audit = audit_dxf(prediction)
    except Exception as error:
        return unavailable("unreadable_prediction", f"Local DXF readback failed: {type(error).__name__}")
    if prediction_audit["filtered_profile"]["count"] == 0 or prediction_audit["filtered_profile"]["issues"]:
        return unavailable("invalid_prediction", "The exported DXF has no valid, evaluable main profile")

    # Only this post-export branch accesses the GT inventory and DXF bytes.
    try:
        catalog = build_catalog(Path(dataset_root))
    except Exception as error:
        return unavailable("reference_inventory_error", f"Read-only GT inventory failed: {type(error).__name__}")
    case = next((row for row in catalog["cases"] if row["id"] == case_id), None)
    if case is None:
        return unavailable("unknown_case", "Case is absent from the read-only source inventory")
    try:
        reference, source = _reference_for_scoring(SimpleNamespace(dataset_root=Path(dataset_root)), case, evaluation)
    except Exception as error:
        return unavailable("reference_open_failed", f"GT could not be opened for independent scoring: {type(error).__name__}")
    if reference is None:
        return unavailable("missing_reference", "No paired GT DXF is available")
    frozen_hash = manifest.get("source_gt_sha256") or manifest.get("reference_sha256")
    if not isinstance(frozen_hash, str) or len(frozen_hash) != 64 or frozen_hash.lower() != source["reference_sha256"]:
        report = {**prefix, "status": "reference_hash_mismatch", "reference_opened": True,
                  "reference_compared": False,
                  "reason": "Scoring GT bytes differ from or are not identified by the frozen oracle-mask source hash",
                  "frozen_source_gt_sha256": frozen_hash, "scoring_reference_sha256": source["reference_sha256"]}
        _write(evaluation / "summary.json", report)
        return report
    try:
        comparison = compare_dxf_entities(prediction, reference)
    except Exception as error:
        report = {**prefix, "status": "comparison_failed", "reference_opened": True,
                  "reference_compared": False,
                  "reason": f"Independent DXF comparison failed: {type(error).__name__}"}
        _write(evaluation / "summary.json", report)
        return report
    oracle_mask = run_dir / "inputs" / "oracle-mask.png"
    oracle_mask_sha = _digest(oracle_mask) if oracle_mask.is_file() else None
    summary = _summarize(comparison, case_id=case_id,
                         prediction_sha256=prediction_audit["sha256"],
                         reference_sha256=source["reference_sha256"],
                         oracle_mask_sha256=oracle_mask_sha,
                         mask_reference_hash_verified=True)
    summary["reference_source"] = source["reference_source"]
    _write(evaluation / "comparison.json", comparison)
    svg = _svg(comparison, case_id)
    if svg:
        (evaluation / "overlay.svg").write_text(svg, encoding="utf-8")
        summary["overlay"] = "evaluation/overlay.svg"
    # Diagnose where reconstruction gained/lost accuracy without feeding that
    # information back to the predictor. All stage files already exist and
    # remain read-only; every comparison uses the same independent thresholds.
    stages = []
    for stage_id, relative in (
        ("initial_cad", "after/baseline-drawing.dxf"),
        ("source_topology", "after/source-topology-export/drawing.dxf"),
        ("parametric_candidate", "after/parametric-export/drawing.dxf"),
        ("published", "after/drawing.dxf"),
    ):
        artifact = run_dir / relative
        entry = {"stage": stage_id, "artifact": relative, "status": "not_available"}
        stages.append(entry)
        if not artifact.is_file():
            continue
        if not artifact.resolve().is_relative_to(run_dir):
            entry["status"] = "invalid_artifact_location"
            continue
        try:
            stage_audit = audit_dxf(artifact)
            if stage_audit["filtered_profile"]["issues"]:
                entry["status"] = "invalid_prediction"
                continue
            stage_comparison = comparison if stage_id == "published" else compare_dxf_entities(artifact, reference)
            stage_summary = summary if stage_id == "published" else _summarize(
                stage_comparison, case_id=case_id, prediction_sha256=stage_audit["sha256"],
                reference_sha256=source["reference_sha256"], oracle_mask_sha256=oracle_mask_sha,
                mask_reference_hash_verified=True)
            entry.update(status="compared", prediction_sha256=stage_audit["sha256"],
                         objects=stage_summary["objects"],
                         registered_shape=stage_summary["geometry"]["registered_shape_diagnostic"],
                         primitive_matching=stage_summary["primitive_matching"], checks=stage_summary["checks"])
            if stage_id != "published":
                stage_directory = evaluation / "stages"
                stage_directory.mkdir(exist_ok=True)
                _write(stage_directory / (stage_id + ".json"), stage_summary)
                stage_svg = _svg(stage_comparison, case_id + " / " + stage_id)
                if stage_svg:
                    (stage_directory / (stage_id + ".svg")).write_text(stage_svg, encoding="utf-8")
                    entry["overlay"] = "evaluation/stages/" + stage_id + ".svg"
            else:
                entry["overlay"] = summary.get("overlay")
        except Exception as error:
            entry.update(status="comparison_failed", failure_type=type(error).__name__)
    summary["stage_comparison"] = stages
    summary["stage_comparison_scope"] = "Independent post-export diagnostics only; no stage is selected or replaced using GT scores."
    _write(evaluation / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "__dataset")
    parser.add_argument("--case-id", type=str)
    args = parser.parse_args()
    report = evaluate_oracle_mask_run(args.run_dir, dataset_root=args.dataset, case_id=args.case_id)
    print(json.dumps({key: report.get(key) for key in ("case_id", "status", "checks", "overlay")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

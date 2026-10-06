"""Source annotation obligations and exact native-ARC export checks.

No reference geometry is consumed. Numerical satisfaction, source association,
and coverage are deliberately independent: an exact but wrongly associated R
is not a verified annotation, and a missed arrow is not an absent arrow.
"""
from __future__ import annotations

import math


def exact_radius_checks(entities, constraints, *, dxf_document=None):
    indexed = {entity.get("id"): entity for entity in entities}
    exported = list(dxf_document.modelspace()) if dxf_document is not None else None
    positions = {entity.get("id"): i for i, entity in enumerate(entities)}
    rows = []
    for constraint in constraints:
        if constraint.get("kind") != "radius":
            continue
        ids = constraint.get("entities") or []
        entity_id = ids[0] if len(ids) == 1 else None
        entity = indexed.get(entity_id, {})
        nominal = constraint.get("value")
        actual = entity.get("radius") if entity.get("type") == "ARC" else None
        numeric = (isinstance(nominal, (int, float)) and not isinstance(nominal, bool)
                   and math.isfinite(nominal) and nominal > 0
                   and isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual))
        passed = bool(numeric and actual == nominal)
        readback_radius = None
        if exported is not None:
            i = positions.get(entity_id)
            primitive = exported[i] if i is not None and i < len(exported) else None
            if primitive is not None and primitive.dxftype() == "ARC":
                readback_radius = float(primitive.dxf.radius)
            passed = bool(passed and len(exported) == len(entities) and readback_radius == nominal)
        rows.append({"record_id": constraint.get("record_id"), "entity_id": entity_id,
                     "nominal": nominal, "actual": actual, "dxf_radius": readback_radius,
                     "absolute_residual": abs(actual - nominal) if numeric else None,
                     "tolerance": 0.0, "enforcement": "exact", "passed": passed})
    return {"mode": "exact_native_arc_radius", "required_count": len(rows),
            "passed": all(row["passed"] for row in rows), "checks": rows,
            "dxf_readback_performed": exported is not None, "reference_geometry_used": False}


def annotation_radius_contract(bindings, solution):
    coverage = bindings.get("radius_binding_coverage")
    constraints = bindings.get("constraints", [])
    exact = exact_radius_checks(solution.get("entities", []), constraints)
    if coverage is None:
        # Older callers can still export a checked subset, but cannot acquire
        # full source-annotation coverage from the absence of a receipt.
        complete = False
        reasons = ["radius_source_coverage_not_recorded"]
    else:
        complete = coverage.get("all_radius_records_resolved") is True
        reasons = []
        if not coverage.get("all_confirmed_arrows_bound", False):
            reasons.append("confirmed_arrow_radius_unbound")
        if coverage.get("unresolved_radius_text_records"):
            reasons.append("unresolved_radius_text")
        if coverage.get("unknown_arrow_records"):
            reasons.append("radius_arrow_detection_unresolved")
        if coverage.get("ambiguous"):
            reasons.append("radius_target_ambiguous")
        if not complete and not reasons:
            reasons.append("radius_source_coverage_incomplete")
        for mapping in coverage.get("bound_mappings",[]):
            if not any(row["entity_id"]==mapping.get("entity_id") and
                       row["nominal"]==mapping.get("nominal") and row["passed"] for row in exact["checks"]):
                reasons.append("bound_radius_missing_exact_solver_result")
    if not exact["passed"]:
        reasons.append("annotated_radius_not_exact")
    if solution.get("accepted") is not True:
        reasons.append("numerical_solution_not_accepted")
    satisfied = bool(complete and exact["passed"] and solution.get("accepted") is True and not reasons)
    return {"schema_version": "annotation-radius-contract-v1", "satisfied": satisfied,
            "all_annotated_radii_verified": satisfied, "reasons": reasons,
            "coverage": coverage, "exact_radius_validation": exact,
            "missing_arrow_detection_is_exemption": False, "reference_verified": False,
            "scope": "Exact source-linked radii only; other dimensions and reference accuracy are separate checks."}

"""Coverage of sourced attributes and relationships, separate from reference accuracy.

A valid numerical subset (even all radii) is not a complete reconstruction.
Unobserved curved joins remain unresolved; this audit never invents tangency.
"""
from __future__ import annotations


def reconstruction_contract(entities, stage, validation):
    constraints = stage.get("constraints") or []
    counts = stage.get("binding_counts") or {}
    diagnostics = (stage.get("solver") or {}).get("diagnostics") or {}
    numerical = (stage.get("solver") or {}).get("validation") or {}
    numerical_checks = {row.get("id"): row for row in stage.get("solver_constraint_checks", [])}
    strict = validation.get("strict_relation_validation") or {}
    radii = validation.get("annotation_radius_contract") or {}
    recognized = counts.get("recognized_dimensions")
    bound = counts.get("bound_source_records")
    unbound = counts.get("unbound_dimensions")
    dimension_coverage = bool(isinstance(recognized, int) and recognized > 0 and
                              bound == recognized and unbound == 0)
    source_direction = set()
    joint_constraints = {}
    for row in constraints:
        ids = row.get("entities") or []
        nodes = row.get("nodes") or []
        if (row.get("kind") in {"horizontal", "vertical", "angle"} and len(ids) == 1 and
                numerical_checks.get(row.get("id"), {}).get("passed") is True):
            source_direction.add(ids[0])
        if row.get("kind") == "tangent" and len(ids) == 2 and len(nodes) == 1:
            joint_constraints[(frozenset(ids), nodes[0])] = row.get("id")
    checks = {row.get("constraint_id", row.get("id")): row for row in strict.get("checks", [])}
    joints = []
    for index, left in enumerate(entities):
        right = entities[(index + 1) % len(entities)]
        node = left.get("end_node")
        constraint_id = joint_constraints.get((frozenset((left.get("id"), right.get("id"))), node))
        explicit_tangent = constraint_id is not None
        axis_corner = (left.get("type") == right.get("type") == "LINE" and
                       left.get("id") in source_direction and right.get("id") in source_direction)
        connected = bool(node and node == right.get("start_node"))
        passed = bool(connected and (checks.get(constraint_id, {}).get("passed") is True
                                     if explicit_tangent else axis_corner))
        joints.append({"node_id": node, "entities": [left.get("id"), right.get("id")],
                       "types": [left.get("type"), right.get("type")], "constraint_id": constraint_id,
                       "relationship": "source_admitted_tangent" if explicit_tangent else
                                       "sourced_line_directions" if axis_corner else "unresolved",
                       "passed": passed, "tangency_assumed_from_fitted_shape": False})
    unknown = [row for row in joints if row["relationship"] == "unresolved"]
    shape_dof = diagnostics.get("remaining_shape_dof")
    fully_determined = isinstance(shape_dof, (int, float)) and shape_dof == 0 and stage.get("underconstrained") is False
    native = bool(strict.get("passed") is True and strict.get("dxf_readback_performed") is True)
    relation_coverage = bool(joints and all(row["passed"] for row in joints) and native)
    reasons = []
    if not dimension_coverage: reasons.append("recognized_annotation_attributes_unresolved")
    if not radii.get("satisfied"): reasons.append("annotated_radii_unresolved")
    if not relation_coverage: reasons.append("source_relationship_coverage_incomplete")
    if not native: reasons.append("strict_native_relation_certificate_missing_or_failed")
    if not fully_determined: reasons.append("shape_degrees_of_freedom_unresolved")
    if numerical.get("constraint_subset_satisfied") is not True: reasons.append("numerical_constraint_subset_not_certified")
    if (stage.get("solver") or {}).get("accepted") is not True: reasons.append("numerical_solver_not_accepted")
    if validation.get("passed") is not True: reasons.append("geometry_not_certified")
    return {"schema_version": "source-reconstruction-contract-v1", "satisfied": not reasons,
            "status": "source_obligations_satisfied" if not reasons else "unresolved_source_obligations",
            "reasons": reasons, "recognized_dimensions": recognized, "bound_source_records": bound,
            "unbound_dimensions": unbound, "all_recognized_attributes_covered": dimension_coverage,
            "all_join_relationships_certified": relation_coverage,
            "strict_native_relation_subset_satisfied": native, "joints": joints,
            "unresolved_joint_count": len(unknown), "remaining_shape_dof": shape_dof,
            "shape_fully_determined": fully_determined, "entity_count": len(entities),
            "entity_count_by_type": {kind: sum(e.get("type") == kind for e in entities) for kind in ("LINE", "ARC")},
            "source_obligation_inventory_consistent": dimension_coverage and relation_coverage,
            "complete_source_annotation_detection_verified": False,
            "reference_object_count_verified": False, "reference_accuracy_verified": False,
            "scope": "Recognized source obligations only. Unknown joints and unbound annotations remain in the denominator; no GT geometry is read."}

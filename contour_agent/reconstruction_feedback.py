"""Deterministic, public reconstruction diagnostics; never model reasoning or GT.

These diagnostics identify unresolved evidence. They cannot turn a plausible
annotation target into a verified dimensional constraint.
"""
from __future__ import annotations

import hashlib
import json
import math


def geometry_fingerprint(graph):
    """Ignore display IDs and cyclic start index when detecting repeated geometry."""
    rows = []
    for entity in graph.get("entities", []):
        row = {key: entity[key] for key in ("type", "start", "end", "center", "radius", "clockwise") if key in entity}
        rows.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    # Canonical rotation, without reversing the contour's winding.
    canonical = min((rows[i:] + rows[:i] for i in range(len(rows))), default=[])
    return hashlib.sha256("\n".join(canonical).encode()).hexdigest()


def reconstruction_feedback(graph, bindings=None, solution=None):
    bindings, solution = bindings or {}, solution or {}
    effective_entities=solution.get("entities",graph.get("entities",[])) if solution.get("accepted") else graph.get("entities",[])
    entities = {row["id"]: row for row in effective_entities}
    issues = []
    bound = {row.get("record_id") for row in bindings.get("constraints", []) if row.get("record_id")}
    for annotation in graph.get("annotation_support", []):
        record_id, entity_id = annotation.get("record_id"), annotation.get("candidate_entity_id")
        entity = entities.get(entity_id)
        if annotation.get("kind") != "radius" or not entity:
            continue
        solved_record=any(row.get("record_id")==record_id and row.get("kind")=="radius" and
                          entity_id in row.get("entities",[]) and row.get("passed") is True
                          for row in solution.get("constraints",[]))
        code = "radius_annotation_unbound" if record_id not in bound or not solved_record else None
        if entity.get("type") != "ARC":
            code = "radius_target_is_line"
        elif graph.get("units") == "mm":
            nominal, actual = annotation.get("nominal"), entity.get("radius")
            if isinstance(nominal, (float, int)) and isinstance(actual, (float, int)) and abs(actual-nominal) > .05:
                code = "radius_value_unresolved"
        if code:
            issues.append({"code": code, "record_id": record_id, "entity_id": entity_id,
                           "stable_id": entity.get("stable_id"),
                           "binding_verified": record_id in bound,"constraint_satisfied":solved_record,
                           "arrowhead_verified": annotation.get("arrowhead_verified") is True})
    for constraint in solution.get("constraints", []):
        if constraint.get("passed") is False:
            issues.append({"code": "constraint_residual_failed", "record_id": constraint.get("record_id"),
                           "entity_ids": constraint.get("entities", []), "kind": constraint.get("kind")})
    for row in bindings.get("bindings", []):
        if row.get("accepted") is False and row.get("reason") in {"conflicting_constraints", "conflicting_relations"}:
            issues.append({"code": row["reason"], "record_id": row.get("record_id"),
                           "relation_id": row.get("relation_id")})
    constraints = bindings.get("constraints", [])
    structural=[]
    def identity(entity_id):
        entity=entities.get(entity_id,{})
        return entity.get("stable_id") or f"{graph.get('candidate_id','graph')}:{entity_id}"
    for row in constraints:
        if row.get("kind") in {"horizontal","vertical","tangent"}:
            structural.append({"kind":row["kind"],"stable_ids":[identity(e) for e in row.get("entities",[])]})
    ancestry={identity(eid):list(set([identity(eid),*entity.get("parent_stable_ids",[]),*entity.get("ancestor_stable_ids",[])]))
              for eid,entity in entities.items()}
    # A measured turn can be a deliberate design corner. Keep observations out
    # of issues, ranking and acceptance; source-image review decides its meaning.
    review_items=[]
    if len(effective_entities)>1:
        for row in primitive_diagnostics(effective_entities)["primitives"]:
            angle=row.get("tangent_jump_deg")
            if angle is not None and math.isfinite(angle) and angle>5.:
                ids=[row["entity_id"],row["next_entity_id"]]
                review_items.append({"code":"tangent_jump_requires_source_review",
                                     "entity_ids":ids,"stable_ids":[identity(eid) for eid in ids],
                                     "tangent_jump_deg":round(angle,6),"tangency_required":False,
                                     "advisory_only":True})
    return {"schema_version": "reconstruction-feedback-v1", "ground_truth_used": False,
            "candidate_id": graph.get("candidate_id"), "geometry_sha256": geometry_fingerprint(graph),
            "issues": issues[:64], "issue_count": len(issues),
            "review_items":review_items[:12],
            "review_scope":"Advisory observations only; use the original drawing to distinguish intended corners from missing tangency. Not defects, constraints, scores or acceptance criteria.",
            "constraint_count": len(constraints), "bound_record_ids": sorted(bound),
            "structural_constraints":structural,"entity_ancestry":ancestry,
            "structural_constraint_count": sum(row.get("kind") in {"horizontal", "vertical", "tangent"} for row in constraints),
            "solver_status": solution.get("status", "not_run"), "solver_accepted": solution.get("accepted") is True,
            "remaining_shape_dof": solution.get("diagnostics", {}).get("remaining_shape_dof"),
            "unbound_dimensions": bindings.get("counts", {}).get("unbound_dimensions"),
            "dimensions_verified": False, "reference_verified": False}


def constraint_regression(before, after):
    """A new topology must not discard already independently bound records."""
    lost = sorted(set(before.get("bound_record_ids", []))-set(after.get("bound_record_ids", [])))
    reasons = []
    if lost:
        reasons.append("previously_bound_source_records_lost")
    lost_structural=[]
    ancestry=after.get("entity_ancestry",{})
    for previous in before.get("structural_constraints",[]):
        required=set(previous["stable_ids"])
        preserved=any(current["kind"]==previous["kind"] and required.issubset(
            {ancestor for eid in current["stable_ids"] for ancestor in ancestry.get(eid,[eid])})
            for current in after.get("structural_constraints",[]))
        # A merged native primitive has no internal tangent discontinuity.
        if previous["kind"]=="tangent" and any(required.issubset(parents) for parents in ancestry.values()):
            preserved=True
        if not preserved:lost_structural.append(previous)
    if lost_structural:reasons.append("previously_verified_structural_relations_lost")
    if before.get("constraint_count",0)>0 and after.get("constraint_count",0)==0:
        reasons.append("all_previous_constraints_lost")
    if before.get("solver_accepted") and after.get("constraint_count") and not after.get("solver_accepted"):
        reasons.append("previously_feasible_constraints_became_unsatisfied")
    if after.get("constraint_count") and after.get("solver_status") not in {"accepted", "not_run"}:
        reasons.append("candidate_constraints_not_jointly_satisfied")
    return {"passed": not reasons, "reasons": reasons, "lost_record_ids": lost,"lost_structural_constraints":lost_structural}


def primitive_diagnostics(entities):
    """Report measured orientation and tangent discontinuities, not desired truth."""
    result = []
    def tangent(entity, endpoint):
        if entity["type"] == "LINE":
            return [entity["end"][i]-entity["start"][i] for i in (0, 1)]
        radial = [entity[endpoint][i]-entity["center"][i] for i in (0, 1)]
        return [radial[1], -radial[0]] if entity["clockwise"] else [-radial[1], radial[0]]
    for index, entity in enumerate(entities):
        following = entities[(index+1) % len(entities)]
        u, v = tangent(entity, "end"), tangent(following, "start")
        denominator = math.hypot(*u)*math.hypot(*v)
        angle = math.degrees(math.acos(max(-1., min(1., sum(a*b for a, b in zip(u, v))/denominator)))) if denominator else None
        result.append({"entity_id": entity.get("id"), "stable_id": entity.get("stable_id"),
                       "type": entity["type"], "radius": entity.get("radius"),
                       "line_angle_deg": math.degrees(math.atan2(u[1], u[0])) if entity["type"] == "LINE" else None,
                       "next_entity_id": following.get("id"), "tangent_jump_deg": angle,
                       "tangency_required": False})
    return {"scope": "Measured geometry only; corners can be intentional and require source evidence.",
            "primitives": result, "reference_verified": False}

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
    # The initial candidate target is a hypothesis. Once the current binding
    # stage has independently selected an object, diagnose that actual object.
    radius_targets = {}
    for row in bindings.get("constraints", []):
        if row.get("kind") == "radius" and row.get("record_id"):
            targets = radius_targets.setdefault(row["record_id"], {})
            for entity_id in row.get("entities", []):
                if entity_id in entities:
                    targets[entity_id] = row
    seen_targets = set()
    for annotation in graph.get("annotation_support", []):
        if annotation.get("kind") != "radius":
            continue
        record_id = annotation.get("record_id")
        current_targets = radius_targets.get(record_id) or {annotation.get("candidate_entity_id"): {}}
        for entity_id, binding in current_targets.items():
            entity = entities.get(entity_id)
            if not entity or (record_id, entity_id) in seen_targets:
                continue
            seen_targets.add((record_id, entity_id))
            solved_record = any(row.get("record_id") == record_id and row.get("kind") == "radius" and
                                entity_id in row.get("entities", []) and row.get("passed") is True
                                for row in solution.get("constraints", []))
            code = "radius_annotation_unbound" if record_id not in bound or not solved_record else None
            if entity.get("type") != "ARC":
                code = "radius_target_is_line"
            elif graph.get("units") == "mm":
                nominal = binding.get("value", annotation.get("nominal"))
                actual = entity.get("radius")
                if isinstance(nominal, (float, int)) and isinstance(actual, (float, int)) and actual != nominal:
                    code = "radius_value_unresolved"
            if code:
                issues.append({"code": code, "record_id": record_id, "entity_id": entity_id,
                               "stable_id": entity.get("stable_id"),
                               "binding_verified": record_id in bound, "constraint_satisfied": solved_record,
                               "arrowhead_verified": binding.get("source_arrow_verified") is True or
                                                     annotation.get("arrowhead_verified") is True})
    for constraint in solution.get("constraints", []):
        if constraint.get("passed") is False:
            issues.append({"code": "constraint_residual_failed", "record_id": constraint.get("record_id"),
                           "entity_ids": constraint.get("entities", []), "kind": constraint.get("kind")})
    coverage=bindings.get("radius_binding_coverage") or {}
    seen_radius_records={row.get("record_id") for row in issues}
    for row in coverage.get("unresolved",[]):
        if row.get("reason")=="requires_topology_repartition":
            existing=next((issue for issue in issues if issue.get("record_id")==row.get("record_id")),None)
            detail={"code":"radius_requires_topology_repartition","record_id":row.get("record_id"),
                    "reason":row["reason"],"binding_verified":False,
                    "entity_ids":row.get("candidate_entity_ids",[]),
                    "arrowhead_verified":row.get("source_arrow_verified") is True}
            if existing is not None:existing.update(detail)
            else:issues.append(detail)
            continue
        if row.get("record_id") not in seen_radius_records:
            issues.append({"code":"radius_annotation_unresolved","record_id":row.get("record_id"),
                           "reason":row.get("reason"),"binding_verified":False,
                           "arrowhead_verified":row.get("record_id") in coverage.get("confirmed_arrow_records",[])})
    for row in bindings.get("bindings", []):
        if row.get("accepted") is False and row.get("reason") in {"conflicting_constraints", "conflicting_relations"}:
            issues.append({"code": row["reason"], "record_id": row.get("record_id"),
                           "relation_id": row.get("relation_id")})
    constraints = bindings.get("constraints", [])
    satisfied_records = {row.get("record_id") for row in solution.get("constraints", [])
                         if row.get("passed") is True and row.get("record_id") in bound}
    confirmed_arrows = set(coverage.get("confirmed_arrow_records") or [])
    verified_radii = {row.get("record_id") for row in solution.get("constraints", [])
                      if row.get("kind") == "radius" and row.get("passed") is True
                      and row.get("record_id") in satisfied_records and row.get("record_id") in confirmed_arrows}
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
            "satisfied_record_ids": sorted(satisfied_records),
            "verified_radius_record_ids": sorted(verified_radii),
            "bound_record_entities": {row["record_id"]: row.get("entities", []) for row in constraints if row.get("record_id")},
            "satisfied_structural_constraint_count": sum(row.get("passed") is True and
                row.get("kind") in {"horizontal", "vertical", "tangent"} for row in solution.get("constraints", [])),
            "structural_constraints":structural,"entity_ancestry":ancestry,
            "structural_constraint_count": sum(row.get("kind") in {"horizontal", "vertical", "tangent"} for row in constraints),
            "solver_status": solution.get("status", "not_run"), "solver_accepted": solution.get("accepted") is True,
            "remaining_shape_dof": solution.get("diagnostics", {}).get("remaining_shape_dof"),
            "unbound_dimensions": bindings.get("counts", {}).get("unbound_dimensions"),
            "radius_binding_coverage":coverage,
            "dimensions_verified": False, "reference_verified": False}


def source_failure_feedback(feedback, graph, validation, mask_diagnostics=None):
    """Expose failed *bound* intervals too; radius coverage alone is insufficient."""
    feedback["source_validation"] = validation
    feedback["post_solve_source_accepted"] = validation.get("passed") is True
    if isinstance(mask_diagnostics, dict):
        feedback["source_mask_diagnostics"] = mask_diagnostics
    if validation.get("passed") is not False:
        return feedback
    entities = {row["id"]: row for row in graph.get("entities", [])}
    mappings = feedback.get("bound_record_entities") or {}
    issues = feedback.setdefault("issues", [])
    for row in (mask_diagnostics or {}).get("entities", [])[:12]:
        if row.get("exceeds_original_budget") is not True or row.get("entity_id") not in entities:
            continue
        entity_id = row["entity_id"]
        records = sorted(record for record, ids in mappings.items() if entity_id in ids)
        radius_records = [record for record in records if record in (feedback.get("verified_radius_record_ids") or [])]
        issue = {"code": "bound_radius_source_interval_failed" if radius_records and entities[entity_id].get("type") == "ARC"
                 else "source_interval_failed_after_solve", "entity_id": entity_id,
                 "binding_verified": bool(records), "constraint_satisfied": bool(set(records) & set(feedback.get("satisfied_record_ids") or [])),
                 "source_max_deviation_px": row.get("conservative_max_deviation_px"),
                 "original_deviation_budget_px": mask_diagnostics.get("original_deviation_budget_px"),
                 "source_failed_quarters": row.get("failed_quarters") or [],
                 "required_action": "source_supported_interval_repartition_not_radius_relaxation"}
        if records:
            issue["record_id"] = (radius_records or records)[0]
            issue["bound_record_ids"] = records[:8]
        issues.insert(0, issue)
    feedback["issues"] = issues[:64]
    feedback["issue_count"] = len(issues)
    return feedback


def source_mask_interval_diagnostics(baseline, graph, entities):
    """Localize a failed immutable-mask gate; these samples do not approve edits."""
    import numpy as np
    from scipy.spatial import cKDTree
    from .topology_candidates import _affines
    from .vectorize import _closed_ring, _line_entities, _sample_entities
    raw = (baseline.get("extraction") or {}).get("raw_polyline_px")
    budget = (baseline.get("curve_fit") or {}).get("total_deviation_budget_px")
    if raw is None or type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        return {"status": "unavailable", "ground_truth_used": False}
    d2s, _, determinant = _affines(graph, baseline)
    scale = math.sqrt(abs(determinant))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid_source_mask_diagnostic_scale")
    source, source_spacing, _ = _sample_entities(_line_entities(_closed_ring(raw)), .5)
    samples, ids, row_slices = [], [], []
    fit_spacing, count = 0., 0
    for index, entity in enumerate(entities):
        points, spacing, _ = _sample_entities([entity], .5 / scale)
        points = d2s(points)
        if count + len(points) > 250000:
            raise ValueError("Source fidelity audit exceeds the 250000 sample limit")
        samples.append(points)
        ids.extend([index] * len(points))
        row_slices.append((count, count + len(points)))
        count += len(points)
        fit_spacing = max(fit_spacing, spacing * scale)
    fitted = np.vstack(samples)
    forward, nearest = cKDTree(fitted).query(source, workers=1)
    reverse = cKDTree(source).query(fitted, workers=1)[0]
    nearest_entities = np.asarray(ids, int)[nearest]
    uncertainty = max(source_spacing, fit_spacing) / 2.
    rows = []
    for index, (entity, (start, end)) in enumerate(zip(entities, row_slices)):
        forward_indices=np.flatnonzero(nearest_entities==index)
        assigned = forward[forward_indices]
        local_reverse = reverse[start:end]
        maximum = max(float(assigned.max()) if len(assigned) else 0., float(local_reverse.max())) + uncertainty
        quarters = [{"quarter": q + 1, "conservative_max_deviation_px": float(values.max()) + uncertainty}
                    for q, values in enumerate(np.array_split(local_reverse, 4)) if len(values)]
        failed_quarters={row["quarter"] for row in quarters if row["conservative_max_deviation_px"] > budget}
        if len(forward_indices):
            local_indices=nearest[forward_indices]-start
            labels=np.minimum(3,(local_indices*4//max(1,end-start)))+1
            failed_quarters.update(int(value) for value in labels[assigned+uncertainty>budget])
            worst_index=int(forward_indices[int(np.argmax(assigned))])
            worst_point={"worst_source_point_px":source[worst_index].tolist(),
                         "closest_candidate_point_px":fitted[nearest[worst_index]].tolist(),
                         "worst_source_interval_quarter":int(labels[int(np.argmax(assigned))])}
        else:worst_point={}
        rows.append({"entity_id": entity["id"], "type": entity["type"],
                     "conservative_max_deviation_px": maximum, "exceeds_original_budget": maximum > budget,
                     "source_to_fit_sample_max_px": float(assigned.max()) if len(assigned) else None,
                     "fit_to_source_sample_max_px": float(local_reverse.max()),
                     "failed_quarters": sorted(failed_quarters),
                     "quarters": quarters, **worst_point})
    rows.sort(key=lambda row: row["conservative_max_deviation_px"], reverse=True)
    return {"status": "measured", "ground_truth_used": False, "original_deviation_budget_px": float(budget),
            "entities": rows[:12], "geometry_stage": "solver_candidate", "acceptance_thresholds_changed": False,
            "scope": "Diagnostic nearest-primitive localization of the immutable input-mask error; never a new acceptance gate."}


def constraint_regression(before, after):
    """A new topology must not discard already independently bound records."""
    if after.get("solver_status") in {"source_validation_failed", "preflight_failed"}:
        # No binding/solve ran: absent receipts are not evidence that previously
        # verified records became geometrically infeasible.
        reasons = (after.get("source_validation") or {}).get("reasons") or [after["solver_status"]]
        return {"passed": False, "reasons": list(reasons),
                "comparison_status": "not_run", "lost_record_ids": [],
                "lost_structural_constraints": []}
    lost = sorted(set(before.get("bound_record_ids", []))-set(after.get("bound_record_ids", [])))
    reasons = []
    source_validation = after.get("source_validation") or {}
    if source_validation.get("passed") is False:
        reasons.extend(source_validation.get("reasons") or ["solved_source_validation_failed"])
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

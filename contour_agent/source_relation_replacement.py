"""Replace an inferred old contact by a freshly sourced G1 contact path."""
import math


def inferred_contact_path(parent, graph, row, mapping, current):
    """Explicit annotated relations are never replaced by this rule."""
    if (row.get("kind") != "tangent" or row.get("source") != "source_geometry" or
            row.get("record_id") is not None or row.get("required") is True or
            len(row.get("entities") or []) != 2 or len(row.get("nodes") or []) != 1):
        return None
    old = {e["id"]: e for e in parent["entities"]}
    new = {e["id"]: e for e in graph["entities"]}
    left, right = [old.get(e, {}) for e in row["entities"]]
    if left.get("start_node") == right.get("end_node"):
        left, right = right, left
    node = row["nodes"][0]
    if left.get("end_node") != node or right.get("start_node") != node:
        return None
    a, b = new.get(mapping.get(left.get("id")), {}), new.get(mapping.get(right.get("id")), {})
    if a.get("type") != "ARC" or b.get("type") != "ARC" or a.get("id") == b.get("id"):
        return None
    lines = [e for e in new.values() if e.get("type") == "LINE" and
             e.get("start_node") == a.get("end_node") and e.get("end_node") == b.get("start_node")]
    if len(lines) != 1:
        return None
    line = lines[0]; proof = line.get("source_contact_construction") or {}
    if (proof.get("method") != "source_nominal_exact_arc_line_arc_v1" or
            proof.get("ground_truth_used") is not False or
            any(proof.get(k) is not True for k in ("radius_exact", "angle_exact", "contacts_exact", "outer_endpoints_fixed"))):
        return None
    for oldpoint, point in ((left.get("start"), a.get("start")), (right.get("end"), b.get("end"))):
        if not oldpoint or not point or math.dist(oldpoint, point) > 1e-7:
            return None
    radii = proof.get("radius_record_ids") or []
    if len(radii) != 2 or len(set(radii)) != 2:
        return None
    for arc, rid in zip((a, b), radii):
        bindings = [c for c in current if c.get("kind") == "radius" and c.get("record_id") == rid and
                    c.get("entities") == [arc["id"]] and c.get("source_arrow_verified") is True and
                    c.get("required") is True and c.get("enforcement") == "exact" and
                    c.get("nominal_source") == "source_ocr" and c.get("value") == arc.get("radius")]
        if len(bindings) != 1:
            return None
    angles = [c for c in current if c.get("kind") == "angle" and c.get("entities") == [line["id"]] and
              c.get("record_id") == proof.get("angle_record_id") and c.get("source_arrow_verified") is True and
              c.get("required") is True and c.get("enforcement") == "exact" and c.get("nominal_source") == "source_ocr"]
    if len(angles) != 1:
        return None
    from .parametric_solver import native_source_nominal_angle_checks
    from .relation_contract import relation_checks
    if native_source_nominal_angle_checks(graph["entities"], angles).get("passed") is not True:
        return None
    selected = []
    for ids, node in (([a["id"], line["id"]], line["start_node"]),
                      ([line["id"], b["id"]], line["end_node"])):
        matches = [c for c in current if c.get("kind") == "tangent" and
                   c.get("entities") == ids and c.get("nodes") == [node]]
        if len(matches) != 1:
            return None
        evidence = matches[0].get("source_contact_path_evidence") or {}
        if (evidence.get("method") != "source_bound_exact_arc_line_arc_tangent_v1" or
                evidence.get("verified") is not True or evidence.get("original_source_image_rechecked") is not True or
                evidence.get("ground_truth_used") is not False or
                evidence.get("radius_record_ids") != radii or evidence.get("angle_record_id") != proof.get("angle_record_id") or
                evidence.get("shared_node") != node or evidence.get("independent_source_tangent_measurement") is not False):
            return None
        selected.append(matches[0])
    if relation_checks(graph["entities"], selected).get("passed") is not True:
        return None
    return {"old_constraint_id": row.get("id"), "new_constraint_ids": [c["id"] for c in selected],
            "path_entities": [a["id"], line["id"], b["id"]],
            "method": "fresh_source_design_path_replaces_inferred_adjacency_v1",
            "old_inferred_adjacency_replaced": True, "old_explicit_annotation_removed": False,
            "independent_source_tangent_measurement": False, "ground_truth_used": False}

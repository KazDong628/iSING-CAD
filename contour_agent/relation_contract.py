"""Recompute admitted tangency obligations on solved and native DXF geometry.

This is a certificate for the supplied constraint subset, not a discovery of
which joins ought to be tangent.  It consumes no reference geometry and never
accepts cached solver ``passed`` flags as geometric evidence.
"""
from __future__ import annotations

import math


STRICT_TANGENT_CERT_TOLERANCE_DEG = 1e-7
STRICT_RELATION_ENDPOINT_TOLERANCE = 1e-7


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("nonfinite_or_invalid_number")
    return float(value)


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("invalid_point")
    return (_number(value[0]), _number(value[1]))


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_or_invalid_id")
    return value


def _geometry(entity):
    if not isinstance(entity, dict):
        raise ValueError("invalid_entity")
    result = {"id": _identifier(entity.get("id")), "type": entity.get("type"),
              "start_node": _identifier(entity.get("start_node")),
              "end_node": _identifier(entity.get("end_node")),
              "start": _point(entity.get("start")), "end": _point(entity.get("end"))}
    chord = math.dist(result["start"], result["end"])
    if result["start_node"] == result["end_node"] or not math.isfinite(chord) or chord <= 1e-12:
        raise ValueError("degenerate_entity")
    if result["type"] == "ARC":
        result["center"] = _point(entity.get("center"))
        result["radius"] = _number(entity.get("radius"))
        result["clockwise"] = entity.get("clockwise")
        if result["radius"] <= 0 or not isinstance(result["clockwise"], bool):
            raise ValueError("invalid_arc")
        if any(abs(math.dist(result[key], result["center"]) - result["radius"]) >
               STRICT_RELATION_ENDPOINT_TOLERANCE for key in ("start", "end")):
            raise ValueError("arc_endpoint_incidence_failed")
    elif result["type"] != "LINE":
        raise ValueError("unsupported_entity_type")
    return result


def _native_point(value):
    point = tuple(float(v) for v in value)
    if len(point) not in (2, 3) or not all(math.isfinite(v) for v in point):
        raise ValueError("invalid_native_point")
    if len(point) == 3 and abs(point[2]) > STRICT_RELATION_ENDPOINT_TOLERANCE:
        raise ValueError("native_geometry_not_xy_planar")
    return point[:2]


def _native_geometry(primitive, source):
    if primitive.dxftype() != source["type"]:
        raise ValueError("native_entity_type_mismatch")
    result = dict(source)
    if source["type"] == "LINE":
        result.update(start=_native_point(primitive.dxf.start), end=_native_point(primitive.dxf.end))
    else:
        extrusion = tuple(float(v) for v in primitive.dxf.extrusion)
        if (len(extrusion) != 3 or not all(math.isfinite(v) for v in extrusion) or
                math.dist(extrusion, (0., 0., 1.)) > STRICT_RELATION_ENDPOINT_TOLERANCE):
            raise ValueError("unsupported_native_arc_plane")
        center = _native_point(primitive.dxf.center)
        radius = _number(float(primitive.dxf.radius))
        angles = [_number(float(primitive.dxf.start_angle)), _number(float(primitive.dxf.end_angle))]
        if radius <= 0:
            raise ValueError("invalid_native_radius")
        points = [tuple(center[i] + radius * trig(math.radians(angle))
                        for i, trig in enumerate((math.cos, math.sin))) for angle in angles]
        # DXF ARC is always CCW; export_parametric swaps endpoints for a CW
        # contour traversal.  Restore that traversal before checking its join.
        if source["clockwise"]:
            points.reverse()
        result.update(center=center, radius=radius, start=points[0], end=points[1])
    native = _geometry(result)
    for key in ("start", "end", "center"):
        if key in source and math.dist(native[key], source[key]) > STRICT_RELATION_ENDPOINT_TOLERANCE:
            raise ValueError("native_entity_mapping_mismatch")
    if source["type"] == "ARC" and abs(native["radius"] - source["radius"]) > STRICT_RELATION_ENDPOINT_TOLERANCE:
        raise ValueError("native_entity_mapping_mismatch")
    return native


def _tangent(entity, node):
    if entity["type"] == "LINE":
        vector = tuple(entity["end"][i] - entity["start"][i] for i in (0, 1))
    else:
        point = entity["end"] if entity["end_node"] == node else entity["start"]
        radial = tuple(point[i] - entity["center"][i] for i in (0, 1))
        sign = -1. if entity["clockwise"] else 1.
        vector = (-radial[1] * sign, radial[0] * sign)
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("degenerate_tangent")
    return tuple(value / norm for value in vector)


def _joint_check(first, second, node):
    a, b = _tangent(first, node), _tangent(second, node)
    dot = a[0] * b[0] + a[1] * b[1]
    cross = a[0] * b[1] - a[1] * b[0]
    angle = abs(math.degrees(math.atan2(cross, dot)))
    endpoint = lambda entity: entity["end"] if entity["end_node"] == node else entity["start"]
    gap = math.dist(endpoint(first), endpoint(second))
    return {"angle_residual_deg": angle, "endpoint_gap": gap, "forward_dot": dot,
            "passed": bool(dot > 0 and angle <= STRICT_TANGENT_CERT_TOLERANCE_DEG and
                           gap <= STRICT_RELATION_ENDPOINT_TOLERANCE)}


def relation_checks(entities, constraints, *, dxf_document=None):
    """Certify only supplied tangent constraints; zero required is vacuous.

    Native primitives must correspond one-to-one to the model list in the
    exact export order, including unchanged geometry within the fixed numeric
    tolerance.  Missing, malformed, duplicate or ambiguous input fails closed.
    The caller must separately establish annotation/relation coverage.
    """
    issues, rows, indexed, native_indexed = [], [], {}, {}
    native_requested = dxf_document is not None
    readback_performed = False
    if not isinstance(entities, list) or not isinstance(constraints, list):
        return {"schema_version": "strict-relation-contract-v1", "passed": False,
                "required_count": 0, "satisfied_count": 0, "checks": [],
                "issues": ["entities_and_constraints_must_be_lists"],
                "dxf_readback_performed": False, "reference_geometry_used": False,
                "complete_relation_coverage_verified": False}
    positions, incident = {}, {}
    for position, original in enumerate(entities):
        try:
            entity = _geometry(original)
            if entity["id"] in indexed:
                raise ValueError("duplicate_entity_id")
            indexed[entity["id"]] = entity
            positions[entity["id"]] = position
            for node in (entity["start_node"], entity["end_node"]):
                incident.setdefault(node, []).append(entity["id"])
        except (ValueError, TypeError, OverflowError) as error:
            issues.append(f"entity[{position}]:{error}")
    if native_requested:
        try:
            exported = list(dxf_document.modelspace())
            readback_performed = True
            if len(exported) != len(entities):
                raise ValueError("native_entity_count_mismatch")
            for entity_id, entity in indexed.items():
                native_indexed[entity_id] = _native_geometry(exported[positions[entity_id]], entity)
        except (AttributeError, ValueError, TypeError, IndexError, OverflowError) as error:
            issues.append(f"native_readback:{error}")
    seen_ids, seen_relations = set(), set()
    for original in constraints:
        if not isinstance(original, dict):
            issues.append("invalid_constraint")
            continue
        if original.get("kind") != "tangent":
            continue
        constraint_id = original.get("id")
        id_issue = None
        try:
            _identifier(constraint_id)
            if constraint_id in seen_ids:
                raise ValueError("duplicate_constraint_id")
            seen_ids.add(constraint_id)
        except (ValueError, TypeError) as error:
            id_issue = str(error)
            issues.append(f"constraint_id:{error}")
        row = {"constraint_id": constraint_id if isinstance(constraint_id, str) else None,
               "kind": "tangent", "passed": False, "model": None, "dxf": None,
               "angle_tolerance_deg": STRICT_TANGENT_CERT_TOLERANCE_DEG,
               "endpoint_tolerance": STRICT_RELATION_ENDPOINT_TOLERANCE}
        rows.append(row)
        try:
            if id_issue:
                raise ValueError(id_issue)
            ids, nodes = original.get("entities"), original.get("nodes", [])
            if (not isinstance(ids, list) or len(ids) != 2 or
                    any(not isinstance(eid, str) or eid not in indexed for eid in ids) or ids[0] == ids[1]):
                raise ValueError("tangent_requires_two_distinct_known_entities")
            first, second = (indexed[eid] for eid in ids)
            shared = {first["start_node"], first["end_node"]} & {second["start_node"], second["end_node"]}
            if not isinstance(nodes, list):
                raise ValueError("tangent_node_incidence_mismatch")
            if len(nodes) == 1 and isinstance(nodes[0], str) and nodes[0] in shared:
                node = nodes[0]
            elif nodes == [] and len(shared) == 1:
                node = next(iter(shared))
            elif nodes == []:
                raise ValueError("tangent_requires_unique_shared_node")
            else:
                raise ValueError("tangent_node_incidence_mismatch")
            if set(incident[node]) != set(ids) or len(incident[node]) != 2:
                raise ValueError("ambiguous_joint_incidence")
            if (first["end_node"] == node) == (second["end_node"] == node):
                raise ValueError("joint_does_not_follow_contour_traversal")
            nominal = original.get("value")
            if nominal is not None and _number(nominal) != 0.:
                raise ValueError("tangent_nominal_must_be_zero")
            signature = (tuple(sorted(ids)), node)
            if signature in seen_relations:
                raise ValueError("duplicate_tangent_relation")
            seen_relations.add(signature)
            row.update(entity_ids=ids[:], node_id=node, model=_joint_check(first, second, node))
            if native_requested:
                if any(eid not in native_indexed for eid in ids):
                    raise ValueError("native_relation_geometry_unavailable")
                row["dxf"] = _joint_check(*(native_indexed[eid] for eid in ids), node)
            row["passed"] = row["model"]["passed"] and (not native_requested or row["dxf"]["passed"])
            if not row["passed"]:
                row["reason"] = "strict_tangency_or_endpoint_check_failed"
        except (ValueError, TypeError, OverflowError) as error:
            row["reason"] = str(error)
    passed = not issues and all(row["passed"] for row in rows)
    return {"schema_version": "strict-relation-contract-v1", "mode": "recomputed_directed_tangency",
            "passed": passed, "required_count": len(rows),
            "satisfied_count": sum(row["passed"] for row in rows), "checks": rows,
            "issues": issues, "dxf_readback_performed": readback_performed,
            "native_mapping_verified": bool(native_requested and readback_performed and not issues),
            "angle_tolerance_deg": STRICT_TANGENT_CERT_TOLERANCE_DEG,
            "endpoint_tolerance": STRICT_RELATION_ENDPOINT_TOLERANCE,
            "reference_geometry_used": False, "complete_relation_coverage_verified": False,
            "scope": "Admitted tangent constraints only; absent obligations are not inferred or certified."}

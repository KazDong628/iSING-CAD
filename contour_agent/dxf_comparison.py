"""Independent DXF object/primitive audit; never imported by prediction code.

Entity IDs describe local file order only, never cross-file correspondence.
Unknown units remain unknown. Registration is the frozen independent scorer's
D4 + translation diagnostic, and never changes either source DXF.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

import ezdxf
import numpy as np
from scipy.spatial import cKDTree

from .autonomous_evaluation import evaluate_autonomous_artifact, _transform
from .evaluation import _curves, _distances, _samples


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _point(value):
    return [float(value[0]), float(value[1])]


def _raw_entity(entity, index):
    """Original object parameters, before filtering or unit conversion."""
    kind = entity.dxftype()
    row = {"id": f"raw{index:04d}", "type": kind, "layer": str(entity.dxf.layer),
           "handle": str(entity.dxf.handle), "coordinate_unit": "original_drawing_units"}
    if kind == "LINE":
        start, end = _point(entity.dxf.start), _point(entity.dxf.end)
        row.update(start=start, end=end, length=math.dist(start, end))
    elif kind in {"ARC", "CIRCLE"}:
        radius = float(entity.dxf.radius)
        start = float(entity.dxf.start_angle) if kind == "ARC" else 0.0
        sweep = (float(entity.dxf.end_angle) - start) % 360 if kind == "ARC" else 360.0
        row.update(center=_point(entity.dxf.center), radius=radius, start_angle_deg=start,
                   end_angle_deg=float(entity.dxf.end_angle) if kind == "ARC" else 360.0,
                   sweep_deg=sweep, length=radius * math.radians(sweep))
    elif kind == "LWPOLYLINE":
        row.update(closed=bool(entity.closed), vertices_xy_bulge=[list(map(float, point)) for point in entity.get_points("xyb")])
    elif kind == "POLYLINE":
        row.update(closed=bool(entity.is_closed), vertices_xy=[_point(vertex.dxf.location) for vertex in entity.vertices])
    return row


def _entity(curve, index, prefix, unit):
    row = {"id": f"{prefix}{index:04d}", "type": curve["kind"], "layer": curve["layer"],
           "assumption_layer": curve["assumed"], "coordinate_unit": unit,
           "start": _point(curve["start"]), "end": _point(curve["end"]), "length": float(curve["length"])}
    if curve["kind"] == "LINE":
        direction = curve["end"] - curve["start"]
        row["line_angle_deg"] = math.degrees(math.atan2(direction[1], direction[0])) % 360
    else:
        row.update(center=_point(curve["center"]), radius=float(curve["radius"]),
                   start_angle_deg=math.degrees(curve["angle"]) % 360,
                   end_angle_deg=math.degrees(curve["angle"] + curve["sweep"]) % 360,
                   sweep_deg=math.degrees(curve["sweep"]))
    return row


def _connections(curves, prefix, tolerance):
    if not curves:
        return {"tolerance": tolerance, "nodes": [], "edges": [], "components": [], "closed": False}
    points = np.array([point for curve in curves for point in (curve["start"], curve["end"])])
    parent = list(range(len(points)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    tree = cKDTree(points)
    for left, right in sorted(tree.query_pairs(tolerance)):
        parent[find(right)] = find(left)
    groups = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    indexed = {root: i for i, root in enumerate(sorted(groups))}
    nodes = []
    for root in sorted(groups):
        indices = groups[root]
        ids = [f"{prefix}{index // 2:04d}" for index in indices]
        incidents = []
        for endpoint in indices:
            curve = curves[endpoint // 2]; is_end = bool(endpoint % 2)
            if curve["kind"] == "LINE":
                tangent = curve["end"] - curve["start"]
            else:
                angle = curve["angle"] + (curve["sweep"] if is_end else 0)
                tangent = np.array([-math.sin(angle), math.cos(angle)])
            tangent = tangent / np.linalg.norm(tangent) * (-1 if is_end else 1)
            incidents.append({"entity_id": f"{prefix}{endpoint // 2:04d}", "type": curve["kind"],
                              "endpoint": "end" if is_end else "start", "interior_tangent": tangent.tolist()})
        angle = None
        if len(incidents) == 2:
            angle = math.degrees(math.acos(float(np.clip(np.dot(incidents[0]["interior_tangent"], incidents[1]["interior_tangent"]), -1, 1))))
        nodes.append({"id": f"n{indexed[root]:04d}", "point": _point(points[indices].mean(axis=0)),
                      "degree": len(indices), "entity_ids": sorted(set(ids)),
                      "maximum_snap_distance": max(float(np.linalg.norm(points[a] - points[b])) for a in indices for b in indices),
                      "incidents": incidents, "joint_type": "-".join(sorted(item["type"] for item in incidents)),
                      "interior_tangent_angle_deg": angle,
                      "deviation_from_tangent_continuity_deg": 180 - angle if angle is not None else None,
                      "intended_tangency_known": False,
                      "nearest_other_endpoint_distance": float(tree.query(points[indices[0]], k=2)[0][1]) if len(indices) == 1 else None})
    edges = []
    adjacency = {node["id"]: set() for node in nodes}
    for index in range(len(curves)):
        start, end = (f"n{indexed[find(2 * index + offset)]:04d}" for offset in (0, 1))
        edges.append({"entity_id": f"{prefix}{index:04d}", "start_node": start, "end_node": end})
        adjacency[start].add(end); adjacency[end].add(start)
    components = []; remaining = set(adjacency)
    while remaining:
        component = set(); pending = [min(remaining)]
        while pending:
            node = pending.pop()
            if node in component: continue
            component.add(node); pending.extend(adjacency[node] - component)
        remaining -= component
        components.append({"node_ids": sorted(component),
                           "entity_ids": [edge["entity_id"] for edge in edges if edge["start_node"] in component],
                           "closed_endpoint_degree": all(node["degree"] == 2 for node in nodes if node["id"] in component)})
    return {"tolerance": tolerance, "nodes": nodes, "edges": edges, "components": components,
            "closed": all(node["degree"] == 2 for node in nodes),
            "meaning": "Endpoint connectivity and incident tangent angles only. Tangent deviation=0 means smooth continuation; a nonzero angle is not an error unless design intent requires tangency. Crossing interiors and intended corners are not inferred."}


def _distribution(values):
    if not values: return {"count": 0, "minimum": None, "p05": None, "median": None, "p95": None, "maximum": None}
    return dict(zip(("count", "minimum", "p05", "median", "p95", "maximum"),
                    (len(values), float(min(values)), float(np.quantile(values,.05)), float(np.median(values)), float(np.quantile(values,.95)), float(max(values)))))


def audit_dxf(path, *, prefix="e", connection_tolerance=1e-5):
    """Return raw objects, filtered primitives and connectivity with explicit units."""
    document = ezdxf.readfile(path)
    raw = [_raw_entity(entity, index) for index, entity in enumerate(document.modelspace())]
    curves, info = _curves(Path(path))
    info = dict(info)
    unit = "mm" if info["source_units"] else "drawing_units_unspecified"
    # The older low-level parser's factor=1 is arithmetic, never a unit claim.
    info["unit_assumption"] = None
    info["coordinate_unit"] = unit
    entities = [_entity(curve, index, prefix, unit) for index, curve in enumerate(curves)]
    connections = _connections(curves, prefix, connection_tolerance)
    connections["coordinate_unit"] = unit
    for entity, edge in zip(entities, connections["edges"]):
        entity.update(start_node=edge["start_node"], end_node=edge["end_node"])
    return {"sha256": _sha256(path), "file_name": Path(path).name, "source_units": int(document.units),
            "raw_modelspace": {"count": len(raw), "types": dict(Counter(row["type"] for row in raw)),
                               "layers": dict(Counter(row["layer"] for row in raw)), "entities": raw},
            "filtered_profile": {**info, "count": len(curves), "entities": entities,
                                 "total_length": sum(curve["length"] for curve in curves),
                                 "length_distribution": _distribution([curve["length"] for curve in curves]),
                                 "arc_diagnostics": {"radius_distribution": _distribution([curve["radius"] for curve in curves if curve["kind"] == "ARC"]),
                                                     "sweep_degrees_distribution": _distribution([math.degrees(curve["sweep"]) for curve in curves if curve["kind"] == "ARC"]),
                                                     "sweep_below_1_degree": sum(curve["kind"] == "ARC" and math.degrees(curve["sweep"]) < 1 for curve in curves),
                                                     "sweep_below_5_degrees": sum(curve["kind"] == "ARC" and math.degrees(curve["sweep"]) < 5 for curve in curves),
                                                     "arc_length_below_0_1mm": sum(curve["kind"] == "ARC" and curve["length"] < .1 for curve in curves) if info["source_units"] else None,
                                                     "meaning": "Descriptive fragmentation indicators, not rejected primitives or proof that a corner should be tangent."}},
            "connections": connections}


def _curve_points(curve, maximum=128):
    count = 2 if curve["kind"] == "LINE" else min(maximum, max(8, math.ceil(curve["sweep"] / math.radians(2)) + 1))
    t = np.linspace(0, 1, count)
    if curve["kind"] == "LINE":
        return curve["start"] + t[:, None] * (curve["end"] - curve["start"])
    angle = curve["angle"] + curve["sweep"] * t
    return curve["center"] + curve["radius"] * np.column_stack([np.cos(angle), np.sin(angle)])


def _bbox(curves):
    points = []
    for curve in curves:
        points.extend([curve["start"], curve["end"]])
        if curve["kind"] != "LINE":
            for angle in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
                if (angle - curve["angle"]) % (2 * math.pi) <= curve["sweep"] + 1e-10:
                    points.append(curve["center"] + curve["radius"] * np.array([math.cos(angle), math.sin(angle)]))
    points = np.asarray(points)
    return points.min(axis=0), points.max(axis=0)


def _visual_curves(curves):
    return [{"type": curve["kind"], "layer": curve["layer"], "assumption_layer": curve["assumed"],
             "points": _curve_points(curve).tolist()} for curve in curves]


def _normalize_visual(curves):
    low, high = _bbox(curves)
    scale = float(max(high - low))
    if scale <= 0: raise ValueError("Degenerate bounding box")
    center = (low + high) / 2
    visual = _visual_curves(curves)
    for row in visual:
        row["points"] = ((np.array(row["points"]) - center) / scale).tolist()
    return visual, {"bbox_center": center.tolist(), "bbox_max_side": scale, "scale": 1 / scale}


def _parameter_errors(predicted, reference):
    result = {"length_error_mm": abs(predicted["length"] - reference["length"])}
    if predicted["kind"] != reference["kind"]: return result
    direct = max(np.linalg.norm(predicted["start"] - reference["start"]), np.linalg.norm(predicted["end"] - reference["end"]))
    reverse = max(np.linalg.norm(predicted["start"] - reference["end"]), np.linalg.norm(predicted["end"] - reference["start"]))
    # ARC traversal is CCW in the parser/aligned representation. Swapping arc
    # endpoints would incorrectly match complementary semicircles on one circle.
    result["endpoint_error_mm"] = float(min(direct, reverse) if predicted["kind"] == "LINE" else direct)
    if predicted["kind"] == "LINE":
        p = predicted["end"] - predicted["start"]; r = reference["end"] - reference["start"]
        cosine = np.clip(abs(np.dot(p, r)) / np.linalg.norm(p) / np.linalg.norm(r), 0, 1)
        result["undirected_angle_error_deg"] = math.degrees(math.acos(float(cosine)))
    else:
        result.update(radius_error_mm=abs(predicted["radius"] - reference["radius"]),
                      center_error_mm=float(np.linalg.norm(predicted["center"] - reference["center"])),
                      sweep_error_deg=math.degrees(abs(predicted["sweep"] - reference["sweep"])))
        if predicted["kind"] == "ARC":
            angular = lambda angle: abs((math.degrees(angle) + 180) % 360 - 180)
            result.update(start_angle_error_deg=angular(predicted["angle"] - reference["angle"]),
                          end_angle_error_deg=angular(predicted["angle"] + predicted["sweep"] - reference["angle"] - reference["sweep"]))
    return result


def _full_parameter_match(predicted, reference, tolerance):
    if predicted["kind"] != reference["kind"]: return False
    errors = _parameter_errors(predicted, reference)
    if errors["length_error_mm"] > 2 * tolerance: return False
    if predicted["kind"] == "LINE": return errors["endpoint_error_mm"] <= tolerance
    arc_displacement = max(predicted["radius"], reference["radius"]) * abs(predicted["sweep"] - reference["sweep"])
    return (errors["center_error_mm"] + errors["radius_error_mm"] + arc_displacement <= tolerance
            and (predicted["kind"] == "CIRCLE" or errors["endpoint_error_mm"] <= tolerance))


def _correspondence(predicted, reference, tolerance=.1):
    """No Hungarian/index pairing: only uniquely supported full primitives match."""
    if len(predicted) * len(reference) > 200_000:
        return {"status": "bounded_pair_limit", "rows": [], "counts": {}, "one_to_many_fragment_candidates": {},
                "unmatched_reference_ids": [f"r{i:04d}" for i in range(len(reference))],
                "reason": "More than 200000 primitive pairs; full-file score and inventories remain available, correspondence is uncomputed."}
    rows = []; gates = {}; distances = {}
    def samples(curve):
        if curve["kind"] == "LINE":
            return curve["start"] + np.linspace(0, 1, 64)[:, None] * (curve["end"] - curve["start"])
        return _curve_points(curve, maximum=256)
    for pi, pc in enumerate(predicted):
        ppoints = samples(pc)
        candidates = []
        for ri, rc in enumerate(reference):
            pd = _distances(ppoints, [rc]); rd = _distances(samples(rc), [pc])
            candidates.append({"reference_id": f"r{ri:04d}", "reference_type": rc["kind"],
                               "sampled_prediction_to_reference_max_mm": float(pd.max()),
                               "sampled_reference_to_prediction_max_mm": float(rd.max()),
                               "sampled_symmetric_max_mm": float(max(pd.max(), rd.max())),
                               "same_type": pc["kind"] == rc["kind"]})
            if _full_parameter_match(pc, rc, tolerance): gates.setdefault(pi, []).append(ri)
        distances[pi] = candidates
    reverse_gates = {ri: [pi for pi, indices in gates.items() if ri in indices] for ri in range(len(reference))}
    matched_reference = set(); fragment_groups = {}
    for pi, pc in enumerate(predicted):
        full = gates.get(pi, [])
        ranked = sorted(distances[pi], key=lambda row: (row["sampled_symmetric_max_mm"], row["reference_id"]))
        row = {"prediction_id": f"p{pi:04d}", "prediction_type": pc["kind"], "status": "unmatched_geometry",
               "reference_id": None, "parameter_errors": None, "nearest_candidates": ranked[:3],
               "full_parameter_candidates": [f"r{ri:04d}" for ri in full]}
        if len(full) == 1 and len(reverse_gates[full[0]]) == 1:
            ri = full[0]; matched_reference.add(ri)
            row.update(status="unique_full_parameter_agreement", reference_id=f"r{ri:04d}",
                       parameter_errors=_parameter_errors(pc, reference[ri]))
        elif full:
            row["status"] = "ambiguous_multiple_full_candidates"
        else:
            # One-way sampled coverage can flag fragmentation, never verify it.
            fragments = [candidate for candidate in distances[pi]
                         if candidate["sampled_prediction_to_reference_max_mm"] <= tolerance
                         and candidate["sampled_reference_to_prediction_max_mm"] > tolerance]
            if fragments:
                row["status"] = "possible_fragment_coverage"
                row["fragment_candidate_ids"] = [candidate["reference_id"] for candidate in fragments]
                for candidate in fragments: fragment_groups.setdefault(candidate["reference_id"], []).append(row["prediction_id"])
        rows.append(row)
    return {"status": "physical_diagnostic", "parameter_tolerance_mm": tolerance,
            "policy": "Never pair by entity index, force equal counts, or merge fragments into true design primitives. Parameter errors exist only for mutually unique full-primitive agreements. Nearest/fragment candidates are sampled geometric diagnostics, not verified correspondences.",
            "counts": dict(Counter(row["status"] for row in rows)), "rows": rows,
            "one_to_many_fragment_candidates": {key: value for key, value in fragment_groups.items() if len(value) > 1},
            "unmatched_reference_ids": [f"r{ri:04d}" for ri in range(len(reference)) if ri not in matched_reference],
            "matching_is_engineering_verification": False}


def _error_localization(predicted, reference, step):
    def stats(values):
        return {"max_error_mm": float(values.max()), "p95_error_mm": float(np.quantile(values, .95)),
                "rms_error_mm": float(np.sqrt(np.mean(values ** 2))), "sample_count": len(values)} if len(values) else None
    output = {"coordinate_unit": "mm", "sample_step_mm": step,
              "meaning": "Directed sampled-to-exact-curve errors after frozen registration. Reference core/closure coverage excludes extra predicted geometry and must not replace the complete symmetric score."}
    groups = {"reference_core": [], "reference_assumptions": []}
    layers = {}
    for side, source, target, prefix, target_prefix in (("reference", reference, predicted, "r", "p"),
                                                       ("prediction", predicted, reference, "p", "r")):
        rows = []
        for index, curve in enumerate(source):
            points = _samples([curve], step)
            distances = _distances(points, target)
            worst = int(np.argmax(distances)); point = points[worst]
            nearest = [float(_distances(point[None, :], [candidate])[0]) for candidate in target]
            minimum = min(nearest)
            rows.append({"entity_id": f"{prefix}{index:04d}", "type": curve["kind"], "layer": curve["layer"],
                         "assumption_layer": curve["assumed"], **stats(distances), "worst_point_mm": _point(point),
                         "nearest_target_entity_ids_at_worst_point": [f"{target_prefix}{i:04d}" for i, value in enumerate(nearest) if abs(value - minimum) <= 1e-8],
                         "parameters": _entity(curve, index, prefix, "mm")})
            if side == "reference":
                groups["reference_assumptions" if curve["assumed"] else "reference_core"].append(distances)
                layers.setdefault(curve["layer"], []).append(distances)
        output[f"per_{side}_curve"] = rows
        output[f"worst_{side}_curves"] = sorted(rows, key=lambda row: (-row["max_error_mm"], row["entity_id"]))[:5]
    output["reference_directed_scope"] = {name: stats(np.concatenate(chunks)) if chunks else None for name, chunks in groups.items()}
    output["reference_directed_layers"] = {name: stats(np.concatenate(chunks)) for name, chunks in layers.items()}
    return output


def compare_dxf_entities(prediction, reference):
    """Audit two finished files and retain the frozen 0.1 mm physical scorer."""
    pa = audit_dxf(prediction, prefix="p"); ra = audit_dxf(reference, prefix="r")
    pc, _ = _curves(Path(prediction)); rc, _ = _curves(Path(reference))
    physical = evaluate_autonomous_artifact(Path(prediction), Path(reference))
    for key in ("prediction_info", "reference_info"):
        if key in physical:
            information = physical[key]
            information["unit_assumption"] = None
            information["coordinate_unit"] = "mm" if information["source_units"] else "drawing_units_unspecified"
            if not information["source_units"]:
                information["native_coordinate_multiplier"] = information.pop("millimetre_conversion_factor", 1.0)
                information["millimetre_conversion_factor"] = None
    validation = physical.get("candidate_validation")
    if pa["source_units"] == 0 and validation and "endpoint_tolerance_mm" in validation:
        validation["endpoint_tolerance_drawing_units"] = validation.pop("endpoint_tolerance_mm")
        validation["coordinate_unit"] = "drawing_units_unspecified"
    available = pa["source_units"] != 0 and ra["source_units"] != 0
    transform = physical.get("transform")
    if available and transform:
        aligned = _transform(pc, np.array(transform["matrix"]), transform["translation_mm"])
        alignment = {"kind": "D4_translation_shape_diagnostic", "coordinate_unit": "mm",
                     "transform": transform, "scale_fitted": False, "engineering_verified": False}
        correspondence = _correspondence(aligned, rc)
        localization = _error_localization(aligned, rc, physical["sample_step_mm"])
        visual = {"prediction": _visual_curves(aligned), "reference": _visual_curves(rc)}
        aligned_entities = [_entity(curve, index, "p", "mm") for index, curve in enumerate(aligned)]
    else:
        alignment = {"kind": "independent_bbox_normalization_visualization_only", "coordinate_unit": "dimensionless",
                     "scale_fitted": False, "separate_display_scales": True, "engineering_verified": False,
                     "reason": "Physical units are missing or physical registration failed. Display normalization is not a scale estimate or parameter comparison."}
        if pc and rc:
            pv, pnorm = _normalize_visual(pc); rv, rnorm = _normalize_visual(rc)
            alignment.update(prediction_display_transform=pnorm, reference_display_transform=rnorm)
            visual = {"prediction": pv, "reference": rv}
        else: visual = {"prediction": [], "reference": []}
        correspondence = {"status": "not_computed", "reason": alignment["reason"], "rows": [], "counts": {},
                          "one_to_many_fragment_candidates": {}, "unmatched_reference_ids": [f"r{i:04d}" for i in range(len(rc))]}
        aligned_entities = []
        localization = {"status": "not_computed", "reason": "Physical units or registration are unavailable; no mm localization is manufactured."}
    return {"schema_version": "dxf-entity-comparison-v1", "prediction": pa, "reference": ra,
            "physical_units_available": available, "physical_score": physical, "alignment": alignment,
            "entity_correspondence": correspondence, "aligned_prediction_entities": aligned_entities,
            "error_localization": localization,
            "visualization": visual, "engineering_verified": False,
            "limitations": ["Entity count differences can indicate fragmentation, omitted design primitives or extra geometry; counts alone cannot identify which.",
                            "Reference closure/simplification layers remain in the frozen score and are marked on every primitive.",
                            "Raw object parameters are original drawing units; filtered known-unit primitives are converted to mm. Unspecified units are never assumed mm."]}

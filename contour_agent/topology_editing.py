"""Bounded local LINE/ARC topology edits proposed by a multimodal agent.

The online model may name an ordered chain and an edit intent.  It never emits
coordinates or dimensions.  This module refits the named chain against the
source segmentation boundary, rebuilds the complete graph, and rejects edits
that lose source support or invalidate the closed contour.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .ocr import canonical_records
from .topology import _StrokeEvidence, _primitive_distance
from .topology_candidates import (
    _affines, _annotation_inventory, _base_source_entities, _base_source_ring,
    _entity_annotation_support, _planner_fields, _read_image, _render_overlay,
    _to_graph,
)
from .vectorize import _arc, _closed_ring, _line, _sample_entities, assess_fit_quality


EDIT_ACTIONS = {"merge_chain_as_line", "merge_chain_as_arc", "merge_chain_best_fit",
                "refit_chain_as_annotated_arc", "refit_entity_as_line",
                "split_chain_at_source_features", "insert_annotated_fillet"}


def _edit_minimum(action):
    return 2 if action.startswith("merge_chain_") else 1


def propose_annotation_arc_edits(graph, inventory, *, limit=4):
    """Create bounded source hypotheses; the executor/evaluator still validate.

    Reserve room for a directed small fillet and a shallow unlabelled ARC->LINE
    repair, so repeated numeric refits cannot starve structural corrections.
    """
    if not isinstance(graph, dict) or not isinstance(inventory, list) or limit <= 0:
        return []
    entities = {row.get("id"): row for row in graph.get("entities", []) if isinstance(row, dict)}
    records = {row.get("record_id"): row for row in inventory if isinstance(row, dict)
               and row.get("record_id") and row.get("kind") == "radius"}
    supports = [row for row in graph.get("annotation_support", []) if isinstance(row, dict)
                and row.get("kind") == "radius" and row.get("status") == "candidate_supported"
                and row.get("candidate_entity_id") in entities and row.get("record_id") in records]
    by_entity = {}
    for row in supports:
        by_entity.setdefault(row["candidate_entity_id"], []).append(row)
    ranked, fillets = [], []
    ordered_entities = list(entities.values())
    nodes = {row.get("id"): row for row in graph.get("nodes", []) if isinstance(row, dict)}

    def source_endpoints(entity):
        a = nodes.get(entity.get("start_node"), {}).get("source_px")
        b = nodes.get(entity.get("end_node"), {}).get("source_px")
        if a is None or b is None:
            if graph.get("units") != "pixel":
                return None
            a, b = entity.get("start"), entity.get("end")
        pair = np.asarray([a, b], float)
        return pair if pair.shape == (2, 2) and np.isfinite(pair).all() else None

    def propose_endpoint_fillet(entity_id, support, record, mismatch, primitive_conflict):
        # Two labels may name different small features at opposite ends of a
        # long initial primitive. Do not refit that whole primitive to either
        # radius; propose only an endpoint-local exact fillet hypothesis.
        target = (support.get("source_evidence") or {}).get("target_source_px")
        if target is None:
            target = (record.get("leader") or {}).get("target_source_px")
        endpoints = source_endpoints(entities[entity_id])
        if not (support.get("arrowhead_verified") and target is not None and endpoints is not None
                and len(ordered_entities) >= 4 and (primitive_conflict or mismatch > math.log(2.))):
            return
        target = np.asarray(target, float)
        if target.shape != (2,) or not np.isfinite(target).all():
            return
        gaps = np.linalg.norm(endpoints - target, axis=1)
        near = int(np.argmin(gaps))
        chord = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        if float(gaps[near]) > max(8 * float(graph.get("source_grid_pitch_px") or 1.), .30 * chord):
            return
        position = next(i for i, row in enumerate(ordered_entities) if row["id"] == entity_id)
        pair = ([ordered_entities[(position - 1) % len(ordered_entities)]["id"], entity_id]
                if near == 0 else [entity_id, ordered_entities[(position + 1) % len(ordered_entities)]["id"]])
        fillets.append((mismatch, {"action": "insert_annotated_fillet", "entity_ids": pair,
            "record_id": record["record_id"], "evidence_tags": ["annotation_target", "source_boundary"]}))

    for entity_id, rows in by_entity.items():
        entity = entities[entity_id]
        primitive_conflict = entity.get("type") == "LINE"
        fitted = entity.get("radius")
        for support in rows:
            record = records[support["record_id"]]
            nominal = record.get("nominal")
            if (isinstance(nominal, bool) or not isinstance(nominal, (int, float)) or
                    not math.isfinite(nominal) or nominal <= 0):
                continue
            mismatch = 0.0
            if (graph.get("units") == "mm" and isinstance(fitted, (int, float)) and fitted > 0):
                mismatch = abs(math.log(float(fitted) / float(nominal)))
            if not primitive_conflict and mismatch < .20:
                continue
            if not (support.get("arrowhead_verified") or
                    float(support.get("target_gap_px", math.inf)) <=
                    2.25 * float(graph.get("source_grid_pitch_px") or 1.)):
                continue
            if len(rows) == 1:
                operation = {
                    "action": "refit_chain_as_annotated_arc", "entity_ids": [entity_id],
                    "record_id": support["record_id"],
                    "evidence_tags": ["annotation_target", "source_boundary"],
                }
                # A type conflict is strongest. For already-ARC entities,
                # larger radii usually span more pixels and condition better.
                key = (primitive_conflict, float(nominal), mismatch,
                       bool(support.get("arrowhead_verified")),
                       -float(support.get("target_gap_px", math.inf)))
                ranked.append((key, operation))
            propose_endpoint_fillet(entity_id, support, record, mismatch, primitive_conflict)
    ranked.sort(key=lambda row: row[0], reverse=True)
    protected = {row.get("candidate_entity_id") for row in graph.get("annotation_support", [])
                 if isinstance(row, dict) and row.get("kind") == "radius" and row.get("status") == "candidate_supported"}
    protected.update(hypothesis.get("entity_id")
                     for row in graph.get("annotation_support", []) if isinstance(row, dict)
                     and row.get("kind") == "radius" and row.get("status") == "candidate_supported"
                     and row.get("arrowhead_verified")
                     for hypothesis in row.get("adjacent_target_hypotheses", [])
                     if isinstance(hypothesis, dict) and hypothesis.get("entity_type") == "ARC")
    straight = []
    for entity in ordered_entities:
        if entity.get("type") != "ARC" or entity.get("id") in protected or entity.get("radius_binding"):
            continue
        endpoints = source_endpoints(entity)
        if endpoints is None:
            continue
        try:
            a, b, center = (np.asarray(entity[key], float) for key in ("start", "end", "center"))
            radius = float(entity["radius"])
            chord = float(np.linalg.norm(b - a))
            source_chord = float(np.linalg.norm(endpoints[1] - endpoints[0]))
            sign = -1 if entity.get("clockwise") else 1
            sweep = ((math.atan2(b[1] - center[1], b[0] - center[0]) -
                      math.atan2(a[1] - center[1], a[0] - center[0])) * sign) % (2 * math.pi)
            sagitta_px = radius * (1 - math.cos(sweep / 2)) * source_chord / chord
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        grid = float(graph.get("source_grid_pitch_px") or 1.)
        if sweep >= math.pi or source_chord < 8 * grid or sagitta_px > max(2 * grid, float(graph.get("proposal_tolerance_px") or grid) * 1.5):
            continue
        straight.append((source_chord / max(sagitta_px, .1), {
            "action": "refit_entity_as_line", "entity_ids": [entity["id"]], "record_id": None,
            "evidence_tags": ["source_boundary", "collinear_support"]}))
    reserved = [row[1] for row in sorted(fillets, key=lambda row: row[0], reverse=True)[:1]]
    reserved.extend(row[1] for row in sorted(straight, key=lambda row: row[0], reverse=True)[:1])
    return [*reserved[:limit], *[operation for _, operation in ranked[:max(0, limit - len(reserved))]]]


def _source_ring(baseline):
    extraction = baseline.get("extraction") if isinstance(baseline, dict) else None
    values = (
        extraction.get("raw_polyline_px") if isinstance(extraction, dict) else None,
        baseline.get("raw_polyline_px") if isinstance(baseline, dict) else None,
        baseline.get("polyline_px") if isinstance(baseline, dict) else None,
    )
    for value in values:
        if value is not None:
            return value
    return None


def _ordered_indices(graph, entity_ids, *, minimum=2):
    entities = graph.get("entities", [])
    if not isinstance(entity_ids, list) or not minimum <= len(entity_ids) <= 8:
        raise ValueError("edit_entity_count_outside_bounds")
    if len(set(entity_ids)) != len(entity_ids) or not all(isinstance(value, str) for value in entity_ids):
        raise ValueError("invalid_edit_entity_ids")
    by_id = {row.get("id"): index for index, row in enumerate(entities) if isinstance(row, dict)}
    if len(by_id) != len(entities) or any(value not in by_id for value in entity_ids):
        raise ValueError("unknown_edit_entity_id")
    start = by_id[entity_ids[0]]
    expected = [entities[(start + offset) % len(entities)].get("id") for offset in range(len(entity_ids))]
    if expected != entity_ids:
        raise ValueError("edit_entities_not_ordered_consecutive_chain")
    return [(start + offset) % len(entities) for offset in range(len(entity_ids))]


def _ring_path(ring, start, end, chain_samples):
    """Return the raw-boundary path that best overlaps the named fitted chain."""
    points = _closed_ring(ring)[:-1]
    first = int(np.argmin(np.linalg.norm(points - start, axis=1)))
    last = int(np.argmin(np.linalg.norm(points - end, axis=1)))

    def path(step):
        result = [points[first]]
        index = first
        for _ in range(len(points)):
            if index == last:
                break
            index = (index + step) % len(points)
            result.append(points[index])
        return np.asarray(result, float)

    options = [path(1), path(-1)]

    def mismatch(candidate):
        if len(candidate) < 2:
            return math.inf
        first_distance = cKDTree(chain_samples).query(candidate, workers=1)[0]
        second_distance = cKDTree(candidate).query(chain_samples, workers=1)[0]
        # P95 is less sensitive to the exact endpoint rasterization; the small
        # length term prevents selecting the rest of the closed profile.
        return float(np.percentile(first_distance, 95) + np.percentile(second_distance, 95)
                     + .001 * np.linalg.norm(np.diff(candidate, axis=0), axis=1).sum())

    selected = min(options, key=mismatch).copy()
    selected[0], selected[-1] = start, end
    return selected, mismatch(selected)


def _fixed_radius_arc(points, radius, tolerance):
    """Fit the locally observed path to a declared radius without moving endpoints."""
    if not math.isfinite(radius) or radius <= 0 or len(points) < 3:
        return None
    a, b = np.asarray(points[0], float), np.asarray(points[-1], float)
    chord = b - a
    length = float(np.linalg.norm(chord))
    if length < 1e-8 or length >= 2 * radius:
        return None
    middle = (a + b) / 2
    normal = np.array([-chord[1], chord[0]]) / length
    offset = math.sqrt(max(0., radius * radius - (length / 2) ** 2))
    options = []
    for center in (middle + offset * normal, middle - offset * normal):
        errors = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        theta = np.unwrap(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
        sweep = float(theta[-1] - theta[0])
        reversals = np.diff(theta) * (1 if sweep >= 0 else -1)
        if not .02 < abs(sweep) < math.pi * 1.8 or np.any(reversals < -.025):
            continue
        p95 = float(np.percentile(errors, 95))
        maximum = float(errors.max())
        if p95 <= max(tolerance * 1.5, 1.) and maximum <= max(tolerance * 3., 2.):
            options.append((p95, maximum, {"type": "ARC", "start": a.tolist(), "end": b.tolist(),
                                           "center": center.tolist(), "radius": float(radius),
                                           "clockwise": sweep < 0, "fit_error_px": maximum}))
    return min(options, key=lambda row: (row[0], row[1]))[2] if options else None


def _fit_replacement(points, action, tolerance, *, force_arc=False, annotated_radius_px=None,
                     radius_binding=None):
    line = _line(points, tolerance)
    arc = _arc(points, tolerance)
    applied_radius_binding = None
    if annotated_radius_px is not None:
        selected = _fixed_radius_arc(points, annotated_radius_px, tolerance)
        if selected is not None:
            applied_radius_binding = radius_binding
        else:
            # The label still establishes ARC semantics when noisy segmentation
            # or misplaced split points prevent an exact fixed-radius fit. Keep
            # the numeric value unbound and let the downstream solver report it
            # as unresolved instead of flattening the region into a LINE.
            selected = arc or _arc(points, max(tolerance * 2., tolerance + 2.))
    elif action in {"merge_chain_as_line", "refit_entity_as_line"}:
        selected = line
    elif action in {"merge_chain_as_arc", "refit_chain_as_annotated_arc"} or force_arc:
        selected = arc
    elif action == "merge_chain_best_fit":
        options = [row for row in (line, arc) if row is not None]
        selected = min(options, key=lambda row: (float(row.get("fit_error_px", math.inf)),
                                                 0 if row["type"] == "LINE" else 1)) if options else None
    else:
        raise ValueError("unsupported_topology_edit_action")
    if selected is None:
        raise ValueError("source_boundary_does_not_support_requested_primitive")
    result = copy.deepcopy(selected)
    result.update(parameter_source="multimodal_edit_local_refit",
                  source_support_vertex_count=int(len(points)))
    if radius_binding:
        result.update(parameter_source="multimodal_annotation_guided_arc_refit",
                      radius_annotation_evidence=copy.deepcopy(radius_binding),
                      radius_binding_status="applied" if applied_radius_binding else "unresolved_fixed_radius_fit_failed")
    if applied_radius_binding:
        result["radius_binding"] = copy.deepcopy(applied_radius_binding)
    return result


def _radius_edit_evidence(graph, operation, inventory, entity_ids, design_to_source):
    """Resolve radius semantics and protect labelled arcs from line simplification."""
    grid = float(graph.get("source_grid_pitch_px") or 1.)
    support = [row for row in graph.get("annotation_support", []) if isinstance(row, dict)
               and row.get("kind") == "radius" and row.get("status") == "candidate_supported"
               and row.get("candidate_entity_id") in entity_ids
               and (row.get("arrowhead_verified") or float(row.get("target_gap_px", math.inf)) <= 2.25 * grid)]
    # The initial nearest-object label is ambiguous at a shared endpoint. A
    # radius can also protect the immediately adjacent ARC from being made a
    # LINE, but only when a directed source arrow actually reaches that ARC.
    # Recompute distances from the original target pixel; graph-side hints and
    # model-selected IDs alone do not grant an annotation binding.
    entities = graph.get("entities", [])
    ids = [row.get("id") for row in entities]
    action = operation.get("action")
    requested_record = operation.get("record_id") if action in {
        "refit_chain_as_annotated_arc", "insert_annotated_fillet"} else None
    source_entities = None
    for original in graph.get("annotation_support", []):
        if (not isinstance(original, dict) or original.get("kind") != "radius" or
                original.get("status") != "candidate_supported" or
                not original.get("arrowhead_verified") or
                original.get("candidate_entity_id") in entity_ids or
                (requested_record is not None and original.get("record_id") != requested_record)):
            continue
        nearest = original.get("candidate_entity_id")
        if nearest not in ids or len(ids) <= 2:
            continue
        target = (original.get("source_evidence") or {}).get("target_source_px")
        if target is None:
            record = next((row for row in inventory if row.get("record_id") == original.get("record_id")), None)
            target = ((record or {}).get("leader") or {}).get("target_source_px")
        try:
            target = np.asarray(target, float)
        except (TypeError, ValueError):
            continue
        if target.shape != (2,) or not np.isfinite(target).all():
            continue
        index = ids.index(nearest)
        neighbors = {ids[(index - 1) % len(ids)], ids[(index + 1) % len(ids)]}
        allowed = neighbors.intersection(entity_ids)
        if action in {"merge_chain_as_line", "refit_entity_as_line"}:
            allowed = {value for value in allowed if entities[ids.index(value)].get("type") == "ARC"}
        if not allowed:
            continue
        if source_entities is None:
            axes = design_to_source(np.asarray([[0., 0.], [1., 0.], [0., 1.]], float))
            determinant = float(np.linalg.det(np.column_stack([axes[1] - axes[0], axes[2] - axes[0]])))
            source_entities = _base_source_entities(graph, design_to_source, determinant)
        nearest_gap = float(_primitive_distance(target[None, :], source_entities[index])[0])
        gap = min(float(_primitive_distance(target[None, :], source_entities[ids.index(value)])[0])
                  for value in allowed)
        if nearest_gap > max(12., 2.25 * grid) or gap > 2.25 * grid or gap - nearest_gap > 2.25 * grid:
            continue
        support.append({**original, "target_gap_px": gap,
                        "target_match_method": "bounded_adjacent_source_target_hypothesis",
                        "source_evidence": {**(original.get("source_evidence") or {}),
                                            "target_source_px": target.tolist()}})
    if operation.get("action") in {"merge_chain_as_line", "refit_entity_as_line"} and support:
        raise ValueError("radius_annotation_protects_arc_chain")
    force_arc = bool(support and operation.get("action") == "merge_chain_best_fit")
    if operation.get("action") not in {"refit_chain_as_annotated_arc", "insert_annotated_fillet"}:
        return force_arc, None, None
    record_id = operation.get("record_id")
    record = next((row for row in inventory if row.get("record_id") == record_id), None)
    nominal = record.get("nominal") if isinstance(record, dict) else None
    matched = next((row for row in support if row.get("record_id") == record_id), None)
    if not matched or isinstance(nominal, bool) or not isinstance(nominal, (int, float)) or not math.isfinite(nominal) or nominal <= 0:
        raise ValueError("radius_annotation_not_supported_by_named_chain")
    if graph.get("units") != "mm":
        raise ValueError("annotated_radius_requires_mm_scale")
    mapped = design_to_source(np.asarray([[0., 0.], [float(nominal), 0.], [0., float(nominal)]], float))
    scales = [float(np.linalg.norm(mapped[1] - mapped[0])), float(np.linalg.norm(mapped[2] - mapped[0]))]
    radius_px = sum(scales) / 2
    if not math.isfinite(radius_px) or radius_px <= 0 or abs(scales[0] - scales[1]) > .02 * radius_px:
        raise ValueError("annotated_radius_coordinate_scale_invalid")
    binding = {"record_id": record_id, "text": record.get("text"), "nominal": float(nominal),
               "method": "multimodal target plus local leader and fixed-radius source fit",
               "arrowhead_verified": bool(matched.get("arrowhead_verified")),
               "target_gap_px": float(matched.get("target_gap_px", 0.))}
    if matched.get("target_match_method"):
        binding["target_match_method"] = matched["target_match_method"]
        binding["target_match_entity_ids"] = list(entity_ids)
    target = (matched.get("source_evidence") or {}).get("target_source_px")
    if target is None:
        target = (record.get("leader") or {}).get("target_source_px")
    if target is not None:
        binding["target_source_px"] = copy.deepcopy(target)
    return True, radius_px, binding


def _source_landmarks(points, maximum=32):
    """Bound search by source arclength; preserve observed sharp corners."""
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distances[-1] <= 1e-8:
        raise ValueError("source_chain_has_zero_length")
    indices = {0, len(points) - 1}
    indices.update(np.searchsorted(distances, np.linspace(0., distances[-1], maximum)).tolist())
    return sorted(indices)


def _split_source_chain(points, tolerance, *, maximum_pieces=4):
    """Refit multiple primitives with shared movable ORIGINAL source joints.

    The DP minimizes count before residual; it does not create new pieces just
    to make noisy pixels look smoother. No model-supplied coordinates are used.
    """
    landmarks = _source_landmarks(points)
    edges = {}
    for i, first in enumerate(landmarks[:-1]):
        for last in landmarks[i + 1:]:
            part = points[first:last + 1]
            primitive = _line(part, tolerance) or _arc(part, tolerance)
            if primitive is not None:
                edges[first, last] = primitive
    best = {(0, 0): (0., [])}
    for count in range(1, maximum_pieces + 1):
        for last in landmarks[1:]:
            choices = []
            for first in landmarks:
                previous = best.get((count - 1, first))
                primitive = edges.get((first, last))
                if previous is not None and primitive is not None:
                    cost = previous[0] + primitive["fit_error_px"] ** 2 * (last - first)
                    choices.append((cost, [*previous[1], copy.deepcopy(primitive)]))
            if choices:
                best[count, last] = min(choices, key=lambda row: row[0])
        solution = best.get((count, len(points) - 1))
        if solution:
            if count == 1:
                raise ValueError("no_source_feature_requires_split")
            return solution[1]
    raise ValueError("source_features_exceed_bounded_replacement_budget")


def _annotated_line_fillet(points, radius, tolerance, binding):
    """Construct an exact tangent LINE/ARC/LINE corner from source evidence.

    Outer endpoints are fixed. Internal tangent points are constructed locally
    from the two observed support directions; a radius alone never invents a
    corner. Other primitive combinations must remain unresolved for later edits.
    """
    if not binding or not binding.get("arrowhead_verified"):
        raise ValueError("fillet_requires_verified_directed_radius_target")
    target = np.asarray(binding.get("target_source_px"), float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("fillet_requires_source_target_location")
    landmarks = _source_landmarks(points, maximum=26)
    a, b = points[0], points[-1]
    choices = []
    for i in landmarks[1:-1]:
        u = points[i] - a
        ul = float(np.linalg.norm(u))
        if ul <= max(tolerance * 2., radius * .1):
            continue
        u = u / ul
        for j in landmarks[1:-1]:
            if j <= i:
                continue
            v = b - points[j]
            vl = float(np.linalg.norm(v))
            if vl <= max(tolerance * 2., radius * .1):
                continue
            v = v / vl
            cross = float(u[0] * v[1] - u[1] * v[0])
            turn = math.atan2(cross, float(u @ v))
            if not math.radians(5) <= abs(turn) <= math.radians(160):
                continue
            try:
                lengths = np.linalg.solve(np.column_stack([u, v]), b - a)
            except np.linalg.LinAlgError:
                continue
            tangent_distance = radius * math.tan(abs(turn) / 2)
            if min(lengths) <= tangent_distance + tolerance:
                continue
            corner = a + lengths[0] * u
            first, last = corner - tangent_distance * u, corner + tangent_distance * v
            center = first + np.array([-u[1], u[0]]) * math.copysign(radius, turn)
            arc = {"type": "ARC", "start": first.tolist(), "end": last.tolist(),
                   "center": center.tolist(), "radius": float(radius), "clockwise": turn < 0}
            prefix = _line(np.vstack([points[:i + 1], first]), tolerance)
            suffix = _line(np.vstack([last, points[j:]]), tolerance)
            if prefix is None or suffix is None:
                continue
            primitive_distances = np.asarray([_primitive_distance(points, item) for item in (prefix, arc, suffix)])
            assignments = np.argmin(primitive_distances, axis=0)
            if np.count_nonzero(assignments == 1) < 3 or np.any(np.diff(assignments) < 0):
                continue
            error = float(primitive_distances.min(axis=0).max())
            target_gap = float(_primitive_distance(target[None, :], arc)[0])
            # Source-label target must reach the new small arc, not just some
            # different primitive in the same large named chain.
            if error > tolerance or target_gap > max(1., tolerance):
                continue
            arc.update(fit_error_px=error, radius_binding=copy.deepcopy(binding),
                       radius_annotation_evidence=copy.deepcopy(binding), radius_binding_status="applied")
            candidate = [prefix, arc, suffix]
            sampled, _, _ = _sample_entities(candidate, max_step_px=max(.25, min(1., tolerance / 2)))
            # Local bidirectional check prevents an analytic tangent arc from
            # taking an unsupported shortcut between valid endpoints.
            forward = float(cKDTree(sampled).query(points, workers=1)[0].max())
            reverse = float(cKDTree(points).query(sampled, workers=1)[0].max())
            if max(forward, reverse) > max(tolerance * 1.5, 1.):
                continue
            choices.append((max(forward, reverse), sum(p["fit_error_px"] for p in candidate), candidate))
    if not choices:
        raise ValueError("source_does_not_support_exact_annotated_fillet")
    return min(choices, key=lambda row: row[:2])[2]


def _replacement_chain(support, action, tolerance, selected_entities, *, force_arc=False,
                       annotated_radius_px=None, radius_binding=None):
    if action == "split_chain_at_source_features":
        replacements = _split_source_chain(support, tolerance)
    elif action == "insert_annotated_fillet":
        replacements = _annotated_line_fillet(support, annotated_radius_px, tolerance, radius_binding)
    else:
        replacements = [_fit_replacement(support, action, tolerance, force_arc=force_arc,
                                          annotated_radius_px=annotated_radius_px, radius_binding=radius_binding)]
    # Previously realized numeric radii are explicit constraints, not optional
    # display tags. An edit may retain/refit them, but cannot silently discard
    # them while replacing the surrounding chain.
    for original in selected_entities:
        bound = original.get("radius_binding")
        if not bound:
            continue
        target = bound.get("target_source_px")
        if target is None:
            samples, _, _ = _sample_entities([original], max_step_px=max(.25, tolerance))
            target = samples[len(samples) // 2]
        target = np.asarray(target, float)[None, :]
        matches = [row for row in replacements if row["type"] == "ARC"
                   and math.isclose(row["radius"], original["radius"], rel_tol=1e-8, abs_tol=1e-8)
                   and float(_primitive_distance(target, row)[0]) <= max(1., tolerance)]
        if not matches:
            raise ValueError("edit_would_discard_bound_radius")
        matches[0].update(radius_binding=copy.deepcopy(bound), radius_binding_status="applied")
    parents = sorted({sid for entity in selected_entities for sid in entity.get("parent_stable_ids", [])})
    ancestors = sorted(set(parents) | {sid for entity in selected_entities for sid in entity.get("ancestor_stable_ids", [])})
    displays = [sid for entity in selected_entities for sid in entity.get("parent_entity_ids", [])]
    for index, entity in enumerate(replacements):
        payload = json.dumps([parents, action, index, entity], sort_keys=True, allow_nan=False)
        entity.update(stable_id="e-" + hashlib.sha256(payload.encode("utf8")).hexdigest()[:20],
                      parent_entity_ids=displays, parent_stable_ids=parents, ancestor_stable_ids=ancestors,
                      parameter_source="multimodal_edit_local_refit")
        if len(replacements) == len(selected_entities) == 1:
            original = selected_entities[0]
            keys = ("start", "end", "center", "radius") if entity["type"] == "ARC" else ("start", "end")
            if (original["type"] == entity["type"] and
                    original.get("clockwise") == entity.get("clockwise") and
                    all(np.allclose(entity[key], original[key], rtol=1e-10, atol=1e-10) for key in keys)):
                entity["stable_id"] = original["stable_id"]
    return replacements


def _apply_one(graph, baseline, operation, inventory):
    action = operation.get("action")
    if action not in EDIT_ACTIONS:
        raise ValueError("unsupported_topology_edit_action")
    entity_ids = operation.get("entity_ids")
    minimum = _edit_minimum(action)
    indices = _ordered_indices(graph, entity_ids, minimum=minimum)
    if action == "refit_entity_as_line" and len(indices) != 1:
        raise ValueError("single_entity_refit_requires_one_entity")
    grid = float(graph.get("source_grid_pitch_px") or 1.)
    if not math.isfinite(grid) or grid <= 0:
        raise ValueError("invalid_source_grid")
    design_to_source, source_to_design, orientation_det = _affines(graph, baseline)
    force_arc, annotated_radius_px, radius_binding = _radius_edit_evidence(
        graph, operation, inventory, entity_ids, design_to_source)
    source_entities = _base_source_entities(graph, design_to_source, orientation_det)
    selected_entities = [source_entities[index] for index in indices]
    chain_samples, _, _ = _sample_entities(selected_entities, max_step_px=max(.35, min(1., grid / 2)))
    raw = _source_ring(baseline)
    ring = _closed_ring(raw) if raw is not None else _base_source_ring(graph, design_to_source, grid)
    start = np.asarray(selected_entities[0]["start"], float)
    end = np.asarray(selected_entities[-1]["end"], float)
    support, localization_mismatch = _ring_path(ring, start, end, chain_samples)
    tolerance = max(.25, float(graph.get("proposal_tolerance_px") or grid))
    replacements = _replacement_chain(support, action, tolerance, selected_entities, force_arc=force_arc,
                                      annotated_radius_px=annotated_radius_px, radius_binding=radius_binding)

    start_index = indices[0]
    rotated = source_entities[start_index:] + source_entities[:start_index]
    updated = [*replacements, *rotated[len(indices):]]
    if len(updated) < 3:
        raise ValueError("edit_would_collapse_closed_profile")
    for index, entity in enumerate(updated):
        following = updated[(index + 1) % len(updated)]
        if np.linalg.norm(np.asarray(entity["end"], float)-np.asarray(following["start"], float)) > 1e-6:
            raise ValueError("edited_chain_endpoint_gap")
    quality = assess_fit_quality(ring, updated, max_step_px=max(.5, min(2., grid / 2)))
    base_quality = assess_fit_quality(ring, source_entities, max_step_px=max(.5, min(2., grid / 2)))
    candidate_upper = quality["source_boundary_deviation_px"]["conservative_upper_bound_px"]
    base_upper = base_quality["source_boundary_deviation_px"]["conservative_upper_bound_px"]
    if not quality["sampled_topology_valid"]:
        raise ValueError("edited_topology_invalid")
    if candidate_upper > max(tolerance, base_upper + grid):
        raise ValueError("edited_boundary_deviation_exceeds_budget")
    return updated, quality, {
        "action": action, "entity_ids": list(entity_ids),
        "record_id": operation.get("record_id"),
        "evidence_tags": list(operation.get("evidence_tags") or []),
        "removed_entity_count": len(indices), "replacement_count": len(replacements),
        "replacement_type": replacements[0]["type"] if len(replacements) == 1 else "LINE_ARC_CHAIN",
        "replacement_types": [row["type"] for row in replacements],
        "net_entity_reduction": len(indices) - len(replacements),
        "replacement_fit_error_px": max(float(row.get("fit_error_px", 0.)) for row in replacements),
        "annotation_guided": bool(radius_binding),
        "radius_binding_applied": any(bool(row.get("radius_binding")) for row in replacements),
        "radius_binding": next((row["radius_binding"] for row in replacements if row.get("radius_binding")), None),
        "radius_binding_status": ("applied" if any(row.get("radius_binding") for row in replacements)
                                  else "unresolved_fixed_radius_fit_failed" if radius_binding else "not_requested"),
        "feature_restoration_validated": action == "insert_annotated_fillet" and any(bool(row.get("radius_binding")) for row in replacements),
        "source_support_vertex_count": int(len(support)),
        "source_path_localization_mismatch": float(localization_mismatch),
        "base_source_deviation_upper_px": float(base_upper),
        "edited_source_deviation_upper_px": float(candidate_upper),
        "ground_truth_used": False,
    }


def _apply_combined(graph, baseline, operations, inventory):
    """Apply non-overlapping, non-wrapping edits against the same source cycle."""
    if len(operations) < 2:
        raise ValueError("combined_edit_requires_multiple_operations")
    grid = float(graph.get("source_grid_pitch_px") or 1.)
    if not math.isfinite(grid) or grid <= 0:
        raise ValueError("invalid_source_grid")
    design_to_source, _, orientation_det = _affines(graph, baseline)
    source_entities = _base_source_entities(graph, design_to_source, orientation_det)
    raw = _source_ring(baseline)
    ring = _closed_ring(raw) if raw is not None else _base_source_ring(graph, design_to_source, grid)
    tolerance = max(.25, float(graph.get("proposal_tolerance_px") or grid))
    occupied, replacements, details = set(), {}, []
    for operation in operations:
        action = operation.get("action")
        if action not in EDIT_ACTIONS:
            raise ValueError("unsupported_topology_edit_action")
        minimum = _edit_minimum(action)
        indices = _ordered_indices(graph, operation.get("entity_ids"), minimum=minimum)
        # A chain crossing the cycle origin is still valid as a standalone
        # candidate.  Keeping combined edits non-wrapping makes the splice
        # auditable and prevents ambiguous remapping of the original IDs.
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("combined_edit_chain_wraps_cycle_origin")
        if occupied.intersection(indices):
            raise ValueError("combined_edit_chains_overlap")
        occupied.update(indices)
        individual, _, detail = _apply_one(graph, baseline, operation, inventory)
        # _apply_one rotates the named replacement to the beginning of its
        # standalone cycle, and explicitly reports how many pieces it emitted.
        replacements[indices[0]] = (len(indices), individual[:detail["replacement_count"]])
        details.append(detail)
    updated, index = [], 0
    while index < len(source_entities):
        replacement = replacements.get(index)
        if replacement is None:
            updated.append(source_entities[index])
            index += 1
        else:
            consumed, entities = replacement
            updated.extend(entities)
            index += consumed
    if len(updated) < 3:
        raise ValueError("edit_would_collapse_closed_profile")
    for index, entity in enumerate(updated):
        following = updated[(index + 1) % len(updated)]
        if np.linalg.norm(np.asarray(entity["end"], float) - np.asarray(following["start"], float)) > 1e-6:
            raise ValueError("edited_chain_endpoint_gap")
    quality = assess_fit_quality(ring, updated, max_step_px=max(.5, min(2., grid / 2)))
    base_quality = assess_fit_quality(ring, source_entities, max_step_px=max(.5, min(2., grid / 2)))
    candidate_upper = quality["source_boundary_deviation_px"]["conservative_upper_bound_px"]
    base_upper = base_quality["source_boundary_deviation_px"]["conservative_upper_bound_px"]
    if not quality["sampled_topology_valid"]:
        raise ValueError("edited_topology_invalid")
    if candidate_upper > max(tolerance, base_upper + grid):
        raise ValueError("edited_boundary_deviation_exceeds_budget")
    return updated, quality, {
        "action": "apply_nonoverlapping_edits",
        "operations": details,
        "removed_entity_count": int(sum(row["removed_entity_count"] for row in details)),
        "replacement_count": sum(row["replacement_count"] for row in details),
        "net_entity_reduction": int(sum(row["net_entity_reduction"] for row in details)),
        "radius_binding_applied": any(row["radius_binding_applied"] for row in details),
        "feature_restoration_validated": any(row["feature_restoration_validated"] for row in details),
        "base_source_deviation_upper_px": float(base_upper),
        "edited_source_deviation_upper_px": float(candidate_upper),
        "ground_truth_used": False,
    }


def execute_topology_edits(image_path, document, baseline, base_candidate, bundle, operations, output_dir):
    """Materialize independently auditable edit candidates.

    Each proposed operation produces one alternative.  Failed operations remain
    in the audit and never mutate the supplied candidate or bundle.
    """
    if not isinstance(operations, list) or len(operations) > 5:
        raise ValueError("topology_edit_operation_count_outside_bounds")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    image = _read_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    graph = base_candidate.get("graph")
    if not isinstance(graph, dict):
        raise ValueError("topology_edit_base_graph_missing")
    source_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if source_sha256 != bundle.get("source_sha256") or graph.get("source_sha256") != source_sha256:
        raise ValueError("topology_edit_source_hash_mismatch")
    records = canonical_records(document)
    design_to_source, source_to_design, orientation_det = _affines(graph, baseline)
    grid = float(graph.get("source_grid_pitch_px") or 1.)
    raw = _source_ring(baseline)
    ring = _closed_ring(raw) if raw is not None else _base_source_ring(graph, design_to_source, grid)
    inventory, _ = _annotation_inventory(gray, records, ring, grid)
    stroke = _StrokeEvidence(gray, records, grid)
    candidates, audit, accepted_operations = [], [], []

    def materialize(source_entities, quality, execution, candidate_id, strategy_description):
        sampled, _, _ = _sample_entities(source_entities, max_step_px=max(.5, min(2., grid / 2)))
        stroke_support = stroke.summarize(sampled)
        support, support_summary = _entity_annotation_support(source_entities, inventory, grid)
        support_summary["source_stroke_support"] = stroke_support
        before = graph.get("source_evidence", {}).get("proposal_stroke_support") or stroke_support
        support_gate = bool(
            float(stroke_support["edge_supported_fraction"]) >= float(before.get("edge_supported_fraction", 0.))-.035 and
            float(stroke_support["p90_edge_distance_px"]) <= float(before.get("p90_edge_distance_px", math.inf))+grid)
        if not support_gate:
            raise ValueError("edited_source_stroke_support_degraded")
        candidate_graph = _to_graph(
            source_entities, source_to_design, orientation_det, graph, candidate_id,
            source_sha256, bundle["base_graph_sha256"],
            float(graph.get("proposal_tolerance_px") or grid), quality, support, support_summary,
            {"name": "multimodal_local_topology_edit", "description": strategy_description},
        )
        candidate_graph["baseline_modified"] = True
        candidate_graph["source_evidence"].update(topology_edit=execution)
        counts = {"total": len(source_entities),
                  "LINE": sum(item["type"] == "LINE" for item in source_entities),
                  "ARC": sum(item["type"] == "ARC" for item in source_entities)}
        overlay_path = output / f"topology-candidate-{candidate_id}.png"
        candidate = {
            "id": candidate_id, "strategy": candidate_graph["source_evidence"]["strategy"],
            "parent_topology_sha256": bundle["base_graph_sha256"], "source_sha256": source_sha256,
            "graph": candidate_graph, "entity_counts": counts,
            "annotation_support": support_summary,
            "complexity": {"entity_count": counts["total"],
                           "relative_to_base": counts["total"] / max(1, len(graph.get("entities", []))),
                           "fit_tolerance_px": float(graph.get("proposal_tolerance_px") or grid)},
            "source_residual": quality, "source_stroke_support": stroke_support,
            "planner_signals": {"prior_rank": 0, "source_support_gate_passed": support_gate,
                                "reference_accuracy_measured": False,
                                "requires_constraint_binding_and_solve": True},
            "ground_truth_used": False, "overlay_path": str(overlay_path.resolve()),
        }
        _planner_fields(candidate, support, counts["total"])
        _render_overlay(image, source_entities, candidate_id, overlay_path)
        candidates.append(candidate)
        return counts

    for index, operation in enumerate(operations, 1):
        row = {"operation_index": index, "operation": copy.deepcopy(operation), "status": "rejected",
               "ground_truth_used": False}
        try:
            source_entities, quality, execution = _apply_one(
                graph, baseline, operation, bundle.get("annotation_inventory", []))
            suffix = {"merge_chain_as_line": "line", "merge_chain_as_arc": "arc",
                      "merge_chain_best_fit": "best",
                      "refit_chain_as_annotated_arc": "annotated-arc",
                      "refit_entity_as_line": "retype-line",
                      "split_chain_at_source_features": "source-split",
                      "insert_annotated_fillet": "annotated-fillet"}[operation["action"]]
            candidate_id = f"cand-edit-{index:02d}-{suffix}"
            counts = materialize(
                source_entities, quality, execution, candidate_id,
                "Agent proposed one entity-ID edit; the local source-only geometry kernel executed and refit it.")
            row.update(status="accepted_as_candidate", candidate_id=candidate_id, execution=execution,
                       entity_counts=counts)
            accepted_operations.append((index, operation))
        except (ValueError, ArithmeticError, KeyError, TypeError) as error:
            row["reason"] = str(error)[:160]
        audit.append(row)
    # A failed proposal cannot poison successful independent edits. Preserve
    # the full per-operation audit, then combine a deterministic disjoint subset.
    subset, occupied, excluded = [], set(), []
    for index, operation in accepted_operations:
        indices = _ordered_indices(graph, operation.get("entity_ids"), minimum=_edit_minimum(operation["action"]))
        if occupied.intersection(indices) or indices != list(range(indices[0], indices[0] + len(indices))):
            excluded.append(index)
            continue
        subset.append((index, operation))
        occupied.update(indices)
    if len(subset) > 1:
        row = {"operation_index": "combined", "operation": {"action": "apply_nonoverlapping_edits",
               "operation_count": len(subset), "included_operation_indices": [index for index, _ in subset],
               "excluded_accepted_operation_indices": excluded}, "status": "rejected", "ground_truth_used": False}
        try:
            source_entities, quality, execution = _apply_combined(
                graph, baseline, [operation for _, operation in subset], bundle.get("annotation_inventory", []))
            candidate_id = "cand-edit-all"
            counts = materialize(
                source_entities, quality, execution, candidate_id,
                "Successful non-overlapping edits applied together by the local source-only geometry kernel.")
            row.update(status="accepted_as_candidate", candidate_id=candidate_id, execution=execution,
                       entity_counts=counts)
        except (ValueError, ArithmeticError, KeyError, TypeError) as error:
            row["reason"] = str(error)[:160]
        audit.append(row)
    return candidates, {"schema_version": "local-topology-edit-execution-v1",
                        "base_candidate_id": base_candidate.get("id"),
                        "proposed": len(operations), "accepted_candidates": len(candidates),
                        "operations": audit, "ground_truth_used": False}

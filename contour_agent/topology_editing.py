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
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .ocr import canonical_records
from .topology import _StrokeEvidence, _primitive_distance, stroke_support_fraction
from .topology_candidates import (
    _affines, _annotation_inventory, _base_source_entities, _base_source_ring,
    _entity_annotation_support, _planner_fields, _read_image, _render_overlay,
    _to_graph,
)
from .vectorize import _arc, _closed_ring, _line, _sample_entities, assess_fit_quality


EDIT_ACTIONS = {"merge_chain_as_line", "merge_chain_as_arc", "merge_chain_best_fit",
                "refit_chain_as_annotated_arc", "refit_entity_as_line",
                "split_chain_at_source_features", "insert_annotated_fillet",
                "restore_annotated_line_support"}


def _edit_minimum(action):
    return 2 if action.startswith("merge_chain_") else 1


def _source_endpoint_pair(graph, entity):
    nodes = {row.get("id"): row for row in graph.get("nodes", []) if isinstance(row, dict)}
    a = nodes.get(entity.get("start_node"), {}).get("source_px")
    b = nodes.get(entity.get("end_node"), {}).get("source_px")
    if a is None or b is None:
        if graph.get("units") != "pixel":
            return None
        a, b = entity.get("start"), entity.get("end")
    pair = np.asarray([a, b], float)
    return pair if pair.shape == (2, 2) and np.isfinite(pair).all() else None


def _unique_source_targeted_unbound_arc(graph, inventory, entity_id, record_id):
    """Admit an unbound ARC only as a source-backed fillet *proposal*.

    A rough fitted radius is not dimensional evidence. The named OCR radius
    must have one directed, full-shaft, first-contour source claim on this ARC;
    fresh binding and solving remain mandatory after construction.
    """
    arc = next((row for row in graph.get("entities", []) if row.get("id") == entity_id), None)
    record = next((row for row in inventory if row.get("record_id") == record_id and
                   row.get("kind") == "radius"), None)
    nominal = (record or {}).get("nominal")
    annotation = arc.get("radius_annotation_evidence") if isinstance(arc, dict) else None
    if (not isinstance(arc, dict) or arc.get("type") != "ARC" or
            arc.get("radius_binding") or arc.get("radius_constructed") or
            arc.get("dimension_bound") or arc.get("constraint_ids") or
            arc.get("radius_binding_status") not in
            {None, "not_requested", "unresolved_fixed_radius_fit_failed"} or
            (annotation is not None and
             (not isinstance(annotation, dict) or annotation.get("record_id") != record_id)) or
            isinstance(nominal, bool) or not isinstance(nominal, (int, float)) or
            not math.isfinite(nominal) or nominal <= 0):
        return False
    claims = [row for row in graph.get("annotation_support", []) if isinstance(row, dict) and
              row.get("kind") == "radius" and row.get("status") == "candidate_supported" and
              row.get("arrowhead_verified") is True and row.get("record_id") == record_id]
    if len(claims) != 1 or claims[0].get("candidate_entity_id") != entity_id:
        return False
    if any(row.get("record_id") != record_id and row.get("candidate_entity_id") == entity_id and
           row.get("kind") == "radius" and row.get("arrowhead_verified") is True
           for row in graph.get("annotation_support", []) if isinstance(row, dict)):
        return False
    claim = claims[0]
    if any(row.get("entity_type") == "ARC" and row.get("entity_id") != entity_id
           for row in claim.get("adjacent_target_hypotheses", []) if isinstance(row, dict)):
        return False
    evidence = claim.get("source_evidence") or {}
    target = evidence.get("target_source_px")
    try: target = np.asarray(target, float)
    except (TypeError, ValueError): return False
    grid = graph.get("source_grid_pitch_px") or 1.
    gap = claim.get("target_gap_px")
    visibility = evidence.get("contour_visibility") or {}
    first_hit = visibility.get("first_intersection_px")
    try: first_hit = np.asarray(first_hit, float)
    except (TypeError, ValueError): return False
    return bool(target.shape == (2,) and np.isfinite(target).all() and
                first_hit.shape == (2,) and np.isfinite(first_hit).all() and
                isinstance(grid, (int, float)) and math.isfinite(grid) and grid > 0 and
                isinstance(gap, (int, float)) and math.isfinite(gap) and
                gap <= 2.25 * grid and
                (evidence.get("shaft_evidence") or {}).get("verified") is True and
                visibility.get("verified") is True and
                (evidence.get("arrowhead") or {}).get("verified") is True)


def _complete_fillet_support_scope(graph, inventory, operation):
    """Include the missing support beyond a short, source-targeted corner.

    Only the two existing outermost primitives can trigger one-neighbor
    expansion. This is a bounded edit hypothesis, never acceptance evidence.
    """
    requested = list(operation.get("entity_ids") or [])
    if operation.get("action") != "insert_annotated_fillet" or graph.get("units") != "mm":
        return requested
    indices = _ordered_indices(graph, requested, minimum=1)
    entities = graph["entities"]
    # An existing source-bound radius between two finite LINE supports is
    # already the complete local domain.  Expanding past either support would
    # let an unrelated neighboring feature influence a tangency repair.
    if (len(indices) == 3 and [entities[index].get("type") for index in indices] ==
            ["LINE", "ARC", "LINE"] and
            ((entities[indices[1]].get("radius_binding") or {}).get("record_id") == operation.get("record_id") or
             _unique_source_targeted_unbound_arc(
                 graph, inventory, entities[indices[1]]["id"], operation.get("record_id")))):
        return requested
    record = next((r for r in inventory if r.get("record_id") == operation.get("record_id")), {})
    nominal = record.get("nominal")
    if isinstance(nominal, bool) or not isinstance(nominal, (int, float)) or not math.isfinite(nominal) or nominal <= 0:
        return requested
    support = next((r for r in graph.get("annotation_support", [])
                    if r.get("record_id") == operation.get("record_id") and r.get("arrowhead_verified") is True
                    and r.get("candidate_entity_id") in requested), None)
    if support is None:
        return requested
    target = (support.get("source_evidence") or {}).get("target_source_px")
    if target is None:
        target = (record.get("leader") or {}).get("target_source_px")
    target = np.asarray(target, float)
    if target.shape != (2,) or not np.isfinite(target).all():
        return requested
    expanded = list(indices)
    for endpoint, step in ((indices[0], -1), (indices[-1], 1)):
        if len(expanded) >= min(8, len(entities)-1):
            break
        entity = entities[endpoint]
        pair = _source_endpoint_pair(graph, entity)
        if pair is None or entity.get("type") not in {"LINE", "ARC"}:
            continue
        if not all(isinstance(entity.get(key), (list, tuple)) and len(entity[key]) == 2
                   for key in ("start", "end")):
            continue
        length = math.dist(entity["start"], entity["end"])
        source_length = float(np.linalg.norm(pair[1]-pair[0]))
        if length <= 1e-9 or source_length <= 1e-9 or length > 2*nominal:
            continue
        radius_px = nominal*source_length/length
        if float(np.linalg.norm(pair-target, axis=1).min()) > 2*radius_px:
            continue
        neighbor = (endpoint+step) % len(entities)
        other = entities[neighbor]
        if (neighbor in expanded or other.get("type") not in {"LINE", "ARC"}
                or math.dist(other["start"], other["end"]) <= 1.5*length):
            continue
        expanded = [neighbor, *expanded] if step < 0 else [*expanded, neighbor]
    # A provider may include the preceding annotated ARC when pointing at a
    # short LINE connector.  Completing the other side then produces a four-
    # primitive scope even though the directed fillet is owned by the two
    # finite LINE supports and their short connector.  Select that smallest
    # source-witnessed chain before geometric construction; the ordinary
    # source-fit, arrow, constraint and solver gates still decide acceptance.
    if len(expanded) > 3:
        windows = []
        for offset in range(len(expanded) - 2):
            window = expanded[offset:offset + 3]
            first, connector, last = (entities[index] for index in window)
            if any(row.get("type") != "LINE" for row in (first, connector, last)):
                continue
            if len({row["id"] for row in (first, connector, last)}.intersection(requested)) < 2:
                continue
            if (connector.get("radius_binding") or connector.get("dimension_bound")
                    or connector.get("constraint_ids")):
                continue
            lengths = [math.dist(row["start"], row["end"])
                       for row in (first, connector, last)]
            if (lengths[1] <= 1e-8 or lengths[1] > 2 * nominal
                    or min(lengths[0], lengths[2]) <= 1.5 * lengths[1]):
                continue
            source_pair = _source_endpoint_pair(graph, connector)
            if source_pair is None:
                continue
            source_length = float(np.linalg.norm(source_pair[1] - source_pair[0]))
            if source_length <= 1e-8:
                continue
            radius_px = nominal * source_length / lengths[1]
            target_gap = float(np.linalg.norm(source_pair - target, axis=1).min())
            if target_gap > 2 * radius_px:
                continue
            windows.append((target_gap / radius_px, window))
        if len(windows) == 1:
            expanded = windows[0][1]
        elif windows:
            windows.sort(key=lambda row: row[0])
            # Two equally plausible corners require a separate source review.
            if windows[1][0] - windows[0][0] > .25:
                expanded = windows[0][1]
    return [entities[index]["id"] for index in expanded]


def propose_annotation_arc_edits(graph, inventory, *, limit=4, unresolved_radius_record_ids=None):
    """Create bounded source hypotheses; the executor/evaluator still validate.

    Reserve room for a directed small fillet and a shallow unlabelled ARC->LINE
    repair, so repeated numeric refits cannot starve structural corrections.
    """
    if not isinstance(graph, dict) or not isinstance(inventory, list) or limit <= 0:
        return []
    from .annotation_line_support import propose_annotation_line_edits
    angle_lines = propose_annotation_line_edits(graph, inventory, limit=1)
    entities = {row.get("id"): row for row in graph.get("entities", []) if isinstance(row, dict)}
    records = {row.get("record_id"): row for row in inventory if isinstance(row, dict)
               and row.get("record_id") and row.get("kind") == "radius"}
    supports = [row for row in graph.get("annotation_support", []) if isinstance(row, dict)
                and row.get("kind") == "radius" and row.get("status") == "candidate_supported"
                and row.get("candidate_entity_id") in entities and row.get("record_id") in records]
    by_entity = {}
    for row in supports:
        by_entity.setdefault(row["candidate_entity_id"], []).append(row)
    unresolved = set(unresolved_radius_record_ids or [])
    ranked, fillets = [], []
    ordered_entities = list(entities.values())
    def source_endpoints(entity):
        return _source_endpoint_pair(graph, entity)

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
        # Legacy graphs may only carry node IDs/source-pixel endpoints.  Only
        # native millimetre endpoints can justify the radius-based reach here.
        native_endpoints = np.asarray([
            entities[entity_id].get("start"), entities[entity_id].get("end")
        ], dtype=float)
        corner_transition = (
            graph.get("units") == "mm"
            and native_endpoints.shape == (2, 2)
            and np.isfinite(native_endpoints).all()
            and math.dist(*native_endpoints) <= 2 * float(record["nominal"])
        )
        reach = chord if corner_transition else .30*chord
        if float(gaps[near]) > max(8 * float(graph.get("source_grid_pitch_px") or 1.), reach):
            return
        position = next(i for i, row in enumerate(ordered_entities) if row["id"] == entity_id)
        pair = ([ordered_entities[(position - 1) % len(ordered_entities)]["id"], entity_id]
                if near == 0 else [entity_id, ordered_entities[(position + 1) % len(ordered_entities)]["id"]])
        operation = {"action": "insert_annotated_fillet", "entity_ids": pair,
            "record_id": record["record_id"], "evidence_tags": ["annotation_target", "source_boundary"]}
        operation["entity_ids"] = _complete_fillet_support_scope(graph, inventory, operation)
        fillets.append(((2 if primitive_conflict and record["record_id"] in unresolved else 0,
                         mismatch), operation))

    def propose_bound_arc_tangent_reinsertion(entity_id, support, record):
        """Propose exact-R reinsertion of a source-targeted ARC between LINEs."""
        arc = entities[entity_id]
        binding = arc.get("radius_binding") or {}
        already_bound = binding.get("record_id") == record["record_id"]
        unique_unbound = _unique_source_targeted_unbound_arc(
            graph, inventory, entity_id, record["record_id"])
        if (arc.get("type") != "ARC" or not (already_bound or unique_unbound) or
                support.get("arrowhead_verified") is not True or
                (support.get("source_evidence") or {}).get("target_source_px") is None
                or len(ordered_entities) < 4):
            return
        position = next(i for i, row in enumerate(ordered_entities) if row["id"] == entity_id)
        before, after = (ordered_entities[(position-1) % len(ordered_entities)],
                         ordered_entities[(position+1) % len(ordered_entities)])
        if before.get("type") != "LINE" or after.get("type") != "LINE":
            return
        if unique_unbound and any(math.dist(line["start"], line["end"]) < float(record["nominal"])
                                  for line in (before, after)):
            return
        # The construction kernel checks the complete source ring and arrow;
        # this geometric deviation is only a trigger, never acceptance proof.
        deviations = [float(_fillet_tangent(before, "end") @ _fillet_tangent(arc, "start")),
                      float(_fillet_tangent(arc, "end") @ _fillet_tangent(after, "start"))]
        radius_matches = math.isclose(float(arc.get("radius", 0.)), float(record["nominal"]),
                                      rel_tol=1e-8, abs_tol=1e-8)
        if radius_matches and min(deviations) >= math.cos(math.radians(1.)):
            return
        operation = {"action": "insert_annotated_fillet",
                     "entity_ids": [before["id"], entity_id, after["id"]],
                     "record_id": record["record_id"],
                     "evidence_tags": ["annotation_target",
                                       "existing_radius_arc" if already_bound else "unique_source_targeted_unbound_arc",
                                       "finite_line_supports", "source_boundary"]}
        fillets.append(((1 if record["record_id"] in unresolved else 0,
                         1.+max(1.-value for value in deviations)), operation))

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
            if not primitive_conflict and len(rows) == 1:
                propose_bound_arc_tangent_reinsertion(entity_id, support, record)
                if mismatch < .20:
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
    # A verified radius can land on a primitive containing two true source
    # arcs. Reserve a partition hypothesis before repeatedly forcing either
    # annotation onto that whole primitive. The executor still checks every
    # target, exact radius, source sample and whole-contour acceptance gate.
    positions={row["id"]:i for i,row in enumerate(ordered_entities)}
    verified=sorted([row for row in supports if row.get("arrowhead_verified") is True],
                    key=lambda row:positions[row["candidate_entity_id"]])
    partitions=[]
    for first,second in zip(verified,verified[1:]):
        lo,hi=positions[first["candidate_entity_id"]],positions[second["candidate_entity_id"]]
        if (hi-lo>2 or first["record_id"]==second["record_id"] or
                records[first["record_id"]].get("nominal")==records[second["record_id"]].get("nominal")):
            continue
        # Start with the smallest named span containing both observed targets;
        # adding unlabelled neighbors can introduce a straight tail or another
        # fillet outside this exact-radius partition's declared scope.
        indices=range(lo,hi+1)
        ids=[ordered_entities[index]["id"] for index in indices]
        included={row["record_id"] for row in verified if row["candidate_entity_id"] in ids}
        if 2<=len(included)<=3 and 1<=len(ids)<=8:
            # A target on an *outer* endpoint needs the neighbor on its other
            # side. Prefer enclosing it at an internal joint, and avoid
            # repeatedly consuming the sole partition slot on an already
            # constructed radius merely because it occurs first in the cycle.
            first_pair=source_endpoints(ordered_entities[lo]);last_pair=source_endpoints(ordered_entities[hi])
            outer_targets=0
            if first_pair is not None and last_pair is not None:
                outer=np.asarray([first_pair[0],last_pair[1]])
                for support in (row for row in verified if row["record_id"] in included):
                    target=(support.get("source_evidence") or {}).get("target_source_px")
                    if target is None:target=(records[support["record_id"]].get("leader") or {}).get("target_source_px")
                    if target is not None and np.asarray(target).shape==(2,):
                        outer_targets+=int(np.linalg.norm(outer-np.asarray(target,float),axis=1).min()<=2.25*float(graph.get("source_grid_pitch_px") or 1.))
            protected_count=sum(bool(entities[entity_id].get("radius_binding")) for entity_id in ids)
            priority=(len(included&unresolved),-outer_targets,-protected_count,-len(ids))
            partitions.append((priority,{"action":"split_chain_at_source_features","entity_ids":ids,"record_id":None,
                                          "evidence_tags":["annotation_target","source_boundary","continuity"]}))
    if partitions:
        reserved.insert(0,max(partitions,key=lambda row:row[0])[1])
    reserved = [*angle_lines, *reserved]
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
        second_distance = _source_polyline_distance(chain_samples, candidate)
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
    if (action == "insert_annotated_fillet" and len(entity_ids) == 3 and
            _unique_source_targeted_unbound_arc(graph, inventory, entity_ids[1], record_id)):
        # This admits a bounded construction trial, not a dimension binding.
        binding["unique_unbound_arc_source_claim"] = True
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


def _source_polyline_distance(samples, points):
    """Exact point-to-segment distance; sparse contour vertices are not holes."""
    samples=np.asarray(samples,float);points=np.asarray(points,float)
    starts=points[:-1];vectors=np.diff(points,axis=0);lengths=np.sum(vectors*vectors,axis=1)
    valid=lengths>1e-20;starts=starts[valid];vectors=vectors[valid];lengths=lengths[valid]
    if not len(starts):return np.linalg.norm(samples-points[0],axis=1)
    distances=[]
    for offset in range(0,len(samples),128):
        relative=samples[offset:offset+128,None,:]-starts[None,:,:]
        fraction=np.clip(np.sum(relative*vectors[None,:,:],axis=2)/lengths[None,:],0.,1.)
        distances.extend(np.min(np.linalg.norm(relative-fraction[:,:,None]*vectors[None,:,:],axis=2),axis=1))
    return np.asarray(distances)


def _refined_straight_fillet(points, radius, tolerance, binding):
    """Jointly adjust two source support directions with fixed outer endpoints.

    The circle is constructed from the annotation at every trial. Only two
    support angles and a minimax epigraph variable are optimized; no free
    radius, source points, outer endpoints or acceptance budgets are changed.
    """
    a,b=points[0],points[-1];target=np.asarray(binding["target_source_px"],float)
    stations=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if stations[-1]<=1e-8:raise ValueError("source_does_not_support_exact_annotated_fillet")
    def station_point(fraction):
        return np.array([np.interp(fraction*stations[-1],stations,points[:,axis]) for axis in range(2)])
    directions=[station_point(.1)-a,b-station_point(.9)]
    angles=np.array([math.atan2(d[1],d[0]) for d in directions])
    angle_scale=tolerance/max(radius,float(np.linalg.norm(b-a)),tolerance)
    def construct(vector):
        headings=angles+vector[:2]*angle_scale
        u=np.array([math.cos(headings[0]),math.sin(headings[0])]);v=np.array([math.cos(headings[1]),math.sin(headings[1])])
        turn=math.atan2(float(u[0]*v[1]-u[1]*v[0]),float(u@v))
        if not math.radians(5)<=abs(turn)<=math.radians(160):return None
        try:lengths=np.linalg.solve(np.column_stack([u,v]),b-a)
        except np.linalg.LinAlgError:return None
        distance=radius*math.tan(abs(turn)/2)
        if min(lengths)<=distance+tolerance:return None
        corner=a+lengths[0]*u;first=corner-distance*u;last=corner+distance*v
        center=first+np.array([-u[1],u[0]])*math.copysign(radius,turn)
        return [{"type":"LINE","start":a.tolist(),"end":first.tolist()},
                {"type":"ARC","start":first.tolist(),"end":last.tolist(),"center":center.tolist(),"radius":float(radius),"clockwise":turn<0},
                {"type":"LINE","start":last.tolist(),"end":b.tolist()}]
    def errors(vector):
        candidate=construct(vector)
        if candidate is None:return np.full(len(points)+1,1e6)
        distance=np.min(np.asarray([_primitive_distance(points,e) for e in candidate]),axis=0)/tolerance
        target_gap=float(_primitive_distance(target[None,:],candidate[1])[0])/max(1.,tolerance)
        return np.r_[distance,target_gap]
    initial=np.array([0.,0.,max(1.,float(errors(np.zeros(3)).max()))])
    if initial[2]>1e5:raise ValueError("source_does_not_support_exact_annotated_fillet")
    fit=minimize(lambda x:x[2]+1e-9*float(x[:2]@x[:2]),initial,method="SLSQP",
                 bounds=[(-8.,8.),(-8.,8.),(0.,max(8.,initial[2]+1.))],
                 constraints=[{"type":"ineq","fun":lambda x:x[2]-errors(x)}],
                 options={"maxiter":120,"ftol":1e-10})
    candidate=construct(fit.x)
    if not fit.success or candidate is None or float(errors(fit.x).max())>1.+1e-10:
        raise ValueError("source_does_not_support_exact_annotated_fillet")
    distances=np.asarray([_primitive_distance(points,e) for e in candidate]);assignments=np.argmin(distances,axis=0)
    if np.count_nonzero(assignments==1)<3 or np.any(np.diff(assignments)<0):
        raise ValueError("source_does_not_support_exact_annotated_fillet")
    sampled=_sample_entities(candidate,max_step_px=max(.25,min(1.,tolerance/2)))[0]
    forward=float(cKDTree(sampled).query(points,workers=1)[0].max())
    reverse=float(_source_polyline_distance(sampled,points).max())
    if max(forward,reverse)>max(tolerance*1.5,1.):
        raise ValueError("source_does_not_support_exact_annotated_fillet")
    for index,e in enumerate(candidate):
        assigned=distances[index,assignments==index]
        e["fit_error_px"]=float(assigned.max()) if len(assigned) else 0.
    candidate[1].update(radius_binding=copy.deepcopy(binding),radius_annotation_evidence=copy.deepcopy(binding),
        radius_binding_status="applied",fillet_construction="source_support_joint_angle_minimax",
        source_refinement={"outer_endpoints_fixed":True,"radius_exact":True,"optimizer_iterations":int(fit.nit),
                           "maximum_source_error_px":float(distances.min(axis=0).max()),"maximum_reverse_error_px":reverse})
    return candidate


def _annotated_straight_fillet(points, radius, tolerance, binding):
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
            reverse = float(_source_polyline_distance(sampled, points).max())
            if max(forward, reverse) > max(tolerance * 1.5, 1.):
                continue
            choices.append((max(forward, reverse), sum(p["fit_error_px"] for p in candidate), candidate))
    if not choices:
        raise ValueError("source_does_not_support_exact_annotated_fillet")
    return min(choices, key=lambda row: row[:2])[2]


def _fillet_offset_loci(primitive, radius):
    """Possible fillet-center loci for a locally source-fitted support."""
    if primitive["type"] == "LINE":
        a, b = np.asarray(primitive["start"]), np.asarray(primitive["end"])
        direction = (b-a)/np.linalg.norm(b-a)
        normal = np.array([-direction[1], direction[0]])
        return [{"type": "LINE", "point": a+sign*radius*normal, "direction": direction}
                for sign in (-1, 1)]
    center = np.asarray(primitive["center"])
    return [{"type": "CIRCLE", "center": center, "radius": distance}
            for distance in (primitive["radius"]+radius, abs(primitive["radius"]-radius))
            if distance > 1e-8]


def _fillet_locus_intersections(first, second):
    if first["type"] == second["type"] == "LINE":
        try:
            distance = np.linalg.solve(np.column_stack([first["direction"], -second["direction"]]),
                                       second["point"]-first["point"])[0]
            return [first["point"]+distance*first["direction"]]
        except np.linalg.LinAlgError:
            return []
    if first["type"] == "LINE" or second["type"] == "LINE":
        line, circle = (first, second) if first["type"] == "LINE" else (second, first)
        projection = float((circle["center"]-line["point"]) @ line["direction"])
        nearest = line["point"]+projection*line["direction"]
        height_squared = circle["radius"]**2-float(np.sum((nearest-circle["center"])**2))
        if height_squared < -1e-9:
            return []
        height = math.sqrt(max(0., height_squared))
        return [nearest+sign*height*line["direction"] for sign in (-1, 1)]
    delta = second["center"]-first["center"]
    distance = float(np.linalg.norm(delta))
    r1, r2 = first["radius"], second["radius"]
    # Coincident offset circles do not establish a unique source-supported
    # fillet center. Reject the underdetermined family rather than invent one.
    if distance <= 1e-8 or distance > r1+r2+1e-9 or distance < abs(r1-r2)-1e-9:
        return []
    along = (r1*r1-r2*r2+distance*distance)/(2*distance)
    height_squared = r1*r1-along*along
    if height_squared < -1e-9:
        return []
    unit = delta/distance; middle = first["center"]+along*unit
    normal = np.array([-unit[1], unit[0]])
    return [middle+sign*math.sqrt(max(0., height_squared))*normal for sign in (-1, 1)]


def _fillet_contact(primitive, center, radius):
    if primitive["type"] == "LINE":
        a, b = np.asarray(primitive["start"]), np.asarray(primitive["end"])
        unit = (b-a)/np.linalg.norm(b-a)
        return [a+float((center-a) @ unit)*unit]
    source_center = np.asarray(primitive["center"])
    distance = float(np.linalg.norm(center-source_center))
    if distance <= 1e-8:
        return []
    unit = (center-source_center)/distance
    return [point for sign in (-1, 1)
            if abs(float(np.linalg.norm((point := source_center+sign*primitive["radius"]*unit)-center))-radius)
            <= max(1e-7, radius*1e-10)]


def _fillet_tangent(primitive, endpoint):
    if primitive["type"] == "LINE":
        tangent = np.asarray(primitive["end"])-primitive["start"]
    else:
        radial = np.asarray(primitive[endpoint])-primitive["center"]
        tangent = np.array([-radial[1], radial[0]])*(-1 if primitive["clockwise"] else 1)
    return tangent/max(float(np.linalg.norm(tangent)), 1e-12)


def _annotated_curved_fillet(points, radius, tolerance, binding):
    """Construct exact tangent fillets with one or two circular supports.

    Only source-fitted side circles determine the offset loci. Their outer
    endpoints remain fixed; no model coordinates or guessed support centers
    enter construction. The existing whole-chain gates remain authoritative.
    """
    target = np.asarray(binding["target_source_px"], float)
    landmarks = _source_landmarks(points, maximum=26)
    prefixes, suffixes = {}, {}
    for index in landmarks[1:-1]:
        for table, part in ((prefixes, points[:index+1]), (suffixes, points[index:])):
            if np.linalg.norm(part[-1]-part[0]) <= max(tolerance*2., radius*.1):
                continue
            primitive = _line(part, tolerance) or _arc(part, tolerance)
            if primitive is not None:
                table[index] = primitive
    choices = []
    for i, before in prefixes.items():
        for j, after in suffixes.items():
            if j <= i or before["type"] == after["type"] == "LINE":
                continue
            for first_locus in _fillet_offset_loci(before, radius):
                for last_locus in _fillet_offset_loci(after, radius):
                    for center in _fillet_locus_intersections(first_locus, last_locus):
                        for first in _fillet_contact(before, center, radius):
                            for last in _fillet_contact(after, center, radius):
                                prefix, suffix = copy.deepcopy(before), copy.deepcopy(after)
                                prefix["end"], suffix["start"] = first.tolist(), last.tolist()
                                if any(np.linalg.norm(np.asarray(p["end"])-p["start"]) <= tolerance
                                       for p in (prefix, suffix)):
                                    continue
                                for clockwise in (False, True):
                                    arc = {"type": "ARC", "start": first.tolist(), "end": last.tolist(),
                                           "center": center.tolist(), "radius": float(radius), "clockwise": clockwise}
                                    if (_fillet_tangent(prefix, "end") @ _fillet_tangent(arc, "start") < 1-1e-8 or
                                            _fillet_tangent(arc, "end") @ _fillet_tangent(suffix, "start") < 1-1e-8):
                                        continue
                                    start_angle = math.atan2(first[1]-center[1], first[0]-center[0])
                                    end_angle = math.atan2(last[1]-center[1], last[0]-center[0])
                                    sweep = ((start_angle-end_angle) if clockwise else (end_angle-start_angle)) % (2*math.pi)
                                    if not math.radians(5) <= sweep <= math.radians(160):
                                        continue
                                    candidate = [prefix, arc, suffix]
                                    distances = np.asarray([_primitive_distance(points, item) for item in candidate])
                                    assignments = np.argmin(distances, axis=0)
                                    if (np.count_nonzero(assignments == 1) < 3 or np.any(np.diff(assignments) < 0) or
                                            float(distances.min(axis=0).max()) > tolerance or
                                            float(_primitive_distance(target[None, :], arc)[0]) > max(1., tolerance)):
                                        continue
                                    try:
                                        sampled, _, _ = _sample_entities(candidate, max_step_px=max(.25, min(1., tolerance/2)))
                                    except ValueError:
                                        continue
                                    forward = float(cKDTree(sampled).query(points, workers=1)[0].max())
                                    reverse = float(_source_polyline_distance(sampled, points).max())
                                    if max(forward, reverse) > max(tolerance*1.5, 1.):
                                        continue
                                    for index, primitive in enumerate(candidate):
                                        assigned = distances[index, assignments == index]
                                        primitive["fit_error_px"] = float(assigned.max()) if len(assigned) else math.inf
                                    arc.update(radius_binding=copy.deepcopy(binding), radius_annotation_evidence=copy.deepcopy(binding),
                                               radius_binding_status="applied", fillet_construction="source_support_offset_loci")
                                    choices.append((max(forward, reverse), sum(p["fit_error_px"] for p in candidate), candidate))
    if not choices:
        raise ValueError("source_does_not_support_exact_annotated_fillet")
    return min(choices, key=lambda row: row[:2])[2]


def _annotated_existing_line_fillet(points, radius, tolerance, binding, selected):
    """Preserve existing LINE supports; a fillet cannot rotate or replace them.

    Only a two-line junction, one short unbound connector, or one already
    source-bound radius ARC between those same finite supports is eligible.
    A three-piece edit may redistribute only its existing middle domain.
    Source residuals and the directed target still have to justify the exact R.
    An infeasible construction remains unresolved instead of refitting supports.
    """
    failure = "source_does_not_support_exact_annotated_fillet"
    if not binding or binding.get("arrowhead_verified") is not True:
        raise ValueError("fillet_requires_verified_directed_radius_target")
    target = np.asarray(binding.get("target_source_px"), float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("fillet_requires_source_target_location")
    points = np.asarray(points, float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 5
            or not np.isfinite(points).all() or not math.isfinite(radius)
            or radius <= 0 or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError(failure)
    middle_arc = (len(selected) == 3 and selected[1].get("type") == "ARC")
    if (len(selected) not in (2, 3) or selected[0].get("type") != "LINE"
            or selected[-1].get("type") != "LINE"
            or (len(selected) == 3 and selected[1].get("type") not in {"LINE", "ARC"})):
        raise ValueError("fillet_requires_two_existing_line_supports")
    for first, second in zip(selected, selected[1:]):
        if math.dist(first["end"], second["start"]) > 1e-6:
            raise ValueError("fillet_support_chain_not_connected")
    a, b = np.asarray(selected[0]["start"], float), np.asarray(selected[-1]["end"], float)
    u = np.asarray(selected[0]["end"], float)-a
    v = b-np.asarray(selected[-1]["start"], float)
    ul, vl = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if min(ul, vl) <= max(1e-8, tolerance):
        raise ValueError("fillet_would_consume_line_support")
    if (np.linalg.norm(points[0]-a) > 1e-6 or np.linalg.norm(points[-1]-b) > 1e-6):
        raise ValueError("fillet_source_endpoints_do_not_match_supports")
    if middle_arc:
        prior = selected[1].get("radius_binding") or {}
        if prior:
            if (prior.get("record_id") != binding.get("record_id") or
                    not math.isclose(float(selected[1].get("radius", 0.)), radius,
                                     rel_tol=1e-8, abs_tol=1e-8)):
                raise ValueError("fillet_reinsertion_requires_same_bound_radius")
        elif (binding.get("unique_unbound_arc_source_claim") is not True or
              selected[1].get("radius_constructed") or
              selected[1].get("radius_binding_status") not in
              {None, "not_requested", "unresolved_fixed_radius_fit_failed"}):
            raise ValueError("fillet_unbound_arc_requires_unique_source_claim")
    elif len(selected) == 3:
        connector = selected[1]
        length = math.dist(connector["start"], connector["end"])
        if (connector.get("radius_binding") or connector.get("dimension_bound")
                or connector.get("constraint_ids") or length <= 1e-8
                or length > 2*radius or min(ul, vl) <= 1.5*length):
            raise ValueError("fillet_connector_is_not_a_short_unbound_corner")
    u, v = u/ul, v/vl
    turn = math.atan2(float(u[0]*v[1]-u[1]*v[0]), float(u@v))
    if not math.radians(5) <= abs(turn) <= math.radians(160):
        raise ValueError(failure)
    try:
        lengths = np.linalg.solve(np.column_stack([u, v]), b-a)
    except np.linalg.LinAlgError as error:
        raise ValueError(failure) from error
    distance = radius*math.tan(abs(turn)/2)
    retained = lengths-distance
    # Both original outer endpoints and LINE directions stay fixed. A
    # tangent may redistribute one unbound short connector, but never reach
    # outside its finite local domain or consume an outer support LINE.
    if not np.isfinite(retained).all() or min(retained) <= max(1e-8, tolerance):
        raise ValueError("fillet_contact_outside_finite_line_support")
    first, last = a+retained[0]*u, b-retained[1]*v
    redistribution = []
    for side, contact, original_length, remaining in (
            ("before", first, ul, retained[0]), ("after", last, vl, retained[1])):
        extension = float(remaining-original_length)
        if extension <= 1e-7:
            continue
        if len(selected) != 3:
            raise ValueError("fillet_contact_outside_finite_line_support")
        c0, c1 = np.asarray(selected[1]["start"], float), np.asarray(selected[1]["end"], float)
        middle_chord = float(np.linalg.norm(c1-c0))
        gap = float(_primitive_distance(contact[None, :], selected[1])[0])
        if middle_arc:
            # Reuse only the old radius arc's finite footprint.  Its current
            # tangent is not trusted; the source ring below must also support
            # the newly constructed contact.
            source_gap = float(_source_polyline_distance(contact[None, :], points)[0])
            if (extension > middle_chord or gap > max(2*tolerance, 2.)
                    or source_gap > tolerance):
                raise ValueError("fillet_contact_outside_replaced_radius_arc_domain")
            redistribution.append({"side": side, "extension_px": extension,
                "replaced_arc_gap_px": gap, "source_path_gap_px": source_gap,
                "replaced_arc_chord_px": middle_chord,
                "source_tolerance_px": float(tolerance)})
        else:
            connector_vector = c1-c0
            fraction = float((contact-c0) @ connector_vector)/(middle_chord**2)
            if (extension > middle_chord or not -1e-7 <= fraction <= 1.+1e-7
                    or gap > tolerance):
                raise ValueError("fillet_contact_outside_finite_line_support_or_short_connector_domain")
            redistribution.append({"side": side, "extension_px": extension,
                "connector_projection_fraction": fraction, "finite_connector_gap_px": gap,
                "connector_length_px": middle_chord, "source_tolerance_px": float(tolerance)})
    center = first+np.array([-u[1], u[0]])*math.copysign(radius, turn)
    candidate = [{"type": "LINE", "start": a.tolist(), "end": first.tolist()},
                 {"type": "ARC", "start": first.tolist(), "end": last.tolist(),
                  "center": center.tolist(), "radius": float(radius), "clockwise": turn < 0},
                 {"type": "LINE", "start": last.tolist(), "end": b.tolist()}]
    distances = np.asarray([_primitive_distance(points, e) for e in candidate])
    assignments = np.argmin(distances, axis=0)
    if (np.count_nonzero(assignments == 1) < 3 or np.any(np.diff(assignments) < 0)
            or float(distances.min(axis=0).max()) > tolerance
            or float(_primitive_distance(target[None, :], candidate[1])[0]) > max(1., tolerance)):
        raise ValueError(failure)
    if len(selected) == 3 and not middle_arc:
        # A named short diagonal can be a real chamfer. Its removal needs
        # observed curvature beyond raster uncertainty, not merely a nearby R.
        corner_source = points[assignments == 1]
        bend = float(_primitive_distance(corner_source, selected[1]).max())
        if bend <= max(.5, tolerance*.25):
            raise ValueError("fillet_connector_has_no_resolved_source_curvature")
    sampled = _sample_entities(candidate, max_step_px=max(.25, min(1., tolerance/2)))[0]
    forward = float(cKDTree(sampled).query(points, workers=1)[0].max())
    reverse = float(_source_polyline_distance(sampled, points).max())
    if max(forward, reverse) > max(tolerance*1.5, 1.):
        raise ValueError(failure)
    for index, entity in enumerate(candidate):
        assigned = distances[index, assignments == index]
        entity["fit_error_px"] = float(assigned.max()) if len(assigned) else 0.
    middle_source = ({key: copy.deepcopy(selected[1][key])
                      for key in ("type", "start", "end", "center", "radius", "clockwise")
                      if key in selected[1]} if len(selected) == 3 else None)
    candidate[1].update(radius_binding=copy.deepcopy(binding),
        radius_annotation_evidence=copy.deepcopy(binding), radius_binding_status="applied",
        fillet_construction=("existing_line_arc_line_tangent_reinsertion" if middle_arc
                             else "existing_finite_line_supports"),
        source_refinement={"outer_endpoints_fixed": True, "support_directions_fixed": True,
                           "support_types_fixed": True, "radius_exact": True,
                           "source_path_and_arrow_verified": True,
                           "ground_truth_used": False,
                           "original_finite_line_supports_px": [
                               [selected[0]["start"], selected[0]["end"]],
                               [selected[-1]["start"], selected[-1]["end"]]],
                           "original_middle_source_geometry": middle_source,
                           "constructed_contact_points_px": [first.tolist(), last.tolist()],
                           "maximum_source_error_px": float(distances.min(axis=0).max()),
                           "maximum_reverse_error_px": reverse})
    if redistribution:
        if middle_arc:
            candidate[1]["source_refinement"]["replaced_arc_domain_redistribution"] = redistribution
        else:
            candidate[1]["fillet_construction"] = "existing_lines_short_connector_redistribution"
            candidate[1]["source_refinement"]["connector_domain_redistribution"] = redistribution
    # A later radius fillet can extend an already restored angular LINE into
    # the replaced ARC domain. Preserve its source-stroke witness only when
    # the replacement retains the entire old finite, directed LINE. Fresh
    # angle binding still rechecks the original stroke and both arrowheads.
    for old_line, new_line in ((selected[0], candidate[0]),
                               (selected[-1], candidate[-1])):
        witness = old_line.get("angle_support_evidence")
        if (not isinstance(witness, dict) or
                witness.get("method") != "verified_two_radius_joint_line_topology_restore" or
                witness.get("ground_truth_used") is not False or
                not isinstance(witness.get("record_id"), str)):
            continue
        old_segment = np.asarray([old_line["start"], old_line["end"]], float)
        new_segment = np.asarray([new_line["start"], new_line["end"]], float)
        old_vector, new_vector = np.diff(old_segment, axis=0)[0], np.diff(new_segment, axis=0)[0]
        old_length, new_length = float(np.linalg.norm(old_vector)), float(np.linalg.norm(new_vector))
        if min(old_length, new_length) <= tolerance:
            continue
        direction = new_vector/new_length
        if float(old_vector @ direction)/old_length < 1-1e-10:
            continue
        offsets = old_segment-new_segment[0]
        projections = offsets @ direction
        normal_gap = np.abs(offsets[:, 0]*direction[1]-offsets[:, 1]*direction[0])
        if (np.max(normal_gap) > 1e-4 or np.min(projections) < -1e-4 or
                np.max(projections) > new_length+1e-4):
            continue
        new_line["angle_support_evidence"] = copy.deepcopy(witness)
    return candidate


def _annotated_line_fillet(points, radius, tolerance, binding):
    """Historical entry point, now supporting LINE or ARC on either side."""
    try:
        return _annotated_straight_fillet(points, radius, tolerance, binding)
    except ValueError as error:
        if str(error) != "source_does_not_support_exact_annotated_fillet":
            raise
        try:
            return _annotated_curved_fillet(points, radius, tolerance, binding)
        except ValueError as curved_error:
            if str(curved_error) != "source_does_not_support_exact_annotated_fillet":
                raise
            return _refined_straight_fillet(points, radius, tolerance, binding)


def _replacement_chain(support, action, tolerance, selected_entities, *, force_arc=False,
                       annotated_radius_px=None, radius_binding=None, radius_targets=None, candidate_scorer=None,
                       angle_evidence=None):
    if action == "restore_annotated_line_support":
        from .annotation_line_support import restore_annotated_joint_line_support, restore_annotated_line_support
        if len(selected_entities) == 2:
            replacements = restore_annotated_joint_line_support(
                support, angle_evidence, tolerance, radius_targets=radius_targets,
                fixed_radius_fitter=_fixed_radius_arc)
        else:
            replacements = restore_annotated_line_support(
                support, angle_evidence, tolerance, radius_targets=radius_targets,
                fixed_radius_fitter=_fixed_radius_arc)
    elif action == "split_chain_at_source_features":
        if radius_targets and len(radius_targets)>=2:
            from .annotated_arc_resegmentation import resegment_annotated_arcs
            partition=resegment_annotated_arcs(support,radius_targets,tolerance,source_entity_count=len(selected_entities),
                                              candidate_scorer=candidate_scorer)
            replacements=partition["entities"]
            for entity in replacements:
                entity["source_partition_evidence"]={key:value for key,value in partition.items() if key!="entities"}
        else:
            replacements = _split_source_chain(support, tolerance)
    elif action == "insert_annotated_fillet":
        if len(selected_entities) in (2, 3) and (all(e.get("type") == "LINE" for e in selected_entities)
                or [e.get("type") for e in selected_entities] == ["LINE", "ARC", "LINE"]):
            replacements = _annotated_existing_line_fillet(
                support, annotated_radius_px, tolerance, radius_binding, selected_entities)
        else:
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


def _verified_partition_targets(graph, inventory, entity_ids, design_to_source, support=None):
    targets=[];seen=set()
    for row in graph.get("annotation_support",[]):
        if (row.get("kind")!="radius" or row.get("arrowhead_verified") is not True or
                row.get("candidate_entity_id") not in entity_ids or row.get("record_id") in seen):
            continue
        _,radius,binding=_radius_edit_evidence(graph,{"action":"refit_chain_as_annotated_arc",
                                                     "record_id":row["record_id"]},inventory,entity_ids,design_to_source)
        if binding and binding.get("arrowhead_verified") is True and binding.get("target_source_px") is not None:
            if support is not None and min(math.dist(binding["target_source_px"],support[index]) for index in (0,-1))<=1e-6:
                # A target exactly at the open chain boundary does not locate
                # an interior arc: its feature may start in the neighboring
                # chain. Keep that record unresolved in the global inventory,
                # rather than inventing a zero-length final radius segment.
                continue
            targets.append({"record_id":row["record_id"],"radius_px":radius,"nominal":binding["nominal"],
                            "source_arrow_verified":True,"target_source_px":binding["target_source_px"],"binding":binding})
            seen.add(row["record_id"])
    return targets


def _apply_one(graph, baseline, operation, inventory, *, candidate_scorer=None):
    action = operation.get("action")
    if action not in EDIT_ACTIONS:
        raise ValueError("unsupported_topology_edit_action")
    requested_entity_ids = operation.get("entity_ids")
    entity_ids = _complete_fillet_support_scope(graph, inventory, operation)
    minimum = _edit_minimum(action)
    indices = _ordered_indices(graph, entity_ids, minimum=minimum)
    if action == "refit_entity_as_line" and len(indices) != 1:
        raise ValueError("single_entity_refit_requires_one_entity")
    if action == "restore_annotated_line_support" and len(indices) not in (1, 2):
        raise ValueError("angle_line_restore_requires_one_or_two_entities")
    angle_evidence = None
    if action == "restore_annotated_line_support":
        from .annotation_line_support import angle_edit_evidence
        angle_evidence = angle_edit_evidence(graph, inventory, operation)
    grid = float(graph.get("source_grid_pitch_px") or 1.)
    if not math.isfinite(grid) or grid <= 0:
        raise ValueError("invalid_source_grid")
    design_to_source, source_to_design, orientation_det = _affines(graph, baseline)
    force_arc, annotated_radius_px, radius_binding = _radius_edit_evidence(
        graph, operation, inventory, entity_ids, design_to_source)
    source_entities = _base_source_entities(graph, design_to_source, orientation_det)
    selected_entities = [source_entities[index] for index in indices]
    if action == "insert_annotated_fillet" and len(indices) == 3:
        # Source-coordinate conversion intentionally carries only geometry and
        # radius provenance. Check other formal connector semantics on the
        # original graph before a short diagonal can be removed.
        connector = graph["entities"][indices[1]]
        if (all(e.get("type") == "LINE" for e in selected_entities)
                and (connector.get("dimension_bound") or connector.get("constraint_ids"))):
            raise ValueError("fillet_connector_is_not_a_short_unbound_corner")
    chain_samples, _, _ = _sample_entities(selected_entities, max_step_px=max(.35, min(1., grid / 2)))
    raw = _source_ring(baseline)
    ring = _closed_ring(raw) if raw is not None else _base_source_ring(graph, design_to_source, grid)
    start = np.asarray(selected_entities[0]["start"], float)
    end = np.asarray(selected_entities[-1]["end"], float)
    support, localization_mismatch = _ring_path(ring, start, end, chain_samples)
    tolerance = max(.25, float(graph.get("proposal_tolerance_px") or grid))
    radius_targets=(_verified_partition_targets(graph,inventory,entity_ids,design_to_source,support)
                    if action in {"split_chain_at_source_features", "restore_annotated_line_support"} else None)
    replacements = _replacement_chain(support, action, tolerance, selected_entities, force_arc=force_arc,
                                      annotated_radius_px=annotated_radius_px, radius_binding=radius_binding,
                                      radius_targets=radius_targets,candidate_scorer=candidate_scorer,
                                      angle_evidence=angle_evidence)
    from .annotation_line_support import protect_annotated_straight_supports
    protect_annotated_straight_supports(graph, entity_ids, selected_entities, replacements, tolerance)

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
        "requested_entity_ids": list(requested_entity_ids),
        "support_scope_expanded": list(entity_ids) != list(requested_entity_ids),
        "support_scope_trimmed": bool(set(requested_entity_ids) - set(entity_ids)),
        "record_id": operation.get("record_id"),
        "evidence_tags": list(operation.get("evidence_tags") or []),
        "removed_entity_count": len(indices), "replacement_count": len(replacements),
        "replacement_type": replacements[0]["type"] if len(replacements) == 1 else "LINE_ARC_CHAIN",
        "replacement_types": [row["type"] for row in replacements],
        "net_entity_reduction": len(indices) - len(replacements),
        "replacement_fit_error_px": max(float(row.get("fit_error_px", 0.)) for row in replacements),
        "annotation_guided": bool(angle_evidence) or bool(radius_binding) or bool(radius_targets and len(radius_targets)>=2),
        "resegmentation_applied": (action == "restore_annotated_line_support" or
                                   any(bool(row.get("source_partition_evidence")) for row in replacements)),
        "angle_support_record_ids": [angle_evidence["record_id"]] if angle_evidence else [],
        "angle_numeric_binding_applied": False,
        "bound_record_ids": sorted({row["radius_binding"]["record_id"] for row in replacements
                                    if row.get("radius_binding") and row["radius_binding"].get("record_id")}),
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


def _apply_combined(graph, baseline, operations, inventory, *, candidate_scorer=None):
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
        effective_ids = _complete_fillet_support_scope(graph, inventory, operation)
        indices = _ordered_indices(graph, effective_ids, minimum=minimum)
        # A chain crossing the cycle origin is still valid as a standalone
        # candidate.  Keeping combined edits non-wrapping makes the splice
        # auditable and prevents ambiguous remapping of the original IDs.
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("combined_edit_chain_wraps_cycle_origin")
        if occupied.intersection(indices):
            raise ValueError("combined_edit_chains_overlap")
        occupied.update(indices)
        individual, _, detail = _apply_one(graph, baseline, operation, inventory,candidate_scorer=candidate_scorer)
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
        "annotation_guided": any(row["annotation_guided"] for row in details),
        "resegmentation_applied": any(row["resegmentation_applied"] for row in details),
        "bound_record_ids": sorted({record for row in details for record in row["bound_record_ids"]}),
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
    inventory, _ = _annotation_inventory(
        gray, records, ring, grid,
        [*(graph.get("annotation_support") or []),
         *(graph.get("radius_source_segment_hypotheses") or [])])
    stroke = _StrokeEvidence(gray, records, grid)
    candidates, audit, accepted_operations = [], [], []

    def candidate_scorer(entities):
        sampled, _, _ = _sample_entities(entities,max_step_px=max(.5,min(2.,grid/2)))
        measured=stroke.summarize(sampled)
        return stroke_support_fraction(measured)

    def materialize(source_entities, quality, execution, candidate_id, strategy_description):
        sampled, _, _ = _sample_entities(source_entities, max_step_px=max(.5, min(2., grid / 2)))
        stroke_support = stroke.summarize(sampled)
        support, support_summary = _entity_annotation_support(source_entities, inventory, grid)
        support_summary["source_stroke_support"] = stroke_support
        before = graph.get("source_evidence", {}).get("proposal_stroke_support") or stroke_support
        support_gate = bool(
            stroke_support_fraction(stroke_support,against=before) >= stroke_support_fraction(before,against=stroke_support)-.035 and
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
                graph, baseline, operation, bundle.get("annotation_inventory", []),candidate_scorer=candidate_scorer)
            suffix = {"merge_chain_as_line": "line", "merge_chain_as_arc": "arc",
                      "merge_chain_best_fit": "best",
                      "refit_chain_as_annotated_arc": "annotated-arc",
                      "refit_entity_as_line": "retype-line",
                      "restore_annotated_line_support": "annotated-line",
                      "split_chain_at_source_features": "source-split",
                      "insert_annotated_fillet": "annotated-fillet"}[operation["action"]]
            candidate_id = f"cand-edit-{index:02d}-{suffix}"
            counts = materialize(
                source_entities, quality, execution, candidate_id,
                "Agent proposed one entity-ID edit; the local source-only geometry kernel executed and refit it.")
            row.update(status="accepted_as_candidate", candidate_id=candidate_id, execution=execution,
                       entity_counts=counts)
            accepted_operations.append((index, {**operation,"entity_ids":execution["entity_ids"]}))
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
                graph, baseline, [operation for _, operation in subset], bundle.get("annotation_inventory", []),
                candidate_scorer=candidate_scorer)
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

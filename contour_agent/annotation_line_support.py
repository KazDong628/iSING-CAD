"""Restore straight source intervals located by independently checked angle ink.

This module proposes topology only.  It never takes a fitted slope as a
dimension, and leaves the exact annotated angle to binding and solving.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .topology import _primitive_distance
from .vectorize import _arc, _line


def _segment(observation):
    line = observation.get("source_line") or {}
    points = np.asarray([line.get("start_px"), line.get("end_px")], dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("angle_source_line_invalid")
    if float(np.linalg.norm(points[1] - points[0])) <= 1e-6:
        raise ValueError("angle_source_line_degenerate")
    return points


def _verified_arc_joint(graph, requested, observation):
    """Locate a source-observed ARC/ARC joint crossed by an angled stroke.

    This only authorizes a topology *candidate*. Both existing curved sides
    must have separate directed radius arrows; their numerical values are
    checked again when the replacement is built and later solved.
    """
    entities = graph.get("entities") or []
    if len(requested) != 2 or len(entities) < 3:
        return None
    index = next((i for i, row in enumerate(entities) if row.get("id") == requested[0]), None)
    if index is None or entities[(index + 1) % len(entities)].get("id") != requested[1]:
        return None
    first, second = entities[index], entities[(index + 1) % len(entities)]
    if first.get("type") != second.get("type") or first.get("type") != "ARC":
        return None
    radius_owners = {row.get("candidate_entity_id") for row in graph.get("annotation_support", [])
                     if row.get("kind") == "radius" and row.get("status") == "candidate_supported"
                     and row.get("arrowhead_verified") is True}
    if not set(requested) <= radius_owners:
        return None
    nodes = {row.get("id"): row for row in graph.get("nodes", [])}
    joint = nodes.get(first.get("end_node"), {}).get("source_px")
    if first.get("end_node") != second.get("start_node") or joint is None:
        return None
    joint = np.asarray(joint, dtype=float)
    if joint.shape != (2,) or not np.isfinite(joint).all():
        return None
    segment = _segment(observation)
    direction = segment[1] - segment[0]
    extent = float(np.linalg.norm(direction))
    direction /= extent
    projection = float((joint - segment[0]) @ direction)
    gap = abs(float(np.linalg.det(np.stack([direction, joint - segment[0]]))))
    band = max(5., 2 * float(graph.get("proposal_tolerance_px") or 1.))
    if gap > band or projection < -band or projection > extent + band:
        return None
    return joint.tolist()


def angle_edit_evidence(graph, inventory, operation):
    """Resolve trusted local observations; operation-supplied coordinates are ignored."""
    record_id = operation.get("record_id")
    record = next((row for row in inventory if row.get("record_id") == record_id), {})
    nominal = record.get("nominal")
    if (record.get("kind") != "angle" or isinstance(nominal, bool)
            or not isinstance(nominal, (float, int)) or not math.isfinite(nominal)
            or not 0 < nominal < 90):
        raise ValueError("angle_edit_requires_valid_angle_record")
    requested = operation.get("entity_ids") or []
    if len(requested) not in (1, 2):
        raise ValueError("angle_line_restore_requires_one_entity")
    for original in graph.get("angle_source_observations", []):
        if (original.get("record_id") != record_id or original.get("verified") is not True
                or original.get("reference_axis") not in {"vertical", "horizontal"}
                or original.get("nominal") != nominal):
            continue
        matches = [row for row in original.get("target_candidates", [])
                   if row.get("entity_id") in requested]
        if not matches:
            continue
        _segment(original)
        evidence = copy.deepcopy(original)
        if len(requested) == 2:
            joint = _verified_arc_joint(graph, requested, original)
            if joint is None:
                continue
            evidence["joint_source_px"] = joint
            evidence["joint_radius_arrow_verified"] = True
        return evidence
    raise ValueError("angle_line_source_evidence_not_verified_for_entity")


def propose_annotation_line_edits(graph, inventory, *, limit=1):
    """Offer angle-supported ARC->LINE interval repairs before radius refits."""
    if limit <= 0:
        return []
    entities = {row.get("id"): row for row in graph.get("entities", [])}
    proposed = []
    seen = set()
    for observation in graph.get("angle_source_observations", []):
        if any(entities.get(row.get("entity_id"), {}).get("type") == "LINE"
               and row.get("whole_line_supported") is True
               for row in observation.get("target_candidates", [])):
            # A fully witnessed LINE already carries this angle. Its adjacent
            # arc can share a short tangent-like ink interval; that alone must
            # not create another straight object or consume the repair budget.
            continue
        for target in observation.get("target_candidates", []):
            entity_id = target.get("entity_id")
            entity = entities.get(entity_id, {})
            if entity.get("type") != "ARC":
                continue
            index = next((i for i, row in enumerate(graph.get("entities", [])) if row.get("id") == entity_id), None)
            scopes = [[entity_id]]
            if index is not None:
                ordered = graph["entities"]
                scopes = [[ordered[(index - 1) % len(ordered)]["id"], entity_id],
                          [entity_id, ordered[(index + 1) % len(ordered)]["id"]],
                          [entity_id]]
            for scope in scopes:
                operation = {"action": "restore_annotated_line_support", "entity_ids": scope,
                             "record_id": observation.get("record_id"),
                             "evidence_tags": ["annotation_target", "collinear_support", "source_boundary"]}
                try:
                    evidence = angle_edit_evidence(graph, inventory, operation)
                except (ValueError, TypeError):
                    continue
                key = (operation["record_id"], tuple(scope))
                if key in seen:
                    continue
                seen.add(key)
                span = float(np.linalg.norm(np.diff(_segment(evidence), axis=0)))
                proposed.append((span, len(scope), operation))
    return [row[2] for row in sorted(proposed, key=lambda row: (-row[0], -row[1], row[2]["entity_ids"]))[:limit]]


def _dense_path(points, tolerance):
    """Sample the unchanged source polyline, including every original vertex."""
    parts = []
    step = max(.5, tolerance / 2.)
    count_budget = 0
    for a, b in zip(points, points[1:]):
        count = max(1, int(math.ceil(float(np.linalg.norm(b-a)) / step)))
        count_budget += count
        if count_budget > 8192:
            raise ValueError("angle_line_source_sampling_budget_exhausted")
        parts.append(np.linspace(a, b, count + 1)[:-1])
    return np.vstack([*parts, points[-1:]])


def restore_annotated_joint_line_support(points, observation, tolerance, *, radius_targets,
                                         fixed_radius_fitter):
    """Test a finite LINE across an ARC/ARC joint using source data alone.

    Both radius arrows must belong to opposite curved tails. A tangent-like
    mask interval or an angle label alone is insufficient to replace an ARC.
    Every proposed tail is refitted to its own independently verified radius.
    """
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 7
            or not np.isfinite(points).all() or fixed_radius_fitter is None
            or len(radius_targets or []) != 2):
        raise ValueError("angle_line_joint_requires_two_verified_radius_tails")
    joint = np.asarray(observation.get("joint_source_px"), dtype=float)
    if joint.shape != (2,) or not np.isfinite(joint).all() or observation.get("joint_radius_arrow_verified") is not True:
        raise ValueError("angle_line_joint_source_evidence_missing")
    joint_index = int(np.argmin(np.linalg.norm(points - joint, axis=1)))
    if (joint_index < 3 or joint_index > len(points) - 4
            or float(np.linalg.norm(points[joint_index] - joint)) > tolerance):
        raise ValueError("angle_line_joint_not_on_source_path")
    located = []
    for target in radius_targets:
        if (target.get("source_arrow_verified") is not True or
                not isinstance(target.get("binding"), dict) or
                target["binding"].get("arrowhead_verified") is not True):
            raise ValueError("angle_line_joint_radius_arrow_unverified")
        tip = np.asarray(target.get("target_source_px"), dtype=float)
        if tip.shape != (2,) or not np.isfinite(tip).all():
            raise ValueError("angle_line_joint_radius_target_invalid")
        target_index = int(np.argmin(np.linalg.norm(points - tip, axis=1)))
        if target_index == joint_index:
            raise ValueError("angle_line_joint_radius_target_ambiguous")
        located.append((target_index, target, tip))
    located.sort(key=lambda row: row[0])
    if (located[0][0] >= joint_index or located[1][0] <= joint_index
            or located[0][1]["record_id"] == located[1][1]["record_id"]):
        raise ValueError("angle_line_joint_radius_targets_on_same_side")
    segment = _segment(observation)
    direction = segment[1] - segment[0]
    observed_length = float(np.linalg.norm(direction))
    direction /= observed_length
    normal = np.array([-direction[1], direction[0]])
    source_span = max((float(row.get("supported_span_px") or 0.)
                       for row in observation.get("target_candidates", [])), default=0.)
    window = max(16 * tolerance, 1.5 * min(observed_length, source_span or observed_length))
    arclength = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    left_indices = [i for i in range(2, joint_index)
                    if arclength[joint_index] - arclength[i] <= window]
    right_indices = [i for i in range(joint_index + 1, len(points) - 2)
                     if arclength[i] - arclength[joint_index] <= window]
    # Bound work independently of source polyline vertex density.
    def bounded(indices):
        if len(indices) <= 64:
            return indices
        return sorted({indices[i] for i in np.linspace(0, len(indices) - 1, 64).astype(int)})
    left_indices, right_indices = bounded(left_indices), bounded(right_indices)
    left, right = located
    left_arcs, right_arcs = {}, {}
    for index in left_indices:
        arc = fixed_radius_fitter(points[:index + 1], float(left[1]["radius_px"]), tolerance)
        if arc is not None and float(_primitive_distance(left[2][None, :], arc)[0]) <= 2.25 * tolerance:
            left_arcs[index] = arc
    for index in right_indices:
        arc = fixed_radius_fitter(points[index:], float(right[1]["radius_px"]), tolerance)
        if arc is not None and float(_primitive_distance(right[2][None, :], arc)[0]) <= 2.25 * tolerance:
            right_arcs[index] = arc
    minimum_span = max(8 * tolerance, 12.)
    band = max(5., 2 * tolerance)
    candidates = []
    for lo, first in left_arcs.items():
        for hi, last in right_arcs.items():
            chord = points[hi] - points[lo]
            span = float(np.linalg.norm(chord))
            if span < minimum_span or span > observed_length + 2 * band:
                continue
            if abs(float(chord @ direction / span)) < math.cos(math.radians(3.)):
                continue
            ends = points[[lo, hi]] - segment[0]
            if (max(np.abs(ends @ normal)) > band or
                    min(ends @ direction) < -band or max(ends @ direction) > observed_length + band):
                continue
            line = _line(points[lo:hi + 1], tolerance)
            if line is None or float(line["fit_error_px"]) > min(tolerance, 1.5 * float(observation.get("source_grid_pitch_px") or tolerance)):
                continue
            # A radius arrow landing on the putative LINE cannot be reassigned
            # to a remote curved tail just because the same radius can be fit.
            if any(float(_primitive_distance(tip[None, :], line)[0]) <= 2.25 * tolerance
                   for _, _, tip in located):
                continue
            ranking = (max(float(first["fit_error_px"]), float(last["fit_error_px"])),
                       float(first["fit_error_px"]) + float(line["fit_error_px"]) + float(last["fit_error_px"]),
                       max(np.abs(ends @ normal)), -span)
            candidates.append((ranking, first, line, last))
    if not candidates:
        raise ValueError("angle_line_joint_has_no_two_radius_source_fit")
    _, first, line, last = min(candidates, key=lambda row: row[0])
    for arc, (_, target, _) in ((first, left), (last, right)):
        arc["radius_binding"] = copy.deepcopy(target["binding"])
        # The fixed-radius fitter just matched this original source interval
        # and its independently directed arrow. Record that construction for
        # exact geometric preservation; fresh binding still decides whether
        # the OCR label owns this ARC on the new graph.
        arc["radius_annotation_evidence"] = copy.deepcopy(target["binding"])
        arc["parameter_source"] = "multimodal_annotation_guided_arc_refit"
        arc["radius_binding_status"] = "applied"
    line["angle_support_evidence"] = {
        "record_id": observation["record_id"], "nominal": observation["nominal"],
        "reference_axis": observation["reference_axis"],
        "method": "verified_two_radius_joint_line_topology_restore",
        "source_line": copy.deepcopy(observation["source_line"]),
        "source_interval_endpoints_px": [line["start"], line["end"]],
        "numeric_angle_bound": False, "requires_angle_binding_and_solve": True,
        "ground_truth_used": False}
    return [first, line, last]


def restore_annotated_line_support(points, observation, tolerance, *, radius_targets=None,
                                   fixed_radius_fitter=None):
    """Preserve any curved tails around a locally witnessed straight interval.

    At most three primitives replace one initial primitive. Endpoints are
    unchanged; every piece must pass the original primitive fit tolerance.
    The caller still performs whole-contour, stroke and dimensional checks.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("angle_line_source_path_invalid")
    segment = _segment(observation)
    direction = segment[1] - segment[0]
    observed_length = float(np.linalg.norm(direction))
    direction /= observed_length
    minimum_span = max(8 * tolerance, 4.)
    if observed_length < minimum_span:
        raise ValueError("angle_line_source_span_too_short")
    dense = _dense_path(points, tolerance)
    offsets = dense-segment[0]
    projection = offsets @ direction
    normal_gap = np.abs(offsets[:, 0]*direction[1]-offsets[:, 1]*direction[0])
    valid = ((normal_gap <= tolerance) & (projection >= -tolerance)
             & (projection <= observed_length+tolerance))
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        raise ValueError("angle_line_source_interval_not_supported")
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
    # The observed segment may stop inside a long straight boundary. Try a
    # bounded neighborhood of each end, plus the original entity endpoints.
    windows = []
    for group in groups:
        if len(group) < 2:
            continue
        first, last = int(group[0]), int(group[-1])
        if np.linalg.norm(dense[last]-dense[first]) < minimum_span:
            continue
        windows.append((first, last))
    candidates = []
    for first, last in windows[:4]:
        stride = max(1, int(math.ceil((last-first)/40)))
        starts = {0, *[max(0, min(len(dense)-1, first+k*stride)) for k in range(-3, 4)]}
        ends = {len(dense)-1, *[max(0, min(len(dense)-1, last+k*stride)) for k in range(-3, 4)]}
        for lo in sorted(starts):
            for hi in sorted(ends):
                if hi <= lo or np.linalg.norm(dense[hi]-dense[lo]) < minimum_span:
                    continue
                line = _line(dense[lo:hi+1], tolerance)
                if line is None:
                    continue
                vector = np.asarray(line["end"])-line["start"]
                cosine = abs(float(vector @ direction / np.linalg.norm(vector)))
                # This checks association to the observed ink, not agreement
                # with the nominal angle, which is a later solver constraint.
                if cosine < math.cos(math.radians(3.)):
                    continue
                if hi < first or lo > last or min(hi, last)-max(lo, first) < .5*(last-first):
                    continue
                pieces = []
                failed = False
                for a, b in ((0, lo), (hi, len(dense)-1)):
                    if a == b:
                        pieces.append(None)
                        continue
                    support = dense[a:b+1]
                    tail = _arc(support, tolerance) or _line(support, tolerance)
                    if tail is None:
                        failed = True
                        break
                    for target in radius_targets or []:
                        point = np.asarray(target["target_source_px"], dtype=float)[None, :]
                        if (tail["type"] == "ARC" and fixed_radius_fitter is not None
                                and float(_primitive_distance(point, tail)[0]) <= 2.25*tolerance):
                            fixed = fixed_radius_fitter(support, target["radius_px"], tolerance)
                            if fixed is not None:
                                tail = fixed
                                tail["radius_binding"] = copy.deepcopy(target["binding"])
                                tail["radius_binding_status"] = "applied"
                    pieces.append(tail)
                if failed:
                    continue
                replacements = [row for row in (pieces[0], line, pieces[1]) if row is not None]
                # A directed R inside the proposed LINE remains unresolved;
                # it cannot be silently erased by the angle-type repair.
                for target in radius_targets or []:
                    px = np.asarray(target["target_source_px"], dtype=float)[None, :]
                    line_gap = float(_primitive_distance(px, line)[0])
                    curved_gap = min((float(_primitive_distance(px, row)[0]) for row in replacements
                                      if row["type"] == "ARC"), default=math.inf)
                    if line_gap <= 2.25*tolerance and curved_gap > 2.25*tolerance:
                        failed = True
                        break
                if failed:
                    continue
                line["angle_support_evidence"] = {
                    "record_id": observation["record_id"], "nominal": observation["nominal"],
                    "reference_axis": observation["reference_axis"],
                    "method": "locally_verified_angle_ink_interval_topology_restore",
                    "source_line": copy.deepcopy(observation["source_line"]),
                    "source_interval_endpoints_px": [line["start"], line["end"]],
                    "numeric_angle_bound": False, "requires_angle_binding_and_solve": True,
                    "ground_truth_used": False}
                # Prefer the largest supported physical straight extent, then
                # fewer tails. The reference DXF never enters this ordering.
                rank = (-float(np.linalg.norm(vector)), len(replacements),
                        max(float(row.get("fit_error_px", 0.)) for row in replacements))
                candidates.append((rank, replacements))
    if not candidates:
        raise ValueError("angle_line_interval_cannot_preserve_source_and_radius_support")
    return min(candidates, key=lambda row: row[0])[1]


def protect_annotated_straight_supports(graph, selected_ids, selected_source_entities, replacements, tolerance):
    """Later merge/fillet edits cannot consume an independently witnessed LINE."""
    original = dict(zip(selected_ids, selected_source_entities))
    for observation in graph.get("angle_source_observations", []):
        if observation.get("verified") is not True:
            continue
        for target in observation.get("target_candidates", []):
            line = original.get(target.get("entity_id"), {})
            if line.get("type") != "LINE" or target.get("whole_line_supported") is not True:
                continue
            a, b = np.asarray(line["start"]), np.asarray(line["end"])
            direction = b-a
            length = float(np.linalg.norm(direction))
            if length <= 1e-6:
                continue
            direction /= length
            preserved = False
            for replacement in replacements:
                if replacement["type"] != "LINE":
                    continue
                pair = np.asarray([replacement["start"], replacement["end"]], dtype=float)
                vector = pair[1]-pair[0]
                span = float(np.linalg.norm(vector))
                if span <= 1e-6 or abs(float(vector @ direction / span)) < math.cos(math.radians(3.)):
                    continue
                offsets = pair-a
                normal_gap = np.abs(offsets[:, 0]*direction[1]-offsets[:, 1]*direction[0])
                projection = np.sort(offsets @ direction)
                overlap = max(0., min(length, projection[1])-max(0., projection[0]))
                if max(normal_gap) <= tolerance and overlap >= min(length/2, max(8*tolerance, 4.)):
                    preserved = True
                    break
            if not preserved:
                raise ValueError("edit_would_discard_angle_supported_line")

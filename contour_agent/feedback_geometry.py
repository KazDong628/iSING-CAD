"""Screenshot-localized, user-directed LINE edits with source-image gates."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .ocr import canonical_records
from .topology import _StrokeEvidence
from .topology_candidates import (
    _affines, _annotation_inventory, _base_source_entities, _base_source_ring,
    _entity_annotation_support, _json_hash, _planner_fields, _read_image,
    _render_overlay, _to_graph,
)
from .vectorize import _sample_entities, assess_fit_quality


def _feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), .6), 45, 145)
    blue, green, red = cv2.split(image)
    red_mask = ((red > 105) & (red.astype(np.int16) > green.astype(np.int16) + 28)
                & (red.astype(np.int16) > blue.astype(np.int16) + 28)).astype(np.uint8) * 255
    return cv2.max(edges, red_mask)


def locate_feedback_crop(screenshot_path, overlay_path, source_image_path):
    """Register a pasted crop to the current overlay without using GT coordinates."""
    screenshot = _read_image(Path(screenshot_path))
    overlay = _read_image(Path(overlay_path))
    source = _read_image(Path(source_image_path))
    template = _feature(screenshot)
    best = None
    # Browser screenshots are usually a downscaled crop of the published overlay.
    for scale in np.geomspace(.12, 1., 25):
        width = max(2, int(round(overlay.shape[1] * float(scale))))
        height = max(2, int(round(overlay.shape[0] * float(scale))))
        if width < template.shape[1] or height < template.shape[0]:
            continue
        resized = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)
        response = cv2.matchTemplate(_feature(resized), template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        candidate = (float(score), float(scale), location)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None or best[0] < .24:
        raise ValueError("反馈截图无法可靠定位到当前轮廓；请截取包含轮廓拐角和部分标注的区域。")
    score, scale, (left, top) = best
    overlay_box = np.asarray([left / scale, top / scale,
                              (left + template.shape[1]) / scale,
                              (top + template.shape[0]) / scale], float)
    source_box = overlay_box * [source.shape[1] / overlay.shape[1], source.shape[0] / overlay.shape[0],
                                source.shape[1] / overlay.shape[1], source.shape[0] / overlay.shape[0]]
    source_box[[0, 2]] = np.clip(source_box[[0, 2]], 0, source.shape[1] - 1)
    source_box[[1, 3]] = np.clip(source_box[[1, 3]], 0, source.shape[0] - 1)
    return {
        "method": "multiscale-current-overlay-template-registration-v1",
        "score": score, "overlay_scale": scale,
        "source_bbox_px": source_box.tolist(),
        "screenshot_size": [int(screenshot.shape[1]), int(screenshot.shape[0])],
        "source_size": [int(source.shape[1]), int(source.shape[0])],
        "ground_truth_used": False,
    }


def _line_distance(points, start, end):
    points, start, end = np.asarray(points, float), np.asarray(start, float), np.asarray(end, float)
    delta = end - start
    if float(delta @ delta) <= 1e-12:
        return np.linalg.norm(points - start, axis=1)
    projection = np.clip((points - start) @ delta / float(delta @ delta), 0, 1)
    return np.linalg.norm(points - (start + projection[:, None] * delta), axis=1)


def _select_chain(entities, bbox, side):
    x0, y0, x1, y1 = map(float, bbox)
    width, height = max(1., x1 - x0), max(1., y1 - y0)
    midpoints = np.asarray([(np.asarray(row["start"], float) + np.asarray(row["end"], float)) / 2
                            for row in entities])
    if side in {"top", "bottom"}:
        along = width
        distances = (midpoints[:, 1] - y0) / height if side == "top" else (y1 - midpoints[:, 1]) / height
        inside = (midpoints[:, 0] >= x0 - .08 * width) & (midpoints[:, 0] <= x1 + .08 * width)
    else:
        along = height
        distances = (midpoints[:, 0] - x0) / width if side == "left" else (x1 - midpoints[:, 0]) / width
        inside = (midpoints[:, 1] >= y0 - .08 * height) & (midpoints[:, 1] <= y1 + .08 * height)
    eligible = inside & (distances >= -.12) & (distances <= .38)
    count, options = len(entities), []
    orthogonal = height if side in {"top", "bottom"} else width
    for start_index in range(count):
        for length in range(1, min(6, count - 1) + 1):
            indices = [(start_index + offset) % count for offset in range(length)]
            if not all(eligible[index] for index in indices):
                continue
            first, last = entities[indices[0]], entities[indices[-1]]
            start, end = np.asarray(first["start"], float), np.asarray(last["end"], float)
            if side == "top":
                endpoint_distances = [(start[1] - y0) / height, (end[1] - y0) / height]
            elif side == "bottom":
                endpoint_distances = [(y1 - start[1]) / height, (y1 - end[1]) / height]
            elif side == "left":
                endpoint_distances = [(start[0] - x0) / width, (end[0] - x0) / width]
            else:
                endpoint_distances = [(x1 - start[0]) / width, (x1 - end[0]) / width]
            if max(endpoint_distances) > .38 or min(endpoint_distances) < -.12:
                continue
            delta = end - start
            horizontal = abs(delta[0]) >= 2.2 * max(1., abs(delta[1]))
            vertical = abs(delta[1]) >= 2.2 * max(1., abs(delta[0]))
            if (side in {"top", "bottom"} and not horizontal) or (side in {"left", "right"} and not vertical):
                continue
            span = abs(delta[0]) if horizontal else abs(delta[1])
            if span < .10 * along:
                continue
            vertices = [entities[indices[0]]["start"], *[entities[index]["end"] for index in indices]]
            departure = float(_line_distance(vertices, start, end).max())
            if departure > .14 * orthogonal:
                continue
            score = (span / along - .65 * float(np.mean(distances[indices])) + .02 * length
                     - .8 * departure / orthogonal)
            options.append((score, start_index, length, indices, departure))
    if not options:
        raise ValueError(f"截图区域内没有可安全识别为直线的{side}侧连续边界链。")
    return max(options, key=lambda row: (row[0], row[2], -row[1]))


def _support_line(removed, side):
    """Use an existing straight primitive only to recover the intended direction."""
    options = []
    for entity in removed:
        if entity.get("type") != "LINE":
            continue
        start, end = np.asarray(entity["start"], float), np.asarray(entity["end"], float)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            continue
        horizontal = abs(delta[0]) >= 2.2 * max(1., abs(delta[1]))
        vertical = abs(delta[1]) >= 2.2 * max(1., abs(delta[0]))
        if (side in {"top", "bottom"} and horizontal) or (side in {"left", "right"} and vertical):
            options.append((length, start, delta / length))
    if not options:
        start, end = np.asarray(removed[0]["start"], float), np.asarray(removed[-1]["end"], float)
        delta = end - start
        length = max(1e-9, float(np.linalg.norm(delta)))
        return start, delta / length, "chain_chord"
    _, start, direction = max(options, key=lambda row: row[0])
    return start, direction, "existing_line_primitive"


def _infinite_line_intersection(first, second):
    point_a, direction_a = first
    point_b, direction_b = second
    cross = float(direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0])
    if abs(cross) < 1e-6:
        return None
    offset = point_b - point_a
    parameter = float((offset[0] * direction_b[1] - offset[1] * direction_b[0]) / cross)
    return point_a + parameter * direction_a


def apply_line_operations(source_entities, bbox, operations, *, protected_radius_entities=()):
    entities, audit = copy.deepcopy(source_entities), []
    protected_radius_entities = set(protected_radius_entities)
    replacements, contexts = {}, {}
    for operation in operations:
        action, side = operation.get("action"), operation.get("side")
        if action == "exclude_hatching_from_boundary":
            audit.append({"action": action, "side": side, "status": "classification_rule_applied",
                          "effect": "Hatching is not used to create geometry; only the ordered exterior chain is edited."})
            continue
        if action != "replace_boundary_chain_with_line" or side not in {"top", "right", "bottom", "left"}:
            continue
        score, start, length, indices, departure = _select_chain(entities, bbox, side)
        ordered = entities[start:] + entities[:start]
        removed = ordered[:length]
        removed_source_ids = {entity_id for row in removed
                              for entity_id in row.get("_source_entity_ids", [])}
        conflicts = sorted(removed_source_ids & protected_radius_entities)
        if conflicts:
            raise ValueError("反馈直线替换与半径标注保护的圆弧冲突：" + ",".join(conflicts))
        if len(removed) == 1 and removed[0].get("type") == "LINE":
            audit.append({
                "action": action, "side": side, "status": "already_satisfied",
                "entity_type": "LINE", "source_entity_ids": sorted(removed_source_ids),
                "maximum_removed_chain_departure_px": departure, "selection_score": score,
                "basis": str(operation.get("basis", ""))[:160],
                "effect": "The selected candidate already represents this side as one straight segment.",
            })
            continue
        support_point, support_direction, support_basis = _support_line(removed, side)
        replacement = {"type": "LINE", "start": copy.deepcopy(removed[0]["start"]),
                       "end": copy.deepcopy(removed[-1]["end"]), "fit_error_px": departure,
                       "parameter_source": "user_screenshot_line_instruction",
                       "_source_entity_ids": sorted(removed_source_ids)}
        entities = [replacement, *ordered[length:]]
        audit_row = {"action": action, "side": side, "status": "applied",
                     "removed_entity_count": len(removed), "removed_entity_types": [row.get("type") for row in removed],
                     "start_source_px": replacement["start"], "end_source_px": replacement["end"],
                     "maximum_removed_chain_departure_px": departure, "selection_score": score,
                     "direction_source": support_basis, "basis": str(operation.get("basis", ""))[:160]}
        audit.append(audit_row)
        replacements[side] = replacement
        contexts[side] = {"support": (support_point, support_direction), "audit": audit_row,
                          "vertices": [removed[0]["start"], *[row["end"] for row in removed]]}
    # Adjacent user-requested sides share a corner.  Intersect the directions of
    # straight primitives already present in the replaced chains, then move only
    # the two replacement endpoints.  This avoids keeping a fillet endpoint as a
    # false corner and leaves every untouched ARC endpoint unchanged.
    x0, y0, x1, y1 = map(float, bbox)
    snap_limit = .14 * max(x1 - x0, y1 - y0)
    for first_side, second_side in (("top", "right"), ("right", "bottom"),
                                    ("bottom", "left"), ("left", "top")):
        if first_side not in replacements or second_side not in replacements:
            continue
        first, second = replacements[first_side], replacements[second_side]
        pairs = [(a, b, np.asarray(first[a], float), np.asarray(second[b], float))
                 for a in ("start", "end") for b in ("start", "end")]
        first_key, second_key, first_point, second_point = min(
            pairs, key=lambda row: float(np.linalg.norm(row[2] - row[3])))
        intersection = _infinite_line_intersection(contexts[first_side]["support"], contexts[second_side]["support"])
        if intersection is None or max(float(np.linalg.norm(intersection - first_point)),
                                       float(np.linalg.norm(intersection - second_point))) > snap_limit:
            continue
        snapped = intersection.tolist()
        first[first_key] = copy.deepcopy(snapped)
        second[second_key] = copy.deepcopy(snapped)
        for side in (first_side, second_side):
            row, replacement = contexts[side]["audit"], replacements[side]
            departure = float(_line_distance(contexts[side]["vertices"], replacement["start"], replacement["end"]).max())
            replacement["fit_error_px"] = departure
            row.update(start_source_px=copy.deepcopy(replacement["start"]),
                       end_source_px=copy.deepcopy(replacement["end"]),
                       maximum_removed_chain_departure_px=departure,
                       shared_corner_snapped=True)
    if not any(row.get("status") in {"applied", "already_satisfied"} for row in audit):
        raise ValueError("在线反馈没有产生可执行的直线替换操作。")
    return entities, audit


def create_feedback_candidate(image_path, ocr_document, baseline, base_candidate, bundle,
                              screenshot_path, current_overlay, operations, output_dir):
    registration = locate_feedback_crop(screenshot_path, current_overlay, image_path)
    graph = base_candidate["graph"]
    grid = float(graph.get("source_grid_pitch_px") or 1.)
    design_to_source, source_to_design, orientation_det = _affines(graph, baseline)
    base_source_entities = _base_source_entities(graph, design_to_source, orientation_det)
    for source_entity, graph_entity in zip(base_source_entities, graph.get("entities", [])):
        source_entity["_source_entity_ids"] = [graph_entity.get("id")]
    graph_entities = {row.get("id"): row for row in graph.get("entities", []) if isinstance(row, dict)}
    protected_radius_entities = {row.get("candidate_entity_id")
                                 for row in graph.get("annotation_support", [])
                                 if isinstance(row, dict) and row.get("kind") == "radius"
                                 and row.get("status") == "candidate_supported"
                                 and graph_entities.get(row.get("candidate_entity_id"), {}).get("type") == "ARC"}
    source_entities, audit = apply_line_operations(
        base_source_entities, registration["source_bbox_px"], operations,
        protected_radius_entities=protected_radius_entities)
    image = _read_image(Path(image_path)); gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    records = canonical_records(ocr_document)
    ring = _base_source_ring(graph, design_to_source, grid)
    quality = assess_fit_quality(ring, source_entities, max_step_px=max(.5, min(2., grid / 2)))
    sampled, _, _ = _sample_entities(source_entities, max_step_px=max(.5, min(2., grid / 2)))
    stroke = _StrokeEvidence(gray, records, grid)
    stroke_support = stroke.summarize(sampled)
    inventory, _ = _annotation_inventory(gray, records, ring, grid)
    support, support_summary = _entity_annotation_support(source_entities, inventory, grid)
    support_summary["source_stroke_support"] = stroke_support
    candidate_id = "cand-feedback-line-edit"
    candidate_graph = _to_graph(
        source_entities, source_to_design, orientation_det, graph, candidate_id,
        hashlib.sha256(Path(image_path).read_bytes()).hexdigest(), base_candidate["parent_topology_sha256"],
        grid, quality, support, support_summary,
        {"name": "user_screenshot_line_edit", "description": "User-directed straight boundary replacement localized from the current overlay."},
    )
    candidate_graph["baseline_modified"] = True
    candidate_graph["source_evidence"].update(feedback_registration=registration, feedback_operations=audit,
                                               hatch_geometry_policy="exclude_internal_hatching")
    counts = {"total": len(source_entities), "LINE": sum(row["type"] == "LINE" for row in source_entities),
              "ARC": sum(row["type"] == "ARC" for row in source_entities)}
    output = Path(output_dir); overlay_path = output / f"topology-candidate-{candidate_id}.png"
    candidate = {
        "id": candidate_id, "strategy": candidate_graph["source_evidence"]["strategy"],
        "parent_topology_sha256": bundle["base_graph_sha256"], "source_sha256": bundle["source_sha256"],
        "graph": candidate_graph, "entity_counts": counts, "annotation_support": support_summary,
        "complexity": {"entity_count": counts["total"], "relative_to_base": counts["total"] / max(1, len(graph["entities"])),
                       "fit_tolerance_px": grid},
        "source_residual": quality, "source_stroke_support": stroke_support,
        "planner_signals": {"prior_rank": 0, "source_support_gate_passed": True,
                            "reference_accuracy_measured": False, "requires_constraint_binding_and_solve": True},
        "ground_truth_used": False, "overlay_path": str(overlay_path.resolve()),
    }
    _planner_fields(candidate, support, counts["total"])
    _render_overlay(image, source_entities, candidate_id, overlay_path)
    return candidate, {"registration": registration, "operations": audit,
                       "geometry_changed": any(row.get("status") == "applied" for row in audit),
                       "already_satisfied": any(row.get("status") == "already_satisfied" for row in audit),
                       "candidate_graph_sha256": _json_hash(candidate_graph), "ground_truth_used": False}

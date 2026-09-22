"""Bounded source-only topology hypotheses for planner/evaluator agents.

The segmentation-derived topology is an observation, not a prescribed CAD
object count.  This module refits that same ordered material boundary at a
small number of declared complexity levels and measures each result against
source pixels and source annotations.  It never opens a reference DXF or a
training label and it does not select a candidate as geometrically correct.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .constraint_binding import _arrowhead_evidence, _leaders
from .ocr import canonical_records
from .topology import _StrokeEvidence, _primitive_distance, _relations
from .vectorize import _closed_ring, _sample_entities, assess_fit_quality, fit_polyline


_STRATEGIES = (
    ("annotation_conservative", 0.65, "Retain small source-supported transitions around annotation targets."),
    ("fidelity_first", 1.00, "Prefer boundary fidelity while allowing adjacent compatible primitives to merge."),
    ("balanced", 1.65, "Balance primitive count, source residual and annotation compatibility."),
    ("compact", 2.75, "Test a lower object-count explanation under an explicit source residual audit."),
    ("very_compact", 4.25, "Stress-test the simplest supported LINE/ARC explanation; never auto-accept it."),
)


def _json_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot read source image")
    if image.shape[0] * image.shape[1] > 80_000_000:
        raise ValueError("Source image exceeds 80 million pixels")
    return image


def _affines(graph, model):
    """Return design->source and source->design affine maps.

    Existing shared nodes are the strongest declaration of the graph's
    coordinate system.  The declared origin/scale is only a bounded fallback.
    """
    nodes = graph.get("nodes", [])
    if len(nodes) >= 3:
        design = np.asarray([[float(n["x"]), float(n["y"]), 1.] for n in nodes], float)
        source = np.asarray([n["source_px"] for n in nodes], float)
        if source.ndim == 2 and source.shape[1] == 2 and np.isfinite(source).all():
            source_h = np.c_[source, np.ones(len(source))]
            if np.linalg.matrix_rank(design) == 3 and np.linalg.matrix_rank(source_h) == 3:
                d2s = np.linalg.lstsq(design, source, rcond=None)[0]
                s2d = np.linalg.lstsq(source_h, design[:, :2], rcond=None)[0]
                error = np.max(np.linalg.norm(source_h @ s2d - design[:, :2], axis=1))
                if math.isfinite(float(error)) and error <= 1e-5:
                    return (lambda points: np.c_[np.asarray(points, float), np.ones(len(points))] @ d2s,
                            lambda points: np.c_[np.asarray(points, float), np.ones(len(points))] @ s2d,
                            float(np.linalg.det(d2s[:2, :])))
    system = graph.get("coordinate_system") or model.get("coordinate_system") or {}
    origin = np.asarray(system.get("origin_source_px", [0., 0.]), float)
    units = graph.get("units", system.get("units", "pixel"))
    ppm = float((model.get("scale") or {}).get("pixels_per_mm") or 1.) if units == "mm" else 1.
    if origin.shape != (2,) or not np.isfinite(origin).all() or not math.isfinite(ppm) or ppm <= 0:
        raise ValueError("Topology coordinate transform is unavailable")
    return (lambda points: np.asarray(points, float) * [ppm, -ppm] + origin,
            lambda points: (np.asarray(points, float) - origin) / [ppm, -ppm],
            -ppm * ppm)


def _entity_design_samples(entity, maximum_step):
    start, end = np.asarray(entity["start"], float), np.asarray(entity["end"], float)
    if entity.get("type") == "LINE":
        count = max(2, int(math.ceil(float(np.linalg.norm(end-start)) / maximum_step)) + 1)
        return np.linspace(start, end, count)
    if entity.get("type") != "ARC":
        raise ValueError("Topology candidates support only LINE and ARC entities")
    center = np.asarray(entity["center"], float)
    radius = float(entity["radius"])
    if radius <= 0 or not np.isfinite(center).all() or not math.isfinite(radius):
        raise ValueError("Invalid base ARC")
    first = math.atan2(start[1]-center[1], start[0]-center[0])
    last = math.atan2(end[1]-center[1], end[0]-center[0])
    sweep = -((first-last) % (2*math.pi)) if entity.get("clockwise") else ((last-first) % (2*math.pi))
    count = max(4, int(math.ceil(abs(sweep)*radius / maximum_step)) + 1)
    angles = np.linspace(first, first+sweep, count)
    return center + radius*np.c_[np.cos(angles), np.sin(angles)]


def _base_source_ring(graph, design_to_source, grid):
    entities = graph.get("entities", [])
    if not entities or len(entities) > 5000:
        raise ValueError("Base topology must contain 1 to 5000 entities")
    points = []
    previous_end = None
    for index, entity in enumerate(entities):
        expected_start = f"v{index:03d}"
        if entity.get("start_node") != expected_start:
            # Arbitrary IDs are allowed, but the chain must still be explicit.
            expected_start = entity.get("start_node")
        following = entities[(index+1) % len(entities)]
        if entity.get("end_node") != following.get("start_node"):
            raise ValueError("Base topology is not an ordered shared-node cycle")
        design_samples = _entity_design_samples(entity, max(.25, .35*grid))
        source_samples = design_to_source(design_samples)
        if previous_end is not None and np.linalg.norm(source_samples[0]-previous_end) > 1e-4:
            raise ValueError("Base topology contains a geometric endpoint gap")
        points.extend(source_samples[:-1])
        previous_end = source_samples[-1]
    ring = _closed_ring(points)
    if len(ring) > 30000:
        # Deterministic arclength thinning keeps the candidate search bounded.
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))]
        targets = np.linspace(0., distance[-1], 30000, endpoint=False)
        ring = _closed_ring(np.c_[np.interp(targets, distance, ring[:, 0]),
                                  np.interp(targets, distance, ring[:, 1])])
    return ring


def _base_source_entities(graph, design_to_source, orientation_det):
    reflected = orientation_det < 0
    result = []
    for entity in graph.get("entities", []):
        start, end = design_to_source(np.asarray([entity["start"], entity["end"]], float))
        item = {"type": entity["type"], "start": start.tolist(), "end": end.tolist(),
                "fit_error_px": 0.}
        if entity["type"] == "ARC":
            center = design_to_source(np.asarray([entity["center"]], float))[0]
            item.update(center=center.tolist(), radius=float(np.linalg.norm(start-center)),
                        clockwise=not bool(entity.get("clockwise")) if reflected else bool(entity.get("clockwise")))
        result.append(item)
    return result


def _point_box_gap(point, box):
    low, high = box.min(axis=0), box.max(axis=0)
    return float(np.linalg.norm(np.maximum(low-point, 0.) + np.minimum(high-point, 0.)))


def _annotation_inventory(gray, records, ring, grid):
    """Find source-observed leader candidates without assigning CAD truth."""
    lines = _leaders(gray, records)
    tree = cKDTree(ring[:-1])
    inventory = []
    for row in records[:250]:
        parsed = row.get("parsed", {})
        box_value = row.get("box")
        try:
            box = np.asarray(box_value, float)
        except (TypeError, ValueError):
            box = np.empty((0, 2))
        item = {"record_id": row.get("id"), "text": str(row.get("text", "")),
                "kind": parsed.get("kind", "unknown"), "nominal": parsed.get("nominal"),
                "source_box": box_value, "leader_status": "not_detected"}
        if box.ndim != 2 or box.shape[1] != 2 or len(box) < 2 or not np.isfinite(box).all():
            item["leader_status"] = "invalid_source_box"
            inventory.append(item)
            continue
        size = max(12., float(np.linalg.norm(box.max(axis=0)-box.min(axis=0))))
        options = []
        for line_index, segment in enumerate(lines):
            for label_end, target_end in (segment, segment[::-1]):
                label_gap = _point_box_gap(label_end, box)
                target_gap, contour_index = tree.query(target_end)
                if label_gap > max(18., .75*size) or target_gap > max(12., 2.25*grid):
                    continue
                direction = target_end-label_end
                length = float(np.linalg.norm(direction))
                if length < max(8., grid):
                    continue
                arrow = _arrowhead_evidence(gray, target_end, direction/length, size, max(4., 2*grid))
                options.append((target_gap+.2*label_gap+(0. if arrow else grid), {
                    "method": "source_hough_segment_between_ocr_box_and_material_boundary",
                    "leader_id": f"line{line_index:03d}", "segment_px": segment.tolist(),
                    "label_endpoint_gap_px": float(label_gap), "boundary_endpoint_gap_px": float(target_gap),
                    "target_source_px": ring[int(contour_index)].tolist(),
                    "arrowhead_verified": bool(arrow), "arrowhead": arrow,
                    "status": "directed_arrow_candidate" if arrow else "undirected_leader_candidate",
                }))
        if options:
            _, best = min(options, key=lambda value: value[0])
            item.update(leader_status=best["status"], leader=best)
        inventory.append(item)
    return inventory, {"detected_source_line_segments": len(lines), "record_budget": 250,
                       "records_considered": len(inventory),
                       "method": "OCR-box-to-source-boundary leader candidate detection",
                       "limitation": "An undirected Hough segment is a hypothesis, not a verified dimension-to-primitive binding."}


def _entity_annotation_support(entities, inventory, grid):
    result = []
    supported = compatible = 0
    for row in inventory:
        leader = row.get("leader")
        base = {key: row.get(key) for key in ("record_id", "text", "kind", "nominal", "leader_status")}
        if not leader:
            result.append({**base, "status": "no_source_leader_candidate"})
            continue
        target = np.asarray(leader["target_source_px"], float)[None, :]
        distances = [float(_primitive_distance(target, entity)[0]) for entity in entities]
        index = int(np.argmin(distances))
        entity = entities[index]
        gap = distances[index]
        type_compatible = row.get("kind") != "radius" or entity["type"] == "ARC"
        source_supported = gap <= max(12., 2.25*grid)
        supported += int(source_supported)
        compatible += int(source_supported and type_compatible)
        result.append({**base, "status": "candidate_supported" if source_supported else "target_missed",
                       "candidate_entity_id": f"g{index:03d}", "candidate_entity_type": entity["type"],
                       "target_gap_px": gap, "type_compatible": bool(type_compatible),
                       "arrowhead_verified": bool(leader.get("arrowhead_verified")),
                       "source_evidence": leader})
    target_count = sum(bool(row.get("leader")) for row in inventory)
    return result, {"leader_target_count": target_count, "targets_reached": supported,
                    "type_compatible_targets": compatible,
                    "compatibility_fraction": compatible/target_count if target_count else None}


def _to_graph(source_entities, source_to_design, orientation_det, base_graph, candidate_id,
              source_sha256, parent_hash, tolerance, quality, support, support_summary, strategy):
    starts = np.asarray([entity["start"] for entity in source_entities], float)
    design_starts = source_to_design(starts)
    nodes = [{"id": f"v{index:03d}", "x": float(point[0]), "y": float(point[1]),
              "source_px": starts[index].tolist()}
             for index, point in enumerate(design_starts)]
    entities = []
    reflected = orientation_det < 0
    for index, source_entity in enumerate(source_entities):
        start = design_starts[index]
        end = design_starts[(index+1) % len(source_entities)]
        entity = {"id": f"g{index:03d}", "type": source_entity["type"],
                  "start_node": f"v{index:03d}", "end_node": f"v{(index+1)%len(source_entities):03d}",
                  "start": start.tolist(), "end": end.tolist(),
                  "parameter_source": "bounded_source_topology_candidate", "dimension_bound": False,
                  "source_fit_error_px": float(source_entity.get("fit_error_px", 0.))}
        if source_entity["type"] == "ARC":
            center = source_to_design(np.asarray([source_entity["center"]], float))[0]
            # Affine transforms in this project are rigid scale/reflection maps.
            radius = float(np.linalg.norm(start-center))
            entity.update(center=center.tolist(), radius=radius,
                          clockwise=not bool(source_entity["clockwise"]) if reflected else bool(source_entity["clockwise"]))
            if source_entity.get("radius_annotation_evidence"):
                entity.update(parameter_source="multimodal_annotation_guided_arc_refit",
                              radius_annotation_evidence=copy.deepcopy(source_entity["radius_annotation_evidence"]))
            if source_entity.get("radius_binding"):
                entity.update(parameter_source="multimodal_annotation_guided_arc_refit", dimension_bound=True,
                              radius_binding=copy.deepcopy(source_entity["radius_binding"]))
        entities.append(entity)
    base_evidence = base_graph.get("source_evidence") or {}
    baseline_support = base_evidence.get("baseline_stroke_support") or support_summary["source_stroke_support"]
    graph = {"schema_version": "source-topology-v1", "status": "candidate_proposal",
             "candidate_id": candidate_id, "parent_topology_sha256": parent_hash,
             "source_sha256": source_sha256, "units": base_graph.get("units", "pixel"),
             "coordinate_system": base_graph.get("coordinate_system", {}), "nodes": nodes, "entities": entities,
             "relations": _relations(entities), "source_grid_pitch_px": base_graph.get("source_grid_pitch_px"),
             "proposal_tolerance_px": tolerance, "ground_truth_used": False, "baseline_modified": False,
             "requires_dimension_binding": True,
             "source_evidence": {"baseline_stroke_support": baseline_support,
                                 "proposal_stroke_support": support_summary["source_stroke_support"],
                                 "source_deviation": quality,
                                 "selection": "bounded_candidate_for_separate_planner",
                                 "source_residual": quality, "annotation_support": support_summary,
                                 "strategy": strategy},
             "annotation_support": support,
             "validation": {"closed": True, "connected": True,
                            "simple": bool(quality.get("sampled_topology_valid")),
                            "ordered_entity_cycle": True, "dimensions_solved": False,
                            "engineering_verified": False},
             "scope": "A source-only LINE/ARC topology hypothesis for planner comparison and later dimension solving."}
    return graph


def _render_overlay(image, source_entities, candidate_id, path):
    scale = min(1., 2400/max(image.shape[:2]))
    canvas = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    colors = {"LINE": (20, 125, 20), "ARC": (20, 95, 210)}
    for index, entity in enumerate(source_entities):
        samples, _, _ = _sample_entities([entity], max_step_px=max(.5, 1/scale))
        pixels = np.rint(samples*scale).astype(np.int32)
        cv2.polylines(canvas, [pixels], False, colors[entity["type"]], 2, cv2.LINE_AA)
        if index < 250:
            point = tuple(np.rint(samples[len(samples)//2]*scale).astype(int))
            cv2.putText(canvas, f"g{index:03d}", point, cv2.FONT_HERSHEY_SIMPLEX, .34,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"g{index:03d}", point, cv2.FONT_HERSHEY_SIMPLEX, .34,
                        (25, 65, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, candidate_id, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .60,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, candidate_id, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .60,
                (15, 70, 15), 1, cv2.LINE_AA)
    cv2.imencode(".png", canvas)[1].tofile(str(path))


def _planner_fields(candidate, support, entity_count):
    supported_ids = {row.get("candidate_entity_id") for row in support
                     if row.get("status") == "candidate_supported" and row.get("candidate_entity_id")}
    evidence_ids = sorted({row.get("record_id") for row in support if row.get("record_id")})
    evidence_ids.extend(sorted({row.get("source_evidence", {}).get("leader_id") for row in support
                                if row.get("source_evidence", {}).get("leader_id")}))
    candidate.update(annotation_coverage=float(candidate["annotation_support"].get("compatibility_fraction") or 0.),
                     unsupported_primitive_count=max(0, int(entity_count)-len(supported_ids)),
                     relation_ids=[relation["id"] for relation in candidate["graph"].get("relations", [])][:32],
                     binding_candidate_ids=[], evidence_ids=list(dict.fromkeys(evidence_ids))[:32],
                     topology=candidate["graph"])


def materialize_selected_candidate(candidate, bundle, output_dir):
    """Publish one planner-selected candidate under the legacy artifact names.

    This is a representation copy only.  It does not claim solver success or
    reference accuracy.  The candidate must be an unchanged member of the
    supplied bundle and retain the bundle's source and parent hashes.
    """
    members = {row.get("id"): row for row in bundle.get("candidates", [])}
    selected = members.get(candidate.get("id"))
    if selected is None or _json_hash(selected.get("graph")) != _json_hash(candidate.get("graph")):
        raise ValueError("Selected candidate is not an unchanged member of the bundle")
    if candidate.get("source_sha256") != bundle.get("source_sha256"):
        raise ValueError("Selected candidate source hash does not match the bundle")
    if candidate.get("parent_topology_sha256") != bundle.get("base_graph_sha256"):
        raise ValueError("Selected candidate parent hash does not match the bundle")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    topology_path = output/"topology.json"
    topology_path.write_text(json.dumps(candidate["graph"], ensure_ascii=False, indent=2,
                                        allow_nan=False), encoding="utf-8")
    overlay_target = output/"topology-overlay.png"
    overlay = candidate.get("overlay_path")
    overlay_source = output/overlay if overlay else None
    if overlay_source is not None and overlay_source.is_file() and overlay_source.resolve() != overlay_target.resolve():
        shutil.copyfile(overlay_source, overlay_target)
    return {"status": "materialized", "candidate_id": candidate["id"],
            "topology_path": str(topology_path),
            "overlay_path": str(overlay_target) if overlay_target.is_file() else None,
            "ground_truth_used": False, "reference_accuracy_verified": False}


def generate_topology_candidates(image_path, document, baseline_model, base_graph, output_dir=None,
                                 *, max_candidates=5):
    """Generate three to five bounded LINE/ARC topology hypotheses.

    `base_graph` must itself be source-only.  The returned candidates preserve
    shared endpoints and carry sufficient evidence for a separate planner to
    compare object count, annotation compatibility and source residual.  A
    candidate rank is a planning prior only; it is not a reference-accuracy
    result and does not authorize final CAD publication.
    """
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 3 <= max_candidates <= 5:
        raise ValueError("max_candidates must be an integer from 3 to 5")
    if base_graph.get("ground_truth_used") is not False:
        raise ValueError("Base topology must explicitly declare ground_truth_used=false")
    image_path = Path(image_path)
    source_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if base_graph.get("source_sha256") != source_sha256:
        raise ValueError("Base topology source hash does not match the source image")
    image = _read_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grid = float(base_graph.get("source_grid_pitch_px") or 1.)
    if not math.isfinite(grid) or grid <= 0:
        raise ValueError("Base topology requires a positive source_grid_pitch_px")
    design_to_source, source_to_design, orientation_det = _affines(base_graph, baseline_model)
    ring = _base_source_ring(base_graph, design_to_source, grid)
    base_source_entities = _base_source_entities(base_graph, design_to_source, orientation_det)
    records = canonical_records(document)
    inventory, inventory_summary = _annotation_inventory(gray, records, ring, grid)
    stroke = _StrokeEvidence(gray, records, grid)
    parent_hash = _json_hash(base_graph)
    candidates = []
    failures = []
    # The first hypothesis always preserves the existing source-only topology
    # exactly.  Later hypotheses may simplify it, but cannot silently replace
    # the baseline in the candidate set.
    attempts = [("base_topology", None, "Preserve the supplied ordered LINE/ARC topology exactly.",
                 base_source_entities)]
    attempts.extend((name, multiple, description, None)
                    for name, multiple, description in _STRATEGIES[:max_candidates-1])
    for index, (strategy_name, multiple, description, supplied_entities) in enumerate(attempts):
        tolerance = float(base_graph.get("proposal_tolerance_px") or max(.25, .5*grid)) if multiple is None else max(.25, multiple*grid)
        try:
            source_entities = supplied_entities or fit_polyline(ring, tolerance_px=tolerance)
            quality = assess_fit_quality(ring, source_entities, max_step_px=max(.5, min(2., grid/2)))
            sampled, _, _ = _sample_entities(source_entities, max_step_px=max(.5, min(2., grid/2)))
            stroke_support = stroke.summarize(sampled)
            support, support_summary = _entity_annotation_support(source_entities, inventory, grid)
            support_summary["source_stroke_support"] = stroke_support
            candidate_id = f"cand-{index+1:02d}-{strategy_name}"
            graph = _to_graph(source_entities, source_to_design, orientation_det, base_graph, candidate_id,
                              source_sha256, parent_hash, tolerance, quality, support, support_summary,
                              {"name": strategy_name, "description": description,
                               "tolerance_grid_multiple": multiple})
            counts = {"total": len(source_entities),
                      "LINE": sum(e["type"] == "LINE" for e in source_entities),
                      "ARC": sum(e["type"] == "ARC" for e in source_entities)}
            deviation = quality["source_boundary_deviation_px"]
            compatibility = support_summary["compatibility_fraction"]
            annotation_penalty = 1.-compatibility if compatibility is not None else 1.
            before_support = graph["source_evidence"]["baseline_stroke_support"]
            after_support = graph["source_evidence"]["proposal_stroke_support"]
            support_gate = bool(
                float(after_support["edge_supported_fraction"]) >= float(before_support["edge_supported_fraction"])-.035 and
                float(after_support["p90_edge_distance_px"]) <= float(before_support["p90_edge_distance_px"])+grid)
            planning_penalty = (counts["total"]/max(1, len(base_graph.get("entities", []))) +
                                deviation["conservative_upper_bound_px"]/max(grid, 1e-6) +
                                2.*annotation_penalty +
                                3.*(not support_gate) +
                                2.*(not quality["sampled_topology_valid"]))
            candidates.append({"id": candidate_id, "strategy": graph["source_evidence"]["strategy"],
                               "parent_topology_sha256": parent_hash, "source_sha256": source_sha256,
                               "graph": graph, "entity_counts": counts,
                               "annotation_support": support_summary,
                               "complexity": {"entity_count": counts["total"],
                                              "relative_to_base": counts["total"]/max(1, len(base_graph.get("entities", []))),
                                              "fit_tolerance_px": tolerance},
                               "source_residual": quality, "source_stroke_support": stroke_support,
                               "planner_signals": {"planning_penalty": float(planning_penalty),
                                                   "lower_is_better": True,
                                                   "source_support_gate_passed": support_gate,
                                                   "reference_accuracy_measured": False,
                                                   "requires_constraint_binding_and_solve": True},
                               "ground_truth_used": False,
                               "overlay_path": str((Path(output_dir)/f"topology-candidate-{candidate_id}.png").resolve())
                                               if output_dir is not None else None})
            _planner_fields(candidates[-1], support, counts["total"])
            if output_dir is not None:
                output = Path(output_dir)
                output.mkdir(parents=True, exist_ok=True)
                _render_overlay(image, source_entities, candidate_id, Path(candidates[-1]["overlay_path"]))
        except (ValueError, ArithmeticError) as error:
            if index == 0:
                raise ValueError(f"Base topology candidate is invalid: {error}") from error
            failures.append({"strategy": strategy_name, "tolerance_px": tolerance,
                             "error": str(error), "ground_truth_used": False})
    if len(candidates) < 3:
        raise ValueError("Fewer than three valid source-only topology candidates could be generated")
    ordered = sorted(candidates, key=lambda item: (item["planner_signals"]["planning_penalty"], item["id"]))
    for rank, candidate in enumerate(ordered, 1):
        candidate["planner_signals"]["prior_rank"] = rank
    bundle = {"schema_version": "source-topology-candidate-set-v1", "status": "proposal_set",
              "source_sha256": source_sha256, "base_graph_sha256": parent_hash,
              "ground_truth_used": False, "baseline_modified": False,
              "candidate_count": len(candidates), "candidates": candidates,
              "annotation_inventory": inventory, "annotation_inventory_summary": inventory_summary,
              "generation": {"requested_candidate_count": max_candidates,
                             "attempted_strategies": len(attempts),
                             "failed_strategies": failures, "bounded": True,
                             "allowed_entity_types": ["LINE", "ARC"],
                             "selection_policy": "Separate planner compares source evidence; no candidate is certified here."},
              "limitations": ["The base material boundary remains segmentation-derived.",
                              "An undirected leader candidate is not a verified dimension binding.",
                              "Planning rank is source-only and cannot establish GT or engineering accuracy."]}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output/"topology-candidates.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return bundle

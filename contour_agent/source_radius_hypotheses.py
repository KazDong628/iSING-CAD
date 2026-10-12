"""Bounded nominal-radius hypotheses before final online certification.

The segmentation intervals are initial unknowns. A uniquely owned source
arrow may therefore justify trying an exact radius even when the OLD finite
interval is not one admissible circle. No hypothesis becomes a binding until
the moved interval is independently rediscovered and all prior obligations,
original source gates and native equations have been checked again.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def _source_identity(image_path, document):
    from .source_ocr_review import canonical_document_sha256
    return {"source_image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "effective_ocr_sha256": canonical_document_sha256(document)}


def _same_record_sources(document, inventory, record_ids):
    from .ocr import canonical_records
    current = {row["id"]: row for row in canonical_records(document)}
    previous = {row["id"]: row for row in inventory.get("all_records", inventory.get("records", []))}
    keys = ("id", "text", "box")
    return all(rid in current and rid in previous and
        _json_hash({key: current[rid].get(key) for key in keys}) ==
        _json_hash({key: previous[rid].get(key) for key in keys}) for rid in record_ids if rid is not None)


def radius_hypotheses(inventory, graph, constraints):
    """Return source-owned hypotheses, never proximity-only alternatives."""
    entities = {row["id"]: row for row in graph.get("entities", [])}
    records = {row["id"]: row for row in inventory.get("all_records", inventory.get("records", []))}
    bound_records = {row.get("record_id") for row in constraints}
    bound_arcs = {row["entities"][0] for row in constraints
                  if row.get("kind") == "radius" and len(row.get("entities", [])) == 1}
    eligible = []
    for candidate in inventory.get("all_candidates", inventory.get("candidates", [])):
        rid = candidate.get("record_id")
        ids = candidate.get("entities", [])
        evidence = candidate.get("evidence") or {}
        interval = evidence.get("whole_primitive_radius") or {}
        leader = evidence.get("leader") or {}
        ownership = leader.get("source_arrow_ownership") or {}
        arrow = leader.get("arrowhead") or {}
        text = leader.get("source_text_shaft_attachment") or {}
        record = records.get(rid, {})
        nominal = candidate.get("value")
        # A source-owned two-LINE fillet can have the right nominal R but
        # wrong contacts. Local ink rejection must schedule a numerical
        # hypothesis, not make the candidate permanently unreachable.
        fillet_retry=(interval.get('status')=='constructed_fillet_original_image_not_verified' and
            interval.get('method')=='original_image_exact_two_line_fillet_admission_v1' and
            interval.get('observation')=='fresh_original_image_ink' and
            len(interval.get('source_ink_checks') or [])==3 and
            entities.get(ids[0] if len(ids)==1 else '',{}).get('fillet_construction')==
                'original_image_two_line_exact_radius_seed' and
            (entities.get(ids[0] if len(ids)==1 else '',{}).get('source_refinement') or {}).get('ground_truth_used') is False)
        if (candidate.get("kind") != "radius" or rid in bound_records or
                len(ids) != 1 or ids[0] in bound_arcs or
                entities.get(ids[0], {}).get("type") != "ARC" or
                type(nominal) not in (int, float) or not math.isfinite(nominal) or nominal <= 0 or
                record.get("parsed", {}).get("kind") != "radius" or
                record.get("parsed", {}).get("nominal") != nominal or
                interval.get("checked") is not True or interval.get("passed") is not False or
                (interval.get("status") != "requires_topology_repartition" and not fillet_retry) or
                evidence.get("uniquely_supported_leader") is not True or
                evidence.get("directed_source_target_count") != 1 or
                (evidence.get("source_text") or {}).get("symbol_confusion") is True or
                leader.get("arrowhead_verified") is not True or arrow.get("verified") is not True or
                (leader.get("shaft_evidence") or {}).get("verified") is not True or
                (leader.get("contour_visibility") or {}).get("verified") is not True or
                text.get("strong_text_adjacency") is not True or
                ownership.get("status") != "globally_unique_source_claim" or
                ownership.get("duplicate_target_ambiguity_preserved") is True):
            continue
        targets = leader.get("target_candidates") or []
        if len(targets) != 1 or targets[0].get("entity_id") != ids[0]:
            continue
        eligible.append(candidate)
    # Multiple source records proposing incompatible values for the same arc
    # remain a partition/ownership ambiguity, not a radius-value vote.
    by_arc = {}
    by_record = {}
    for row in eligible:
        by_arc.setdefault(row["entities"][0], set()).add((row["record_id"], row["value"]))
        by_record.setdefault(row["record_id"], set()).add(row["entities"][0])
    unique = {}
    for row in eligible:
        if len(by_arc[row["entities"][0]]) == 1 and len(by_record[row["record_id"]]) == 1:
            unique.setdefault((row["record_id"], row["entities"][0]), row)
    return [copy.deepcopy(unique[key]) for key in sorted(unique)]


def graph_from_solution(original, baseline, solution):
    """Move unknowns while preserving IDs, traversal and the original affine."""
    from .topology_candidates import _affines
    if solution.get("accepted") is not True:
        raise ValueError("A rejected numerical trial cannot seed a source-radius graph")
    old_entities = original.get("entities", [])
    entities = solution.get("entities", [])
    identity = ("id", "type", "start_node", "end_node", "clockwise")
    if len(entities) != len(old_entities) or any(
            any(old.get(key) != new.get(key) for key in identity)
            for old, new in zip(old_entities, entities)):
        raise ValueError("Radius hypothesis cannot change object count, type or traversal")
    design_to_source, _, _ = _affines(original, baseline)
    graph = copy.deepcopy(original)
    graph["entities"] = copy.deepcopy(entities)
    endpoints = {}
    for entity in entities:
        for end in ("start", "end"):
            point = np.asarray(entity[end], float)
            node_id = entity[end + "_node"]
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError("Nonfinite source-radius endpoint")
            if node_id in endpoints and np.linalg.norm(endpoints[node_id] - point) > 1e-8:
                raise ValueError("Source-radius shared endpoints disagree")
            endpoints[node_id] = point
    if set(endpoints) != {row["id"] for row in graph["nodes"]}:
        raise ValueError("Source-radius node identity changed")
    for node in graph["nodes"]:
        point = endpoints[node["id"]]
        node.update(x=float(point[0]), y=float(point[1]),
                    source_px=design_to_source([point])[0].tolist())
    graph["requires_dimension_binding"] = True
    graph.setdefault("source_evidence", {})["source_radius_hypothesis"] = {
        "method": "exact_nominal_joint_solve_then_current_interval_reverification_v1",
        "topology_changed": False, "coordinates_are_initial_solution": True,
        "reference_geometry_used": False, "final_online_binding_required": True}
    return graph


def _carried_radius_finite_ownership(leader, record_id, entity_id, solved_graph,
                                     transform, inverse, band, fresh_inventory, *, gray=None,
                                     certified_radius=None, verified_interval=None):
    """Recheck one physical arrow against every current finite primitive.

    A rejected *different* arrow hypothesis for the same OCR record is not a
    contradiction. Conversely, a fresh competing claim on this same original
    shaft cannot be hidden by the absence of an accepted numeric constraint.
    This validates ownership only; it never binds a radius or certifies DXF.
    """
    from .source_arrow_localization import _same_source_arrow, same_original_ink_arrow_shaft
    from .topology import _primitive_distance
    target_band = max(10., 1.7 * float(band))
    result = {"method": "same_physical_source_arrow_current_finite_ownership_v1",
              "passed": False, "record_id": record_id, "entity_id": entity_id,
              "original_arrow_target_band_px": target_band, "current_target_candidates": [],
              "fresh_contradictions": [], "ground_truth_used": False,
              "numeric_binding_approved": False}
    try:
        point = np.asarray(leader["arrowhead"]["tip_px"], float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("invalid original arrow tip")
        source_scale = transform([[1., 0.], [0., 0.]])
        ppm = float(np.linalg.norm(source_scale[0] - source_scale[1]))
        if not math.isfinite(ppm) or ppm <= 0:
            raise ValueError("invalid original source affine")
        design_point = inverse([point])
        entities = solved_graph.get("entities", [])
        if not entities or len({e["id"] for e in entities}) != len(entities):
            raise ValueError("invalid current entity identity")
        by_id={entity['id']:entity for entity in entities}
        typed_radius=certified_radius is not None
        if typed_radius:
            interval=verified_interval or {};owned=by_id.get(entity_id,{})
            if (type(certified_radius) not in (int,float) or not math.isfinite(certified_radius) or
                    owned.get('type')!='ARC' or owned.get('radius')!=certified_radius or
                    interval.get('checked') is not True or interval.get('passed') is not True or
                    interval.get('entity_id')!=entity_id or interval.get('nominal')!=certified_radius):
                raise ValueError('certified radius interval unavailable')
        for entity in entities:
            gap = float(_primitive_distance(design_point, entity)[0]) * ppm
            if not math.isfinite(gap):
                raise ValueError("nonfinite primitive distance")
            if gap <= target_band:
                result["current_target_candidates"].append({"entity_id": entity["id"],
                    "entity_type": entity["type"], "tip_gap_px": gap})
        result["current_target_candidates"].sort(key=lambda row: (row["tip_gap_px"], row["entity_id"]))
        radius_targets=[row for row in result['current_target_candidates'] if row['entity_type']=='ARC']
        ownership_targets=radius_targets if typed_radius else result['current_target_candidates']
        if typed_radius:
            result.update(type_semantics='previous_exact_radius_and_verified_current_finite_arc_interval',
                coarse_line_hypotheses=[row for row in result['current_target_candidates'] if row['entity_type']=='LINE'])
        if [row["entity_id"] for row in ownership_targets] != [entity_id]:
            result["reason"] = "original_arrow_current_finite_target_not_unique"
            return result

        def same_arrow(row):
            return (_same_source_arrow(leader, row, band) or
                    same_original_ink_arrow_shaft(gray, leader, row))

        positive_rejection_reasons = {"shared_source_arrow_ownership_ambiguous",
            "source_arrow_claim_owned_by_other_label", "outside_selected_boundary_or_scope_conflict"}
        for row in fresh_inventory.get("radius_source_rejections", []):
            if (row.get("record_id") == record_id and row.get("reason") in positive_rejection_reasons
                    and same_arrow(row)):
                result["fresh_contradictions"].append({"reason": row["reason"],
                    "record_id": row.get("record_id"), "selected_record_id": row.get("selected_record_id")})
        observations = list(fresh_inventory.get("radius_source_observations", []))
        for candidate in fresh_inventory.get("all_candidates", fresh_inventory.get("candidates", [])):
            observed = (candidate.get("evidence") or {}).get("leader")
            if observed:
                observations.append({**observed, "record_id": candidate.get("record_id")})
        for observed in observations:
            if not same_arrow(observed):
                continue
            ownership = observed.get("source_arrow_ownership") or {}
            if ownership.get("status") != "globally_unique_source_claim":
                continue
            other_label = observed.get("record_id") != record_id
            targets = {row.get("entity_id") for row in observed.get("target_candidates", [])}
            line_ids={eid for eid,entity in by_id.items() if entity.get('type')=='LINE'}
            resolved_lines=bool(typed_radius and targets.intersection(line_ids) and
                                targets-line_ids=={entity_id})
            typed_targets=targets-line_ids if typed_radius else targets
            conflict = (other_label or (ownership.get("duplicate_target_ambiguity_preserved") is True and not resolved_lines) or
                        bool(typed_targets and typed_targets != {entity_id}))
            if conflict:
                result["fresh_contradictions"].append({"reason": "same_source_arrow_positive_ownership_conflict",
                    "record_id": observed.get("record_id"), "target_entity_ids": sorted(targets - {None}),
                    "duplicate_target_ambiguity_preserved": ownership.get("duplicate_target_ambiguity_preserved") is True})
        if result["fresh_contradictions"]:
            result["reason"] = "same_original_arrow_has_fresh_source_contradiction"
            return result
        result.update(passed=True, reason="same_original_arrow_still_uniquely_owns_current_finite_primitive")
    except (ValueError, TypeError, KeyError, ArithmeticError) as error:
        result.update(reason="current_finite_arrow_ownership_invalid", error_type=type(error).__name__)
    return result


def _same_solved_geometry(reconstructed, supplied):
    """Exact CAD identity, with roundoff allowance only for derived pixel caches.

    Re-estimating the same affine can change source_px by a few ulps. These
    caches are not solved CAD coordinates and must not revoke an exact solve.
    All entity data, node identities, x/y and other node metadata stay exact.
    """
    if _json_hash(reconstructed['entities']) != _json_hash(supplied['entities']):
        return False
    first, second = reconstructed['nodes'], supplied['nodes']
    if len(first) != len(second):
        return False
    for old, new in zip(first, second):
        if _json_hash({k:v for k,v in old.items() if k != 'source_px'}) != \
                _json_hash({k:v for k,v in new.items() if k != 'source_px'}):
            return False
        try:
            a, b = np.asarray(old['source_px'], float), np.asarray(new['source_px'], float)
            if (a.shape != (2,) or b.shape != (2,) or not np.isfinite(a).all() or
                    not np.isfinite(b).all() or np.max(np.abs(a-b)) > 1e-7):
                return False
        except (ValueError, TypeError, KeyError):
            return False
    return True


def carried_binding_certificate(image_path, document, baseline, original_graph,
                                solved_graph, original_inventory, original_constraints,
                                fresh_inventory, fresh_bindings, solution):
    """Retain already established semantics when repeat detection is missing.

    This is not new binding admission. The original arrow/straight-support
    evidence and its immutable inputs remain the authority; native equations,
    finite ownership and any newly contradictory source evidence are rechecked.
    """
    from .constraint_binding import _constraint, _radius_primitive_feasibility
    from .parametric_pipeline import _constraint_signatures, _constraint_receipts_complete
    from .topology_candidates import _affines
    from .topology import _primitive_distance
    from .ocr import canonical_records
    source_gray = None
    source_records = canonical_records(document)
    identity = _source_identity(image_path, document)
    result = {"schema_version": "carried-source-binding-certificate-v1", "passed": False,
        "input_identity": identity, "checks": [], "constraints": [],
        "current_API_reconfirmed": False, "ground_truth_used": False,
        "basis": "previous_independent_source_binding_same_inputs_and_topology",
        "repeat_detector_nondetection_is_not_source_contradiction": True}
    if original_inventory.get("source_image_sha256") != identity["source_image_sha256"]:
        result["reason"] = "original_binding_image_changed"
        return result
    if not _same_record_sources(document, original_inventory,
                                [row.get("record_id") for row in original_constraints]):
        result["reason"] = "original_binding_ocr_text_or_box_changed"
        return result
    try:
        # graph_from_solution independently checks the traversal and shared
        # node identity; no proof can migrate to a replacement primitive.
        reconstructed = graph_from_solution(original_graph, baseline, solution)
        if not _same_solved_geometry(reconstructed, solved_graph):
            result["reason"] = "supplied_graph_differs_from_exact_solution"
            return result
        if not _constraint_receipts_complete({"constraints": original_constraints}, solution, original_graph):
            result["reason"] = "original_constraints_not_numerically_satisfied"
            return result
        old = original_inventory.get("all_candidates", original_inventory.get("candidates", []))
        new = fresh_inventory.get("all_candidates", fresh_inventory.get("candidates", []))
        entities = {row["id"]: row for row in solved_graph["entities"]}
        transform, inverse, _ = _affines(original_graph, baseline)
        ppm = float(np.linalg.norm(transform([[1., 0.], [0., 0.]])[0] -
                                   transform([[1., 0.], [0., 0.]])[1]))
        band = float(original_inventory.get("proposal_tolerance_px") or original_graph.get("proposal_tolerance_px") or 1.)
        target_band = max(10., 1.7 * band)  # existing directed-arrow association band
        fresh_signatures = _constraint_signatures(solved_graph, fresh_bindings.get("constraints", []))
        for constraint in original_constraints:
            signature = _constraint_signatures(original_graph, [constraint])
            check = {"constraint_id": constraint["id"], "record_id": constraint.get("record_id"),
                     "kind": constraint["kind"], "passed": False}
            result["checks"].append(check)
            if signature <= fresh_signatures:
                check.update(passed=True, reason="fresh_source_binding_retained")
                continue
            rid = constraint.get("record_id")
            contradictory = [row for row in fresh_bindings.get("constraints", [])
                if rid is not None and row.get("record_id") == rid and
                _constraint_signatures(solved_graph, [row]) != signature]
            rejected_text = [row for row in fresh_bindings.get("bindings", [])
                if rid is not None and row.get("record_id") == rid and
                row.get("reason") in {"source_observed_text_mismatch", "source_symbol_confusion_requires_confirmation"}]
            if contradictory or rejected_text:
                check["reason"] = "new_positive_source_contradiction"
                continue
            if constraint["source"] == "source_geometry":
                matches = [row for row in original_inventory.get("relations", [])
                    if row.get("local_reliable") is True and (row.get("evidence") or {}).get("verified") is True
                    and row.get("type") == constraint["kind"]
                    and set(row.get("entities", [])) == set(constraint.get("entities", []))]
                if not matches:
                    check["reason"] = "original_relation_source_proof_missing"
                    continue
                check.update(passed=True, reason="same_source_relation_same_shared_objects_exact_native_equation",
                             source_evidence_sha256=_json_hash(matches[0].get("evidence")))
            else:
                matches = [row for row in old if row.get("local_reliable") is True and
                    row.get("record_id") == rid and
                    _constraint_signatures(original_graph, [_constraint(row, "ocr_local_binding")]) == signature]
                if len(matches) != 1:
                    check["reason"] = "unique_original_admitted_source_binding_missing"
                    continue
                previous = matches[0]
                evidence = previous.get("evidence") or {}
                if any(row.get("local_reliable") is True and row.get("record_id") == rid and
                       _constraint_signatures(solved_graph, [_constraint(row, "ocr_local_binding")]) != signature
                       for row in new):
                    check["reason"] = "new_positive_source_target_contradiction"
                    continue
                if constraint["kind"] == "radius":
                    leader = evidence.get("leader") or {}
                    arrow = leader.get("arrowhead") or {}
                    entity = entities[constraint["entities"][0]]
                    if (leader.get("arrowhead_verified") is not True or arrow.get("verified") is not True or
                            (leader.get("shaft_evidence") or {}).get("verified") is not True):
                        check["reason"] = "original_radius_arrow_proof_missing"
                        continue
                    tip = inverse([arrow["tip_px"]])
                    gap = float(_primitive_distance(tip, entity)[0]) * ppm
                    if source_gray is None:
                        from PIL import Image
                        with Image.open(image_path) as source_image:
                            source_gray = np.asarray(source_image.convert("L"))
                    interval = _radius_primitive_feasibility(entity, constraint["value"], leader,
                        baseline, solved_graph, transform, band,
                        gray=source_gray, records=source_records)
                    check.update(current_arrow_to_finite_primitive_px=gap,
                                 original_arrow_target_band_px=target_band, current_interval_check=interval)
                    if gap > target_band or interval.get("passed") is not True:
                        check["reason"] = "old_radius_arrow_no_longer_owns_current_finite_interval"
                        continue
                    ownership = _carried_radius_finite_ownership(leader, rid, entity["id"], solved_graph,
                        transform, inverse, band, fresh_inventory, gray=source_gray,
                        certified_radius=constraint['value'],verified_interval=interval)
                    check["current_finite_arrow_ownership"] = ownership
                    if ownership.get("passed") is not True:
                        check["reason"] = ownership["reason"]
                        continue
                elif constraint["kind"] == "angle" and constraint.get("reference_axis"):
                    observed = evidence.get("angle_observation") or {}
                    proof = observed.get("evidence") or {}
                    if (observed.get("verified") is not True or
                            proof.get("reference_arrow", {}).get("verified") is not True or
                            proof.get("target_arrow", {}).get("verified") is not True):
                        check["reason"] = "original_angle_arrow_pair_missing"
                        continue
                    source = observed["source_line"]
                    a, b = np.asarray(source["start_px"], float), np.asarray(source["end_px"], float)
                    direction = (b - a) / np.linalg.norm(b - a)
                    entity = entities[constraint["entities"][0]]
                    ends = transform([entity["start"], entity["end"]])
                    projections = (ends - a) @ direction
                    overlap = max(0., min(float(np.linalg.norm(b - a)), float(max(projections))) - max(0., float(min(projections))))
                    finite_span = float(np.linalg.norm(ends[1] - ends[0]))
                    old_span = max((float(row.get("supported_span_px") or 0.) for row in observed.get("target_candidates", [])
                                    if row.get("entity_id") == entity["id"] and row.get("whole_line_supported") is True), default=0.)
                    cross = np.array([-direction[1], direction[0]])
                    gap = float(np.max(np.abs((ends - a) @ cross)))
                    # Preserve the already proved semantic association within
                    # its original arrow target corridor. Do not repeat the
                    # fitted Hough whole-LINE admission test after correcting
                    # its slope to the exact annotation; that would turn the
                    # noisy initial fit back into an attribute lock.
                    check.update(current_finite_support_overlap_px=overlap,
                        previous_certified_support_span_px=old_span,
                        current_support_corridor_gap_px=gap, original_arrow_target_band_px=target_band)
                    if old_span <= 0 or finite_span <= 0 or overlap < .78 * min(old_span, finite_span) or gap > target_band:
                        check["reason"] = "old_angle_support_no_longer_owns_current_finite_line"
                        continue
                else:
                    # Existing symbolic station/anchor identity is unchanged.
                    # A fresh positive competing station above revokes it;
                    # nondetection alone cannot rewrite a known equation.
                    check["symbolic_anchor_identity_unchanged"] = True
                check.update(passed=True, reason="same_source_binding_finite_ownership_and_exact_native_value",
                             source_evidence_sha256=_json_hash(evidence))
            if check["passed"]:
                carried = copy.deepcopy(constraint)
                if carried.get("source") != "source_geometry":
                    carried["source"] = "ocr_local_binding"
                carried.update(admission_method="carried_source_binding_certificate",
                               prior_binding_source=constraint["source"], current_API_reconfirmed=False)
                result["constraints"].append(carried)
        result["passed"] = all(row["passed"] for row in result["checks"])
    except (ValueError, TypeError, KeyError, ArithmeticError) as error:
        result.update(reason="carried_source_certificate_invalid", error_type=type(error).__name__)
    return result


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def refine_source_owned_radius_hypotheses(image_path, document, baseline, graph,
                                        bindings, inventory, output_dir, *,
                                        max_attempts=2, max_wall_seconds=90.,
                                        solver=None, binding_analyzer=None,
                                        source_validator=None, topology_validator=None,
                                        seed_entities=None):
    """Return a fully source-rechecked seed, never a published DXF certificate.

    All attempts share one immutable graph, mask budget and wall allowance.
    The caller must subsequently run online binding, preserve the returned
    typed obligations, solve again and perform its normal native DXF checks.
    """
    from .constraint_binding import _constraint, analyze_constraint_bindings
    from .parametric_solver import solve_parametric
    # Runtime imports avoid the pipeline/module import cycle. These helpers
    # contain the existing original-input checks; no alternate relaxed gates.
    from .parametric_pipeline import (_constraint_signatures, _constraint_receipts_complete,
        _solver_source_observation, _solved_source_validation, _topology_source_validation)
    if (type(max_attempts) is not int or not 0 <= max_attempts <= 2 or
            isinstance(max_wall_seconds, bool) or not math.isfinite(max_wall_seconds) or
            not 0 < max_wall_seconds <= 90):
        raise ValueError("Source radius hypotheses require at most two attempts and 90 seconds")
    solver = solver or solve_parametric
    # A full fresh inventory/API pass is mandatory after this bounded numeric
    # phase. Repeating the whole drawing detector inside each hypothesis would
    # consume the numerical budget and silently re-open established equations.
    # Here only the moved radius interval is remeasured against the immutable
    # globally resolved source-arrow inventory.
    supplied_binding_analyzer = binding_analyzer
    source_validator = source_validator or _solved_source_validation
    topology_validator = topology_validator or _topology_source_validation
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + float(max_wall_seconds)
    constraints = copy.deepcopy(bindings.get("constraints", []))
    required = copy.deepcopy(constraints)
    hypotheses = radius_hypotheses(inventory, graph, constraints)
    receipt = {"schema_version": "source-radius-hypotheses-v1", "status": "not_applicable",
        "maximum_attempts": max_attempts, "maximum_wall_seconds": max_wall_seconds,
        "eligible_hypothesis_count": len(hypotheses), "attempts": [], "accepted_record_ids": [],
        "admitted_constraints": required, "ground_truth_used": False, "network_requests": 0,
        "original_graph_and_mask_immutable": True, "acceptance_thresholds_changed": False,
        "numerical_seed_only": True, "final_online_binding_required": True,
        "native_DXF_export_performed": False, "all_dimensions_verified": False}
    receipt['certified_checkpoint_used_as_initial_values'] = seed_entities is not None
    receipt['optimization_domain_recentered'] = False

    def persist():
        receipt["elapsed_seconds"] = time.monotonic() - started
        (output / "source-radius-hypotheses.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf8")

    persist()
    if not hypotheses or not max_attempts:
        return graph, receipt
    identity = _source_identity(image_path, document)
    receipt["input_identity"] = identity
    actual_hash = identity["source_image_sha256"]
    if inventory.get("source_image_sha256") != actual_hash:
        receipt.update(status="rejected", reason="inventory_source_image_changed")
        persist()
        return graph, receipt
    if not _same_record_sources(document, inventory,
            [row.get("record_id") for row in constraints] + [row["record_id"] for row in hypotheses]):
        receipt.update(status="rejected", reason="inventory_ocr_text_or_box_changed")
        persist()
        return graph, receipt
    from .source_topology_initial_guess import line_seed_problem_graphs
    problem_graph,_=line_seed_problem_graphs(graph)
    observation = _solver_source_observation(baseline, problem_graph)
    if not observation or not any(k in observation for k in ('boundary_error_budget','optimization_budget')):
        receipt.update(status="rejected", reason="immutable_source_boundary_budget_missing")
        persist()
        return graph, receipt
    result_graph = graph
    remaining_evaluations = 30000
    for index, candidate in enumerate(hypotheses[:max_attempts]):
        remaining = deadline - time.monotonic()
        if remaining <= 15. or remaining_evaluations <= 0:
            receipt["stop_reason"] = "shared_budget_reserved_for_source_reverification"
            break
        trial_dir = output / f"attempt-{index + 1:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        row = {"candidate_id": candidate["id"], "record_id": candidate["record_id"],
            "entity_id": candidate["entities"][0], "status": "running",
            "original_interval_check": copy.deepcopy(candidate["evidence"]["whole_primitive_radius"]),
            "all_prior_constraints_retained": True, "binding_approved": False}
        receipt["attempts"].append(row)
        persist()
        provisional = _constraint(candidate, "ocr_local_binding")
        provisional["id"] = f"provisional_radius_{index:03d}"
        trial_constraints = copy.deepcopy(required) + [provisional]
        try:
            trial_observation = copy.deepcopy(observation)
            from .reconstruction_priority import mask_is_hard
            if not mask_is_hard(baseline):
                from .source_stroke_prior import source_stroke_observation
                trial_observation['image_stroke_prior'] = source_stroke_observation(
                    image_path, document, baseline, problem_graph,
                    probe_entities=(seed_entities if seed_entities is not None else graph['entities']))
                row['original_image_observation_supplied_to_solver'] = True
            budget=trial_observation.get('optimization_budget',trial_observation.get('boundary_error_budget'))
            budget["max_wall_seconds"] = min(60., remaining - 15.)
            budget["max_objective_evaluations"] = remaining_evaluations
            seed_options=({"seed_entities":copy.deepcopy(seed_entities)} if seed_entities is not None else
                          {"seed_entities":graph["entities"]} if problem_graph is not graph else {})
            solution = solver(problem_graph, trial_constraints, output_dir=trial_dir / "solve",
                              source_observation=trial_observation,**seed_options)
            row.update(solver_status=solution.get("status"), solver_accepted=solution.get("accepted") is True,
                source_budget_search=copy.deepcopy((solution.get("diagnostics") or {}).get("source_budget_search")))
            used = (((solution.get("diagnostics") or {}).get("optimization_budget") or {}).get("objective_evaluations") or
                    ((solution.get("diagnostics") or {}).get("source_budget_search") or {}).get("objective_evaluations", 0))
            remaining_evaluations = max(0, remaining_evaluations - int(used))
            receipt["remaining_objective_evaluations"] = remaining_evaluations
            if (solution.get("accepted") is not True or
                    not _constraint_receipts_complete({"constraints": trial_constraints}, solution, graph)):
                row.update(status="rejected", reason="joint_exact_constraint_solve_failed")
                continue
            if time.monotonic() >= deadline:
                row.update(status="rejected", reason="shared_deadline_before_source_reverification")
                break
            candidate_graph = graph_from_solution(graph, baseline, solution)
            source_check = source_validator(image_path, document, baseline, graph, solution["entities"])
            topology_check = topology_validator(image_path, baseline, candidate_graph)
            row.update(original_source_check=source_check, topology_check=topology_check)
            if source_check.get("passed") is not True or topology_check.get("passed") is not True:
                row.update(status="rejected", reason="original_source_or_topology_check_failed")
                continue
            if supplied_binding_analyzer is not None:
                fresh = supplied_binding_analyzer(image_path, document, baseline, candidate_graph,
                                         trial_dir / "source-rebinding", use_api=False)
            else:
                fresh = _scoped_radius_rebinding(image_path, document, baseline, graph, candidate_graph,
                    inventory, required[len(constraints):] + [provisional], trial_dir / "source-rebinding")
            old_signatures = _constraint_signatures(graph, trial_constraints)
            new_signatures = _constraint_signatures(candidate_graph, fresh.get("constraints", []))
            if not _constraint_signatures(graph, required[len(constraints):] + [provisional]) <= new_signatures:
                row.update(status="rejected", reason="new_radius_not_independently_rebound")
                continue
            fresh_inventory = json.loads((trial_dir / "source-rebinding" / "binding-candidates.json").read_text(encoding="utf8"))
            carry = carried_binding_certificate(image_path, document, baseline, graph, candidate_graph,
                inventory, constraints, fresh_inventory, fresh, solution)
            row["carried_binding_certificate"] = carry
            row["fresh_binding_retained_all_required_constraints"] = old_signatures <= new_signatures
            if not carry["passed"] or _source_identity(image_path, document) != identity:
                row.update(status="rejected", reason="source_rebinding_dropped_or_contradicted_required_obligation")
                continue
            if time.monotonic() >= deadline:
                row.update(status="rejected", reason="shared_deadline_during_source_reverification")
                break
            # Only the required, numerically solved obligations become locks.
            # Newly observed extra bindings still require the final joint solve.
            required = trial_constraints
            result_graph = candidate_graph
            # The next independent hypothesis continues the current numerical
            # solution; the immutable problem graph and bounds remain fixed.
            seed_entities = copy.deepcopy(solution['entities'])
            row.update(status="source_verified_seed", binding_approved=False,
                       provisional_source_checks_passed=True,
                       final_online_binding_and_DXF_readback_pending=True)
            receipt["accepted_record_ids"].append(candidate["record_id"])
            receipt["admitted_constraints"] = copy.deepcopy(required)
            receipt["carried_binding_certificate"] = copy.deepcopy(carry)
            receipt["selected_attempt_directory"] = str(trial_dir)
            receipt["seed_solution_artifact"] = str(trial_dir / "solve" / "parametric-solve.json")
            receipt["seed_solution_sha256"] = _json_hash(solution)
            (trial_dir / "source-verified-seed.json").write_text(
                json.dumps(result_graph, ensure_ascii=False, indent=2), encoding="utf8")
        except (ValueError, TypeError, KeyError, ArithmeticError, OSError) as error:
            row.update(status="rejected", reason="bounded_hypothesis_verification_error",
                       error_type=type(error).__name__)
        finally:
            persist()
    receipt["status"] = "source_verified_seed" if receipt["accepted_record_ids"] else "no_admitted_hypothesis"
    persist()
    return result_graph, receipt


def _scoped_radius_rebinding(image_path, document, baseline, original_graph, solved_graph,
                             inventory, provisional_constraints, output_dir):
    """Remeasure provisional intervals; full source/API admission remains pending."""
    from .constraint_binding import _radius_primitive_feasibility
    from .topology_candidates import _affines
    from .topology import _primitive_distance
    from .ocr import canonical_records
    from PIL import Image
    with Image.open(image_path) as source_image:
        gray = np.asarray(source_image.convert("L"))
    records = canonical_records(document)
    original_transform, inverse, _ = _affines(original_graph, baseline)
    ppm = float(np.linalg.norm(original_transform([[1., 0.], [0., 0.]])[0] -
                               original_transform([[1., 0.], [0., 0.]])[1]))
    band = float(inventory.get('proposal_tolerance_px') or original_graph.get('proposal_tolerance_px') or 1.)
    candidates = inventory.get('all_candidates', inventory.get('candidates', []))
    current = {row['id']: row for row in solved_graph['entities']}
    fresh = copy.deepcopy(inventory)
    fresh['provisional_remeasurement_only'] = True
    fresh['full_fresh_source_detection_performed'] = False
    fresh['input_identity'] = _source_identity(image_path, document)
    for row in fresh.get('all_candidates', fresh.get('candidates', [])):
        row['local_reliable'] = False
    admitted, checks = [], []
    for constraint in provisional_constraints:
        previous = next((row for row in candidates if row.get('kind') == 'radius' and
            row.get('record_id') == constraint['record_id'] and row.get('entities') == constraint['entities']), None)
        if previous is None:
            continue
        leader = previous['evidence']['leader']
        entity = current[constraint['entities'][0]]
        interval = _radius_primitive_feasibility(entity, constraint['value'], leader,
            baseline, solved_graph, original_transform, band, gray=gray, records=records)
        gap = float(_primitive_distance(inverse([leader['arrowhead']['tip_px']]), entity)[0]) * ppm
        ownership = _carried_radius_finite_ownership(leader, constraint['record_id'], entity['id'],
            solved_graph, original_transform, inverse, band, {}, gray=gray,
            certified_radius=constraint['value'], verified_interval=interval)
        passed = interval.get('passed') is True and ownership.get('passed') is True
        check = {'record_id': constraint['record_id'], 'current_interval': interval,
                 'current_finite_arrow_distance_px': gap, 'passed': passed,
                 'current_finite_ownership': ownership,
                 'original_global_source_ownership_retained': True}
        checks.append(check)
        if passed:
            admitted.append(copy.deepcopy(constraint))
            for row in fresh.get('all_candidates', fresh.get('candidates', [])):
                if row.get('id') == previous['id']:
                    row['local_reliable'] = True
                    row['evidence']['whole_primitive_radius'] = interval
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fresh['provisional_radius_checks'] = checks
    (output / 'binding-candidates.json').write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding='utf8')
    return {'constraints': admitted, 'bindings': [], 'provisional_remeasurement_only': True}


class LockedSourceBindingProvider:
    """Pass bounded known-source equations without changing selection policy."""
    def __init__(self, provider, constraints):
        self.provider = provider
        self.constraints = copy.deepcopy(constraints)

    def select(self, image_path, topology_path, inventory):
        packet = copy.deepcopy(inventory)
        packet['locked_known_constraints'] = copy.deepcopy(self.constraints)
        return self.provider.select(image_path, topology_path, packet)


def merge_carried_bindings(original_inventory, fresh_inventory, fresh, carry):
    """Merge only a fully revalidated source certificate; preserve API abstentions."""
    if carry.get("passed") is not True:
        raise ValueError("unverified_source_binding_certificate")
    fresh_inventory=copy.deepcopy(fresh_inventory)
    fresh = copy.deepcopy(fresh)
    fresh['carried_binding_certificate'] = carry
    fresh['api_abstentions_retained_in_binding_decisions'] = True
    for index, row in enumerate(carry['constraints']):
        retained = copy.deepcopy(row)
        retained['id'] = f'carried_known_{index:03d}'
        retained.update(admission_method='carried_source_binding_certificate',current_API_reconfirmed=False)
        decision = {'constraint_id': retained['id'],
            'record_id': retained.get('record_id'), 'source': retained['source'], 'accepted': True,
            'admission_method': 'carried_source_binding_certificate', 'current_API_reconfirmed': False,
            'reason': None, 'prior_binding_source': retained.get('prior_binding_source')}
        if retained['source'] == 'source_geometry':
            original_relation = next(item for item in original_inventory.get('relations', [])
                if item.get('type') == retained['kind'] and
                set(item.get('entities', [])) == set(retained.get('entities', [])) and
                item.get('local_reliable') is True and (item.get('evidence') or {}).get('verified') is True)
            relation_id = f'carried_relation_{index:03d}'
            retained['id'] = 'k' + relation_id
            decision.update(constraint_id=retained['id'], relation_id=relation_id,
                prior_relation_id=original_relation['id'], evidence=copy.deepcopy(original_relation['evidence']))
            carried_relation=copy.deepcopy(original_relation)
            carried_relation.update(id=relation_id,admission_method='carried_source_binding_certificate',
                                    current_API_reconfirmed=False,fresh_detector_reconfirmed=False)
            fresh_inventory.setdefault('relations',[]).append(carried_relation)
        else:
            previous = next(item for item in original_inventory.get('all_candidates', [])
                if item.get('kind') == retained['kind'] and item.get('record_id') == retained['record_id'] and
                item.get('entities') == retained['entities'] and item.get('local_reliable') is True)
            carried_candidate = copy.deepcopy(previous)
            carried_candidate.update(id=f'carried_{retained["kind"]}_evidence_{index:03d}',
                admission_method='carried_source_binding_certificate', current_API_reconfirmed=False,
                fresh_detector_reconfirmed=False)
            if retained['kind']=='radius':
                check = next(item for item in carry['checks'] if item['constraint_id'] == row['id'])
                carried_candidate['evidence']['whole_primitive_radius'] = check['current_interval_check']
            fresh_inventory.setdefault('all_candidates', []).append(carried_candidate)
        fresh['constraints'].append(retained)
        fresh.setdefault('bindings', []).append(decision)
    return fresh,fresh_inventory


def certify_refined_radius_seed(image_path, document, baseline, original_graph, original_bindings,
                                original_inventory, candidate_graph, hypothesis_receipt, output_dir,
                                *, provider=None, use_api=False, binding_analyzer=None):
    """One fresh source/API pass; failed certification retains the original seed."""
    from .constraint_binding import analyze_constraint_bindings, radius_binding_coverage
    from .parametric_pipeline import _constraint_signatures
    analyzer = binding_analyzer or analyze_constraint_bindings
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt = {'status': 'rejected', 'full_fresh_source_binding_required': True,
               'all_known_constraints_preserved': False, 'native_solve_and_export_pending': True,
               'provisional_phase_elapsed_seconds': hypothesis_receipt.get('elapsed_seconds')}
    def finish(graph, bindings):
        receipt['formal_binding_elapsed_seconds'] = time.monotonic() - started
        receipt['overall_elapsed_seconds'] = receipt['formal_binding_elapsed_seconds'] + float(
            hypothesis_receipt.get('elapsed_seconds') or 0.)
        (output / 'radius-seed-certification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf8')
        return graph, bindings, receipt
    try:
        if hypothesis_receipt.get('status') != 'source_verified_seed':
            receipt['reason'] = 'no_source_verified_radius_seed'
            return finish(original_graph, original_bindings)
        if _source_identity(image_path, document) != hypothesis_receipt.get('input_identity'):
            receipt['reason'] = 'source_inputs_changed_before_formal_binding'
            return finish(original_graph, original_bindings)
        solution = json.loads(Path(hypothesis_receipt['seed_solution_artifact']).read_text(encoding='utf8'))
        if _json_hash(solution) != hypothesis_receipt.get('seed_solution_sha256'):
            receipt['reason'] = 'provisional_solution_artifact_changed'
            return finish(original_graph, original_bindings)
        required = hypothesis_receipt['admitted_constraints']
        known = original_bindings.get('constraints', [])
        new = [row for row in required if row.get('record_id') in hypothesis_receipt['accepted_record_ids']]
        wrapped = LockedSourceBindingProvider(provider, known) if provider is not None else None
        fresh = analyzer(image_path, document, baseline, candidate_graph, output / 'full-source-binding',
                         provider=wrapped, use_api=use_api)
        fresh_inventory = json.loads((output / 'full-source-binding' / 'binding-candidates.json').read_text(encoding='utf8'))
        receipt['provider'] = copy.deepcopy(fresh.get('provider', {}))
        receipt['new_radius_records'] = [row['record_id'] for row in new]
        if not _constraint_signatures(candidate_graph, new) <= _constraint_signatures(candidate_graph, fresh.get('constraints', [])):
            receipt['reason'] = 'new_radius_not_formally_rebound'
            return finish(original_graph, original_bindings)
        carry = carried_binding_certificate(image_path, document, baseline, original_graph, candidate_graph,
            original_inventory, known, fresh_inventory, fresh, solution)
        receipt['carried_binding_certificate'] = carry
        if carry.get('passed') is not True:
            receipt['reason'] = 'known_source_binding_certificate_failed'
            return finish(original_graph, original_bindings)
        fresh,fresh_inventory=merge_carried_bindings(original_inventory,fresh_inventory,fresh,carry)
        if not _constraint_signatures(original_graph, required) <= _constraint_signatures(candidate_graph, fresh['constraints']):
            receipt['reason'] = 'required_typed_constraint_regression'
            return finish(original_graph, original_bindings)
        if _source_identity(image_path, document) != hypothesis_receipt.get('input_identity'):
            receipt['reason'] = 'source_inputs_changed_during_formal_binding'
            return finish(original_graph, original_bindings)
        counts = fresh.setdefault('counts', {})
        bound = {row.get('record_id') for row in fresh['constraints']} - {None}
        counts.update(constraints=len(fresh['constraints']), bound_source_records=len(bound),
            bound_angle_records=sum(row['kind'] == 'angle' for row in fresh['constraints']),
            carried_source_constraints=len(carry['constraints']))
        if 'recognized_dimensions' in counts:
            counts['unbound_dimensions'] = counts['recognized_dimensions'] - len(bound)
        fresh['radius_binding_coverage'] = radius_binding_coverage(fresh_inventory, candidate_graph,
                                                                 fresh['constraints'], fresh['bindings'])
        fresh_inventory['carried_binding_certificate'] = carry
        fresh['inventory_artifact'] = str(output / 'binding-candidates-with-carried-certificates.json')
        Path(fresh['inventory_artifact']).write_text(json.dumps(fresh_inventory, ensure_ascii=False, indent=2), encoding='utf8')
        (output / 'constraint-bindings.json').write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding='utf8')
        receipt.update(status='accepted_seed', all_known_constraints_preserved=True,
                       fresh_new_radius_binding_complete=True, topology_changed=False)
        return finish(candidate_graph, fresh)
    except (ValueError, TypeError, KeyError, ArithmeticError, OSError) as error:
        receipt.update(reason='source_radius_formal_binding_failed', error_type=type(error).__name__)
        return finish(original_graph, original_bindings)

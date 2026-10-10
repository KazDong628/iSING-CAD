"""Bounded source-only candidate search bookkeeping and evidence ranking.

No reference counts, coordinates or evaluation scores are accepted here.
Numerical radius construction is deliberately absent from binding coverage.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import cv2
import numpy as np

from .reconstruction_feedback import geometry_fingerprint


def operation_fingerprint(parent_geometry, operation):
    payload = {"parent_geometry_sha256": parent_geometry,
               "operation": {key: operation.get(key) for key in ("action", "entity_ids", "record_id")}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def preflight_admissible(feedback):
    if (feedback.get("topology_source_validation") or {}).get("passed") is False:
        return False
    if (feedback.get("source_validation") or {}).get("passed") is False:
        return False
    if feedback.get("solver_status") in {"source_validation_failed", "preflight_failed", "preflight_not_run"}:
        return False
    return not feedback.get("constraint_count") or feedback.get("solver_accepted") is True


def unmet_source_angle_joint_operations(candidate, feedback, inventory, *, limit=3):
    """Find bounded, unfinished finite LINE edits at two arrow-backed ARC tails."""
    if not inventory or not isinstance(feedback.get("satisfied_record_ids"), list):
        return []
    from .annotation_line_support import angle_edit_evidence, propose_annotation_line_edits
    graph = candidate.get("graph") or {}
    grid = graph.get("source_grid_pitch_px")
    if type(grid) not in (int, float) or not math.isfinite(grid) or grid <= 0:
        return []
    result = []
    for operation in propose_annotation_line_edits(graph, inventory, limit=max(3, limit * 3)):
        if (len(operation.get("entity_ids") or []) != 2 or
                operation.get("record_id") in feedback["satisfied_record_ids"]):
            continue
        try:
            observation = angle_edit_evidence(graph, inventory, operation)
        except (ValueError, TypeError, KeyError):
            continue
        if observation.get("joint_radius_arrow_verified") is not True:
            continue
        requested = set(operation["entity_ids"])
        if any(row.get("entity_id") in requested and
               type(row.get("supported_span_px")) in (int, float) and
               math.isfinite(row["supported_span_px"]) and
               row["supported_span_px"] >= 4. * grid
               for row in observation.get("target_candidates", [])):
            result.append(operation)
            if len(result) >= limit:
                break
    return result


def _unmet_source_angle_joint(candidate, feedback, inventory):
    return bool(unmet_source_angle_joint_operations(candidate, feedback, inventory, limit=1))


def near_nominal_joint_bootstrap_refits(graph, inventory, feedback, *, limit=1):
    """Offer one unresolved exact-radius refit before a dependent angle joint.

    This only proposes source-backed operations. A new two-radius joint must be
    observed after execution, and the compound graph must pass the ordinary
    source, binding, solver, and publication gates before entering the beam.
    """
    if not isinstance(graph, dict) or not isinstance(inventory, list) or limit <= 0:
        return []
    if graph.get("units") != "mm" or not isinstance(feedback.get("satisfied_record_ids"), list):
        return []
    unresolved = {row.get("record_id") for row in
                  ((feedback.get("radius_binding_coverage") or {}).get("unresolved") or [])
                  if isinstance(row, dict)}
    if not unresolved:
        return []
    angle_ids = {row.get("record_id") for row in inventory if isinstance(row, dict)
                 and row.get("kind") == "angle"}
    entities = graph.get("entities") or []
    by_id = {row.get("id"): row for row in entities if isinstance(row, dict)}
    unsatisfied_angle_arcs = {target.get("entity_id")
        for observation in graph.get("angle_source_observations") or []
        if isinstance(observation, dict) and observation.get("verified") is True
        and observation.get("record_id") in angle_ids
        and observation.get("record_id") not in feedback["satisfied_record_ids"]
        and not any(target.get("whole_line_supported") is True and
                    by_id.get(target.get("entity_id"), {}).get("type") == "LINE"
                    for target in observation.get("target_candidates") or [])
        for target in observation.get("target_candidates") or []
        if isinstance(target, dict) and by_id.get(target.get("entity_id"), {}).get("type") == "ARC"}
    if not unsatisfied_angle_arcs or _unmet_source_angle_joint({"graph": graph}, feedback, inventory):
        return []
    records = {row.get("record_id"): row for row in inventory if isinstance(row, dict)
               and row.get("kind") == "radius"}
    support_by_entity = {}
    for row in graph.get("annotation_support") or []:
        if (isinstance(row, dict) and row.get("kind") == "radius" and
                row.get("status") == "candidate_supported"):
            support_by_entity.setdefault(row.get("candidate_entity_id"), []).append(row)
    grid = graph.get("source_grid_pitch_px")
    if type(grid) not in (int, float) or not math.isfinite(grid) or grid <= 0:
        return []
    positions = {row.get("id"): index for index, row in enumerate(entities)}
    ranked = []
    for entity_id, supports in support_by_entity.items():
        entity = by_id.get(entity_id) or {}
        if entity.get("type") != "ARC" or len(supports) != 1:
            continue
        support = supports[0]
        record_id = support.get("record_id")
        record = records.get(record_id) or {}
        nominal, fitted = record.get("nominal"), entity.get("radius")
        gap = support.get("target_gap_px")
        if (record_id not in unresolved or support.get("arrowhead_verified") is not True
                or type(nominal) not in (int, float) or type(fitted) not in (int, float)
                or type(gap) not in (int, float) or not all(map(math.isfinite, (nominal, fitted, gap)))
                or nominal <= 0 or fitted <= 0 or gap < 0 or gap > 2.25 * grid):
            continue
        mismatch = abs(math.log(fitted / nominal))
        if mismatch >= .20:
            continue
        try:
            chord = math.dist(entity["start"], entity["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(chord) or chord > 2 * nominal * 1.01:
            continue
        index = positions[entity_id]
        distance = min(min(abs(index - positions[target]), len(entities) - abs(index - positions[target]))
                       for target in unsatisfied_angle_arcs)
        operation = {"action": "refit_chain_as_annotated_arc", "entity_ids": [entity_id],
                     "record_id": record_id,
                     "evidence_tags": ["annotation_target", "source_boundary"]}
        ranked.append(((distance, gap / grid, mismatch, str(record_id)), operation))
    return [row[1] for row in sorted(ranked, key=lambda row: row[0])[:limit]]


def refresh_source_angle_observations(image_path, document, baseline, graph):
    """Rerun only the source-ink angle detector after source-only topology edits."""
    from pathlib import Path
    from .constraint_binding import (_angle_source_observations, _source_text_evidence,
                                     _source_transform, canonical_records)
    gray = cv2.imdecode(np.fromfile(str(Path(image_path)), np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("source_image_unreadable_for_angle_observation")
    records = [{"id": row["id"], "text": str(row.get("text", "")), "parsed": row["parsed"],
                "box": row.get("box"), "source_arrow_proposals": row.get("source_arrow_proposals", [])}
               for row in canonical_records(document)]
    for row in records:
        row["source_text_evidence"] = _source_text_evidence(gray, row)
    band = max(4., float(graph.get("proposal_tolerance_px", max(gray.shape) / 500)))
    observations = _angle_source_observations(gray, records, graph, _source_transform(baseline, graph), band)
    graph["angle_source_observations"] = observations
    return observations


def candidate_priority(candidate, feedback=None, *, preferred_id=None, annotation_inventory=None):
    """Rank certified radii, then feasible OCR topology, then source/solver fit."""
    feedback = feedback or candidate.get("constraint_feedback") or {}
    admitted = preflight_admissible(feedback)
    radius = len(set(feedback.get("verified_radius_record_ids") or [])) if admitted else 0
    records = len(set(feedback.get("satisfied_record_ids") or [])) if admitted else 0
    structural = int(feedback.get("satisfied_structural_constraint_count") or 0) if admitted else 0
    dof = feedback.get("remaining_shape_dof")
    if not admitted or type(dof) not in (int, float) or not math.isfinite(dof) or dof < 0:
        dof = math.inf
    validation = (feedback.get("source_validation") if admitted else feedback.get("topology_source_validation")) or feedback.get("topology_source_validation") or {}
    mask = validation.get("oracle_mask_validation") or {}
    error = mask.get("conservative_max_deviation_px")
    if type(error) not in (int, float) or not math.isfinite(error):
        error = (((candidate.get("source_residual") or {}).get("source_boundary_deviation_px") or {})
                 .get("conservative_upper_bound_px"))
    if type(error) not in (int, float) or not math.isfinite(error):
        error = math.inf
    joint_candidate = candidate
    edit_observations = candidate.get("edit_source_observations")
    if isinstance(edit_observations, dict):
        # Rank a possible edit using current source observations without
        # changing the graph whose binding/solve admitted this candidate.
        edit_graph = dict(candidate.get("graph") or {})
        for field in ("angle_source_observations", "radius_source_segment_hypotheses"):
            if field in edit_observations:
                edit_graph[field] = edit_observations[field]
        joint_candidate = {"graph": edit_graph}
    joint = admitted and _unmet_source_angle_joint(joint_candidate, feedback, annotation_inventory)
    # A feasible but untried OCR joint deserves a bounded search before an
    # extra generic structural relation. Among such joints, use the already
    # validated input-mask error to spend the first branch on the closer shape.
    return (not admitted, -radius, -records, -int(joint),
            float(error) if joint else math.inf, -structural, float(dof), float(error),
            candidate.get("id") != preferred_id, str(candidate.get("id") or ""))


def retain_distinct(candidates, *, width=3, preferred_id=None, annotation_inventory=None):
    if type(width) is not int or not 1 <= width <= 3:
        raise ValueError("topology_beam_width_must_be_1_to_3")
    result, seen = [], set()
    for candidate in sorted(candidates, key=lambda row: candidate_priority(
            row, preferred_id=preferred_id, annotation_inventory=annotation_inventory)):
        key = geometry_fingerprint(candidate["graph"])
        if key not in seen:
            result.append(candidate)
            seen.add(key)
        if len(result) >= width:
            break
    return result


def preflight_order(candidates, source_validations):
    """Give one source-valid angular line repair and combined edit early slots.

    This only schedules local checks; source validity and eventual constraint
    acceptance remain separate decisions. Other source-valid candidates then
    precede source-invalid ones, retaining order within each group. No provider
    or reference data is used.
    """
    def edit(candidate):
        return (((candidate.get("graph") or {}).get("source_evidence") or {})
                .get("topology_edit") or {})

    def source_valid(candidate):
        return (source_validations.get(candidate.get("id")) or {}).get("passed") is True

    angular = next((row for row in candidates if source_valid(row) and
                    edit(row).get("action") == "restore_annotated_line_support" and
                    edit(row).get("resegmentation_applied") is True and
                    edit(row).get("angle_support_record_ids")), None)
    combined = next((row for row in candidates if source_valid(row) and
                     edit(row).get("action") == "apply_nonoverlapping_edits"), None)
    preferred = [row for row in (angular, combined) if row is not None]
    preferred_ids = {id(row) for row in preferred}
    remaining = [row for row in candidates if id(row) not in preferred_ids]
    return [*preferred, *(row for row in remaining if source_valid(row)),
            *(row for row in remaining if not source_valid(row))]


class SearchBudget:
    def __init__(self, *, max_preflights=18, max_provider_calls=12, max_seconds=600., clock=None):
        if type(max_preflights) is not int or not 1 <= max_preflights <= 18:
            raise ValueError("topology_preflight_budget_must_be_1_to_18")
        if type(max_provider_calls) is not int or not 0 <= max_provider_calls <= 12:
            raise ValueError("topology_provider_budget_must_be_0_to_12")
        if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 600:
            raise ValueError("topology_time_budget_must_be_positive_and_at_most_600")
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.max_preflights, self.max_provider_calls, self.max_seconds = max_preflights, max_provider_calls, float(max_seconds)
        self.extension_seconds = 0.
        self.extension_reason = None
        self.preflights, self.provider_calls = 0, 0
        self.visited_operations = {}

    def remaining_seconds(self):
        return max(0., self.max_seconds + self.extension_seconds - (self.clock() - self.started))

    def extend_once(self, *, seconds=None, reason):
        """Extend this search clock once without resetting any count or visit state."""
        seconds=min(600.,self.max_seconds) if seconds is None else seconds
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or
                not 0 < seconds <= min(600.,self.max_seconds) or self.max_seconds + seconds > 1200.):
            raise ValueError("topology_time_extension_must_be_positive_and_at_most_original_budget_total_1200")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("topology_time_extension_requires_a_reason")
        if self.extension_seconds:
            return False
        self.extension_seconds = float(seconds)
        self.extension_reason = reason
        return True

    def preflight_block_reason(self):
        if self.remaining_seconds() <= 0:
            return "topology_search_time_budget_exhausted"
        if self.preflights >= self.max_preflights:
            return "topology_search_preflight_count_budget_exhausted"
        return None

    def reserve_preflight(self):
        if self.preflight_block_reason() is not None:
            return False
        self.preflights += 1
        return True

    def reserve_provider(self, role, parent_geometry):
        if self.provider_calls >= self.max_provider_calls or self.remaining_seconds() <= 0:
            return None
        self.provider_calls += 1
        return min(self.remaining_seconds(), 600.)

    def visit_operation(self, parent_geometry, operation):
        key = operation_fingerprint(parent_geometry, operation)
        if key in self.visited_operations:
            return False, key
        self.visited_operations[key] = {"parent_geometry_sha256": parent_geometry,
                                        "operation_sha256": key,
                                        "operation": {name: operation.get(name) for name in ("action", "entity_ids", "record_id")},
                                        "status": "pending"}
        return True, key

    def summary(self):
        return {"max_preflights": self.max_preflights, "preflights_used": self.preflights,
                "max_provider_calls": self.max_provider_calls, "provider_calls_reserved": self.provider_calls,
                "max_seconds": self.max_seconds, "original_max_seconds": self.max_seconds,
                "extension_seconds": self.extension_seconds,
                "total_max_seconds": self.max_seconds + self.extension_seconds,
                "extension_granted": bool(self.extension_seconds),
                "extension_reason": self.extension_reason,
                "elapsed_seconds": round(self.clock() - self.started, 3),
                "remaining_seconds": round(self.remaining_seconds(), 3),
                "time_limit_stops_new_work": True}

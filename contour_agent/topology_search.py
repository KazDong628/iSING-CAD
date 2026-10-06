"""Bounded source-only candidate search bookkeeping and evidence ranking.

No reference counts, coordinates or evaluation scores are accepted here.
Numerical radius construction is deliberately absent from binding coverage.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

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
    if feedback.get("solver_status") in {"source_validation_failed", "preflight_failed"}:
        return False
    return not feedback.get("constraint_count") or feedback.get("solver_accepted") is True


def candidate_priority(candidate, feedback=None, *, preferred_id=None):
    """Lexicographic evidence objective; lower is better, never object count."""
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
    return (not admitted, -radius, -records, -structural, float(dof), float(error),
            candidate.get("id") != preferred_id, str(candidate.get("id") or ""))


def retain_distinct(candidates, *, width=3, preferred_id=None):
    if type(width) is not int or not 1 <= width <= 3:
        raise ValueError("topology_beam_width_must_be_1_to_3")
    result, seen = [], set()
    for candidate in sorted(candidates, key=lambda row: candidate_priority(row, preferred_id=preferred_id)):
        key = geometry_fingerprint(candidate["graph"])
        if key not in seen:
            result.append(candidate)
            seen.add(key)
        if len(result) >= width:
            break
    return result


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
        self.preflights, self.provider_calls = 0, 0
        self.visited_operations = {}

    def remaining_seconds(self):
        return max(0., self.max_seconds - (self.clock() - self.started))

    def reserve_preflight(self):
        if self.preflights >= self.max_preflights or self.remaining_seconds() <= 0:
            return False
        self.preflights += 1
        return True

    def reserve_provider(self, role, parent_geometry):
        if self.provider_calls >= self.max_provider_calls or self.remaining_seconds() <= 0:
            return None
        self.provider_calls += 1
        return self.remaining_seconds()

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
                "max_seconds": self.max_seconds, "elapsed_seconds": round(self.clock() - self.started, 3),
                "remaining_seconds": round(self.remaining_seconds(), 3),
                "time_limit_stops_new_work": True}

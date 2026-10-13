"""Explicit user-scoped terminal simplification, independent of GT geometry.

This produces a separate partial-profile artifact. It cannot turn the original
full-source reconstruction contract into a passed contract or relax a gate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import ezdxf

from .ocr import canonical_records
from .radius_contract import exact_radius_checks
from .relation_contract import relation_checks
from .parametric_solver import (native_axis_relation_checks, native_measurement_checks,
                                native_source_nominal_angle_checks)


def source_annotation_identity(document):
    records = [{k: r.get(k) for k in ("id", "text", "box", "parsed")}
               for r in canonical_records(document)]
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf8")).hexdigest()


def _references(value, key=None):
    if isinstance(value, dict):
        return set().union(*(_references(v, k) for k, v in value.items())) if value else set()
    if isinstance(value, list):
        return set().union(*(_references(v, key) for v in value)) if value else set()
    return {value} if isinstance(value, str) and key in {
        "entities", "nodes", "entity_id", "other_entity_id", "node_id"} else set()


def _checks(entities, constraints, document=None):
    if any(c.get("kind") not in {"radius", "angle", "tangent", "horizontal", "vertical",
                                "distance", "distance_x", "distance_y"} for c in constraints):
        raise ValueError("unsupported_retained_source_constraint")
    checks = {
        "radii": exact_radius_checks(entities, constraints, dxf_document=document),
        "angles": native_source_nominal_angle_checks(entities, constraints, dxf_document=document),
        "tangencies": relation_checks(entities, constraints, dxf_document=document),
        "measurements": native_measurement_checks(entities, constraints, dxf_document=document),
        "source_axes": native_axis_relation_checks(entities, constraints, dxf_document=document),
    }
    kinds = {"radii": {"radius"}, "angles": {"angle"}, "tangencies": {"tangent"},
             "measurements": {"distance", "distance_x", "distance_y"},
             "source_axes": {"horizontal", "vertical"}}
    if any(checks[name].get("required_count") != sum(c["kind"] in selected for c in constraints)
           for name, selected in kinds.items()):
        raise ValueError("retained_source_constraint_not_covered_by_readback")
    return checks


def simplify_terminal_scope(entities, constraints, source_document, policy):
    """Replace a declared contiguous end-face chain by one exact axis LINE.

    Adjacent LINE supports are extended/trimmed to that axis. All retained
    sourced equations must still pass; an undeclared annotated feature cannot
    be erased. The unknown support position starts from the existing contour.
    """
    if policy.get("authority") != "explicit_user_request" or policy.get("mode") != "axis_line":
        raise ValueError("explicit_user_scope_required")
    if policy.get("source_annotation_sha256") != source_annotation_identity(source_document):
        raise ValueError("terminal_scope_annotation_identity_mismatch")
    selected = policy.get("entity_ids")
    deferred = policy.get("deferred_record_ids")
    if (not isinstance(selected, list) or not selected or len(selected) != len(set(selected)) or
            not isinstance(deferred, list) or len(deferred) != len(set(deferred))):
        raise ValueError("invalid_scope_selection")
    records = {r["id"]: r for r in canonical_records(source_document)}
    if any(r not in records or records[r]["parsed"]["kind"] != "radius" for r in deferred):
        raise ValueError("scope_requires_explicit_radius_record_ids")
    ids = [e["id"] for e in entities]
    if len(ids) != len(set(ids)) or not set(selected) <= set(ids) or len(selected) >= len(ids) - 1:
        raise ValueError("invalid_scope_entity_ids")
    starts = [i for i, e in enumerate(entities)
              if e["id"] in selected and entities[i - 1]["id"] not in selected]
    if len(starts) != 1:
        raise ValueError("scope_must_be_one_contiguous_terminal_chain")
    start = starts[0]
    rotated = copy.deepcopy(entities[start:] + entities[:start])
    chain, remainder = rotated[:len(selected)], rotated[len(selected):]
    if {e["id"] for e in chain} != set(selected):
        raise ValueError("scope_must_be_one_contiguous_terminal_chain")
    before, after = remainder[-1], remainder[0]
    lines = [e for e in chain if e["type"] == "LINE"]
    if not lines or before["type"] != "LINE" or after["type"] != "LINE":
        raise ValueError("terminal_scope_requires_LINE_body_and_LINE_neighbors")
    body = max(lines, key=lambda e: math.dist(e["start"], e["end"]))
    axis = policy.get("axis", "auto")
    if axis == "auto":
        axis = "vertical" if abs(body["end"][1] - body["start"][1]) >= abs(
            body["end"][0] - body["start"][0]) else "horizontal"
    if axis not in {"vertical", "horizontal"}:
        raise ValueError("invalid_terminal_axis")
    fixed = 0 if axis == "vertical" else 1
    coordinate = (body["start"][fixed] + body["end"][fixed]) / 2
    budget = policy.get("max_endpoint_adjustment_mm")
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("finite_positive_scope_adjustment_budget_required")

    def intersection(line):
        a, b = line["start"], line["end"]
        delta = b[fixed] - a[fixed]
        if abs(delta) <= 1e-12:
            raise ValueError("terminal_neighbor_parallel_to_replacement")
        t = (coordinate - a[fixed]) / delta
        result = [a[k] + t * (b[k] - a[k]) for k in (0, 1)]
        result[fixed] = coordinate
        return result

    p, q = intersection(before), intersection(after)
    adjustment = max(math.dist(before["end"], p), math.dist(after["start"], q))
    if adjustment > budget or math.dist(p, q) <= 1e-7:
        raise ValueError("terminal_scope_exceeds_declared_region_budget")
    if any(sum((e["end"][k] - e["start"][k]) * (end[k] - begin[k]) for k in (0, 1)) <= 0
           for e, begin, end in ((before, before["start"], p), (after, q, after["end"]))):
        raise ValueError("terminal_scope_would_reverse_neighbor")
    removed_ids = set(selected) - {chain[0]["id"]}
    removed_nodes = {e["end_node"] for e in chain[:-1]}
    dropped, retained = [], []
    for c in constraints:
        touches_removed = bool(_references(c) & (removed_ids | removed_nodes))
        excluded = c.get("record_id") in deferred
        if touches_removed and not excluded and not (c.get("kind") == "tangent" and c.get("record_id") is None):
            raise ValueError("terminal_scope_would_drop_undeclared_source_constraint")
        (dropped if touches_removed or excluded else retained).append(copy.deepcopy(c))
    before["end"], after["start"] = p, q
    replacement = {"id": chain[0]["id"], "type": "LINE", "start": p, "end": q,
                   "start_node": chain[0]["start_node"], "end_node": chain[-1]["end_node"],
                   "construction_basis": "explicit_user_scope_axis_closure"}
    projected = [replacement, *remainder]
    if any(math.dist(e["end"], projected[(i + 1) % len(projected)]["start"]) > 1e-7
           for i, e in enumerate(projected)):
        raise ValueError("terminal_scope_not_closed")
    checks = _checks(projected, retained)
    if not all(v["passed"] is True for v in checks.values()):
        raise ValueError("terminal_scope_retained_source_equations_failed")
    active_radii = {r for r, row in records.items() if row["parsed"]["kind"] == "radius"} - set(deferred)
    exact_ids = {r["record_id"] for r in checks["radii"]["checks"] if r["passed"]}
    receipt = {
        "schema_version": "user-terminal-profile-scope-v1", "policy": copy.deepcopy(policy),
        "axis": axis, "replacement_entity_id": replacement["id"], "removed_entity_ids": sorted(removed_ids),
        "deferred_annotations": [{"record_id": r, "text": records[r]["text"],
            "nominal": records[r]["parsed"]["nominal"], "status": "deferred_by_explicit_user_scope"}
            for r in deferred], "dropped_constraints": dropped,
        "max_endpoint_adjustment_mm": adjustment, "retained_source_checks": checks,
        "active_radius_count": len(active_radii), "exact_active_radius_count": len(active_radii & exact_ids),
        "all_active_radii_exact": active_radii <= exact_ids,
        "full_source_radius_count": len(active_radii) + len(deferred),
        "full_source_annotation_target_satisfied": False, "reference_accuracy_verified": False,
        "new_online_api_calls": 0, "new_optimization_performed": False,
        "axis_position_basis": "existing_prediction_LINE_initial_support; not newly dimension-solved",
        "scope": "Explicitly simplified terminal face only; original full-source results remain unchanged.",
    }
    return projected, retained, receipt


def export_terminal_scope(image_path, source_document, baseline, bindings, policy, output_dir):
    """Export to a fresh sibling directory and read back every retained equation."""
    from .parametric_pipeline import _export_chain, _write
    image_sha = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    if (policy.get("source_image_sha256") != image_sha or
            baseline.get("source_input_identity", {}).get("source_image_sha256") != image_sha or
            baseline.get("coordinate_system", {}).get("units") != "mm"):
        raise ValueError("terminal_scope_source_identity_or_units_mismatch")
    output = Path(output_dir)
    if output.exists():
        raise ValueError("terminal_scope_requires_fresh_output_directory")
    entities, retained, receipt = simplify_terminal_scope(
        baseline["entities"], bindings["constraints"], source_document, policy)
    parent = copy.deepcopy(baseline)
    for key in ("parameterization", "parameterization_snapshot_scope"):
        parent.pop(key, None)
    model = _export_chain(image_path, parent, entities, output,
        constraint_validation={"status": "retained_equations_rechecked_after_user_scope_projection",
                               "constraint_subset_satisfied": True, "new_solver_run": False})
    native_document = ezdxf.readfile(output / "drawing.dxf")
    native = _checks(entities, retained, native_document)
    if not all(v["passed"] is True for v in native.values()):
        raise ValueError("terminal_scope_native_equations_failed")
    receipt["retained_native_checks"] = native
    fixed = 0 if receipt["axis"] == "vertical" else 1
    native_line = list(native_document.modelspace())[0]
    receipt["axis_exact_in_model_and_native_dxf"] = bool(
        entities[0]["start"][fixed] == entities[0]["end"][fixed] and
        native_line.dxftype() == "LINE" and native_line.dxf.start[fixed] == native_line.dxf.end[fixed])
    if not receipt["axis_exact_in_model_and_native_dxf"]:
        raise ValueError("terminal_scope_axis_not_exact_in_dxf")
    receipt["output_dxf_sha256"] = hashlib.sha256((output / "drawing.dxf").read_bytes()).hexdigest()
    model.update(algorithm_version="explicit-user-terminal-axis-scope-v1",
        completion_class="user_scoped_partial_profile", reconstruction_scope=receipt,
        scope="User-requested simplified terminal LINE; remaining source constraints checked; full reconstruction not certified.")
    model["validation"].update(scope_adjusted=True, full_source_annotation_target_satisfied=False,
        validation_basis="explicit_user_scope_projection_and_retained_native_equations",
        retained_native_checks=native)
    _write(output / "scope-policy.json", policy)
    _write(output / "scope-audit.json", receipt)
    _write(output / "constraint-bindings.json", {"constraints": retained,
        "scope": receipt["scope"], "excluded_constraints": receipt["dropped_constraints"]})
    _write(output / "model.json", model)
    _write(output / "validation.json", model["validation"])
    return model, receipt

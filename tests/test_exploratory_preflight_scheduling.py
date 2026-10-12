"""Source-valid partial repairs must get depth within the existing search budget."""
from copy import deepcopy

import pytest

from contour_agent import parametric_pipeline as pipeline


def _candidate(state, length):
    points = [[0., 0.], [length, 0.], [length, 2.], [0., 2.]]
    return {"id": state, "graph": {"candidate_id": state, "repair_state": state,
        "units": "mm", "source_grid_pitch_px": 1., "ground_truth_used": False,
        "entities": [{"id": f"g{i:03d}", "stable_id": f"edge-{i}", "type": "LINE",
                      "start": point, "end": points[(i + 1) % 4]}
                     for i, point in enumerate(points)]}}


def _repair_fixture(monkeypatch, *, finish_accepted=True, preserve_incumbent=True,
                    source_valid=True, same_geometry=False):
    import contour_agent.constraint_binding as binding
    import contour_agent.planning_provider as planning
    import contour_agent.topology_editing as editing
    import contour_agent.topology_search as search

    clock, solves, parents, kernel_attempts = [0.], [], [], []
    monkeypatch.setattr(search.time, "monotonic", lambda: clock[0])
    base = _candidate("base", 2.)
    operations = [{"action": "restore_annotated_line_support" if index == 0 else "insert_annotated_fillet",
                   "entity_ids": [f"g{index:03d}"], "record_id": f"r-{index}"}
                  for index in range(4)]
    finish = {"action": "insert_annotated_fillet", "entity_ids": ["g001"], "record_id": "r-finish"}

    def propose(graph, inventory, **kwargs):
        return deepcopy(operations if graph["repair_state"] == "base" else
                        [finish] if graph["repair_state"] == "edit-a" else [])

    def execute(image, document, baseline, parent, bundle, requested, output):
        state = parent["graph"]["repair_state"]
        parents.append(state)
        edited, receipts = [], []
        for operation in requested:
            kernel_attempts.append((state, operation["record_id"]))
            if state == "base":
                index = int(operation["record_id"].split("-")[-1])
                name, length = ("edit-a", 3.) if index == 0 else (f"sibling-{index}", 4. + index)
            else:
                name, length = "ready-child", 10.
            child = _candidate(name, 2. if same_geometry else length)
            action = {**operation, "resegmentation_applied": True,
                      "angle_support_record_ids": [operation["record_id"]]}
            child["graph"]["source_evidence"] = {"topology_edit": action}
            edited.append(child)
            receipts.append({"operation": deepcopy(operation), "candidate_id": name,
                             "status": "accepted_as_candidate"})
        if state == "base" and len(requested) == len(operations):
            combined = _candidate("edit-all", 2. if same_geometry else 9.)
            combined["graph"]["source_evidence"] = {"topology_edit": {
                "action": "apply_nonoverlapping_edits", "operations": deepcopy(requested)}}
            edited.append(combined)
        return edited, {"operations": receipts}

    def source_check(image, baseline, graph):
        passed = source_valid or graph["repair_state"] == "base"
        return {"passed": passed, "reasons": [] if passed else ["source_stroke_support_degraded"],
                "max_deviation_px": .2, "maximum_allowed_deviation_px": .5}

    def analyze(image, document, baseline, graph, output, **kwargs):
        state = graph["repair_state"]
        records = ["r-base"] if state == "base" else ["r-middle"]
        if state == "ready-child":
            records += ["r-final"] + (["r-base"] if preserve_incumbent else [])
        return {"constraints": [{"kind": "distance", "record_id": name,
                                  "entities": ["g000"], "value": 2.} for name in records]}

    def solve(solver, graph, constraints, baseline, output, *, budget_seconds=None, **kwargs):
        state = graph["repair_state"]
        cost = 100. if state == "base" else 80. if state == "ready-child" else 120.
        clock[0] += min(cost, budget_seconds)
        solves.append((state, clock[0]))
        accepted = state == "base" or (state == "ready-child" and finish_accepted)
        return {"status": "accepted" if accepted else "source_budget_search_failed",
                "accepted": accepted,
                "entities" if accepted else "candidate_entities": deepcopy(graph["entities"]),
                "constraints": [{**row, "passed": accepted} for row in constraints],
                "diagnostics": {"remaining_shape_dof": 4 if state == "base" else 2}}

    def evaluate(rows, **kwargs):
        return {"admissible_candidate_ids": [row["id"] for row in rows],
                "evaluated": [{"candidate_id": row["id"], "admissible": True, "score": .9,
                    "metrics": {"entity_count": 4, "source_boundary_support": .95,
                                "unsupported_primitive_count": 0}} for row in rows]}

    monkeypatch.setattr(editing, "propose_annotation_arc_edits", propose)
    monkeypatch.setattr(editing, "execute_topology_edits", execute)
    monkeypatch.setattr(planning, "evaluate_candidates", evaluate)
    monkeypatch.setattr(binding, "analyze_constraint_bindings", analyze)
    monkeypatch.setattr(pipeline, "_topology_source_validation", source_check)
    monkeypatch.setattr(pipeline, "_solve_with_source_observation", solve)
    monkeypatch.setattr(pipeline, "_solved_source_validation", lambda *args: {
        "passed": True, "reasons": [], "max_deviation_px": .2, "maximum_allowed_deviation_px": .5})
    monkeypatch.setattr(pipeline, "_preflight_input_identity", lambda *args: {
        "schema_version": "synthetic-preflight", "reference_geometry_used": False})
    monkeypatch.setattr(pipeline, "_record_fillet_tangent_contract", lambda *args: None)
    return base, clock, solves, parents, kernel_attempts


@pytest.mark.parametrize("finish_accepted", [False, True])
def test_two_step_source_repair_gets_depth_under_original_600_seconds(tmp_path, monkeypatch, finish_accepted):
    base, clock, solves, parents, attempts = _repair_fixture(monkeypatch, finish_accepted=finish_accepted)
    final, report = pipeline._topology_edit_loop("unused", {}, {}, base,
        {"annotation_inventory": [], "candidates": [base]}, tmp_path, max_rounds=2)
    assert len(report["rounds"]) == 2
    first = report["rounds"][0]["branches"][0]
    assert first["completed_source_preflights"] == 2
    assert [row["status"] for row in first["preflight_queue"]] == [
        "rejected", "rejected", "deferred_not_run", "deferred_not_run", "deferred_not_run"]
    assert first["deferred_for_exploration_after_candidate_id"].endswith("edit-a")
    assert first["deferred_for_next_round_after_candidate_id"] is None
    assert report["rounds"][0]["final_candidate_id"] == "base"
    assert parents[:2] == ["base", "edit-a"]
    assert any(state == "ready-child" and elapsed < 600. for state, elapsed in solves)
    assert final["graph"]["repair_state"] == ("ready-child" if finish_accepted else "base")
    assert not final.get("diagnostic_only")
    # Deferred siblings of the original parent remain eligible on its later visit.
    assert attempts.count(("base", "r-1")) == 2
    assert report["budget"]["max_seconds"] == 600.
    assert report["budget"]["extension_seconds"] == 0.
    assert clock[0] <= 600. and report["budget"]["preflights_used"] <= 18
    assert report["max_rounds"] == 2 and report["beam_width"] == 3
    assert all(row["validation"]["maximum_allowed_deviation_px"] == .5
               for row in report["final_source_rechecks"])


def test_failed_intermediate_cannot_erase_formal_incumbent_constraints(tmp_path, monkeypatch):
    base, _, solves, _, _ = _repair_fixture(monkeypatch, preserve_incumbent=False)
    bundle = {"annotation_inventory": [], "candidates": [base]}
    final, _ = pipeline._topology_edit_loop("unused", {}, {}, base, bundle, tmp_path, max_rounds=2)
    assert any(state == "ready-child" for state, _ in solves)
    child = next(row for row in bundle["candidates"] if row["graph"]["repair_state"] == "ready-child")
    assert child["constraint_regression_gate"]["passed"] is True
    gate = child["incumbent_constraint_regression_gate"]
    assert gate["passed"] is False and gate["lost_record_ids"] == ["r-base"]
    assert "previously_bound_source_records_lost" in gate["reasons"]
    assert final["id"] == "base" and final["constraint_feedback"]["bound_record_ids"] == ["r-base"]


@pytest.mark.parametrize("source_valid,same_geometry", [(False, False), (True, True)])
def test_no_exploration_deferral_without_different_source_valid_geometry(
        tmp_path, monkeypatch, source_valid, same_geometry):
    base, _, _, _, _ = _repair_fixture(monkeypatch, source_valid=source_valid, same_geometry=same_geometry)
    _, report = pipeline._topology_edit_loop("unused", {}, {}, base,
        {"annotation_inventory": [], "candidates": [base]}, tmp_path, max_rounds=2)
    assert report["rounds"][0]["deferred_preflight_candidate_ids"] == []


def test_last_round_does_not_defer_for_unavailable_depth(tmp_path, monkeypatch):
    base, _, _, _, _ = _repair_fixture(monkeypatch)
    _, report = pipeline._topology_edit_loop("unused", {}, {}, base,
        {"annotation_inventory": [], "candidates": [base]}, tmp_path, max_rounds=1)
    assert report["rounds"][0]["deferred_preflight_candidate_ids"] == []
    assert all(row["status"] != "deferred_not_run" for row in report["rounds"][0]["preflight_queue"])


def test_no_exploration_deferral_when_beam_has_no_diagnostic_slot(tmp_path, monkeypatch):
    base, _, _, _, _ = _repair_fixture(monkeypatch)
    _, report = pipeline._topology_edit_loop("unused", {}, {}, base,
        {"annotation_inventory": [], "candidates": [base]}, tmp_path, max_rounds=2, beam_width=1)
    assert report["rounds"][0]["deferred_preflight_candidate_ids"] == []


def test_final_source_recheck_cannot_promote_failed_exploratory_geometry(tmp_path, monkeypatch):
    base, _, _, _, _ = _repair_fixture(monkeypatch, finish_accepted=False)
    original = pipeline._topology_source_validation
    base_checks = []

    def recheck(image, baseline, graph):
        if graph["repair_state"] == "base":
            base_checks.append(graph["candidate_id"])
            if len(base_checks) > 1:
                return {"passed": False, "reasons": ["source_stroke_support_degraded"]}
        return original(image, baseline, graph)

    monkeypatch.setattr(pipeline, "_topology_source_validation", recheck)
    final, report = pipeline._topology_edit_loop("unused", {}, {}, base,
        {"annotation_inventory": [], "candidates": [base]}, tmp_path, max_rounds=2)
    assert len(base_checks) == 2
    assert report["stop_reason"] == "final_source_recheck_failed"
    assert any(row["candidate_id"] != "base" and row["validation"]["passed"] is True
               for row in report["final_source_rechecks"])
    assert final["id"] == "base" and not final.get("diagnostic_only")

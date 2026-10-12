"""A failed or unstarted numerical preflight must retain its actual stage."""
from copy import deepcopy
import json

import pytest

from contour_agent import parametric_pipeline as pipeline, topology_search
from test_parametric_solver import rectangle


def _setup(monkeypatch):
    now = [0.]
    real_budget = topology_search.SearchBudget
    monkeypatch.setattr(topology_search, "SearchBudget",
                        lambda **kwargs: real_budget(**kwargs, clock=lambda: now[0]))
    monkeypatch.setattr(pipeline, "_topology_source_validation", lambda *args: {"passed": True, "reasons": []})
    monkeypatch.setattr(pipeline, "_record_fillet_tangent_contract", lambda *args: None)
    def unchanged(image, document, baseline, parent, bundle, output, **kwargs):
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
                        "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
                        "editor": {}, "evaluator": {}, "trusted_candidate_ids": []}
    monkeypatch.setattr(pipeline, "_topology_edit_stage", unchanged)
    graph = rectangle()
    graph["candidate_id"] = "base"
    candidate = {"id": "base", "graph": graph}
    binding = {"status": "completed", "constraints": [{"id": "width", "kind": "distance_x",
        "nodes": ["v0", "v1"], "entities": [], "record_id": "r_width", "value": 10.,
        "source": "ocr_local_binding"}], "counts": {"unbound_dimensions": 2}}
    return now, candidate, binding


def _run(tmp_path, candidate):
    final, _ = pipeline._topology_edit_loop("unused", {}, {}, candidate,
        {"candidates": [candidate], "annotation_inventory": []}, tmp_path,
        max_rounds=1, beam_width=1, max_seconds=10.)
    directory = next((tmp_path / "topology-iterations" / "constraint-preflight").iterdir())
    receipt = json.loads((directory / "reconstruction-feedback.json").read_text("utf8"))
    failure = json.loads((directory / "preflight-failure.json").read_text("utf8"))
    assert receipt == final["constraint_feedback"]
    assert receipt["preflight_failure"] == failure
    assert not topology_search.preflight_admissible(receipt)
    assert receipt["solver_accepted"] is False
    assert receipt["satisfied_record_ids"] == []
    assert failure["infeasibility_proven"] is False
    assert not (directory / "preflight-input-identity.json").exists()
    assert not (directory / "topology.json").exists()
    return directory, receipt, failure


def test_binding_exhausts_budget_before_solver_retains_real_binding_receipt(tmp_path, monkeypatch):
    now, candidate, binding = _setup(monkeypatch)
    original = deepcopy(candidate["graph"])
    calls = []
    def bind(image, document, baseline, graph, output, **kwargs):
        pipeline._write(output / "constraint-bindings.json", binding)
        now[0] = 11.
        return deepcopy(binding)
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings", bind)
    monkeypatch.setattr(pipeline, "_solve_with_source_observation", lambda *args, **kwargs: calls.append(True))
    directory, receipt, failure = _run(tmp_path, candidate)
    assert calls == []
    assert candidate["graph"] == original
    assert json.loads((directory / "constraint-bindings.json").read_text("utf8")) == binding
    assert receipt["binding_status"] == "completed"
    assert receipt["constraint_count"] == 1 and receipt["bound_record_ids"] == ["r_width"]
    assert receipt["solver_status"] == "preflight_not_run"
    assert receipt["preflight_status"] == "not_run_budget"
    assert receipt["preflight_skip_reason"] == "topology_search_time_budget_exhausted"
    assert receipt["source_validation"]["passed"] is None
    assert receipt["topology_source_validation"]["passed"] is True
    assert failure == {"category": "budget_exhausted", "stage": "budget_before_solver",
        "error_type": "TimeoutError", "binding_completed": True, "solver_started": False,
        "solver_completed": False, "infeasibility_proven": False, "exception_message_recorded": False}


@pytest.mark.parametrize("kind", ["solver", "invalid_binding", "binding", "post_solve"])
def test_preflight_records_failure_stage_without_leaking_exception_message(tmp_path, monkeypatch, kind):
    _, candidate, binding = _setup(monkeypatch)
    private_message = "private-key-and-local-path-must-never-be-logged"
    def bind(*args, **kwargs):
        if kind == "binding": raise OSError(private_message)
        return {"constraints": "malformed"} if kind == "invalid_binding" else deepcopy(binding)
    def solve(solver, graph, constraints, baseline, output, **kwargs):
        if kind == "solver": raise RuntimeError(private_message)
        return {"status": "accepted", "accepted": True, "entities": graph["entities"], "constraints": constraints}
    def validate(*args):
        raise ValueError(private_message)
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings", bind)
    monkeypatch.setattr(pipeline, "_solve_with_source_observation", solve)
    monkeypatch.setattr(pipeline, "_solved_source_validation", validate)
    directory, receipt, failure = _run(tmp_path, candidate)
    expected = {"solver": ("solver_failed", "solver", "RuntimeError"),
                "invalid_binding": ("invalid_binding", "binding_validation", "ValueError"),
                "binding": ("binding_failed", "binding", "OSError"),
                "post_solve": ("post_solve_failed", "post_solve_validation", "ValueError")}[kind]
    assert (failure["category"], failure["stage"], failure["error_type"]) == expected
    assert receipt["solver_status"] == "preflight_failed"
    completed_binding = kind in {"solver", "post_solve"}
    assert failure["binding_completed"] is completed_binding
    assert failure["solver_started"] is completed_binding
    assert failure["solver_completed"] is (kind == "post_solve")
    assert receipt["binding_status"] == ("completed" if completed_binding else "failed")
    assert receipt["bound_record_ids"] == (["r_width"] if completed_binding else [])
    for path in directory.glob("*.json"):
        assert private_message not in path.read_text("utf8")

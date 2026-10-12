import json

from contour_agent.model_transcript import build_model_transcript


def _transcript(tmp_path, documents):
    output = tmp_path / "job"
    output.mkdir()
    for name, document in documents.items():
        (output / name).write_text(json.dumps(document), encoding="utf8")
    return build_model_transcript({"artifact_directory": str(output)}, tmp_path)


def test_rejected_solver_subset_is_not_assigned_to_published_topology(tmp_path):
    result = _transcript(tmp_path, {
        "model.json": {"parameterization": {"status": "source_topology_exported",
            "accepted": False, "constraint_subset_accepted": False}},
        "parametric-stage.json": {"status": "candidate_rejected", "accepted": False},
        "parametric-solution.json": {"status": "source_budget_search_failed", "accepted": False,
            "diagnostics": {"remaining_shape_dof": 29},
            "constraints": [{"id": "k1", "kind": "radius", "passed": True, "actual": 176}]},
        "constraint-bindings.json": {"counts": {"recognized_dimensions": 37,
            "bound_source_records": 11, "constraints": 20}},
    })["parameterization"]
    assert result["scope"] == "published_dxf"
    assert result["status"] == "source_topology_exported"
    assert result["counts"]["bound_source_records"] == 0
    assert result["counts"]["unbound_dimensions"] == 37
    assert result["constraints"] == [] and result["diagnostics"] == {}
    assert result["attempted"]["counts"]["bound_source_records"] == 11
    assert result["attempted"]["diagnostics"]["remaining_shape_dof"] == 29
    assert result["attempted"]["constraints"][0]["actual"] == 176


def test_preserved_publication_uses_committed_counts_not_later_attempt(tmp_path):
    result = _transcript(tmp_path, {
        "model.json": {"parameterization": {
            "status": "completed_with_unresolved_attributes", "accepted": False,
            "constraint_subset_accepted": True,
            "binding_counts": {"recognized_dimensions": 35, "bound_source_records": 7},
            "solver": {"accepted": True, "underconstrained": True,
                "diagnostics": {"remaining_shape_dof": 30}},
            "solver_constraint_checks": [{"id": "formal", "passed": True, "actual": 135}],
        }},
        "parametric-solution.json": {"status": "rejected", "accepted": False,
            "diagnostics": {"remaining_shape_dof": 19},
            "constraints": [{"id": "attempt", "passed": True, "actual": 134.99}]},
        "constraint-bindings.json": {"counts": {"recognized_dimensions": 35, "bound_source_records": 16}},
    })["parameterization"]
    assert result["constraint_subset_accepted"] and not result["pipeline_accepted"]
    assert result["counts"]["bound_source_records"] == 7
    assert result["diagnostics"]["remaining_shape_dof"] == 30
    assert result["constraints"][0]["id"] == "formal"
    assert result["attempted"]["counts"]["bound_source_records"] == 16


def test_public_preflight_failure_keeps_phase_without_exception_text(tmp_path):
    result = _transcript(tmp_path, {
        "topology-iterations.json": {"rounds": [{"round": 1, "feedback": {
            "binding_status": "completed", "preflight_status": "not_run_budget",
            "preflight_failure": {"category": "budget_exhausted", "stage": "before_solver",
                "error_type": "TimeoutError", "binding_completed": True,
                "solver_started": False, "solver_completed": False, "infeasibility_proven": False,
                "exception_message_recorded": False, "exception": "DO-NOT-EXPOSE"},
        }}]},
    })
    feedback = result["iterations"]["rounds"][0]["feedback"]
    assert feedback["binding_status"] == "completed"
    assert feedback["preflight_status"] == "not_run_budget"
    assert feedback["preflight_failure"]["solver_started"] is False
    assert feedback["preflight_failure"]["infeasibility_proven"] is False
    assert "DO-NOT-EXPOSE" not in json.dumps(result)

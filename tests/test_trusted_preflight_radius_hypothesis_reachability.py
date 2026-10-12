"""The trusted-parent path must not bypass bounded source-radius refinement."""
from copy import deepcopy
import json

import pytest

from contour_agent import source_radius_hypotheses
from contour_agent.parametric_pipeline import (CORE, refine_parametric,
    _certify_preflight_checkpoint, _final_checkpoint_replacement_gate)
from test_preflight_publication_guard import setup_case


def test_trusted_parent_gets_one_bounded_radius_attempt_with_known_source_locks(tmp_path, monkeypatch):
    image, baseline, graph, output, saved, binding_calls = setup_case(tmp_path, monkeypatch)
    original = deepcopy(graph)
    calls = []

    def refine(image_path, document, source_baseline, current_graph, bindings, inventory, directory, **kwargs):
        calls.append({"graph": deepcopy(current_graph), "constraints": deepcopy(bindings["constraints"]),
                      "baseline": deepcopy(source_baseline)})
        return current_graph, {"status": "no_admitted_hypothesis", "accepted_record_ids": [],
                               "maximum_attempts": 2, "maximum_wall_seconds": 90.}

    monkeypatch.setattr(source_radius_hypotheses, "refine_source_owned_radius_hypotheses", refine)
    model, stage = refine_parametric(image, {}, baseline, output, use_api=True,
                                    frozen_topology=graph, trusted_preflight_dir=saved)
    assert len(calls) == 1  # Regression: the old trusted branch returned before this phase.
    assert {row["record_id"] for row in calls[0]["constraints"]} == {"r1"}
    assert calls[0]["graph"] == original
    assert calls[0]["baseline"] == baseline
    assert graph == original
    assert binding_calls == [False, True]
    assert stage["binding_selection"]["status"] == "certified_preflight_retained"
    assert model["validation"]["exact_radius_validation"]["checks"][0]["dxf_radius"] == 5.
    assert all((output / name).is_file() for name in CORE)


@pytest.mark.parametrize("new_source_counterevidence", [False, True])
def test_failed_seed_certification_preserves_original_parent_policy(
        tmp_path, monkeypatch, new_source_counterevidence):
    image, baseline, graph, output, saved, binding_calls = setup_case(
        tmp_path, monkeypatch, changed_evidence=new_source_counterevidence)
    original = deepcopy(graph)
    refinement_calls, certification_calls = [], []

    def refine(image_path, document, source_baseline, current_graph, bindings, inventory, directory, **kwargs):
        refinement_calls.append(deepcopy(bindings["constraints"]))
        proposed = deepcopy(current_graph)
        proposed["candidate_id"] = "unapproved-provisional-seed"
        return proposed, {"status": "source_verified_seed", "accepted_record_ids": ["r2"],
                          "numerical_seed_only": True, "all_dimensions_verified": False}

    def certify(image_path, document, source_baseline, original_graph, original_bindings,
                original_inventory, candidate_graph, receipt, directory, **kwargs):
        certification_calls.append(candidate_graph["candidate_id"])
        return original_graph, original_bindings, {"status": "rejected", "reason": "fresh_new_arrow_conflict"}

    monkeypatch.setattr(source_radius_hypotheses, "refine_source_owned_radius_hypotheses", refine)
    monkeypatch.setattr(source_radius_hypotheses, "certify_refined_radius_seed", certify)
    model, stage = refine_parametric(image, {}, baseline, output, use_api=True,
                                    frozen_topology=graph, trusted_preflight_dir=saved)
    assert len(refinement_calls) == 1
    assert certification_calls == ["unapproved-provisional-seed"]
    assert graph == original
    assert binding_calls == [False, True]
    guard = json.loads((output / "binding-publication-guard.json").read_text())["replacement_gate"]
    assert not guard["replace_checkpoint"]
    selected = json.loads((output / "constraint-bindings.json").read_text())
    if new_source_counterevidence:
        assert not guard["retain_checkpoint"]
        assert selected["constraints"] == []
        assert model["algorithm_version"] == "source-topology-draft-v1"
    else:
        assert guard["retain_checkpoint"]
        assert stage["binding_selection"]["status"] == "certified_preflight_retained"
        assert {row["record_id"] for row in selected["constraints"]} == {"r1"}
        assert model["validation"]["exact_radius_validation"]["checks"][0]["dxf_radius"] == 5.
    assert "r2" not in {row.get("record_id") for row in selected["constraints"]}


def _transition():
    return {"status": "accepted_seed", "all_known_constraints_preserved": True,
            "fresh_new_radius_binding_complete": True, "topology_changed": False,
            "carried_binding_certificate": {"passed": True},
            "input_identity_verified_against_checkpoint": True}


@pytest.mark.parametrize("failure", ["source_identity", "old_binding_proof", "new_radius_proof", "topology_changed"])
def test_changed_interval_evidence_needs_complete_controlled_source_transition(tmp_path, monkeypatch, failure):
    image, baseline, graph, output, saved, _ = setup_case(tmp_path, monkeypatch)
    trusted, _ = _certify_preflight_checkpoint(image, {}, baseline, graph, saved, output / "cert")
    attempt = deepcopy(trusted)
    certificate = _transition()
    if failure == "source_identity":
        certificate["input_identity_verified_against_checkpoint"] = False
    elif failure == "old_binding_proof":
        certificate["carried_binding_certificate"]["passed"] = False
    elif failure == "new_radius_proof":
        certificate["fresh_new_radius_binding_complete"] = False
    else:
        certificate["topology_changed"] = True
    attempt["source_binding_transition_certificate"] = certificate
    guard = _final_checkpoint_replacement_gate(trusted, attempt, evidence_sha256="remeasured-finite-interval")
    assert not guard["replace_checkpoint"]
    assert "independent_source_evidence_changed" in guard["reasons"]


@pytest.mark.parametrize("failure", ["missing_numeric_receipt", "source_budget", "native_readback", "old_exact_value"])
def test_controlled_transition_does_not_bypass_numeric_source_or_native_gates(tmp_path, monkeypatch, failure):
    image, baseline, graph, output, saved, _ = setup_case(tmp_path, monkeypatch)
    trusted, _ = _certify_preflight_checkpoint(image, {}, baseline, graph, saved, output / "cert")
    attempt = deepcopy(trusted)
    attempt["source_binding_transition_certificate"] = _transition()
    if failure == "missing_numeric_receipt":
        attempt["solution"]["constraints"] = []
        reason = "final_constraint_receipts_incomplete"
    elif failure == "source_budget":
        attempt["source_validation"] = {"passed": False, "reasons": ["original_source_budget_failed"]}
        reason = "original_source_budget_failed"
    elif failure == "native_readback":
        attempt["model"]["validation"]["exact_radius_validation"]["dxf_readback_performed"] = False
        reason = "final_native_radius_and_dxf_readback_not_verified"
    else:
        attempt["bindings"]["constraints"][0]["value"] = 6.
        attempt["solution"]["constraints"][0]["value"] = 6.
        reason = "previously_verified_constraint_set_lost"
    guard = _final_checkpoint_replacement_gate(trusted, attempt, evidence_sha256="remeasured-finite-interval")
    assert not guard["replace_checkpoint"]
    assert guard["retain_checkpoint"]
    assert reason in guard["reasons"]


def test_certified_source_transition_can_remeasure_interval_without_changing_known_equation(tmp_path, monkeypatch):
    image, baseline, graph, output, saved, _ = setup_case(tmp_path, monkeypatch)
    trusted, _ = _certify_preflight_checkpoint(image, {}, baseline, graph, saved, output / "cert")
    attempt = deepcopy(trusted)
    attempt["source_binding_transition_certificate"] = _transition()
    guard = _final_checkpoint_replacement_gate(trusted, attempt, evidence_sha256="remeasured-finite-interval")
    assert guard["replace_checkpoint"]
    assert guard["source_evidence_remeasured_with_certified_known_bindings"]
    assert guard["same_independent_source_evidence"] is False

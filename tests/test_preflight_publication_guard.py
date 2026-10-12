"""A repeated model selection cannot replace a certified subset with less evidence."""
from copy import deepcopy
import json

import pytest

from contour_agent.parametric_pipeline import (
    CORE, _certify_preflight_checkpoint, _preflight_input_identity,
    _final_checkpoint_replacement_gate, _constraint_receipts_complete, refine_parametric)
from contour_agent.reconstruction_feedback import geometry_fingerprint, reconstruction_feedback
from test_parametric_pipeline import prepared, supported_graph


def setup_case(tmp_path, monkeypatch, *, changed_evidence=False, elapsed_seconds_change=False):
    image, baseline, output = prepared(tmp_path)
    graph = supported_graph(image, baseline)
    graph["candidate_id"] = "frozen-synthetic"
    graph["annotation_support"] = [{"record_id": "r1", "kind": "radius", "nominal": 5.,
                                    "candidate_entity_id": "g002"}]
    constraint = {"id": "kr1", "kind": "radius", "record_id": "r1", "entities": ["g002"],
                  "nodes": [], "value": 5., "source_arrow_verified": True, "source": "ocr_local_binding"}
    coverage = {"recognized_count": 1, "required_count": 1, "bound_count": 1,
                "confirmed_arrow_records": ["r1"], "all_radius_records_resolved": True,
                "all_confirmed_arrows_bound": True, "unknown_arrow_records": [], "unresolved": []}
    bindings = {"constraints": [constraint], "bindings": [], "provider": {"status": "disabled", "network_requests": 0},
                "radius_binding_coverage": coverage, "counts": {"constraints": 1}, "issues": []}
    solution = {"accepted": True, "status": "accepted", "entities": deepcopy(graph["entities"]),
                "constraints": [{**constraint, "passed": True}], "units": "mm", "underconstrained": True,
                "validation": {"passed": True, "constraint_subset_satisfied": True},
                "diagnostics": {"remaining_shape_dof": 10}}
    feedback = reconstruction_feedback(graph, bindings, solution)
    feedback["source_validation"] = {"passed": True}
    checkpoint = tmp_path / "saved-preflight"
    checkpoint.mkdir()
    for name, data in [("constraint-bindings.json", bindings), ("parametric-solve.json", solution),
                       ("reconstruction-feedback.json", feedback),
                       ("preflight-input-identity.json", _preflight_input_identity(image, {}, baseline, graph))]:
        (checkpoint / name).write_text(json.dumps(data))
    inventory = {"units": "mm", "source_image_sha256": graph["source_sha256"],
                 "all_records": [{"id": "r1", "parsed": {"kind": "radius", "nominal": 5.}}],
                 "all_candidates": [{"id": "c1", "record_id": "r1", "entities": ["g002"], "local_reliable": True}],
                 "source_arrow_ownership": {"ocr_local_candidate_searches": [
                     {"record_id": "r1", "status": "completed", "time_limit_seconds": 8., "elapsed_seconds": .172}]}}
    calls = []
    def binding(image, doc, base, current_graph, directory, **kwargs):
        calls.append(kwargs.get("use_api"))
        directory.mkdir(parents=True, exist_ok=True)
        current = deepcopy(inventory)
        if kwargs.get("use_api") and changed_evidence:
            current["all_candidates"][0]["local_reliable"] = False
        if kwargs.get("use_api") and elapsed_seconds_change:
            current["source_arrow_ownership"]["ocr_local_candidate_searches"][0]["elapsed_seconds"] = .187
        (directory / "binding-candidates.json").write_text(json.dumps(current))
        if not kwargs.get("use_api"):
            result=deepcopy(bindings)
            (directory / "constraint-bindings.json").write_text(json.dumps(result))
            return result
        # Successful transport with a model abstention on the sent label.
        result={"constraints": [], "bindings": [{"record_id": "r1", "accepted": False,
                "reason": "provider_abstained_from_sent_record"}], "counts": {"constraints": 0}, "issues": [],
                "provider": {"status": "succeeded", "network_requests": 1, "http_success": True,
                             "schema_success": True, "input_record_ids": ["r1"], "bindings": []},
                "radius_binding_coverage": {**coverage, "bound_count": 0, "all_confirmed_arrows_bound": False,
                                            "all_radius_records_resolved": False}}
        (directory / "constraint-bindings.json").write_text(json.dumps(result))
        return result
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings", binding)
    def radius_coverage(inventory, graph, constraints, decisions):
        count=sum(row.get("kind")=="radius" for row in constraints)
        return {**deepcopy(coverage),"bound_count":count,
                "all_confirmed_arrows_bound":count==1,"all_radius_records_resolved":count==1}
    monkeypatch.setattr("contour_agent.constraint_binding.radius_binding_coverage", radius_coverage)
    monkeypatch.setattr("contour_agent.parametric_pipeline._solved_source_validation", lambda *args: {"passed": True})
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric", lambda current_graph, constraints, **kwargs:
        {**deepcopy(solution), "constraints": [{**row, "passed": True} for row in constraints],
         "diagnostics": {"remaining_shape_dof": 11 if not constraints else 10}})
    return image, baseline, graph, output, checkpoint, calls


def test_checkpoint_requires_numerical_receipt_for_every_authoritative_constraint(tmp_path, monkeypatch):
    image, baseline, graph, output, saved, calls = setup_case(tmp_path, monkeypatch)
    p = saved / "parametric-solve.json"
    solution = json.loads(p.read_text()); solution["constraints"] = []
    p.write_text(json.dumps(solution))
    result, receipt = _certify_preflight_checkpoint(image, {}, baseline, graph, saved, output / "cert")
    assert result is None and receipt["reason"] == "checkpoint_constraint_receipts_incomplete"
    assert calls == []


def test_native_attribute_candidate_is_inspectable_but_source_failure_never_publishes(tmp_path,monkeypatch):
    image,baseline,graph,output,saved,calls=setup_case(tmp_path,monkeypatch)
    measurements=[]
    def source_check(*args):
        measurements.append(True)
        return {'passed':len(measurements)==1,
                'reasons':[] if len(measurements)==1 else ['solved_source_stroke_support_degraded']}
    monkeypatch.setattr('contour_agent.parametric_pipeline._solved_source_validation',source_check)
    model,stage=refine_parametric(image,{},baseline,output,frozen_topology=graph,
        trusted_preflight_dir=saved,use_api=True)
    candidate=output/'final-binding-attempt/parametric-export/drawing.dxf'
    assert candidate.is_file()
    receipt=stage['final_binding_attempt']['native_attribute_candidate']
    assert receipt['source_gate_passed'] is False and receipt['publication_approved'] is False
    assert stage['final_binding_attempt']['replacement_gate']['replace_checkpoint'] is False
    assert json.loads((candidate.parent/'model.json').read_text())['candidate_scope']['publication_approved'] is False


def test_checkpoint_is_not_reused_with_changed_current_object_ids(tmp_path, monkeypatch):
    image, baseline, graph, output, saved, _ = setup_case(tmp_path, monkeypatch)
    graph["entities"][2]["id"] = "other-current-object"
    result, receipt = _certify_preflight_checkpoint(image, {}, baseline, graph, saved, output / "cert")
    assert result is None and receipt["reason"] == "checkpoint_input_identity_mismatch"


def test_certification_rechecks_original_source_and_native_dxf_radius(tmp_path, monkeypatch):
    image, baseline, graph, output, saved, calls = setup_case(tmp_path, monkeypatch)
    result, receipt = _certify_preflight_checkpoint(image, {}, baseline, graph, saved, output / "cert")
    assert receipt["status"] == "certified" and calls == [False]
    assert result["model"]["validation"]["exact_radius_validation"]["dxf_readback_performed"]
    assert result["model"]["validation"]["exact_radius_validation"]["checks"][0]["dxf_radius"] == 5.


@pytest.mark.parametrize('repaired_ok',[True,False])
def test_old_axis_contract_reoptimizes_same_reverified_subset_before_certification(tmp_path,monkeypatch,repaired_ok):
    from contour_agent import parametric_pipeline as pipeline, parametric_solver as solver
    image,baseline,graph,output,saved,calls=setup_case(tmp_path,monkeypatch)
    old=json.loads((saved/'parametric-solve.json').read_text())
    constraints=json.loads((saved/'constraint-bindings.json').read_text())['constraints']
    axis=solver.native_axis_relation_checks;checks=[];solves=[]
    def legacy_axis(*args,**kwargs):
        checks.append(True)
        return {'passed':False,'checks':[{'absolute_residual':.099999}]} if len(checks)==1 else axis(*args,**kwargs)
    def reoptimize(image,doc,base,domain,known,directory,**kwargs):
        solves.append(True)
        assert domain is graph and known==constraints
        assert kwargs['initial_seed_entities']==old['entities']
        assert kwargs['budget_seconds']==60.
        return {**deepcopy(old),'accepted':repaired_ok}
    monkeypatch.setattr(solver,'native_axis_relation_checks',legacy_axis)
    monkeypatch.setattr(pipeline,'_final_source_multistart',reoptimize)
    result,receipt=_certify_preflight_checkpoint(image,{},baseline,graph,saved,output/'cert')
    assert len(solves)==1 and calls==[False]
    if repaired_ok:assert receipt['status']=='certified' and result is not None
    else:assert result is None and receipt['reason']=='checkpoint_current_axis_reoptimization_not_certified'


def test_same_source_verified_final_candidate_can_replace_whole_checkpoint(tmp_path, monkeypatch):
    image, baseline, graph, output, saved, _ = setup_case(tmp_path, monkeypatch)
    trusted, receipt = _certify_preflight_checkpoint(image, {}, baseline, graph, saved, output / "cert")
    attempt = deepcopy(trusted)
    gate = _final_checkpoint_replacement_gate(trusted, attempt, evidence_sha256=receipt["source_evidence_sha256"])
    assert gate["replace_checkpoint"] and not gate["retain_checkpoint"]


@pytest.mark.parametrize("kind",["vertical","tangent"])
def test_receipts_allow_only_legitimate_current_graph_node_expansion(tmp_path,monkeypatch,kind):
    image, baseline, graph, output, saved, _ = setup_case(tmp_path, monkeypatch)
    if kind=="vertical":
        entity=graph["entities"][0]
        eids=[entity["id"]];nodes=[entity["start_node"],entity["end_node"]]
    else:
        eids=[row["id"] for row in graph["entities"][1:3]]
        nodes=[graph["entities"][1]["end_node"]]
    constraint={"id":"ks1","kind":kind,"source":"source_geometry","record_id":None,
                "entities":eids,"nodes":[],"value":None}
    bindings={"constraints":[constraint]}
    solution={"constraints":[{**constraint,"nodes":nodes,"passed":True}]}
    assert _constraint_receipts_complete(bindings,solution,graph)
    solution["constraints"][0]["nodes"]=[graph["entities"][3]["start_node"]]
    assert not _constraint_receipts_complete(bindings,solution,graph)


def test_angle_mode_is_part_of_the_verified_equation(tmp_path,monkeypatch):
    image, baseline, graph, output, saved, _ = setup_case(tmp_path, monkeypatch)
    constraint={"id":"ka1","kind":"angle","source":"ocr_local_binding","record_id":"a1",
                "entities":[row["id"] for row in graph["entities"][:2]],"nodes":[],"value":90.}
    bindings={"constraints":[constraint]}
    receipt={**constraint,"nodes":[graph["entities"][0]["end_node"]],"angle_mode":"unsigned","passed":True}
    assert _constraint_receipts_complete(bindings,{"constraints":[receipt]},graph)
    receipt["angle_mode"]="signed"
    assert not _constraint_receipts_complete(bindings,{"constraints":[receipt]},graph)


@pytest.mark.parametrize("change",["different_node","reverse_measurement_order","reverse_support_order"])
def test_typed_measurement_receipt_preserves_endpoint_identity_and_direction(change):
    # The bottom/right support intersection is (10,0). Measuring to v2 gives
    # +10 mm, to v3 gives +20 mm, and reversing the endpoints gives -10 mm.
    xy=[[0.,0.],[10.,0.],[10.,10.],[0.,20.]]
    graph={"units":"mm","nodes":[{"id":f"v{i}","x":p[0],"y":p[1]} for i,p in enumerate(xy)],
           "entities":[{"id":f"g{i}","type":"LINE","start":xy[i],"end":xy[(i+1)%4],
                        "start_node":f"v{i}","end_node":f"v{(i+1)%4}"} for i in range(4)]}
    required={"id":"km0","kind":"distance_y","record_id":"r000","source":"ocr_local_binding",
              "entities":["g0","g1"],"nodes":[],"value":10.,
              "anchors":[{"type":"line_intersection","entities":["g0","g1"]},
                         {"type":"node","node_id":"v2"}]}
    receipt={**deepcopy(required),"passed":True}
    if change=="different_node":receipt["anchors"][1]["node_id"]="v3"
    elif change=="reverse_measurement_order":receipt["anchors"].reverse()
    else:receipt["anchors"][0]["entities"].reverse()
    assert _constraint_receipts_complete({"constraints":[required]},
        {"constraints":[receipt]},graph) is (change=="reverse_support_order")


def test_equal_rank_and_radius_coverage_cannot_replace_a_different_dimension(tmp_path, monkeypatch):
    image, baseline, graph, output, saved, _ = setup_case(tmp_path, monkeypatch)
    trusted, receipt = _certify_preflight_checkpoint(image, {}, baseline, graph, saved, output / "cert")
    dimension = {"id": "kd1", "kind": "distance", "record_id": "d1", "entities": [],
                 "nodes": [graph["nodes"][0]["id"],graph["nodes"][1]["id"]], "value": 178.,
                 "source":"ocr_local_binding"}
    trusted["bindings"]["constraints"].append(dimension)
    trusted["solution"]["constraints"].append({**dimension, "passed": True})
    attempt = deepcopy(trusted)
    attempt["bindings"]["constraints"][-1]["value"] = 179.
    attempt["solution"]["constraints"][-1]["value"] = 179.
    gate = _final_checkpoint_replacement_gate(trusted, attempt, evidence_sha256=receipt["source_evidence_sha256"])
    assert gate["previous_verified_radius_records"] == gate["attempt_verified_radius_records"]
    assert not gate["replace_checkpoint"] and gate["retain_checkpoint"]
    assert "previously_verified_constraint_set_lost" in gate["reasons"]


@pytest.mark.parametrize("failure", ["missing_receipt", "failed_readback"])
def test_final_candidate_requires_complete_numeric_and_native_dxf_proof(tmp_path, monkeypatch, failure):
    image, baseline, graph, output, saved, _ = setup_case(tmp_path, monkeypatch)
    trusted, receipt = _certify_preflight_checkpoint(image, {}, baseline, graph, saved, output / "cert")
    attempt = deepcopy(trusted)
    if failure == "missing_receipt":
        del attempt["solution"]["constraints"][0]["value"]
        reason = "final_constraint_receipts_incomplete"
    else:
        attempt["model"]["validation"]["exact_radius_validation"]["passed"] = False
        reason = "final_native_radius_and_dxf_readback_not_verified"
    gate = _final_checkpoint_replacement_gate(trusted, attempt, evidence_sha256=receipt["source_evidence_sha256"])
    assert not gate["replace_checkpoint"] and gate["retain_checkpoint"]
    assert reason in gate["reasons"]


@pytest.mark.parametrize("changed_evidence", [False, True])
def test_final_model_abstention_preserves_certified_subset_but_new_source_counterevidence_revokes_it(
        tmp_path, monkeypatch, changed_evidence):
    image, baseline, graph, output, saved, calls = setup_case(tmp_path, monkeypatch,
                                                             changed_evidence=changed_evidence)
    def forbidden(*args, **kwargs):
        pytest.fail("Final-stage replay must not rerun upstream planning or localization")
    monkeypatch.setattr("contour_agent.topology.build_topology", forbidden)
    monkeypatch.setattr("contour_agent.parametric_pipeline._plan_topology", forbidden)
    model, stage = refine_parametric(image, {}, baseline, output, use_api=True,
                                    frozen_topology=graph, trusted_preflight_dir=saved)
    assert calls == [False, True]
    assert stage["provider"]["http_success"] is True
    attempt = json.loads((output / "final-binding-attempt/constraint-bindings.json").read_text())
    assert attempt["constraints"] == []  # No old constraint was injected into the new attempt.
    guard = json.loads((output / "binding-publication-guard.json").read_text())["replacement_gate"]
    assert not guard["replace_checkpoint"]
    if changed_evidence:
        assert not guard["retain_checkpoint"] and "independent_source_evidence_changed" in guard["reasons"]
        assert model["algorithm_version"] == "source-topology-draft-v1"
        assert not stage["constraint_subset_accepted"]
        contract=json.loads((output/"radius-contract.json").read_text())
        assert contract["publication_status"]=="revoked" and not contract["current_dxf_verified"]
        assert not contract["satisfied"] and contract["coverage"]["bound_count"]==0
        active=json.loads((output/"constraint-bindings.json").read_text())
        assert active["constraints"]==[]
        assert not stage["geometry_updated_by_api"] and not stage["dimensions_updated_by_api"]
        core_model=json.loads((output/"model.json").read_text())
        assert not core_model["parameterization"]["constraint_subset_accepted"]
        assert core_model["parameterization"]["constraints"]==[]
        provenance=json.loads((output/"workflow-provenance.json").read_text())
        assert not provenance["strict_radius_contract"]["current_dxf_verified"]
        assert not provenance["constraint_subset_accepted"]
    else:
        assert guard["retain_checkpoint"] and "previously_verified_radius_coverage_lost" in guard["reasons"]
        assert stage["binding_selection"]["status"] == "certified_preflight_retained"
        assert stage["annotation_radius_contract"]["exact_radius_validation"]["checks"][0]["dxf_radius"] == 5.
        assert stage["reconstruction_feedback"]["verified_radius_record_ids"] == ["r1"]
        selected = json.loads((output / "constraint-bindings.json").read_text())
        assert selected["constraints"][0]["record_id"] == "r1"
    assert all((output / name).is_file() for name in CORE)
    selected_inventory=json.loads((output/"binding-candidates.json").read_text())
    assert selected_inventory["all_candidates"][0]["local_reliable"] is not changed_evidence
    assert json.loads((output/"constraint-bindings.json").read_text())["inventory_artifact"]==str(output/"binding-candidates.json")


def test_elapsed_wall_time_change_does_not_revoke_an_independently_certified_preflight(tmp_path, monkeypatch):
    image, baseline, graph, output, saved, calls = setup_case(tmp_path, monkeypatch, elapsed_seconds_change=True)
    model, stage = refine_parametric(image, {}, baseline, output, use_api=True,
                                    frozen_topology=graph, trusted_preflight_dir=saved)
    guard = json.loads((output/"binding-publication-guard.json").read_text())["replacement_gate"]
    assert calls == [False, True]
    assert guard["retain_checkpoint"] and not guard["replace_checkpoint"]
    assert "independent_source_evidence_changed" not in guard["reasons"]
    assert stage["binding_selection"]["status"] == "certified_preflight_retained"
    assert stage["constraint_subset_accepted"]
    assert model["validation"]["exact_radius_validation"]["checks"][0]["dxf_radius"] == 5.
    latest = json.loads((output/"final-binding-attempt/constraint-bindings.json").read_text())
    assert latest["constraints"] == []  # Timing normalization never promotes the abstaining attempt.

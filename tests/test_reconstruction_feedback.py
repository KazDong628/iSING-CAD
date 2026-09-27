from copy import deepcopy
import json

import pytest

from contour_agent.reconstruction_feedback import geometry_fingerprint, reconstruction_feedback, constraint_regression
from contour_agent.parametric_pipeline import _topology_edit_loop


def candidate(name="base", length=2.):
    return {"id":name,"graph":{"candidate_id":name,"units":"mm","entities":[
        {"id":"g000","type":"LINE","start":[0.,0.],"end":[length,0.]},
        {"id":"g001","type":"LINE","start":[length,0.],"end":[0.,0.]}]}}


def test_geometry_fingerprint_survives_cyclic_display_renumbering():
    graph=candidate()["graph"]
    rotated=deepcopy(graph)
    rotated["entities"]=rotated["entities"][1:]+rotated["entities"][:1]
    for i,row in enumerate(rotated["entities"]):row["id"]=f"other{i}"
    assert geometry_fingerprint(graph)==geometry_fingerprint(rotated)
    rotated["entities"][0]["start"][0]+=.01
    assert geometry_fingerprint(graph)!=geometry_fingerprint(rotated)


def test_unresolved_radius_is_not_mistaken_for_successful_binding():
    graph=candidate()["graph"]
    graph["entities"][0].update(type="ARC",radius=43.,center=[1.,43.],clockwise=False)
    graph["annotation_support"]=[{"record_id":"r1","kind":"radius","nominal":3.,"candidate_entity_id":"g000"}]
    report=reconstruction_feedback(graph)
    assert report["issues"][0]["code"]=="radius_value_unresolved"
    assert not report["issues"][0]["binding_verified"]
    solved=deepcopy(graph["entities"]);solved[0]["radius"]=3.
    assert reconstruction_feedback(graph,solution={"accepted":True,"entities":solved})["issues"][0]["code"]=="radius_annotation_unbound"
    constraint={"kind":"radius","record_id":"r1","entities":["g000"],"passed":True}
    assert not reconstruction_feedback(graph,{"constraints":[constraint]},
        {"accepted":True,"entities":solved,"constraints":[constraint]})["issues"]


def test_constraint_regression_keeps_independent_records_and_feasibility():
    before={"bound_record_ids":["r1"],"solver_accepted":True}
    after={"bound_record_ids":[],"constraint_count":1,"solver_status":"conflict","solver_accepted":False}
    result=constraint_regression(before,after)
    assert not result["passed"] and result["lost_record_ids"]==["r1"]
    assert "candidate_constraints_not_jointly_satisfied" in result["reasons"]


def test_designed_right_angle_is_only_a_source_review_observation():
    points = [[0., 0.], [10., 0.], [10., 6.], [0., 6.]]
    graph = {"candidate_id": "rectangle", "units": "mm", "entities": [
        {"id": f"g{i:03d}", "stable_id": f"edge-{i}", "type": "LINE",
         "start": point, "end": points[(i + 1) % 4]} for i, point in enumerate(points)]}
    report = reconstruction_feedback(graph)
    assert report["issue_count"] == 0 and report["issues"] == []
    assert report["constraint_count"] == 0 and report["structural_constraints"] == []
    assert len(report["review_items"]) == 4
    for i, item in enumerate(report["review_items"]):
        assert item["entity_ids"] == [f"g{i:03d}", f"g{(i + 1) % 4:03d}"]
        assert item["stable_ids"] == [f"edge-{i}", f"edge-{(i + 1) % 4}"]
        assert item["tangent_jump_deg"] == pytest.approx(90.)
        assert item["advisory_only"] is True and item["tangency_required"] is False
    assert constraint_regression(report, {**report, "review_items": []})["passed"]


def test_constraint_regression_protects_structural_evidence_across_reindexing_and_merging():
    before={"constraint_count":2,"structural_constraints":[
        {"kind":"horizontal","stable_ids":["old-a"]},{"kind":"horizontal","stable_ids":["old-b"]}]}
    after={"constraint_count":1,"structural_constraints":[{"kind":"horizontal","stable_ids":["merged"]}],
           "entity_ancestry":{"merged":["merged","old-a","old-b"]},"solver_status":"accepted"}
    assert constraint_regression(before,after)["passed"]
    after["structural_constraints"]=[]
    assert not constraint_regression(before,after)["passed"]
    assert not constraint_regression(before,{"constraint_count":0})["passed"]


@pytest.mark.parametrize("repeat",[False,True])
def test_iteration_budget_stops_and_persists_each_round(tmp_path,monkeypatch,repeat):
    seen=[]
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation",lambda *a:{"passed":True})
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",lambda *a,**k:{"constraints":[]})
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",lambda *a,**k:{"status":"no_constraints"})
    def stage(image,document,baseline,selected,bundle,output,**kwargs):
        seen.append(kwargs["round_index"])
        if len(seen)>1:
            saved=json.loads((tmp_path/"topology-iterations.json").read_text())
            assert len(saved["rounds"])==len(seen)-1
        next_candidate=candidate(f"r{len(seen)}",2. if repeat else 2.+len(seen))
        return next_candidate,{"base_candidate_id":selected["id"],"final_candidate_id":next_candidate["id"],
            "acceptance_gate":{"accepted":True},"execution":{"operations":[]},"editor":{},"evaluator":{}}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage",stage)
    result,audit=_topology_edit_loop("unused",{}, {},candidate(),{},tmp_path)
    assert len(seen)==(1 if repeat else 3)
    assert audit["stop_reason"]==("repeated_geometry" if repeat else "round_budget_exhausted")
    assert result["id"]==("base" if repeat else "r3")
    assert audit["accepted_round_count"]==(0 if repeat else 3)
    assert (tmp_path/"topology-edit-proposals.json").is_file()


def test_iteration_cancellation_is_not_swallowed(tmp_path,monkeypatch):
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation",lambda *a:{"passed":True})
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",lambda *a,**k:{"constraints":[]})
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",lambda *a,**k:{"status":"no_constraints"})
    def cancel(*args):raise InterruptedError()
    with pytest.raises(InterruptedError):
        _topology_edit_loop("unused",{}, {},candidate(),{},tmp_path,progress=cancel)
    assert json.loads((tmp_path/"topology-iterations.json").read_text())["status"]=="running"


def test_rejected_fresh_online_proposal_gets_bounded_feedback_retry(tmp_path,monkeypatch):
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation",lambda *a:{"passed":True})
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",lambda *a,**k:{"constraints":[]})
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",lambda *a,**k:{"status":"no_constraints"})
    calls=[]
    def stage(image,document,baseline,selected,bundle,output,**kwargs):
        calls.append(kwargs["feedback"])
        op={"action":"merge_chain_as_line","entity_ids":["g000","g001"],"record_id":None}
        return selected,{"acceptance_gate":{"accepted":False},"execution":{"operations":[{"operation":op,"status":"rejected","reason":"source_support"}]},
                         "editor":{"schema_success":True,"operations":[op]}}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage",stage)
    final,report=_topology_edit_loop("unused",{}, {},candidate(),{},tmp_path,use_api=True)
    assert len(calls)==2  # Same rejected request cannot consume an unbounded loop.
    assert calls[1]["previous_operations"][0]["reason"]=="source_support"
    assert final["id"]=="base" and report["accepted_round_count"]==0

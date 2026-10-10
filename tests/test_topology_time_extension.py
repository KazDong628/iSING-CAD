"""The optional second search clock buys work, never constraint or source credit."""
from copy import deepcopy

import pytest

from contour_agent import parametric_pipeline as pipeline
from contour_agent import topology_editing, topology_search
from contour_agent.reconstruction_feedback import geometry_fingerprint


def _candidate(name, length):
    graph={"candidate_id":name,"units":"pixel","entities":[
        {"id":"g000","type":"LINE","start":[0.,0.],"end":[length,0.]},
        {"id":"g001","type":"LINE","start":[length,0.],"end":[0.,0.]}]}
    return {"id":name,"graph":graph}


def _extension_record():
    return {"stop_reason":"topology_search_time_budget_exhausted",
            "time_budget_extension":{"status":"not_granted","reason":None,
                                     "original_max_seconds":600.,"extension_seconds":0.,
                                     "total_max_seconds":600.}}


def test_search_budget_one_extension_retains_count_and_visit_limits():
    now=[0.]
    budget=topology_search.SearchBudget(max_preflights=2,max_provider_calls=2,
                                        max_seconds=600.,clock=lambda:now[0])
    operation={"action":"insert_annotated_fillet","entity_ids":["g000"],"record_id":"r1"}
    assert budget.visit_operation("parent",operation)[0]
    now[0]=601.
    assert budget.remaining_seconds()==0.
    assert budget.extend_once(reason="certified_progress_with_fresh_source_witnessed_topology_obligation")
    assert not budget.extend_once(reason="cannot_extend_again")
    assert budget.remaining_seconds()==599.
    assert budget.reserve_preflight() and budget.reserve_preflight()
    assert not budget.reserve_preflight()
    assert budget.reserve_provider("editor","parent")==599.
    assert budget.reserve_provider("evaluator","parent")==599.
    assert budget.reserve_provider("editor","parent") is None
    assert not budget.visit_operation("parent",operation)[0]
    receipt=budget.summary()
    assert (receipt["original_max_seconds"],receipt["extension_seconds"],
            receipt["total_max_seconds"])==(600.,600.,1200.)
    assert (receipt["preflights_used"],receipt["provider_calls_reserved"])==(2,2)
    now[0]=1200.
    assert budget.remaining_seconds()==0.
    with pytest.raises(ValueError,match="total_1200"):
        topology_search.SearchBudget(max_seconds=600.).extend_once(
            seconds=601.,reason="invalid")


def test_short_custom_clock_cannot_gain_disproportionate_time():
    now=[0.]
    budget=topology_search.SearchBudget(max_seconds=10.,clock=lambda:now[0])
    now[0]=11.
    assert budget.extend_once(reason="certified_source_progress")
    assert budget.remaining_seconds()==9.
    assert budget.summary()["total_max_seconds"]==20.
    with pytest.raises(ValueError,match="original_budget"):
        topology_search.SearchBudget(max_seconds=10.).extend_once(
            seconds=600.,reason="invalid")


def test_pending_topology_requires_fresh_arrow_source_witness_and_unvisited_operation(monkeypatch):
    candidate=_candidate("edited",3.)
    operation={"action":"insert_annotated_fillet","entity_ids":["g000","g001"],
               "record_id":"r5","evidence_tags":["annotation_target","source_boundary"]}
    monkeypatch.setattr(topology_editing,"propose_annotation_arc_edits",
                        lambda *args,**kwargs:[deepcopy(operation)])
    feedback={"radius_binding_coverage":{"required_mappings":[
        {"record_id":"r5","source_arrow_verified":False}]}}
    assert pipeline._pending_source_topology_operations(candidate,feedback,[],{})==[]
    feedback["radius_binding_coverage"]["required_mappings"][0]["source_arrow_verified"]=True
    pending=pipeline._pending_source_topology_operations(candidate,feedback,[],{})
    assert [(row["action"],row["record_id"]) for row in pending]==[("insert_annotated_fillet","r5")]
    key=topology_search.operation_fingerprint(geometry_fingerprint(candidate["graph"]),operation)
    assert pipeline._pending_source_topology_operations(candidate,feedback,[],{key:{}})==[]
    operation["evidence_tags"]=["annotation_target"]
    assert pipeline._pending_source_topology_operations(candidate,feedback,[],{})==[]


def test_extension_requires_certified_progress_and_fresh_source_work(monkeypatch):
    now=[601.]
    budget=topology_search.SearchBudget(max_seconds=600.,clock=lambda:now[0])
    previous=_candidate("base",2.);selected=_candidate("edited",3.)
    previous["constraint_feedback"]={"satisfied_record_ids":["r1"]}
    selected["constraint_feedback"]={"satisfied_record_ids":["r1","r2"]}
    record=_extension_record()
    monkeypatch.setattr(pipeline,"_source_verified_primary_progress",lambda *args:False)
    monkeypatch.setattr(pipeline,"_pending_source_topology_operations",
                        lambda *args,**kwargs:[{"action":"insert_annotated_fillet"}])
    assert not pipeline._maybe_extend_topology_time_budget(
        budget,record,previous,selected,[],round_number=1,max_rounds=3,use_api=True)
    assert record["time_budget_extension"]["reason"]=="no_certified_source_numeric_regression_progress"
    assert budget.summary()["total_max_seconds"]==600.
    monkeypatch.setattr(pipeline,"_source_verified_primary_progress",lambda *args:True)
    monkeypatch.setattr(pipeline,"_pending_source_topology_operations",lambda *args,**kwargs:[])
    assert not pipeline._maybe_extend_topology_time_budget(
        budget,record,previous,selected,[],round_number=1,max_rounds=3,use_api=True)
    assert record["time_budget_extension"]["reason"]=="no_fresh_source_witnessed_topology_obligation"
    monkeypatch.setattr(pipeline,"_pending_source_topology_operations",
                        lambda *args,**kwargs:[{"action":"insert_annotated_fillet"}])
    assert pipeline._maybe_extend_topology_time_budget(
        budget,record,previous,selected,[],round_number=1,max_rounds=3,use_api=True)
    assert record["stop_reason"] is None
    assert record["time_budget_extension"]["trigger_candidate_id"]=="edited"
    assert record["time_budget_extension"]["gained_satisfied_record_ids"]==["r2"]
    assert not pipeline._maybe_extend_topology_time_budget(
        budget,record,previous,selected,[],round_number=2,max_rounds=3,use_api=True)
    assert budget.summary()["total_max_seconds"]==1200.


def test_extension_cannot_bypass_final_round_or_count_caps(monkeypatch):
    now=[0.]
    previous=_candidate("base",2.);selected=_candidate("edited",3.)
    selected["constraint_feedback"]={"satisfied_record_ids":["r2"]}
    monkeypatch.setattr(pipeline,"_source_verified_primary_progress",lambda *args:True)
    monkeypatch.setattr(pipeline,"_pending_source_topology_operations",
                        lambda *args,**kwargs:[{"action":"insert_annotated_fillet"}])
    final=topology_search.SearchBudget(max_seconds=600.,clock=lambda:now[0])
    now[0]=601.
    assert not pipeline._maybe_extend_topology_time_budget(
        final,_extension_record(),previous,selected,[],round_number=3,max_rounds=3,use_api=True)
    count=topology_search.SearchBudget(max_preflights=1,max_seconds=600.,clock=lambda:now[0])
    assert count.reserve_preflight()
    assert not pipeline._maybe_extend_topology_time_budget(
        count,_extension_record(),previous,selected,[],round_number=1,max_rounds=3,use_api=True)
    assert count.extension_seconds==0.
    provider=topology_search.SearchBudget(max_provider_calls=1,max_seconds=600.,clock=lambda:now[0])
    provider.provider_calls=1
    assert not pipeline._maybe_extend_topology_time_budget(
        provider,_extension_record(),previous,selected,[],round_number=1,max_rounds=3,use_api=True)
    assert provider.extension_seconds==0.
    late=topology_search.SearchBudget(max_seconds=600.,clock=lambda:now[0])
    now[0]=1802.
    assert not pipeline._maybe_extend_topology_time_budget(
        late,_extension_record(),previous,selected,[],round_number=1,max_rounds=3,use_api=True)
    assert late.extension_seconds==0.


def test_extension_evidence_failure_is_audited_without_grant(monkeypatch):
    previous=_candidate("base",2.);selected=_candidate("edited",3.)
    selected["constraint_feedback"]={"satisfied_record_ids":["r2"]}
    budget=topology_search.SearchBudget(max_seconds=600.)
    record=_extension_record()
    monkeypatch.setattr(pipeline,"_source_verified_primary_progress",
                        lambda *args:(_ for _ in ()).throw(ValueError("invalid evidence")))
    assert not pipeline._maybe_extend_topology_time_budget(
        budget,record,previous,selected,[],round_number=1,max_rounds=3,use_api=False)
    assert record["time_budget_extension"]["reason"]=="progress_evidence_unavailable"
    assert budget.extension_seconds==0.


def test_extended_clock_allows_second_round_without_restarting_preflights(tmp_path,monkeypatch):
    now=[0.]
    real_budget=topology_search.SearchBudget
    monkeypatch.setattr(topology_search,"SearchBudget",
                        lambda **kwargs:real_budget(**kwargs,clock=lambda:now[0]))
    monkeypatch.setattr(pipeline,"_topology_source_validation",lambda *args:{"passed":True})
    monkeypatch.setattr(pipeline,"_solved_source_validation",lambda *args:{"passed":True})
    monkeypatch.setattr(pipeline,"_preflight_input_identity",lambda *args:{"current":True})
    monkeypatch.setattr(pipeline,"_record_fillet_tangent_contract",lambda *args:None)
    monkeypatch.setattr(pipeline,"source_failure_feedback",
                        lambda feedback,*args:feedback.update(post_solve_source_accepted=True,
                                                             source_validation={"passed":True}))
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",
                        lambda *args,**kwargs:{"constraints":[{"id":"c1"}]})
    monkeypatch.setattr(pipeline,"_solve_with_source_observation",
                        lambda solver,graph,*args,**kwargs:{"accepted":True,"status":"accepted",
                            "entities":graph["entities"]})
    def feedback(graph,*args,**kwargs):
        edited=graph["candidate_id"]=="edited"
        return {"geometry_sha256":geometry_fingerprint(graph),"solver_status":"accepted",
                "solver_accepted":True,"constraint_count":1,
                "satisfied_record_ids":["r1","r2"] if edited else ["r1"],
                "source_validation":{"passed":True},"topology_source_validation":{"passed":True},
                "post_solve_source_accepted":True}
    monkeypatch.setattr(pipeline,"reconstruction_feedback",feedback)
    monkeypatch.setattr(pipeline,"_source_verified_primary_progress",lambda *args:True)
    monkeypatch.setattr(pipeline,"_pending_source_topology_operations",
                        lambda *args,**kwargs:[{"action":"insert_annotated_fillet"}])
    calls=[]
    def stage(image,document,baseline,parent,bundle,output,**kwargs):
        number=kwargs["round_index"];calls.append(number)
        if number==1:
            result=_candidate("edited",3.)
            result["constraint_regression_gate"]={"passed":True,"reasons":[]}
            result["constraint_feedback"]=kwargs["verifier"](result)
            now[0]=601.
        else:
            result=parent
            now[0]=1201.
        return result,{"base_candidate_id":parent["id"],"final_candidate_id":result["id"],
                       "acceptance_gate":{"accepted":number==1},"execution":{"operations":[]},
                       "editor":{},"evaluator":{},"trusted_candidate_ids":[result["id"]]}
    monkeypatch.setattr(pipeline,"_topology_edit_stage",stage)
    base=_candidate("base",2.)
    final,report=pipeline._topology_edit_loop("unused",{}, {},base,
        {"candidates":[base],"annotation_inventory":[]},tmp_path,
        max_rounds=3,beam_width=1,max_seconds=600.)
    assert calls==[1,2] and final["id"]=="edited"
    assert len(report["rounds"])==2 and report["accepted_round_count"]==1
    assert report["budget"]["preflights_used"]==2
    assert report["budget"]["original_max_seconds"]==600.
    assert report["budget"]["total_max_seconds"]==1200.
    assert report["time_budget_extension"]["status"]=="granted"

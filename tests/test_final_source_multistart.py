"""The optional final solve must be source-only and never replace a valid incumbent on regression."""

import json
import copy

import pytest

from contour_agent import parametric_pipeline as pipeline
from contour_agent.parametric_solver import solve_parametric, native_measurement_checks, native_source_nominal_angle_checks
from test_parametric_solver import rectangle
from test_exact_source_nominal_solver import exact, native


def _export_receipt(passed=True):
    return {"validation":{key:{"passed":passed,"dxf_readback_performed":True} for key in
        ("exact_radius_validation","strict_relation_validation","native_measurement_validation","native_source_angle_validation")}}


def _solution(name, *, accepted=True, evaluations=10):
    return {"status":"accepted" if accepted else "failed", "accepted":accepted,
            "underconstrained":True,"entities":[{"id":name}],
            "constraints":[{"id":"c0","passed":True}],
            "diagnostics":{"source_budget_search":{"objective_evaluations":evaluations}}}


def _source_check(mask, support, p90=2., *, passed=True):
    return {"passed":passed,"before":{"stroke_supported_fraction":.8,
                "support_measurement_version":"source-stroke-support-v2"},
            "after":{"stroke_supported_fraction":support,"p90_edge_distance_px":p90,
                     "support_measurement_version":"source-stroke-support-v2"},
            "oracle_mask_validation":{"conservative_max_deviation_px":mask}}


@pytest.mark.parametrize('same_problem',[True,False])
def test_budget_checkpoint_continues_only_same_problem_without_publication_claim(monkeypatch,tmp_path,same_problem):
    from contour_agent.reconstruction_priority import source_first_model
    from contour_agent.parametric_solver import numerical_problem_identity
    graph={};constraints=[{'id':'c0'}];partial=[{'id':'partial'}]
    first=_solution('baseline',accepted=False);first['status']='optimization_budget_exhausted'
    first.update(candidate_entities=partial)
    first['diagnostics']['numerical_continuation_checkpoint']={
        'available':True,'seed_only':True,'publication_approved':False,
        'problem_sha256':numerical_problem_identity(graph,constraints) if same_problem else 'changed'}
    second=_solution('continued');second['underconstrained']=False
    monkeypatch.setattr('contour_agent.source_stroke_prior.source_stroke_observation',lambda *a,**kw:{})
    monkeypatch.setattr(pipeline,'_solver_source_observation',lambda *a:{
        'optimization_budget':{},'mask_representation_budget_diagnostic':{'sampling_step':.05}})
    calls=[]
    def solve(*a,**kw):
        calls.append(kw)
        return first if len(calls)==1 else second
    monkeypatch.setattr(pipeline,'_solve_with_source_observation',solve)
    monkeypatch.setattr(pipeline,'_constraint_receipts_complete',lambda *a:True)
    monkeypatch.setattr(pipeline,'_solved_source_validation',lambda *a:_source_check(.5,.9))
    result=pipeline._final_source_multistart('source.png',{},source_first_model({}),graph,
        constraints,tmp_path,solver=object())
    assert len(calls)==(2 if same_problem else 1)
    if same_problem:
        assert calls[1]['seed_entities']==partial
        assert result is second and calls[1]['budget_seconds']<=15.
    else:assert result is first


def test_final_multistart_selects_only_pareto_improved_source_candidate(monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline,"_constraint_receipts_complete",lambda *args:True)
    observation={"boundary_error_budget":{"sampling_step":.05}}
    monkeypatch.setattr(pipeline,"_solver_source_observation",lambda *args:observation)
    offsets=[{"v0":[.1,0.]},{"v0":[-.1,0.]}]
    monkeypatch.setattr(pipeline,"_source_restart_offsets",lambda *args:(offsets,{"status":"prepared"}))
    solves=iter([_solution("canonical"),_solution("better"),_solution("ink-regressed")])
    budgets=[]
    def solve(*args,**kwargs):
        budgets.append((kwargs["budget_seconds"],kwargs["budget_evaluations"],kwargs.get("seed_node_offsets")))
        return next(solves)
    monkeypatch.setattr(pipeline,"_solve_with_source_observation",solve)
    scores={"canonical":_source_check(3.,.8),"better":_source_check(2.8,.81),
            "ink-regressed":_source_check(2.5,.79)}
    monkeypatch.setattr(pipeline,"_solved_source_validation",
                        lambda *args:scores[args[-1][0]["id"]])
    monkeypatch.setattr(pipeline,"export_parametric",lambda *args,**kwargs:_export_receipt())
    selected=pipeline._final_source_multistart("source.png",{}, {}, {},[{"id":"c0"}],tmp_path,solver=object())
    assert selected["entities"][0]["id"]=="better"
    receipt=json.loads((tmp_path/"source-multistart.json").read_text(encoding="utf8"))
    assert receipt["selected_seed"]=="source-normal-1" and receipt["ground_truth_used"] is False
    assert receipt["attempts"][2]["reason"]=="source_evidence_not_improved"
    assert len(budgets)==3 and [row[1] for row in budgets]==[30000,3000,3000]
    assert budgets[1][2]==offsets[0] and budgets[2][2]==offsets[1]
    assert budgets[0][0]<=120. and all(row[0]<=15. for row in budgets[1:])


@pytest.mark.parametrize('source_pass',[False,True])
def test_source_first_nominals_use_original_ink_in_canonical_solve_and_shared_recovery_budget(monkeypatch,tmp_path,source_pass):
    from contour_agent.reconstruction_priority import source_first_model
    ink={'source':'original_image_direction_compatible_ink','reference_dxf_read':False}
    monkeypatch.setattr('contour_agent.source_stroke_prior.source_stroke_observation',lambda *args,**kwargs:ink)
    monkeypatch.setattr(pipeline,'_solver_source_observation',lambda *args:{
        'optimization_budget':{'max_wall_seconds':120.},
        'mask_representation_budget_diagnostic':{'sampling_step':.05}})
    monkeypatch.setattr(pipeline,'_constraint_receipts_complete',lambda *args:True)
    monkeypatch.setattr(pipeline,'_source_restart_offsets',lambda *args,**kwargs:(
        [{'v0':[.1,0.]},{'v0':[-.1,0.]}],{'status':'prepared'}))
    calls=[]
    def solve(*args,**kwargs):
        calls.append(kwargs)
        assert kwargs['image_stroke_prior']==ink
        result=_solution('canonical' if len(calls)==1 else 'recovered')
        result['underconstrained']=False
        return result
    monkeypatch.setattr(pipeline,'_solve_with_source_observation',solve)
    monkeypatch.setattr(pipeline,'_solved_source_validation',lambda *args:
        _source_check(.1,.9,passed=source_pass or args[-1][0]['id']=='recovered'))
    monkeypatch.setattr(pipeline,'export_parametric',lambda *args,**kwargs:_export_receipt())
    selected=pipeline._final_source_multistart('source.png',{},source_first_model({}),{},
        [{'id':'c0'}],tmp_path,solver=object())
    receipt=json.loads((tmp_path/'source-multistart.json').read_text('utf8'))
    assert receipt['canonical_original_ink_prior_enabled'] is True
    assert receipt['acceptance_thresholds_changed'] is False
    assert len(calls)==(1 if source_pass else 2)
    if not source_pass:
        assert 15.<calls[1]['budget_seconds']<=120.
        assert 3000<calls[1]['budget_evaluations']<30000
        assert selected['entities'][0]['id']=='recovered'


@pytest.mark.parametrize('native_pass',[True,False])
def test_source_failed_native_exact_progress_is_saved_without_promoting_or_resetting_constraints(monkeypatch,tmp_path,native_pass):
    from contour_agent.reconstruction_priority import source_first_model
    monkeypatch.setattr('contour_agent.source_stroke_prior.source_stroke_observation',lambda *a,**kw:{})
    monkeypatch.setattr(pipeline,'_solver_source_observation',lambda *a:{
        'optimization_budget':{},'mask_representation_budget_diagnostic':{'sampling_step':.05}})
    monkeypatch.setattr(pipeline,'_constraint_receipts_complete',lambda *a:True)
    monkeypatch.setattr(pipeline,'_source_restart_offsets',lambda *a,**kw:([
        {'v0':[.1,0.]},{'v0':[-.1,0.]}],{}))
    sequence=iter([_solution('canonical'),_solution('better'),_solution('worse')]);calls=[]
    def solve(*args,**kwargs):
        calls.append(kwargs)
        assert args[2]==[{'id':'c0'}]
        return next(sequence)
    monkeypatch.setattr(pipeline,'_solve_with_source_observation',solve)
    scores={'canonical':.70,'better':.75,'worse':.74}
    def check(*args):
        value=_source_check(.5,scores[args[-1][0]['id']],passed=False)
        value['reasons']=['solved_source_stroke_support_degraded'];return value
    monkeypatch.setattr(pipeline,'_solved_source_validation',check)
    monkeypatch.setattr(pipeline,'export_parametric',lambda *a,**kw:_export_receipt(native_pass))
    result=pipeline._final_source_multistart('source',{},source_first_model({}),{},
        [{'id':'c0'}],tmp_path,solver=object())
    assert result['entities'][0]['id']=='canonical'
    receipt=json.loads((tmp_path/'source-multistart.json').read_text('utf8'))
    assert receipt['selected_seed']=='canonical'
    if native_pass:
        draft=receipt['unpublished_numerical_candidate']
        assert draft['seed']=='source-normal-1'
        assert draft['source_gate_passed'] is False and draft['publication_approved'] is False
        assert calls[2]['seed_entities']==[{'id':'better'}]
        assert draft['native_attributes_verified'] is True
    else:
        assert 'unpublished_numerical_candidate' not in receipt
        assert calls[2]['seed_entities']==[{'id':'canonical'}]


@pytest.mark.parametrize("readback",[False,True])
def test_final_multistart_keeps_canonical_when_alternative_fails_a_gate(monkeypatch,tmp_path,readback):
    monkeypatch.setattr(pipeline,"_constraint_receipts_complete",lambda *args:True)
    monkeypatch.setattr(pipeline,"_solver_source_observation",
                        lambda *args:{"boundary_error_budget":{"sampling_step":.05}})
    monkeypatch.setattr(pipeline,"_source_restart_offsets",
                        lambda *args:([{"v0":[.1,0.]}],{"status":"prepared"}))
    solves=iter([_solution("canonical"),_solution("alternative")])
    monkeypatch.setattr(pipeline,"_solve_with_source_observation",lambda *args,**kwargs:next(solves))
    checks={"canonical":_source_check(3.,.8),
            "alternative":_source_check(2.,.9,passed=readback)}
    monkeypatch.setattr(pipeline,"_solved_source_validation",
                        lambda *args:checks[args[-1][0]["id"]])
    monkeypatch.setattr(pipeline,"export_parametric",lambda *args,**kwargs:_export_receipt(False))
    selected=pipeline._final_source_multistart("source.png",{}, {}, {},[{"id":"c0"}],tmp_path,solver=object())
    assert selected["entities"][0]["id"]=="canonical"
    receipt=json.loads((tmp_path/"source-multistart.json").read_text(encoding="utf8"))
    assert receipt["selected_seed"]=="canonical"
    assert receipt["attempts"][1]["source_gate_passed"] is readback
    assert receipt["attempts"][1]["dxf_readback_passed"] is False


def _failed_source_case():
    graph=rectangle()
    constraints=[exact({"id":"width","kind":"distance_x","nodes":["v0","v1"],"value":10.})]
    good=solve_parametric(graph,constraints)
    assert good["accepted"]
    failed=copy.deepcopy(good)
    failed.update(status="source_budget_search_failed",accepted=False,
                  entities=copy.deepcopy(graph["entities"]),nodes=copy.deepcopy(graph["nodes"]))
    failed["validation"].update(passed=False,source_boundary_budget_passed=False)
    for node in failed["candidate_nodes"]:node["y"]+=.3
    for entity in failed["candidate_entities"]:
        for endpoint in ("start","end"):entity[endpoint][1]+=.3
    failed["diagnostics"]["source_budget_search"]={"objective_evaluations":38,"enabled":True}
    return graph,constraints,failed,good


def _native_export_check(image,baseline,solution,directory):
    entities=solution["entities"];constraints=solution["constraints"];document=native(entities)
    return {"validation":{
        "exact_radius_validation":pipeline.exact_radius_checks(entities,constraints,dxf_document=document),
        "strict_relation_validation":pipeline.relation_checks(entities,constraints,dxf_document=document),
        "native_measurement_validation":native_measurement_checks(entities,constraints,dxf_document=document),
        "native_source_angle_validation":native_source_nominal_angle_checks(entities,constraints,dxf_document=document)}}


def _recovery_setup(monkeypatch,solutions):
    monkeypatch.setattr(pipeline,"_solver_source_observation",lambda *args:
        {"boundary_error_budget":{"sampling_step":.05,"maximum_deviation":.1}})
    values=iter(solutions);calls=[]
    def solve(solver,graph,constraints,baseline,directory,**kwargs):
        calls.append((copy.deepcopy(graph),copy.deepcopy(constraints),kwargs))
        return next(values)
    monkeypatch.setattr(pipeline,"_solve_with_source_observation",solve)
    seed_inputs=[]
    def seeds(image,document,baseline,graph,entities,step,**kwargs):
        seed_inputs.append(copy.deepcopy(entities))
        return [{"v0":[.1,0.]}],{"status":"prepared","ground_truth_used":False}
    monkeypatch.setattr(pipeline,"_source_restart_offsets",seeds)
    monkeypatch.setattr(pipeline,"_solved_source_validation",lambda *args:_source_check(.05,.9))
    monkeypatch.setattr(pipeline,"export_parametric",_native_export_check)
    return calls,seed_inputs


def test_failed_source_feasibility_can_restart_original_problem_and_pass_every_native_gate(monkeypatch,tmp_path):
    graph,constraints,failed,good=_failed_source_case()
    calls,seed_inputs=_recovery_setup(monkeypatch,[failed,good])
    result=pipeline._final_source_multistart("source.png",{}, {},graph,constraints,tmp_path,solver=object())
    receipt=json.loads((tmp_path/"source-multistart.json").read_text(encoding="utf8"))
    assert result is good and result["accepted"]
    assert receipt["status"]=="selected_source_feasibility_recovery"
    assert receipt["recovery_admission"]["candidate_constraints_independently_checked"]
    assert receipt["attempts"][1]["dxf_readback_passed"]
    assert receipt["attempts"][1]["reason"]=="source_feasibility_recovered_all_gates_passed"
    assert seed_inputs[0]==failed["candidate_entities"]
    assert len(calls)==2 and all(call[0]==graph and call[1]==constraints for call in calls)
    assert calls[1][2]["budget_seconds"]<=15. and calls[1][2]["budget_evaluations"]<=3000
    saved=json.loads((tmp_path/"source-multistart/canonical/parametric-solution.json").read_text(encoding="utf8"))
    assert saved["accepted"] is False


@pytest.mark.parametrize("failure",["value","branch","actual","geometry"])
def test_invalid_failed_candidate_does_not_gain_restart_eligibility(failure):
    graph,constraints,failed,_=_failed_source_case()
    if failure=="value":failed["constraints"][0]["value"]=9.9
    if failure=="branch":failed["candidate_entities"][0]["id"]="wrong"
    if failure=="actual":
        failed["candidate_nodes"][1]["x"]-=.01
        failed["candidate_entities"][0]["end"][0]-=.01
        failed["candidate_entities"][1]["start"][0]-=.01
    if failure=="geometry":failed["validation"]["geometry_valid"]=False
    assert not pipeline._source_feasibility_restart_admission(graph,constraints,failed)["eligible"]


@pytest.mark.parametrize("failure",["source","native","constraints"])
def test_restart_cannot_publish_a_candidate_with_a_failed_independent_gate(monkeypatch,tmp_path,failure):
    graph,constraints,failed,good=_failed_source_case()
    _recovery_setup(monkeypatch,[failed,good])
    if failure=="source":
        monkeypatch.setattr(pipeline,"_solved_source_validation",lambda *args:_source_check(.2,.9,passed=False))
    elif failure=="native":
        def export(*args):
            result=_export_receipt();result["validation"]["native_measurement_validation"]["passed"]=False
            return result
        monkeypatch.setattr(pipeline,"export_parametric",export)
    else:good["constraints"][0]["value"]=10.01
    result=pipeline._final_source_multistart("source.png",{}, {},graph,constraints,tmp_path,solver=object())
    assert result is failed and not result["accepted"]
    assert (tmp_path/"source-multistart/source-normal-1/parametric-solution.json").exists()


@pytest.mark.parametrize("late_phase",["solve","source","native","promotion"])
def test_shared_deadline_prevents_late_restart_promotion(monkeypatch,tmp_path,late_phase):
    graph,constraints,failed,good=_failed_source_case()
    calls,_=_recovery_setup(monkeypatch,[failed,good]);clock=[0.]
    monkeypatch.setattr(pipeline.time,"monotonic",lambda:clock[0])
    original_solve=pipeline._solve_with_source_observation
    def solve(*args,**kwargs):
        result=original_solve(*args,**kwargs)
        if len(calls)==2 and late_phase=="solve":clock[0]=121.
        return result
    monkeypatch.setattr(pipeline,"_solve_with_source_observation",solve)
    def source(*args):
        if late_phase=="source":clock[0]=121.
        return _source_check(.05,.9)
    monkeypatch.setattr(pipeline,"_solved_source_validation",source)
    def export(*args):
        if late_phase=="native":clock[0]=121.
        return _native_export_check(*args)
    monkeypatch.setattr(pipeline,"export_parametric",export)
    if late_phase=="promotion":
        original_score=pipeline._source_solution_score
        def score(*args):
            result=original_score(*args);clock[0]=121.;return result
        monkeypatch.setattr(pipeline,"_source_solution_score",score)
    result=pipeline._final_source_multistart("source.png",{}, {},graph,constraints,tmp_path,solver=object())
    assert result is failed
    receipt=json.loads((tmp_path/"source-multistart.json").read_text(encoding="utf8"))
    assert receipt["selected_seed"]=="canonical"
    assert receipt["stop_reason"]=="shared_wall_deadline_reached"


def test_legacy_already_strict_axis_angle_keeps_input_output_constraint_identity():
    graph=rectangle()
    constraint={"id":"angle","kind":"angle","record_id":"r_angle","source":"ocr_api_binding",
        "entities":["g0"],"nodes":[],"value":0.,"reference_axis":"horizontal","angle_mode":"unsigned",
        "required":True,"nominal_source":"source_ocr","source_arrow_verified":True}
    result=solve_parametric(graph,[constraint])
    assert result["accepted"]
    assert pipeline._constraint_receipts_complete({"constraints":[constraint]},result,graph)


@pytest.mark.parametrize("recovered", [True, False])
def test_numerically_accepted_but_source_rejected_candidate_receives_bounded_recovery(monkeypatch,tmp_path,recovered):
    graph,constraints,_,good=_failed_source_case()
    canonical=copy.deepcopy(good)
    calls,_=_recovery_setup(monkeypatch,[canonical,good])
    source_calls=[0]
    def source(*args):
        source_calls[0]+=1
        result=_source_check(.05,.9,passed=source_calls[0]>1 and recovered)
        result["reasons"]=[] if result["passed"] else ["solved_source_stroke_support_degraded"]
        return result
    monkeypatch.setattr(pipeline,"_solved_source_validation",source)
    result=pipeline._final_source_multistart("source.png",{}, {},graph,constraints,tmp_path,solver=object())
    receipt=json.loads((tmp_path/"source-multistart.json").read_text(encoding="utf8"))
    assert len(calls)==2 and all(call[0]==graph and call[1]==constraints for call in calls)
    assert receipt["search_purpose"]=="recover_original_source_support_without_changing_the_problem"
    assert receipt["attempts"][0]["solver_accepted"] is True
    assert receipt["attempts"][0]["source_gate_passed"] is False
    assert calls[1][2]["budget_seconds"]<=15. and calls[1][2]["budget_evaluations"]<=3000
    assert result is (good if recovered else canonical)
    assert receipt["selected_seed"]==("source-normal-1" if recovered else "canonical")
    assert receipt["attempts"][1]["dxf_readback_passed"] is recovered


def test_ink_recovery_probes_new_support_loss_instead_of_preexisting_unsupported_closure(monkeypatch):
    from contour_agent import source_support_diagnostics as diagnostics
    from contour_agent import reconstruction_feedback as feedback
    graph=rectangle()
    def row(entity,support,count):
        return {"entity_id":entity,"sample_count":count,"quarters":[
            {"quarter":q,"stroke_supported_fraction":support,"p90_edge_distance_px":2.} for q in range(1,5)]}
    before={"status":"measured","entities":[row("g0",0.,500),row("g1",1.,40),row("g2",1.,100)]}
    after={"status":"measured","entities":[row("g0",0.,500),row("g1",.8,40),row("g2",.5,100)]}
    measurements=iter([before,after])
    monkeypatch.setattr(diagnostics,"source_support_diagnostics",lambda *args:next(measurements))
    monkeypatch.setattr(feedback,"source_mask_interval_diagnostics",lambda *args:{"entities":[]})
    offsets,evidence=pipeline._source_restart_offsets("source.png",{}, {},graph,graph["entities"],.05,
                                                      recover_stroke_support=True)
    assert evidence["entity_id"]=="g2"
    assert evidence["basis"]=="largest_original_image_ink_support_loss_from_same_source_graph"
    assert evidence["ground_truth_used"] is False
    assert len(offsets)==2 and all(set(seed)=={"v2","v3"} for seed in offsets)
    assert offsets[0]["v2"]==offsets[0]["v3"]
    assert evidence["offset_magnitude"]==pytest.approx(.1)


def test_moved_initial_seed_cannot_reset_original_problem_or_source_comparator(monkeypatch,tmp_path):
    graph=rectangle()
    initial=copy.deepcopy(graph['entities'])
    for entity in initial:
        entity['start'][0]+=.12
        entity['end'][0]+=.12
    before=copy.deepcopy(graph)
    observation={'boundary_error_budget':{'sampling_step':.05}}
    monkeypatch.setattr(pipeline,'_solver_source_observation',lambda _baseline,original:
        observation if original==before else pytest.fail('mask comparator was replaced by seed'))
    seen=[]
    def solve(_solver,original,constraints,baseline,directory,**kwargs):
        assert original==before and kwargs['seed_entities']==initial
        seen.append('original_domain')
        return {'accepted':True,'status':'accepted','underconstrained':False,'entities':initial,
                'constraints':[], 'diagnostics':{}}
    def check(_image,_document,_baseline,original,entities):
        assert original==before and entities==initial
        seen.append('original_ink_comparator')
        return {'passed':True}
    monkeypatch.setattr(pipeline,'_solve_with_source_observation',solve)
    monkeypatch.setattr(pipeline,'_solved_source_validation',check)
    result=pipeline._final_source_multistart('source.png',{}, {},graph,[],tmp_path,
                                           solver=object(),initial_seed_entities=initial)
    assert result['accepted'] and seen==['original_domain','original_ink_comparator']
    assert graph==before

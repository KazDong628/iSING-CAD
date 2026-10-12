"""Independent rejection tests for source-owned topology initial guesses."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from contour_agent import source_topology_initial_guess as repair


def _case():
    arc = {"id": "arc", "type": "ARC", "radius": 65., "start": [0., 0.], "end": [5., 5.]}
    line = {"id": "new-line", "type": "LINE", "start": [5., 5.], "end": [10., 10.],
        "angle_support_evidence": {"record_id": "a24", "nominal": 24., "reference_axis": "vertical",
            "requires_angle_binding_and_solve": True, "ground_truth_used": False,
            "method": "locally_verified_angle_ink_interval_topology_restore",
            "source_interval_endpoints_px": [[0., 0.], [8., 20.]]}}
    graph = {"units": "mm", "entities": [arc, line], "source_grid_pitch_px": 1.,
        "entity_identity": {"parent_to_current": {"arc": ["arc"], "old-line": ["new-line"]}},
        "source_evidence": {"topology_edit": {"action": "restore_annotated_line_support",
            "resegmentation_applied": True, "ground_truth_used": False, "record_id": "a24",
            "angle_support_record_ids": ["a24"], "net_entity_reduction": -1}}}
    parent = {"entities": [deepcopy(arc), {"id": "old-line", "type": "LINE"}]}
    radius = {"id": "k65", "record_id": "r65", "kind": "radius", "value": 65.,
        "entities": ["arc"], "nodes": [], "required": True, "enforcement": "exact",
        "source": "ocr_local_binding", "nominal_source": "source_ocr", "tolerance": 0.}
    angle = {"id": "k24", "record_id": "a24", "kind": "angle", "value": 24.,
        "entities": ["new-line"], "nodes": [], "required": True, "enforcement": "exact",
        "source": "ocr_local_binding", "nominal_source": "source_ocr", "reference_axis": "vertical"}
    source = {"passed": False, "reasons": ["original_oracle_mask_budget_exceeded"]}
    return {"id": "seed", "graph": graph}, parent, [radius], [deepcopy(radius), angle], source


def test_raw_mask_exception_never_approves_publication_or_mutates_input():
    candidate, _, _, _, source = _case()
    old = deepcopy(candidate)
    receipt = repair.line_seed_admission(candidate["graph"], source)
    assert receipt["eligible"] is True and receipt["publication_approved"] is False
    assert candidate == old


def _two_radius_case():
    candidate,parent,old,current,source=_case()
    line=candidate['graph']['entities'][1]
    line['angle_support_evidence']['method']='verified_two_radius_joint_line_topology_restore'
    line['source_contact_construction']=dict(method='source_nominal_exact_arc_line_arc_v1',
        angle_record_id='a24',radius_record_ids=['r15','r65'],ground_truth_used=False,
        radius_exact=True,angle_exact=True,contacts_exact=True,outer_endpoints_fixed=True,
        source_line_support_checked_during_search=True)
    return candidate,parent,old,current,source


def test_two_radius_contact_proposal_can_enter_a_solve_without_approving_export():
    candidate,_,_,_,source=_two_radius_case()
    receipt=repair.line_seed_admission(candidate['graph'],source)
    assert receipt['eligible'] and receipt['publication_approved'] is False


@pytest.mark.parametrize('defect',['unobserved','duplicate_radius','wrong_angle','free_radius','false_contact'])
def test_two_radius_seed_needs_nominal_contact_and_source_search_witness(defect):
    candidate,_,_,_,source=_two_radius_case()
    contact=candidate['graph']['entities'][1]['source_contact_construction']
    if defect=='unobserved':contact['source_line_support_checked_during_search']=False
    elif defect=='duplicate_radius':contact['radius_record_ids']=['r15','r15']
    elif defect=='wrong_angle':contact['angle_record_id']='other'
    elif defect=='free_radius':contact['radius_exact']=False
    elif defect=='false_contact':contact['contacts_exact']=False
    assert not repair.line_seed_admission(candidate['graph'],source)['eligible']


@pytest.mark.parametrize("reason", ["source_hash_mismatch", "source_stroke_support_degraded",
    "topology_validation_failed", "node_source_mapping_mismatch", "units_mismatch"])
def test_any_non_mask_failure_is_ineligible(reason):
    candidate, _, _, _, source = _case()
    source["reasons"].append(reason)
    assert repair.line_seed_admission(candidate["graph"], source)["eligible"] is False


@pytest.mark.parametrize("field,value", [("type", "ARC"), ("requires_angle_binding_and_solve", False),
    ("ground_truth_used", True), ("reference_axis", "unknown"), ("nominal", float("nan")),
    ("source_interval_endpoints_px", [[0., 0.], [0., 2.]])])
def test_missing_finite_independent_angle_line_evidence_is_ineligible(field, value):
    candidate, _, _, _, source = _case()
    entity = candidate["graph"]["entities"][1]
    (entity if field == "type" else entity["angle_support_evidence"])[field] = value
    assert repair.line_seed_admission(candidate["graph"], source)["eligible"] is False


@pytest.mark.parametrize("alteration", [{"required": False}, {"enforcement": "soft"},
    {"nominal_source": "fitted"}, {"source": "unverified_proposal"}])
def test_old_exact_radius_cannot_be_retained_only_by_value_with_weaker_enforcement(alteration):
    candidate, parent, old, current, _ = _case()
    current[0].update(alteration)
    assert repair.required_constraints_preserved(parent, candidate["graph"], old, current)["passed"] is False


def test_type_lineage_cannot_rebind_a_radius_to_an_unrelated_or_ambiguous_child():
    candidate, parent, old, current, _ = _case()
    graph = candidate["graph"]
    graph["entity_identity"]["parent_to_current"]["arc"] = ["new-line"]
    assert repair.required_constraints_preserved(parent, graph, old, current)["passed"] is False
    graph["entities"].append({"id": "another-arc", "type": "ARC"})
    graph["entity_identity"]["parent_to_current"]["arc"] = ["arc", "another-arc"]
    assert repair.required_constraints_preserved(parent, graph, old, current)["passed"] is False


def _fresh_arrow_lineage_case():
    candidate,parent,old,current,_=_case()
    graph=candidate['graph']
    graph['entities'].append({'id':'other-arc','type':'ARC'})
    graph['entity_identity']['parent_to_current']['arc']=['arc','new-line','other-arc']
    current[0]['source_arrow_verified']=True
    return parent,graph,old,current


def test_fresh_independent_arrow_disambiguates_radius_identity_after_resegmentation():
    parent,graph,old,current=_fresh_arrow_lineage_case()
    result=repair.required_constraints_preserved(parent,graph,old,current)
    assert result['passed']
    assert result['source_arrow_radius_identity_mapping']['arc']['entity_id']=='arc'


@pytest.mark.parametrize('defect',['arrow','record','value','source','enforcement','duplicate'])
def test_radius_identity_cannot_be_inferred_from_a_matching_fitted_radius(defect):
    parent,graph,old,current=_fresh_arrow_lineage_case()
    if defect=='arrow':current[0]['source_arrow_verified']=False
    elif defect=='record':current[0]['record_id']='other'
    elif defect=='value':current[0]['value']=64.
    elif defect=='source':current[0]['source']='fitted'
    elif defect=='enforcement':current[0]['enforcement']='soft'
    elif defect=='duplicate':current.append({**current[0],'id':'other','entities':['other-arc']})
    assert not repair.required_constraints_preserved(parent,graph,old,current)['passed']


def _anchored_measurement_case():
    parent={"entities":[{"id":"a","type":"LINE"},{"id":"b","type":"LINE"}]}
    graph={"entities":[{"id":"x","type":"LINE"},{"id":"y","type":"LINE"}],
        "entity_identity":{"parent_to_current":{"a":["x"],"b":["y"]}}}
    old={"id":"k65","kind":"distance_y","record_id":"r65","value":65.,
        "entities":["a","b"],"nodes":[],"source":"ocr_local_binding",
        "required":True,"enforcement":"exact","nominal_source":"source_ocr",
        "anchors":[{"type":"support_level","entity_id":"a","axis":"y"},
                   {"type":"support_level","entity_id":"b","axis":"y"}]}
    fresh=deepcopy(old);fresh.update(id="knew",entities=["y","x"],
        anchors=[{"type":"support_level","entity_id":"x","axis":"y"},
                 {"type":"support_level","entity_id":"y","axis":"y"}])
    return parent,graph,old,fresh


def test_anchored_distance_membership_order_is_not_measurement_direction():
    parent,graph,old,fresh=_anchored_measurement_case()
    assert repair.required_constraints_preserved(parent,graph,[old],[fresh])["passed"] is True


@pytest.mark.parametrize("change",["anchors","value","owner","duplicate","axis"])
def test_anchored_distance_direction_and_ownership_still_strict(change):
    parent,graph,old,fresh=_anchored_measurement_case()
    if change=="anchors":fresh["anchors"].reverse()
    elif change=="value":fresh["value"]=-65.
    elif change=="owner":fresh["entities"]=["y","unrelated"]
    elif change=="duplicate":fresh["entities"]=["x","x"]
    elif change=="axis":fresh["anchors"][0]["axis"]="x"
    assert repair.required_constraints_preserved(parent,graph,[old],[fresh])["passed"] is False


def _mock_stages(monkeypatch, candidate, current, *, strict=True, parent_ink=True, materialized=True):
    calls = []
    def binding(*args,**kwargs):
        output=Path(args[4]);output.mkdir(parents=True,exist_ok=True)
        result={"constraints":deepcopy(current)}
        (output/'constraint-bindings.json').write_text(json.dumps(result),encoding='utf8')
        inventory={'all_candidates':[{**deepcopy(row),'local_reliable':True,'evidence':{}} for row in current]}
        (output/'binding-candidates.json').write_text(json.dumps(inventory),encoding='utf8')
        return result
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",binding)
    monkeypatch.setattr("contour_agent.parametric_pipeline._solver_source_observation",
        lambda *args: {"boundary_error_budget": {"max_mm": .8}})
    def solve(graph, rows, **kwargs):
        calls.append((deepcopy(graph), deepcopy(rows), kwargs))
        result={"accepted": strict, "status": "accepted" if strict else "candidate_rejected",
            "entities": deepcopy(graph["entities"]), "constraints": [{**r, "passed": True} for r in rows]}
        output=Path(kwargs['output_dir']);output.mkdir(parents=True,exist_ok=True)
        (output/'parametric-solve.json').write_text(json.dumps(result),encoding='utf8')
        return result
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric", solve)
    def source(image, document, baseline, graph, entities):
        is_parent = "source_evidence" not in graph
        return {"passed": parent_ink if is_parent else True, "before": {"support_fraction": 1.},
            "after": {"support_fraction": 1.}}
    monkeypatch.setattr("contour_agent.parametric_pipeline._solved_source_validation", source)
    monkeypatch.setattr("contour_agent.source_radius_hypotheses.graph_from_solution",
        lambda graph, *args: deepcopy(graph))
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation",
        lambda *args: {"passed": materialized})
    return calls


def test_valid_strict_result_is_only_a_seed_and_preserves_original_problem(tmp_path, monkeypatch):
    candidate, parent, old, current, source = _case()
    initial = deepcopy(candidate)
    calls = _mock_stages(monkeypatch, candidate, current)
    updated, receipt = repair.repair_source_owned_line_seed("unused", {}, {}, candidate, tmp_path,
        parent_graph=parent, parent_constraints=old, source_validation=source, budget_seconds=90.)
    assert updated is not None and receipt["accepted"] is True
    assert receipt["publication_approved"] is False and receipt["formal_binding_required"] is True
    assert receipt["original_initial_source_validation"] == source
    assert candidate == initial
    assert calls[0][0] == initial["graph"] and calls[0][2]["source_warm_start"] is True
    assert calls[0][2]["source_observation"]["boundary_error_budget"]["max_mm"] == .8
    assert calls[0][2]["source_observation"]["boundary_error_budget"]["max_wall_seconds"] <= 90.


@pytest.mark.parametrize("failure", ["strict", "parent_ink", "materialized", "angle", "old_radius", "budget"])
def test_failed_stage_never_replaces_parent_or_yields_publishable_seed(tmp_path, monkeypatch, failure):
    candidate, parent, old, current, source = _case()
    initial = deepcopy(candidate)
    if failure == "angle": current.pop()
    if failure == "old_radius": current.pop(0)
    calls = _mock_stages(monkeypatch, candidate, current, strict=failure != "strict",
        parent_ink=failure != "parent_ink", materialized=failure != "materialized")
    updated, receipt = repair.repair_source_owned_line_seed("unused", {}, {}, candidate, tmp_path,
        parent_graph=parent, parent_constraints=old, source_validation=source,
        budget_seconds=0. if failure == "budget" else 90.)
    assert updated is None and receipt["accepted"] is False and receipt["publication_approved"] is False
    assert candidate == initial
    if failure in {"angle", "old_radius", "budget"}: assert calls == []


def test_moved_line_seed_reuses_original_solver_domain_and_hashes(tmp_path, monkeypatch):
    from contour_agent import parametric_pipeline as pipeline
    candidate,parent,old,current,source=_case()
    _mock_stages(monkeypatch,candidate,current)
    updated,_=repair.repair_source_owned_line_seed("unused",{}, {},candidate,tmp_path,
        parent_graph=parent,parent_constraints=old,source_validation=source,budget_seconds=90.)
    updated['graph']['entities'][0]['end']=[6.,6.]
    domain,comparator=repair.line_seed_problem_graphs(updated['graph'])
    assert domain == candidate['graph']
    calls=[]
    def solve(graph,rows,*,source_observation,output_dir,seed_entities):
        calls.append((deepcopy(graph),deepcopy(seed_entities)))
        return {'accepted':False}
    pipeline._solve_with_source_observation(solve,updated['graph'],current,{},tmp_path/'formal')
    assert calls[0][0] == candidate['graph']
    assert calls[0][1] == updated['graph']['entities']
    (tmp_path/'initial-topology.json').write_text('{}',encoding='utf8')
    with pytest.raises(ValueError,match='snapshot_changed'):
        repair.line_seed_problem_graphs(updated['graph'])


def test_changed_seed_entity_identity_cannot_reuse_original_domain(tmp_path, monkeypatch):
    candidate,parent,old,current,source=_case()
    _mock_stages(monkeypatch,candidate,current)
    updated,_=repair.repair_source_owned_line_seed("unused",{}, {},candidate,tmp_path,
        parent_graph=parent,parent_constraints=old,source_validation=source,budget_seconds=90.)
    updated['graph']['entities'][0]['id']='unrelated'
    with pytest.raises(ValueError,match='domain_identity_changed'):
        repair.line_seed_problem_graphs(updated['graph'])


def test_later_api_abstention_retains_revalidated_seed_angle_and_old_decision(tmp_path,monkeypatch):
    candidate,parent,old,current,source=_case()
    _mock_stages(monkeypatch,candidate,current)
    updated,_=repair.repair_source_owned_line_seed('unused',{}, {},candidate,tmp_path/'seed',
        parent_graph=parent,parent_constraints=old,source_validation=source,budget_seconds=90.)
    fresh={'constraints':[deepcopy(current[0])],
        'bindings':[{'record_id':'a24','accepted':False,'reason':'api_abstained'}],
        'counts':{'recognized_dimensions':2}}
    def certificate(*args):
        assert args[3]==candidate['graph'] and args[6]==current
        return {'passed':True,'constraints':[deepcopy(current[1])],'checks':[]}
    monkeypatch.setattr('contour_agent.source_radius_hypotheses.carried_binding_certificate',certificate)
    # This fixture isolates receipt handoff. Real finite ownership/native
    # signature checks are exercised by the source-radius certificate tests.
    monkeypatch.setattr('contour_agent.parametric_pipeline._constraint_signatures',
        lambda graph,rows:{(r['kind'],r.get('record_id'),tuple(r['entities']),r['value'],r.get('reference_axis')) for r in rows})
    bound,inventory,receipt=repair.retain_line_seed_bindings('unused',{}, {},updated['graph'],
        fresh,{},tmp_path/'fresh')
    assert len(bound['constraints'])==2
    assert bound['constraints'][-1]['value']==24.
    assert bound['constraints'][-1]['current_API_reconfirmed'] is False
    assert bound['bindings'][0]['accepted'] is False
    assert bound['counts']['bound_source_records']==2
    assert receipt['carried_count']==1 and fresh['constraints']==[current[0]]
    (tmp_path/'seed/solve/parametric-solve.json').write_text('{}',encoding='utf8')
    with pytest.raises(ValueError,match='certificate_changed'):
        repair.retain_line_seed_bindings('unused',{}, {},updated['graph'],fresh,{},tmp_path/'failed')


def test_positive_counterevidence_never_reinstates_old_seed_binding(tmp_path,monkeypatch):
    candidate,parent,old,current,source=_case()
    _mock_stages(monkeypatch,candidate,current)
    updated,_=repair.repair_source_owned_line_seed('unused',{}, {},candidate,tmp_path/'seed',
        parent_graph=parent,parent_constraints=old,source_validation=source,budget_seconds=90.)
    monkeypatch.setattr('contour_agent.source_radius_hypotheses.carried_binding_certificate',
        lambda *args:{'passed':False,'constraints':[],'reason':'new_positive_source_contradiction'})
    with pytest.raises(ValueError,match='not_retained'):
        repair.retain_line_seed_bindings('unused',{}, {},updated['graph'],{'constraints':[]},{},tmp_path/'fresh')

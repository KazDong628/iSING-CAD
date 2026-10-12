"""Structural completion must rebind and retain every prior typed obligation."""
from copy import deepcopy
import json
import numpy as np
import pytest

from contour_agent import constraint_binding as binding, parametric_pipeline as pipeline
from contour_agent import topology_candidates as candidates, topology_editing as editing
from contour_agent.parametric_solver import solve_parametric
from contour_agent.source_topology_completion import complete_source_topology, prioritize_structural_obligations
from contour_agent import source_topology_completion as completion
from test_parametric_solver import rectangle


@pytest.mark.parametrize('retain_known,source_pass,native_pass,accepted',[
    (True,True,True,True),(False,True,True,False),
    (True,False,True,False),(True,True,False,False)])
@pytest.mark.parametrize('recordless',['no','yes'])
@pytest.mark.parametrize('max_rounds',[1,2])
def test_structural_candidate_requires_all_unchanged_gates(tmp_path,monkeypatch,
        retain_known,source_pass,native_pass,accepted,recordless,max_rounds):
    graph=rectangle();graph.update(source_sha256='source',ground_truth_used=False)
    known=dict(id='known',kind='horizontal',record_id=None,entities=['g0'],nodes=[],value=0.,source='source_geometry')
    angle=dict(id='angle',kind='angle',record_id='r1',entities=['g1'],nodes=[],value=90.,
               source='ocr_local_binding',reference_axis='horizontal')
    if recordless=='yes':
        angle=dict(id='axis',kind='horizontal',record_id=None,entities=['g2'],nodes=[],value=0.,source='source_geometry')
        monkeypatch.setattr(completion,'unconfirmed_curved_joins',lambda *a:[['g0','g1']])
    bound=dict(constraints=[known],bindings=[],radius_binding_coverage={'unresolved':[]})
    inv={'all_records':[{'id':'r1','parsed':{'kind':'angle','nominal':90.}}] if recordless=='no' else []}
    solution=solve_parametric(graph,[known]);original=deepcopy((graph,bound,solution,inv))
    child=deepcopy(graph);child['candidate_id']='edited'
    monkeypatch.setattr(candidates,'_read_image',lambda *a:np.zeros((20,20,3),dtype=np.uint8))
    monkeypatch.setattr(candidates,'_affines',lambda *a:(object(),object(),1.))
    monkeypatch.setattr(candidates,'_base_source_ring',lambda *a:np.zeros((4,2)))
    monkeypatch.setattr(candidates,'_base_source_entities',lambda g,*a:g['entities'])
    monkeypatch.setattr(candidates,'_annotation_inventory',lambda *a:([],{}))
    monkeypatch.setattr(candidates,'_render_overlay',lambda *a:None)
    monkeypatch.setattr(editing,'_fresh_edit_source_graph',lambda g,*a:(deepcopy(g),{}))
    monkeypatch.setattr(editing,'propose_annotation_arc_edits',lambda *a,**kw:[{'action':'refit_entity_as_line','entity_ids':['g1']}])
    monkeypatch.setattr(editing,'execute_topology_edits',lambda *a:([{'id':'edited','graph':child}],{}))
    calls=[]
    def analyze(image,doc,baseline,g,output,**kw):
        calls.append('binding');output.mkdir(parents=True)
        path=output/'binding-candidates.json';path.write_text(json.dumps(inv),'utf8')
        return dict(constraints=([known] if retain_known else [])+[angle],bindings=[],inventory_artifact=str(path))
    monkeypatch.setattr(binding,'analyze_constraint_bindings',analyze)
    def solve(image,doc,baseline,g,constraints,output,**kw):
        calls.append('solve');return solve_parametric(g,constraints)
    monkeypatch.setattr(pipeline,'_final_source_multistart',solve)
    monkeypatch.setattr(pipeline,'_solved_source_validation',lambda *a:{'passed':source_pass})
    def export(*a,**kw):
        calls.append('export')
        return {'validation':{k:{'passed':native_pass,'dxf_readback_performed':True} for k in (
            'exact_radius_validation','strict_relation_validation','native_measurement_validation',
            'native_source_angle_validation','native_axis_relation_validation')}}
    monkeypatch.setattr(pipeline,'export_parametric',export)
    result=complete_source_topology('source.png',{}, {},graph,bound,solution,inv,tmp_path,max_rounds=max_rounds)
    assert result[-1]['accepted']==accepted
    assert (graph,bound,solution,inv)==original
    if not accepted:assert result[:4]==(graph,bound,solution,inv)
    if not retain_known:assert calls==['binding']
    assert not result[-1]['reference_geometry_used']
    assert not result[-1]['acceptance_thresholds_changed']
    assert len([call for call in calls if call=='binding'])==1
    if max_rounds==2 and not accepted:
        assert len(result[-1]['rounds'])==2
    if len(result[-1]['rounds'])==2:
        assert result[-1]['rounds'][1]['operations']==[]


def test_rejected_previous_solution_cannot_trigger_network_or_edit(tmp_path):
    result=complete_source_topology('missing.png',{}, {},{}, {},{'accepted':False},{},tmp_path)
    assert not result[-1]['accepted'] and result[-1]['rounds']==[]


def test_line_radius_conflict_is_not_starved_by_repeated_existing_arc_refits():
    graph={'entities':[{'id':'line','type':'LINE'},{'id':'arc','type':'ARC'}],
        'annotation_support':[{'record_id':'R16','kind':'radius','candidate_entity_id':'line'},
                              {'record_id':'R6','kind':'radius','candidate_entity_id':'arc'}]}
    old={'action':'refit_chain_as_annotated_arc','record_id':'R6','entity_ids':['arc']}
    repair={'action':'insert_annotated_fillet','record_id':'R16','entity_ids':['line']}
    assert prioritize_structural_obligations(graph,[old,repair],{'R16','R6'})[0]==repair


def test_missing_angle_precedes_repair_of_an_already_bound_angle():
    old={'action':'restore_annotated_line_support','record_id':'r022','entity_ids':['g007']}
    missing={**old,'record_id':'r035','entity_ids':['g016']}
    assert prioritize_structural_obligations({},[old,missing],set(),missing_angles={'r035'})[0]==missing


def test_already_bound_angle_is_not_a_goal_without_a_pending_contact_dependency():
    op={'action':'restore_annotated_line_support','record_id':'r022','entity_ids':['g007']}
    pending=[['g013','g014']]
    assert not completion.proposal_targets_obligation(op,{'r035'},pending,{'r022'})
    assert completion.proposal_targets_obligation({**op,'record_id':'r035'},{'r035'},pending,{'r022'})
    assert completion.proposal_targets_obligation({**op,'entity_ids':['g013']},{'r035'},pending,{'r022'})


def test_unknown_contacts_are_reviews_even_below_visual_jump_threshold():
    graph={'entities':[{'id':'a','type':'LINE'},{'id':'b','type':'ARC'},{'id':'c','type':'LINE'}]}
    assert completion.unconfirmed_curved_joins(graph,{'constraints':[]})==[['a','b'],['b','c']]
    bound={'constraints':[{'kind':'tangent','entities':['a','b']}]}
    assert completion.unconfirmed_curved_joins(graph,bound)==[['b','c']]
    assert all(c.get('tangency_required') is not True for c in bound['constraints'])

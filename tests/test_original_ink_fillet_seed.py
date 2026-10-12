"""A simplified raster corner cannot veto a source-owned exact radius seed."""
import copy
import math

import cv2
import numpy as np
import pytest

from contour_agent import topology_editing as editing
from contour_agent.topology import _StrokeEvidence,stroke_support_fraction,_tangent
from contour_agent.vectorize import _sample_entities
from test_fillet_preserves_finite_line_supports import _case


@pytest.mark.parametrize('rotation,reflection',[(0.,1.),(37.,1.),(13.,-1.)])
def test_original_ink_admits_exact_corner_with_sharp_mask(rotation,reflection):
    observed,selected,binding=_case(rotation=rotation,reflection=reflection)
    # Original image contains the physical round. The input mask was simplified
    # into a sharp corner; it is deliberately not used to choose the R value.
    shift=np.array([100.,160.])-observed.min(axis=0)
    observed+=shift
    for e in selected:
        for key in ['start','end']:e[key]=(np.array(e[key])+shift).tolist()
    binding['target_source_px']=(np.array(binding['target_source_px'])+shift).tolist()
    gray=np.full((600,600),255,np.uint8)
    cv2.polylines(gray,[np.rint(observed).astype(np.int32)],False,0,2)
    evidence=_StrokeEvidence(gray,[],1.)
    def check(arc):
        points=_sample_entities([arc],max_step_px=.25)[0]
        metrics=evidence.summarize(points)
        return {**metrics,'passed':stroke_support_fraction(metrics)>=.60 and metrics['p90_edge_distance_px']<=2.,
                'ground_truth_used':False}
    sharp=_sample_entities(selected,max_step_px=.25)[0];old=copy.deepcopy(selected)
    with pytest.raises(ValueError,match='source_does_not_support_exact_annotated_fillet'):
        editing._replacement_chain(sharp,'insert_annotated_fillet',.5,selected,
            annotated_radius_px=20.,radius_binding=binding)
    result=editing._replacement_chain(sharp,'insert_annotated_fillet',.5,selected,
        annotated_radius_px=20.,radius_binding=binding,source_fillet_validator=check)
    assert selected==old and result[1]['radius']==20.
    assert [e['type'] for e in result]==['LINE','ARC','LINE']
    assert float(_tangent(result[0],True)@_tangent(result[1],False))==pytest.approx(1.,abs=1e-12)
    assert float(_tangent(result[1],True)@_tangent(result[2],False))==pytest.approx(1.,abs=1e-12)
    assert result[1]['source_refinement']['mask_path_used_as_numeric_evidence'] is False


@pytest.mark.parametrize('problem',['no_ink','wrong_arrow','oversized','unverified_arrow'])
def test_original_ink_seed_does_not_invent_radius_or_erase_finite_support(problem):
    _,selected,binding=_case();sharp=_sample_entities(selected,max_step_px=.25)[0]
    radius=20.
    if problem=='wrong_arrow':binding['target_source_px']=[0.,0.]
    if problem=='oversized':radius=110.
    if problem=='unverified_arrow':binding['arrowhead_verified']=False
    def check(arc):return dict(passed=problem!='no_ink',ground_truth_used=False)
    with pytest.raises(ValueError):
        editing._replacement_chain(sharp,'insert_annotated_fillet',.5,selected,
            annotated_radius_px=radius,radius_binding=binding,source_fillet_validator=check)


def test_observed_arrow_tip_is_separate_from_projection_on_missing_mask_corner():
    _,selected,binding=_case()
    observed=copy.deepcopy(binding['target_source_px'])
    binding['verified_arrow_tip_source_px']=observed
    binding['target_source_px']=copy.deepcopy(selected[0]['end'])
    sharp=_sample_entities(selected,max_step_px=.25)[0]
    result=editing._replacement_chain(sharp,'insert_annotated_fillet',.5,selected,
        annotated_radius_px=20.,radius_binding=binding,
        source_fillet_validator=lambda arc:{'passed':True,'ground_truth_used':False})
    assert result[1]['radius']==20.
    assert result[1]['radius_binding']['target_source_px']!=observed
    assert result[1]['radius_binding']['verified_arrow_tip_source_px']==observed


def _source_corner_admission_case():
    from contour_agent.source_fillet_seed import two_line_fillet_seed
    observed,selected,binding=_case()
    shift=np.array([120.,170.])-observed.min(axis=0);observed+=shift
    for e in selected:
        for k in ('start','end'):e[k]=(np.array(e[k])+shift).tolist()
    binding['target_source_px']=(np.array(binding['target_source_px'])+shift).tolist()
    rows=two_line_fillet_seed(selected,20.,.5,binding,lambda arc:dict(passed=True,ground_truth_used=False))
    for i,row in enumerate(rows):row.update(id=str(i),start_node=str(i),end_node=str(i+1))
    gray=np.full((600,600),255,np.uint8)
    cv2.polylines(gray,[np.rint(observed).astype(np.int32)],False,0,2)
    tip=np.asarray(binding['target_source_px']);radial=tip-np.asarray(rows[1]['center'])
    leader=dict(record_id=binding['record_id'],arrowhead_verified=True,
        shaft_evidence=dict(verified=True),source_arrow_ownership=dict(status='globally_unique_source_claim'),
        arrowhead=dict(tip_px=tip.tolist(),direction_px=(radial/np.linalg.norm(radial)).tolist()))
    graph=dict(entities=rows,source_grid_pitch_px=1.,proposal_tolerance_px=.5)
    return gray,graph,leader


def test_fresh_original_image_rechecks_constructed_radius_instead_of_sharp_mask():
    from contour_agent.source_fillet_seed import original_ink_fillet_feasibility
    from contour_agent.reconstruction_priority import source_first_model
    gray,graph,leader=_source_corner_admission_case()
    result=original_ink_fillet_feasibility(gray,[],graph['entities'][1],20.,leader,
        source_first_model({}),graph,np.asarray,.5)
    assert result['passed'] and result['status']=='feasible_pending_joint_solve'
    assert result['mask_path_used_as_numeric_evidence'] is False
    assert result['reference_accuracy_verified'] is False
    # Metadata cannot substitute for current ink or a globally owned arrow.
    gray[:]=255
    assert not original_ink_fillet_feasibility(gray,[],graph['entities'][1],20.,leader,
        source_first_model({}),graph,np.asarray,.5)['passed']


@pytest.mark.parametrize('problem',['radius','ownership','tangency','legacy'])
def test_source_corner_admission_rejects_changed_known_attribute_or_unverified_evidence(problem):
    from contour_agent.source_fillet_seed import original_ink_fillet_feasibility
    from contour_agent.reconstruction_priority import source_first_model
    gray,graph,leader=_source_corner_admission_case();model=source_first_model({})
    if problem=='radius':graph['entities'][1]['radius']=19.
    if problem=='ownership':leader['source_arrow_ownership']['status']='ambiguous'
    if problem=='tangency':graph['entities'][0]['start'][1]+=15.
    if problem=='legacy':model={}
    result=original_ink_fillet_feasibility(gray,[],graph['entities'][1],20.,leader,model,graph,np.asarray,.5)
    assert result is None if problem=='legacy' else result['passed'] is False


@pytest.mark.parametrize('rotation,reflection',[(0.,1.),(37.,1.),(13.,-1.)])
def test_existing_line_arc_line_reinsertion_has_original_ink_seed(rotation,reflection):
    observed,sharp,binding=_case(rotation=rotation,reflection=reflection)
    from contour_agent.source_fillet_seed import two_line_fillet_seed
    selected=two_line_fillet_seed(sharp,20.,.5,binding,lambda arc:dict(passed=True,ground_truth_used=False))
    selected[1]['radius_binding']=copy.deepcopy(binding)
    # A sharp oracle-mask corner differs from the actual round in the source.
    # Neither this mask nor the previous arc's radius chooses the nominal R.
    mask=_sample_entities(sharp,max_step_px=.25)[0]
    original=copy.deepcopy(selected)
    result=editing._replacement_chain(mask,'insert_annotated_fillet',.5,selected,
        annotated_radius_px=20.,radius_binding=binding,
        source_fillet_validator=lambda arc:dict(passed=True,ground_truth_used=False))
    assert selected==original and result[1]['radius']==20.
    assert result[1]['source_refinement']['replaced_middle_arc'] is not None
    assert result[1]['source_refinement']['requires_fresh_binding_and_joint_solve'] is True
    for a,b in zip(result,result[1:]):
        assert float(_tangent(a,True)@_tangent(b,False))==pytest.approx(1.,abs=1e-12)
    with pytest.raises(ValueError,match='original_ink_not_supported'):
        editing._replacement_chain(mask,'insert_annotated_fillet',.5,selected,
            annotated_radius_px=20.,radius_binding=binding,
            source_fillet_validator=lambda arc:dict(passed=False,ground_truth_used=False))

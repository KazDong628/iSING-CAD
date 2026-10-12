"""Scope conflicts use independently seen arrows and the immutable input mask."""
from copy import deepcopy

import cv2
import numpy as np

import contour_agent.constraint_binding as binding
import contour_agent.source_arrow_localization as localization
from test_source_arrow_ownership import image, record, observation


def scope_source():
    gray=image();row=record()
    row["source_arrow_proposals"]=[{"record_id":row["id"],"tip_px":[100.,75.],"shaft_px":[229.,75.]}]
    boundary=np.array([[160.,10.],[390.,10.],[390.,150.],[160.,150.],[160.,10.]])
    return gray,row,boundary


def inner_claim():
    result=observation(tip=(180.,120.),direction=(-.707,.707))
    result["segment_px"]=[[229.,80.],[180.,120.]]
    return result


def test_real_source_arrow_outside_mask_withholds_different_inside_hatch_claim():
    gray,row,boundary=scope_source()
    scope=localization.observe_source_arrow_scope(gray,row,boundary,4.)
    assert scope["status"]=="outside_selected_boundary_or_scope_conflict",scope
    proof=scope["observations"][0]
    assert proof["independent_source_verified"] and proof["source_arrow"]["target_conditioned_detection"] is False
    assert proof["signed_distance_to_input_mask_px"] < -50.
    assert proof["source_text_shaft_attachment"]["strong_text_adjacency"]
    kept,rejected=localization.guard_source_arrow_scope(gray,row,[inner_claim()],scope,4.)
    assert kept==[] and len(rejected)==1
    assert rejected[0]["reason"]=="outside_selected_boundary_or_scope_conflict"
    assert rejected[0]["recognized_annotation_retained"] and not rejected[0]["out_of_scope_exemption_granted"]


def test_api_claim_of_outside_arrow_has_no_veto_without_original_ink():
    gray,row,boundary=scope_source();gray[:, :210]=255
    scope=localization.observe_source_arrow_scope(gray,row,boundary,4.)
    assert scope["status"]=="checked_no_far_outside_source_proof"
    observations=[inner_claim()]
    kept,rejected=localization.guard_source_arrow_scope(gray,row,observations,scope,4.)
    assert kept==observations and rejected==[]


def test_same_physical_shaft_is_not_overridden_by_a_distant_tip_hypothesis():
    gray,row,boundary=scope_source()
    scope=localization.observe_source_arrow_scope(gray,row,boundary,4.)
    same=observation(tip=(180.,75.),direction=(-1.,0.));same["segment_px"]=[[229.,75.],[180.,75.]]
    kept,rejected=localization.guard_source_arrow_scope(gray,row,[same],scope,4.)
    assert kept==[same] and rejected==[]


def test_arrow_near_mask_boundary_is_unresolved_without_scope_veto():
    gray,row,boundary=scope_source();boundary[boundary[:,0]==160.,0]=102.
    scope=localization.observe_source_arrow_scope(gray,row,boundary,4.)
    assert scope["status"]=="checked_no_far_outside_source_proof"
    assert scope["observations"][0]["status"]=="near_boundary_without_scope_exclusion"
    assert abs(scope["observations"][0]["signed_distance_to_input_mask_px"])<5.
    assert localization.guard_source_arrow_scope(gray,row,[inner_claim()],scope,4.)==([inner_claim()],[])


def test_inside_source_arrow_remains_eligible_and_nominal_does_not_change_scope():
    gray,row,boundary=scope_source();boundary[boundary[:,0]==160.,0]=20.
    scope=localization.observe_source_arrow_scope(gray,row,boundary,4.)
    assert scope["observations"][0]["status"]=="inside_selected_boundary"
    changed=deepcopy(row);changed["parsed"]["nominal"]=999.
    assert localization.observe_source_arrow_scope(gray,changed,boundary,4.)==scope
    assert localization.guard_source_arrow_scope(gray,row,[inner_claim()],scope,4.)==([inner_claim()],[])


def test_missing_immutable_mask_never_substitutes_current_edited_graph():
    gray,row,_=scope_source()
    scope=localization.observe_source_arrow_scope(gray,row,None,4.)
    assert scope["status"]=="not_checked_no_immutable_source_mask_boundary"
    assert not scope["immutable_source_mask_used"]
    assert localization.guard_source_arrow_scope(gray,row,[inner_claim()],scope,4.)==([inner_claim()],[])


def test_glyph_ownership_is_required_even_when_arrow_and_shaft_are_real():
    gray,row,boundary=scope_source();gray[50:90,230:301]=255
    scope=localization.observe_source_arrow_scope(gray,row,boundary,4.)
    assert scope["observations"][0]["status"]=="source_glyph_attachment_not_verified"
    assert not scope["observations"][0]["conflicting_candidate_veto_allowed"]


def test_target_conditioned_wrapper_still_refuses_the_same_off_boundary_arrow():
    gray,row,boundary=scope_source()
    proposal=row["source_arrow_proposals"][0]
    assert binding._verify_source_arrow_ink(gray,row,proposal,4.) is not None
    assert binding.verify_source_arrow_proposal(gray,row,proposal,boundary,4.) is None


def test_scope_then_target_snapping_reuses_bounded_hough_cache(monkeypatch):
    gray,row,boundary=scope_source();calls=[]
    row["source_arrow_proposals"][0].update(tip_px=[101.,74.],shaft_px=[245.,59.])
    original=localization.source_arrow_hypotheses
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(localization,"source_arrow_hypotheses",counted)
    cache={};localization.observe_source_arrow_scope(gray,row,boundary,4.,hypothesis_cache=cache)
    assert len(calls)==1
    localization.localize_source_arrow_proposal(gray,row,row["source_arrow_proposals"][0],boundary,4.,hypothesis_cache=cache)
    assert len(calls)==1


def test_scope_search_is_bounded_to_two_proposals_and_has_no_wall_clock_identity():
    gray,row,boundary=scope_source();row["source_arrow_proposals"]*=4
    scope=localization.observe_source_arrow_scope(gray,row,boundary,4.,hypothesis_cache={})
    assert len(scope["observations"])==2 and scope["maximum_proposals"]==2
    assert scope["proposal_count_beyond_scope_budget"]==2
    assert "elapsed_seconds" not in repr(scope)


def test_full_observation_pipeline_keeps_real_arrow_scope_before_internal_taper_targeting():
    gray,row,boundary=scope_source()
    shaft=np.array([229.,80.]);tip=np.array([180.,120.])
    direction=(tip-shaft)/np.linalg.norm(tip-shaft)
    normal=np.array([-direction[1],direction[0]])
    # A competing internal stroke with a taper passes the old source head and
    # full shaft checks, so this exercises the actual observation collection
    # path rather than only the guard helper or a trusted stub verdict.
    cv2.line(gray,tuple(shaft.astype(int)),tuple(tip.astype(int)),0,2)
    cv2.fillConvexPoly(gray,np.rint([tip,tip-direction*16+normal*5,
                                   tip-direction*16-normal*5]).astype(np.int32),0)
    graph={"entities":[{"id":"g000","type":"LINE","start":[175.,115.],"end":[185.,125.]}]}
    arguments=(gray,[row],graph,lambda points:np.asarray(points),[np.array([shaft,tip])],4.)
    before=binding._radius_source_observations(*arguments)
    assert before and all(r["record_id"]==row["id"] for r in before)
    rejected=[];audit={}
    after=binding._radius_source_observations(*arguments,rejections=rejected,ownership_audit=audit,
                                             source_boundary_points=boundary)
    assert not after
    assert any(item["reason"]=="outside_selected_boundary_or_scope_conflict" for item in rejected)
    assert audit["source_scope_observations"][0]["recognized_annotation_retained"]

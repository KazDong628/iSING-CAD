"""Radial-gap candidates require independent original-source ownership proofs."""
from copy import deepcopy
import math

import cv2
import numpy as np
import pytest

from contour_agent import radial_gap_binding as binding


def circular(identifier,radius,center=(30.,30.),start=30.,end=60.):
    def point(angle):
        return [center[0]+radius*math.cos(math.radians(angle)),
                center[1]+radius*math.sin(math.radians(angle))]
    return {"id":identifier,"type":"ARC","center":list(center),"radius":radius,
            "start":point(start),"end":point(end),"clockwise":False}


def fixture(monkeypatch):
    graph={"units":"mm","entities":[circular("inner",70.*math.sqrt(2.)),
                                          circular("outer",90.*math.sqrt(2.))]}
    records=[{"id":"r020","text":"20 +3/0","parsed":{"kind":"length","nominal":20.,
        "lower_deviation":0.,"upper_deviation":3.,"reference":False},
        "box":[[95.,80.],[115.,80.],[115.,95.],[95.,95.]]}]
    records.extend({"id":identifier,"parsed":{"kind":"radius","nominal":nominal}}
        for identifier,nominal in (("rR1",70.*math.sqrt(2.)),("rR2",90.*math.sqrt(2.))))
    direction=np.asarray([1.,1.])/math.sqrt(2.)
    witness={"record_id":"r020","endpoints_source_px":[[100.,100.],[120.,120.]],
        "direction_px":direction.tolist(),"arrow_pair_verified":True,"evidence":{
            "verified":True,"same_source_shaft":True,"arrow_directions_opposed":True,
            "label_association_unique":True,"nominal_used_to_rank":False,
            "nominal_used_to_set_span":False,"ground_truth_used":False,
            "endpoint_arrows":[{"length_px":20.},{"length_px":20.}]}}
    center={"anchor":{"type":"common_curve_center","entity_id":"inner","other_entity_id":"outer","axis":"x"},
        "evidence":{"verified":True,"center_semantics_verified":True,"nominal_used":False,
            "coordinates_locked":False,"required_center_equalities":["x","y"],
            "source_supported_entities":["inner","outer"],"source_radius_record_ids":["rR1","rR2"],
            "source_arrow_ids":["radius-arrow-1","radius-arrow-2"],
            "tail_support":[{"verified":True},{"verified":True}],"observed_source_center_px":[30.,30.]}}
    monkeypatch.setattr(binding,"oblique_dimension_witnesses",lambda *args:[deepcopy(witness)])
    monkeypatch.setattr(binding,"source_curve_center_locations",lambda *args,**kwargs:[deepcopy(center)])
    calls=[]
    def refine(gray,previous,contacts):
        calls.append(np.asarray(contacts).copy())
        return {**deepcopy(previous),"endpoints_source_px":np.asarray(contacts).tolist(),
            "evidence":{**deepcopy(previous["evidence"]),"contact_pixel_tapers_verified":True,
                        "requires_contact_tip_recheck":False}}
    monkeypatch.setattr(binding,"refine_oblique_arrow_contacts",refine)
    gray=np.full((220,220),255,np.uint8)
    return gray,records,graph,witness,center,calls


def run(data,**kwargs):
    gray,records,graph,*_=data
    diagnostics=[]
    candidates=binding.radial_gap_candidates(gray,records,graph,lambda p:np.asarray(p,float),3.,[],
                                              diagnostics=diagnostics,**kwargs)
    return candidates,diagnostics


def test_nominal_is_kept_while_owners_and_common_center_are_independently_verified(monkeypatch):
    data=fixture(monkeypatch);before=deepcopy(data[:5]);candidates,diagnostics=run(data)
    assert len(candidates)==1
    candidate=candidates[0]
    assert candidate["entities"]==["inner","outer"]
    assert candidate["kind"]=="radial_gap" and candidate["nodes"]==[]
    assert candidate["value"]==20.
    assert candidate["value"]!=pytest.approx(data[2]["entities"][1]["radius"]-data[2]["entities"][0]["radius"])
    assert candidate["measurement_policy"]=="source_tolerance_interval"
    assert candidate["evidence"]["required_center_equalities"]==["x","y"]
    assert candidate["evidence"]["fitted_radius_difference_used_as_dimension"] is False
    assert candidate["evidence"]["nominal_used_for_ownership"] is False
    assert data[-1]  # Original-pixel contact refiner is mandatory.
    assert diagnostics[-1]["status"]=="source_radial_gap_candidate_available"
    np.testing.assert_array_equal(data[0],before[0])
    assert data[1:5]==before[1:5]


def test_changing_dimension_and_radius_nominals_does_not_select_new_owners(monkeypatch):
    data=fixture(monkeypatch);first,_=run(data)
    data[1][0]["parsed"]["nominal"]=2000.
    data[1][1]["parsed"]["nominal"]=1.
    data[1][2]["parsed"]["nominal"]=5000.
    second,_=run(data)
    assert second[0]["entities"]==first[0]["entities"]
    assert second[0]["value"]==2000.
    assert second[0]["evidence"]["finite_contact_owners"]==first[0]["evidence"]["finite_contact_owners"]


def test_coarse_tip_close_to_fitted_arc_cannot_override_failed_fresh_pixel_check(monkeypatch):
    data=fixture(monkeypatch)
    direction=np.asarray(data[3]["direction_px"])
    data[3]["endpoints_source_px"][1]=(np.asarray([120.,120.])+16.35*direction).tolist()
    monkeypatch.setattr(binding,"refine_oblique_arrow_contacts",lambda *args:None)
    candidates,diagnostics=run(data)
    assert not candidates
    assert diagnostics[-1]["contact_pixel_recheck_attempts"]>0
    assert diagnostics[-1]["status"]=="source_contact_taper_recheck_failed"


@pytest.mark.parametrize("competitor",["duplicate_arc","contact_line"])
def test_each_contact_must_have_one_finite_arc_owner(monkeypatch,competitor):
    data=fixture(monkeypatch)
    if competitor=="duplicate_arc":
        other=deepcopy(data[2]["entities"][0]);other["id"]="duplicate"
    else:
        other={"id":"contact_line","type":"LINE","start":[98.,102.],"end":[102.,98.]}
    data[2]["entities"].append(other)
    candidates,diagnostics=run(data)
    assert not candidates
    assert diagnostics[-1]["status"]=="finite_arc_contact_owner_not_unique"


@pytest.mark.parametrize("defect",["single_center","nominal_used","missing_equalities","unverified_tail",
                                   "same_radius_arrow","withheld_radius","different_arc_pair"])
def test_fitted_or_incomplete_center_semantics_cannot_certify_concentricity(monkeypatch,defect):
    data=fixture(monkeypatch);center=data[4]
    if defect=="single_center":center["anchor"]["type"]="curve_center"
    elif defect=="nominal_used":center["evidence"]["nominal_used"]=True
    elif defect=="missing_equalities":center["evidence"]["required_center_equalities"]=["x"]
    elif defect=="unverified_tail":center["evidence"]["tail_support"][0]["verified"]=False
    elif defect=="same_radius_arrow":center["evidence"]["source_arrow_ids"]=["same","same"]
    elif defect=="withheld_radius":data[1][1]["source_text_admission"]={"status":"withheld"}
    else:center["anchor"]["other_entity_id"]="different"
    candidates,diagnostics=run(data)
    assert not candidates
    assert diagnostics[-1]["status"]=="independent_source_common_center_unresolved"


@pytest.mark.parametrize("source_center",[[30.,40.],[110.,110.]])
def test_shaft_must_pass_source_center_and_measure_same_radial_half_ray(monkeypatch,source_center):
    data=fixture(monkeypatch);data[4]["evidence"]["observed_source_center_px"]=source_center
    candidates,diagnostics=run(data)
    assert not candidates
    assert diagnostics[-1]["status"]=="source_dimension_shaft_not_a_common_radial_gap"


def test_missing_oblique_pair_remains_unknown_in_diagnostics(monkeypatch):
    data=fixture(monkeypatch)
    monkeypatch.setattr(binding,"oblique_dimension_witnesses",lambda *args:[])
    candidates,diagnostics=run(data)
    assert not candidates
    assert diagnostics==[{"record_id":"r020","status":"oblique_source_arrow_pair_unresolved","candidate_count":0}]


@pytest.mark.parametrize("defect",["reference","withheld","nan","reused_nominal","pixel_units"])
def test_unadmitted_source_semantics_or_units_cannot_generate_a_candidate(monkeypatch,defect):
    data=fixture(monkeypatch)
    if defect=="reference":data[1][0]["parsed"]["reference"]=True
    elif defect=="withheld":data[1][0]["source_text_admission"]={"status":"withheld"}
    elif defect=="nan":data[1][0]["parsed"]["nominal"]=math.nan
    elif defect=="reused_nominal":data[3]["evidence"]["nominal_used_to_rank"]=True
    else:data[2]["units"]="pixel"
    assert run(data)[0]==[]


def test_nonuniform_or_sheared_mapping_cannot_treat_ellipse_as_source_circle(monkeypatch):
    data=fixture(monkeypatch);diagnostics=[]
    assert not binding.radial_gap_candidates(data[0],data[1],data[2],
        lambda p:np.asarray(p,float)@np.asarray([[1.,.2],[0.,1.]]),3.,[],diagnostics=diagnostics)
    assert diagnostics[-1]["status"]=="radial_gap_source_mapping_unresolved"


def test_real_original_pixels_arrow_contacts_and_independent_radius_tails(tmp_path):
    from test_oblique_dimension_evidence import _drawing
    gray,records,contacts=_drawing(tmp_path,angle=37.)
    direction=(contacts[1]-contacts[0])/np.linalg.norm(contacts[1]-contacts[0])
    center=contacts[0]-100.*direction
    angle=math.degrees(math.atan2(direction[1],direction[0]))
    graph={"units":"mm","entities":[circular("inner",100.,center,angle-20.,100.),
                                          circular("outer",250.,center,-10.,angle+20.)]}
    records[0]["parsed"].update(nominal=150.,lower_deviation=0.,upper_deviation=3.)
    observations=[]
    for index,(identifier,radius,theta) in enumerate((("inner",100.,90.),("outer",250.,0.))):
        unit=np.asarray([math.cos(math.radians(theta)),math.sin(math.radians(theta))])
        tip=center+radius*unit;tail=tip+.45*(center-tip)
        cv2.line(gray,tuple(np.rint(tip).astype(int)),tuple(np.rint(center).astype(int)),0,2)
        record_id="rR"+str(index)
        records.append({"id":record_id,"parsed":{"kind":"radius","nominal":radius},
            "box":[[350.,350.],[370.,350.],[370.,370.],[350.,370.]]})
        observations.append({"record_id":record_id,"segment_px":[tip.tolist(),tail.tolist()],
            "arrowhead_verified":True,"arrowhead":{"verified":True,"tip_px":tip.tolist(),"direction_px":unit.tolist()},
            "shaft_evidence":{"verified":True},"source_arrow_ownership":{
                "status":"globally_unique_source_claim","physical_arrow_id":"radius-arrow-"+str(index)},
            "target_candidates":[{"entity_id":identifier,"entity_type":"ARC"}]})
    diagnostics=[]
    candidates=binding.radial_gap_candidates(gray,records,graph,lambda p:np.asarray(p,float),3.,observations,
                                              diagnostics=diagnostics)
    assert len(candidates)==1,diagnostics
    assert candidates[0]["value"]==150.
    assert candidates[0]["entities"]==["inner","outer"]
    assert candidates[0]["evidence"]["source_common_center"]["method"]=="two_source_owned_radius_shafts_common_finite_termination_v2"
    assert candidates[0]["evidence"]["source_oblique_dimension"]["evidence"]["contact_pixel_tapers_verified"]

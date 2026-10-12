import copy
import hashlib
import math

import cv2
import numpy as np
import pytest

from contour_agent.source_contact_construction import tangent_arc_line_arc
from contour_agent.source_contact_evidence import contact_construction_relations
from contour_agent.topology import _tangent


def source_chain():
    angle=math.radians(30.);d=np.array([math.cos(angle),math.sin(angle)])
    n=np.array([-d[1],d[0]]);p=np.array([100.,100.]);q=p+30*d;theta=angle-math.pi/2
    def arc(contact,r,angles):return contact+r*n+r*np.c_[np.cos(angles),np.sin(angles)]
    first=arc(p,15.,np.linspace(theta-.7,theta,35))
    last=arc(q,65.,np.linspace(theta,theta+.8,70))
    points=np.vstack([first,np.linspace(p,q,30)[1:-1],last])
    observation=dict(record_id="angle",nominal=30.,reference_axis="horizontal",
        source_line=dict(start_px=p.tolist(),end_px=q.tolist()),
        evidence=dict(method="source_angular_arrows_axis_and_straight_support_v1",
                      reference_arrow=dict(verified=True),target_arrow=dict(verified=True)))
    return points,observation


def test_contact_equations_keep_radii_angle_incidence_and_tangency_exact():
    points,observation=source_chain()
    chain=tangent_arc_line_arc(points,observation,[15.,65.],[False,False],2.)
    assert chain is not None
    assert chain[0]["radius"]==15. and chain[2]["radius"]==65.
    assert np.allclose(chain[0]["start"],points[0]) and np.allclose(chain[-1]["end"],points[-1])
    for arc in [chain[0],chain[2]]:
        for side in ["start","end"]:
            assert abs(np.linalg.norm(np.array(arc[side])-arc["center"])-arc["radius"])<1e-10
    assert _tangent(chain[0],True)@_tangent(chain[1],False)>1-1e-12
    assert _tangent(chain[1],True)@_tangent(chain[2],False)>1-1e-12
    delta=np.array(chain[1]["end"])-chain[1]["start"]
    assert abs(math.degrees(math.atan2(delta[1],delta[0]))-30.)<1e-10
    assert chain[1]["source_contact_construction"]["publication_approved"] is False


def test_incompatible_source_direction_or_radius_cannot_be_constructed():
    points,observation=source_chain();observation["nominal"]=70.
    assert tangent_arc_line_arc(points,observation,[15.,65.],[False,False],2.) is None
    assert tangent_arc_line_arc(points,observation,[-1.,65.],[False,False],2.) is None


def test_contact_search_keeps_source_feasibility_as_well_as_nominal_equations():
    points,observation=source_chain()
    direction=np.array([math.cos(math.radians(30.)),math.sin(math.radians(30.))])
    normal=np.array([-direction[1],direction[0]])
    offset=float(np.array(observation['source_line']['start_px'])@normal)
    calls=[]
    def observed_stroke(line):
        gap=abs(float(np.array(line['start'])@normal)-offset)
        calls.append(gap)
        return .3<=gap<=1.2
    chain=tangent_arc_line_arc(points,observation,[15.,65.],[False,False],2.,
                               source_line_validator=observed_stroke)
    assert chain is not None and observed_stroke(chain[1])
    assert len(calls)<=16*(33+49)+1
    assert chain[1]['source_contact_construction']['source_line_support_checked_during_search']
    assert _tangent(chain[0],True)@_tangent(chain[1],False)>1-1e-12
    assert _tangent(chain[1],True)@_tangent(chain[2],False)>1-1e-12
    assert tangent_arc_line_arc(points,observation,[15.,65.],[False,False],2.,
                               source_line_validator=lambda line:False) is None


def fixture(tmp_path):
    points,observation=source_chain()
    chain=tangent_arc_line_arc(points,observation,[15.,65.],[False,False],2.)
    chain.append(dict(type="LINE",start=chain[-1]["end"],end=chain[0]["start"]))
    nodes=[]
    for i,e in enumerate(chain):
        e.update(id=f"g{i}",start_node=f"v{i}",end_node=f"v{(i+1)%4}")
        x,y=e["start"];nodes.append(dict(id=f"v{i}",x=x,y=y,source_px=[x,y]))
    chain[1]["source_contact_construction"].update(radius_record_ids=["r15","r65"],angle_record_id="angle")
    gray=np.full((240,240),255,np.uint8)
    cv2.polylines(gray,[np.round(points).astype(np.int32)],False,0,2)
    image=tmp_path/'source.png';cv2.imwrite(str(image),gray)
    graph=dict(entities=chain,nodes=nodes,units="mm",source_grid_pitch_px=1.,proposal_tolerance_px=2.,
               source_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    model=dict(extraction=dict(raw_polyline_px=np.vstack([points,points[0]]).tolist()),scale=dict(pixels_per_mm=1.))
    candidates=[dict(id=f"c{i}",kind="radius",record_id=rid,entities=[eid],value=value,local_reliable=True,
        evidence=dict(leader=dict(arrowhead_verified=True),whole_primitive_radius=dict(passed=True)))
        for i,(rid,eid,value) in enumerate([("r15","g0",15.),("r65","g2",65.)])]
    candidates.append(dict(kind="angle",record_id="angle",entities=["g1"],value=30.,
                           reference_axis="horizontal",local_reliable=True))
    return gray,[],image,model,graph,lambda xy:np.asarray(xy,float),[],candidates


def test_source_design_contacts_need_fresh_nominals_source_ink_and_raw_boundary(tmp_path):
    args=fixture(tmp_path);relations=contact_construction_relations(*args)
    assert len(relations)==2 and all(r["local_reliable"] for r in relations)
    assert all(r["evidence"]["evidence_class"]=="source_bound_design_construction" for r in relations)
    assert all(r["evidence"]["independent_source_tangent_measurement"] is False for r in relations)


@pytest.mark.parametrize("change",["radius","angle","contact","ink","hash","boundary"])
def test_missing_or_tampered_contact_source_never_admits_relations(tmp_path,change):
    args=list(fixture(tmp_path))
    if change=="radius":args[-1][0]["local_reliable"]=False
    elif change=="angle":args[-1][-1]["value"]=31.
    elif change=="contact":args[4]["entities"][1]["source_contact_construction"]["constructed_contacts_px"][0][0]+=1.
    elif change=="ink":args[0][:]=255
    elif change=="hash":args[4]["source_sha256"]="changed"
    elif change=="boundary":args[3]["extraction"]["raw_polyline_px"]=(np.asarray(args[3]["extraction"]["raw_polyline_px"])+50.).tolist()
    assert contact_construction_relations(*args)==[]

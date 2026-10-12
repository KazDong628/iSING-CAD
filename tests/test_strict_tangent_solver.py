"""Strict admitted joint constraints, measured independently on returned geometry."""
import copy
import math

import numpy as np
import pytest

from contour_agent import parametric_solver as solver
from test_parametric_solver import rounded_graph, dimensions_and_relations


def angle_row(entity_id, value, axis="horizontal"):
    return {"id":"angle_"+entity_id,"kind":"angle","entities":[entity_id],"nodes":[],
            "value":value,"record_id":"r_"+entity_id,"source":"ocr_local_binding",
            "reference_axis":axis,"required":True,"nominal_source":"source_ocr",
            "source_arrow_verified":True}


def tangent_row(first, second, joint):
    return {"id":"tan_"+first+"_"+second,"kind":"tangent","entities":[first,second],
            "nodes":[joint],"value":None,"record_id":None,"source":"source_geometry"}


def independently_measure_joints(result):
    indexed={entity["id"]:entity for entity in result["entities"]}
    angles=[]
    for row in result["constraints"]:
        if row["kind"]!="tangent":continue
        vectors=[];points=[]
        for eid in row["entities"]:
            entity=indexed[eid];point=entity["end"] if entity["end_node"]==row["nodes"][0] else entity["start"]
            points.append(point)
            if entity["type"]=="LINE":vector=np.asarray(entity["end"])-entity["start"]
            else:
                radial=np.asarray(point)-entity["center"]
                vector=np.array([-radial[1],radial[0]])*(-1 if entity["clockwise"] else 1)
            assert np.linalg.norm(vector)>1e-9
            vectors.append(vector/np.linalg.norm(vector))
        assert math.dist(*points)<=1e-7
        a,b=vectors
        assert np.dot(a,b)>0
        angles.append(abs(math.degrees(math.atan2(a[0]*b[1]-a[1]*b[0],np.dot(a,b)))))
    return angles


def source_observation(graph, maximum_deviation=.5):
    samples=np.concatenate([solver._sample_entity(entity,31) for entity in graph["entities"]])
    return {"units":"mm","points":samples.tolist(),"provenance":"input_mask_boundary",
            "reference_dxf_read":False,"boundary_error_budget":{"units":"mm",
            "maximum_deviation":maximum_deviation,"sampling_step":.02,
            "source":"initial_curve_fit_total_deviation_budget_px"}}


def test_fixed_radius_and_verified_angle_and_tangent_are_joint_equalities():
    graph=rounded_graph(width=20,height=12,radius=3,noise=.018)
    theta=math.radians(15);rotation=np.array([[math.cos(theta),-math.sin(theta)],
                                           [math.sin(theta),math.cos(theta)]])
    for node in graph["nodes"]:
        node["x"],node["y"]=(rotation@np.array([node["x"],node["y"]])).tolist()
    for entity in graph["entities"]:
        for key in ("start","end","center"):
            if key in entity:entity[key]=(rotation@np.asarray(entity[key])).tolist()
    constraints=[row for row in dimensions_and_relations() if row["kind"] in {"radius","tangent"}]
    constraints.extend([angle_row("g000",15),angle_row("g002",15,"vertical")])
    before=copy.deepcopy(graph)
    result=solver.solve_parametric(graph,constraints)
    assert result["accepted"],result["validation"]
    assert graph==before
    assert max(independently_measure_joints(result))<=1e-7
    assert result["strict_tangent_contract"]["required_count"]==8
    assert result["strict_tangent_contract"]["satisfied"]
    assert result["strict_ocr_angle_contract"]["satisfied"]
    assert all(row["absolute_error_deg"]<=1e-7 for row in result["strict_ocr_angle_contract"]["checks"])
    assert all(entity["radius"]==3. for entity in result["entities"] if entity["type"]=="ARC")
    assert all(row["tolerance"]==1e-7 for row in result["constraints"] if row["kind"]=="tangent")


def test_mixed_arc_arc_and_line_arc_relations_are_strict_without_unbound_join_claims():
    graph=rounded_graph(width=20,height=12,radius=3,noise=0)
    arc=graph["entities"][1];center=np.array(arc["center"])
    midpoint=center+3*np.array([math.cos(-math.pi/4),math.sin(-math.pi/4)])
    original_end=arc["end"];original_node=arc["end_node"]
    arc["end"]=midpoint.tolist();arc["end_node"]="extra"
    following={**copy.deepcopy(arc),"id":"split","start":midpoint.tolist(),"end":original_end,
               "start_node":"extra","end_node":original_node}
    graph["entities"].insert(2,following)
    graph["nodes"].append({"id":"extra","x":float(midpoint[0]),"y":float(midpoint[1])})
    # A small independently declared radius change forces a nontrivial joint solve.
    constraints=[{"id":"rad_"+e["id"],"kind":"radius","entities":[e["id"]],"nodes":[],
                  "value":3.02,"record_id":"r_"+e["id"],"source":"ocr_local_binding"}
                 for e in graph["entities"] if e["type"]=="ARC"]
    for index,entity in enumerate(graph["entities"]):
        following=graph["entities"][(index+1)%len(graph["entities"])]
        constraints.append(tangent_row(entity["id"],following["id"],entity["end_node"]))
    result=solver.solve_parametric(graph,constraints)
    assert result["accepted"],result["validation"]
    assert max(independently_measure_joints(result))<=1e-7
    assert result["strict_tangent_contract"]["required_count"]==9
    assert result["strict_tangent_contract"]["unbound_joint_coverage_verified"] is False


def collinear_graph():
    points=[[0.,0.],[3.,.01],[7.,-.01],[10.,0.],[10.,5.],[0.,5.]]
    return {"units":"mm","nodes":[{"id":f"v{i}","x":p[0],"y":p[1]} for i,p in enumerate(points)],
            "entities":[{"id":f"g{i}","type":"LINE","start_node":f"v{i}",
                         "end_node":f"v{(i+1)%len(points)}","start":point,
                         "end":points[(i+1)%len(points)]} for i,point in enumerate(points)]}


@pytest.mark.parametrize("already_collinear",[False,True])
def test_redundant_angle_and_collinear_tangent_obligations_remain_certified(already_collinear):
    graph=collinear_graph()
    if already_collinear:
        for node in graph["nodes"][:4]:node["y"]=0.
        for entity in graph["entities"][:3]:entity["start"][1]=entity["end"][1]=0.
    constraints=[angle_row(f"g{i}",0) for i in range(3)]
    constraints.extend([tangent_row("g0","g1","v1"),tangent_row("g1","g2","v2")])
    result=solver.solve_parametric(graph,constraints)
    assert result["accepted"],result["validation"]
    assert max(independently_measure_joints(result))<=1e-7
    assert result["strict_tangent_contract"]["satisfied"]
    assert result["strict_ocr_angle_contract"]["required_count"]==3
    if already_collinear:
        system=result["diagnostics"]["strict_equality_system"]
        assert system["optimizer_basis_count"]<system["required_count"]
        assert system["all_original_rows_finally_certified"]


def test_conflicting_strict_angles_and_tangent_cannot_publish_a_compromise():
    graph=collinear_graph();before=copy.deepcopy(graph)
    constraints=[angle_row("g0",0),angle_row("g1",2),tangent_row("g0","g1","v1")]
    result=solver.solve_parametric(graph,constraints)
    assert not result["accepted"]
    assert result["entities"]==before["entities"]
    assert not result["validation"].get("strict_tangent_satisfied",False) or not result["validation"].get("strict_ocr_angle_satisfied",False)


def test_antiparallel_unit_vectors_do_not_pass_zero_cross_certificate():
    entities=[{"id":"a","type":"LINE","start_node":"v0","end_node":"v1","start":[0.,0.],"end":[1.,0.]},
              {"id":"b","type":"LINE","start_node":"v1","end_node":"v2","start":[1.,0.],"end":[0.,0.]}]
    result=solver._strict_tangent_contract(entities,[tangent_row("a","b","v1")])
    assert not result["satisfied"]
    assert result["checks"][0]["absolute_error_deg"]==180.


@pytest.mark.parametrize("success",[False,True])
def test_fake_optimizer_success_or_feasible_checkpoint_cannot_hide_point_one_degree_tangency(monkeypatch, success):
    graph=rounded_graph(width=20,height=12,radius=3,noise=.01)
    constraints=[row for row in dimensions_and_relations() if row["kind"]=="tangent"]
    def unrepaired(fun,x0,**kwargs):
        return solver.OptimizeResult(x=np.asarray(x0).copy(),jac=np.zeros_like(x0),success=success,
                                     status=0 if success else 9,nfev=1,nit=1,message="synthetic unrepaired iterate")
    monkeypatch.setattr(solver,"minimize",unrepaired)
    result=solver.solve_parametric(graph,constraints,source_observation=source_observation(graph))
    assert not result["accepted"]
    assert result["entities"]==graph["entities"]
    assert not result["strict_tangent_contract"]["satisfied"]
    assert not result["diagnostics"]["source_budget_search"]["feasible_checkpoint_found"]


def test_no_admitted_tangent_means_no_invented_smoothness_obligation():
    result=solver.solve_parametric(collinear_graph(),[angle_row("g0",0)])
    assert result["accepted"]
    assert result["strict_tangent_contract"]["required_count"]==0
    assert result["strict_tangent_contract"]["unbound_joint_coverage_verified"] is False


def test_nonconverged_but_independently_strict_feasible_checkpoint_is_retained(monkeypatch):
    graph=rounded_graph(width=20,height=12,radius=3,noise=0)
    constraints=[row for row in dimensions_and_relations() if row["kind"] in {"radius","tangent"}]
    constraints.append(angle_row("g000",0))
    actual=solver.minimize
    def force_optimality_unknown(fun,*args,**kwargs):
        result=actual(fun,*args,**kwargs)
        result.success=False;result.status=9;result.message="iteration limit after valid iterate"
        return result
    monkeypatch.setattr(solver,"minimize",force_optimality_unknown)
    result=solver.solve_parametric(graph,constraints,source_observation=source_observation(graph))
    assert result["accepted"],result["validation"]
    assert result["diagnostics"]["optimizer_converged"] is False
    assert result["diagnostics"]["independently_verified_feasible_checkpoint"]
    assert result["diagnostics"]["optimality_proven"] is False
    assert result["strict_tangent_contract"]["satisfied"]
    assert max(independently_measure_joints(result))<=1e-7
    assert result["strict_ocr_angle_contract"]["satisfied"]
    assert result["strict_radius_contract"]["satisfied"]

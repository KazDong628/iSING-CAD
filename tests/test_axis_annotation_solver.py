"""Drawing angle semantics must reach the numerical solve, never ARC tangents."""
import copy
import math

import numpy as np
import pytest

from contour_agent.parametric_solver import _constraint_value, _validate_inputs, solve_parametric
from contour_agent.parametric_pipeline import _constraint_signatures


def graph(angle=11.0):
    x=math.tan(math.radians(angle))*20.
    points=[(0.,0.),(10.,0.),(10.+x,20.),(0.,20.)]
    return {"units":"mm", "nodes":[{"id":f"v{i}","x":a,"y":b,"source_px":[a*8,200-b*8]} for i,(a,b) in enumerate(points)],
            "entities":[{"id":f"g{i}","type":"LINE","start_node":f"v{i}","end_node":f"v{(i+1)%4}","start":list(points[i]),"end":list(points[(i+1)%4])} for i in range(4)]}


def angle_row(axis="vertical",value=12.):
    return {"id":"inclination","record_id":"angle_label","kind":"angle","entities":["g1"],"nodes":[],
            "value":value,"source":"ocr_local_binding","reference_axis":axis,"angle_mode":"unsigned"}


def strict_angle_row(axis="vertical",value=12.):
    return {**angle_row(axis,value),"required":True,"nominal_source":"source_ocr",
            "source_arrow_verified":True}


def source_observation(source,*,maximum=None):
    points=[[node["x"],node["y"]] for node in source["nodes"]]
    observation={"units":"mm","points":points+[points[0]],"provenance":"input_mask_boundary",
                 "oracle_mask_conditioned":False,"reference_dxf_read":False}
    if maximum is not None:
        observation["boundary_error_budget"]={"units":"mm","maximum_deviation":maximum,
            "sampling_step":.02,"source":"initial_curve_fit_total_deviation_budget_px"}
    return observation


def test_arrow_verified_ocr_axis_angle_is_strict_without_source_budget():
    initial=graph(11.95);saved=copy.deepcopy(initial)
    solved=solve_parametric(initial,[strict_angle_row()])
    assert solved["accepted"],solved["validation"]
    check=solved["constraints"][0]
    assert check["enforcement"]=="linear_endpoint_direction_equality"
    assert check["tolerance"]==1e-7 and abs(check["actual"]-12.)<=1e-7
    assert solved["strict_ocr_angle_contract"]["satisfied"]
    assert solved["strict_ocr_angle_contract"]["checks"][0]["source_direction_quadrant_preserved"]
    assert solved["diagnostics"]["optimizer"]=="SLSQP_strict_ocr_angle_equalities"
    assert initial==saved


def test_strict_ocr_angle_survives_source_prior_and_bounded_warm_start():
    target=graph(12.);initial=graph(11.95);source=source_observation(target)
    plain=solve_parametric(initial,[strict_angle_row()],source_observation=source)
    assert plain["accepted"],plain["validation"]
    assert abs(plain["constraints"][0]["actual"]-12.)<=1e-7
    bounded=solve_parametric(initial,[strict_angle_row()],source_observation=source_observation(target,maximum=.1),
                             seed_node_offsets={"v2":[.01,0.]})
    assert bounded["accepted"],bounded["validation"]
    assert bounded["validation"]["source_boundary_budget_passed"]
    assert bounded["diagnostics"]["source_budget_search"]["strict_ocr_angle_equalities_active"]
    assert abs(bounded["constraints"][0]["actual"]-12.)<=1e-7
    assert bounded["strict_ocr_angle_contract"]["satisfied"]


def test_unverified_axis_angle_keeps_existing_tolerance():
    solved=solve_parametric(graph(11.95),[angle_row()])
    assert solved["accepted"],solved["validation"]
    assert solved["constraints"][0]["tolerance"]==.1
    assert solved["constraints"][0]["enforcement"]=="fixed_tolerance"
    assert solved["strict_ocr_angle_contract"]["required_count"]==0


def test_compatible_strict_axis_labels_on_one_line_share_direction_equality():
    vertical=strict_angle_row()
    horizontal={**strict_angle_row("horizontal",78.),"id":"complement","record_id":"complement_label"}
    solved=solve_parametric(graph(11.95),[vertical,horizontal])
    assert solved["accepted"],solved["validation"]
    assert solved["strict_ocr_angle_contract"]["required_count"]==2
    assert all(check["passed"] and check["absolute_residual"]<=1e-7 for check in solved["constraints"])


def test_conflicting_strict_angles_fail_closed_without_moving_graph():
    initial=graph(11.95);saved=copy.deepcopy(initial)
    other={**strict_angle_row(value=14.),"id":"other","record_id":"other_label"}
    solved=solve_parametric(initial,[strict_angle_row(),other])
    assert not solved["accepted"] and solved["status"]=="conflict"
    assert solved["entities"]==saved["entities"] and solved["nodes"]==saved["nodes"]
    assert solved["diagnostics"]["strict_ocr_angle_setup_issues"][0]["reason"]=="conflicting_strict_ocr_angles_on_one_line"
    assert not solved["strict_ocr_angle_contract"]["satisfied"]


def test_strict_angle_without_source_line_quadrant_fails_closed():
    initial=graph(0.);saved=copy.deepcopy(initial)
    solved=solve_parametric(initial,[strict_angle_row()])
    assert not solved["accepted"] and solved["status"]=="conflict"
    assert solved["entities"]==saved["entities"] and solved["nodes"]==saved["nodes"]
    assert solved["diagnostics"]["strict_ocr_angle_setup_issues"][0]["reason"]=="source_line_direction_quadrant_ambiguous"


@pytest.mark.parametrize("axis,value",[("vertical",0.),("horizontal",90.)])
def test_strict_axis_limit_has_smooth_equality_and_numerical_certificate(axis,value):
    solved=solve_parametric(graph(.05),[strict_angle_row(axis,value)])
    assert solved["accepted"],solved["validation"]
    assert solved["diagnostics"]["constraint_rank"]>=1
    assert abs(solved["constraints"][0]["actual"]-value)<=1e-7
    assert solved["strict_ocr_angle_contract"]["satisfied"]


def test_strict_angle_source_budget_exhaustion_keeps_uncertified_baseline():
    initial=graph(11.95);saved=copy.deepcopy(initial)
    observation=source_observation(graph(12.),maximum=.1)
    observation["boundary_error_budget"]["max_objective_evaluations"]=1
    solved=solve_parametric(initial,[strict_angle_row()],source_observation=observation)
    assert not solved["accepted"] and solved["status"]=="source_budget_search_exhausted"
    assert solved["entities"]==saved["entities"] and solved["nodes"]==saved["nodes"]
    assert not solved["strict_ocr_angle_contract"]["satisfied"]
    assert solved["diagnostics"]["source_budget_search"]["exhausted"]


def test_annotation_angle_changes_fitted_direction_in_joint_solve(tmp_path):
    initial=graph();saved=copy.deepcopy(initial)
    solved=solve_parametric(initial,[angle_row()],output_dir=tmp_path)
    assert solved["accepted"],solved.get("validation")
    check=solved["constraints"][0]
    assert check["passed"] and abs(check["actual"]-12.)<=.1
    assert check["reference_axis"]=="vertical" and check["record_id"]=="angle_label"
    assert abs(check["actual"]-11.)>.8
    assert initial==saved
    assert solved["entities"][1]["type"]=="LINE"


@pytest.mark.parametrize("axis,value",[("vertical",12.),("horizontal",78.)])
def test_axis_angle_is_independent_of_line_traversal(axis,value):
    original=graph(12.); row=angle_row(axis,value)
    for reverse in (False,True):
        entity=copy.deepcopy(original["entities"][1])
        if reverse:entity["start"],entity["end"]=entity["end"],entity["start"]
        actual=_constraint_value(row,{}, {"g1":entity})
        assert actual==pytest.approx(value)


@pytest.mark.parametrize("change",[
    {"reference_axis":"diagonal"},{"angle_mode":"signed"},{"value":91.},
    {"entities":["g1","g2"]},{"nodes":["v1"]},{"source":"source_geometry","record_id":None},
])
def test_invalid_or_unsupported_axis_claim_is_rejected(change):
    with pytest.raises(ValueError):_validate_inputs(graph(),[{**angle_row(),**change}])


def test_arc_tangent_is_not_admitted_as_an_annotated_straight_support():
    altered=graph(); arc=altered["entities"][1]; a=np.array(arc["start"]);b=np.array(arc["end"])
    center=(a+b)/2+np.array([30.,-30.*(b[0]-a[0])/(b[1]-a[1])])
    arc.update(type="ARC",center=center.tolist(),radius=float(np.linalg.norm(a-center)),clockwise=True)
    with pytest.raises(ValueError,match="single LINE"):_validate_inputs(altered,[angle_row()])


def test_reference_axis_is_part_of_publication_constraint_identity():
    vertical=_constraint_signatures(graph(),[angle_row("vertical",12.)])
    horizontal=_constraint_signatures(graph(),[angle_row("horizontal",12.)])
    assert vertical!=horizontal


def test_old_nonadjacent_line_angle_contract_remains_supported():
    row=angle_row();row.pop("reference_axis");row["entities"]=["g1","g3"]
    nodes,entities,prepared=_validate_inputs(graph(),[row])
    assert prepared[0]["entities"]==["g1","g3"]

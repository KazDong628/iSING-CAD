"""Annotation equations stay exact through solve and independent native readback."""
import copy
import math

import ezdxf
import numpy as np
import pytest

from contour_agent import parametric_solver as solver
from test_parametric_solver import rectangle


def exact(row):
    return {"id":"dimension","record_id":"r_dimension","source":"ocr_api_binding",
            "entities":[],"nodes":[],"required":True,"nominal_source":"source_ocr",
            "enforcement":"exact",**row}


def native(entities):
    document=ezdxf.new()
    for entity in entities:
        if entity["type"]=="LINE":
            document.modelspace().add_line(entity["start"],entity["end"])
        else:
            angles=[math.degrees(math.atan2(entity[key][1]-entity["center"][1],
                                           entity[key][0]-entity["center"][0])) for key in ("start","end")]
            if entity["clockwise"]:angles.reverse()
            document.modelspace().add_arc(entity["center"],entity["radius"],*angles)
    return document


@pytest.mark.parametrize("kind,nodes,value",[
    ("distance_x",["v0","v1"],10.5),
    ("distance_y",["v2","v1"],-5.25),
    ("distance",["v0","v2"],11.3),
])
def test_admitted_node_dimensions_are_equalities_not_point_zero_five_tolerance(kind,nodes,value):
    graph=rectangle();before=copy.deepcopy(graph)
    row=exact({"kind":kind,"nodes":nodes,"value":value})
    result=solver.solve_parametric(graph,[row])
    assert result["accepted"],result["validation"]
    assert graph==before
    check=result["constraints"][0]
    assert check["tolerance"]==1e-7 and check["absolute_residual"]<1e-7
    assert check["enforcement"]=="source_nominal_equality"
    assert result["strict_measurement_contract"]["required_count"]==1
    assert result["underconstrained"] and not result["all_dimensions_verified"]
    audit=solver.native_measurement_checks(result["entities"],result["constraints"],
                                           dxf_document=native(result["entities"]))
    assert audit["passed"] and audit["dxf_readback_performed"]
    assert audit["checks"][0]["tolerance_mm"]==1e-7


def test_forged_passed_length_receipt_does_not_turn_small_nominal_error_into_success():
    graph=rectangle(width=9.99)
    row=exact({"kind":"distance_x","nodes":["v0","v1"],"value":10.,
               "actual":10.,"absolute_residual":0.,"passed":True})
    result=solver.native_measurement_checks(graph["entities"],[row],dxf_document=native(graph["entities"]))
    assert not result["passed"]
    assert result["checks"][0]["model"]["absolute_residual"]==pytest.approx(.01)


@pytest.mark.parametrize("signed,value",[(False,87.),(True,-87.)])
def test_general_node_angles_use_hard_nominal_equations_and_native_certificate(signed,value):
    row=exact({"kind":"angle","nodes":["v0","v1","v2"],"value":value,
               "angle_mode":"signed" if signed else "unsigned"})
    result=solver.solve_parametric(rectangle(),[row])
    assert result["accepted"],result["validation"]
    assert result["strict_source_nominal_angle_contract"]["satisfied"]
    assert result["constraints"][0]["absolute_residual"]<1e-7
    assert result["constraints"][0]["enforcement"]=="source_nominal_angle_equality"
    audit=solver.native_source_nominal_angle_checks(result["entities"],result["constraints"],
                                                   dxf_document=native(result["entities"]))
    assert audit["passed"] and audit["required_count"]==1


def test_forged_passed_angle_receipt_below_legacy_tolerance_still_fails():
    graph=rectangle()
    row=exact({"kind":"angle","nodes":["v0","v1","v2"],"value":89.95,
               "actual":89.95,"absolute_residual":0.,"passed":True})
    audit=solver.native_source_nominal_angle_checks(graph["entities"],[row],
                                                   dxf_document=native(graph["entities"]))
    assert not audit["passed"]
    assert audit["checks"][0]["dxf"]["absolute_residual"]==pytest.approx(.05)


def test_conflicting_exact_lengths_cannot_average_inside_legacy_tolerance():
    first=exact({"kind":"distance_x","nodes":["v0","v1"],"value":10.})
    second={**first,"id":"other","record_id":"other_record","value":10.01}
    result=solver.solve_parametric(rectangle(),[first,second])
    assert not result["accepted"]
    assert not result["candidate_strict_measurement_contract"]["satisfied"]
    assert result["entities"]==rectangle()["entities"]


def test_derived_levels_enforce_axis_relations_without_extra_nodes_and_count_their_rank():
    graph=rectangle();graph["nodes"][2]["y"]+=.02
    for entity in graph["entities"]:
        for endpoint in ("start","end"):
            item=next(node for node in graph["nodes"] if node["id"]==entity[endpoint+"_node"])
            entity[endpoint]=[item["x"],item["y"]]
    row=exact({"kind":"distance_y","entities":["g0","g2"],"value":5.2,"anchors":[
        {"type":"support_level","entity_id":"g0","axis":"y"},
        {"type":"support_level","entity_id":"g2","axis":"y"}]})
    result=solver.solve_parametric(graph,[row])
    assert result["accepted"],result["validation"]
    assert len(result["nodes"])==4
    assert result["diagnostics"]["variable_count"]==8
    assert result["diagnostics"]["constraint_rank"]==3
    for entity in (result["entities"][0],result["entities"][2]):
        assert abs(entity["end"][1]-entity["start"][1])<1e-7
    audit=solver.native_measurement_checks(result["entities"],result["constraints"],
                                           dxf_document=native(result["entities"]))
    assert audit["passed"]


def test_derived_arc_height_recomputes_extremum_from_free_center_and_radius():
    graph={"units":"mm","nodes":[{"id":"v0","x":5.,"y":0.},{"id":"v1","x":-5.,"y":0.}],
           "entities":[{"id":"g0","type":"ARC","start_node":"v0","end_node":"v1",
                        "start":[5.,0.],"end":[-5.,0.],"center":[0.,0.],"radius":5.,"clockwise":False},
                       {"id":"g1","type":"LINE","start_node":"v1","end_node":"v0",
                        "start":[-5.,0.],"end":[5.,0.]}]}
    row=exact({"kind":"distance_y","entities":["g0","g1"],"value":5.3,"anchors":[
        {"type":"support_level","entity_id":"g1","axis":"y"},
        {"type":"curve_extremum","entity_id":"g0","axis":"y","sense":"maximum"}]})
    result=solver.solve_parametric(graph,[row])
    assert result["accepted"],result["validation"]
    assert len(result["nodes"])==2
    assert result["diagnostics"]["constraint_rank"]==2
    actual=result["entities"][0]["center"][1]+result["entities"][0]["radius"]-result["entities"][1]["start"][1]
    assert actual==pytest.approx(5.3,abs=1e-7)
    audit=solver.native_measurement_checks(result["entities"],result["constraints"],
                                           dxf_document=native(result["entities"]))
    assert audit["passed"]

"""Measurements on theoretical sharp corners stay attached to free supports."""
import copy
import math

import numpy as np
import pytest

from contour_agent import parametric_solver as solver
from test_parametric_solver import rounded_graph, dimensions_and_relations, rectangle


def intersection(first,second):
    return {"type":"line_intersection","entities":[first,second]}


def node(identifier):
    return {"type":"node","node_id":identifier}


def measurement(anchors,value=20.,kind="distance_x",identifier="width"):
    return {"id":identifier,"kind":kind,"anchors":anchors,"nodes":[],
            "entities":sorted({eid for anchor in anchors for eid in anchor.get("entities",[])}),
            "value":value,"record_id":"r_"+identifier,"source":"ocr_local_binding"}


def independent_intersection(first,second):
    a=np.asarray(first["start"]);u=np.asarray(first["end"])-a
    b=np.asarray(second["start"]);v=np.asarray(second["end"])-b
    parameter=np.linalg.solve(np.column_stack((u,-v)),b-a)[0]
    return a+parameter*u


def test_fillet_corner_dimensions_jointly_move_supports_without_new_cad_objects():
    graph=rounded_graph();before=copy.deepcopy(graph)
    constraints=dimensions_and_relations()[2:]
    constraints.extend([
        measurement([intersection("g006","g000"),intersection("g000","g002")]),
        measurement([intersection("g006","g000"),intersection("g006","g004")],12.,"distance_y","height")])
    result=solver.solve_parametric(graph,constraints)
    assert result["accepted"],result
    assert graph==before
    assert len(result["nodes"])==len(graph["nodes"])
    assert len(result["entities"])==len(graph["entities"])
    indexed={entity["id"]:entity for entity in result["entities"]}
    corners=[independent_intersection(indexed[a],indexed[b]) for a,b in
             [("g006","g000"),("g000","g002"),("g006","g004")]]
    assert corners[1][0]-corners[0][0]==pytest.approx(20.,abs=1e-7)
    assert corners[2][1]-corners[0][1]==pytest.approx(12.,abs=1e-7)
    assert all(entity["radius"]==3. for entity in result["entities"] if entity["type"]=="ARC")
    assert result["strict_tangent_contract"]["satisfied"]
    assert result["strict_measurement_contract"]["satisfied"]
    assert result["strict_measurement_contract"]["required_count"]==2
    assert result["diagnostics"]["remaining_shape_dof"]==0
    assert result["diagnostics"]["variable_count"]==16
    assert set(row["enforcement"] for row in result["constraints"] if "anchors" in row)=={"derived_measurement_anchor_equality"}
    assert not result["all_dimensions_verified"]


def test_mixed_node_and_virtual_corner_uses_current_support_endpoints():
    graph=rounded_graph(noise=0.)
    bound=measurement([node("v004"),intersection("g006","g004")],-17.1)
    result=solver.solve_parametric(graph,[bound])
    assert result["accepted"],result
    indexed={entity["id"]:entity for entity in result["entities"]}
    nodes={value["id"]:np.array([value["x"],value["y"]]) for value in result["nodes"]}
    actual=independent_intersection(indexed["g006"],indexed["g004"])[0]-nodes["v004"][0]
    assert actual==pytest.approx(-17.1,abs=1e-7)
    assert result["diagnostics"]["constraint_rank"]==1
    assert result["diagnostics"]["optimizer_active_variable_count"]==8
    assert result["underconstrained"]


def test_anchor_rank_excludes_prior_and_translations_remain_gauges():
    graph=rectangle()
    bound=measurement([node("v0"),node("v1")],10.5)
    result=solver.solve_parametric(graph,[bound])
    assert result["accepted"],result
    assert result["constraints"][0]["absolute_residual"]<=1e-7
    assert result["diagnostics"]["constraint_rank"]==1
    assert result["diagnostics"]["variable_count"]==8
    assert result["diagnostics"]["remaining_shape_dof"]>0
    assert result["diagnostics"]["rank_excludes_soft_prior"]


@pytest.mark.parametrize("anchors,match",[
    ([node("missing"),node("v001")],"unknown node"),
    ([intersection("g000","g001"),node("v004")],"non-LINE"),
    ([intersection("g000","g004"),node("v004")],"parallel"),
    ([intersection("g000","g000"),node("v004")],"distinct LINE"),
    ([{"type":"fixed","x":0.,"y":0.},node("v004")],"Unsupported"),
    ([{"type":"node","node_id":"v004","x":0.},node("v005")],"only contain"),
    ([node("v004"),node("v004")],"distinct"),
])
def test_invalid_and_literal_coordinate_anchors_are_rejected(anchors,match):
    with pytest.raises(ValueError,match=match):
        solver.solve_parametric(rounded_graph(noise=0.),[measurement(anchors)])


def test_nearly_parallel_supports_fail_before_optimization():
    graph=rectangle();graph["nodes"][2]["y"]+=.01
    for entity in graph["entities"]:
        for key in ("start","end"):
            reference=next(value for value in graph["nodes"] if value["id"]==entity[key+"_node"])
            entity[key]=[reference["x"],reference["y"]]
    with pytest.raises(ValueError,match="near-parallel"):
        solver.solve_parametric(graph,[measurement([intersection("g0","g2"),node("v1")])])


def test_anchor_reference_union_and_legacy_node_list_cannot_disagree():
    graph=rounded_graph(noise=0.);row=measurement([node("v004"),intersection("g006","g004")])
    with pytest.raises(ValueError,match="support union"):
        solver.solve_parametric(graph,[{**row,"entities":["g006"]}])
    with pytest.raises(ValueError,match="empty nodes"):
        solver.solve_parametric(graph,[{**row,"nodes":["v004","v005"]}])


def test_reversed_intersection_supports_do_not_create_extra_independent_dimensions():
    graph=rounded_graph(noise=0.)
    row=measurement([intersection("g006","g000"),intersection("g000","g002")])
    duplicate=measurement([intersection("g002","g000"),intersection("g000","g006")],-20.,identifier="reversed")
    with pytest.raises(ValueError,match="equivalent"):
        solver.solve_parametric(graph,[row,duplicate])


def test_conflicting_typed_measurements_retain_original_geometry():
    graph=rectangle();before=copy.deepcopy(graph)
    rows=[measurement([node("v0"),node("v1")],value,identifier="w"+str(index))
          for index,value in enumerate([10.,10.01])]
    result=solver.solve_parametric(graph,rows)
    assert not result["accepted"]
    assert result["entities"]==before["entities"]
    assert result["nodes"]==before["nodes"]
    assert not result["strict_measurement_contract"]["satisfied"]


def test_final_contract_rejects_parallel_support_after_geometry_changes():
    graph=rectangle();nodes={row["id"]:np.array([row["x"],row["y"]]) for row in graph["nodes"]}
    row=measurement([intersection("g0","g1"),node("v2")],5.,"distance_y")
    changed=copy.deepcopy(graph["entities"]);changed[1]["end"]=[20.,0.]
    contract=solver._strict_measurement_contract(nodes,changed,[row],graph["entities"])
    assert not contract["satisfied"]
    assert contract["checks"][0]["reason"]=="invalid_or_ill_conditioned_measurement_anchor"

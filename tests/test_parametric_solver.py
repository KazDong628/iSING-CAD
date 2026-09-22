import copy
import json
import math

import numpy as np
import pytest

from contour_agent.parametric_solver import solve_parametric


def rounded_graph(width=19.3,height=12.4,radius=2.5,noise=.03):
    points=np.array([(radius,0),(width-radius,0),(width,radius),(width,height-radius),
                     (width-radius,height),(radius,height),(0,height-radius),(0,radius)],float)
    points+=np.random.default_rng(41).normal(0,noise,points.shape)
    nodes=[{"id":f"v{i:03d}","x":float(x),"y":float(y),"source_px":[float(x*8),float(100-y*8)]} for i,(x,y) in enumerate(points)]
    entities=[]
    for index in range(8):
        a=points[index];b=points[(index+1)%8]
        entity={"id":f"g{index:03d}","type":"ARC" if index%2 else "LINE",
                "start_node":nodes[index]["id"],"end_node":nodes[(index+1)%8]["id"],"start":a.tolist(),"end":b.tolist()}
        if index%2:
            chord=b-a;center=(a+b)/2+np.array([-chord[1],chord[0]])/2
            entity.update(center=center.tolist(),radius=float(np.linalg.norm(a-center)),clockwise=False)
        entities.append(entity)
    return {"units":"mm","coordinate_system":{"units":"mm","x":"image right","y":"image up"},"nodes":nodes,"entities":entities}


def dimensions_and_relations():
    dimensions=[{"id":"width","kind":"distance_x","nodes":["v006","v002"],"entities":[],"record_id":"r000","value":20.,"source":"ocr_local_binding"},
                {"id":"height","kind":"distance_y","nodes":["v000","v004"],"entities":[],"record_id":"r001","value":12.,"source":"ocr_api_binding"}]
    for index in (1,3,5,7):
        dimensions.append({"id":f"radius{index}","kind":"radius","nodes":[],"entities":[f"g{index:03d}"],"record_id":f"r{index+1:03d}","value":3.,"source":"ocr_local_binding"})
    for index in (0,2,4,6):
        dimensions.append({"id":f"axis{index}","kind":"horizontal" if index in (0,4) else "vertical","entities":[f"g{index:03d}"],"nodes":[],"value":None,"record_id":None,"source":"source_geometry"})
    for index in range(8):
        dimensions.append({"id":f"tangent{index}","kind":"tangent","entities":[f"g{index:03d}",f"g{(index+1)%8:03d}"],"nodes":[],"value":None,"record_id":None,"source":"source_geometry"})
    return dimensions


def rectangle(width=10,height=5):
    points=[(0,0),(width,0),(width,height),(0,height)]
    return {"units":"mm","nodes":[{"id":f"v{i}","x":x,"y":y,"source_px":[x*10,100-y*10]} for i,(x,y) in enumerate(points)],
            "entities":[{"id":f"g{i}","type":"LINE","start_node":f"v{i}","end_node":f"v{(i+1)%4}","start":list(points[i]),"end":list(points[(i+1)%4])} for i in range(4)]}


def test_joint_solve_recovers_noisy_dimensions_radius_and_tangency(tmp_path):
    graph=rounded_graph();before=copy.deepcopy(graph);constraints=dimensions_and_relations()
    result=solve_parametric(graph,constraints,output_dir=tmp_path)
    assert result["accepted"],result["validation"]
    assert graph==before
    assert result["dimension_solve_success"]
    assert all(row["passed"] for row in result["constraints"])
    assert max(row["absolute_residual"] for row in result["constraints"] if row["kind"] in {"radius","distance_x","distance_y"})<.05
    assert max(row["absolute_residual"] for row in result["constraints"] if row["kind"]=="tangent")<.1
    assert result["validation"]["closed"] and not result["validation"]["self_intersection"]
    assert result["validation"]["max_gap"]==0
    assert result["validation"]["max_radial_error"]<1e-10
    assert [node["id"] for node in result["nodes"]]==[node["id"] for node in graph["nodes"]]
    assert not result["all_dimensions_verified"] and not result["engineering_verified"]
    assert not result["diagnostics"]["source_coordinates_hard_locked"]
    assert result["diagnostics"]["rank_excludes_soft_prior"]
    assert result["diagnostics"]["remaining_shape_dof"]==0
    saved=json.loads((tmp_path/"parametric-solve.json").read_text(encoding="utf8"))
    assert saved["accepted"] and saved["entities"]==result["entities"]
    for first,second in zip(result["entities"],result["entities"][1:]+result["entities"][:1]):
        assert first["end"]==second["start"]


def test_partial_underconstrained_dimension_can_be_accepted_without_overclaim():
    graph=rectangle()
    constraints=[{"id":"width","kind":"distance_x","entities":[],"nodes":["v0","v1"],"value":10.5,"record_id":"r0","source":"ocr_local_binding"}]
    result=solve_parametric(graph,constraints)
    assert result["accepted"] and result["underconstrained"]
    assert result["diagnostics"]["remaining_shape_dof"]>0
    assert result["constraints"][0]["passed"]
    assert result["constraints"][0]["source"]=="ocr_local_binding"
    assert not result["all_dimensions_verified"] and not result["validation"]["dimensions_verified"]


def test_conflicting_dimensions_keep_exact_baseline_and_candidate_diagnostics():
    graph=rectangle()
    constraints=[{"id":f"width{i}","kind":"distance_x","entities":[],"nodes":["v0","v1"],"value":value,"record_id":f"r{i}","source":"ocr_api_binding"} for i,value in enumerate((10,12))]
    result=solve_parametric(graph,constraints)
    assert not result["accepted"] and result["status"]=="conflict"
    assert result["entities"]==graph["entities"] and result["nodes"]==graph["nodes"]
    assert result["candidate_entities"] is not None
    assert all(not row["passed"] for row in result["constraints"])
    assert not result["dimension_solve_success"]


def test_large_displacement_is_rejected_without_discarding_baseline():
    graph=rectangle()
    constraints=[{"id":"width","kind":"distance_x","entities":[],"nodes":["v0","v1"],"value":30.,"record_id":"r0","source":"ocr_local_binding"}]
    result=solve_parametric(graph,constraints)
    assert not result["accepted"]
    assert result["status"]=="displacement_rejected"
    assert result["entities"]==graph["entities"]
    assert result["diagnostics"]["maximum_node_displacement"]>result["diagnostics"]["displacement_limit"]


def test_pixel_relations_never_create_millimetre_dimensions():
    graph=rectangle();graph["units"]="pixel"
    relation={"id":"flat","kind":"horizontal","entities":["g0"],"nodes":[],"record_id":None,"value":None,"source":"source_geometry"}
    result=solve_parametric(graph,[relation])
    assert result["accepted"] and not result["dimension_solve_success"]
    assert result["constraints"][0]["residual_unit"]=="degree"
    assert result["diagnostics"]["displacement_unit"]=="pixel"
    dimension={"id":"width","kind":"distance_x","nodes":["v0","v1"],"entities":[],"record_id":"r0","value":10,"source":"ocr_local_binding"}
    with pytest.raises(ValueError,match="mm graph"):solve_parametric(graph,[dimension])


@pytest.mark.parametrize("value",[float("nan"),float("inf"),-float("inf"),True,"10"])
def test_nonfinite_or_nonreal_values_are_rejected(value):
    constraint={"id":"width","kind":"distance_x","nodes":["v0","v1"],"entities":[],"record_id":"r0","value":value,"source":"ocr_api_binding"}
    with pytest.raises(ValueError,match="finite real"):solve_parametric(rectangle(),[constraint])


def test_duplicate_ambiguous_and_unknown_bindings_are_rejected():
    graph=rounded_graph();radius=dimensions_and_relations()[2]
    duplicate={**radius,"id":"other"}
    with pytest.raises(ValueError,match="record_id is reused"):solve_parametric(graph,[radius,duplicate])
    duplicate["record_id"]="another"
    with pytest.raises(ValueError,match="equivalent"):solve_parametric(graph,[radius,duplicate])
    with pytest.raises(ValueError,match="Duplicate constraint id"):solve_parametric(graph,[radius,radius])
    with pytest.raises(ValueError,match="unknown entity/node"):solve_parametric(graph,[{**radius,"entities":["missing"]}])
    with pytest.raises(ValueError,match="record_id"):solve_parametric(graph,[{**radius,"record_id":None}])
    with pytest.raises(ValueError,match="OCR binding"):solve_parametric(graph,[{**radius,"source":"source_geometry","record_id":None}])


def test_major_arc_branch_and_clockwise_are_preserved():
    graph={"units":"mm","nodes":[{"id":"a","x":0.,"y":0.},{"id":"b","x":4.,"y":0.}],
           "entities":[{"id":"arc","type":"ARC","start_node":"a","end_node":"b","start":[0.,0.],"end":[4.,0.],"center":[2.,-math.sqrt(5)],"radius":3.,"clockwise":False},
                       {"id":"line","type":"LINE","start_node":"b","end_node":"a","start":[4.,0.],"end":[0.,0.]}]}
    constraint={"id":"radius","kind":"radius","entities":["arc"],"nodes":[],"value":3.1,"record_id":"r0","source":"ocr_local_binding"}
    result=solve_parametric(graph,[constraint])
    assert result["accepted"],result["validation"]
    assert result["diagnostics"]["arc_branches"]["arc"]["major"]
    entity=result["entities"][0]
    assert entity["clockwise"] is False
    a=math.atan2(entity["start"][1]-entity["center"][1],entity["start"][0]-entity["center"][0])
    b=math.atan2(entity["end"][1]-entity["center"][1],entity["end"][0]-entity["center"][0])
    assert (b-a)%(2*math.pi)>math.pi


def test_angle_and_euclidean_distance_have_explicit_measurements():
    graph=rectangle()
    constraints=[{"id":"diag","kind":"distance","nodes":["v0","v2"],"entities":[],"record_id":"r0","value":math.sqrt(125),"source":"ocr_local_binding"},
                 {"id":"corner","kind":"angle","nodes":["v0","v1","v2"],"entities":[],"record_id":"r1","value":90.,"source":"ocr_api_binding"}]
    result=solve_parametric(graph,constraints)
    assert result["accepted"]
    assert result["constraints"][0]["actual"]==pytest.approx(math.sqrt(125),abs=.05)
    assert result["constraints"][1]["actual"]==pytest.approx(90,abs=.1)


def test_self_intersecting_input_cannot_be_published_as_valid_solution():
    graph=rectangle()
    graph["nodes"][1].update(x=10,y=5);graph["nodes"][2].update(x=0,y=5);graph["nodes"][3].update(x=10,y=0)
    for index,entity in enumerate(graph["entities"]):
        entity["start"]=[graph["nodes"][index][key] for key in ("x","y")]
        entity["end"]=[graph["nodes"][(index+1)%4][key] for key in ("x","y")]
    constraint={"id":"width","kind":"distance_x","nodes":["v0","v1"],"entities":[],"record_id":"r0","value":10.,"source":"ocr_local_binding"}
    result=solve_parametric(graph,[constraint])
    assert not result["accepted"] and result["validation"]["self_intersection"]
    assert result["entities"]==graph["entities"]


def test_solver_reads_no_files_and_empty_constraints_preserve_graph(monkeypatch):
    from pathlib import Path
    def forbidden(*args,**kwargs):raise AssertionError("Solver must not read files")
    monkeypatch.setattr(Path,"read_text",forbidden);monkeypatch.setattr(Path,"read_bytes",forbidden)
    graph=rectangle();result=solve_parametric(graph,[])
    assert not result["accepted"] and result["entities"]==graph["entities"]
    assert result["diagnostics"]["ground_truth_used"] is False
    assert result["diagnostics"]["network_used"] is False


def test_clockwise_rounded_chain_uses_same_joint_constraints():
    graph=rounded_graph()
    for entity in graph["entities"]:
        entity["start"],entity["end"]=entity["end"],entity["start"]
        entity["start_node"],entity["end_node"]=entity["end_node"],entity["start_node"]
        if entity["type"]=="ARC":entity["clockwise"]=True
    graph["entities"].reverse()
    result=solve_parametric(graph,dimensions_and_relations())
    assert result["accepted"],result["validation"]
    assert result["validation"]["signed_area"]<0
    assert all(entity["clockwise"] for entity in result["entities"] if entity["type"]=="ARC")


def test_large_arbitrary_origin_does_not_destroy_signed_area():
    graph=rectangle()
    for node in graph["nodes"]:node["x"]+=1e8;node["y"]+=1e8
    for entity in graph["entities"]:
        for key in ("start","end"):entity[key]=[value+1e8 for value in entity[key]]
    constraint={"id":"width","kind":"distance_x","nodes":["v0","v1"],"entities":[],"record_id":"r0","value":10.,"source":"ocr_local_binding"}
    result=solve_parametric(graph,[constraint])
    assert result["accepted"],result["validation"]
    assert result["validation"]["signed_area"]==pytest.approx(50)


def test_reversed_equivalent_axis_dimension_is_rejected():
    first={"id":"one","kind":"distance_x","nodes":["v0","v1"],"entities":[],"record_id":"r0","value":10.,"source":"ocr_local_binding"}
    second={**first,"id":"two","nodes":["v1","v0"],"record_id":"r1","value":-10.}
    with pytest.raises(ValueError,match="equivalent"):solve_parametric(rectangle(),[first,second])


def test_radius_correction_does_not_collapse_chord_to_preserve_wrong_initial_center():
    graph={"units":"mm","nodes":[{"id":"a","x":0.,"y":0.},{"id":"b","x":6.,"y":0.}],
           "entities":[{"id":"arc","type":"ARC","start_node":"a","end_node":"b","start":[0.,0.],"end":[6.,0.],"center":[3.,math.sqrt(55)],"radius":8.,"clockwise":False},
                       {"id":"line","type":"LINE","start_node":"b","end_node":"a","start":[6.,0.],"end":[0.,0.]}]}
    constraint={"id":"radius","kind":"radius","entities":["arc"],"nodes":[],"value":5.,"record_id":"r0","source":"ocr_local_binding"}
    result=solve_parametric(graph,[constraint])
    assert result["accepted"],result["validation"]
    assert result["entities"][0]["radius"]==pytest.approx(5,abs=1e-5)
    assert math.dist(result["entities"][0]["start"],result["entities"][0]["end"])==pytest.approx(6,abs=1e-3)
    assert result["diagnostics"]["evaluations"]<100
    assert result["diagnostics"]["radius_bound_arc_offsets_without_initial_h_penalty"]==["arc"]


def test_infeasible_initial_chord_is_solved_without_fake_radius_or_zero_chord():
    graph={"units":"mm","nodes":[{"id":"a","x":0.,"y":0.},{"id":"b","x":12.,"y":0.}],
           "entities":[{"id":"arc","type":"ARC","start_node":"a","end_node":"b","start":[0.,0.],"end":[12.,0.],"center":[6.,math.sqrt(28)],"radius":8.,"clockwise":False},
                       {"id":"line","type":"LINE","start_node":"b","end_node":"a","start":[12.,0.],"end":[0.,0.]}]}
    constraint={"id":"radius","kind":"radius","entities":["arc"],"nodes":[],"value":5.,"record_id":"r0","source":"ocr_local_binding"}
    result=solve_parametric(graph,[constraint])
    assert result["diagnostics"]["converged"]
    assert result["constraints"][0]["passed"]
    assert result["candidate_entities"][0]["radius"]==pytest.approx(5,abs=.05)
    assert math.dist(result["candidate_entities"][0]["start"],result["candidate_entities"][0]["end"])>9.9
    assert not result["accepted"] and result["status"]=="displacement_rejected"
    assert result["entities"]==graph["entities"]


def test_relation_solve_uses_common_length_scale_for_short_and_long_primitives():
    # The 0.2-unit straight segments and ~140-unit arcs have very different
    # angular derivatives. Changing drawing units/origin must not change the
    # dimensionless source-regularized geometric problem.
    graph=rounded_graph(width=400,height=200,radius=99.9,noise=.01)
    constraints=[row for row in dimensions_and_relations() if row["source"]=="source_geometry"]
    original=solve_parametric(graph,constraints)
    shifted=copy.deepcopy(graph);shifted["units"]="pixel"
    factor=10.;offset=np.array([1234.,-4321.])
    for node in shifted["nodes"]:
        point=np.array([node["x"],node["y"]])*factor+offset
        node.update(x=float(point[0]),y=float(point[1]))
    for entity in shifted["entities"]:
        for key in ("start","end","center"):
            if key in entity:entity[key]=(np.array(entity[key])*factor+offset).tolist()
        if "radius" in entity:entity["radius"]*=factor
    transformed=solve_parametric(shifted,constraints)
    assert original["accepted"] and transformed["accepted"]
    for first,second in zip(original["nodes"],transformed["nodes"]):
        actual=(np.array([second["x"],second["y"]])-offset)/factor
        assert actual==pytest.approx([first["x"],first["y"]],abs=2e-5)
    for result in (original,transformed):
        assert result["diagnostics"]["converged"]
        assert result["validation"]["geometry_valid"]
        assert all(check["absolute_residual"]<.1 for check in result["constraints"])
        assert not result["dimension_solve_success"]
        assert not result["all_dimensions_verified"]


def test_feasible_residuals_do_not_override_optimizer_nonconvergence(monkeypatch):
    import contour_agent.parametric_solver as solver
    actual=solver.least_squares
    def exhausted(*args,**kwargs):
        fit=actual(*args,**kwargs)
        fit.success=False;fit.status=0;fit.message="Bounded regression: evaluation budget exhausted"
        return fit
    monkeypatch.setattr(solver,"least_squares",exhausted)
    graph=rectangle()
    relation={"id":"flat","kind":"horizontal","entities":["g0"],"nodes":[],"record_id":None,"value":None,"source":"source_geometry"}
    result=solver.solve_parametric(graph,[relation])
    assert result["validation"]["constraint_subset_satisfied"]
    assert result["validation"]["geometry_valid"]
    assert result["validation"]["source_displacement_passed"]
    assert not result["accepted"] and result["status"]=="not_converged"
    assert result["entities"]==graph["entities"]
    assert "evaluation budget exhausted" in result["diagnostics"]["optimizer_message"]

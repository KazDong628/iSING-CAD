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
    radii=[row for row in result["constraints"] if row["kind"]=="radius"]
    assert all(row["actual"]==row["value"] and row["absolute_residual"]==0. and row["tolerance"]==0. for row in radii)
    assert result["strict_radius_contract"]["satisfied"]
    assert result["strict_radius_contract"]["required_count"]==4
    assert result["diagnostics"]["eliminated_exact_radius_constraint_count"]==4
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
    assert not result["diagnostics"]["source_curve_prior"]["enabled"]


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


def test_close_but_distinct_radius_annotations_are_an_exact_conflict(tmp_path):
    graph=rounded_graph();first=dimensions_and_relations()[2]
    second={**first,"id":"different_radius","record_id":"different_record","value":first["value"]+.001}
    # Both values fitted within 0.05 mm must never count as satisfying either
    # annotation exactly; conflicting targets cannot be averaged together.
    result=solve_parametric(graph,[first,second],output_dir=tmp_path)
    assert not result["accepted"] and result["status"]=="conflict"
    assert result["entities"]==graph["entities"]
    assert result["candidate_entities"] is None
    assert result["strict_radius_contract"]["required_count"]==2
    assert result["strict_radius_contract"]["required_entity_count"]==1
    assert not result["strict_radius_contract"]["satisfied"]
    assert result["diagnostics"]["conflicting_radius_entity_ids"]==[first["entities"][0]]
    assert (tmp_path/"parametric-solve.json").is_file()


def test_rejected_radius_candidate_never_certifies_baseline_radii():
    graph=rounded_graph(width=10,height=10,radius=4.8);constraint={**dimensions_and_relations()[2],"value":.001}
    result=solve_parametric(graph,[constraint])
    assert not result["accepted"]
    assert result["entities"]==graph["entities"]
    assert not result["strict_radius_contract"]["satisfied"]
    assert result["strict_radius_contract"]["failed_entity_ids"]==constraint["entities"]
    assert result["candidate_strict_radius_contract"]["satisfied"]
    assert result["candidate_entities"][1]["radius"]==.001


def test_exact_radius_cannot_hide_infeasible_circle_incidence(monkeypatch):
    from types import SimpleNamespace
    import contour_agent.parametric_solver as solver
    graph={"units":"mm","nodes":[{"id":"a","x":0.,"y":0.},{"id":"b","x":12.,"y":0.}],
           "entities":[{"id":"arc","type":"ARC","start_node":"a","end_node":"b","start":[0.,0.],"end":[12.,0.],"center":[6.,math.sqrt(28)],"radius":8.,"clockwise":False},
                       {"id":"line","type":"LINE","start_node":"b","end_node":"a","start":[12.,0.],"end":[0.,0.]}]}
    constraint={"id":"radius","kind":"radius","entities":["arc"],"nodes":[],"value":5.,"record_id":"r0","source":"ocr_local_binding"}
    def invalid_success(*args,**kwargs):
        return SimpleNamespace(x=np.array([-.5,0.,.5,0.]),jac=np.zeros(4),success=True,status=0,nfev=1,message="infeasible trial")
    monkeypatch.setattr(solver,"minimize",invalid_success)
    result=solver.solve_parametric(graph,[constraint])
    assert not result["accepted"] and result["status"]=="invalid_geometry"
    assert result["candidate_entities"][0]["radius"]==5.
    assert result["candidate_strict_radius_contract"]["satisfied"]
    assert not result["validation"]["exact_radius_chords_feasible"]
    assert not result["validation"]["geometry_valid"]
    assert result["validation"]["max_radial_error"]>0.9
    assert result["entities"]==graph["entities"]


def test_absent_radius_bindings_do_not_claim_annotation_coverage():
    graph=rounded_graph();result=solve_parametric(graph,[])
    assert result["entities"]==graph["entities"]
    assert result["strict_radius_contract"]["required_count"]==0
    assert not result["strict_radius_contract"]["unbound_radius_coverage_verified"]


def test_exact_radius_optimizer_nonconvergence_is_not_accepted(monkeypatch):
    import contour_agent.parametric_solver as solver
    actual=solver.minimize
    def exhausted(*args,**kwargs):
        fit=actual(*args,**kwargs)
        fit.success=False;fit.status=9;fit.message="Bounded regression: iteration budget exhausted"
        return fit
    monkeypatch.setattr(solver,"minimize",exhausted)
    graph=rounded_graph();constraint=dimensions_and_relations()[2]
    result=solver.solve_parametric(graph,[constraint])
    assert result["candidate_strict_radius_contract"]["satisfied"]
    assert not result["accepted"] and result["status"]=="not_converged"
    assert result["entities"]==graph["entities"]
    assert not result["strict_radius_contract"]["satisfied"]


def test_source_curve_prior_preserves_arc_interior_while_radius_stays_exact():
    graph={"units":"mm","nodes":[{"id":"a","x":0.,"y":0.},{"id":"b","x":6.,"y":0.}],
           "entities":[{"id":"arc","type":"ARC","start_node":"a","end_node":"b","start":[0.,0.],"end":[6.,0.],"center":[3.,math.sqrt(55)],"radius":8.,"clockwise":False},
                       {"id":"line","type":"LINE","start_node":"b","end_node":"a","start":[6.,0.],"end":[0.,0.]}]}
    constraint={"id":"radius","kind":"radius","entities":["arc"],"nodes":[],"value":5.,"record_id":"r0","source":"ocr_local_binding"}
    endpoint_only=solve_parametric(graph,[constraint])
    supported=copy.deepcopy(graph)
    for entity in supported["entities"]:entity["source_fit_error_px"]=.25
    shape_preserving=solve_parametric(supported,[constraint])
    assert endpoint_only["accepted"] and shape_preserving["accepted"]
    def midpoint(entity):
        # Analytic midpoint of this known counterclockwise minor branch,
        # independent of the solver's quadrature sample generation.
        a=np.array(entity["start"]);b=np.array(entity["end"]);length=np.linalg.norm(b-a)
        normal=np.array([-(b-a)[1],(b-a)[0]])/length
        sagitta=entity["radius"]-math.sqrt(entity["radius"]**2-(length/2)**2)
        return (a+b)/2-normal*sagitta
    source_middle=midpoint(graph["entities"][0])
    old_error=np.linalg.norm(midpoint(endpoint_only["entities"][0])-source_middle)
    new_error=np.linalg.norm(midpoint(shape_preserving["entities"][0])-source_middle)
    assert new_error<old_error*.8
    assert shape_preserving["entities"][0]["radius"]==5.
    assert shape_preserving["strict_radius_contract"]["satisfied"]
    assert shape_preserving["validation"]["max_gap"]==0
    prior=shape_preserving["diagnostics"]["source_curve_prior"]
    assert prior["enabled"] and not prior["hard_constraint"]
    assert prior["source"]=="input_graph_entities"


def test_curve_prior_releases_neighboring_unbound_geometry_without_radius_snapping():
    graph=rounded_graph()
    for entity in graph["entities"]:entity["source_fit_error_px"]=.25
    result=solve_parametric(graph,[dimensions_and_relations()[2]])
    assert result["accepted"] and result["strict_radius_contract"]["required_count"]==1
    assert result["entities"][1]["radius"]==3.
    # The source-curve objective couples the chain, including neighboring arc
    # centers with no dimension. Free radii can adjust but cannot inherit the
    # nearby annotation merely because they belong to the same contour.
    assert result["entities"][3]["radius"]!=graph["entities"][3]["radius"]
    assert result["entities"][3]["radius"]!=3.
    assert result["diagnostics"]["optimizer_active_variable_count"]==result["diagnostics"]["variable_count"]
    assert result["diagnostics"]["rank_excludes_soft_prior"]
    assert result["underconstrained"]


def _mask_observation(points,**overrides):
    return {"units":"mm","points":points,"provenance":"input_mask_boundary",
            "oracle_mask_conditioned":False,"reference_dxf_read":False,**overrides}


def test_raw_mask_prior_preserves_exact_radius_and_improves_source_shape_without_file_reads(monkeypatch):
    from pathlib import Path
    graph={"units":"mm","nodes":[{"id":"a","x":0.,"y":0.},{"id":"b","x":6.,"y":0.}],
           "entities":[{"id":"arc","type":"ARC","start_node":"a","end_node":"b","start":[0.,0.],"end":[6.,0.],"center":[3.,math.sqrt(55)],"radius":8.,"clockwise":False},
                       {"id":"line","type":"LINE","start_node":"b","end_node":"a","start":[6.,0.],"end":[0.,0.]}]}
    angles=np.linspace(math.atan2(-math.sqrt(55),-3),math.atan2(-math.sqrt(55),3),121)
    source_arc=np.column_stack([3+8*np.cos(angles),math.sqrt(55)+8*np.sin(angles)])
    source_arc[0]=[0.,0.];source_arc[-1]=[6.,0.]
    points=np.vstack([source_arc,np.column_stack([np.linspace(6,0,61)[1:],np.zeros(60)])]).tolist()
    observation=_mask_observation(points);original=copy.deepcopy(observation)
    constraint={"id":"radius","kind":"radius","entities":["arc"],"nodes":[],"value":5.,"record_id":"r0","source":"ocr_local_binding"}
    endpoint_only=solve_parametric(graph,[constraint])
    def forbidden(*args,**kwargs):raise AssertionError("Solver may not read files")
    monkeypatch.setattr(Path,"read_text",forbidden);monkeypatch.setattr(Path,"read_bytes",forbidden)
    result=solve_parametric(graph,[constraint],source_observation=observation)
    def radial_error(solution):
        arc=solution["entities"][0]
        return float(np.mean(np.abs(np.linalg.norm(source_arc[10:-10]-arc["center"],axis=1)-arc["radius"])))
    assert result["accepted"] and radial_error(result)<radial_error(endpoint_only)*.8
    assert result["entities"][0]["radius"]==5. and result["strict_radius_contract"]["satisfied"]
    assert result["validation"]["max_gap"]==0 and result["validation"]["geometry_valid"]
    prior=result["diagnostics"]["raw_source_prior"]
    assert prior["enabled"] and not prior["hard_constraint"] and not prior["reference_dxf_read"]
    assert not result["all_dimensions_verified"] and not result["diagnostics"]["source_curve_prior"]["enabled"]
    assert observation==original


def test_raw_mask_prior_supports_nonradius_constraints_without_changing_empty_constraint_behavior():
    graph=rectangle();points=[[0.,.2],[10.,.2],[10.,5.2],[0.,5.2],[0.,.2]]
    observation=_mask_observation(points)
    unchanged=solve_parametric(graph,[],source_observation=observation)
    assert unchanged["entities"]==graph["entities"] and not unchanged["accepted"]
    constraint={"id":"width","kind":"distance_x","nodes":["v0","v1"],"entities":[],"record_id":"r0","value":10.,"source":"ocr_local_binding"}
    result=solve_parametric(graph,[constraint],source_observation=observation)
    assert result["accepted"] and result["constraints"][0]["passed"]
    assert sum(abs(node["y"]-points[index][1]) for index,node in enumerate(result["nodes"]))<.6
    assert result["diagnostics"]["optimizer"]=="SLSQP_source_observation"


@pytest.mark.parametrize("change",[
    {"units":"pixel"},{"provenance":"reference_geometry"},{"reference_dxf_read":True},
    {"points":[[0.,0.],[10.,0.],[10.,5.],[0.,5.]]},
    {"points":[[0.,0.],[float("nan"),0.],[10.,5.],[0.,0.]]},
])
def test_raw_source_prior_rejects_invalid_observation(change):
    observation={**_mask_observation([[0.,0.],[10.,0.],[10.,5.],[0.,5.],[0.,0.]]),**change}
    with pytest.raises(ValueError,match="Source observation"):
        solve_parametric(rectangle(),[],source_observation=observation)


def test_raw_mask_evidence_is_invariant_to_vertex_density():
    from contour_agent.parametric_solver import _raw_source_prior
    graph=rectangle();scale=math.sqrt(125)
    sparse=[[0.,0.],[10.,0.],[10.,5.],[0.,5.],[0.,0.]]
    # The same straight source edge can be stored as two vertices or hundreds
    # of raster samples. It must not change the fitted evidence objective.
    dense=[[float(x),0.] for x in np.linspace(0,10,401)]+sparse[2:]
    candidate=copy.deepcopy(graph["entities"])
    for entity in candidate:
        for key in ("start","end"):entity[key][1]+=.2
    first,receipt=_raw_source_prior(_mask_observation(sparse),graph,graph["entities"],scale)
    second,_=_raw_source_prior(_mask_observation(dense),graph,graph["entities"],scale)
    assert float(np.dot(first(candidate),first(candidate)))==pytest.approx(
        float(np.dot(second(candidate),second(candidate))),rel=1e-10,abs=1e-12)
    assert receipt["source_quadrature"]=="uniform_physical_arclength_midpoints"
    assert receipt["weighting"]=="source_arclength_forward_and_primitive_arclength_reverse"


def test_raw_mask_evidence_does_not_multiply_when_a_line_is_split():
    from contour_agent.parametric_solver import _raw_source_prior
    graph=rectangle();split=copy.deepcopy(graph)
    split["nodes"].append({"id":"middle","x":5.,"y":0.})
    split["entities"][0].update(end=[5.,0.],end_node="middle")
    split["entities"].insert(1,{"id":"second_half","type":"LINE","start_node":"middle","end_node":"v1",
                                 "start":[5.,0.],"end":[10.,0.]})
    points=[[0.,0.],[10.,0.],[10.,5.],[0.,5.],[0.,0.]]
    costs=[]
    for topology in (graph,split):
        prior,_=_raw_source_prior(_mask_observation(points),topology,topology["entities"],math.sqrt(125))
        candidate=copy.deepcopy(topology["entities"])
        for entity in candidate:
            for key in ("start","end"):entity[key][1]+=.2
        residual=prior(candidate);costs.append(float(np.dot(residual,residual)))
    # Bounded midpoint quadrature may differ slightly near corners, but an
    # equivalent partition must not add another full-primitive RMS penalty.
    assert costs[0]==pytest.approx(costs[1],rel=.01)


def _declare_constructed_radius(entity):
    evidence={"record_id":"historical_r","nominal":entity["radius"],
              "arrowhead_verified":True,"target_source_px":[12.,34.]}
    entity.update(radius_constructed=True,parameter_source="multimodal_annotation_guided_arc_refit",
                  radius_binding_status="constructed_unverified",source_fit_error_px=.25,
                  radius_binding=copy.deepcopy(evidence),radius_annotation_evidence=copy.deepcopy(evidence))


def test_constructed_radius_is_preserved_without_promoting_ambiguous_history_to_binding():
    from contour_agent.parametric_solver import _sample_entity
    graph=rounded_graph(noise=0);_declare_constructed_radius(graph["entities"][1])
    points=np.vstack([_sample_entity(entity,31)[:-1] for entity in graph["entities"]])
    points=np.vstack([points,points[0]]).tolist()
    constraints=[dimensions_and_relations()[0]]
    result=solve_parametric(graph,constraints,source_observation=_mask_observation(points))
    assert result["accepted"],result["validation"]
    assert result["entities"][1]["radius"]==graph["entities"][1]["radius"]
    assert result["strict_radius_contract"]["required_count"]==0
    assert not result["strict_radius_contract"]["unbound_radius_coverage_verified"]
    assert len(result["constraints"])==1 and result["constraints"][0]["record_id"]=="r000"
    preservation=result["constructed_radius_preservation"]
    assert preservation["required_count"]==1 and preservation["satisfied"]
    assert not preservation["binding_verified"] and preservation["annotation_constraint_count_added"]==0
    assert all(not row["binding_verified"] for row in preservation["checks"])
    assert result["diagnostics"]["eliminated_exact_radius_constraint_count"]==0
    assert result["diagnostics"]["eliminated_constructed_geometry_radius_count"]==1
    assert result["diagnostics"]["remaining_shape_dof"]==result["diagnostics"]["numerical_remaining_shape_dof"]+1
    assert not result["engineering_verified"] and not result["all_dimensions_verified"]


def test_constructed_radius_conflicting_with_fresh_constraint_is_rejected_without_averaging():
    graph=rounded_graph(noise=0);_declare_constructed_radius(graph["entities"][1])
    constraint={**dimensions_and_relations()[2],"value":3.}
    result=solve_parametric(graph,[constraint])
    assert not result["accepted"] and result["status"]=="conflict"
    assert result["entities"]==graph["entities"] and result["candidate_entities"] is None
    assert result["diagnostics"]["conflicting_constructed_radius_entity_ids"]==["g001"]
    assert result["strict_radius_contract"]["required_count"]==1
    assert not result["strict_radius_contract"]["satisfied"]
    assert result["constructed_radius_preservation"]["satisfied"]


@pytest.mark.parametrize("change",[
    {"radius_constructed":False},{"parameter_source":"source_topology_hypothesis"},
    {"radius_binding_status":"unresolved_fixed_radius_fit_failed"},
    {"source_fit_error_px":None},{"radius_annotation_evidence":None},
    {"radius_binding":{"record_id":"historical_r","nominal":3.,"arrowhead_verified":True,"target_source_px":[12.,34.]}},
])
def test_incomplete_or_inconsistent_construction_history_never_becomes_a_radius_constraint(change):
    graph=rounded_graph(noise=0);_declare_constructed_radius(graph["entities"][1])
    graph["entities"][1].update(change)
    result=solve_parametric(graph,[dimensions_and_relations()[0]])
    assert result["constructed_radius_preservation"]["required_count"]==0
    assert result["constructed_radius_preservation"]["ignored"]
    assert result["diagnostics"]["eliminated_constructed_geometry_radius_count"]==0
    assert result["strict_radius_contract"]["required_count"]==0


def _budgeted_mask(points,maximum=.04,step=.01):
    return _mask_observation(points,boundary_error_budget={"units":"mm","maximum_deviation":maximum,
        "sampling_step":step,"source":"initial_curve_fit_total_deviation_budget_px"})


def test_source_budget_search_repairs_a_local_maximum_hidden_by_RMS():
    from contour_agent.vectorize import assess_fit_quality
    graph=rectangle();points=[[0.,0.],[10.,0.],[10.4,5.],[0.,5.],[0.,0.]]
    constraint={"id":"width","kind":"distance_x","nodes":["v0","v1"],"entities":[],
                "record_id":"r0","value":10.,"source":"ocr_local_binding"}
    rms_only=solve_parametric(graph,[constraint],source_observation=_mask_observation(points))
    maximum=.02;step=.005
    assert rms_only["accepted"]
    before=assess_fit_quality(points,rms_only["entities"],max_step_px=step)
    assert before["source_boundary_deviation_px"]["conservative_upper_bound_px"]>maximum
    observation=_budgeted_mask(points,maximum,step);unchanged=copy.deepcopy(observation)
    result=solve_parametric(graph,[constraint],source_observation=observation)
    assert result["accepted"],result["validation"]
    assert result["validation"]["source_boundary_budget_passed"]
    assert result["validation"]["source_boundary_budget_audit"]["conservative_max_deviation"]<=maximum
    assert result["constraints"][0]["passed"] and result["validation"]["max_gap"]==0
    receipt=result["diagnostics"]["source_budget_search"]
    assert len(receipt["rounds"])<=5 and not receipt["acceptance_thresholds_changed"]
    assert not receipt["infeasibility_proven"] and not receipt["reference_dxf_read"]
    assert observation==unchanged and result["underconstrained"]


def test_source_budget_search_moves_shared_joints_while_radius_remains_exact():
    from contour_agent.parametric_solver import _sample_entity
    graph=rounded_graph(noise=0)
    source=rounded_graph(width=20.,height=12.,radius=3.,noise=0)
    points=np.vstack([_sample_entity(entity,61)[:-1] for entity in source["entities"]])
    points=np.vstack([points,points[0]]).tolist()
    result=solve_parametric(graph,dimensions_and_relations(),source_observation=_budgeted_mask(points,.06,.01))
    assert result["accepted"],result["validation"]
    assert result["validation"]["source_boundary_budget_passed"]
    assert all(entity["radius"]==3. for entity in result["entities"] if entity["type"]=="ARC")
    assert result["strict_radius_contract"]["satisfied"] and result["strict_radius_contract"]["required_count"]==4
    assert result["validation"]["max_gap"]==0 and result["validation"]["max_radial_error"]<1e-10
    assert result["diagnostics"]["maximum_node_displacement"]>.1
    search=result["diagnostics"]["source_budget_search"]
    assert search["feasible_checkpoint_found"] or search["warm_RMS_reused"]
    assert not result["engineering_verified"] and not result["all_dimensions_verified"]


def test_incompatible_source_budget_keeps_fallback_without_claiming_infeasibility():
    graph=rectangle();points=[[0.,0.],[11.,0.],[11.,5.],[0.,5.],[0.,0.]]
    constraints=[{"id":"width","kind":"distance_x","nodes":["v0","v1"],"entities":[],
                  "record_id":"r0","value":10.,"source":"ocr_local_binding"}]
    for index in range(4):
        constraints.append({"id":f"axis{index}","kind":"horizontal" if index%2==0 else "vertical",
                            "entities":[f"g{index}"],"nodes":[],"value":None,"record_id":None,"source":"source_geometry"})
    result=solve_parametric(graph,constraints,source_observation=_budgeted_mask(points,.1,.02))
    assert not result["accepted"] and result["status"]=="source_budget_search_failed"
    assert result["entities"]==graph["entities"] and result["nodes"]==graph["nodes"]
    search=result["diagnostics"]["source_budget_search"]
    assert not search["passed"] and len(search["rounds"])<=5 and not search["infeasibility_proven"]
    assert "not proof" in " ".join(result["validation"]["issues"])


@pytest.mark.parametrize("budget",[
    {"units":"pixel","maximum_deviation":.04,"sampling_step":.01,"source":"initial_curve_fit_total_deviation_budget_px"},
    {"units":"mm","maximum_deviation":.04,"sampling_step":.04,"source":"initial_curve_fit_total_deviation_budget_px"},
    {"units":"mm","maximum_deviation":.04,"sampling_step":.01,"source":"updated_candidate_budget"},
])
def test_source_budget_rejects_incompatible_units_or_mutable_provenance(budget):
    observation=_mask_observation([[0.,0.],[10.,0.],[10.,5.],[0.,5.],[0.,0.]],boundary_error_budget=budget)
    with pytest.raises(ValueError,match="Source boundary"):
        solve_parametric(rectangle(),[],source_observation=observation)


def test_source_budget_evaluation_exhaustion_is_bounded_and_retains_original_graph():
    graph=rectangle();points=[[0.,0.],[10.,0.],[10.4,5.],[0.,5.],[0.,0.]]
    observation=_budgeted_mask(points,.02,.005)
    observation["boundary_error_budget"]["max_objective_evaluations"]=1
    constraint={"id":"width","kind":"distance_x","nodes":["v0","v1"],"entities":[],
                "record_id":"r0","value":10.,"source":"ocr_local_binding"}
    result=solve_parametric(graph,[constraint],source_observation=observation)
    assert not result["accepted"] and result["status"]=="source_budget_search_exhausted"
    assert result["entities"]==graph["entities"] and result["nodes"]==graph["nodes"]
    search=result["diagnostics"]["source_budget_search"]
    assert search["objective_evaluations"]==1 and search["exhaustion_reason"]=="objective_evaluations"
    assert not search["infeasibility_proven"] and not search["acceptance_thresholds_changed"]


def test_feasible_checkpoint_survives_RMS_wall_exhaustion_after_complete_revalidation(monkeypatch,tmp_path):
    import contour_agent.parametric_solver as solver
    actual_minimize=solver.minimize;actual_clock=solver.time.monotonic
    elapsed=[0.];calls=[0]
    monkeypatch.setattr(solver.time,"monotonic",lambda:actual_clock()+elapsed[0])
    def exhaust_optional_refinement(*args,**kwargs):
        calls[0]+=1
        result=actual_minimize(*args,**kwargs)
        # Warm RMS fails maximum-distance certification; the next optimizer
        # finds a source-feasible checkpoint. Expire wall budget before the
        # optional RMS refinement, as a slow online preflight could do.
        if calls[0]==2:elapsed[0]=121.
        return result
    monkeypatch.setattr(solver,"minimize",exhaust_optional_refinement)
    graph=rectangle();points=[[0.,0.],[10.,0.],[10.4,5.],[0.,5.],[0.,0.]]
    constraint={"id":"width","kind":"distance_x","nodes":["v0","v1"],"entities":[],
                "record_id":"r0","value":10.,"source":"ocr_local_binding"}
    result=solver.solve_parametric(graph,[constraint],source_observation=_budgeted_mask(points,.02,.005),output_dir=tmp_path)
    search=result["diagnostics"]["source_budget_search"]
    assert search["exhausted"] and search["feasible_checkpoint_found"]
    assert not search["RMS_refinement_accepted"] and search["exhaustion_reason"]=="wall_seconds"
    assert result["accepted"] and result["status"]=="accepted",result["validation"]
    assert all(row["passed"] for row in result["constraints"])
    assert result["validation"]["source_boundary_budget_passed"] and result["validation"]["geometry_valid"]
    assert result["validation"]["source_displacement_passed"] and result["validation"]["winding_preserved"]
    assert (tmp_path/"source-budget-checkpoint.json").is_file()
    assert not result["engineering_verified"] and not result["all_dimensions_verified"]

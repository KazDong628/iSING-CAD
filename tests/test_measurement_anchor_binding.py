"""Original-ink corner observations survive API selection and native readback."""
from copy import deepcopy
import json
import math

import ezdxf
import numpy as np
from PIL import Image, ImageDraw
import pytest

from contour_agent.binding_provider import bounded_inventory
from contour_agent.constraint_binding import (_length_candidates, _samples, _source_transform,
                                             analyze_constraint_bindings)
from contour_agent.measurement_anchors import virtual_corner_locations, support_intersection
from contour_agent.parametric_solver import solve_parametric, native_measurement_checks
from contour_agent.relation_contract import relation_checks
from contour_agent.service import _recover_numerical_constraints
from test_parametric_solver import rounded_graph, rectangle


def source(tmp_path):
    graph=rounded_graph(width=60.,height=40.,radius=8.,noise=0.)
    transform=lambda points:np.asarray(points)*[5.,-5.]+[80.,330.]
    for row in graph["nodes"]:
        row["source_px"]=transform([[row["x"],row["y"]]])[0].tolist()
    graph.update(proposal_tolerance_px=5.,relations=[])
    graph["coordinate_system"]={"units":"mm","origin_source_px":[80.,330.]}
    image=Image.new("RGB",(470,400),"white");draw=ImageDraw.Draw(image)
    for entity in graph["entities"]:
        draw.line([tuple(point) for point in _samples(entity,transform)],fill="black",width=2)
    draw.line((80,70,380,70),fill="black",width=2)
    for x in (80,380):draw.line((x,60,x,145),fill="black",width=2)
    records=[{"id":"r000","text":"60","parsed":{"kind":"length","nominal":60.},
              "box":[[215,42],[245,42],[245,60],[215,60]]}]
    path=tmp_path/"source.png";image.save(path)
    model={"scale":{"pixels_per_mm":5.},"coordinate_system":graph["coordinate_system"]}
    return path,np.asarray(image.convert("L")),records,graph,model


def test_actual_source_extensions_bind_common_material_levels_and_joint_solve(tmp_path):
    path,gray,records,graph,model=source(tmp_path)
    candidates=_length_candidates(gray,records,graph,5.,5.)
    assert len(candidates)==1,candidates
    candidate=candidates[0]
    assert candidate["local_reliable"] and candidate["nodes"]==[]
    # These axis-aligned support lines make the corner's measured coordinate
    # the material face level. Do not lock one arbitrary fitted endpoint.
    assert all(anchor["type"]=="support_level" and anchor["axis"]=="x" for anchor in candidate["anchors"])
    assert candidate["entities"]==["g002","g006"]
    assert all(row["support_kind"]=="source_supported_derived_location"
               for row in candidate["evidence"]["observed_station_groups"])
    document={"records":[{k:v for k,v in records[0].items() if k not in {"id","parsed"}}]}
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"bindings")
    dimensions=[row for row in result["constraints"] if row.get("record_id") is not None]
    assert len(dimensions)==1 and dimensions[0]["anchors"]==candidate["anchors"]
    solved=solve_parametric(graph,result["constraints"])
    assert solved["accepted"],solved
    assert solved["strict_measurement_contract"]["satisfied"]
    assert solved["strict_measurement_contract"]["checks"][0]["actual"]==pytest.approx(60.,abs=1e-7)


def test_corner_candidates_do_not_use_nominal_to_locate_supports(tmp_path):
    _,gray,records,graph,_=source(tmp_path)
    transform=_source_transform({},graph)
    first=virtual_corner_locations(gray,records,graph,transform,5.)
    records[0]["parsed"]["nominal"]=999.
    second=virtual_corner_locations(gray,records,graph,transform,5.)
    assert first==second and first
    assert all(row["evidence"]["nominal_used"] is False for row in first)


def test_unobserved_or_text_only_supports_cannot_create_virtual_corner(tmp_path):
    _,gray,records,graph,_=source(tmp_path)
    transform=_source_transform({},graph)
    assert virtual_corner_locations(np.full_like(gray,255),records,graph,transform,5.)==[]
    text=[{"box":[[0,0],[469,0],[469,399],[0,399]]}]
    assert virtual_corner_locations(gray,text,graph,transform,5.)==[]


def test_parallel_and_remote_support_intersections_are_not_corner_evidence(tmp_path):
    _,gray,records,graph,_=source(tmp_path)
    assert support_intersection({"start":[0.,0.],"end":[100.,0.]},
                                {"start":[0.,5.],"end":[100.,5.1]}) is None
    # A local short connector between almost parallel finite supports has a
    # well-defined but remote theoretical intersection. It is not a fillet
    # corner witness even if both strokes are perfectly dark.
    graph["entities"]=deepcopy(graph["entities"][:3])
    graph["entities"][0].update(start=[0.,0.],end=[30.,0.])
    graph["entities"][1].update(start=[30.,0.],end=[30.,3.])
    graph["entities"][2].update(start=[30.,3.],end=[0.,4.])
    assert virtual_corner_locations(gray,records,graph,_source_transform({},graph),5.)==[]


def test_provider_packet_keeps_typed_references_but_drops_arbitrary_coordinates(tmp_path):
    _,gray,records,graph,_=source(tmp_path)
    candidate=_length_candidates(gray,records,graph,5.,5.)[0]
    expected_anchors=deepcopy(candidate["anchors"])
    candidate.update(id="c000",ground_truth_coordinates=[[999.,999.]])
    candidate["anchors"][0]["literal_private_coordinates"]=[999.,999.]
    inventory={"units":"mm","records":records,"candidates":[candidate],"relations":[]}
    packet=bounded_inventory(inventory)
    assert packet["candidates"][0]["anchors"]==expected_anchors
    assert "literal_private_coordinates" not in json.dumps(packet)
    assert "ground_truth" not in json.dumps(packet)


def native_document(graph):
    document=ezdxf.new("R2010");document.units=4
    for entity in graph["entities"]:
        if entity["type"]=="LINE":document.modelspace().add_line(entity["start"],entity["end"])
        else:
            angles=[math.degrees(math.atan2(point[1]-entity["center"][1],point[0]-entity["center"][0]))
                    for point in (entity["start"],entity["end"])]
            if entity["clockwise"]:angles.reverse()
            document.modelspace().add_arc(entity["center"],entity["radius"],*angles)
    return document


@pytest.mark.parametrize("error",[0.,1e-5])
def test_native_dxf_recomputes_hard_corner_measurement_instead_of_old_residual(error):
    graph=rectangle(width=10.+error,height=5.)
    constraint={"id":"width","kind":"distance_x","nodes":[],"entities":["g0","g1","g3"],
        "anchors":[{"type":"line_intersection","entities":["g3","g0"]},
                   {"type":"line_intersection","entities":["g0","g1"]}],
        "value":10.,"record_id":"r000","source":"ocr_api_binding","passed":True,"actual":10.}
    model={"entities":graph["entities"],"coordinate_system":{"units":"mm"},
           "parameterization":{"constraints":[constraint]}}
    document=native_document(graph)
    strict=relation_checks(graph["entities"],[constraint],dxf_document=document)
    result=_recover_numerical_constraints(model,document,strict)
    assert result["passed"]==(error==0.),result
    assert result["checks"][0]["tolerance"]==1e-7
    assert result["checks"][0]["dxf"]["actual"]==pytest.approx(10.+error,abs=1e-10)
    assert result["checks"][0]["dxf"]["derived_anchor_points"]
    assert result["solver_called"] is False


def test_api_selects_source_candidate_identity_without_creating_corner_coordinates(tmp_path):
    path,_,records,graph,model=source(tmp_path)
    class Selector:
        def select(self,image,topology,inventory):
            candidate=next(row for row in inventory["candidates"] if row.get("anchors"))
            self.packet=bounded_inventory(inventory)
            return {"status":"succeeded","network_requests":1,"http_success":True,"schema_success":True,
                "bindings":[{"record_id":candidate["record_id"],"candidate_id":candidate["id"],"observed_text":"60"}],
                "relations":[],"input_record_ids":[row["id"] for row in inventory["records"]],
                "input_candidate_ids":[row["id"] for row in inventory["candidates"]],
                "input_relation_ids":[row["id"] for row in inventory["relations"]]}
    provider=Selector()
    document={"records":[{k:v for k,v in records[0].items() if k not in {"id","parsed"}}]}
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"online",provider=provider,use_api=True)
    assert result["counts"]["api_accepted"]==1
    constraint=next(row for row in result["constraints"] if row.get("record_id"))
    assert constraint["source"]=="ocr_api_binding" and constraint["anchors"]
    assert constraint["value"]==60.
    solution=solve_parametric(graph,result["constraints"])
    assert solution["accepted"],solution


@pytest.mark.parametrize("error,anchors,passed",[(0.,True,True),(1e-5,True,False),(.04,False,True),(.06,False,False)])
def test_independent_native_measurement_certificate_keeps_separate_tolerances(error,anchors,passed):
    graph=rectangle(width=10.+error)
    row={"id":"w","kind":"distance_x","nodes":["v0","v1"],"entities":[],"value":10.,
         "record_id":"r000","source":"ocr_local_binding","passed":True}
    if anchors:row.update(nodes=[],anchors=[{"type":"node","node_id":node} for node in ("v0","v1")])
    result=native_measurement_checks(graph["entities"],[row],dxf_document=native_document(graph))
    assert result["passed"]==passed,result
    assert result["dxf_readback_performed"] and result["native_mapping_verified"]
    assert result["checks"][0]["tolerance_mm"]==(1e-7 if anchors else .05)


def test_native_measurement_readback_handles_clockwise_arcs_and_rejects_reordered_entities(tmp_path):
    from test_parametric_pipeline import fixture_model
    _,model=fixture_model(tmp_path)
    row={"id":"height","kind":"distance_y","nodes":[],"entities":["g000","g001","g003"],
         "anchors":[{"type":"line_intersection","entities":["g000","g003"]},
                    {"type":"line_intersection","entities":["g000","g001"]}],
         "value":10.,"record_id":"r000","source":"ocr_api_binding"}
    valid=native_measurement_checks(model["entities"],[row],dxf_document=native_document(model))
    assert valid["passed"],valid
    reordered=deepcopy(model);reordered["entities"]=[model["entities"][1],model["entities"][0],*model["entities"][2:]]
    invalid=native_measurement_checks(model["entities"],[row],dxf_document=native_document(reordered))
    assert not invalid["passed"] and invalid["issues"]==["native_entity_mapping_mismatch"]


def test_export_rejects_false_solver_claim_on_hard_anchor_measurement(tmp_path):
    from contour_agent.parametric_pipeline import export_parametric
    from test_parametric_pipeline import fixture_model
    path,model=fixture_model(tmp_path)
    row={"id":"height","kind":"distance_y","nodes":[],"entities":[],
         "anchors":[{"type":"node","node_id":"v000"},{"type":"node","node_id":"v001"}],
         "value":10.00001,"record_id":"r000","source":"ocr_api_binding","passed":True,"actual":10.00001}
    with pytest.raises(ValueError,match="linear measurements"):
        export_parametric(path,model,{"accepted":True,"entities":model["entities"],"constraints":[row]},tmp_path/"bad")
    row["value"]=10.
    exported=export_parametric(path,model,{"accepted":True,"entities":model["entities"],"constraints":[row]},tmp_path/"good")
    audit=exported["validation"]["native_measurement_validation"]
    assert audit["passed"] and audit["dxf_readback_performed"] and audit["required_count"]==1

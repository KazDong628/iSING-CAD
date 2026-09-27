"""A complete oracle raster remains the observation across later CAD stages."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from contour_agent.parametric_pipeline import (_oracle_mask_validation,
                                              _solved_source_validation,
                                              _topology_source_validation)
from contour_agent.topology import build_topology


def fixture(tmp_path):
    image=tmp_path/"drawing.png"
    pixels=np.array([[30.,30.],[130.,30.],[130.,130.],[30.,130.],[30.,30.]])
    gray=np.full((160,160),255,np.uint8)
    cv2.polylines(gray,[pixels.astype(np.int32)],True,0,2)
    cv2.imencode(".png",gray)[1].tofile(str(image))
    points=pixels*[1.,-1.]
    entities=[{"id":f"g{i:03d}","type":"LINE","start":start.tolist(),"end":end.tolist()}
              for i,(start,end) in enumerate(zip(points[:-1],points[1:]))]
    model={"entities":entities,"coordinate_system":{"units":"pixel","origin_source_px":[0.,0.]},
           "scale":{"status":"unresolved"},"extraction":{"raw_polyline_px":pixels.tolist()},
           "curve_fit":{"tolerance_px":1.,"total_deviation_budget_px":1.},"oracle_mask_conditioned":True}
    return image,model


def test_oracle_initial_topology_preserves_cad_and_never_repairs_mask(tmp_path,monkeypatch):
    image,model=fixture(tmp_path)
    def forbidden(*args,**kwargs):pytest.fail("Oracle mask is a fixed observation, not a segmentation repair target")
    monkeypatch.setattr("contour_agent.topology._repair_local_shortcuts",forbidden)
    monkeypatch.setattr("contour_agent.topology._repair_text_occlusions",forbidden)
    monkeypatch.setattr("contour_agent.topology._candidate",forbidden)
    graph=build_topology(image,{},model,tmp_path/"topology")
    assert [(e["type"],e["start"],e["end"]) for e in graph["entities"]]==[
        (e["type"],e["start"],e["end"]) for e in model["entities"]]
    assert graph["source_evidence"]["selection"]=="oracle_mask_preserved_initial_cad_geometry"
    assert graph["source_evidence"]["local_corrections"]==0
    assert _topology_source_validation(image,model,graph)["passed"]


def shifted(entities,offset):
    result=deepcopy(entities)
    for entity in result:
        for key in ("start","end","center"):
            if key in entity:entity[key][0]+=offset
    return result


def test_oracle_guard_never_rebases_to_intermediate_boundary(tmp_path):
    _,model=fixture(tmp_path)
    assert _oracle_mask_validation(model,shifted(model["entities"],.5))["passed"]
    receipt=_oracle_mask_validation(model,shifted(model["entities"],1.5))
    assert not receipt["passed"]
    assert receipt["original_deviation_budget_px"]==1.
    assert receipt["observation"]=="initial_extraction_raw_polyline_px"
    assert receipt["reason"]=="original_oracle_mask_budget_exceeded"


def test_oracle_topology_gate_uses_original_mask_not_candidate_residual(tmp_path):
    image,model=fixture(tmp_path)
    graph=build_topology(image,{},model,tmp_path/"topology")
    graph["entities"]=shifted(graph["entities"],3.)
    for node in graph["nodes"]:
        node["x"]+=3.;node["source_px"][0]+=3.
    # Candidate generation can report zero residual to its own input topology;
    # that report must never replace the original raster observation.
    graph["source_evidence"]["source_deviation"]={"source_boundary_deviation_px":{"conservative_upper_bound_px":0.}}
    receipt=_topology_source_validation(image,model,graph)
    assert receipt["coordinate_mapping"]["passed"]
    assert receipt["reasons"]==["original_oracle_mask_budget_exceeded"]
    assert receipt["mask_fidelity_gate_used"]


def test_oracle_solver_guard_cannot_trade_mask_for_good_ink_support(tmp_path,monkeypatch):
    image,model=fixture(tmp_path)
    graph=build_topology(image,{},model,tmp_path/"topology")
    # A nearby dimension or hatch can provide perfect ink support while being
    # the wrong boundary. The original raster observation remains independent.
    monkeypatch.setattr("contour_agent.topology._StrokeEvidence.summarize",
                        lambda *a,**k:{"edge_supported_fraction":1.,"p90_edge_distance_px":0.})
    changed=shifted(graph["entities"],3.)
    receipt=_solved_source_validation(image,{},model,graph,changed)
    assert not receipt["passed"]
    assert receipt["before"]==receipt["after"]
    assert "original_oracle_mask_budget_exceeded" in receipt["reasons"]
    ordinary=deepcopy(model);ordinary["oracle_mask_conditioned"]=False
    assert _solved_source_validation(image,{},ordinary,graph,changed)["passed"]


def test_missing_oracle_observation_fails_closed_but_normal_flow_unchanged(tmp_path):
    _,model=fixture(tmp_path)
    del model["curve_fit"]["total_deviation_budget_px"]
    assert _oracle_mask_validation(model,model["entities"])["reason"]=="original_oracle_mask_observation_unavailable"
    model["oracle_mask_conditioned"]=False
    assert _oracle_mask_validation(model,model["entities"])=={"applicable":False,"passed":True}

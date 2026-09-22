"""Integration checks use synthetic source geometry; no models, references or API."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import ezdxf
import numpy as np
from PIL import Image
import pytest

from contour_agent.config import Settings
from contour_agent.parametric_pipeline import CORE, export_parametric, refine_parametric
from contour_agent.service import AgentService
from contour_agent.store import JobStore
from contour_agent.vectorize import _sample_entities


def fixture_model(tmp_path, units="mm"):
    image=tmp_path/"source.png"
    Image.new("RGB",(200,200),"white").save(image)
    entities=[{"id":"g000","type":"LINE","start":[-4.,-5.],"end":[-4.,5.]},
              {"id":"g001","type":"LINE","start":[-4.,5.],"end":[0.,5.]},
              {"id":"g002","type":"ARC","start":[0.,5.],"end":[0.,-5.],"center":[0.,0.],"radius":5.,"clockwise":True},
              {"id":"g003","type":"LINE","start":[0.,-5.],"end":[-4.,-5.]}]
    scale=5. if units=="mm" else 1.
    raw,_,_=_sample_entities(entities)
    pixels=np.c_[raw[:,0]*scale+100,100-raw[:,1]*scale].tolist()
    baseline={"entities":entities,"coordinate_system":{"units":units,"origin_source_px":[100,100]},
              "scale":{"status":"resolved" if units=="mm" else "unresolved","pixels_per_mm":scale if units=="mm" else None},
              "extraction":{"raw_polyline_px":pixels},"polyline_px":pixels,"issues":[],"curve_fit":{"passed":True}}
    return image,baseline


@pytest.mark.parametrize("units",["mm","pixel"])
def test_parametric_export_preserves_arc_branch_image_y_and_dxf_units(tmp_path,units):
    image,baseline=fixture_model(tmp_path,units)
    solution={"accepted":True,"entities":baseline["entities"],"validation":{"passed":True,"constraint_subset_satisfied":True}}
    output=tmp_path/"export"
    model=export_parametric(image,baseline,solution,output)
    document=ezdxf.readfile(output/"drawing.dxf")
    assert document.units==(4 if units=="mm" else 0)
    arc=list(document.modelspace().query("ARC"))[0]
    assert arc.dxf.radius==5
    assert arc.dxf.start_angle==270 and arc.dxf.end_angle==90
    pixels=np.asarray(model["fitted_polyline_px"])
    scale=5. if units=="mm" else 1.
    assert pixels[:,0].max()==pytest.approx(100+5*scale,abs=.05)
    assert pixels[:,1].min()==pytest.approx(100-5*scale)
    assert pixels[:,1].max()==pytest.approx(100+5*scale)
    assert model["validation"]["dxf_readback"]["passed"]
    assert not model["validation"]["dimensions_verified"] and not model["validation"]["reference_verified"]
    assert model["curve_fit"]["passed"] is None and not model["curve_fit"]["used_as_acceptance_gate"]


def prepared(tmp_path):
    image,source=fixture_model(tmp_path)
    output=tmp_path/"runtime"/"jobs"/"persisted"/"automatic-001"
    baseline=export_parametric(image,source,{"accepted":True,"entities":source["entities"],"validation":{"passed":True}},output)
    return image,baseline,output


def test_valid_parametric_export_cannot_upgrade_incomplete_material_draft(tmp_path):
    image,baseline=fixture_model(tmp_path)
    baseline["complete_material_exterior"]=False
    baseline["validation"]={"material_connectivity":{"components_4":2,"complete_exterior_candidate":False}}
    solution={"accepted":True,"entities":baseline["entities"],"validation":{"passed":True}}
    model=export_parametric(image,baseline,solution,tmp_path/"export")
    assert model["validation"]["passed"]  # CAD validity remains independently reported.
    assert not model["automatic_completion"]
    assert not model["validation"]["complete_material_exterior"]
    assert model["completion_class"]=="incomplete_material_exterior_draft"
    assert model["validation"]["material_connectivity"]["components_4"]==2


def test_failed_source_validation_preserves_exact_baseline_without_api(tmp_path,monkeypatch):
    image,baseline,output=prepared(tmp_path)
    before={name:(output/name).read_bytes() for name in CORE}
    graph=supported_graph(image,baseline)
    graph["source_evidence"]["proposal_stroke_support"]["edge_supported_fraction"]=.1
    monkeypatch.setattr("contour_agent.topology.build_topology",lambda *a,**k:graph)
    def failed_binding(*args,**kwargs):pytest.fail("Unvalidated topology must not enter online binding")
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",failed_binding)
    model,stage=refine_parametric(image,{},baseline,output,use_api=True)
    assert model==baseline and not stage["accepted"] and stage["status"]=="candidate_rejected"
    assert stage["provider"]["status"]=="not_invoked" and stage["provider"]["network_requests"]==0
    assert {name:(output/name).read_bytes() for name in CORE}==before
    assert {name:(output/("baseline-"+name)).read_bytes() for name in CORE}==before
    assert "do not publish arbitrary" not in (output/"parametric-stage.json").read_text()


def seed_pending_job(tmp_path,baseline,output):
    stage={"status":"running","stage":"constraint_binding","accepted":False,
           "provider":{"status":"pending","network_requests":None,"http_success":None,"schema_success":False}}
    job={"id":"persisted","mode":"autonomous_image","status":"solving","updated_at":"2026-09-21T00:00:00+00:00",
         "events":[],"issues":[],"artifacts":{},"artifact_directory":str(output),"parameterization":stage,
         "automatic_completion":True,"validation":baseline["validation"],"geometry":{"entities":baseline["entities"]},
         "provider":{"status":"pending","network_requests":0},"use_api":True}
    JobStore(tmp_path/"runtime").save(job)
    return job


def recover(tmp_path,monkeypatch):
    monkeypatch.setattr("contour_agent.service.build_catalog",lambda *args:{"cases":[]})
    def no_network(*args,**kwargs):pytest.fail("Restart must not repeat any API stage")
    monkeypatch.setattr("contour_agent.binding_provider.BindingProvider.select",no_network)
    monkeypatch.setattr("contour_agent.vision_provider.VisionProvider.inspect",no_network)
    return AgentService(Settings(runtime_root=tmp_path/"runtime",api_key=""),recover_running=True)


def test_restart_keeps_valid_fragmented_draft_without_claiming_complete(tmp_path,monkeypatch):
    image,baseline,output=prepared(tmp_path)
    baseline["complete_material_exterior"]=False
    baseline["validation"]["complete_material_exterior"]=False
    baseline=export_parametric(image,baseline,{"accepted":True,"entities":baseline["entities"]},output)
    seed_pending_job(tmp_path,baseline,output)
    service=recover(tmp_path,monkeypatch)
    recovered=service.store.get("persisted")
    assert recovered["validation"]["passed"]
    assert recovered["status"]=="needs_review"
    assert not recovered["automatic_completion"]
    assert recovered["completion_class"]=="incomplete_material_exterior_draft"


def test_restart_never_marks_mixed_parametric_publication_completed(tmp_path,monkeypatch):
    image,baseline,output=prepared(tmp_path)
    for name in CORE:shutil.copyfile(output/name,output/("baseline-"+name))
    seed_pending_job(tmp_path,baseline,output)
    # Simulate process termination after the first CORE replacement: the newer
    # DXF is individually valid, but model.json and validation still describe
    # the old contour. Files merely existing must not imply a validated set.
    altered=deepcopy(baseline["entities"])
    for entity in altered:
        for key in ("start","end","center"):
            if key in entity:entity[key][0]+=1.
    candidate=tmp_path/"candidate"
    export_parametric(image,baseline,{"accepted":True,"entities":altered,"validation":{"passed":True}},candidate)
    shutil.copyfile(candidate/"drawing.dxf",output/"drawing.dxf")
    service=recover(tmp_path,monkeypatch)
    try:
        job=service.store.get("persisted")
        from contour_agent.automatic import _verify_dxf_readback
        model=json.loads((output/"model.json").read_text(encoding="utf-8"))
        integrity=_verify_dxf_readback(ezdxf.readfile(output/"drawing.dxf"),model["entities"],expected_units=4)
        assert not job["automatic_completion"] or integrity["passed"], "stale journal marked a mixed DXF/model set complete"
        if job["status"]=="completed":assert integrity["passed"]
    finally:service.close()


def test_restart_uses_already_persisted_binding_receipt_without_retry(tmp_path,monkeypatch):
    _,baseline,output=prepared(tmp_path)
    seed_pending_job(tmp_path,baseline,output)
    receipt={"status":"succeeded","network_requests":1,"http_success":True,"schema_success":True,"bindings":[],"relations":[]}
    (output/"constraint-bindings.json").write_text(json.dumps({"status":"completed","provider":receipt,"constraints":[],"counts":{"api_selected":0}}))
    # API and durable binding artifact completed just before the job checkpoint.
    service=recover(tmp_path,monkeypatch)
    try:
        job=service.store.get("persisted")
        assert job["parameterization"]["provider"]["http_success"] is True
        assert job["parameterization"]["provider"]["network_requests"]==1
        assert not job["parameterization"]["accepted"]
    finally:service.close()


def test_restart_checks_publication_preview_hashes_not_only_dxf_model(tmp_path,monkeypatch):
    image,baseline,output=prepared(tmp_path)
    before={name:(output/name).read_bytes() for name in CORE}
    for name in CORE:shutil.copyfile(output/name,output/("baseline-"+name))
    job=seed_pending_job(tmp_path,baseline,output)
    altered=deepcopy(baseline["entities"])
    for entity in altered:
        for key in ("start","end","center"):
            if key in entity:entity[key][0]+=1.
    candidate=tmp_path/"candidate"
    export_parametric(image,baseline,{"accepted":True,"entities":altered,"validation":{"passed":True}},candidate)
    for name in CORE:
        if name!="overlay.png":shutil.copyfile(candidate/name,output/name)
    job["parameterization"]["publication"]={"status":"pending",
        "candidate_sha256":{name:hashlib.sha256((candidate/name).read_bytes()).hexdigest() for name in CORE},
        "baseline_sha256":{name:hashlib.sha256(content).hexdigest() for name,content in before.items()}}
    JobStore(tmp_path/"runtime").save(job)
    service=recover(tmp_path,monkeypatch)
    try:
        recovered=service.store.get("persisted")
        assert recovered["automatic_completion"] and recovered["status"]=="completed"
        assert recovered["parameterization"]["publication"]["status"]=="rolled_back"
        assert {name:(output/name).read_bytes() for name in CORE}==before
    finally:service.close()


def test_cancel_at_publication_checkpoint_restores_every_baseline_file(tmp_path,monkeypatch):
    image,baseline,output=prepared(tmp_path)
    before={name:(output/name).read_bytes() for name in CORE}
    graph=supported_graph(image,baseline)
    monkeypatch.setattr("contour_agent.topology.build_topology",lambda *a,**k:graph)
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",lambda *a,**k:{
        "constraints":[],"provider":{"status":"disabled","network_requests":0},"counts":{},"issues":[]})
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",lambda *a,**k:{
        "accepted":True,"entities":baseline["entities"],"underconstrained":True,"validation":{"passed":True}})
    def cancel(stage,message):
        if stage=="topology_export":raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        refine_parametric(image,{},baseline,output,progress=cancel)
    assert {name:(output/name).read_bytes() for name in CORE}==before
    stage=json.loads((output/"parametric-stage.json").read_text(encoding="utf8"))
    assert stage["publication"]["status"]=="rolled_back" and not stage["accepted"]


def supported_graph(image,baseline):
    entities=deepcopy(baseline["entities"])
    for entity in entities:
        for key in ("start","end","center"):
            if key in entity:entity[key][0]+=.5
    scale=baseline["scale"]["pixels_per_mm"]
    origin=baseline["coordinate_system"]["origin_source_px"]
    nodes=[]
    for index,entity in enumerate(entities):
        x,y=entity["start"]
        nodes.append({"id":f"v{index:03d}","x":x,"y":y,"source_px":[x*scale+origin[0],origin[1]-y*scale]})
        entity.update(start_node=f"v{index:03d}",end_node=f"v{(index+1)%len(entities):03d}")
    return {"units":"mm","entities":entities,"nodes":nodes,"coordinate_system":deepcopy(baseline["coordinate_system"]),
            "validation":dict.fromkeys(("closed","connected","simple","ordered_entity_cycle"),True),
            "source_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),"ground_truth_used":False,"source_grid_pitch_px":2.,
            "source_evidence":{"baseline_stroke_support":{"edge_supported_fraction":.8,"p90_edge_distance_px":2.},
                               "proposal_stroke_support":{"edge_supported_fraction":.95,"p90_edge_distance_px":1.},"local_corrections":1}}


@pytest.mark.parametrize("failure",["no_constraints","binding_failure","solver_failure"])
def test_supported_topology_survives_missing_constraints_or_later_failure(tmp_path,monkeypatch,failure):
    image,baseline,output=prepared(tmp_path)
    before={name:(output/name).read_bytes() for name in CORE}
    graph=supported_graph(image,baseline)
    monkeypatch.setattr("contour_agent.topology.build_topology",lambda *a,**k:graph)
    def binding(*args,**kwargs):
        if failure=="binding_failure":raise OSError("secret must not reach report")
        return {"constraints":[],"provider":{"status":"failed","network_requests":1,"error_code":"timeout"},"counts":{},"issues":[]}
    def solve(*args,**kwargs):
        if failure=="solver_failure":raise ValueError("private detail must not reach report")
        return {"accepted":False,"status":"no_constraints","entities":graph["entities"]}
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",binding)
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",solve)
    model,stage=refine_parametric(image,{},baseline,output,use_api=True)
    assert not stage["accepted"] and stage["topology_exported"]
    assert model["completion_class"]=="source_topology_draft"
    assert model["entities"]==graph["entities"] and model["entities"]!=baseline["entities"]
    assert model["validation"]["constraint_validation"]["status"]=="not_run"
    assert model["validation"]["source_topology_validation"]["passed"]
    assert not model["validation"]["dimensions_verified"]
    assert {name:(output/("baseline-"+name)).read_bytes() for name in CORE}==before
    stored=json.loads((output/"model.json").read_text(encoding="utf8"))
    assert stored["algorithm_version"]=="source-topology-draft-v1" and stored["parameterization"]["topology_exported"]
    assert "must not reach report" not in (output/"parametric-stage.json").read_text(encoding="utf8")


def test_unsupported_source_topology_is_not_promoted(tmp_path,monkeypatch):
    image,baseline,output=prepared(tmp_path)
    graph=supported_graph(image,baseline)
    graph["source_evidence"]["proposal_stroke_support"]["edge_supported_fraction"]=.1
    monkeypatch.setattr("contour_agent.topology.build_topology",lambda *a,**k:graph)
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",lambda *a,**k:{"constraints":[],"provider":{"status":"disabled"}})
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",lambda *a,**k:{"accepted":False,"status":"no_constraints"})
    model,stage=refine_parametric(image,{},baseline,output)
    assert model==baseline and not stage.get("topology_exported",False)
    assert stage["topology_source_validation"]["reasons"]==["source_stroke_support_degraded"]


def test_cancel_second_publication_rolls_back_to_corrected_topology_not_mask(tmp_path,monkeypatch):
    image,baseline,output=prepared(tmp_path)
    graph=supported_graph(image,baseline)
    monkeypatch.setattr("contour_agent.topology.build_topology",lambda *a,**k:graph)
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",lambda *a,**k:{"constraints":[],"provider":{"status":"disabled"}})
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",lambda *a,**k:{
        "accepted":True,"entities":graph["entities"],"validation":{"passed":True}})
    def cancel(stage,message):
        if stage=="parametric_export":raise InterruptedError()
    with pytest.raises(InterruptedError):refine_parametric(image,{},baseline,output,progress=cancel)
    model=json.loads((output/"model.json").read_text(encoding="utf8"))
    assert model["completion_class"]=="source_topology_draft" and model["entities"]==graph["entities"]
    stage=json.loads((output/"parametric-stage.json").read_text(encoding="utf8"))
    assert stage["topology_exported"] and not stage["accepted"] and stage["publication"]["status"]=="rolled_back"


def test_restart_partial_second_publication_restores_last_valid_topology(tmp_path,monkeypatch):
    image,baseline,output=prepared(tmp_path)
    graph=supported_graph(image,baseline)
    monkeypatch.setattr("contour_agent.topology.build_topology",lambda *a,**k:graph)
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",lambda *a,**k:{"constraints":[],"provider":{"status":"disabled"}})
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",lambda *a,**k:{"accepted":False,"status":"no_constraints"})
    topology,_=refine_parametric(image,{},baseline,output)
    for name in CORE:shutil.copyfile(output/name,output/("last-valid-"+name))
    job=seed_pending_job(tmp_path,topology,output)
    stage=job["parameterization"]
    stage.update(topology_exported=True,publication={"status":"pending","rollback_prefix":"last-valid-",
        "candidate_sha256":dict.fromkeys(CORE,"not-the-current-set"),
        "rollback_sha256":{name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in CORE},
        "baseline_sha256":{name:hashlib.sha256((output/("baseline-"+name)).read_bytes()).hexdigest() for name in CORE}})
    (output/"parametric-stage.json").write_text(json.dumps(stage))
    JobStore(tmp_path/"runtime").save(job)
    (output/"drawing.dxf").write_bytes(b"interrupted write")
    service=recover(tmp_path,monkeypatch)
    try:
        recovered=service.store.get("persisted")
        assert recovered["status"]=="completed" and recovered["automatic_completion"]
        assert recovered["completion_class"]=="source_topology_draft"
        assert recovered["geometry"]["entities"]==graph["entities"]
        assert recovered["parameterization"]["topology_exported"] and not recovered["parameterization"]["accepted"]
    finally:service.close()


def test_source_topology_coordinate_mapping_control_is_consistent(tmp_path):
    from contour_agent.parametric_pipeline import _topology_source_validation
    image,baseline=fixture_model(tmp_path)
    graph=supported_graph(image,baseline)
    assert _topology_source_validation(image,baseline,graph)["passed"]


@pytest.mark.parametrize("mismatch",["origin","pixel_scale","source_node","axis_direction","coordinate_units"])
def test_source_topology_rejects_stale_or_inconsistent_coordinate_mapping(tmp_path,mismatch):
    from contour_agent.parametric_pipeline import _topology_source_validation
    image,baseline=fixture_model(tmp_path)
    graph=supported_graph(image,baseline)
    if mismatch=="origin":
        graph["coordinate_system"]["origin_source_px"][0]-=50.
        for node in graph["nodes"]:node["source_px"][0]-=50.
    elif mismatch=="pixel_scale":
        for node in graph["nodes"]:node["source_px"]=[100+6.*node["x"],100-6.*node["y"]]
    elif mismatch=="source_node":graph["nodes"][0]["source_px"][0]+=1.
    elif mismatch=="axis_direction":
        for node in graph["nodes"]:node["source_px"][1]=100+5.*node["y"]
    elif mismatch=="coordinate_units":graph["coordinate_system"]["units"]="pixel"
    validation=_topology_source_validation(image,baseline,graph)
    assert not validation["passed"], "Source hash/stroke scores cannot validate a mismatched image-to-CAD transform"


def test_bad_coordinate_mapping_is_rejected_before_api_or_solver(tmp_path,monkeypatch):
    image,baseline,output=prepared(tmp_path)
    before={name:(output/name).read_bytes() for name in CORE}
    graph=supported_graph(image,baseline)
    graph["coordinate_system"]["origin_source_px"][0]-=50.
    for node in graph["nodes"]:node["source_px"][0]-=50.
    monkeypatch.setattr("contour_agent.topology.build_topology",lambda *a,**k:graph)
    calls=[]
    def binding(*args,**kwargs):
        calls.append("binding")
        return {"constraints":[],"provider":{"status":"disabled"}}
    def solver(*args,**kwargs):
        calls.append("solver")
        return {"accepted":True,"entities":graph["entities"],"validation":{"passed":True}}
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",binding)
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",solver)
    model,stage=refine_parametric(image,{},baseline,output,use_api=True)
    assert calls==[], "Do not send or solve geometry in an unverified source coordinate frame"
    assert model==baseline and not stage["accepted"] and not stage.get("topology_exported",False)
    assert {name:(output/name).read_bytes() for name in CORE}==before


@pytest.mark.parametrize("selection,schema,solved,expected_geometry,expected_dimensions",[
    ({"relation_id":"rel000","source":"source_geometry","accepted":True},True,True,True,False),
    ({"candidate_id":"c000","source":"ocr_api_binding","accepted":True},True,True,True,True),
    ({"relation_id":"rel000","source":"source_geometry","accepted":False},True,True,False,False),
    ({"relation_id":"rel000","source":"source_geometry","accepted":True},False,True,False,False),
    ({"relation_id":"rel000","source":"source_geometry","accepted":True},True,False,False,False),
    ({"candidate_id":"c000","source":"ocr_local_binding","accepted":True},True,True,False,False),
])
def test_api_geometry_and_dimension_update_flags_require_accepted_published_choices(
    tmp_path,monkeypatch,selection,schema,solved,expected_geometry,expected_dimensions
):
    image,baseline,output=prepared(tmp_path)
    graph=supported_graph(image,baseline)
    monkeypatch.setattr("contour_agent.topology.build_topology",lambda *a,**k:graph)
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",lambda *a,**k:{
        "constraints":[],"bindings":[selection],"provider":{"schema_success":schema,"network_requests":1}})
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",lambda *a,**k:{
        "accepted":solved,"status":"accepted" if solved else "no_constraints","entities":graph["entities"],"validation":{"passed":solved}})
    model,stage=refine_parametric(image,{},baseline,output,use_api=True)
    assert stage["geometry_updated_by_api"] is expected_geometry
    assert stage["dimensions_updated_by_api"] is expected_dimensions
    assert model["parameterization"]["geometry_updated_by_api"] is expected_geometry
    assert model["parameterization"]["dimensions_updated_by_api"] is expected_dimensions
    saved=json.loads((output/"model.json").read_text(encoding="utf8"))
    assert saved["parameterization"]["geometry_updated_by_api"] is expected_geometry
    assert saved["parameterization"]["dimensions_updated_by_api"] is expected_dimensions

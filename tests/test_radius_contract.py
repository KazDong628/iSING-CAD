from copy import deepcopy

import ezdxf
import pytest

from contour_agent.radius_contract import annotation_radius_contract, exact_radius_checks
from test_parametric_pipeline import fixture_model, prepared, supported_graph
from contour_agent.parametric_pipeline import export_parametric, refine_parametric
import json


def equation(value=5.):
    return {"kind":"radius","record_id":"r001","entities":["g002"],"value":value}


@pytest.mark.parametrize("radius",[5.000001,4.999999,True])
def test_exact_radius_audit_cannot_accept_soft_tolerance_or_boolean(tmp_path,radius):
    _,model=fixture_model(tmp_path)
    model["entities"][2]["radius"]=radius
    audit=exact_radius_checks(model["entities"],[equation()])
    assert not audit["passed"] and audit["checks"][0]["tolerance"]==0


def test_dxf_readback_independently_checks_radius(tmp_path):
    _,model=fixture_model(tmp_path)
    doc=ezdxf.new()
    for row in model["entities"]:
        if row["type"]=="ARC":doc.modelspace().add_arc((0,0),5.000001,0,180)
        else:doc.modelspace().add_line(row["start"],row["end"])
    audit=exact_radius_checks(model["entities"],[equation()],dxf_document=doc)
    assert not audit["passed"] and audit["dxf_readback_performed"]


def test_export_does_not_trust_accepted_flag_over_wrong_radius(tmp_path):
    image,model=fixture_model(tmp_path)
    with pytest.raises(ValueError,match="exactly"):
        export_parametric(image,model,{"accepted":True,"entities":model["entities"],
                                      "constraints":[equation(5.000001)]},tmp_path/"bad")


def test_exact_export_records_source_value_and_native_arc_readback(tmp_path):
    image,model=fixture_model(tmp_path)
    result=export_parametric(image,model,{"accepted":True,"entities":model["entities"],
                                        "constraints":[equation()]},tmp_path/"good")
    audit=result["validation"]["exact_radius_validation"]
    assert audit["passed"] and audit["checks"][0]["dxf_radius"]==5.
    assert not result["validation"]["dimensions_verified"]


@pytest.mark.parametrize("defect",["unknown","unbound","ambiguous","missing_equation"])
def test_partial_or_inconsistent_coverage_never_satisfies_full_contract(tmp_path,defect):
    _,model=fixture_model(tmp_path)
    coverage={"all_radius_records_resolved":True,"all_confirmed_arrows_bound":True,
              "bound_mappings":[{"record_id":"r001","entity_id":"g002","nominal":5.}]}
    bindings={"radius_binding_coverage":coverage,"constraints":[equation()]}
    if defect=="unknown":coverage["unknown_arrow_records"]=["r002"]
    if defect=="unbound":coverage["all_confirmed_arrows_bound"]=False
    if defect=="ambiguous":coverage["ambiguous"]=[{"record_id":"r002"}]
    if defect=="missing_equation":bindings["constraints"]=[]
    contract=annotation_radius_contract(bindings,{"accepted":True,"entities":model["entities"]})
    assert not contract["satisfied"] and contract["reasons"]


def test_complete_radius_contract_does_not_certify_other_dimensions(tmp_path):
    _,model=fixture_model(tmp_path)
    coverage={"all_radius_records_resolved":True,"all_confirmed_arrows_bound":True,
              "bound_mappings":[{"record_id":"r001","entity_id":"g002","nominal":5.}]}
    contract=annotation_radius_contract({"radius_binding_coverage":coverage,"constraints":[equation()]},
                                       {"accepted":True,"entities":model["entities"]})
    assert contract["satisfied"] and not contract["reference_verified"]
    assert not contract["missing_arrow_detection_is_exemption"]


@pytest.mark.parametrize("complete",[True,False])
def test_pipeline_publishes_exact_subset_without_claiming_unresolved_radii_complete(tmp_path,monkeypatch,complete):
    image,baseline,output=prepared(tmp_path)
    graph=supported_graph(image,baseline)
    monkeypatch.setattr("contour_agent.topology.build_topology",lambda *a,**k:graph)
    monkeypatch.setattr("contour_agent.parametric_pipeline._plan_topology",lambda *a,**k:(graph,{
        "selected_candidate_id":"source","selection_source":"local_evaluator","local_evaluation":{}}))
    monkeypatch.setattr("contour_agent.parametric_pipeline._solved_source_validation",lambda *a,**k:{"passed":True})
    coverage={"all_radius_records_resolved":complete,"all_confirmed_arrows_bound":True,
              "unknown_arrow_records":[] if complete else ["r002"],
              "bound_mappings":[{"record_id":"r001","entity_id":"g002","nominal":5.}]}
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",lambda *a,**k:{
        "constraints":[equation()],"radius_binding_coverage":coverage,"provider":{"status":"disabled","network_requests":0}})
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",lambda *a,**k:{
        "accepted":True,"status":"accepted","entities":graph["entities"],"constraints":[equation()],"validation":{"passed":True}})
    model,stage=refine_parametric(image,{},baseline,output)
    assert stage["constraint_subset_accepted"] and stage["accepted"] is False
    assert stage["annotation_radius_contract"]["satisfied"] is complete
    assert stage["status"]==("completed_with_unresolved_attributes" if complete else "completed_with_unresolved_radii")
    assert (output/"drawing.dxf").is_file() and not model["validation"]["dimensions_verified"]
    contract=json.loads((output/"radius-contract.json").read_text(encoding="utf8"))
    assert contract["exact_radius_validation"]["dxf_readback_performed"]
    assert contract["exact_radius_validation"]["checks"][0]["dxf_radius"]==5
    provenance=json.loads((output/"workflow-provenance.json").read_text(encoding="utf8"))
    assert provenance["solver_executed"] and provenance["published_artifact_kind"]=="parametric"
    assert not provenance["reference_dxf_used_as_prediction_geometry"]
    saved=json.loads((output/"model.json").read_text(encoding="utf8"))
    assert saved["parameterization"]["accepted"] is False
    assert saved["validation"]["reconstruction_contract"]["satisfied"] is False

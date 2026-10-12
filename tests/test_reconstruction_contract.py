from copy import deepcopy

import pytest

from contour_agent.reconstruction_contract import reconstruction_contract
from contour_agent.parametric_pipeline import export_parametric
from test_parametric_pipeline import fixture_model


def rectangle():
    points = [[0., 0.], [10., 0.], [10., 5.], [0., 5.]]
    entities = [dict(id=f"g{i}", type="LINE", start=p, end=points[(i+1)%4],
                     start_node=f"v{i}", end_node=f"v{(i+1)%4}") for i,p in enumerate(points)]
    stage = {"constraints": [dict(id=f"k{i}", kind="horizontal" if i%2 == 0 else "vertical",
                                  entities=[f"g{i}"], nodes=[]) for i in range(4)],
             "binding_counts": dict(recognized_dimensions=2, bound_source_records=2, unbound_dimensions=0),
             "solver": dict(accepted=True,diagnostics=dict(remaining_shape_dof=0),
                            validation=dict(constraint_subset_satisfied=True)), "underconstrained": False}
    stage["solver_constraint_checks"] = [{**row,"passed":True} for row in stage["constraints"]]
    validation = dict(passed=True, strict_relation_validation=dict(passed=True, dxf_readback_performed=True),
                      annotation_radius_contract=dict(satisfied=True))
    return entities, stage, validation


def test_complete_sourced_rectangle_never_claims_gt_or_complete_ocr_detection():
    result = reconstruction_contract(*rectangle())
    assert result["satisfied"] and result["all_join_relationships_certified"]
    assert not result["reference_object_count_verified"]
    assert not result["complete_source_annotation_detection_verified"]


@pytest.mark.parametrize("defect", ["unbound", "unknown_count", "shape_dof", "unknown_dof", "native_failed", "native_missing", "numerical_failed", "radius_failed", "geometry_failed", "solver_rejected", "direction_failed"])
def test_no_radius_only_or_stale_subset_upgrade(defect):
    entities,stage,validation = rectangle()
    if defect == "unbound": stage["binding_counts"].update(bound_source_records=1, unbound_dimensions=1)
    if defect == "unknown_count": stage["binding_counts"] = {}
    if defect == "shape_dof": stage["solver"]["diagnostics"]["remaining_shape_dof"] = 1
    if defect == "unknown_dof": stage["solver"]["diagnostics"] = {}
    if defect == "native_failed": validation["strict_relation_validation"]["passed"] = False
    if defect == "native_missing": validation["strict_relation_validation"] = {}
    if defect == "numerical_failed": stage["solver"]["validation"]["constraint_subset_satisfied"] = False
    if defect == "radius_failed": validation["annotation_radius_contract"]["satisfied"] = False
    if defect == "geometry_failed": validation["passed"] = False
    if defect == "solver_rejected": stage["solver"]["accepted"] = False
    if defect == "direction_failed": stage["solver_constraint_checks"][0]["passed"] = False
    result = reconstruction_contract(entities,stage,validation)
    assert not result["satisfied"] and result["reasons"]


def test_unadmitted_arc_arc_join_is_visible_without_invented_tangency():
    entities,stage,validation = rectangle()
    entities[1]["type"] = entities[2]["type"] = "ARC"
    result = reconstruction_contract(entities,stage,validation)
    assert not result["satisfied"]
    assert result["unresolved_joint_count"] == 3
    assert any(j["types"] == ["ARC","ARC"] and j["relationship"] == "unresolved" for j in result["joints"])


def test_export_rejects_stale_accepted_flag_for_non_tangent_join(tmp_path):
    image,model = fixture_model(tmp_path)
    # Left vertical -> top horizontal is a genuine corner, not a tangent.
    solution = dict(accepted=True, entities=deepcopy(model["entities"]),
                    constraints=[dict(id="wrong",kind="tangent",entities=["g000","g001"],nodes=["v001"])])
    with pytest.raises(ValueError, match="strictly satisfied"):
        export_parametric(image,model,solution,tmp_path/"wrong")
    assert not (tmp_path/"wrong"/"drawing.dxf").exists()


def test_export_certifies_native_tangencies_and_preserves_partial_coverage(tmp_path):
    image,model = fixture_model(tmp_path)
    solution = dict(accepted=True, entities=model["entities"],
                    constraints=[dict(id="good",kind="tangent",entities=["g001","g002"],nodes=["v002"])])
    result = export_parametric(image,model,solution,tmp_path/"good")
    audit = result["validation"]["strict_relation_validation"]
    assert audit["passed"] and audit["dxf_readback_performed"] and audit["required_count"] == 1
    assert not audit["complete_relation_coverage_verified"]

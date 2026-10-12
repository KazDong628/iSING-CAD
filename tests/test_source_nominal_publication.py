"""Known source values cannot regain an older approximate certificate."""
from copy import deepcopy

import pytest

from contour_agent.constraint_binding import _constraint
from contour_agent.parametric_pipeline import _constraint_receipts_complete, export_parametric
from contour_agent.service import _recover_numerical_constraints
from contour_agent.relation_contract import relation_checks
from test_measurement_anchor_binding import native_document
from test_parametric_solver import rectangle
from test_parametric_pipeline import fixture_model


@pytest.mark.parametrize("kind", ["distance_x", "distance_y", "distance", "angle"])
def test_admitted_source_attribute_explicitly_requests_exact_nominal(kind):
    row = _constraint({"id": "c0", "kind": kind, "record_id": "r0", "entities": [],
                       "nodes": ["v0", "v1"], "value": 10.}, "ocr_api_binding")
    assert row["required"] is True
    assert row["enforcement"] == "exact" and row["nominal_source"] == "source_ocr"


def test_approximate_receipt_cannot_substitute_for_exact_source_measurement():
    graph = rectangle(width=10., height=5.)
    required = {"id": "k0", "kind": "distance_x", "entities": [], "nodes": ["v0", "v1"],
                "record_id": "r0", "value": 10., "source": "ocr_api_binding", "required": True,
                "enforcement": "exact", "nominal_source": "source_ocr"}
    approximate = {k: v for k, v in required.items() if k not in {"required", "enforcement", "nominal_source"}}
    approximate.update(passed=True, actual=10.)
    assert not _constraint_receipts_complete({"constraints": [required]}, {"constraints": [approximate]}, graph)
    exact = {**required, "enforcement": "source_nominal_equality", "passed": True}
    assert _constraint_receipts_complete({"constraints": [required]}, {"constraints": [exact]}, graph)


@pytest.mark.parametrize("kind,value,nodes", [
    ("distance_x", 10.00001, ["v0", "v1"]),
    ("angle", 90.00001, ["v0", "v1", "v2"]),
])
def test_recovery_recomputes_strict_source_values_despite_forged_cached_success(kind, value, nodes):
    graph = rectangle(width=10., height=5.)
    row = {"id": "k0", "kind": kind, "nodes": nodes, "entities": [], "record_id": "r0",
           "value": value, "source": "ocr_api_binding", "required": True, "enforcement": "exact",
           "nominal_source": "source_ocr", "passed": True, "actual": value}
    model = {"entities": graph["entities"], "coordinate_system": {"units": "mm"},
             "parameterization": {"constraints": [row]}}
    native = native_document(graph)
    strict = relation_checks(graph["entities"], [row], dxf_document=native)
    check = _recover_numerical_constraints(model, native, strict)
    assert check["dxf_readback_performed"] and check["solver_called"] is False
    assert not check["passed"] and check["checks"][0]["tolerance"] == 1e-7


def test_export_blocks_accepted_flag_with_inexact_source_node_length(tmp_path):
    source, model = fixture_model(tmp_path)
    row = {"id": "height", "kind": "distance_y", "nodes": ["v000", "v001"], "entities": [],
           "value": 10.00001, "record_id": "r0", "source": "ocr_api_binding", "required": True,
           "enforcement": "exact", "nominal_source": "source_ocr", "passed": True}
    solution = {"accepted": True, "units": "mm", "entities": deepcopy(model["entities"]),
                "constraints": [row], "validation": {"constraint_subset_satisfied": True}}
    with pytest.raises(ValueError, match="linear measurements"):
        export_parametric(source, model, solution, tmp_path/"forged")

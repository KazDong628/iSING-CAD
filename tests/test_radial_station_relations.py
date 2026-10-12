"""A common drawing axis is eliminated algebraically, never fitted and locked."""
from copy import deepcopy

import pytest

from contour_agent.measurement_relations import radial_station_relation, radial_station_value
from contour_agent.parametric_solver import solve_parametric, native_measurement_checks
from test_parametric_solver import rounded_graph, dimensions_and_relations
from test_measurement_anchor_binding import native_document


def relation(kind="diameter", direction=1):
    return {"type": "half_section_common_axis_eliminated", "axis": "x", "direction": direction,
        "primary_kind": kind, "reference_record_id": "r_reference", "reference_dimension": {
            "kind": "diameter", "nominal": 14., "lower_deviation": None, "upper_deviation": None}}


def constraint():
    return {"id": "radial_width", "kind": "distance_x", "record_id": "r_primary", "value": 54.,
        "source": "ocr_local_binding", "nodes": [], "entities": ["g002", "g006"],
        "anchors": [{"type": "support_level", "entity_id": "g006", "axis": "x"},
                    {"type": "support_level", "entity_id": "g002", "axis": "x"}],
        "source_station_relation": relation()}


def test_signed_difference_preserves_both_source_values_without_absolute_value_shortcut():
    assert radial_station_value(20., relation()) == 54.
    assert radial_station_value(-20., relation(direction=-1)) == 54.
    assert radial_station_value(-20., relation()) == -26.
    assert radial_station_value(20., relation(kind="length")) == 27.
    audit = radial_station_relation(relation())
    assert not audit["absolute_axis_position_verified"]
    assert not audit["reference_diameter_independently_solved"]


def test_reference_tolerance_midpoint_is_declared_annotation_policy():
    meta = relation()
    meta["reference_dimension"].update(nominal=186., lower_deviation=-2., upper_deviation=0.)
    assert radial_station_value(40.5, meta) == 266.
    audit = radial_station_relation(meta)
    assert audit["reference_target"] == 185.
    assert audit["reference_tolerance"]["minimum"] == 184.
    assert audit["reference_tolerance"]["maximum"] == 186.


@pytest.mark.parametrize("change", [
    {"axis_origin": 42.}, {"direction": True}, {"direction": 0}, {"primary_kind": "radius"},
    {"axis": "z"}, {"reference_record_id": "../secret"}, {"type": "fitted_axis"},
])
def test_literal_or_untyped_coordinate_construction_is_rejected(change):
    with pytest.raises(ValueError):
        radial_station_relation({**relation(), **change})


def test_joint_solve_uses_relative_diameter_equation_with_exact_radii_and_tangency():
    graph = rounded_graph(); before = deepcopy(graph)
    constraints = dimensions_and_relations(); constraints[0] = constraint()
    result = solve_parametric(graph, constraints)
    assert result["accepted"], result
    assert graph == before
    assert result["strict_radius_contract"]["satisfied"]
    assert result["strict_tangent_contract"]["satisfied"]
    assert result["strict_measurement_contract"]["satisfied"]
    assert result["diagnostics"]["remaining_shape_dof"] == 0
    indexed = {row["id"]: row for row in result["entities"]}
    assert indexed["g002"]["start"][0]-indexed["g006"]["start"][0] == pytest.approx(20., abs=1e-7)
    measured = next(row for row in result["constraints"] if row["id"] == "radial_width")
    assert measured["actual"] == pytest.approx(54., abs=1e-7)
    # Native readback evaluates the affine source equation again, not a cached
    # diameter result or a hidden absolute datum coordinate.
    check = native_measurement_checks(result["entities"], [constraint()],
                                      dxf_document=native_document(result))
    assert check["passed"]
    translated = deepcopy(result)
    for entity in translated["entities"]:
        for key in ("start", "end", "center"):
            if key in entity:
                entity[key][0] += 100.
    check = native_measurement_checks(translated["entities"], [constraint()],
                                      dxf_document=native_document(translated))
    assert check["passed"]


def test_same_ocr_cannot_be_used_as_its_own_eliminated_datum():
    row = constraint(); row["source_station_relation"]["reference_record_id"] = row["record_id"]
    with pytest.raises(ValueError, match="distinct OCR"):
        solve_parametric(rounded_graph(), [row])


def test_affine_relation_equivalent_to_plain_distance_is_not_double_counted():
    row = constraint()
    duplicate = {**deepcopy(row), "id": "plain", "record_id": "r_plain", "value": 20.}
    duplicate.pop("source_station_relation")
    with pytest.raises(ValueError, match="Duplicate equivalent"):
        solve_parametric(rounded_graph(), [row, duplicate])


def binding_inventory():
    row = constraint()
    records = [{"id": record_id, "text": "Ø"+str(nominal), "box": [[0, 0], [2, 0], [2, 2], [0, 2]],
                "parsed": {"kind": "diameter", "nominal": nominal, "lower_deviation": None, "upper_deviation": None}}
               for record_id, nominal in (("r_primary", 54.), ("r_reference", 14.))]
    candidate = {key: deepcopy(row[key]) for key in ("kind", "record_id", "value", "nodes", "entities", "anchors", "source_station_relation")}
    candidate.update(id="c1", local_reliable=True, evidence={}, source_dimension_semantics=records[0]["parsed"])
    return {"units": "mm", "all_records": records, "records": records, "all_candidates": [candidate],
        "candidates": [candidate], "relations": [], "counts": {"recognized_dimensions": 2},
        "artifacts": {"inventory": "inventory.json", "topology": "topology.png"}}


def test_relative_binding_keeps_reference_diameter_dependency_in_unresolved_denominator(monkeypatch, tmp_path):
    from contour_agent import constraint_binding
    inventory = binding_inventory()
    monkeypatch.setattr(constraint_binding, "build_binding_candidates", lambda *args, **kwargs: deepcopy(inventory))
    result = constraint_binding.analyze_constraint_bindings("unused.png", {"records": []}, {}, rounded_graph(), tmp_path)
    assert len(result["constraints"]) == 1
    assert result["counts"]["bound_source_records"] == 1
    assert result["counts"]["unbound_dimensions"] == 1
    assert result["counts"]["relative_station_bound_records"] == 1
    assert result["counts"]["unsolved_reference_diameter_records"] == ["r_reference"]
    assert result["constraints"][0]["value"] == 54.
    assert result["constraints"][0]["source_station_relation"] == relation()


@pytest.mark.parametrize("issue", ["changed_reference", "withheld_reference"])
def test_stale_or_withheld_reference_nominal_cannot_enter_solver(monkeypatch, tmp_path, issue):
    from contour_agent import constraint_binding
    inventory = binding_inventory()
    if issue == "changed_reference":
        inventory["all_records"][1]["parsed"]["nominal"] = 18.
    else:
        inventory["all_records"][1]["source_text_admission"] = {"status": "withheld"}
    monkeypatch.setattr(constraint_binding, "build_binding_candidates", lambda *args, **kwargs: deepcopy(inventory))
    result = constraint_binding.analyze_constraint_bindings("unused.png", {"records": []}, {}, rounded_graph(), tmp_path)
    assert not result["constraints"]
    assert result["counts"]["bound_source_records"] == 0
    assert result["counts"]["unbound_dimensions"] == 2


def test_provider_sees_declared_source_algebra_but_not_injected_coordinates():
    import json
    from contour_agent.binding_provider import bounded_inventory
    inventory = binding_inventory()
    inventory["candidates"][0]["source_station_relation"]["fit_axis_px"] = 999.
    packet = bounded_inventory(inventory)
    assert packet["candidates"][0]["source_station_relation"] == relation()
    assert "fit_axis_px" not in json.dumps(packet)

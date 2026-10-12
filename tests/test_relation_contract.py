"""Strict tangency must hold in native geometry, independent of old receipts."""
import copy
import math

import ezdxf
import pytest

from contour_agent.relation_contract import relation_checks, STRICT_TANGENT_CERT_TOLERANCE_DEG


def _line(entity_id, start, end, start_node, end_node):
    return dict(id=entity_id, type="LINE", start=start, end=end,
                start_node=start_node, end_node=end_node)


def _arc(entity_id, start, end, center, radius, clockwise, start_node, end_node):
    return dict(id=entity_id, type="ARC", start=start, end=end, center=center,
                radius=radius, clockwise=clockwise, start_node=start_node, end_node=end_node)


def _tangent(cid, first, second, node):
    return dict(id=cid, kind="tangent", entities=[first, second], nodes=[node],
                value=None, passed=True, residual=0.)


def _fillet():
    entities = [_line("a", [-1., 0.], [0., 0.], "v0", "v1"),
                _arc("b", [0., 0.], [1., 1.], [0., 1.], 1., False, "v1", "v2"),
                _line("c", [1., 1.], [1., 2.], "v2", "v3")]
    return entities, [_tangent("k1", "a", "b", "v1"), _tangent("k2", "b", "c", "v2")]


def _native(entities):
    document = ezdxf.new("R2010")
    document.units = 4
    model = document.modelspace()
    for row in entities:
        if row["type"] == "LINE":
            model.add_line(row["start"], row["end"])
        else:
            center = row["center"]
            a, b = [math.degrees(math.atan2(row[key][1] - center[1], row[key][0] - center[0])) % 360
                    for key in ("start", "end")]
            model.add_arc(center, row["radius"], b if row["clockwise"] else a,
                          a if row["clockwise"] else b)
    return document


def test_line_arc_line_readback_recomputes_without_mutating_or_trusting_flags():
    entities, constraints = _fillet()
    for row in constraints:
        row["passed"] = False
    previous = copy.deepcopy((entities, constraints))
    result = relation_checks(entities, constraints, dxf_document=_native(entities))
    assert result["passed"] and result["required_count"] == result["satisfied_count"] == 2
    assert result["dxf_readback_performed"] and result["native_mapping_verified"]
    assert result["complete_relation_coverage_verified"] is False
    assert previous == (entities, constraints)
    assert all(row["dxf"]["angle_residual_deg"] <= STRICT_TANGENT_CERT_TOLERANCE_DEG
               for row in result["checks"])


def test_clockwise_native_arc_swapped_endpoints_are_restored():
    entities, constraints = _fillet()
    entities.reverse()
    for row in entities:
        row["start"], row["end"] = row["end"], row["start"]
        row["start_node"], row["end_node"] = row["end_node"], row["start_node"]
        if row["type"] == "ARC":
            row["clockwise"] = True
    result = relation_checks(entities, constraints, dxf_document=_native(entities))
    assert result["passed"]
    assert result["required_count"] == 2


def test_saved_and_reread_native_dxf_is_the_certified_artifact(tmp_path):
    entities, constraints = _fillet()
    destination = tmp_path / "drawing.dxf"
    _native(entities).saveas(destination)
    assert relation_checks(entities, constraints, dxf_document=ezdxf.readfile(destination))["passed"]


def test_arc_arc_join_is_audited_including_opposite_curvature():
    entities, _ = _fillet()
    entities = [entities[1], _arc("d", [1., 1.], [2., 2.], [2., 1.], 1., True, "v2", "v3")]
    result = relation_checks(entities, [_tangent("arc-arc", "b", "d", "v2")], dxf_document=_native(entities))
    assert result["passed"] and result["required_count"] == 1
    assert result["checks"][0]["model"]["forward_dot"] == pytest.approx(1.)


def test_stale_solver_pass_cannot_hide_point_one_degree_tangent_error():
    entities, constraints = _fillet()
    entities[0]["start"][1] = -math.tan(math.radians(.099999))
    result = relation_checks(entities, constraints, dxf_document=_native(entities))
    assert not result["passed"]
    assert result["checks"][0]["model"]["angle_residual_deg"] == pytest.approx(.099999)
    assert result["checks"][0]["dxf"]["passed"] is False


def test_antiparallel_directions_are_180_degrees_not_zero():
    entities, constraints = _fillet()
    entities[0]["start"] = [1., 0.]
    result = relation_checks(entities, constraints)
    check = result["checks"][0]
    assert not result["passed"]
    assert check["model"]["angle_residual_deg"] == 180.
    assert check["model"]["forward_dot"] == -1.


def test_endpoint_gap_fails_even_for_parallel_tangents():
    entities, constraints = _fillet()
    entities[0]["start"][1] = entities[0]["end"][1] = 1e-5
    result = relation_checks(entities, constraints)
    assert not result["passed"]
    assert result["checks"][0]["model"]["angle_residual_deg"] == 0.
    assert result["checks"][0]["model"]["endpoint_gap"] == pytest.approx(1e-5)


@pytest.mark.parametrize("perturbation", ["line_endpoint", "arc_center", "arc_radius", "arc_angle", "arc_plane"])
def test_perturbed_native_geometry_fails_mapping(perturbation):
    entities, constraints = _fillet()
    document = _native(entities)
    line, arc = list(document.modelspace())[:2]
    if perturbation == "line_endpoint":
        line.dxf.end = (0., .01)
    elif perturbation == "arc_center":
        arc.dxf.center = (.01, 1.)
    elif perturbation == "arc_radius":
        arc.dxf.radius = 1.01
    elif perturbation == "arc_angle":
        arc.dxf.start_angle += 1.
    else:
        arc.dxf.extrusion = (0., 0., -1.)
    result = relation_checks(entities, constraints, dxf_document=document)
    assert not result["passed"] and not result["native_mapping_verified"]
    assert result["issues"]


def test_sub_mapping_tolerance_native_change_still_requires_strict_tangent():
    entities, constraints = _fillet()
    entities[0]["start"] = [-1e-4, 0.]
    document = _native(entities)
    list(document.modelspace())[0].dxf.start = (-1e-4, -1e-9)
    result = relation_checks(entities, constraints, dxf_document=document)
    assert result["native_mapping_verified"]
    assert result["checks"][0]["model"]["passed"] is True
    assert result["checks"][0]["dxf"]["passed"] is False
    assert result["passed"] is False


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_type", "reordered", "broken_document"])
def test_missing_or_mismatched_native_entity_list_fails_closed(mutation):
    entities, constraints = _fillet()
    document = _native(entities)
    model = document.modelspace()
    if mutation == "missing":
        model.delete_entity(list(model)[-1])
    elif mutation == "extra":
        model.add_line((4., 0.), (5., 0.))
    elif mutation == "wrong_type":
        document = ezdxf.new()
        for _ in entities:
            document.modelspace().add_line((0., 0.), (1., 0.))
    elif mutation == "reordered":
        document = _native([entities[2], entities[1], entities[0]])
    else:
        document = object()
    result = relation_checks(entities, constraints, dxf_document=document)
    assert not result["passed"]
    assert not result["native_mapping_verified"]


@pytest.mark.parametrize("mutation", ["duplicate_id", "unknown_id", "empty_id", "bad_nodes", "duplicate_constraint",
                                     "duplicate_relation", "same_entity", "nonzero_nominal", "nan_nominal"])
def test_malformed_constraint_and_conflicting_incidence_fail_closed(mutation):
    entities, constraints = _fillet()
    if mutation == "duplicate_id":
        entities[2]["id"] = "a"
    elif mutation == "unknown_id":
        constraints[0]["entities"][0] = "missing"
    elif mutation == "empty_id":
        constraints[0]["id"] = ""
    elif mutation == "bad_nodes":
        constraints[0]["nodes"] = ["v2"]
    elif mutation == "duplicate_constraint":
        constraints[1]["id"] = constraints[0]["id"]
    elif mutation == "duplicate_relation":
        extra = copy.deepcopy(constraints[0]); extra["id"] = "new"
        extra["entities"].reverse(); constraints.append(extra)
    elif mutation == "same_entity":
        constraints[0]["entities"] = ["a", "a"]
    else:
        constraints[0]["value"] = 1. if mutation == "nonzero_nominal" else float("nan")
    assert not relation_checks(entities, constraints)["passed"]


@pytest.mark.parametrize("mutation", ["nan", "degenerate", "no_id", "bad_type", "no_clockwise", "off_circle", "bad_point"])
def test_malformed_model_geometry_fails_closed(mutation):
    entities, constraints = _fillet()
    if mutation == "nan":
        entities[0]["start"][0] = float("nan")
    elif mutation == "degenerate":
        entities[0]["start"] = entities[0]["end"][:]
    elif mutation == "no_id":
        entities[0].pop("id")
    elif mutation == "bad_type":
        entities[0]["type"] = "SPLINE"
    elif mutation == "no_clockwise":
        entities[1].pop("clockwise")
    elif mutation == "off_circle":
        entities[1]["radius"] = 2.
    else:
        entities[0]["start"] = [0.]
    assert not relation_checks(entities, constraints)["passed"]


def test_third_incident_entity_is_not_a_unique_tangent_joint():
    entities, constraints = _fillet()
    entities.append(_line("branch", [0., 0.], [0., -1.], "v1", "branch_end"))
    result = relation_checks(entities, constraints)
    assert not result["passed"]
    assert result["checks"][0]["reason"] == "ambiguous_joint_incidence"


def test_same_end_node_orientation_is_not_a_valid_contour_joint():
    entities, constraints = _fillet()
    entities[0]["start_node"], entities[0]["end_node"] = "v1", "v0"
    result = relation_checks(entities, constraints)
    assert not result["passed"]
    assert result["checks"][0]["reason"] == "joint_does_not_follow_contour_traversal"


def test_empty_node_list_can_infer_only_unique_joint():
    entities, constraints = _fillet()
    constraints[0]["nodes"] = []
    assert relation_checks(entities, constraints)["passed"]
    entities[0]["start_node"] = "v2"
    assert not relation_checks(entities, constraints)["passed"]


def test_two_arc_circle_requires_explicit_node_for_each_tangent():
    entities = [_arc("upper", [1., 0.], [-1., 0.], [0., 0.], 1., False, "a", "b"),
                _arc("lower", [-1., 0.], [1., 0.], [0., 0.], 1., False, "b", "a")]
    constraints = [_tangent("left", "upper", "lower", "b"), _tangent("right", "upper", "lower", "a")]
    assert relation_checks(entities, constraints, dxf_document=_native(entities))["passed"]
    constraints[0]["nodes"] = []
    assert not relation_checks(entities, constraints)["passed"]


def test_zero_obligations_does_not_claim_relation_coverage():
    entities, _ = _fillet()
    result = relation_checks(entities, [], dxf_document=_native(entities))
    assert result["passed"] and result["required_count"] == result["satisfied_count"] == 0
    assert result["complete_relation_coverage_verified"] is False


def test_non_tangent_constraint_protocol_is_not_redefined_by_relation_audit():
    entities, constraints = _fillet()
    constraints.append({"kind": "radius", "entities": ["b"], "value": 1.})
    result = relation_checks(entities, constraints, dxf_document=_native(entities))
    assert result["passed"] and result["required_count"] == 2
    constraints[0].pop("id")
    assert not relation_checks(entities, constraints)["passed"]


@pytest.mark.parametrize("entities,constraints", [(None, []), ([], None), ([None], []), ([], [None])])
def test_invalid_containers_fail_closed_without_exception(entities, constraints):
    assert not relation_checks(entities, constraints)["passed"]

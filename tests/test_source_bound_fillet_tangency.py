"""An annotated fillet's construction witness survives a child graph and fresh binding."""
import copy
import hashlib
import math

import cv2
import numpy as np
import pytest

from contour_agent import constraint_binding as binding
from contour_agent import topology_candidates as candidates
from contour_agent import topology_editing as editing
from contour_agent.parametric_solver import solve_parametric
from contour_agent.vectorize import _sample_entities


def _fillet_source():
    exact = [
        {"type": "LINE", "start": [20., 20.], "end": [100., 20.]},
        {"type": "ARC", "start": [100., 20.], "end": [120., 40.],
         "center": [100., 40.], "radius": 20., "clockwise": False},
        {"type": "LINE", "start": [120., 40.], "end": [120., 110.]},
    ]
    old = copy.deepcopy(exact)
    old[1]["center"][0] += 1.
    old[1]["radius_binding"] = {"record_id": "r001", "nominal": 20.,
                                "arrowhead_verified": True,
                                "target_source_px": [100.+20/math.sqrt(2), 40.-20/math.sqrt(2)]}
    source = _sample_entities(exact, max_step_px=.25)[0]
    return source, old, copy.deepcopy(old[1]["radius_binding"])


def _child(tmp_path):
    source, old, radius = _fillet_source()
    replacement = editing._replacement_chain(
        source, "insert_annotated_fillet", .5, old,
        annotated_radius_px=20., radius_binding=radius)
    assert replacement[1]["fillet_construction"] == "existing_line_arc_line_tangent_reinsertion"
    closure = [
        {"type": "LINE", "start": [120., 110.], "end": [20., 110.]},
        {"type": "LINE", "start": [20., 110.], "end": [20., 20.]},
    ]
    gray = np.full((140, 150), 255, np.uint8)
    for entity in [replacement[0], replacement[2], *closure]:
        cv2.line(gray, tuple(map(round, entity["start"])),
                 tuple(map(round, entity["end"])), 0, 2)
    arc_points = _sample_entities([replacement[1]], max_step_px=.25)[0]
    cv2.polylines(gray, [np.rint(arc_points).astype(np.int32)], False, 0, 2)
    image = tmp_path / "original-source.png"
    assert cv2.imwrite(str(image), gray)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    base = {"units": "mm", "source_grid_pitch_px": 1.,
            "coordinate_system": {"origin_source_px": [0., 0.]}}
    graph = candidates._to_graph(
        [*replacement, *closure], lambda points: np.asarray(points, float), 1.,
        base, "source-fillet-child", digest, "parent", .5,
        {"sampled_topology_valid": True}, [], {"source_stroke_support": {}},
        {"name": "test_source_only"})
    return image, gray, graph


def _radius_candidate():
    return {"id": "b001", "kind": "radius", "record_id": "r001",
            "entities": ["g001"], "value": 20., "local_reliable": True,
            "evidence": {"whole_primitive_radius": {"passed": True},
                         "leader": {"arrowhead_verified": True}}}


def _tangencies(image, gray, graph, radius=None):
    return [row for row in binding._constructed_fillet_tangent_relations(
        gray, [], image, graph, binding._source_transform({}, graph), [],
        [_radius_candidate()] if radius is None else radius)
            if row.get("source") == "source_bound_design_fillet"]


def test_reinsertion_witness_survives_child_and_admits_only_two_design_tangencies(tmp_path):
    image, gray, graph = _child(tmp_path)
    arc = graph["entities"][1]
    assert arc["fillet_construction"] == "existing_line_arc_line_tangent_reinsertion"
    assert arc["source_refinement"]["ground_truth_used"] is False
    assert arc["radius"] == 20.
    relations = _tangencies(image, gray, graph)
    assert len(relations) == 2
    assert {tuple(row["entities"]) for row in relations} == {
        ("g000", "g001"), ("g001", "g002")}
    assert all(row["evidence"]["evidence_class"] == "source_bound_design_construction"
               and row["evidence"]["independent_source_tangent_measurement"] is False
               and row["evidence"]["directed_tangent_deviation_degrees"] < 1e-4
               for row in relations)


def test_reinserted_fillet_stays_tangent_when_solver_changes_a_dimension(tmp_path):
    image, gray, graph = _child(tmp_path)
    constraints = [
        {"id": "radius", "kind": "radius", "entities": ["g001"], "nodes": [],
         "value": 20., "record_id": "r001", "source": "ocr_local_binding"},
        {"id": "width", "kind": "distance_x", "entities": [],
         "nodes": ["v000", "v002"], "value": 100.5, "record_id": "r002",
         "source": "ocr_local_binding"},
    ]
    for index, relation in enumerate(_tangencies(image, gray, graph)):
        constraints.append({"id": f"design-tangent-{index}", "kind": "tangent",
                            "entities": relation["entities"], "nodes": relation["nodes"],
                            "value": None, "record_id": None, "source": "source_geometry",
                            "evidence_class": "source_bound_design_construction",
                            "independent_source_tangent_measurement": False})
    assert len(constraints) == 4
    solved = solve_parametric(graph, constraints)
    assert solved["accepted"]
    assert solved["validation"]["strict_radius_satisfied"]
    assert all(row["passed"] and row["absolute_residual"] < .1
               for row in solved["constraints"] if row["kind"] == "tangent")
    assert next(row for row in solved["constraints"] if row["id"] == "width")["actual"] == pytest.approx(100.5)


@pytest.mark.parametrize("defect", ["no_fresh_radius", "stale_source", "changed_line",
                                    "ordinary_arc_refit", "changed_contact"])
def test_design_tangency_fails_closed_without_current_source_and_geometry(tmp_path, defect):
    image, gray, graph = _child(tmp_path)
    radius = None
    if defect == "no_fresh_radius":
        radius = []
    elif defect == "stale_source":
        graph["source_sha256"] = "wrong"
    elif defect == "changed_line":
        graph["entities"][0]["start"][1] += 2.
    elif defect == "ordinary_arc_refit":
        graph["entities"][1].pop("fillet_construction")
    else:
        graph["entities"][1]["source_refinement"]["constructed_contact_points_px"][0][0] += 1.
    assert not _tangencies(image, gray, graph, radius)


def test_exact_bound_arc_with_bad_tangent_still_proposes_local_reinsertion():
    _, old, radius = _fillet_source()
    graph = {"units": "mm", "source_grid_pitch_px": 1.,
             "nodes": [{"id": f"v{i}", "source_px": point} for i, point in enumerate(
                 [[20., 20.], [100., 20.], [120., 40.], [120., 110.], [20., 110.]])],
             "entities": [
                 {**entity, "id": f"g{i:03d}", "start_node": f"v{i}",
                  "end_node": f"v{(i+1)%5}"}
                 for i, entity in enumerate([*old,
                     {"type": "LINE", "start": [120., 110.], "end": [20., 110.]},
                     {"type": "LINE", "start": [20., 110.], "end": [20., 20.]}])],
             "annotation_support": [{"record_id": "r001", "kind": "radius",
                 "status": "candidate_supported", "candidate_entity_id": "g001",
                 "arrowhead_verified": True,
                 "source_evidence": {"target_source_px": radius["target_source_px"]}}]}
    operations = editing.propose_annotation_arc_edits(
        graph, [{"record_id": "r001", "kind": "radius", "nominal": 20.}], limit=4)
    assert any(row["action"] == "insert_annotated_fillet"
               and row["entity_ids"] == ["g000", "g001", "g002"]
               and "existing_radius_arc" in row["evidence_tags"] for row in operations)


def test_restored_angle_line_witness_survives_full_finite_support_extension():
    source, old, radius = _fillet_source()
    old[1]["end"] = [120., 44.]
    old[2]["start"] = [120., 44.]
    witness = {"record_id": "r035", "method": "verified_two_radius_joint_line_topology_restore",
               "ground_truth_used": False, "requires_angle_binding_and_solve": True,
               "source_line": {"start_px": [120., 40.], "end_px": [120., 110.]}}
    old[2]["angle_support_evidence"] = copy.deepcopy(witness)
    replacement = editing._replacement_chain(
        source, "insert_annotated_fillet", 2.5, old,
        annotated_radius_px=20., radius_binding=radius)
    assert replacement[2]["start"] == pytest.approx([120., 40.])
    assert replacement[2]["angle_support_evidence"] == witness
    assert replacement[2]["angle_support_evidence"] is not old[2]["angle_support_evidence"]
    closure = [{"type": "LINE", "start": [120., 110.], "end": [20., 110.]},
               {"type": "LINE", "start": [20., 110.], "end": [20., 20.]}]
    graph = candidates._to_graph(
        [*replacement, *closure], lambda points: np.asarray(points, float), 1.,
        {"units": "mm", "source_grid_pitch_px": 1.}, "child", "image-sha", "parent",
        2.5, {"sampled_topology_valid": True}, [],
        {"source_stroke_support": {}}, {"name": "test_source_only"})
    assert graph["entities"][2]["angle_support_evidence"] == witness


@pytest.mark.parametrize("defect", ["old_line_trimmed", "untrusted_witness"])
def test_angle_line_witness_is_not_carried_without_preserved_source_support(defect):
    source, old, radius = _fillet_source()
    witness = {"record_id": "r035", "method": "verified_two_radius_joint_line_topology_restore",
               "ground_truth_used": False}
    if defect == "old_line_trimmed":
        old = [{"type": "LINE", "start": [20., 20.], "end": [120., 20.]},
               {"type": "LINE", "start": [120., 20.], "end": [120., 110.]}]
        old[1]["angle_support_evidence"] = witness
        suffix_index = 2
    else:
        witness["ground_truth_used"] = True
        old[2]["angle_support_evidence"] = witness
        suffix_index = 2
    replacement = editing._replacement_chain(
        source, "insert_annotated_fillet", .5, old,
        annotated_radius_px=20., radius_binding=radius)
    assert "angle_support_evidence" not in replacement[suffix_index]

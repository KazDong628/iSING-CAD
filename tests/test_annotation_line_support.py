import copy
import math

import numpy as np
import pytest

from contour_agent.annotation_line_support import (
    angle_edit_evidence, propose_annotation_line_edits, restore_annotated_line_support,
    restore_annotated_joint_line_support, protect_annotated_straight_supports,
)
from contour_agent.topology_editing import _fixed_radius_arc, _replacement_chain, propose_annotation_arc_edits
from contour_agent.vectorize import _sample_entities


def _observation(start=(0., 0.), end=(0., 30.)):
    return {"record_id": "angle-a", "nominal": 15., "reference_axis": "vertical", "verified": True,
            "source_line": {"start_px": list(start), "end_px": list(end)},
            "target_candidates": [{"entity_id": "edge-a"}]}


def _inventory():
    return [{"record_id": "angle-a", "kind": "angle", "nominal": 15.}]


def _graph():
    return {"entities": [{"id": "edge-a", "type": "ARC"}],
            "angle_source_observations": [_observation()]}


def _operation():
    return {"action": "restore_annotated_line_support", "entity_ids": ["edge-a"], "record_id": "angle-a"}


def test_proposal_reserves_angle_type_correction_without_case_or_value_constants():
    graph = _graph()
    original = copy.deepcopy(graph)
    proposed = propose_annotation_line_edits(graph, _inventory())
    assert len(proposed) == 1
    assert proposed[0]["action"] == "restore_annotated_line_support"
    assert proposed[0]["record_id"] == "angle-a"
    assert graph == original
    assert propose_annotation_arc_edits(graph, _inventory(), limit=1) == proposed


@pytest.mark.parametrize("change", [
    {"verified": False}, {"reference_axis": "unidentified"}, {"nominal": 12.},
    {"target_candidates": [{"entity_id": "other"}]},
])
def test_unknown_mismatched_or_unverified_angle_does_not_authorize_an_edit(change):
    graph = _graph()
    graph["angle_source_observations"][0].update(change)
    assert propose_annotation_line_edits(graph, _inventory()) == []
    with pytest.raises(ValueError, match="angle_line_source_evidence_not_verified"):
        angle_edit_evidence(graph, _inventory(), _operation())


def test_provider_coordinates_do_not_override_local_source_observation():
    operation = {**_operation(), "source_line": {"start_px": [100., 100.], "end_px": [200., 200.]},
                 "verified": True}
    assert angle_edit_evidence(_graph(), _inventory(), operation)["source_line"] == _observation()["source_line"]
    graph = _graph()
    graph["angle_source_observations"] = []
    with pytest.raises(ValueError):
        angle_edit_evidence(graph, _inventory(), operation)


@pytest.mark.parametrize("nominal", [True, -1., 0., 90., float("nan"), float("inf")])
def test_invalid_angle_records_remain_unresolved(nominal):
    inventory = [{"record_id": "angle-a", "kind": "angle", "nominal": nominal}]
    assert propose_annotation_line_edits(_graph(), inventory) == []


def _straight_then_curve():
    entities = [{"type": "LINE", "start": [0., 0.], "end": [0., 30.]},
                {"type": "ARC", "start": [0., 30.], "end": [20., 50.],
                 "center": [20., 30.], "radius": 20., "clockwise": True}]
    return _sample_entities(entities, max_step_px=.25)[0]


@pytest.mark.parametrize("rotation,reflection", [(0., 1.), (31., 1.), (11., -1.)])
def test_local_line_is_restored_without_flattening_the_real_arc_tail(rotation, reflection):
    source = _straight_then_curve()
    angle = math.radians(rotation)
    transform = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]) @ np.diag([1., reflection])
    source = source @ transform.T
    observation = _observation(*(np.array([[0., 0.], [0., 30.]]) @ transform.T))
    result = restore_annotated_line_support(source, observation, .2)
    assert [row["type"] for row in result] == ["LINE", "ARC"]
    assert result[0]["start"] == pytest.approx(source[0])
    assert result[-1]["end"] == pytest.approx(source[-1])
    assert result[0]["end"] == result[1]["start"]
    assert math.dist(result[0]["start"], result[0]["end"]) >= 28.
    assert result[0]["angle_support_evidence"]["numeric_angle_bound"] is False
    assert result[0]["angle_support_evidence"]["requires_angle_binding_and_solve"] is True


def test_two_real_curved_tails_and_their_exact_radii_survive_line_restoration():
    entities = [{"type": "ARC", "start": [0., 0.], "end": [10., 10.], "center": [10., 0.], "radius": 10., "clockwise": True},
                {"type": "LINE", "start": [10., 10.], "end": [40., 10.]},
                {"type": "ARC", "start": [40., 10.], "end": [50., 20.], "center": [40., 20.], "radius": 10., "clockwise": False}]
    source = _sample_entities(entities, max_step_px=.25)[0]
    targets = [{"record_id": f"R-{i}", "radius_px": 10., "target_source_px": px,
                "binding": {"record_id": f"R-{i}", "nominal": 10., "target_source_px": px}}
               for i, px in enumerate(([10-10/math.sqrt(2), 10/math.sqrt(2)], [40+10/math.sqrt(2), 20-10/math.sqrt(2)]))]
    result = restore_annotated_line_support(source, _observation((10., 10.), (40., 10.)), .15,
                                              radius_targets=targets, fixed_radius_fitter=_fixed_radius_arc)
    assert [row["type"] for row in result] == ["ARC", "LINE", "ARC"]
    assert result[0]["radius"] == result[2]["radius"] == 10.
    assert result[0]["radius_binding"]["record_id"] == "R-0"
    assert result[2]["radius_binding"]["record_id"] == "R-1"
    assert result[0]["start"] == source[0].tolist()
    assert result[-1]["end"] == source[-1].tolist()


def test_real_radius_arrow_inside_proposed_line_is_not_silently_discarded():
    source = _straight_then_curve()
    target = {"record_id": "R-a", "target_source_px": [0., 10.], "radius_px": 5., "binding": {"record_id": "R-a"}}
    with pytest.raises(ValueError, match="cannot_preserve_source_and_radius_support"):
        restore_annotated_line_support(source, _observation(), .2, radius_targets=[target])


def test_existing_exact_radius_is_not_lost_by_retyping_the_entire_arc():
    source = np.c_[np.zeros(80), np.linspace(0., 30., 80)]
    selected = [{"type": "ARC", "start": source[0].tolist(), "end": source[-1].tolist(),
                 "center": [100., 15.], "radius": 101.11874208078342, "clockwise": False,
                 "radius_binding": {"record_id": "R-a", "target_source_px": [0., 15.]}}]
    with pytest.raises(ValueError, match="edit_would_discard_bound_radius"):
        _replacement_chain(source, "restore_annotated_line_support", .2, selected, angle_evidence=_observation())


def test_line_observation_far_from_source_is_rejected_without_tolerance_relaxation():
    with pytest.raises(ValueError, match="angle_line_source_interval_not_supported"):
        restore_annotated_line_support(_straight_then_curve(), _observation((80., 0.), (80., 30.)), .2)


def test_does_not_propose_edits_for_already_straight_objects():
    graph = _graph()
    graph["entities"][0]["type"] = "LINE"
    assert propose_annotation_line_edits(graph, _inventory()) == []


def test_existing_fully_witnessed_line_prevents_duplicate_tangent_arc_repair():
    graph = _graph()
    graph["entities"].append({"id": "line-neighbor", "type": "LINE"})
    graph["angle_source_observations"][0]["target_candidates"].append({"entity_id": "line-neighbor", "whole_line_supported": True})
    assert propose_annotation_line_edits(graph, _inventory()) == []


def test_later_merge_cannot_consume_an_angle_supported_straight_line():
    graph = _graph()
    graph["angle_source_observations"][0]["target_candidates"][0]["whole_line_supported"] = True
    source = {"type": "LINE", "start": [0., 0.], "end": [0., 30.]}
    with pytest.raises(ValueError, match="edit_would_discard_angle_supported_line"):
        protect_annotated_straight_supports(graph, ["edge-a"], [source],
                                            [{"type": "ARC", "start": [0., 0.], "end": [0., 30.]}], .2)
    protect_annotated_straight_supports(graph, ["edge-a"], [source],
                                        [{"type": "LINE", "start": [0., 2.], "end": [0., 28.]}], .2)


def test_unbounded_sampling_request_fails_without_allocating_a_large_path():
    with pytest.raises(ValueError, match="sampling_budget_exhausted"):
        restore_annotated_line_support([[0., 0.], [0., 1e9]], _observation(), .2)


def _two_radius_joint_source(*, finite_line=True, radius_tip_on_line=False):
    first = {"type": "ARC", "start": [0., 0.], "end": [10., 10.],
             "center": [0., 10.], "radius": 10., "clockwise": False}
    second_start = [10., 30.] if finite_line else [10., 10.]
    second = {"type": "ARC", "start": second_start,
              "end": [20., 40.] if finite_line else [20., 20.],
              "center": [20., 30.] if finite_line else [20., 10.],
              "radius": 10., "clockwise": True}
    entities = [first, {"type": "LINE", "start": [10., 10.], "end": [10., 30.]}, second] if finite_line else [first, second]
    rotation = math.radians(15.)
    matrix = np.array([[math.cos(rotation), -math.sin(rotation)],
                       [math.sin(rotation), math.cos(rotation)]])
    for entity in entities:
        for key in ("start", "end", "center"):
            if key in entity:
                entity[key] = (np.asarray(entity[key]) @ matrix.T).tolist()
    source = _sample_entities(entities, max_step_px=.2)[0]
    transform = lambda point: (np.asarray(point) @ matrix.T).tolist()
    joint = transform([10., 20.] if finite_line else [10., 10.])
    observation = {**_observation(transform([10., 10.]), transform([10., 30.])),
                   "joint_source_px": joint, "joint_radius_arrow_verified": True,
                   "target_candidates": [{"entity_id": "arc-a", "supported_span_px": 20.}]}
    tips = [transform([10. / math.sqrt(2), 10. - 10. / math.sqrt(2)]),
            transform([20. - 10. / math.sqrt(2), (30. if finite_line else 10.) + 10. / math.sqrt(2)])]
    if radius_tip_on_line:
        tips[0] = transform([10., 15.])
    targets = [{"record_id": f"R-{index}", "radius_px": 10., "target_source_px": tip,
                "source_arrow_verified": True,
                "binding": {"record_id": f"R-{index}", "nominal": 10.,
                            "target_source_px": tip, "arrowhead_verified": True}}
               for index, tip in enumerate(tips)]
    return source, observation, targets


def test_two_arc_joint_requires_source_angle_and_separate_directed_radius_arrows():
    _, observation, _ = _two_radius_joint_source()
    graph = {"proposal_tolerance_px": .2,
             "entities": [{"id": "arc-a", "type": "ARC", "start_node": "a", "end_node": "joint"},
                          {"id": "arc-b", "type": "ARC", "start_node": "joint", "end_node": "b"},
                          {"id": "closing", "type": "LINE", "start_node": "b", "end_node": "a"}],
             "nodes": [{"id": "joint", "source_px": observation["joint_source_px"]}],
             "annotation_support": [{"kind": "radius", "record_id": f"R-{i}",
                                     "candidate_entity_id": f"arc-{letter}", "status": "candidate_supported",
                                     "arrowhead_verified": True}
                                    for i, letter in enumerate("ab")],
             "angle_source_observations": [observation]}
    operation = propose_annotation_line_edits(graph, _inventory())[0]
    assert operation["entity_ids"] == ["arc-a", "arc-b"]
    assert angle_edit_evidence(graph, _inventory(), operation)["joint_source_px"] == observation["joint_source_px"]
    graph["annotation_support"][1]["arrowhead_verified"] = False
    assert propose_annotation_line_edits(graph, _inventory())[0]["entity_ids"] == ["arc-a"]


def test_short_finite_line_between_two_labelled_arcs_is_restored_without_losing_either_radius():
    source, observation, targets = _two_radius_joint_source()
    result = restore_annotated_joint_line_support(
        source, observation, .2, radius_targets=targets, fixed_radius_fitter=_fixed_radius_arc)
    assert [row["type"] for row in result] == ["ARC", "LINE", "ARC"]
    assert [result[0]["radius_binding"]["record_id"], result[2]["radius_binding"]["record_id"]] == ["R-0", "R-1"]
    assert result[0]["radius"] == result[2]["radius"] == 10.
    assert result[0]["radius_annotation_evidence"] == result[0]["radius_binding"]
    assert result[2]["radius_annotation_evidence"] == result[2]["radius_binding"]
    assert 12. <= math.dist(result[1]["start"], result[1]["end"]) <= 23.
    assert result[0]["start"] == pytest.approx(source[0])
    assert result[-1]["end"] == pytest.approx(source[-1])
    assert result[1]["angle_support_evidence"]["numeric_angle_bound"] is False


def test_curved_only_joint_or_radius_arrow_on_line_cannot_create_false_straight_object():
    for kwargs in ({"finite_line": False}, {"radius_tip_on_line": True}):
        source, observation, targets = _two_radius_joint_source(**kwargs)
        with pytest.raises(ValueError, match="angle_line_joint"):
            restore_annotated_joint_line_support(
                source, observation, .2, radius_targets=targets, fixed_radius_fitter=_fixed_radius_arc)


def test_joint_radius_construction_rejects_unverified_binding_metadata():
    source, observation, targets = _two_radius_joint_source()
    targets[1]["binding"]["arrowhead_verified"] = False
    with pytest.raises(ValueError, match="angle_line_joint_radius_arrow_unverified"):
        restore_annotated_joint_line_support(
            source, observation, .2, radius_targets=targets, fixed_radius_fitter=_fixed_radius_arc)


def test_offline_fallback_selects_only_new_source_verified_annotated_line():
    from copy import deepcopy
    from contour_agent.parametric_pipeline import _offline_edit_selection_eligible

    before = {"graph": {"entities": [{"type": "ARC"}, {"type": "ARC"}]}}
    after = {"graph": {"source_grid_pitch_px": 1., "entities": [
        {"type": "ARC"},
        {"type": "LINE", "angle_support_evidence": {
            "record_id": "new_angle", "requires_angle_binding_and_solve": True,
            "ground_truth_used": False, "source_interval_endpoints_px": [[0., 0.], [0., 24.]]}},
        {"type": "ARC"}],
        "source_evidence": {"topology_edit": {
            "action": "restore_annotated_line_support", "record_id": "new_angle",
            "resegmentation_applied": True, "net_entity_reduction": -1,
            "angle_support_record_ids": ["new_angle"]}}},
        "constraint_regression_gate": {"passed": True},
        "constraint_feedback": {"issue_count": 7, "satisfied_record_ids": ["old", "new_angle"],
                                "source_validation": {"passed": True},
                                "topology_source_validation": {"passed": True}}}
    prior = {"issue_count": 7, "bound_record_ids": ["old"]}
    base_score, candidate_score = {"score": .859}, {"score": .852}
    def eligible(row, score=candidate_score):
        return _offline_edit_selection_eligible(before, row, prior, base_score, score)

    assert eligible(after)
    assert not eligible(after, {"score": .838})
    missing = deepcopy(after)
    missing["constraint_feedback"]["satisfied_record_ids"] = ["old"]
    assert not eligible(missing)
    unsupported = deepcopy(after)
    unsupported["constraint_feedback"]["source_validation"]["passed"] = False
    assert not eligible(unsupported)
    unsupported = deepcopy(after)
    unsupported["constraint_regression_gate"]["passed"] = False
    assert not eligible(unsupported)
    no_line = deepcopy(after)
    no_line["graph"]["entities"][1]["type"] = "ARC"
    assert not eligible(no_line)

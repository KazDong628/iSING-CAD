"""Angular values come from OCR; targets need source arrows and LINE support."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image, ImageDraw
import pytest

from contour_agent import constraint_binding as binding


def angle_source(tmp_path, *, arrows=True, reference=True, horizontal=False, nominal=15.):
    image = Image.new("L", (800, 700), 255)
    draw = ImageDraw.Draw(image)
    if reference:
        draw.line([(250, 100), (250, 620)], fill=0, width=3)
    draw.line([(250, 100), (385.2, 620)], fill=0, width=3)
    if arrows:
        draw.line([(250, 500), (354, 500)], fill=0, width=2)
        draw.polygon([(250, 500), (276, 494), (276, 506)], fill=0)
        draw.polygon([(354, 500), (328, 494), (328, 506)], fill=0)
    box = [[260, 452], [340, 452], [340, 488], [260, 488]]
    pixels = np.asarray([[277, 200], [334.2, 420]], float)
    if horizontal:
        image = image.transpose(Image.Transpose.TRANSPOSE)
        box = [p[::-1] for p in box]
        pixels = pixels[:, ::-1]
    path = tmp_path/"angle.png"
    image.save(path)
    row = {"id": "source_angle", "text": f"{nominal:g}°", "box": box,
           "parsed": {"kind": "angle", "nominal": nominal}}
    coords = pixels*np.asarray([1., -1.])
    graph = {"entities": [{"id": "slanted_side", "type": "LINE", "start": coords[0].tolist(),
                           "end": coords[1].tolist(), "start_node": "joint_a", "end_node": "joint_b"}],
             "nodes": [{"id": name, "x": p[0], "y": p[1], "source_px": s.tolist()}
                       for name, p, s in zip(("joint_a", "joint_b"), coords, pixels)],
             "relations": [], "units": "mm", "proposal_tolerance_px": 5,
             "coordinate_system": {"origin_source_px": [0, 0]}}
    model = {"scale": {"pixels_per_mm": 1}}
    return path, np.asarray(image), row, model, graph


@pytest.mark.parametrize("horizontal", [False, True])
def test_source_extension_arrows_bind_nominal_not_fitted_direction(tmp_path, horizontal):
    path, gray, row, model, graph = angle_source(tmp_path, horizontal=horizontal)
    observations = binding._angle_source_observations(gray, [row], graph, binding._source_transform(model, graph), 5)
    candidates = binding._angle_candidates(observations, graph)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["local_reliable"] and candidate["entities"] == ["slanted_side"]
    assert candidate["reference_axis"] == ("horizontal" if horizontal else "vertical")
    assert candidate["value"] == 15.
    measured = observations[0]["source_line"]["observed_direction_deg_from_axis"]
    assert abs(measured-15.) > .1
    assert observations[0]["evidence"]["nominal_used_to_rank"] is False
    # Arrows lie on an extension, outside the material LINE's finite endpoints.
    tip = observations[0]["evidence"]["target_arrow"]["tip_px"]
    assert tip[0 if horizontal else 1] > 420


@pytest.mark.parametrize("kwargs", [{"arrows": False}, {"reference": False}, {"nominal": 40.}])
def test_missing_arrow_axis_or_wrong_stroke_stays_unresolved(tmp_path, kwargs):
    _, gray, row, model, graph = angle_source(tmp_path, **kwargs)
    result = binding._angle_source_observations(gray, [row], graph, binding._source_transform(model, graph), 5)
    assert result == []


def observation(entity="side", *, axis="vertical", kind="LINE", span=150., whole=True):
    return {"record_id": "angle_label", "nominal": 15., "verified": True, "reference_axis": axis,
            "target_candidates": [{"entity_id": entity, "entity_type": kind, "supported_span_px": span,
                                   "source_interval": [0., 1.], "whole_line_supported": whole}],
            "evidence": {"verified": True, "reference_arrow": {"verified": True}, "target_arrow": {"verified": True}}}


def test_arc_support_remains_topology_evidence_never_an_angle_binding():
    arc = observation(kind="ARC", whole=False)
    assert binding._angle_candidates([arc], {"entities": [{"id": "side", "type": "ARC"}]}) == []


def test_adjacent_tangent_arc_does_not_hide_unique_whole_straight_side():
    obs = observation()
    obs["target_candidates"].extend(observation("arc", kind="ARC", span=60., whole=False)["target_candidates"])
    graph = {"entities": [{"id": "side", "type": "LINE", "start_node": "a", "end_node": "b"},
                          {"id": "arc", "type": "ARC", "start_node": "b", "end_node": "c"}]}
    assert binding._angle_candidates([obs], graph)[0]["local_reliable"]
    graph["entities"][1]["start_node"] = "elsewhere"
    assert not binding._angle_candidates([obs], graph)[0]["local_reliable"]


def test_long_near_tangent_radius_interval_at_shared_joint_does_not_hide_short_line():
    obs = observation(span=79.)
    curved = observation("arc", kind="ARC", span=89., whole=False)["target_candidates"][0]
    curved["source_interval"] = [.76, 1.]
    obs["target_candidates"].append(curved)
    graph = {"entities": [{"id": "side", "type": "LINE", "start_node": "joint", "end_node": "outside"},
                          {"id": "arc", "type": "ARC", "start_node": "inside", "end_node": "joint"}]}
    assert binding._angle_candidates([obs], graph)[0]["local_reliable"]
    curved["source_interval"] = [.25, .65]
    assert not binding._angle_candidates([obs], graph)[0]["local_reliable"]
    curved["source_interval"] = [.76, 1.]
    graph["entities"][1]["end_node"] = "elsewhere"
    assert not binding._angle_candidates([obs], graph)[0]["local_reliable"]


def test_two_supported_lines_or_axes_cannot_silently_choose_one():
    graph = {"entities": [{"id": "side", "type": "LINE"}, {"id": "second", "type": "LINE"}]}
    for rows in ([observation(), observation("second")],
                 [observation(), observation(axis="horizontal")]):
        candidates = binding._angle_candidates(rows, graph)
        assert len(candidates) == 2 and not any(c["local_reliable"] for c in candidates)


def test_conflicting_angles_keep_only_finite_line_arrow_owner():
    graph = {"units": "mm", "source_grid_pitch_px": 1.,
             "coordinate_system": {"origin_source_px": [0., 0.]},
             "entities": [
                 {"id": "lower", "type": "LINE", "start": [0., 0.], "end": [30., -100.]},
                 {"id": "upper", "type": "LINE", "start": [0., 400.], "end": [-30., 300.]},
             ]}
    model = {"scale": {"pixels_per_mm": 1.}}

    def conflict(far_tip=(-65., -200.), near_tip=(-32., -100.), verify=True):
        candidates = {}
        group = []
        for record_id, value, tip in (("far", 16., far_tip), ("near", 18., near_tip)):
            observation = {"verified": True, "target_candidates": [
                {"entity_id": "lower", "whole_line_supported": True}],
                "evidence": {"target_arrow": {"verified": verify, "tip_px": list(tip)},
                             "reference_arrow": {"verified": True}}}
            candidates[record_id] = {"evidence": {"angle_observation": observation}}
            group.append(({"kind": "angle", "entities": ["lower"], "value": value},
                          {"candidate_id": record_id, "record_id": record_id}))
        return group, candidates

    # The two verified arrows are on different finite line extensions. The
    # farther record stays unresolved; proximity alone does not bind it to
    # the upper line without a separately detected source-angle observation.
    group, candidates = conflict()
    owner = binding._source_owned_angle_conflict(group, candidates, graph, model)
    assert owner["record_id"] == "near"
    assert group[0][1]["record_id"] == "far"

    # A missing competing finite LINE, close arrows, or an unverified arrow
    # cannot break a conflict merely by comparing OCR numeric values.
    assert binding._source_owned_angle_conflict(group, candidates,
        {**graph, "entities": graph["entities"][:1]}, model) is None
    group, candidates = conflict(far_tip=(-35., -110.))
    assert binding._source_owned_angle_conflict(group, candidates, graph, model) is None
    group, candidates = conflict(verify=False)
    assert binding._source_owned_angle_conflict(group, candidates, graph, model) is None


def test_graph_not_mutated_and_axis_constraint_persisted_with_denominator(tmp_path):
    path, _, row, model, graph = angle_source(tmp_path)
    original = deepcopy(graph)
    result = binding.analyze_constraint_bindings(path, {"records": [row]}, model, graph, tmp_path/"result")
    assert graph == original
    assert result["counts"]["recognized_angles"] == 1 and result["counts"]["bound_angle_records"] == 1
    constraint, = [c for c in result["constraints"] if c["kind"] == "angle"]
    assert constraint["reference_axis"] == "vertical" and constraint["value"] == 15.
    assert constraint["entities"] == ["slanted_side"] and constraint["nodes"] == []
    assert constraint["required"] and constraint["source_arrow_verified"]
    assert result["angle_source_observations"] and result["ground_truth_used"] is False
    saved = json.loads((tmp_path/"result/constraint-bindings.json").read_text(encoding="utf-8"))
    assert saved["constraints"] == result["constraints"]


def test_rotated_unverified_coordinate_axes_do_not_guess_reference_axis(tmp_path):
    _, gray, row, _, graph = angle_source(tmp_path)
    rotation = np.asarray([[.8, -.6], [.6, .8]])
    observations = binding._angle_source_observations(gray, [row], graph, lambda xy: np.asarray(xy)@rotation, 5)
    assert observations == []


def test_rejected_angle_does_not_remove_an_existing_exact_radius(tmp_path, monkeypatch):
    obs = observation()
    candidates = binding._angle_candidates([obs], {"entities": [{"id": "side", "type": "LINE"}]})
    angle = candidates[0]
    angle.update(id="angle_candidate")
    angle["evidence"]["angle_observation"]["evidence"]["target_arrow"]["verified"] = False
    radius = {"id": "radius_candidate", "record_id": "radius_label", "kind": "radius", "entities": ["arc"],
              "nodes": [], "value": 40., "local_reliable": True, "evidence": {"leader": {"arrowhead_verified": True}}}
    records = [{"id": "angle_label", "text": "15°", "parsed": {"kind": "angle", "nominal": 15.}},
               {"id": "radius_label", "text": "R40", "parsed": {"kind": "radius", "nominal": 40.}}]
    inventory = {"all_records": records, "records": records, "all_candidates": [angle, radius], "candidates": [angle, radius],
                 "units": "mm", "relations": [], "counts": {"recognized_dimensions": 2},
                 "artifacts": {"inventory": "unused", "topology": "unused"}}
    monkeypatch.setattr(binding, "build_binding_candidates", lambda *a, **kw: deepcopy(inventory))
    graph = {"nodes": [], "entities": [{"id": "side", "type": "LINE"}, {"id": "arc", "type": "ARC"}]}
    result = binding.analyze_constraint_bindings("unused", {}, {}, graph, tmp_path)
    assert len(result["constraints"]) == 1
    assert result["constraints"][0]["kind"] == "radius" and result["constraints"][0]["enforcement"] == "exact"
    assert "source_angular_arrows_not_verified" in result["issues"]
    assert result["counts"]["unbound_dimensions"] == 1

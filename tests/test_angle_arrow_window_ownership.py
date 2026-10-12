"""Whole-label arrow search must not confuse nearby finite angular supports."""
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw
import pytest

from contour_agent import constraint_binding as binding
from test_annotation_angle_binding import angle_source, observation


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("slanted_arrow", [False, True])
def test_arrow_above_ocr_bottom_uses_source_direction_and_exact_nominal(tmp_path, horizontal, slanted_arrow):
    _, _, row, model, graph = angle_source(tmp_path, horizontal=horizontal)
    image = Image.new("L", (800, 700), 255)
    draw = ImageDraw.Draw(image)
    draw.line([(250, 100), (250, 620)], fill=0, width=3)
    draw.line([(250, 100), (385.2, 620)], fill=0, width=3)
    draw.line([(250, 500), (354, 500)], fill=0, width=2)
    draw.polygon([(250, 500), (276, 494), (276, 506)], fill=0)
    tip = np.asarray([354., 500.])
    direction = np.asarray([1., -.26 if slanted_arrow else 0.])
    direction /= np.linalg.norm(direction)
    normal = np.asarray([-direction[1], direction[0]])
    triangle = [tip, tip-26*direction+6*normal, tip-26*direction-6*normal]
    draw.polygon([tuple(p) for p in triangle], fill=0)
    box = [[260, 478], [340, 478], [340, 542], [260, 542]]
    if horizontal:
        image = image.transpose(Image.Transpose.TRANSPOSE)
        box = [p[::-1] for p in box]
    row["box"] = box
    observations = binding._angle_source_observations(
        np.asarray(image), [row], graph, binding._source_transform(model, graph), 5.)
    candidate, = binding._angle_candidates(observations, graph)
    assert candidate["local_reliable"] and candidate["entities"] == ["slanted_side"]
    assert candidate["value"] == 15.
    evidence = candidate["evidence"]["angle_observation"]
    assert abs(evidence["source_line"]["observed_direction_deg_from_axis"]-15.) > .1
    # This tip is outside the old bottom-only scan. The recovered angle must
    # still use its OCR nominal, not the independently observed pixel slope.
    arrow_tip = evidence["evidence"]["target_arrow"]["tip_px"]
    assert arrow_tip[0 if horizontal else 1] < 542-80*.38
    assert evidence["evidence"]["nominal_used_to_rank"] is False
    if slanted_arrow:
        arrow_direction = evidence["evidence"]["target_arrow"]["direction_px"]
        assert abs(arrow_direction[0 if horizontal else 1]) > .15


def competing_observations():
    graph = {"units": "mm", "source_grid_pitch_px": 1.,
             "coordinate_system": {"origin_source_px": [0., 0.]},
             "entities": [
                 {"id": "lower", "type": "LINE", "start": [0., 0.], "end": [30., -100.]},
                 {"id": "upper", "type": "LINE", "start": [0., 400.], "end": [-30., 300.]},
             ]}
    rows = []
    for record_id, nominal, entity, tip in (
        ("far", 16., "lower", [-65., -200.]),
        ("near", 18., "lower", [-32., -100.]),
        ("far", 16., "upper", [-32., -300.]),
    ):
        row = observation(entity)
        row.update(record_id=record_id, nominal=nominal)
        row["evidence"]["target_arrow"]["tip_px"] = tip
        rows.append(row)
    transform = binding._source_transform({"scale": {"pixels_per_mm": 1.}}, graph)
    return graph, rows, transform


def test_independent_alternative_and_competing_owner_remove_only_false_extension():
    graph, rows, transform = competing_observations()
    original = deepcopy((graph, rows))
    result = binding._source_owned_angle_observations(rows, graph, transform)
    candidates = binding._angle_candidates(result, graph)
    assert {(r["record_id"], r["entities"][0]) for r in candidates} == {
        ("far", "upper"), ("near", "lower")}
    assert all(r["local_reliable"] for r in candidates)
    evidence, = next(r for r in result if r["record_id"] == "far")["finite_source_ownership_rejections"]
    assert evidence["rejected_entity_id"] == "lower" and evidence["owner_record_id"] == "near"
    assert evidence["independently_observed_alternative_entities"] == ["upper"]
    assert evidence["nominal_used_to_rank"] is False
    assert evidence["rejected_observation"] == rows[0]
    assert (graph, rows) == original


@pytest.mark.parametrize("change", ["no_alternative", "unverified_alternative", "unverified_alternative_arrow", "missing_arrow",
                                    "nearby_arrows", "third_competitor", "no_other_finite_line"])
def test_finite_proximity_alone_cannot_resolve_missing_or_ambiguous_source_ownership(change):
    graph, rows, transform = competing_observations()
    if change == "no_alternative":
        rows.pop()
    elif change == "unverified_alternative":
        rows[-1]["verified"] = False
    elif change == "unverified_alternative_arrow":
        rows[-1]["evidence"]["target_arrow"]["verified"] = False
    elif change == "missing_arrow":
        rows[0]["evidence"]["target_arrow"]["verified"] = False
    elif change == "nearby_arrows":
        rows[0]["evidence"]["target_arrow"]["tip_px"] = [-35., -110.]
    elif change == "third_competitor":
        rows.append(deepcopy(rows[1]))
        rows[-1]["record_id"] = "third"
    elif change == "no_other_finite_line":
        graph["entities"].pop()
    assert binding._source_owned_angle_observations(rows, graph, transform) == rows


def test_ownership_not_selected_by_nominal_angle_value():
    graph, rows, transform = competing_observations()
    expected = [(r["record_id"], r["target_candidates"][0]["entity_id"])
                for r in binding._source_owned_angle_observations(rows, graph, transform)]
    for row in rows:
        row["nominal"] = 18. if row["record_id"] == "far" else 16.
    result = binding._source_owned_angle_observations(rows, graph, transform)
    assert [(r["record_id"], r["target_candidates"][0]["entity_id"]) for r in result] == expected
    assert {r["record_id"]: r["value"] for r in binding._angle_candidates(result, graph)} == {
        "far": 18., "near": 16.}

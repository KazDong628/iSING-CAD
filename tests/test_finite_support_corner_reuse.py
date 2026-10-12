"""Thick source boundaries keep their independently certified finite corners."""
from copy import deepcopy

import cv2
import numpy as np

from contour_agent.constraint_binding import _samples
from contour_agent.measurement_anchors import virtual_corner_locations, _straight_source_support
from contour_agent.structural_evidence import source_ink
from test_parametric_solver import rounded_graph


def thick_source():
    graph = rounded_graph(width=60., height=40., radius=8., noise=0.)
    transform = lambda points: np.asarray(points)*[5., -5.]+[80., 330.]
    gray = np.full((430, 490), 255, np.uint8)
    for entity in graph["entities"]:
        points = np.rint(_samples(entity, transform)).astype(np.int32)
        cv2.polylines(gray, [points], False, 0, 8)
    return gray, graph, transform


def test_thick_line_corners_reuse_fresh_axis_stroke_without_lowering_generic_run_gate():
    gray, graph, transform = thick_source()
    _, ink = source_ink(gray, [])
    first = graph["entities"][0]
    assert not _straight_source_support(ink, transform([first["start"], first["end"]]), 4.)["verified"]
    result = virtual_corner_locations(gray, [], graph, transform, 4.)
    assert len(result) == 4
    for row in result:
        assert row["anchor"]["type"] == "line_intersection"
        assert all(support["verified"] for support in row["evidence"]["source_line_support"])
        assert not row["evidence"]["coordinates_locked"]
        assert not row["evidence"]["nominal_used"]


def test_reused_axis_evidence_remains_source_dependent_not_fitted_corner_proof():
    gray, graph, transform = thick_source()
    assert virtual_corner_locations(np.full_like(gray, 255), [], graph, transform, 4.) == []
    assert virtual_corner_locations(gray, [{"box": [[0, 0], [489, 0], [489, 429], [0, 429]]}],
                                    graph, transform, 4.) == []


def test_reused_support_does_not_turn_far_infinite_intersection_into_local_corner():
    gray, graph, transform = thick_source()
    graph = deepcopy(graph)
    graph["entities"][2].update(start=[200., 1.], end=[201., 20.])
    result = virtual_corner_locations(gray, [], graph, transform, 4.)
    assert all("g002" not in row["anchor"]["entities"] for row in result)

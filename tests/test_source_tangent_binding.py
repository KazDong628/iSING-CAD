"""Source-backed tangent discovery must not promote mask-only smoothness."""
import math

import cv2
import numpy as np
from PIL import Image

from contour_agent.constraint_binding import (
    _raw_boundary_tangent_relations, _source_transform, analyze_constraint_bindings,
)


def _fixture(tmp_path, *, source_kink=0., mask_kink=0., blank=False, coarse=True):
    gray = np.full((260, 260), 255, np.uint8)
    theta = np.linspace(np.pi / 2, 0., 181)
    arc = np.c_[110 + 65 * np.cos(theta), 85 + 65 * np.sin(theta)]
    def rotate(points, degrees):
        angle = math.radians(degrees)
        matrix = np.array([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
        return (points - [110., 150.]) @ matrix.T + [110., 150.]
    if not blank:
        cv2.line(gray, (30, 150), (110, 150), 0, 2)
        cv2.polylines(gray, [np.rint(rotate(arc, source_kink)).astype(np.int32)], False, 0, 2)
    ring = np.vstack([np.c_[np.linspace(30., 110., 81), np.full(81, 150.)],
                      rotate(arc, mask_kink)[1:],
                      [[230., 85.], [230., 230.], [30., 230.], [30., 150.]]])
    source = [[30., 150.], [110., 150.], [175., 85.]]
    design = [[x, 260 - y] for x, y in source]
    graph = {"units": "mm", "source_grid_pitch_px": 1., "proposal_tolerance_px": 4.,
             "nodes": [{"id": f"v{i:03d}", "x": q[0], "y": q[1], "source_px": p}
                       for i, (p, q) in enumerate(zip(source, design))],
             "entities": [{"id": "g000", "type": "LINE", "start_node": "v000", "end_node": "v001",
                           "start": design[0], "end": design[1]},
                          {"id": "g001", "type": "ARC", "start_node": "v001", "end_node": "v002",
                           "start": design[1], "end": design[2], "clockwise": False,
                           "center": [128., 157.] if coarse else [110., 175.],
                           "radius": math.hypot(18., 47.) if coarse else 65.}], "relations": []}
    model = {"extraction": {"raw_polyline_px": ring.tolist()}}
    path = tmp_path / "source.png"
    Image.fromarray(gray).save(path)
    return path, gray, model, graph


def _inspect(gray, model, graph, existing=None):
    return _raw_boundary_tangent_relations(gray, [], model, graph,
                                          _source_transform(model, graph), 4., existing or [])


def test_source_paths_discover_tangency_despite_coarse_arc_angle(tmp_path):
    path, gray, model, graph = _fixture(tmp_path)
    relations = _inspect(gray, model, graph)
    assert len(relations) == 1
    relation = relations[0]
    assert relation["local_reliable"]
    assert relation["evidence"]["mask_observed_deviation_degrees"] <= 3.
    assert relation["evidence"]["observed_deviation_degrees"] <= 3.
    assert all(side["verified"] for side in relation["evidence"]["sides"])
    # Integration: independently measured relation enters the ordinary binding
    # admission path rather than bypassing the established relation checks.
    result = analyze_constraint_bindings(path, {}, model, graph, tmp_path / "bindings")
    assert any(row["kind"] == "tangent" for row in result["constraints"])


def test_visible_corner_is_not_promoted_by_smooth_mask(tmp_path):
    _, gray, model, graph = _fixture(tmp_path, source_kink=15., coarse=False)
    relations = _inspect(gray, model, graph)
    assert not any(row["local_reliable"] for row in relations)


def test_mask_corner_is_not_promoted_by_smooth_source_ink(tmp_path):
    _, gray, model, graph = _fixture(tmp_path, mask_kink=15., coarse=False)
    assert not any(row["local_reliable"] for row in _inspect(gray, model, graph))


def test_right_angle_joint_is_not_tangent(tmp_path):
    _, gray, model, graph = _fixture(tmp_path, source_kink=90., mask_kink=90., coarse=False)
    assert not any(row["local_reliable"] for row in _inspect(gray, model, graph))


def test_mask_without_original_ink_cannot_supply_constraint(tmp_path):
    _, gray, model, graph = _fixture(tmp_path, blank=True)
    assert not any(row["local_reliable"] for row in _inspect(gray, model, graph))


def test_missing_raw_boundary_does_not_reconstruct_from_fitted_graph(tmp_path):
    _, gray, _, graph = _fixture(tmp_path)
    assert _inspect(gray, {}, graph) == []


def test_open_source_path_cannot_be_silently_closed_for_evidence(tmp_path):
    _, gray, model, graph = _fixture(tmp_path)
    model["extraction"]["raw_polyline_px"] = model["extraction"]["raw_polyline_px"][:-1]
    assert _inspect(gray, model, graph) == []


def test_multiple_nearby_ink_runs_do_not_select_a_convenient_stroke(tmp_path):
    _, gray, model, graph = _fixture(tmp_path, blank=True)
    theta = np.linspace(np.pi / 2, 0., 181)
    for offset in (-3, 3):
        cv2.line(gray, (30, 150 + offset), (110, 150 + offset), 0, 1)
        arc = np.c_[110 + (65 + offset) * np.cos(theta),
                    85 + (65 + offset) * np.sin(theta)]
        cv2.polylines(gray, [np.rint(arc).astype(np.int32)], False, 0, 1)
    assert not any(row["local_reliable"] for row in _inspect(gray, model, graph))


def test_new_source_evidence_retains_failed_prior_receipt(tmp_path):
    _, gray, model, graph = _fixture(tmp_path)
    prior = {"id": "rel007", "type": "tangent", "entities": ["g000", "g001"],
             "nodes": ["v001"], "local_reliable": False,
             "evidence": {"verified": False, "reason": "source_junction_strokes_ambiguous"}}
    result = _inspect(gray, model, graph, [prior])
    assert len(result) == 1 and result[0]["id"] == "rel007"
    assert result[0]["local_reliable"]
    assert result[0]["evidence"]["previous_geometry_seeded_evidence"] == prior["evidence"]
    assert prior["local_reliable"] is False

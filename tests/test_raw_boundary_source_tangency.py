"""Mask derivatives locate inspections but cannot decide source tangency."""
import math

import cv2
import numpy as np
import pytest

from contour_agent.constraint_binding import _raw_boundary_tangent_relations, _source_transform


def _fixture(*, noisy_mask=False, source_corner=0., blank=False):
    gray = np.full((260, 260), 255, np.uint8)
    theta = np.linspace(np.pi / 2, 0., 181)
    arc = np.c_[110 + 65 * np.cos(theta), 85 + 65 * np.sin(theta)]
    line = np.c_[np.linspace(30., 110., 81), np.full(81, 150.)]
    if not blank:
        a = math.radians(source_corner)
        rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        source_arc = (arc - [110., 150.]) @ rotation.T + [110., 150.]
        cv2.line(gray, (30, 150), (110, 150), 0, 2)
        cv2.polylines(gray, [np.rint(source_arc).astype(np.int32)], False, 0, 2)
    if noisy_mask:
        line[:, 1] += np.sin((110 - line[:, 0]) * np.pi / 16.)
    ring = np.vstack([line, arc[1:], [[230., 85.], [230., 230.], [30., 230.]], line[:1]])
    source = [[30., 150.], [110., 150.], [175., 85.]]
    design = [[x, 260 - y] for x, y in source]
    graph = {"units": "mm", "source_grid_pitch_px": 1.,
             "nodes": [{"id": f"v{i}", "x": q[0], "y": q[1], "source_px": p}
                       for i, (p, q) in enumerate(zip(source, design))],
             "entities": [{"id": "line", "type": "LINE", "start_node": "v0", "end_node": "v1",
                           "start": design[0], "end": design[1]},
                          {"id": "arc", "type": "ARC", "start_node": "v1", "end_node": "v2",
                           "start": design[1], "end": design[2], "clockwise": False,
                           "center": [110., 175.], "radius": 65.}]}
    return gray, {"extraction": {"raw_polyline_px": ring.tolist()}}, graph


def _inspect(gray, model, graph, relations=()):
    return _raw_boundary_tangent_relations(gray, [], model, graph,
                                          _source_transform(model, graph), 4., list(relations))


def test_noisy_mask_cannot_veto_independently_measured_tangent_source():
    gray, model, graph = _fixture(noisy_mask=True)
    relation, = _inspect(gray, model, graph)
    evidence = relation["evidence"]
    assert not evidence["mask_diagnostic_passed"]
    assert "mask_tangent_scale_disagreement" in evidence["mask_diagnostic_reasons"]
    assert "mask_observed_tangent_deviation_exceeded" in evidence["mask_diagnostic_reasons"]
    assert evidence["mask_sides"][0]["scale_disagreement_degrees"] > 2.
    assert evidence["mask_observed_deviation_degrees"] > 3.
    assert evidence["source_ink_measurement_performed"]
    assert not evidence["mask_derivatives_used_for_admission"]
    assert relation["local_reliable"] and evidence["verified"]
    assert evidence["observed_deviation_degrees"] <= evidence["tolerance_degrees"] == 3.
    for side in evidence["sides"]:
        assert side["verified"] and side["scale_disagreement_degrees"] <= 2.
        assert side["unambiguous_samples"] >= .75 * side["sample_count"]


def test_arc_arc_tangency_is_measured_from_source_despite_mask_noise():
    gray, model, graph = _fixture()
    left_theta = np.linspace(np.pi, 1.5 * np.pi, 181)
    left = np.c_[110 + 65 * np.cos(left_theta), 215 + 65 * np.sin(left_theta)]
    right_theta = np.linspace(np.pi / 2, 0., 181)
    right = np.c_[110 + 65 * np.cos(right_theta), 85 + 65 * np.sin(right_theta)]
    gray[:] = 255
    cv2.polylines(gray, [np.rint(left).astype(np.int32), np.rint(right).astype(np.int32)], False, 0, 2)
    raw_left = left.copy()
    raw_left[:, 1] += np.sin((1.5 * np.pi - left_theta) * 65 * np.pi / 16.)
    model["extraction"]["raw_polyline_px"] = np.vstack([
        raw_left, right[1:], [[230., 85.], [230., 230.], [30., 230.]], raw_left[:1]]).tolist()
    graph["nodes"][0].update(x=45., y=45., source_px=[45., 215.])
    graph["entities"][0].update(type="ARC", start=[45., 45.], center=[110., 45.], radius=65., clockwise=True)
    # These three nodes are collinear, so declare the normal image-axis mapping.
    graph["coordinate_system"] = {"origin_source_px": [0., 260.]}
    relation, = _inspect(gray, model, graph)
    evidence = relation["evidence"]
    assert not evidence["mask_diagnostic_passed"]
    assert "mask_tangent_scale_disagreement" in evidence["mask_diagnostic_reasons"]
    assert relation["local_reliable"] and evidence["source_ink_measurement_performed"]
    assert evidence["observed_deviation_degrees"] <= 3.
    assert all(side["verified"] and side["scale_disagreement_degrees"] <= 2. for side in evidence["sides"])


@pytest.mark.parametrize("noisy_mask", [False, True])
def test_real_source_corner_never_gains_tangency_from_mask(noisy_mask):
    gray, model, graph = _fixture(noisy_mask=noisy_mask, source_corner=15.)
    relation, = _inspect(gray, model, graph)
    assert relation["evidence"]["source_ink_measurement_performed"]
    assert not relation["local_reliable"] and not relation["evidence"]["verified"]
    assert relation["evidence"]["reason"] in {"source_junction_is_not_tangent", "source_tangent_evidence_insufficient"}


@pytest.mark.parametrize("noisy_mask", [False, True])
def test_blank_source_never_gains_tangency_even_when_mask_smooth(noisy_mask):
    gray, model, graph = _fixture(noisy_mask=noisy_mask, blank=True)
    relation, = _inspect(gray, model, graph)
    evidence = relation["evidence"]
    assert evidence["source_ink_measurement_performed"]
    assert not relation["local_reliable"] and not evidence["verified"]
    assert evidence["mask_diagnostic_passed"] is (not noisy_mask)
    assert not any(side["verified"] for side in evidence["sides"])


def test_short_span_retains_explicit_nonadmission_receipt():
    gray, model, graph = _fixture()
    graph["entities"][0]["start"] = [100., 110.]
    relation, = _inspect(gray, model, graph)
    assert not relation["local_reliable"]
    assert relation["evidence"]["reason"] == "insufficient_local_source_span"
    assert not relation["evidence"]["source_ink_measurement_performed"]


def test_joint_far_from_raw_boundary_retains_nonadmission_receipt():
    gray, model, graph = _fixture()
    graph["entities"][0]["end"] = [110., 125.]
    relation, = _inspect(gray, model, graph)
    assert not relation["local_reliable"]
    assert relation["evidence"]["reason"] == "joint_outside_source_boundary_band"
    assert relation["evidence"]["joint_to_raw_boundary_px"] > 4.
    assert not relation["evidence"]["source_ink_measurement_performed"]


def test_unavailable_raw_retry_keeps_prior_failure_without_mutation():
    gray, _, graph = _fixture()
    prior = {"id": "prior", "type": "tangent", "entities": ["line", "arc"], "nodes": ["v1"],
             "local_reliable": False, "evidence": {"verified": False, "reason": "ambiguous_source"}}
    relation, = _inspect(gray, {}, graph, [prior])
    assert relation["evidence"]["reason"] == "raw_boundary_unavailable"
    assert relation["evidence"]["previous_geometry_seeded_evidence"] == prior["evidence"]
    assert prior["evidence"]["reason"] == "ambiguous_source"


def test_source_verified_relation_is_preserved_without_mask_override():
    gray, model, graph = _fixture(noisy_mask=True, blank=True)
    prior = {"id": "prior", "type": "tangent", "entities": ["line", "arc"], "nodes": ["v1"],
             "local_reliable": True, "evidence": {"verified": True, "method": "earlier_source_receipt"}}
    assert _inspect(gray, model, graph, [prior]) == [prior]

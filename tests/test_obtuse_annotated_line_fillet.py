"""Source-drawn obtuse corners cannot use connector length as a radius veto.

The fixtures are analytic synthetic sketches, never dataset/GT geometry.
Their fitted middle LINE deliberately spans a little of both real supports.
"""
import copy
import math

import cv2
import numpy as np
import pytest

from contour_agent import topology_editing as editing
from contour_agent.topology import _StrokeEvidence, stroke_support_fraction
from contour_agent.vectorize import _sample_entities


def _source_sketch(tmp_path, *, rotation=0., reflection=1., chamfer=False):
    radius = 20.
    turn = math.radians(150.)
    u, v = np.array([1., 0.]), np.array([math.cos(turn), math.sin(turn)])
    setback = radius*math.tan(turn/2)
    # The provisional connector stops five pixels before each real contact.
    # Its >2R chord is still local and both retained supports are substantial.
    interval = setback+5.
    a, b = -(interval+240.)*u, (interval+240.)*v
    first, last = -setback*u, setback*v
    center = first+[0., radius]
    ideal = [dict(type="LINE", start=a.tolist(), end=first.tolist()),
             dict(type="ARC", start=first.tolist(), end=last.tolist(),
                  center=center.tolist(), radius=radius, clockwise=False),
             dict(type="LINE", start=last.tolist(), end=b.tolist())]
    vertices = [a, -interval*u, interval*v, b]
    selected = [dict(type="LINE", start=p.tolist(), end=q.tolist())
                for p, q in zip(vertices, vertices[1:])]
    angle = math.radians(rotation)
    transform = np.array([[math.cos(angle), -math.sin(angle)],
                          [math.sin(angle), math.cos(angle)]]) @ np.diag([1., reflection])
    def point(value):
        return (transform @ np.asarray(value)+[420., 420.]).tolist()
    target = point(center+radius*np.array([math.cos(-math.pi/2+turn/2),
                                           math.sin(-math.pi/2+turn/2)]))
    for entity in [*ideal, *selected]:
        for key in ("start", "end", "center"):
            if key in entity:
                entity[key] = point(entity[key])
        if "clockwise" in entity and reflection < 0:
            entity["clockwise"] = not entity["clockwise"]
    source = _sample_entities(selected if chamfer else ideal, max_step_px=.2)[0]
    if chamfer:
        target = ((np.asarray(selected[1]["start"])+selected[1]["end"])/2).tolist()
    binding = dict(record_id="synthetic-r001", nominal=radius,
                   arrowhead_verified=True, target_source_px=target)
    image = np.full((900, 900), 255, np.uint8)
    cv2.polylines(image, [np.rint(source).astype(np.int32)], False, 0, 3, cv2.LINE_AA)
    path = tmp_path/"synthetic-source.png"
    assert cv2.imwrite(str(path), image)
    drawn = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert drawn is not None
    return source, selected, binding, radius, setback, drawn


def _replace(source, selected, binding, radius, tolerance=5.5):
    return editing._replacement_chain(source, "insert_annotated_fillet", tolerance, selected,
        annotated_radius_px=radius, radius_binding=binding)


@pytest.mark.parametrize("rotation,reflection", [(0., 1.), (37., 1.), (13., -1.)])
def test_obtuse_source_round_accepts_connector_longer_than_diameter(tmp_path, rotation, reflection):
    source, selected, binding, radius, setback, image = _source_sketch(
        tmp_path, rotation=rotation, reflection=reflection)
    original = copy.deepcopy(selected)
    connector_length = math.dist(selected[1]["start"], selected[1]["end"])
    assert connector_length > 2*radius
    result = _replace(source, selected, binding, radius)
    assert [e["type"] for e in result] == ["LINE", "ARC", "LINE"]
    assert result[1]["radius"] == radius
    assert result[0]["start"] == selected[0]["start"]
    assert result[-1]["end"] == selected[-1]["end"]
    proof = result[1]["source_refinement"]
    corner = proof["connector_corner_geometry"]
    assert abs(corner["support_turn_deg"]) == pytest.approx(150.)
    assert corner["tangent_setback_px"] == pytest.approx(setback)
    assert corner["nominal_tangent_chord_px"] < 2*radius < corner["connector_length_px"]
    assert corner["resolved_source_curvature_px"] > corner["minimum_resolved_source_curvature_px"]
    assert proof["requires_fresh_binding_and_joint_solve"] is True
    for contact in proof["connector_domain_redistribution"]:
        assert 0. <= contact["connector_projection_fraction"] <= 1.
        assert contact["finite_connector_gap_px"] <= contact["source_tolerance_px"]
    for first, second in zip(result, result[1:]):
        assert first["end"] == pytest.approx(second["start"], abs=1e-12)
        assert float(editing._fillet_tangent(first, "end") @
                     editing._fillet_tangent(second, "start")) > 1-1e-12
    sampled = _sample_entities(result, max_step_px=.25)[0]
    actual_ink = _StrokeEvidence(image, [], 1.).summarize(sampled)
    assert stroke_support_fraction(actual_ink) >= .90
    assert selected == original


@pytest.mark.parametrize("protected", ["dimension_bound", "constraint_ids", "radius_binding"])
def test_obtuse_connector_with_formal_semantics_remains_protected(tmp_path, protected):
    source, selected, binding, radius, _, _ = _source_sketch(tmp_path)
    selected[1][protected] = {"record_id": "other"} if protected == "radius_binding" else True
    with pytest.raises(ValueError, match="fillet_connector_is_not_a_short_unbound_corner"):
        _replace(source, selected, binding, radius)


def test_actual_straight_chamfer_drawing_is_not_erased_even_with_generous_tolerance(tmp_path):
    source, selected, binding, radius, _, image = _source_sketch(tmp_path, chamfer=True)
    assert stroke_support_fraction(_StrokeEvidence(image, [], 1.).summarize(source)) >= .90
    with pytest.raises(ValueError, match="fillet_connector_has_no_resolved_source_curvature"):
        _replace(source, selected, binding, radius, tolerance=22.)


@pytest.mark.parametrize("defect", ["missing_arrow", "wrong_target", "finite_domain", "bad_source"])
def test_long_connector_never_bypasses_directed_source_or_finite_domain_checks(tmp_path, defect):
    source, selected, binding, radius, _, _ = _source_sketch(tmp_path)
    if defect == "missing_arrow":
        binding["arrowhead_verified"] = False
    elif defect == "wrong_target":
        binding["target_source_px"] = source[5].tolist()
    elif defect == "finite_domain":
        # Pull the provisional endpoints farther into both true LINEs. The
        # actual contacts are then outside the connector's source-distance band.
        for endpoint, support in (("start", selected[0]), ("end", selected[-1])):
            outer = np.asarray(support["start" if endpoint == "start" else "end"])
            contact = np.asarray(selected[1][endpoint])
            changed = contact+(outer-contact)/np.linalg.norm(outer-contact)*5.
            selected[1][endpoint] = changed.tolist()
            support["end" if endpoint == "start" else "start"] = changed.tolist()
    else:
        source[len(source)//2] += [40., -30.]
    with pytest.raises(ValueError):
        _replace(source, selected, binding, radius)

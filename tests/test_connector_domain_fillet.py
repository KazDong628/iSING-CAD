"""Only an unbound finite connector may be redistributed by an exact fillet."""
import copy
import math

import numpy as np
import pytest

from contour_agent import topology_editing as editing
from contour_agent.vectorize import _sample_entities


def _fixture(*, extension=.3, rotation=0., reflection=1.):
    ideal = [
        {"type": "LINE", "start": [20., 20.], "end": [100., 20.]},
        {"type": "ARC", "start": [100., 20.], "end": [120., 40.],
         "center": [100., 40.], "radius": 20., "clockwise": False},
        {"type": "LINE", "start": [120., 40.], "end": [120., 110.]},
    ]
    vertices = [[20., 20.], [100.-extension, 20.], [120., 40.+extension], [120., 110.]]
    selected = [{"type": "LINE", "start": a, "end": b,
                 "parent_stable_ids": [f"physical-{i}"], "parent_entity_ids": [f"g{i:03d}"]}
                for i, (a, b) in enumerate(zip(vertices, vertices[1:]))]
    angle = math.radians(rotation)
    transform = np.array([[math.cos(angle), -math.sin(angle)],
                          [math.sin(angle), math.cos(angle)]]) @ np.diag([1., reflection])
    def point(value):
        return (transform @ np.asarray(value)+[137., -23.]).tolist()
    target = point([100.+20/math.sqrt(2), 40.-20/math.sqrt(2)])
    for entity in [*ideal, *selected]:
        for key in ("start", "end", "center"):
            if key in entity:
                entity[key] = point(entity[key])
        if "clockwise" in entity and reflection < 0:
            entity["clockwise"] = not entity["clockwise"]
    binding = {"record_id": "r001", "nominal": 20., "arrowhead_verified": True,
               "target_source_px": target}
    return _sample_entities(ideal, max_step_px=.25)[0], selected, binding


def _replace(source, selected, binding, radius=20., tolerance=.5):
    return editing._replacement_chain(source, "insert_annotated_fillet", tolerance, selected,
        annotated_radius_px=radius, radius_binding=binding)


def _direction(entity):
    vector = np.asarray(entity["end"])-entity["start"]
    return vector/np.linalg.norm(vector)


@pytest.mark.parametrize("rotation,reflection", [(0., 1.), (37., 1.), (13., -1.), (180., -1.)])
def test_small_connector_redistribution_keeps_two_physical_lines(rotation, reflection):
    source, selected, binding = _fixture(rotation=rotation, reflection=reflection)
    original = copy.deepcopy(selected)
    result = _replace(source, selected, binding)
    assert [e["type"] for e in result] == ["LINE", "ARC", "LINE"]
    assert result[1]["radius"] == 20.
    assert result[1]["fillet_construction"] == "existing_lines_short_connector_redistribution"
    domain = result[1]["source_refinement"]["connector_domain_redistribution"]
    assert len(domain) == 2
    for evidence in domain:
        assert evidence["extension_px"] == pytest.approx(.3, abs=1e-10)
        assert 0. <= evidence["connector_projection_fraction"] <= 1.
        assert evidence["finite_connector_gap_px"] <= evidence["source_tolerance_px"] == .5
    for old, new in ((selected[0], result[0]), (selected[-1], result[-1])):
        assert _direction(old) == pytest.approx(_direction(new), abs=1e-12)
        assert math.dist(new["start"], new["end"]) > .5
    assert result[0]["start"] == selected[0]["start"]
    assert result[-1]["end"] == selected[-1]["end"]
    for first, second in zip(result, result[1:]):
        assert first["end"] == pytest.approx(second["start"], abs=1e-12)
        assert float(editing._fillet_tangent(first, "end") @ editing._fillet_tangent(second, "start")) > 1-1e-12
    assert selected == original


def test_extension_away_from_finite_connector_is_rejected_without_refitting(monkeypatch):
    source, selected, binding = _fixture(extension=4.)
    def forbidden(*args, **kwargs):
        raise AssertionError("No rotating or curved-support fallback")
    monkeypatch.setattr(editing, "_annotated_line_fillet", forbidden)
    with pytest.raises(ValueError, match="fillet_contact_outside_finite_line_support_or_short_connector_domain"):
        _replace(source, selected, binding)


@pytest.mark.parametrize("protected", ["radius_binding", "dimension_bound", "constraint_ids"])
def test_a_formally_bound_middle_diagonal_cannot_be_reassigned(protected):
    source, selected, binding = _fixture()
    selected[1][protected] = {"record_id": "another"} if protected == "radius_binding" else True
    with pytest.raises(ValueError, match="fillet_connector_is_not_a_short_unbound_corner"):
        _replace(source, selected, binding)


def test_real_straight_chamfer_is_not_reassigned_to_nearby_r():
    _, selected, binding = _fixture()
    source = _sample_entities(selected, max_step_px=.25)[0]
    with pytest.raises(ValueError, match="source_does_not_support_exact_annotated_fillet|fillet_connector_has_no_resolved_source_curvature"):
        _replace(source, selected, binding, tolerance=6.)


@pytest.mark.parametrize("defect", ["oversized_radius", "long_connector", "wrong_target", "missing_arrow", "broken_source", "disconnected"])
def test_connector_domain_does_not_bypass_existing_source_or_geometry_checks(defect):
    source, selected, binding = _fixture()
    radius = 20.
    if defect == "oversized_radius":
        radius = 110.
    elif defect == "long_connector":
        selected[0]["end"][0] -= 25.
        selected[1]["start"] = selected[0]["end"]
    elif defect == "wrong_target":
        binding["target_source_px"] = source[10].tolist()
    elif defect == "missing_arrow":
        binding["arrowhead_verified"] = False
    elif defect == "broken_source":
        source[len(source)//2] += [30., -20.]
    else:
        selected[1]["start"] = (np.asarray(selected[1]["start"])+[1., 1.]).tolist()
    with pytest.raises(ValueError):
        _replace(source, selected, binding, radius=radius)


def test_two_line_junction_does_not_gain_an_unbounded_extension_family():
    source, selected, binding = _fixture()
    # With no disposable connector, the two original supports must actually
    # meet and both contacts must remain inside their own finite intervals.
    pair = [selected[0], selected[-1]]
    with pytest.raises(ValueError, match="fillet_support_chain_not_connected"):
        _replace(source, pair, binding)


def test_an_ordinary_trim_does_not_claim_connector_redistribution():
    source, selected, binding = _fixture(extension=0.)
    result = _replace(source, selected, binding)
    assert result[1]["fillet_construction"] == "existing_finite_line_supports"
    assert "connector_domain_redistribution" not in result[1]["source_refinement"]


def test_contact_near_connector_but_outside_its_finite_projection_is_rejected():
    # A folded short connector has nearby infinite support but its finite
    # ownership interval points away from the tangent contact. Its distance
    # alone would pass tolerance=3; the negative projection must reject it.
    radius = 10.
    theta = math.radians(30.)
    v = np.array([math.cos(theta), math.sin(theta)])
    distance = radius*math.tan(theta/2)
    a, b = np.array([-100., 0.]), 100.*v
    first, last = np.array([-distance, 0.]), distance*v
    center = first+[0., radius]
    ideal = [{"type": "LINE", "start": a.tolist(), "end": first.tolist()},
             {"type": "ARC", "start": first.tolist(), "end": last.tolist(),
              "center": center.tolist(), "radius": radius, "clockwise": False},
             {"type": "LINE", "start": last.tolist(), "end": b.tolist()}]
    vertices = [a, np.array([-5., 0.]), -8.*v, b]
    selected = [{"type": "LINE", "start": x.tolist(), "end": y.tolist()}
                for x, y in zip(vertices, vertices[1:])]
    vector = vertices[2]-vertices[1]
    assert float((first-vertices[1])@vector)/float(vector@vector) < 0.
    assert float(editing._primitive_distance(first[None, :], selected[1])[0]) < 3.
    target = center+radius*np.array([math.cos(math.radians(-75.)), math.sin(math.radians(-75.))])
    binding = {"record_id": "r010", "nominal": radius, "arrowhead_verified": True,
               "target_source_px": target.tolist()}
    with pytest.raises(ValueError, match="fillet_contact_outside_finite_line_support_or_short_connector_domain"):
        _replace(_sample_entities(ideal, max_step_px=.25)[0], selected, binding, radius=radius, tolerance=3.)

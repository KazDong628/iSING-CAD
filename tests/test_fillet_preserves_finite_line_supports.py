"""A local fillet may trim a physical LINE, never rotate it or turn it into ARC."""
import copy
import math

import numpy as np
import pytest

from contour_agent import topology_editing as editing
from contour_agent.vectorize import _sample_entities


def _case(*, connector=False, corner=(120., 20.), rotation=0., reflection=1.):
    ideal = [
        {"type": "LINE", "start": [20., 20.], "end": [100., 20.]},
        {"type": "ARC", "start": [100., 20.], "end": [120., 40.],
         "center": [100., 40.], "radius": 20., "clockwise": False},
        {"type": "LINE", "start": [120., 40.], "end": [120., 110.]},
    ]
    vertices = ([[20., 20.], [100., 20.], [120., 40.], [120., 110.]]
                if connector else [[20., 20.], list(corner), [120., 110.]])
    selected = [{"type": "LINE", "start": a, "end": b,
                 "parent_stable_ids": [f"physical-{i}"], "parent_entity_ids": [f"g{i:03d}"]}
                for i, (a, b) in enumerate(zip(vertices, vertices[1:]))]
    angle = math.radians(rotation)
    transform = np.array([[math.cos(angle), -math.sin(angle)],
                          [math.sin(angle), math.cos(angle)]]) @ np.diag([1., reflection])
    shift = np.array([137., -23.])
    def point(value):
        return (transform @ np.asarray(value)+shift).tolist()
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
@pytest.mark.parametrize("connector", [False, True])
def test_exact_radius_preserves_finite_sloped_supports_in_all_orientations(rotation, reflection, connector):
    source, selected, binding = _case(connector=connector, rotation=rotation, reflection=reflection)
    original = copy.deepcopy(selected)
    result = _replace(source, selected, binding)
    assert [e["type"] for e in result] == ["LINE", "ARC", "LINE"]
    assert result[1]["radius"] == 20.
    assert result[1]["fillet_construction"] == "existing_finite_line_supports"
    assert result[0]["start"] == selected[0]["start"]
    assert result[-1]["end"] == selected[-1]["end"]
    for before, after in ((selected[0], result[0]), (selected[-1], result[-1])):
        assert _direction(after) == pytest.approx(_direction(before), abs=1e-12)
        original_length = math.dist(before["start"], before["end"])
        assert .5 < math.dist(after["start"], after["end"]) <= original_length+1e-10
    for first, second in zip(result, result[1:]):
        assert first["end"] == pytest.approx(second["start"], abs=1e-12)
        assert float(editing._fillet_tangent(first, "end") @ editing._fillet_tangent(second, "start")) > 1-1e-12
    assert selected == original


def test_existing_slightly_sloped_line_is_not_rotated_to_better_pixel_fit():
    # Baseline re-detects the ideal horizontal/vertical supports from these
    # pixels and silently rotates both existing slightly inclined LINE objects.
    source, selected, binding = _case(corner=(120.08, 20.10), rotation=27.)
    result = _replace(source, selected, binding)
    assert _direction(result[0]) == pytest.approx(_direction(selected[0]), abs=1e-12)
    assert _direction(result[-1]) == pytest.approx(_direction(selected[-1]), abs=1e-12)
    assert result[1]["radius"] == 20.
    old = editing._annotated_line_fillet(source, 20., .5, binding)
    assert np.linalg.norm(_direction(old[0])-_direction(selected[0])) > 1e-4


def test_tangent_cannot_extend_beyond_original_finite_line():
    source, selected, binding = _case(connector=True)
    # Supports ending before the tangent points would need extending through
    # the diagonal. This operation only authorizes trimming the existing lines.
    selected[0]["end"] = [233., -3.]
    selected[1]["start"] = selected[0]["end"]
    with pytest.raises(ValueError, match="fillet_contact_outside_finite_line_support"):
        _replace(source, selected, binding)


def test_infeasible_radius_does_not_fall_back_to_rotated_or_curved_supports(monkeypatch):
    source, selected, binding = _case()
    def forbidden(*args, **kwargs):
        raise AssertionError("Existing LINE supports must not be refitted")
    monkeypatch.setattr(editing, "_annotated_line_fillet", forbidden)
    with pytest.raises(ValueError, match="fillet_contact_outside_finite_line_support"):
        _replace(source, selected, binding, radius=110.)


@pytest.mark.parametrize("change", ["missing_arrow", "wrong_target", "wrong_radius", "bad_pixels"])
def test_source_evidence_and_radius_still_have_to_support_the_construction(change):
    source, selected, binding = _case()
    radius = 20.
    if change == "missing_arrow":
        binding["arrowhead_verified"] = False
    elif change == "wrong_target":
        binding["target_source_px"] = source[10].tolist()
    elif change == "wrong_radius":
        radius = 5.
    else:
        source[len(source)//2] += [30., -20.]
    with pytest.raises(ValueError):
        _replace(source, selected, binding, radius=radius)


def test_short_diagonal_chamfer_is_not_erased_by_a_nearby_radius():
    _, selected, binding = _case(connector=True)
    source = _sample_entities(selected, max_step_px=.25)[0]
    # A generous pre-existing raster tolerance alone must not let this true
    # straight chamfer masquerade as a visibly rounded source corner.
    with pytest.raises(ValueError, match="fillet_connector_has_no_resolved_source_curvature"):
        _replace(source, selected, binding, tolerance=6.)


@pytest.mark.parametrize("protected", ["radius_binding", "dimension_bound", "constraint_ids"])
def test_short_connector_with_existing_formal_semantics_is_not_removed(protected):
    source, selected, binding = _case(connector=True)
    selected[1][protected] = {"record_id": "other"} if protected == "radius_binding" else True
    with pytest.raises(ValueError, match="fillet_connector_is_not_a_short_unbound_corner"):
        _replace(source, selected, binding)


def test_long_third_line_requires_a_separate_topology_edit():
    source, selected, binding = _case(connector=True)
    selected[0]["end"] = [227., -3.]
    selected[1]["start"] = selected[0]["end"]
    selected[1]["end"] = [257., 47.]
    selected[2]["start"] = selected[1]["end"]
    with pytest.raises(ValueError, match="fillet_connector_is_not_a_short_unbound_corner"):
        _replace(source, selected, binding)


@pytest.mark.parametrize("protected", ["dimension_bound", "constraint_ids"])
def test_graph_connector_semantics_survive_source_coordinate_conversion(protected):
    source, selected, binding = _case(connector=True)
    vertices = [e["start"] for e in selected]+[selected[-1]["end"],
                [selected[0]["start"][0], selected[-1]["end"][1]]]
    graph = {"units": "mm", "source_grid_pitch_px": 1., "proposal_tolerance_px": .5,
             "coordinate_system": {"units": "mm", "origin_source_px": [0., 0.]},
             "nodes": [{"id": f"v{i}", "source_px": p, "x": p[0], "y": -p[1]}
                       for i, p in enumerate(vertices)],
             "entities": [{"id": f"g{i}", "type": "LINE", "start": [a[0], -a[1]],
                           "end": [b[0], -b[1]], "start_node": f"v{i}",
                           "end_node": f"v{(i+1)%len(vertices)}"}
                          for i, (a, b) in enumerate(zip(vertices, vertices[1:]+vertices[:1]))],
             "annotation_support": [{"record_id": "r001", "kind": "radius",
                 "status": "candidate_supported", "candidate_entity_id": "g1",
                 "arrowhead_verified": True, "source_evidence": {"target_source_px": binding["target_source_px"]}}]}
    graph["entities"][1][protected] = True
    raw = [*source.tolist(), vertices[-1], vertices[0]]
    baseline = {"extraction": {"raw_polyline_px": raw}, "coordinate_system": graph["coordinate_system"],
                "scale": {"pixels_per_mm": 1.}}
    with pytest.raises(ValueError, match="fillet_connector_is_not_a_short_unbound_corner"):
        editing._apply_one(graph, baseline,
            {"action": "insert_annotated_fillet", "entity_ids": ["g0", "g1", "g2"], "record_id": "r001"},
            [{"record_id": "r001", "kind": "radius", "nominal": 20., "text": "R20"}])


def test_continuous_line_support_is_independent_of_source_vertex_density():
    # A polygon encoder may represent an entire straight support by one end
    # vertex; the other contact is assigned to the adjacent ARC. Continuous
    # source-polyline distance, not assigned vertex count, establishes support.
    source, selected, binding = _case(connector=True)
    angles = np.linspace(-math.pi/2, 0., 40)
    arc = np.column_stack([237.+20.*np.cos(angles), 17.+20.*np.sin(angles)])
    sparse = np.vstack([source[0], arc, source[-1]])
    result = _replace(sparse, selected, binding)
    assert [e["type"] for e in result] == ["LINE", "ARC", "LINE"]
    assert result[1]["radius"] == 20.
    assert _direction(result[0]) == pytest.approx(_direction(selected[0]), abs=1e-12)
    assert _direction(result[-1]) == pytest.approx(_direction(selected[-1]), abs=1e-12)
    assert result[0]["start"] == selected[0]["start"]
    assert result[-1]["end"] == selected[-1]["end"]

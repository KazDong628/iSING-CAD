"""Source-only level/extremum locations retain finite-domain semantics."""
from copy import deepcopy
import math

import numpy as np
from PIL import Image, ImageDraw
import pytest

from contour_agent.measurement_anchors import (
    derived_anchor_domain_conditions,
    derived_anchor_equalities,
    derived_anchor_point,
    source_derived_locations,
    validate_derived_anchor,
)


def arc(start_degrees=0., end_degrees=180., *, clockwise=False, radius=30., center=(40., 40.)):
    center = np.asarray(center, float)
    def point(degrees):
        angle = math.radians(degrees)
        return (center+radius*np.asarray([math.cos(angle), math.sin(angle)])).tolist()
    return {"id": "g_arc", "type": "ARC", "start": point(start_degrees), "end": point(end_degrees),
            "center": center.tolist(), "radius": radius, "clockwise": clockwise}


def extremum(axis="y", sense="maximum"):
    return {"type": "curve_extremum", "entity_id": "g_arc", "axis": axis, "sense": sense}


def level(axis="y"):
    return {"type": "support_level", "entity_id": "g_line", "axis": axis}


def line(start=(0., 4.), end=(10., 4.)):
    return {"id": "g_line", "type": "LINE", "start": list(start), "end": list(end)}


def test_symbolic_level_uses_current_geometry_and_requires_its_direction_equation():
    entity = line(); indexed = {"g_line": entity}
    assert validate_derived_anchor(level(), indexed, "y") == (("support_level", "g_line", "y"), {"g_line"})
    assert derived_anchor_point(level(), indexed).tolist() == [5., 4.]
    entity.update(start=[2., 8.], end=[18., 8.])
    assert derived_anchor_point(level(), indexed).tolist() == [10., 8.]
    entity["end"][1] += .01
    # The projected midpoint must never conceal a missing horizontal relation.
    assert derived_anchor_equalities(level(), indexed).tolist() == pytest.approx([.01])
    assert derived_anchor_point(level(), indexed, allow_infeasible=True)[1] == pytest.approx(8.005)
    with pytest.raises(ValueError, match="not constant"):
        derived_anchor_point(level(), indexed)


@pytest.mark.parametrize("start,end,clockwise,axis,sense,valid", [
    (0, 180, False, "y", "maximum", True),
    (0, 180, False, "y", "minimum", False),
    (180, 0, True, "y", "maximum", True),
    (180, 0, False, "y", "maximum", False),
    (330, 30, False, "x", "maximum", True),
    (30, 330, True, "x", "maximum", True),
    (30, 330, False, "x", "maximum", False),
    (30, 330, False, "x", "minimum", True),
    (90, 180, False, "y", "maximum", True),  # Cardinal endpoint is in domain.
    (90.01, 180, False, "y", "maximum", False),
    (0, 0, False, "x", "maximum", False),  # Zero sweep is not a full circle.
])
def test_extremum_uses_declared_finite_directed_arc_not_full_circle(start, end, clockwise, axis, sense, valid):
    entity = arc(start, end, clockwise=clockwise)
    anchor = extremum(axis, sense); indexed = {entity["id"]: entity}
    if valid:
        point = derived_anchor_point(anchor, indexed)
        expected = np.asarray(entity["center"])
        expected[0 if axis == "x" else 1] += entity["radius"]*(1 if sense == "maximum" else -1)
        np.testing.assert_allclose(point, expected, atol=1e-12)
        assert derived_anchor_domain_conditions(anchor, indexed).min() >= -1e-9
    else:
        with pytest.raises(ValueError, match="finite support domain"):
            derived_anchor_point(anchor, indexed)


def test_arc_extremum_moves_with_current_center_and_radius_without_free_auxiliary_station():
    first = arc(); second = arc(radius=45., center=(12., -9.))
    np.testing.assert_equal(derived_anchor_point(extremum(), {first["id"]: first}), [40., 70.])
    np.testing.assert_equal(derived_anchor_point(extremum(), {second["id"]: second}), [12., 36.])
    assert derived_anchor_equalities(extremum(), {second["id"]: second}).size == 0


@pytest.mark.parametrize("change,measurement_axis,match", [
    ({"x": 100.}, "y", "literal coordinates"),
    ({"station": 100.}, "y", "literal coordinates"),
    ({"axis": "z"}, "y", "axis"),
    ({"axis": "y"}, "x", "axis"),
    ({}, None, "axis"),
    ({"entity_id": "missing"}, "y", "unknown"),
    ({"entity_id": "../file"}, "y", "identifier"),
    ({"sense": "nearest"}, "y", "sense"),
])
def test_bad_schema_or_unprojected_distance_cannot_certify_derived_anchor(change, measurement_axis, match):
    entity = arc()
    with pytest.raises(ValueError, match=match):
        validate_derived_anchor({**extremum(), **change}, {entity["id"]: entity}, measurement_axis)


def test_wrong_entity_type_and_degenerate_support_are_rejected():
    entity = line(start=(5., 8.), end=(5., 8.))
    with pytest.raises(ValueError, match="finite support domain"):
        derived_anchor_point(level(), {entity["id"]: entity})
    with pytest.raises(ValueError, match="non-ARC"):
        validate_derived_anchor({**extremum(), "entity_id": entity["id"]}, {entity["id"]: entity}, "y")
    circular = arc(radius=0.)
    with pytest.raises(ValueError, match="positive radius"):
        validate_derived_anchor(extremum(), {circular["id"]: circular}, "y")


def raster_source(entities):
    image = Image.new("L", (500, 500), "white")
    draw = ImageDraw.Draw(image)
    transform = lambda points: np.asarray(points, float)*[4., -4.]+[80., 420.]
    for entity in entities:
        if entity["type"] == "LINE":
            points = transform([entity["start"], entity["end"]])
        else:
            center = np.asarray(entity["center"])
            first = math.atan2(*(np.asarray(entity["start"])-center)[::-1])
            last = math.atan2(*(np.asarray(entity["end"])-center)[::-1])
            sign = -1 if entity["clockwise"] else 1
            sweep = ((last-first)*sign) % (2*math.pi)
            angles = np.linspace(first, first+sign*sweep, 256)
            points = transform(center+entity["radius"]*np.c_[np.cos(angles), np.sin(angles)])
        draw.line([tuple(point) for point in points], fill="black", width=2)
    return np.asarray(image), transform


def test_source_levels_admit_actual_finite_stroke_without_nominal_or_coordinate_lock():
    entity = line(start=(0., 4.), end=(70., 4.)); gray, transform = raster_source([entity])
    graph = {"entities": [entity]}
    records = [{"id": "r1", "text": "70", "parsed": {"kind": "length", "nominal": 70.},
                "box": [[200, 350], [230, 350], [230, 370], [200, 370]]}]
    before = deepcopy(graph)
    observed = source_derived_locations(gray, records, graph, transform, 5., "y")
    assert len(observed) == 1
    assert observed[0]["anchor"] == level()
    assert observed[0]["evidence"]["source_attachment_interval_px"] == [80., 360.]
    assert not observed[0]["evidence"]["coordinates_locked"]
    assert observed[0]["evidence"]["required_axis_equality"]
    records[0]["text"] = "999"; records[0]["parsed"]["nominal"] = 999.
    assert source_derived_locations(gray, records, graph, transform, 5., "y") == observed
    assert graph == before
    assert source_derived_locations(np.full_like(gray, 255), records, graph, transform, 5., "y") == []


def test_parallel_ink_does_not_force_an_arbitrary_level():
    entity = line(start=(0., 4.), end=(70., 4.)); parallel = line(start=(0., 5.5), end=(70., 5.5))
    gray, transform = raster_source([entity, parallel])
    assert source_derived_locations(gray, [], {"entities": [entity]}, transform, 7., "y") == []


def test_proposal_endpoint_inside_drafting_gap_is_not_an_observed_finite_attachment():
    from contour_agent.constraint_binding import _derived_station_groups
    from contour_agent.dimension_evidence import _axis_lines
    entity = line(start=(0., 4.), end=(70., 4.))
    # Most of the proposed face is genuinely observed, but its last 10 pixels
    # lie in a gap. An unrelated extension begins at that fitted endpoint.
    observed_face = line(start=(0., 4.), end=(67.5, 4.))
    extension = line(start=(70., 4.), end=(75., 4.))
    gray, transform = raster_source([observed_face, extension])
    locations = source_derived_locations(gray, [], {"entities": [entity]}, transform, 4., "y")
    assert len(locations) == 1
    evidence = locations[0]["evidence"]
    assert evidence["proposal_finite_interval_px"] == [80., 360.]
    assert evidence["observed_source_stroke_interval_px"] == [80., 350.]
    assert evidence["source_attachment_interval_px"] == [80., 350.]
    assert not evidence["coordinates_locked"]
    # The prior fitted-only interval would attach x=360 directly. Clipped
    # source ownership now requires crossing the actual gap; its length is too
    # large a fraction of this short route, so the independent gate stays closed.
    groups = _derived_station_groups(locations, _axis_lines(gray, "x"), 1, 404., 370., 4.)
    assert groups == []


def test_curve_extremum_proposal_requires_observed_source_turn_with_uncertainty():
    entity = arc(); gray, transform = raster_source([entity]); graph = {"entities": [entity]}
    locations = source_derived_locations(gray, [], graph, transform, 5., "y")
    assert len(locations) == 1, locations
    location = locations[0]
    assert location["anchor"] == extremum()
    assert location["evidence"]["source_turn_depth_px"] > 1.5
    assert location["evidence"]["source_position_uncertainty_px"] >= 1.
    assert location["evidence"]["section_interpretation_not_resolved"]
    assert location["evidence"]["finite_arc_domain_checked"]
    assert source_derived_locations(np.full_like(gray, 255), [], graph, transform, 5., "y") == []
    # A fitted arc near an actual straight station is not proof of a turn.
    gray_line, _ = raster_source([line(start=(0., 70.), end=(80., 70.))])
    assert source_derived_locations(gray_line, [], graph, transform, 5., "y") == []


def test_arc_proposal_keeps_finite_domain_and_does_not_cross_missing_curve_interval():
    entity = arc(100., 180.); gray, transform = raster_source([entity])
    assert source_derived_locations(gray, [], {"entities": [entity]}, transform, 5., "y") == []


def test_rotated_source_frame_does_not_masquerade_as_axis_aligned_measurement():
    entity = line(start=(0., 4.), end=(70., 4.)); gray, _ = raster_source([entity])
    transform = lambda points: np.asarray(points, float) @ np.array([[3.9, .1], [.1, -3.9]])+[80., 420.]
    assert source_derived_locations(gray, [], {"entities": [entity]}, transform, 5., "y") == []

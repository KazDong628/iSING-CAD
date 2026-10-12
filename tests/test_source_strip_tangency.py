"""Line-width association must not fabricate source tangency or move geometry."""
import math

import cv2
import numpy as np
import pytest

from contour_agent.structural_evidence import (
    _bounded_source_ink_tangent_trace,
    _source_ink_tangent_trace,
)


def _horizontal(*, offset=6., thickness=5):
    ink = np.zeros((200, 260), np.uint8)
    cv2.line(ink, (20, 90), (235, 90), 255, thickness)
    points = np.c_[np.linspace(40., 180., 81), np.full(81, 90. + offset)]
    return ink > 0, points


def test_thick_source_strip_inside_band_has_measureable_tangent():
    ink, points = _horizontal()
    original = points.copy()
    assert not _bounded_source_ink_tangent_trace(ink, points, 4.)["verified"]
    measured = _source_ink_tangent_trace(ink, points, 4.)
    assert measured["verified"]
    assert measured["source_ink_strip_gap_budget_px"] == 4.
    assert measured["association_distance"] == "proposal_to_observed_ink_strip_edge"
    assert measured["stroke_center_displacement_is_not_geometry_permission"]
    assert "centre_displacement_budget_px" not in measured
    assert not measured["previous_center_band_measurement"]["verified"]
    assert measured["unambiguous_samples"] == 96
    assert np.allclose(measured["tangent_direction_px"], [1., 0.], atol=1e-9)
    assert np.array_equal(original, points)


@pytest.mark.parametrize("thickness,offset", [(2, 6.), (5, 8.), (11, 6.)])
def test_outside_or_excessively_wide_ink_stays_unresolved(thickness, offset):
    ink, points = _horizontal(thickness=thickness, offset=offset)
    measured = _source_ink_tangent_trace(ink, points, 4.)
    assert not measured["verified"]
    assert not measured["width_aware_strip_retry"]["verified"]


def test_strip_retry_does_not_select_nearest_of_two_source_runs():
    ink, points = _horizontal()
    # Another visible parallel stroke lies in the observation aperture. Choosing
    # the first/thicker/nearest strip would let the mask prior decide ownership.
    ink[102:104, 20:236] = True
    measured = _source_ink_tangent_trace(ink, points, 4.)
    assert not measured["verified"]
    retry = measured["width_aware_strip_retry"]
    assert all(row["rejected_samples"]["multiple_runs"] == 96
               for row in retry["window_attempts"])


def test_short_crossing_cannot_supply_finite_strip_tangent():
    ink, points = _horizontal()
    ink[:] = False
    ink[60:130, 85:89] = True
    measured = _source_ink_tangent_trace(ink, points, 4.)
    assert not measured["verified"]


def test_nearby_stroke_cannot_be_followed_after_association_drifts_away():
    ink, _ = _horizontal(thickness=2)
    x = np.linspace(40., 180., 81)
    points = np.c_[x, 90. + .25 * (x - 40.)]
    measured = _source_ink_tangent_trace(ink, points, 4.)
    assert not measured["verified"]
    attempts = measured["width_aware_strip_retry"]["window_attempts"]
    rejected = [row for row in attempts
                if row["reason"] == "source_strip_association_drifts_across_stroke"]
    assert rejected
    assert all(row["source_strip_lateral_offset_span_px"] > row["source_strip_observed_width_px"]
               for row in rejected)


def test_original_arc_ink_derivative_survives_mask_stroke_side_offset():
    ink = np.zeros((350, 350), np.uint8)
    theta = np.linspace(.2, 1.25, 161)
    direction = np.c_[np.cos(theta), np.sin(theta)]
    source = [100., 100.] + 140. * direction
    points = [100., 100.] + 146. * direction
    cv2.polylines(ink, [np.rint(source).astype(np.int32)], False, 255, 5)
    measured = _source_ink_tangent_trace(ink > 0, points, 4.)
    assert measured["verified"]
    assert measured["association_distance"] == "proposal_to_observed_ink_strip_edge"
    expected = np.array([-math.sin(theta[0]), math.cos(theta[0])])
    deviation = math.degrees(math.acos(np.clip(
        expected @ measured["tangent_direction_px"], -1., 1.)))
    assert deviation < 2.


def test_two_measured_strips_do_not_turn_source_corner_into_tangency():
    ink = np.zeros((350, 350), np.uint8)
    joint = np.array([170., 170.])
    directions = [np.array([-1., 0.]), np.array([math.cos(.3), math.sin(.3)])]
    paths = []
    for direction in directions:
        source = joint + np.linspace(0., 140., 121)[:, None] * direction
        cv2.polylines(ink, [np.rint(source).astype(np.int32)], False, 255, 5)
        paths.append(source + 6. * np.array([-direction[1], direction[0]]))
    measured = [_source_ink_tangent_trace(ink > 0, path, 4.) for path in paths]
    assert all(row["verified"] for row in measured)
    angle = math.degrees(math.acos(np.clip(-np.dot(
        measured[0]["tangent_direction_px"], measured[1]["tangent_direction_px"]), -1., 1.)))
    assert angle > 10.  # The unchanged 3-degree joint gate rejects this corner.

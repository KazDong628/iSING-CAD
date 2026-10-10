"""A wider measurement aperture must not widen geometric acceptance."""
import math

import cv2
import numpy as np

from contour_agent.structural_evidence import _ink_trace, _source_ink_tangent_trace


def _line(*, thickness=5, offset=0., span=100.):
    ink = np.zeros((180, 200), np.uint8)
    cv2.line(ink, (20, 80), (160, 80), 255, thickness)
    samples = np.c_[np.linspace(25., 25. + span, 65), np.full(65, 80. + offset)]
    return ink > 0, samples


def test_thick_stroke_is_measured_without_clipping_or_moving_centre_budget():
    ink, samples = _line(offset=2.)
    before = samples.copy()
    assert not _ink_trace(ink, samples, 4.)["verified"]
    result = _source_ink_tangent_trace(ink, samples, 4.)
    assert result["verified"]
    assert result["measurement_aperture_half_width_px"] == 8
    assert result["centre_displacement_budget_px"] == 4.
    assert result["independent_source_tangent_measurement"]
    assert not result["fitted_tangency_used_as_evidence"]
    assert result["previous_measurement"]["verified"] is False
    assert abs(result["tangent_direction_px"][1]) < 1e-10
    assert np.array_equal(samples, before)


def test_finite_short_source_line_uses_more_than_arbitrary_45_percent():
    ink, samples = _line(thickness=2, span=40.)
    assert _ink_trace(ink, samples, 4.)["reason"] == "insufficient_local_source_span"
    result = _source_ink_tangent_trace(ink, samples, 4.)
    assert result["verified"]
    assert 24 <= result["span_px"] <= 40 * .85


def test_stroke_outside_original_centre_band_stays_rejected():
    ink, samples = _line(thickness=2, offset=6.)
    result = _source_ink_tangent_trace(ink, samples, 4.)
    assert not result["verified"]
    assert all(row["rejected_samples"]["centre_outside_band"] for row in result["window_attempts"])


def test_multiple_source_strokes_do_not_select_nearest_run():
    ink = np.zeros((180, 200), np.uint8)
    for y in (77, 83):
        cv2.line(ink, (20, y), (160, y), 255, 1)
    samples = np.c_[np.linspace(25., 125., 65), np.full(65, 80.)]
    result = _source_ink_tangent_trace(ink > 0, samples, 4.)
    assert not result["verified"]
    assert all(row["rejected_samples"]["multiple_runs"] == 96 for row in result["window_attempts"])


def test_empty_source_does_not_use_seed_as_tangent_evidence():
    ink, samples = _line(offset=2.)
    result = _source_ink_tangent_trace(np.zeros_like(ink), samples, 4.)
    assert not result["verified"]
    assert all(row["rejected_samples"]["no_ink"] == 96 for row in result["window_attempts"])


def test_aperture_border_clipping_is_not_accepted():
    ink = np.zeros((40, 200), bool)
    ink[2:6] = True
    samples = np.c_[np.linspace(25., 125., 65), np.full(65, 3.)]
    assert not _source_ink_tangent_trace(ink, samples, 4.)["verified"]


def test_unresolved_very_short_stroke_is_not_extrapolated():
    ink, samples = _line(thickness=2, span=20.)
    result = _source_ink_tangent_trace(ink, samples, 4.)
    assert not result["verified"]
    assert not result["window_attempts"]


def test_existing_successful_source_measurement_is_preserved():
    ink, samples = _line(thickness=2)
    previous = _ink_trace(ink, samples, 4.)
    assert previous["verified"]
    assert _source_ink_tangent_trace(ink, samples, 4.) == previous


def test_side_directions_are_measured_without_forcing_tangency():
    ink = np.zeros((260, 260), np.uint8)
    first = np.c_[np.linspace(120., 35., 65), np.full(65, 130.)]
    angle = math.radians(15.)
    second = [120., 130.] + np.linspace(0., 85., 65)[:, None] * [math.cos(angle), math.sin(angle)]
    for points in (first, second):
        cv2.polylines(ink, [np.rint(points).astype(np.int32)], False, 255, 5)
    measured = [_source_ink_tangent_trace(ink > 0, points, 4.) for points in (first, second)]
    assert all(row["verified"] for row in measured)
    deviation = math.degrees(math.acos(np.clip(-np.dot(
        measured[0]["tangent_direction_px"], measured[1]["tangent_direction_px"]), -1., 1.)))
    assert deviation > 10.  # The unchanged 3-degree relation gate must reject it.

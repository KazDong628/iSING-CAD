import cv2
import numpy as np

from contour_agent.structural_evidence import _connected_source_ink_tangent_trace
from contour_agent.constraint_binding import _derived_extension_supports


def test_continuous_boundary_is_distinguished_from_intermittent_hatch_runs():
    ink = np.zeros((220, 260), np.uint8)
    cv2.line(ink, (20, 100), (235, 100), 255, 2)
    for x in (55, 85, 115, 145):
        cv2.line(ink, (x, 108), (x+5, 108), 255, 1)
    points = np.c_[np.linspace(40., 200., 100), np.full(100, 100.)]
    result = _connected_source_ink_tangent_trace(ink > 0, points, 4.)
    assert result['verified']
    assert result['eligible_track_count'] == 1
    assert result['unambiguous_samples'] >= 72
    assert np.allclose(result['tangent_direction_px'], [1., 0.], atol=1e-8)


def test_parallel_track_is_not_discarded_by_fitted_path_distance():
    ink = np.zeros((220, 260), np.uint8)
    cv2.line(ink, (20, 100), (235, 100), 255, 2)
    cv2.line(ink, (20, 108), (235, 108), 255, 1)
    points = np.c_[np.linspace(40., 200., 100), np.full(100, 100.)]
    result = _connected_source_ink_tangent_trace(ink > 0, points, 4.)
    assert not result['verified']
    assert all(row['eligible_track_count'] == 2 for row in result['window_attempts'])


def test_blank_source_never_provides_connected_track():
    points = np.c_[np.linspace(40., 200., 100), np.full(100, 100.)]
    assert not _connected_source_ink_tangent_trace(np.zeros((220,260), bool), points, 4.)['verified']


def test_short_hatch_cannot_supply_required_boundary_coverage():
    ink = np.zeros((220,260), np.uint8)
    cv2.line(ink, (55,100), (65,100), 255, 2)
    points = np.c_[np.linspace(40.,200.,100), np.full(100,100.)]
    assert not _connected_source_ink_tangent_trace(ink > 0, points, 4.)['verified']


def _stroke(lo, hi, width):
    return dict(lo=lo, hi=hi, cross=100., thickness=width, span=hi-lo+1.)


def test_observed_pen_width_can_explain_a_bounded_drafting_clearance():
    segments = [_stroke(20.,100.,3.),_stroke(132.,400.,8.)]
    result = _derived_extension_supports(segments,100.,30.,300.,4.)
    assert len(result) == 1
    assert result[0]['observed_gap_px'] == 32.
    assert result[0]['observed_segments'] == segments
    assert result[0]['nominal_used'] is False
    assert not _derived_extension_supports([_stroke(20.,100.,3.),_stroke(132.,400.,3.)],100.,30.,300.,4.)


def test_broad_or_noncollinear_strokes_cannot_authorize_gap():
    assert not _derived_extension_supports([_stroke(20.,100.,3.),_stroke(132.,400.,20.)],100.,30.,300.,4.)
    assert not _derived_extension_supports([_stroke(20.,100.,3.),_stroke(132.,400.,8.)],100.,30.,150.,4.)

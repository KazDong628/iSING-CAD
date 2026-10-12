import cv2
import numpy as np
import pytest

from contour_agent.constraint_binding import _arrowhead_evidence


@pytest.mark.parametrize("angle", [0., 23., 45., 117.])
def test_thin_rasterized_shaft_is_not_an_arrow(angle):
    gray = np.full((260, 260), 255, np.uint8)
    direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    tip = np.array([130., 130.])
    tail = tip - 95. * direction
    cv2.line(gray, tuple(np.rint(tail).astype(int)), tuple(tip.astype(int)),
             0, 1, cv2.LINE_AA)
    assert _arrowhead_evidence(gray, tip, direction, 125., 4., tip[None]) is None


@pytest.mark.parametrize("angle", [0., 23., 45., 117.])
def test_filled_arrow_survives_a_contour_crossing_at_its_tip(angle):
    gray = np.full((260, 260), 255, np.uint8)
    direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    normal = np.array([-direction[1], direction[0]])
    tip = np.array([130., 130.])
    tail = tip - 95. * direction
    cv2.line(gray, tuple(np.rint(tail).astype(int)), tuple(tip.astype(int)), 0, 2)
    base = tip - 30. * direction
    cv2.fillConvexPoly(gray, np.rint([tip, base + 7. * normal, base - 7. * normal]).astype(np.int32), 0)
    cv2.line(gray, tuple(np.rint(tip - 35. * normal).astype(int)),
             tuple(np.rint(tip + 35. * normal).astype(int)), 0, 2)
    arrow = _arrowhead_evidence(gray, tip, direction, 125., 4., tip[None])
    assert arrow is not None
    assert arrow["body_persistence"]["verified"]
    assert np.linalg.norm(np.array(arrow["tip_px"]) - tip) <= 10.

"""A short restored LINE needs fresh source association, not old crop identity."""
from copy import deepcopy

import numpy as np
import pytest

from contour_agent.constraint_binding import _short_restored_angle_line
from contour_agent.annotation_line_support import _source_nominal_line


def _case():
    samples = np.linspace([20., 60.], [35., 90.], 16)
    entity = {"type": "LINE", "angle_support_evidence": {
        "record_id": "a", "method": "locally_verified_angle_ink_interval_topology_restore",
        "ground_truth_used": False, "requires_angle_binding_and_solve": True,
        "source_interval_endpoints_px": samples[[0, -1]].tolist(),
        "source_line": {"start_px": [10., 40.], "end_px": [40., 100.]}}}
    return entity, samples, np.array([[10., 40.], [50., 120.]])


def test_fresh_longer_stroke_does_not_erase_a_short_finite_restoration():
    entity, samples, fresh = _case()
    original = deepcopy(entity)
    assert _short_restored_angle_line(entity, "a", samples, fresh, 4.)
    assert entity == original


def test_two_radius_restoration_rechecks_finite_line_not_stale_hough_endpoints():
    entity, samples, fresh = _case()
    witness = entity["angle_support_evidence"]
    witness["method"] = "verified_two_radius_joint_line_topology_restore"
    # Source detection can return a different finite extent on the same ink.
    # This only chooses a short inspection window; it cannot bind an angle.
    witness["source_line"] = {"start_px": [5., 30.], "end_px": [45., 110.]}
    assert _short_restored_angle_line(entity, "a", samples, fresh, 4.)
    assert not _short_restored_angle_line(entity, "a", samples, fresh + [40., 0.], 4.)
    witness["source_interval_endpoints_px"][0][0] += 20.
    assert not _short_restored_angle_line(entity, "a", samples, fresh, 4.)


@pytest.mark.parametrize("defect", ["curve", "wrong_record", "stale_interval", "different_stroke",
                                      "too_short", "untrusted", "no_solve", "unknown_method"])
def test_restoration_metadata_cannot_authorize_an_unrelated_or_unproven_line(defect):
    entity, samples, fresh = _case()
    witness = entity["angle_support_evidence"]
    if defect == "curve": entity["type"] = "ARC"
    elif defect == "wrong_record": witness["record_id"] = "other"
    elif defect == "stale_interval": witness["source_interval_endpoints_px"][0][0] += 20.
    elif defect == "different_stroke": fresh[:, 0] += 40.
    elif defect == "too_short": samples = np.linspace([20., 60.], [21., 62.], 16); witness["source_interval_endpoints_px"] = samples[[0, -1]].tolist()
    elif defect == "untrusted": witness["ground_truth_used"] = True
    elif defect == "no_solve": witness["requires_angle_binding_and_solve"] = False
    elif defect == "unknown_method": witness["method"] = "provider_claim"
    assert not _short_restored_angle_line(entity, "a", samples, fresh, 4.)


def test_angle_annotation_constructs_the_direction_instead_of_copying_fitted_slope():
    import math
    points = np.linspace([0., 0.], [11., 20.], 32)
    observation = {"nominal": 24., "reference_axis": "vertical", "evidence": {
        "method": "source_angular_arrows_axis_and_straight_support_v1",
        "reference_arrow": {"verified": True}, "target_arrow": {"verified": True}}}
    line = _source_nominal_line(points, observation, 2., fixed_start=True)
    assert line is not None and line["start"] == [0., 0.]
    assert math.degrees(math.atan2(line["end"][0], line["end"][1])) == pytest.approx(24., abs=1e-10)
    assert line["fit_error_px"] <= 2.
    assert _source_nominal_line(points, observation, .1, fixed_start=True) is None
    assert _source_nominal_line(points, observation, 2., fixed_start=True, fixed_end=True) is None
    observation["evidence"]["target_arrow"]["verified"] = False
    assert _source_nominal_line(points, observation, 2.) is None

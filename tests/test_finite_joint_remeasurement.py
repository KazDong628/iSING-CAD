import math
import cv2
import numpy as np
from contour_agent.structural_evidence import remeasure_finite_source_joint


def tracks(angle):
    ink=np.zeros((260,280),np.uint8);joint=np.array([130.,130.])
    paths=[joint+np.linspace(0,85,129)[:,None]*[-1.,0.],
           joint+np.linspace(0,85,129)[:,None]*[math.cos(angle),math.sin(angle)]]
    for path in paths:cv2.polylines(ink,[np.rint(path).astype(np.int32)],False,255,3)
    return ink>0,paths


def test_longer_original_window_still_rejects_a_designed_corner():
    ink,paths=tracks(math.radians(15.))
    result=remeasure_finite_source_joint(ink,paths,4.)
    assert result['observed_deviation_degrees']>10.
    assert not result['verified'] and result['tolerance_degrees']==3.
    assert all(row['span_px']<=min(96.,85*.85)+1e-8 for row in result['sides'])


def test_remeasurement_does_not_use_the_supplied_path_as_ink():
    ink,paths=tracks(0.)
    result=remeasure_finite_source_joint(np.zeros_like(ink),paths,4.)
    assert not result['verified'] and result['observed_deviation_degrees'] is None


def test_independent_finite_straight_ink_is_certified_without_nominals():
    ink,paths=tracks(0.)
    result=remeasure_finite_source_joint(ink,paths,4.)
    assert result['verified'] and result['observed_deviation_degrees']<1e-5
    assert not result['fitted_tangency_used_as_evidence']
    assert not result['nominal_dimensions_used']

import math
import cv2
import numpy as np

from contour_agent.source_circle_tangent import circular_ink_derivative
from contour_agent.structural_evidence import remeasure_finite_source_joint


def arc(radius=100., angle=.9):
    stations=np.linspace(1.5,radius*angle,96)
    points=np.c_[radius*np.sin(stations/radius),radius*(1-np.cos(stations/radius))]
    return points,stations


def test_known_circle_derivative_uses_ink_geometry_without_nominals():
    points,stations=arc()
    result=circular_ink_derivative(points,stations,np.full(96,3.))
    assert result['verified']
    assert abs(result['observed_radius_px']-100.)<1e-6
    assert np.allclose(result['tangent_direction_px'],[1.,0.],atol=1e-8)
    assert not result['observed_radius_is_dimensional_evidence']
    assert not result['nominal_dimensions_used']


def test_nearly_straight_ink_cannot_identify_a_circle():
    points,stations=arc(radius=100000.,angle=.0009)
    assert not circular_ink_derivative(points,stations,np.full(96,3.))['verified']


def test_non_circular_curvature_transition_is_not_certified():
    points,stations=arc()
    points[:,1]+=12*(stations/stations.max())**3
    assert not circular_ink_derivative(points,stations,np.full(96,3.))['verified']


def circle_joint(deviation):
    joint=np.array([150.,150.]);radius=80.
    stations=np.linspace(0.,100.,129);theta=stations/radius
    left=joint+np.c_[-radius*np.sin(theta),radius*(1-np.cos(theta))]
    right=joint+np.c_[radius*np.sin(theta),radius*(1-np.cos(theta))]@np.array([
        [math.cos(deviation),math.sin(deviation)],[-math.sin(deviation),math.cos(deviation)]])
    ink=np.zeros((300,320),np.uint8)
    for path in (left,right):cv2.polylines(ink,[np.rint(path).astype(np.int32)],False,255,3)
    return ink>0,[left,right]


def test_rasterized_circular_joint_keeps_three_degree_gate():
    ink,paths=circle_joint(0.)
    result=remeasure_finite_source_joint(ink,paths,4.)
    assert result['verified'] and result['observed_deviation_degrees']<3.
    assert result['tolerance_degrees']==3.
    ink,paths=circle_joint(math.radians(15.))
    result=remeasure_finite_source_joint(ink,paths,4.)
    assert not result['verified'] and result['observed_deviation_degrees']>10.

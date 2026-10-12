from copy import deepcopy
import math

import cv2
import numpy as np
import pytest

from contour_agent.constraint_binding import _annotated_line_arc_tangent


def fixture(corner=0.):
    joint=np.array([100.,100.]);radius=100.
    theta=math.radians(corner)-math.pi/2
    centre=joint-radius*np.array([math.cos(theta),math.sin(theta)])
    arc=lambda s:centre+radius*np.c_[np.cos(theta+np.asarray(s)/radius),
                                    np.sin(theta+np.asarray(s)/radius)]
    line=np.array([[80.,100.],[100.,100.]])
    curved=arc(np.linspace(0.,130.,150))
    raster=np.zeros((330,330),np.uint8)
    cv2.line(raster,(80,100),(100,100),255,2)
    cv2.polylines(raster,[np.round(curved).astype(np.int32)],False,255,2)
    entities=[dict(id="l",type="LINE",start_node="a",end_node="j"),
              dict(id="c",type="ARC",start_node="j",end_node="b")]
    observation=dict(record_id="a24",verified=True,nominal=24.,
        source_line=dict(start_px=line[0].tolist(),end_px=line[1].tolist()),
        evidence=dict(method="source_angular_arrows_axis_and_straight_support_v1",
                      reference_arrow=dict(verified=True),target_arrow=dict(verified=True)),
        target_candidates=[dict(entity_id="l",entity_type="LINE",whole_line_supported=True)])
    def path(side,s):
        return arc(s) if side>0 else joint+np.c_[-np.asarray(s),np.zeros(len(s))]
    return raster>0,entities,"j",[line,curved],path,3.,1.,[observation]


def test_short_straight_side_uses_independently_measured_stroke_not_nominal():
    args=fixture()
    result=_annotated_line_arc_tangent(*args)
    assert result["verified"]
    assert result["observed_deviation_degrees"]<3.
    assert result["nominal_angle_used_for_measurement"] is False
    assert result["straight_source_direction_px"]==[1.,0.]
    # The intentionally different nominal cannot turn a source corner tangent.
    args[-1][0]["nominal"]=80.
    assert _annotated_line_arc_tangent(*args)["verified"]


def test_real_corner_does_not_pass_same_three_degree_gate():
    result=_annotated_line_arc_tangent(*fixture(9.))
    assert not result["verified"]
    assert result["reason"]=="source_junction_is_not_tangent"


@pytest.mark.parametrize("missing",["arrows","whole_line","unique","ink","side"])
def test_missing_independent_source_or_ambiguous_ownership_rejects(missing):
    args=list(fixture());args[-1]=deepcopy(args[-1])
    if missing=="arrows":args[-1][0]["evidence"]["target_arrow"]["verified"]=False
    elif missing=="whole_line":args[-1][0]["target_candidates"][0]["whole_line_supported"]=False
    elif missing=="unique":args[-1].append(deepcopy(args[-1][0]))
    elif missing=="ink":args[0]=np.zeros_like(args[0])
    elif missing=="side":args[4]=lambda side,s:np.c_[100.-np.asarray(s),np.full(len(s),100.)]
    assert not _annotated_line_arc_tangent(*args)["verified"]

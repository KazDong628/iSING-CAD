"""Symbolic sections move with solved geometry and retain finite domains."""
from copy import deepcopy
import math

import numpy as np
import pytest

from contour_agent.section_measurements import (
    validate_section_anchor, section_anchor_point, section_anchor_equalities,
    section_anchor_domain_conditions,
)


def line(identifier,start,end):
    return {"id":identifier,"type":"LINE","start":list(start),"end":list(end)}


def arc(start=0.,end=180.,*,clockwise=False,center=(0.,0.),radius=5.):
    def point(angle):
        return [center[0]+radius*math.cos(math.radians(angle)),
                center[1]+radius*math.sin(math.radians(angle))]
    return {"id":"curve","type":"ARC","center":list(center),"radius":radius,
            "start":point(start),"end":point(end),"clockwise":clockwise}


def entities(primary=None):
    primary=arc() if primary is None else primary
    return {"base_h":line("base_h",(-10.,0.),(10.,0.)),
            "base_v":line("base_v",(0.,-10.),(0.,10.)),"curve":primary}


def anchor(*,axis="y",offset=3.,direction=1,branch="positive"):
    return {"type":"section_curve_intersection","entity_id":"curve","axis":axis,"branch":branch,
        "station":{"type":"source_offset_from_line_intersection","entities":["base_h","base_v"],
            "axis":"x" if axis=="y" else "y","direction":direction,
            "source_record_id":"r041","source_offset":offset,"reference":True}}


def test_station_is_symbolic_and_intersection_updates_after_support_and_circle_move():
    indexed=entities();a=anchor();before=deepcopy((a,indexed))
    identity,supports=validate_section_anchor(a,indexed,"y")
    assert supports=={"curve","base_h","base_v"}
    assert "r041" in identity
    np.testing.assert_allclose(section_anchor_point(a,indexed),[3.,4.],atol=1e-12)
    assert section_anchor_equalities(a,indexed).size==0
    assert section_anchor_domain_conditions(a,indexed).min()>=0.
    assert (a,indexed)==before
    indexed["base_v"]=line("base_v",(1.,-10.),(1.,10.))
    indexed["curve"]=arc(center=(2.,1.))
    np.testing.assert_allclose(section_anchor_point(a,indexed),[4.,1.+math.sqrt(21.)],atol=1e-12)
    assert a==before[0]  # No stored source/raster station can become stale.


def test_equivalent_support_order_has_one_identity_without_mutating_input():
    indexed=entities();first=anchor();second=deepcopy(first)
    second["station"]["entities"].reverse()
    assert validate_section_anchor(first,indexed,"y")==validate_section_anchor(second,indexed,"y")
    np.testing.assert_allclose(section_anchor_point(first,indexed),section_anchor_point(second,indexed))
    assert second["station"]["entities"]==["base_v","base_h"]


def test_unknown_circle_center_can_be_solved_from_symbolic_sections_with_fixed_radius():
    from scipy.optimize import least_squares
    anchors=[anchor(offset=3.),anchor(offset=4.)]
    expected=[math.sqrt(25.-(station-1.)**2)-2. for station in (3.,4.)]

    def residual(center):
        indexed=entities(arc(center=center))
        return [section_anchor_point(a,indexed)[1]-value
                for a,value in zip(anchors,expected)]

    solution=least_squares(residual,[0.,0.],bounds=([-.5,-3.],[1.5,1.]),
                           ftol=1e-12,xtol=1e-12,gtol=1e-12)
    np.testing.assert_allclose(solution.x,[1.,-2.],atol=1e-8)
    assert np.linalg.norm(residual(solution.x))<1e-10
    indexed=entities(arc(center=solution.x))
    assert indexed["curve"]["radius"]==5.
    assert all(section_anchor_domain_conditions(a,indexed).min()>=0. for a in anchors)


def test_other_axis_and_signed_station_use_correct_quadratic_root():
    indexed=entities(arc(-90.,90.))
    np.testing.assert_allclose(section_anchor_point(anchor(axis="x"),indexed),[4.,3.],atol=1e-12)
    indexed["curve"]=arc(90.,270.)
    np.testing.assert_allclose(section_anchor_point(anchor(axis="x",branch="negative"),indexed),[-4.,3.],atol=1e-12)
    indexed["curve"]=arc()
    np.testing.assert_allclose(section_anchor_point(anchor(direction=-1),indexed),[-3.,4.],atol=1e-12)


def test_wrong_root_of_same_circle_is_not_an_intersection_of_the_finite_arc():
    a=anchor(branch="negative");indexed=entities()
    with pytest.raises(ValueError,match="finite support domain"):
        section_anchor_point(a,indexed)
    np.testing.assert_allclose(section_anchor_point(a,indexed,allow_infeasible=True),[3.,-4.],atol=1e-12)
    assert section_anchor_domain_conditions(a,indexed).min()<0.


@pytest.mark.parametrize("start,end,clockwise,offset,valid",[
    (330.,30.,False,2.,True), (30.,330.,True,2.,True),
    (30.,330.,False,2.,False), (330.,30.,False,3.,False),
    (0.,0.,False,3.,False),
])
def test_clockwise_major_and_wraparound_finite_arc_branches(start,end,clockwise,offset,valid):
    a=anchor(axis="x",offset=offset);indexed=entities(arc(start,end,clockwise=clockwise))
    if valid:
        assert np.isfinite(section_anchor_point(a,indexed)).all()
        assert section_anchor_domain_conditions(a,indexed).min()>=-1e-9
    else:
        with pytest.raises(ValueError,match="finite support domain"):
            section_anchor_point(a,indexed)


def test_nondegenerate_arc_endpoint_is_in_the_finite_domain():
    a=anchor();indexed=entities(arc(0.,math.degrees(math.atan2(4.,3.))))
    np.testing.assert_allclose(section_anchor_point(a,indexed),indexed["curve"]["end"],atol=1e-12)
    assert section_anchor_domain_conditions(a,indexed).min()>=-1e-9


@pytest.mark.parametrize("offset",[5.,5.-1e-8,6.])
def test_tangent_near_tangent_and_nonintersecting_trials_have_negative_domain(offset):
    a=anchor(offset=offset);indexed=entities()
    with pytest.raises(ValueError,match="near-tangent"):
        section_anchor_point(a,indexed)
    assert np.isfinite(section_anchor_point(a,indexed,allow_infeasible=True)).all()
    assert section_anchor_domain_conditions(a,indexed).min()<0.
    # Schema validation may admit a finite but imperfect initial proposal;
    # it never substitutes for the mandatory strict final domain certificate.
    assert validate_section_anchor(a,indexed,"y")[1]=={"curve","base_h","base_v"}


def test_finite_line_section_updates_and_rejects_its_infinite_extension():
    indexed=entities(line("curve",(-5.,-5.),(5.,5.)));a=anchor(branch="linear")
    np.testing.assert_allclose(section_anchor_point(a,indexed),[3.,3.],atol=1e-12)
    indexed["curve"]=line("curve",(-5.,-3.),(5.,7.))
    np.testing.assert_allclose(section_anchor_point(a,indexed),[3.,5.],atol=1e-12)
    a["station"]["source_offset"]=5.
    np.testing.assert_allclose(section_anchor_point(a,indexed),[5.,7.],atol=1e-12)
    a["station"]["source_offset"]=6.
    with pytest.raises(ValueError,match="finite support domain"):
        section_anchor_point(a,indexed)
    assert np.isfinite(section_anchor_point(a,indexed,allow_infeasible=True)).all()
    assert section_anchor_domain_conditions(a,indexed).min()<0.


@pytest.mark.parametrize("end",[(0.,10.),(.001,10.)])
def test_target_line_parallel_or_near_parallel_to_section_is_not_unique(end):
    indexed=entities(line("curve",(0.,-10.),end));a=anchor(branch="linear")
    with pytest.raises(ValueError,match="parallel"):
        section_anchor_point(a,indexed)
    assert np.isfinite(section_anchor_point(a,indexed,allow_infeasible=True)).all()
    assert section_anchor_domain_conditions(a,indexed).min()<0.


@pytest.mark.parametrize("end",[(10.,0.),(10.,.001)])
def test_parallel_or_near_parallel_corner_supports_cannot_define_initial_station(end):
    indexed=entities();indexed["base_v"]=line("base_v",(-10.,0.),end);a=anchor()
    with pytest.raises(ValueError,match="parallel"):
        validate_section_anchor(a,indexed,"y")
    with pytest.raises(ValueError,match="parallel"):
        section_anchor_point(a,indexed)
    assert np.isfinite(section_anchor_point(a,indexed,allow_infeasible=True)).all()
    assert section_anchor_domain_conditions(a,indexed).min()<0.


@pytest.mark.parametrize("change",[
    {"x":3.}, {"axis":"z"}, {"axis":"x"}, {"entity_id":"missing"},
    {"entity_id":"../file"}, {"branch":"nearest"}, {"branch":"linear"},
])
def test_bad_anchor_schema_axis_and_reference_cannot_enter_measurements(change):
    with pytest.raises(ValueError):
        validate_section_anchor({**anchor(),**change},entities(),"y")


@pytest.mark.parametrize("change",[
    {"x":3.}, {"source_offset":math.nan}, {"source_offset":math.inf},
    {"source_offset":True}, {"source_offset":0.}, {"source_offset":-3.},
    {"source_record_id":"../secret"}, {"direction":True}, {"direction":0},
    {"axis":"y"}, {"entities":["base_h","missing"]},
    {"entities":["base_h","base_h"]}, {"entities":["base_h","curve"]},
    {"reference":"true"}, {"fit_station_px":100.},
])
def test_bad_station_literals_ownership_shapes_and_nonfinite_offsets_rejected(change):
    a=anchor();a["station"].update(change)
    with pytest.raises(ValueError):
        validate_section_anchor(a,entities(),"y")


@pytest.mark.parametrize("change",[
    {"center":[math.nan,0.]}, {"start":[math.inf,0.]}, {"radius":math.nan},
    {"radius":0.}, {"clockwise":"false"}, {"end":[0.,0.]},
])
def test_invalid_arc_geometry_is_not_hidden_by_allow_infeasible(change):
    indexed=entities();indexed["curve"].update(change)
    with pytest.raises(ValueError):
        section_anchor_point(anchor(),indexed,allow_infeasible=True)


def test_unprojected_distance_cannot_use_a_section_anchor():
    with pytest.raises(ValueError,match="measurement axis"):
        validate_section_anchor(anchor(),entities(),None)

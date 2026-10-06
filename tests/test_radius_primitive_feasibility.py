from copy import deepcopy
import math

import numpy as np
import pytest

from contour_agent.constraint_binding import _radius_primitive_feasibility, radius_binding_coverage


def radius_case(*, sweep=math.pi, source_radius=40., reverse=False):
    angles = np.linspace(-sweep/2, sweep/2, 401)
    # Both curves meet the same independently observed arrow tip. Their centres
    # can differ, and no fitted/nominal-radius ratio is an acceptance criterion.
    center = np.array([200., 200.])
    source_center = center + [40-source_radius, 0.]
    observed = source_center+source_radius*np.c_[np.cos(angles), np.sin(angles)]
    points = center+40*np.c_[np.cos(angles), np.sin(angles)]
    if reverse:
        points = points[::-1]
    entity = {"id": "g0", "type": "ARC", "center": center.tolist(), "radius": 40.,
              "start": points[0].tolist(), "end": points[-1].tolist(), "clockwise": reverse}
    model = {"extraction": {"raw_polyline_px": observed.tolist()},
             "curve_fit": {"total_deviation_budget_px": 4.}}
    graph = {"units": "mm", "entities": [entity]}
    leader = {"arrowhead_verified": True, "arrowhead": {"tip_px": [240., 200.], "direction_px": [-1., 0.]}}
    return entity, model, graph, leader


def check(case, nominal):
    entity, model, graph, leader = case
    return _radius_primitive_feasibility(entity, nominal, leader, model, graph, np.asarray, 9.)


def test_verified_tip_does_not_bind_incompatible_whole_primitive():
    case = radius_case()
    original = deepcopy(case)
    diagnostic = check(case, 176.)
    assert diagnostic["status"] == "requires_topology_repartition"
    assert diagnostic["conservative_max_residual_px"] > diagnostic["source_deviation_budget_px"] == 4.
    assert not diagnostic["passed"] and not diagnostic["nominal_used_to_rank"]
    assert case == original
    entity, _, graph, leader = case
    inventory = {"all_records": [{"id": "r0", "text": "R176", "parsed": {"kind": "radius", "nominal": 176.}}],
                 "all_candidates": [{"record_id": "r0", "kind": "radius", "entities": [entity["id"]],
                                     "local_reliable": False,
                                     "evidence": {"leader": leader, "whole_primitive_radius": diagnostic}}]}
    coverage = radius_binding_coverage(inventory, graph)
    assert coverage["confirmed_arrow_records"] == ["r0"]
    assert coverage["required_count"] == 1 and coverage["bound_count"] == 0
    assert coverage["required_mappings"][0]["status"] == "requires_topology_repartition"
    assert coverage["unresolved"][0]["reason"] == "requires_topology_repartition"
    assert not coverage["all_confirmed_arrows_bound"]


def test_source_supported_nominal_is_allowed_despite_large_fitted_radius_ratio():
    case = radius_case(sweep=math.radians(20.), source_radius=80.)
    diagnostic = check(case, 80.)
    assert diagnostic["passed"] and diagnostic["status"] == "feasible_pending_joint_solve"
    assert diagnostic["source_deviation_budget_px"] == 4.  # Never adopt the wider candidate band of 9.
    assert diagnostic["conservative_max_residual_px"] < 1.
    assert diagnostic["endpoints_may_move_in_joint_solve"]


def test_arrow_without_source_interval_cannot_certify_whole_arc():
    case = radius_case()
    case[1].pop("extraction")
    diagnostic = check(case, 40.)
    assert not diagnostic["passed"] and not diagnostic["checked"]
    assert diagnostic["status"] == "source_boundary_observation_unavailable"


def test_nominal_circle_must_remain_radial_to_verified_arrow():
    case = radius_case()
    case[3]["arrowhead"]["direction_px"] = [0., 1.]
    diagnostic = check(case, 40.)
    assert diagnostic["conservative_max_residual_px"] < 1.
    assert diagnostic["nominal_circle_arrow_alignment"] < .88
    assert diagnostic["status"] == "requires_topology_repartition"


@pytest.mark.parametrize("nominal", [40., 176.])
def test_source_interval_direction_and_closed_seam_do_not_change_gate(nominal):
    forward, backward = radius_case(), radius_case(reverse=True)
    # Re-index the closed observation so the arc interval crosses its seam.
    ring = np.asarray(backward[1]["extraction"]["raw_polyline_px"])
    ring = np.vstack([ring, ring[0]])
    backward[1]["extraction"]["raw_polyline_px"] = np.roll(ring[:-1], 150, axis=0).tolist()
    first, second = check(forward, nominal), check(backward, nominal)
    assert first["passed"] == second["passed"]
    assert abs(first["conservative_max_residual_px"]-second["conservative_max_residual_px"]) < .6

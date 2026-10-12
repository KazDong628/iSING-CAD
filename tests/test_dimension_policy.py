import copy
import pytest

from contour_agent.ocr import parse_dimension
from contour_agent.dimension_policy import source_dimension_target, tolerance_interval_check
from contour_agent.constraint_binding import _constraint
from contour_agent.parametric_solver import solve_parametric, native_measurement_checks
from test_parametric_solver import rectangle
from test_exact_source_nominal_solver import native


@pytest.mark.parametrize("text,target,limits",[
    ("135^{+5}_{+2}",138.5,(137.,140.)),("178±2",178.,(176.,180.)),
    ("19^{+3}_{0}",20.5,(19.,22.)),("⌀840^{+6}_{+4}",845.,(844.,846.)),
])
def test_tolerance_target_is_declared_source_midpoint(text,target,limits):
    parsed=parse_dimension(text)
    value,policy=source_dimension_target(parsed)
    assert value==target and (policy["minimum"],policy["maximum"])==limits
    assert tolerance_interval_check(policy,target)["passed"]
    assert not tolerance_interval_check(policy,limits[0]-1.)["passed"]
    changed=copy.deepcopy(policy);changed["construction_target"]+=1.
    with pytest.raises(ValueError):tolerance_interval_check(changed,target)


def test_radius_keeps_exact_nominal_policy():
    assert source_dimension_target(parse_dimension("R40"))==(40.,None)
    assert source_dimension_target(parse_dimension("R40±1"))==(40.,None)


def test_binding_solver_and_native_readback_keep_asymmetric_tolerance():
    candidate={"id":"c0","record_id":"r0","kind":"distance_x","value":10.,
        "entities":[],"nodes":["v0","v1"],"source_dimension_semantics":parse_dimension("10^{+2}_{+1}")}
    constraint=_constraint(candidate,"ocr_api_binding")
    assert candidate["value"]==10. and constraint["value"]==11.5
    graph=rectangle()
    solution=solve_parametric(graph,[constraint])
    assert solution["accepted"] and solution["constraints"][0]["absolute_residual"]<1e-7
    audit=native_measurement_checks(solution["entities"],solution["constraints"],dxf_document=native(solution["entities"]))
    assert audit["passed"] and audit["dxf_readback_performed"]
    assert audit["checks"][0]["dxf"]["source_tolerance_interval"]["passed"]
    wrong=copy.deepcopy(constraint);wrong["value"]=10.
    with pytest.raises(ValueError):solve_parametric(graph,[wrong])

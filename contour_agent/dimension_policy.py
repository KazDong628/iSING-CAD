"""Declared drawing-tolerance semantics, independent of reference CAD.

Linear tolerance midpoints are an explicit construction convention, not a
measured nominal, a hidden template or a reference-accuracy certificate.
Radius annotations keep the user's exact nominal-R contract.
"""
from __future__ import annotations
import math


def source_dimension_target(parsed):
    nominal=parsed.get("nominal")
    if type(nominal) not in (int,float) or not math.isfinite(nominal) or nominal<=0:
        raise ValueError("invalid_source_dimension_nominal")
    lower,upper=parsed.get("lower_deviation"),parsed.get("upper_deviation")
    if lower is None and upper is None:
        return float(nominal),None
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (lower,upper)) or lower>upper:
        raise ValueError("invalid_source_dimension_tolerance")
    if parsed.get("kind")=="radius":
        return float(nominal),None
    target=float(nominal+(lower+upper)/2)
    if target<=0:raise ValueError("nonpositive_source_tolerance_target")
    return target,{"policy":"source_tolerance_midpoint","nominal":float(nominal),
        "lower_deviation":float(lower),"upper_deviation":float(upper),
        "minimum":float(nominal+lower),"maximum":float(nominal+upper),
        "construction_target":target,"reference_geometry_used":False}


def tolerance_interval_check(metadata,actual):
    if metadata is None:return None
    if not isinstance(metadata,dict) or metadata.get("policy")!="source_tolerance_midpoint" or metadata.get("reference_geometry_used") is not False:
        raise ValueError("invalid_source_tolerance_policy")
    target,expected=source_dimension_target({"kind":"length","nominal":metadata.get("nominal"),
        "lower_deviation":metadata.get("lower_deviation"),"upper_deviation":metadata.get("upper_deviation")})
    if expected!=metadata:raise ValueError("source_tolerance_interval_or_target_changed")
    value=abs(float(actual))
    return {"policy":metadata["policy"],"nominal":metadata["nominal"],"actual_magnitude":value,
        "minimum":metadata["minimum"],"maximum":metadata["maximum"],"construction_target":target,
        "passed":math.isfinite(value) and metadata["minimum"]-1e-7<=value<=metadata["maximum"]+1e-7,
        "reference_geometry_used":False}

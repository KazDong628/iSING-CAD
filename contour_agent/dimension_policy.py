"""Declared drawing-tolerance semantics, independent of reference CAD.

Legacy linear tolerance midpoints remain an explicit construction convention.
New interval dimensions declare feasible bounds, never a midpoint equality.
Radius annotations keep the user's exact nominal-R contract in both policies.
"""
from __future__ import annotations
import math
from numbers import Real


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


def source_dimension_interval(parsed):
    """Return the source nominal and its admissible interval, without a target.

    In particular 20 +3/0 permits an exactly nominal thickness of 20. A
    midpoint is neither an extra source annotation nor a required equation.
    Unilateral tolerances can exclude the nominal; callers must then enforce
    the bounds rather than silently treating this return value as an equality.
    """
    if not isinstance(parsed,dict):
        raise ValueError("invalid_source_dimension_nominal")
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
    minimum,maximum=float(nominal+lower),float(nominal+upper)
    if not all(math.isfinite(value) for value in (minimum,maximum)) or maximum<=0:
        raise ValueError("invalid_source_dimension_interval")
    return float(nominal),{"policy":"source_tolerance_interval","nominal":float(nominal),
        "lower_deviation":float(lower),"upper_deviation":float(upper),
        "minimum":minimum,"maximum":maximum,"reference_geometry_used":False}


def _validated_tolerance_metadata(metadata):
    if (not isinstance(metadata,dict) or metadata.get("policy") not in
            {"source_tolerance_midpoint","source_tolerance_interval"} or
            metadata.get("reference_geometry_used") is not False):
        raise ValueError("invalid_source_tolerance_policy")
    midpoint=metadata["policy"]=="source_tolerance_midpoint"
    fields={"nominal","lower_deviation","upper_deviation","minimum","maximum"}
    if midpoint:
        fields.add("construction_target")
    if (set(metadata)!=fields|{"policy","reference_geometry_used"} or
            any(type(metadata[key]) not in (int,float) or not math.isfinite(metadata[key]) for key in fields)):
        raise ValueError("invalid_source_tolerance_metadata")
    parsed={"kind":"length","nominal":metadata["nominal"],
        "lower_deviation":metadata["lower_deviation"],"upper_deviation":metadata["upper_deviation"]}
    _,expected=(source_dimension_target(parsed) if midpoint else source_dimension_interval(parsed))
    if expected!=metadata:
        raise ValueError("source_tolerance_interval_or_target_changed")
    return expected


def tolerance_interval_check(metadata,actual):
    if metadata is None:return None
    expected=_validated_tolerance_metadata(metadata)
    if not isinstance(actual,Real) or isinstance(actual,bool):
        raise ValueError("invalid_source_dimension_actual")
    value=abs(float(actual))
    result={"policy":metadata["policy"],"nominal":metadata["nominal"],"actual_magnitude":value,
        "minimum":metadata["minimum"],"maximum":metadata["maximum"],
        "passed":math.isfinite(value) and metadata["minimum"]-1e-7<=value<=metadata["maximum"]+1e-7,
        "reference_geometry_used":False}
    if "construction_target" in expected:
        result["construction_target"]=expected["construction_target"]
    return result


def tolerance_interval_residual(metadata,actual):
    """Signed distance to the feasible magnitude interval; zero everywhere inside.

    This is an optimization helper, not a substitute for final bound checks.
    The signed projection of a measurement keeps its own direction contract.
    """
    check=tolerance_interval_check(metadata,actual)
    if check is None or not math.isfinite(check["actual_magnitude"]):
        raise ValueError("invalid_source_dimension_actual")
    value=check["actual_magnitude"]
    return value-min(max(value,check["minimum"]),check["maximum"])

"""Finite curve sections at source-declared symbolic measurement stations.

Only an OCR offset relative to two current LINE supports defines the station.
No raster coordinate, fitted datum coordinate, reference CAD or free auxiliary
vertex enters the anchor. Source-arrow and OCR ownership are caller contracts.
"""
from __future__ import annotations

import math
from numbers import Real
import re

import numpy as np


SECTION_MIN_INTERSECTION_ANGLE_DEG = 1.
SECTION_DOMAIN_TOLERANCE = 1e-9
_MIN_CROSSING = math.sin(math.radians(SECTION_MIN_INTERSECTION_ANGLE_DEG))
_MIN_LENGTH = 1e-9
_COORDINATE_LIMIT = 1e9


def _identifier(value):
    if not isinstance(value,str) or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,119}",value) is None:
        raise ValueError("Section anchor needs a bounded identifier")
    return value


def _number(value):
    if (not isinstance(value,Real) or isinstance(value,bool) or
            not math.isfinite(float(value)) or abs(value)>_COORDINATE_LIMIT):
        raise ValueError("Section anchor/support values must be finite bounded real numbers")
    return float(value)


def _point(value):
    if not isinstance(value,(list,tuple,np.ndarray)) or len(value)!=2:
        raise ValueError("Section support must have two-dimensional finite points")
    return np.asarray([_number(component) for component in value],float)


def _entity(identifier,entities):
    _identifier(identifier)
    if not isinstance(entities,dict) or identifier not in entities:
        raise ValueError("Section anchor references an unknown entity")
    row=entities[identifier]
    if not isinstance(row,dict) or row.get("type") not in {"LINE","ARC"}:
        raise ValueError("Section anchor requires LINE or ARC supports")
    if "id" in row and row["id"]!=identifier:
        raise ValueError("Section support identifier disagrees with its entity mapping")
    _point(row.get("start"));_point(row.get("end"))
    if row["type"]=="ARC":
        center=_point(row.get("center"));radius=_number(row.get("radius"))
        if radius<=0:
            raise ValueError("Section ARC requires a positive radius")
        if "clockwise" in row and type(row["clockwise"]) is not bool:
            raise ValueError("Section ARC traversal must be boolean")
        if min(np.linalg.norm(_point(row[key])-center) for key in ("start","end"))<=_MIN_LENGTH:
            raise ValueError("Section ARC endpoints cannot coincide with its center")
    return row


def _schema(anchor,entities,measurement_axis=None):
    required={"type","entity_id","axis","station","branch"}
    if (not isinstance(anchor,dict) or set(anchor)!=required or
            anchor.get("type")!="section_curve_intersection"):
        raise ValueError("Section anchor cannot contain literal coordinates or extra fields")
    axis=anchor["axis"]
    if not isinstance(axis,str) or axis not in {"x","y"} or (measurement_axis is not None and axis!=measurement_axis):
        raise ValueError("Section anchor axis must match its projected measurement axis")
    station=anchor["station"]
    station_fields={"type","entities","axis","direction","source_record_id","source_offset","reference"}
    if (not isinstance(station,dict) or set(station)!=station_fields or
            station.get("type")!="source_offset_from_line_intersection"):
        raise ValueError("Section station needs the exact source-offset schema, without literal coordinates")
    if not isinstance(station["axis"],str) or station["axis"] not in {"x","y"} or station["axis"]==axis:
        raise ValueError("Section station axis must be perpendicular to its measurement axis")
    if type(station["direction"]) is not int or station["direction"] not in {-1,1}:
        raise ValueError("Section station requires a signed source direction")
    _identifier(station["source_record_id"])
    offset=_number(station["source_offset"])
    if offset<=0:
        raise ValueError("Section source offset must be positive")
    if type(station["reference"]) is not bool:
        raise ValueError("Section source reference flag must be boolean")
    support_ids=station["entities"]
    if not isinstance(support_ids,list) or len(support_ids)!=2:
        raise ValueError("Section station requires two distinct LINE supports")
    for identifier in support_ids:
        _identifier(identifier)
    if len(set(support_ids))!=2:
        raise ValueError("Section station requires two distinct LINE supports")
    supports=[_entity(identifier,entities) for identifier in support_ids]
    if any(row["type"]!="LINE" for row in supports):
        raise ValueError("Section station intersection requires two LINE supports")
    primary=_entity(anchor["entity_id"],entities)
    branches={"positive","negative"} if primary["type"]=="ARC" else {"linear"}
    if not isinstance(anchor["branch"],str) or anchor["branch"] not in branches:
        raise ValueError("Section branch does not match its finite curve type")
    return primary,supports


def _support_station(anchor,supports,*,allow_infeasible):
    first,second=supports
    a,b=_point(first["start"]),_point(second["start"])
    u,v=_point(first["end"])-a,_point(second["end"])-b
    lengths=[float(np.linalg.norm(vector)) for vector in (u,v)]
    cross=float(u[0]*v[1]-u[1]*v[0])
    normalized=cross/max(lengths[0]*lengths[1],1e-30)
    conditions=[lengths[0]-_MIN_LENGTH,lengths[1]-_MIN_LENGTH,abs(normalized)-_MIN_CROSSING]
    if not allow_infeasible and (min(lengths)<=_MIN_LENGTH or abs(normalized)<_MIN_CROSSING):
        raise ValueError("Section station supports are parallel, near-parallel or degenerate")
    floor=max(lengths[0]*lengths[1]*1e-12,1e-20)
    denominator=math.copysign(max(abs(cross),floor),cross) if allow_infeasible else cross
    offset=b-a
    corner=a+float(offset[0]*v[1]-offset[1]*v[0])/denominator*u
    k=0 if anchor["station"]["axis"]=="x" else 1
    station=float(corner[k])+anchor["station"]["direction"]*float(anchor["station"]["source_offset"])
    return station,conditions


def _section_geometry(anchor,primary,station,*,allow_infeasible):
    j=0 if anchor["axis"]=="x" else 1
    k=1-j
    start,end=_point(primary["start"]),_point(primary["end"])
    if primary["type"]=="LINE":
        delta=end-start;length=float(np.linalg.norm(delta))
        component=float(delta[k])
        transverse=abs(component)/max(length,1e-30)
        floor=max(length*1e-12,1e-20)
        denominator=math.copysign(max(abs(component),floor),component) if allow_infeasible else component
        if not allow_infeasible and (length<=_MIN_LENGTH or transverse<_MIN_CROSSING):
            raise ValueError("Section LINE is parallel or near-parallel to its section")
        parameter=(station-float(start[k]))/denominator
        point=start+parameter*delta;point[k]=station
        conditions=[length-_MIN_LENGTH,transverse-_MIN_CROSSING,parameter,1.-parameter]
    else:
        center=_point(primary["center"]);radius=float(primary["radius"])
        difference=station-float(center[k])
        discriminant=radius*radius-difference*difference
        normalized=discriminant/(radius*radius)
        if not allow_infeasible and normalized<_MIN_CROSSING**2:
            raise ValueError("Section ARC has no transverse intersection or is near-tangent")
        # Optimizer trial steps outside the circle still receive finite values.
        # Their negative discriminant/domain conditions MUST remain active.
        root=math.sqrt(max(discriminant,radius*radius*1e-16) if allow_infeasible else discriminant)
        point=center.copy();point[k]=station
        point[j]+=root*(1. if anchor["branch"]=="positive" else -1.)
        first=math.atan2(*(start-center)[::-1]);last=math.atan2(*(end-center)[::-1])
        sign=-1. if primary.get("clockwise",False) else 1.
        sweep=(sign*(last-first))%(2*math.pi)
        angle=math.atan2(*(point-center)[::-1]);midpoint=first+sign*sweep/2.
        angular_delta=(angle-midpoint+math.pi)%(2*math.pi)-math.pi
        conditions=[radius-_MIN_LENGTH,sweep-_MIN_LENGTH,normalized-_MIN_CROSSING**2,
                    sweep/2.-abs(angular_delta)]
    return point,conditions


def validate_section_anchor(anchor,entities,measurement_axis):
    """Return symbolic identity/support union; source OCR ownership is external.

    An initial target outside its finite domain can move during optimization.
    Parallel theoretical-corner supports cannot define an initial station.
    Final certificates must additionally call the strict point evaluator.
    """
    if not isinstance(measurement_axis,str) or measurement_axis not in {"x","y"}:
        raise ValueError("Section anchor requires a projected measurement axis")
    _,supports=_schema(anchor,entities,measurement_axis)
    _support_station(anchor,supports,allow_infeasible=False)
    station=anchor["station"]
    identity=(anchor["type"],anchor["entity_id"],anchor["axis"],anchor["branch"],
        *sorted(station["entities"]),station["axis"],station["direction"],
        station["source_record_id"],float(station["source_offset"]),station["reference"])
    return identity,set(station["entities"])|{anchor["entity_id"]}


def section_anchor_point(anchor,entities,*,allow_infeasible=False):
    """Recompute the intersection from current supports, circle and source offset."""
    primary,supports=_schema(anchor,entities)
    station,station_conditions=_support_station(anchor,supports,allow_infeasible=allow_infeasible)
    point,curve_conditions=_section_geometry(anchor,primary,station,allow_infeasible=allow_infeasible)
    if not np.isfinite(point).all():
        raise ValueError("Section measurement point is nonfinite")
    if not allow_infeasible:
        if np.max(np.abs(point))>_COORDINATE_LIMIT:
            raise ValueError("Section measurement point exceeds the bounded coordinate range")
        if (primary["type"]=="ARC" and curve_conditions[1]<=0 or
                min(station_conditions+curve_conditions)<-SECTION_DOMAIN_TOLERANCE):
            raise ValueError("Section measurement point is outside its finite support domain")
    return point


def section_anchor_equalities(anchor,entities):
    """The analytic intersection eliminates auxiliary station/curve variables."""
    _schema(anchor,entities)
    return np.empty(0,float)


def section_anchor_domain_conditions(anchor,entities):
    """Nonnegative support conditioning, transverse crossing and finite domains.

    Conditions mix physical length, dimensionless conditioning and radians,
    like the existing derived-anchor contracts. They are mandatory optimizer
    inequalities; allow_infeasible trial values cannot certify their own domain.
    """
    primary,supports=_schema(anchor,entities)
    station,station_conditions=_support_station(anchor,supports,allow_infeasible=True)
    _,curve_conditions=_section_geometry(anchor,primary,station,allow_infeasible=True)
    return np.asarray(station_conditions+curve_conditions,float)

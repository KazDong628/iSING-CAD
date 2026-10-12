"""Joint source-only dimensional solving for an already declared CAD topology.

This module reads no images, references, training labels or network resources.
Shared nodes enforce closure; signed chord-normal arc centers preserve each
declared minor/major branch. Bound radius annotations are eliminated as exact
constants with explicit diameter feasibility. Source coordinates and supported
source-curve samples are soft priors, never locks or dimensional evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
import time

import numpy as np
from scipy.optimize import OptimizeResult, least_squares, minimize
from scipy.linalg import qr
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from .topology import _primitive_distance
from .vectorize import _sample_entities, _line_entities, assess_fit_quality
from .relation_contract import (STRICT_TANGENT_CERT_TOLERANCE_DEG,
                                STRICT_RELATION_ENDPOINT_TOLERANCE)
from .measurement_anchors import (derived_anchor_point, validate_derived_anchor,
                                  derived_anchor_equalities, derived_anchor_domain_conditions)

LINEAR_TOLERANCE_MM = .05
ANGLE_TOLERANCE_DEG = .1
STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG = 1e-7
STRICT_MEASUREMENT_CERT_TOLERANCE_MM = 1e-7
MIN_MEASUREMENT_INTERSECTION_ANGLE_DEG = 1.
MAX_DISPLACEMENT_FRACTION = .10
MAX_ENTITIES = 256
MAX_CONSTRAINTS = 1024
SOURCE_CURVE_PRIOR_WEIGHT = .3
SOURCE_BUDGET_SEARCH_ROUNDS = 24
_KINDS = {"radius", "distance_x", "distance_y", "distance", "angle", "horizontal", "vertical", "tangent"}
_DIMENSIONAL = {"radius", "distance_x", "distance_y", "distance"}
_SOURCES = {"ocr_local_binding", "ocr_api_binding", "source_geometry"}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite real number")
    if abs(float(value)) > 1e9: raise ValueError(f"{label} exceeds the bounded CAD coordinate/value range")
    return float(value)


def _identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,119}", value):
        raise ValueError(f"{label} must be a bounded identifier")
    return value


def _point(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2: raise ValueError(f"{label} must have two coordinates")
    return [_number(value[0],label), _number(value[1],label)]


def _sweep(entity):
    start = math.atan2(entity["start"][1]-entity["center"][1], entity["start"][0]-entity["center"][0])
    end = math.atan2(entity["end"][1]-entity["center"][1], entity["end"][0]-entity["center"][0])
    return ((start-end) if entity["clockwise"] else (end-start)) % (2*math.pi)


def _measurement_intersection_condition(anchor, entities):
    first,second=(entities[eid] for eid in anchor["entities"])
    u=np.asarray(first["end"],float)-np.asarray(first["start"],float)
    v=np.asarray(second["end"],float)-np.asarray(second["start"],float)
    lengths=float(np.linalg.norm(u)),float(np.linalg.norm(v))
    return float(u[0]*v[1]-u[1]*v[0])/max(lengths[0]*lengths[1],1e-30),lengths


def _measurement_anchor_point(anchor, nodes, entities, *, allow_ill_conditioned=False):
    """Evaluate a source-declared anchor from the CURRENT geometric unknowns.

    Intersections refer to infinite straight supports, not extra CAD entities
    or coordinate locks. Source-arrow ownership remains the caller's evidence
    responsibility. Regularized values are only for optimizer trial steps;
    initial and final checks always reject ill-conditioned intersections.
    """
    if anchor["type"]=="node":return np.asarray(nodes[anchor["node_id"]],float)
    if anchor["type"] in {"support_level","curve_extremum","curve_center","common_curve_center"}:
        return derived_anchor_point(anchor,entities,allow_infeasible=allow_ill_conditioned)
    first,second=(entities[eid] for eid in anchor["entities"])
    a=np.asarray(first["start"],float);b=np.asarray(second["start"],float)
    u=np.asarray(first["end"],float)-a;v=np.asarray(second["end"],float)-b
    condition,lengths=_measurement_intersection_condition(anchor,entities)
    minimum=math.sin(math.radians(MIN_MEASUREMENT_INTERSECTION_ANGLE_DEG))
    if not allow_ill_conditioned and (min(lengths)<=1e-9 or abs(condition)<minimum):
        raise ValueError("Measurement support intersection is parallel, near-parallel or degenerate")
    denominator=float(u[0]*v[1]-u[1]*v[0])
    if allow_ill_conditioned:
        floor=max(lengths[0]*lengths[1]*1e-12,1e-20)
        denominator=math.copysign(max(abs(denominator),floor),denominator)
    offset=b-a;point=a+(offset[0]*v[1]-offset[1]*v[0])/denominator*u
    if not allow_ill_conditioned and (not np.isfinite(point).all() or np.max(np.abs(point))>1e9):
        raise ValueError("Measurement support intersection exceeds the bounded coordinate range")
    return point


def _validate_measurement_anchors(row, node_map, entity_map):
    anchors=row.get("anchors")
    if not isinstance(anchors,list) or len(anchors)!=2 or row["nodes"]:
        raise ValueError("Typed distances require two ordered anchors and an empty nodes list")
    supports=set();identities=[]
    for anchor in anchors:
        if not isinstance(anchor,dict):raise ValueError("Measurement anchor must be an object")
        if anchor.get("type")=="node":
            if set(anchor)!={"type","node_id"}:raise ValueError("Node measurement anchors only contain type and node_id")
            node_id=_identifier(anchor["node_id"],"measurement node reference")
            if node_id not in node_map:raise ValueError("Measurement anchor references an unknown node")
            identities.append(("node",node_id))
        elif anchor.get("type")=="line_intersection":
            if set(anchor)!={"type","entities"}:raise ValueError("Intersection anchors cannot contain literal coordinates")
            eids=anchor["entities"]
            if not isinstance(eids,list) or len(eids)!=2 or len(set(eids))!=2:
                raise ValueError("Measurement intersection requires two distinct LINE references")
            for eid in eids:
                _identifier(eid,"measurement LINE reference")
                if eid not in entity_map or entity_map[eid]["type"]!="LINE":
                    raise ValueError("Measurement intersection references unknown or non-LINE support")
            anchor["entities"]=sorted(eids);supports.update(eids)
            identities.append(("line_intersection",*anchor["entities"]))
        elif anchor.get("type") in {"support_level","curve_extremum","curve_center","common_curve_center"}:
            identity,references=validate_derived_anchor(anchor,entity_map,
                "x" if row["kind"]=="distance_x" else "y" if row["kind"]=="distance_y" else None)
            supports.update(references);identities.append(identity)
        else:raise ValueError("Unsupported measurement anchor type")
        _measurement_anchor_point(anchor,node_map,entity_map,
            allow_ill_conditioned=anchor["type"] in {"support_level","curve_extremum","curve_center","common_curve_center"})
    if identities[0]==identities[1]:raise ValueError("Measurement anchors must be distinct")
    if set(row["entities"])!=supports:raise ValueError("Measurement entities must equal its intersection support union")
    row["entities"]=sorted(supports)
    return tuple(identities)


def _validate_inputs(graph, constraints):
    if not isinstance(graph,dict) or graph.get("units") not in {"mm","pixel"}:
        raise ValueError("Graph units must be explicitly mm or pixel")
    nodes=copy.deepcopy(graph.get("nodes")); entities=copy.deepcopy(graph.get("entities"))
    if not isinstance(nodes,list) or not isinstance(entities,list) or not 2 <= len(entities) <= MAX_ENTITIES:
        raise ValueError(f"An ordered closed graph with 2..{MAX_ENTITIES} LINE/ARC entities is required")
    if len(nodes) < 2 or len(nodes) > MAX_ENTITIES: raise ValueError("Graph node count is outside the bounded topology limit")
    indexed={}
    for node in nodes:
        if not isinstance(node,dict): raise ValueError("Each node must be an object")
        node_id=_identifier(node.get("id"),"node id")
        if node_id in indexed: raise ValueError("Duplicate node id")
        node["x"]=_number(node.get("x"),"node.x");node["y"]=_number(node.get("y"),"node.y")
        if node.get("source_px") is not None:node["source_px"]=_point(node["source_px"],"source_px")
        indexed[node_id]=node
    entity_map={};used_nodes=set()
    for entity in entities:
        if not isinstance(entity,dict): raise ValueError("Each entity must be an object")
        entity_id=_identifier(entity.get("id"),"entity id")
        if entity_id in entity_map:raise ValueError("Duplicate entity id")
        if entity.get("type") not in {"LINE","ARC"}:raise ValueError("Only LINE and ARC topology is supported")
        a,b=entity.get("start_node"),entity.get("end_node")
        if a not in indexed or b not in indexed:raise ValueError("Entity references an unknown node")
        if a==b:raise ValueError("A primitive needs distinct endpoint nodes; full circles require two arcs")
        used_nodes.update((a,b))
        for key,node_id in (("start",a),("end",b)):
            supplied=_point(entity.get(key),f"entity.{key}")
            actual=[indexed[node_id]["x"],indexed[node_id]["y"]]
            if math.dist(supplied,actual)>1e-6:raise ValueError("Entity endpoint differs from its shared graph node")
            entity[key]=supplied
        if math.dist(entity["start"],entity["end"])<1e-9:raise ValueError("A primitive has a degenerate chord")
        if entity["type"]=="ARC":
            entity["center"]=_point(entity.get("center"),"arc.center")
            entity["radius"]=_number(entity.get("radius"),"arc.radius")
            if entity["radius"]<=0:raise ValueError("Arc radius must be positive")
            if not isinstance(entity.get("clockwise"),bool):raise ValueError("Arc clockwise must be an explicit boolean")
            if not 1e-9 < _sweep(entity) < 2*math.pi-1e-9:raise ValueError("Degenerate/full-turn arcs are not supported")
        entity_map[entity_id]=entity
    if used_nodes != set(indexed):raise ValueError("Unused graph nodes are not allowed")
    for index,entity in enumerate(entities):
        if entity["end_node"] != entities[(index+1)%len(entities)]["start_node"]:
            raise ValueError("Entity order must be a closed shared-node chain")
    if not isinstance(constraints,list) or len(constraints)>MAX_CONSTRAINTS:raise ValueError("Constraints must be a bounded list")
    prepared=[];identities=set();semantic=set();records={}
    for original in constraints:
        if not isinstance(original,dict):raise ValueError("Each constraint must be an object")
        row=copy.deepcopy(original);cid=_identifier(row.get("id"),"constraint id")
        if cid in identities:raise ValueError("Duplicate constraint id")
        identities.add(cid)
        kind=row.get("kind");source=row.get("source")
        if kind not in _KINDS or source not in _SOURCES:raise ValueError("Unknown constraint kind or source")
        eids=row.get("entities",[]);nids=row.get("nodes",[])
        if not isinstance(eids,list) or not isinstance(nids,list):raise ValueError("Constraint entity/node references must be lists")
        for item in eids+nids:_identifier(item,"constraint entity/node reference")
        if len(eids)!=len(set(eids)) or len(nids)!=len(set(nids)):
            raise ValueError("Constraint entity/node references must be unique lists")
        if any(item not in entity_map for item in eids) or any(item not in indexed for item in nids):raise ValueError("Constraint references an unknown entity/node")
        row.update(entities=eids,nodes=nids)
        anchor_identities=None
        if "anchors" in row:
            if kind not in {"distance","distance_x","distance_y"}:
                raise ValueError("Measurement anchors are only valid for distances")
            anchor_identities=_validate_measurement_anchors(row,
                {node_id:np.array([node["x"],node["y"]]) for node_id,node in indexed.items()},entity_map)
        radial_relation = None
        if row.get("source_station_relation") is not None:
            from .measurement_relations import radial_station_relation
            if kind not in {"distance_x", "distance_y"} or anchor_identities is None:
                raise ValueError("Source radial station equation requires typed projected anchors")
            radial_relation = radial_station_relation(row["source_station_relation"],
                measurement_axis="x" if kind == "distance_x" else "y")
            if row.get("record_id") == radial_relation["reference_record_id"]:
                raise ValueError("Source radial station reference must be a distinct OCR record")
        value=row.get("value")
        if value is not None:value=_number(value,"constraint.value")
        row["value"]=value
        if row.get("source_tolerance") is not None:
            from .dimension_policy import tolerance_interval_check
            audit=tolerance_interval_check(row["source_tolerance"],value)
            if kind not in {"distance","distance_x","distance_y","angle"} or abs(value)!=audit["construction_target"]:
                raise ValueError("Source tolerance must keep its declared exact construction target")
        record=row.get("record_id")
        if source!="source_geometry":
            _identifier(record,"dimension record_id")
        elif record is not None:raise ValueError("source_geometry must not impersonate an OCR record binding")
        if kind in _DIMENSIONAL:
            if graph["units"]!="mm" or source=="source_geometry" or value is None:
                raise ValueError("Physical dimensions require an mm graph, OCR binding source and record_id/value")
            if kind=="radius" and (len(eids)!=1 or entity_map[eids[0]]["type"]!="ARC" or nids):
                raise ValueError("Radius binds exactly one ARC and no nodes")
            if kind!="radius" and anchor_identities is None and len(nids)!=2:raise ValueError("Distances require two ordered nodes")
            if kind in {"radius","distance"} and value<=0:raise ValueError("Radius/distance must be positive")
        elif kind in {"horizontal","vertical"}:
            if value not in (None,0.):raise ValueError("Horizontal/vertical relation value must be null or zero")
            if len(nids)==2:
                pass
            elif not nids and len(eids)==1 and entity_map[eids[0]]["type"]=="LINE":
                row["nodes"]=[entity_map[eids[0]]["start_node"],entity_map[eids[0]]["end_node"]]
            else:raise ValueError("Axis relation needs two nodes or one LINE")
        elif kind=="angle" and row.get("reference_axis") is not None:
            # A drawing angle to an independently evidenced axis constrains
            # one straight support. Never interpret an ARC tangent as that
            # straight feature, or create fictitious CAD reference nodes.
            if row["reference_axis"] not in {"horizontal","vertical"}:
                raise ValueError("Unknown angle reference_axis")
            if source=="source_geometry" or len(eids)!=1 or nids or entity_map[eids[0]]["type"]!="LINE":
                raise ValueError("Axis angle requires an annotated single LINE and no nodes")
            if row.get("angle_mode","unsigned")!="unsigned" or value is None or not 0<=value<=90:
                raise ValueError("Axis angle requires an unsigned acute angle in degrees")
        elif kind=="tangent" or (kind=="angle" and len(nids)!=3):
            if len(eids)!=2:raise ValueError("Tangent/entity angle requires two entities")
            shared=set((entity_map[eids[0]]["start_node"],entity_map[eids[0]]["end_node"])) & set((entity_map[eids[1]]["start_node"],entity_map[eids[1]]["end_node"]))
            if nids:
                if len(nids)!=1 or nids[0] not in shared:raise ValueError("Joint node must be shared by both entities")
            elif len(shared)==1:row["nodes"]=[next(iter(shared))]
            elif kind=="tangent" or len(shared)>1 or any(entity_map[e]["type"]=="ARC" for e in eids):
                raise ValueError("A unique shared joint is required; ambiguous joints need an explicit node")
        if kind=="tangent" and value not in (None,0.):raise ValueError("Tangent relation value must be null or zero")
        if kind=="angle":
            if value is None:raise ValueError("Angle requires a finite value in degrees")
            mode=row.get("angle_mode","unsigned")
            if mode not in {"unsigned","signed"}:raise ValueError("angle_mode must be unsigned or signed")
            if not (-180<=value<=180) or mode=="unsigned" and value<0:raise ValueError("Angle value is outside its declared angular range")
            row["angle_mode"]=mode
        # The same OCR record cannot silently become two independent dimensions.
        if record is not None and kind in _DIMENSIONAL|{"angle"}:
            if record in records:raise ValueError("A dimension record_id is reused; resolve binding multiplicity first")
            records[record]=cid
        canonical_entities=tuple(row["entities"]);canonical_nodes=tuple(row["nodes"]);canonical_value=value
        if radial_relation is not None:
            canonical_value=(value-radial_relation["offset"])/radial_relation["coefficient"]
        if kind in {"distance","distance_x","distance_y"}:
            canonical_nodes=(anchor_identities if anchor_identities is not None else
                             tuple(("node",node_id) for node_id in canonical_nodes))
        if kind in {"horizontal","vertical","distance"}:canonical_nodes=tuple(sorted(canonical_nodes))
        if kind in {"distance_x","distance_y"} and canonical_nodes[0]>canonical_nodes[1]:
            canonical_nodes=canonical_nodes[::-1];canonical_value=-canonical_value
        if kind=="tangent":canonical_entities=tuple(sorted(canonical_entities))
        if kind=="angle" and row["angle_mode"]=="unsigned":
            canonical_entities=tuple(sorted(canonical_entities))
            if len(canonical_nodes)==3 and canonical_nodes[0]>canonical_nodes[2]:canonical_nodes=canonical_nodes[::-1]
        if kind!="angle" and row.get("reference_axis") is not None:
            raise ValueError("reference_axis is only valid for an annotated angle")
        signature=(kind,canonical_entities,canonical_nodes,canonical_value,row.get("angle_mode"),row.get("reference_axis"))
        if signature in semantic:raise ValueError("Duplicate equivalent constraint")
        semantic.add(signature);prepared.append(row)
    return nodes,entities,prepared


def _tangent(entity, at_end):
    if entity["type"]=="LINE":direction=np.array(entity["end"])-entity["start"]
    else:
        radial=np.array(entity["end"] if at_end else entity["start"])-entity["center"]
        direction=np.array([-radial[1],radial[0]])*(-1 if entity["clockwise"] else 1)
    length=np.linalg.norm(direction)
    return direction/max(length,1e-12)


def _angle(first,second,signed):
    cross=first[0]*second[1]-first[1]*second[0]
    value=math.degrees(math.atan2(cross,float(np.dot(first,second))))
    return value if signed else abs(value)


def _constraint_value(row,nodes,entities,*,allow_ill_conditioned=False):
    kind=row["kind"];nids=row["nodes"];eids=row["entities"]
    if kind=="radius":return entities[eids[0]]["radius"]
    if kind.startswith("distance"):
        points=([_measurement_anchor_point(anchor,nodes,entities,allow_ill_conditioned=allow_ill_conditioned)
                 for anchor in row["anchors"]] if "anchors" in row else [nodes[node_id] for node_id in nids])
        delta=points[1]-points[0]
        measured=float(delta[0] if kind=="distance_x" else delta[1] if kind=="distance_y" else np.linalg.norm(delta))
        if row.get("source_station_relation") is not None:
            from .measurement_relations import radial_station_value
            return radial_station_value(measured,row["source_station_relation"],
                measurement_axis="x" if kind=="distance_x" else "y" if kind=="distance_y" else None)
        return measured
    if kind in {"horizontal","vertical"}:
        delta=nodes[nids[1]]-nodes[nids[0]]
        value=math.degrees(math.atan2(delta[1],delta[0]))-(90 if kind=="vertical" else 0)
        return (value+90)%180-90
    if kind=="angle" and row.get("reference_axis") is not None:
        entity=entities[eids[0]]
        delta=np.asarray(entity["end"])-np.asarray(entity["start"])
        # Both rays of a straight support have the same acute axis angle.
        opposite,adjacent=(abs(delta[0]),abs(delta[1])) if row["reference_axis"]=="vertical" else (abs(delta[1]),abs(delta[0]))
        return math.degrees(math.atan2(opposite,adjacent))
    if kind=="angle" and len(nids)==3:
        first=nodes[nids[0]]-nodes[nids[1]];second=nodes[nids[2]]-nodes[nids[1]]
    else:
        first_entity,second_entity=entities[eids[0]],entities[eids[1]]
        joint=nids[0] if nids else None
        first=_tangent(first_entity,first_entity["end_node"]==joint)
        second=_tangent(second_entity,second_entity["end_node"]==joint)
    return _angle(first,second,kind=="tangent" or row.get("angle_mode")=="signed")


def _constraint_residual(row,actual):
    residual=actual-(row["value"] or 0.)
    if row["kind"]=="tangent" or row["kind"]=="angle" and row.get("angle_mode")=="signed":
        residual=(residual+180)%360-180
    return residual


def _strict_source_nominal(row):
    """Explicitly admitted source nominals cannot trade error for a closer fit.

    Legacy callers which have not declared exact nominal enforcement retain
    their historical tolerance. Certificate modes are accepted because native
    readback must recheck output constraints without weakening the input rule.
    """
    return (row.get("source") in {"ocr_local_binding","ocr_api_binding"}
        and row.get("required") is True and row.get("nominal_source")=="source_ocr"
        and (row.get("enforcement") in {"exact","source_nominal_equality",
            "source_nominal_angle_equality","derived_measurement_anchor_equality",
            "linear_endpoint_direction_equality"} or
            row.get("kind")=="angle" and row.get("reference_axis") in {"horizontal","vertical"}
            and row.get("source_arrow_verified") is True))


def _strict_measurement_row(row):
    return (row.get("kind") in {"distance","distance_x","distance_y"} and
            ("anchors" in row or _strict_source_nominal(row)))


def _strict_general_angle_rows(constraints):
    # Axis relations have already passed source admission. They describe a
    # design relation, not a noisy angle observation that may be averaged
    # against the initial fit. Keep both node-pair and LINE forms exact.
    return [row for row in constraints if _strict_axis_relation(row) or
            row.get("kind")=="angle" and _strict_source_nominal(row)
            and row.get("reference_axis") is None]


def _strict_axis_relation(row):
    return row.get("kind") in {"horizontal", "vertical"} and row.get("source")=="source_geometry"


def _strict_general_angle_contract(nodes,entities,constraints):
    indexed={entity["id"]:entity for entity in entities};checks=[]
    for row in _strict_general_angle_rows(constraints):
        actual=_constraint_value(row,nodes,indexed);error=abs(_constraint_residual(row,actual))
        checks.append({"constraint_id":row["id"],"record_id":row.get("record_id"),
            "actual_angle_deg":actual,"absolute_error_deg":error,
            "passed":math.isfinite(error) and error<=STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG})
    return {"required_count":len(checks),"satisfied":all(row["passed"] for row in checks),
            "checks":checks,"source_binding_verified_by_solver":False}


def _strict_measurement_contract(nodes,entities,constraints,baseline_entities=None):
    indexed={entity["id"]:entity for entity in entities}
    baseline={entity["id"]:entity for entity in (baseline_entities or entities)}
    checks=[]
    for row in constraints:
        if not _strict_measurement_row(row):continue
        actual=None;residual=None;points=[];conditions=[];reason=None
        try:
            actual=_constraint_value(row,nodes,indexed)
            residual=_constraint_residual(row,actual)
            for anchor in row.get("anchors",[]):
                points.append(_measurement_anchor_point(anchor,nodes,indexed).tolist())
                if anchor["type"]=="line_intersection":
                    condition,_=_measurement_intersection_condition(anchor,indexed)
                    initial_condition,_=_measurement_intersection_condition(anchor,baseline)
                    preserved=condition*initial_condition>0
                    conditions.append({"entities":anchor["entities"],"normalized_cross":condition,
                        "initial_branch_preserved":preserved})
                    if not preserved:reason="intersection_support_orientation_branch_changed"
                elif anchor["type"] in {"support_level","curve_extremum","curve_center","common_curve_center"}:
                    equality=derived_anchor_equalities(anchor,indexed)
                    domain=derived_anchor_domain_conditions(anchor,indexed)
                    conditions.append({"anchor":copy.deepcopy(anchor),"support_equalities_mm":list(equality),
                        "support_domain_conditions":list(domain)})
        except (ValueError,ArithmeticError,KeyError):reason="invalid_or_ill_conditioned_measurement_anchor"
        checks.append({"constraint_id":row["id"],"record_id":row.get("record_id"),
            "actual":actual,"absolute_residual":abs(residual) if residual is not None else None,
            "tolerance_mm":STRICT_MEASUREMENT_CERT_TOLERANCE_MM,
            "derived_anchor_points":points,"intersection_conditions":conditions,"reason":reason,
            "passed":bool(reason is None and residual is not None and
                          abs(residual)<=STRICT_MEASUREMENT_CERT_TOLERANCE_MM)})
    return {"required_count":len(checks),"satisfied":all(row["passed"] for row in checks),
        "checks":checks,"literal_coordinates_used_as_constraints":False,
        "source_binding_verified_by_solver":False,"full_drawing_coverage_proven":False}


def native_measurement_checks(entities,constraints,*,dxf_document=None):
    """Recompute supplied linear measurements on exported geometry.

    No cached solver residual or accepted flag is evidence. The native entity
    order/types, coordinates, planar ARC traversal and shared node identities
    are independently checked before evaluating derived support intersections.
    Zero required measurements prove no drawing coverage.
    """
    from .relation_contract import _geometry, _native_geometry
    audit={"schema_version":"source-measurement-readback-v1","passed":False,
           "required_count":0,"satisfied_count":0,"checks":[],"issues":[],
           "dxf_readback_performed":False,"native_mapping_verified":False,
           "reference_geometry_used":False,"complete_measurement_coverage_verified":False}
    try:
        if not isinstance(entities,list) or not isinstance(constraints,list):
            raise ValueError("entities_and_constraints_must_be_lists")
        selected=[row for row in constraints if row.get("kind") in {"distance","distance_x","distance_y"}]
        audit["required_count"]=len(selected)
        model=[_geometry(entity) for entity in entities]
        values=[model]
        if dxf_document is not None:
            records=list(dxf_document.modelspace());audit["dxf_readback_performed"]=True
            if len(records)!=len(model):raise ValueError("native_entity_count_mismatch")
            values.append([_native_geometry(actual,expected) for actual,expected in zip(records,model)])
            audit["native_mapping_verified"]=True
        contexts=[]
        for current in values:
            nodes={}
            for entity in current:
                for endpoint in ("start","end"):
                    node_id=entity[endpoint+"_node"];point=np.asarray(entity[endpoint])
                    if node_id in nodes and np.linalg.norm(nodes[node_id]-point)>STRICT_RELATION_ENDPOINT_TOLERANCE:
                        raise ValueError("shared_native_node_endpoints_disagree")
                    nodes.setdefault(node_id,point)
            graph={"units":"mm","entities":current,"nodes":[
                {"id":key,"x":point[0],"y":point[1]} for key,point in nodes.items()]}
            _,_,prepared=_validate_inputs(graph,selected)
            contract=_strict_measurement_contract(nodes,current,prepared,model)
            contexts.append((nodes,{entity["id"]:entity for entity in current},prepared,
                             {row["constraint_id"]:row for row in contract["checks"]}))
        for row in contexts[0][2]:
            tolerance=STRICT_MEASUREMENT_CERT_TOLERANCE_MM if _strict_measurement_row(row) else LINEAR_TOLERANCE_MM
            samples=[]
            for nodes,indexed,_,checks in contexts:
                actual=float(_constraint_value(row,nodes,indexed));residual=float(_constraint_residual(row,actual))
                check=checks.get(row["id"])
                passed=math.isfinite(actual) and math.isfinite(residual) and abs(residual)<=tolerance
                sample={"actual":actual,"signed_residual":residual,"absolute_residual":abs(residual),
                        "passed":bool(passed and (check is None or check["passed"]))}
                if row.get("source_tolerance") is not None:
                    from .dimension_policy import tolerance_interval_check
                    sample["source_tolerance_interval"]=tolerance_interval_check(row["source_tolerance"],actual)
                    sample["passed"]=sample["passed"] and sample["source_tolerance_interval"]["passed"]
                if check:
                    sample.update(derived_anchor_points=check["derived_anchor_points"],
                                  intersection_conditions=check["intersection_conditions"])
                samples.append(sample)
            audit["checks"].append({"constraint_id":row["id"],"record_id":row.get("record_id"),
                "kind":row["kind"],"tolerance_mm":tolerance,"model":samples[0],
                "dxf":samples[1] if len(samples)>1 else None,"passed":all(sample["passed"] for sample in samples)})
        audit["satisfied_count"]=sum(row["passed"] for row in audit["checks"])
        audit["passed"]=audit["required_count"]==audit["satisfied_count"]
    except (ValueError,TypeError,KeyError,AttributeError,IndexError,ArithmeticError) as error:
        audit["issues"].append(str(error))
    return audit


def native_source_nominal_angle_checks(entities,constraints,*,dxf_document=None, _axis_only=False):
    """Recompute admitted exact angles on native geometry, without cached scores."""
    from .relation_contract import _geometry, _native_geometry
    audit={"schema_version":"source-nominal-angle-readback-v1","passed":False,
           "required_count":0,"satisfied_count":0,"checks":[],"issues":[],
           "dxf_readback_performed":False,"native_mapping_verified":False,
           "reference_geometry_used":False,"complete_angle_coverage_verified":False}
    try:
        if not isinstance(entities,list) or not isinstance(constraints,list):
            raise ValueError("entities_and_constraints_must_be_lists")
        selected=([row for row in constraints if _strict_axis_relation(row)] if _axis_only else
                  [row for row in constraints if row.get("kind")=="angle" and
                   (_strict_source_nominal(row) or _strict_ocr_axis_angle(row))])
        audit["required_count"]=len(selected)
        model=[_geometry(entity) for entity in entities];values=[model]
        if dxf_document is not None:
            records=list(dxf_document.modelspace());audit["dxf_readback_performed"]=True
            if len(records)!=len(model):raise ValueError("native_entity_count_mismatch")
            values.append([_native_geometry(actual,expected) for actual,expected in zip(records,model)])
            audit["native_mapping_verified"]=True
        contexts=[]
        # An exported angle must retain the certified model's ray/quadrant.
        specs=_strict_ocr_angle_specs(model,selected)
        for current in values:
            nodes={}
            for entity in current:
                for endpoint in ("start","end"):
                    node_id=entity[endpoint+"_node"];point=np.asarray(entity[endpoint])
                    if node_id in nodes and np.linalg.norm(nodes[node_id]-point)>STRICT_RELATION_ENDPOINT_TOLERANCE:
                        raise ValueError("shared_native_node_endpoints_disagree")
                    nodes.setdefault(node_id,point)
            graph={"units":"mm","entities":current,"nodes":[
                {"id":key,"x":point[0],"y":point[1]} for key,point in nodes.items()]}
            _,_,prepared=_validate_inputs(graph,selected)
            contract=_strict_ocr_angle_contract(current,specs)
            contexts.append((nodes,{entity["id"]:entity for entity in current},prepared,
                             {row["constraint_id"]:row for row in contract["checks"]}))
        for row in contexts[0][2]:
            samples=[]
            for nodes,indexed,_,checks in contexts:
                actual=float(_constraint_value(row,nodes,indexed));residual=float(_constraint_residual(row,actual))
                check=checks.get(row["id"])
                samples.append({"actual":actual,"signed_residual":residual,"absolute_residual":abs(residual),
                    "passed":bool(math.isfinite(residual) and abs(residual)<=STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG
                                  and (check is None or check["passed"]))})
            audit["checks"].append({"constraint_id":row["id"],"record_id":row.get("record_id"),
                "tolerance_deg":STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG,"model":samples[0],
                "dxf":samples[1] if len(samples)>1 else None,"passed":all(sample["passed"] for sample in samples)})
        audit["satisfied_count"]=sum(row["passed"] for row in audit["checks"])
        audit["passed"]=audit["required_count"]==audit["satisfied_count"]
    except (ValueError,TypeError,KeyError,AttributeError,IndexError,ArithmeticError) as error:
        audit["issues"].append(str(error))
    return audit


def native_axis_relation_checks(entities, constraints, *, dxf_document=None):
    """Certify admitted axes separately from OCR angle coverage."""
    audit = native_source_nominal_angle_checks(entities, constraints,
        dxf_document=dxf_document, _axis_only=True)
    audit["schema_version"] = "source-axis-relation-readback-v1"
    return audit


def _model(nodes,entities,exact_radii=None):
    exact_radii=exact_radii or {}
    points=np.array([[n["x"],n["y"]] for n in nodes]);origin=points.mean(axis=0)
    scale=max(float(np.linalg.norm(np.ptp(points,axis=0))),1e-3)
    node_indices={n["id"]:i for i,n in enumerate(nodes)}
    initial=list(((points-origin)/scale).ravel());arc_params={};branches={}
    for entity in entities:
        if entity["type"]!="ARC":continue
        a=np.array(entity["start"]);b=np.array(entity["end"]);chord=b-a;normal=np.array([-chord[1],chord[0]])/np.linalg.norm(chord)
        h=float(np.dot(np.array(entity["center"])-(a+b)/2,normal))
        major=_sweep(entity)>math.pi+1e-9
        sign=1 if (not entity["clockwise"] and not major) or (entity["clockwise"] and major) else -1
        if entity["id"] not in exact_radii:
            index=len(initial);initial.append(abs(h)/scale);arc_params[entity["id"]]=(index,sign)
        branches[entity["id"]]={"clockwise":entity["clockwise"],"major":major,"initial_sweep_deg":math.degrees(_sweep(entity)),"h_sign":sign}
    initial=np.array(initial);lower=initial-10;upper=initial+10
    for index,sign in arc_params.values():lower[index]=0.;upper[index]=max(50.,initial[index]*5+5)

    def decode(vector):
        xy=vector[:2*len(nodes)].reshape((-1,2))*scale+origin
        node_map={node["id"]:xy[i] for i,node in enumerate(nodes)};result=[]
        for original in entities:
            entity=copy.copy(original);a=node_map[entity["start_node"]];b=node_map[entity["end_node"]]
            entity.update(start=a.tolist(),end=b.tolist())
            if entity["type"]=="ARC":
                chord=b-a;length=float(np.linalg.norm(chord))
                normal=np.array([-chord[1],chord[0]])/max(length,1e-12)
                if entity["id"] in exact_radii:
                    # The annotation value is a constant, not an optimizer
                    # residual. Chord feasibility is an explicit inequality
                    # in the constrained solve below. During infeasible trial
                    # steps the clipped square root stays finite, but circle
                    # incidence validation prevents publishing those trials.
                    radius=exact_radii[entity["id"]]
                    h=branches[entity["id"]]["h_sign"]*math.sqrt(max(0.,radius*radius-(length/2)**2))
                else:
                    index,sign=arc_params[entity["id"]];h=sign*max(vector[index],0)*scale
                    radius=float(math.hypot(length/2,h))
                entity.update(center=((a+b)/2+h*normal).tolist(),radius=radius)
            result.append(entity)
        return node_map,result
    return initial,lower,upper,decode,scale,node_indices,arc_params,branches


def _on_arc(entity,point,epsilon):
    if abs(math.dist(entity["center"],point)-entity["radius"])>epsilon:return False
    center=entity["center"]
    start=math.atan2(entity["start"][1]-center[1],entity["start"][0]-center[0])
    angle=math.atan2(point[1]-center[1],point[0]-center[0])
    travel=((start-angle) if entity["clockwise"] else (angle-start))%(2*math.pi)
    return travel<=_sweep(entity)+epsilon/max(entity["radius"],epsilon) or 2*math.pi-travel<=epsilon/max(entity["radius"],epsilon)


def _intersections(first,second,epsilon):
    if first["type"]==second["type"]=="LINE":
        a=np.array(first["start"]);u=np.array(first["end"])-a;b=np.array(second["start"]);v=np.array(second["end"])-b
        cross=lambda x,y:float(x[0]*y[1]-x[1]*y[0])
        denominator=cross(u,v)
        if abs(denominator)>epsilon*max(np.linalg.norm(u),np.linalg.norm(v),1):
            t=cross(b-a,v)/denominator;s=cross(b-a,u)/denominator
            return [a+t*u] if -1e-10<=t<=1+1e-10 and -1e-10<=s<=1+1e-10 else []
        if abs(cross(b-a,u))>epsilon*max(np.linalg.norm(u),1):return []
        length=float(np.dot(u,u));ts=[float(np.dot(np.array(p)-a,u)/max(length,1e-20)) for p in (second["start"],second["end"])]
        low,high=max(0,min(ts)),min(1,max(ts))
        return [a+low*u,a+high*u,a+(low+high)/2*u] if low<=high else []
    if first["type"]=="LINE" or second["type"]=="LINE":
        line,arc=(first,second) if first["type"]=="LINE" else (second,first)
        start=np.array(line["start"]);direction=np.array(line["end"])-start;offset=start-arc["center"]
        a=float(np.dot(direction,direction));b=2*float(np.dot(direction,offset));c=float(np.dot(offset,offset))-arc["radius"]**2
        determinant=b*b-4*a*c
        if determinant<0 or a<=1e-20:return []
        candidates=[start+t*direction for t in ((-b-math.sqrt(max(0,determinant)))/(2*a),(-b+math.sqrt(max(0,determinant)))/(2*a)) if -1e-10<=t<=1+1e-10]
        return [point for point in candidates if _on_arc(arc,point,epsilon)]
    c1=np.array(first["center"]);c2=np.array(second["center"]);r1=first["radius"];r2=second["radius"]
    distance=float(np.linalg.norm(c2-c1));candidates=[]
    if distance<=epsilon:
        if abs(r1-r2)<=epsilon:
            candidates=[np.array(e[key]) for e in (first,second) for key in ("start","end")]
            candidates.extend(_sample_entity(e,9)[4] for e in (first,second))
    elif abs(r1-r2)-epsilon<=distance<=r1+r2+epsilon:
        x=(r1*r1-r2*r2+distance*distance)/(2*distance);height=math.sqrt(max(0,r1*r1-x*x))
        unit=(c2-c1)/distance;center=c1+x*unit;normal=np.array([-unit[1],unit[0]])
        candidates=[center+height*normal,center-height*normal]
    return [point for point in candidates if _on_arc(first,point,epsilon) and _on_arc(second,point,epsilon)]


def _sample_entity(entity,count=17,*,midpoints=False):
    a=np.array(entity["start"]);b=np.array(entity["end"])
    t=(np.arange(count)+.5)/count if midpoints else np.linspace(0,1,count)
    if entity["type"]=="LINE":return a+t[:,None]*(b-a)
    center=np.array(entity["center"]);angle=math.atan2(a[1]-center[1],a[0]-center[0]);theta=angle+(-1 if entity["clockwise"] else 1)*_sweep(entity)*t
    return center+entity["radius"]*np.column_stack([np.cos(theta),np.sin(theta)])


def _geometry_validation(entities,branches,scale):
    epsilon=max(1e-7,scale*1e-10);issues=[];radial=[];area=0.;joins=[]
    area_origin=np.array(entities[0]["start"])
    for index,entity in enumerate(entities):
        a=np.array(entity["start"]);b=np.array(entity["end"]);gap=math.dist(entity["end"],entities[(index+1)%len(entities)]["start"])
        joins.append({"first":entity["id"],"second":entities[(index+1)%len(entities)]["id"],"gap":gap})
        if np.linalg.norm(b-a)<=epsilon:issues.append(f"{entity['id']}: degenerate chord")
        if entity["type"]=="ARC":
            radius=entity["radius"];center=np.array(entity["center"]);sweep=_sweep(entity)
            radial.extend(abs(math.dist(center,p)-radius) for p in (a,b))
            if radius<=0 or sweep<1e-9 or sweep>2*math.pi-1e-9:issues.append(f"{entity['id']}: degenerate arc")
            major=sweep>math.pi+1e-7
            if major!=branches[entity["id"]]["major"] and abs(sweep-math.pi)>1e-7:issues.append(f"{entity['id']}: arc major/minor branch changed")
            shifted_center=center-area_origin
            area+=.5*(shifted_center[0]*(b[1]-a[1])-shifted_center[1]*(b[0]-a[0])+radius*radius*sweep*(-1 if entity["clockwise"] else 1))
        else:
            first=a-area_origin;last=b-area_origin
            area+=.5*(first[0]*last[1]-first[1]*last[0])
    intersections=[]
    for i,first in enumerate(entities):
        for second in entities[i+1:]:
            shared=set((first["start_node"],first["end_node"]))&set((second["start_node"],second["end_node"]))
            allowed=[first[key] for key in ("start","end") if first[key+"_node"] in shared]
            for point in _intersections(first,second,epsilon):
                if not any(math.dist(point,common)<=epsilon*3 for common in allowed):
                    intersections.append({"first":first["id"],"second":second["id"],"point":point.tolist()});break
    if intersections:issues.append("Contour has a self-intersection or overlapping primitives")
    if max((join["gap"] for join in joins),default=0)>epsilon:issues.append("Contour is not closed")
    if max(radial,default=0)>epsilon:issues.append("Arc endpoint circle incidence failed")
    if abs(area)<=max(1e-12,scale*scale*1e-12):issues.append("Contour has zero signed area")
    return {"passed":not issues,"finite":True,"closed":all(join["gap"]<=epsilon for join in joins),
            "max_gap":max((j["gap"] for j in joins),default=0),"max_radial_error":max(radial,default=0),
            "self_intersection":bool(intersections),"intersections":intersections,"signed_area":float(area),
            "joins":joins,"geometry_epsilon":epsilon,"issues":issues}


def _persist(result,output_dir):
    if output_dir is not None:
        directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
        destination=directory/"parametric-solve.json";temporary=destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8");temporary.replace(destination)
    return result


def _strict_radius_contract(entities,constraints):
    """Report equality of the actual returned radii, never candidate-only radii."""
    indexed={entity["id"]:entity for entity in entities}
    rows=[row for row in constraints if row["kind"]=="radius"]
    checks=[{"constraint_id":row["id"],"record_id":row["record_id"],"entity_id":row["entities"][0],
             "required_radius":row["value"],"actual_radius":indexed[row["entities"][0]]["radius"],
             "passed":indexed[row["entities"][0]]["radius"]==row["value"]} for row in rows]
    return {"mode":"exact_parameter_elimination","required_count":len(rows),
            "required_entity_count":len({row["entities"][0] for row in rows}),
            "satisfied":all(row["passed"] for row in checks),"checks":checks,
            "failed_entity_ids":sorted({row["entity_id"] for row in checks if not row["passed"]}),
            "binding_verified_by_solver":False,"unbound_radius_coverage_verified":False}


def _strict_ocr_axis_angle(row):
    """Admitted exact source angles and legacy arrow-verified angles are hard."""
    return (row["kind"]=="angle" and row.get("reference_axis") in {"horizontal","vertical"}
            and len(row["entities"])==1 and row["nodes"]==[]
            and row.get("required") is True and row.get("nominal_source")=="source_ocr"
            and (row.get("source_arrow_verified") is True or _strict_source_nominal(row)))


def _strict_ocr_angle_specs(entities,constraints):
    indexed={entity["id"]:entity for entity in entities}
    specs=[]
    for row in constraints:
        if not _strict_ocr_axis_angle(row):continue
        entity=indexed[row["entities"][0]]
        delta=np.asarray(entity["end"],float)-np.asarray(entity["start"],float)
        length=float(np.linalg.norm(delta))
        theta=math.radians(row["value"])
        magnitudes=([math.sin(theta),math.cos(theta)] if row["reference_axis"]=="vertical" else
                    [math.cos(theta),math.sin(theta)])
        direction=[];reason=None
        for component,magnitude in zip(delta,magnitudes):
            if abs(magnitude)<1e-14:
                direction.append(0.)
            elif abs(component)<=1e-10*length:
                reason="source_line_direction_quadrant_ambiguous"
                direction.append(0.)
            else:
                direction.append(math.copysign(magnitude,component))
        if reason is None:
            direction=np.asarray(direction,float)
            direction/=np.linalg.norm(direction)
        else:
            direction=None
        specs.append({"row":row,"entity_id":entity["id"],"start_node":entity["start_node"],
                      "end_node":entity["end_node"],"direction":direction,
                      "initial_length":length,"setup_issue":reason})
    by_entity={}
    for spec in specs:
        if spec["direction"] is None:continue
        previous=by_entity.get(spec["entity_id"])
        if previous is not None and abs(float(previous["direction"][0]*spec["direction"][1]-
                                              previous["direction"][1]*spec["direction"][0]))>1e-12:
            spec["setup_issue"]="conflicting_strict_ocr_angles_on_one_line"
        else:
            by_entity[spec["entity_id"]]=spec
    return specs


def _strict_ocr_angle_contract(entities,specs):
    """Certify returned LINE angles and their source-graph direction quadrant."""
    indexed={entity["id"]:entity for entity in entities}
    checks=[]
    for spec in specs:
        row=spec["row"];entity=indexed[spec["entity_id"]]
        actual=float(_constraint_value(row,{},indexed))
        delta=np.asarray(entity["end"],float)-np.asarray(entity["start"],float)
        direction=spec["direction"]
        quadrant=bool(direction is not None and all(
            delta[index]*direction[index]>0 for index in range(2) if direction[index]!=0))
        error=abs(_constraint_residual(row,actual))
        passed=bool(spec["setup_issue"] is None and quadrant and
                    error<=STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG)
        checks.append({"constraint_id":row["id"],"record_id":row["record_id"],
                       "entity_id":spec["entity_id"],"nominal_angle_deg":row["value"],
                       "actual_angle_deg":actual,"absolute_error_deg":error,
                       "source_direction_quadrant_preserved":quadrant,
                       "passed":passed,"reason":spec["setup_issue"]})
    return {"mode":"linear_endpoint_direction_equality",
            "numerical_certificate_tolerance_deg":STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG,
            "required_count":len(checks),"satisfied":all(row["passed"] for row in checks),
            "checks":checks,"failed_record_ids":sorted({row["record_id"] for row in checks if not row["passed"]}),
            "binding_verified_by_solver":False,"reference_verified":False}


def _strict_tangent_contract(entities,constraints):
    """Certify only admitted joint relations, including finite forward traversal.

    A zero cross product alone also describes reversed tangents or degenerate
    vectors. Neither is a valid G1 joint. The directed angular measurement and
    endpoint checks remain independent of the optimizer's equality row basis.
    """
    indexed={entity["id"]:entity for entity in entities};checks=[]
    for row in constraints:
        if row["kind"]!="tangent":continue
        first,second=(indexed[eid] for eid in row["entities"]);joint=row["nodes"][0]
        points=[entity["end"] if entity["end_node"]==joint else entity["start"]
                for entity in (first,second)]
        lengths=[math.dist(entity["start"],entity["end"]) for entity in (first,second)]
        vectors=[_tangent(entity,entity["end_node"]==joint) for entity in (first,second)]
        gap=math.dist(*points);dot=float(np.dot(*vectors));angle=abs(_angle(*vectors,True))
        finite=bool(all(math.isfinite(value) and value>1e-9 for value in lengths) and
                    all(np.isfinite(vector).all() for vector in vectors))
        checks.append({"constraint_id":row["id"],"entity_ids":row["entities"],"joint_node":joint,
                       "absolute_error_deg":angle,"endpoint_gap":gap,"forward_dot":dot,
                       "finite_supports":finite,"passed":bool(finite and dot>0. and
                       gap<=STRICT_RELATION_ENDPOINT_TOLERANCE and
                       angle<=STRICT_TANGENT_CERT_TOLERANCE_DEG)})
    return {"mode":"normalized_tangent_cross_equality_with_forward_orientation",
            "scope":"supplied_validated_tangent_constraints_only",
            "required_count":len(checks),"satisfied":all(row["passed"] for row in checks),
            "numerical_certificate_tolerance_deg":STRICT_TANGENT_CERT_TOLERANCE_DEG,
            "endpoint_certificate_tolerance":STRICT_RELATION_ENDPOINT_TOLERANCE,
            "checks":checks,"unbound_joint_coverage_verified":False}


def _independent_equality_rows(jacobian):
    """Select a well-conditioned local row basis without discarding obligations.

    SLSQP rejects redundant equality rows (even if they describe valid geometry).
    Every original row is still measured and certified before publication. Rank
    reduction concerns the numerical subproblem only, never annotation coverage.
    """
    if not len(jacobian):return np.empty(0,dtype=int),0.
    norms=np.linalg.norm(jacobian,axis=1)
    scaled=jacobian/np.maximum(norms[:,None],1e-15)
    _,triangular,pivots=qr(scaled.T,mode="economic",pivoting=True)
    diagonal=np.abs(np.diag(triangular))
    tolerance=max(float(diagonal[0])*1e-10,1e-12) if len(diagonal) else 1e-12
    rank=int(np.count_nonzero(diagonal>tolerance))
    return np.sort(pivots[:rank]),tolerance


def _constructed_radius_invariants(graph,entities):
    """Preserve implemented geometry without treating history as a binding.

    A fresh binder can reject an ambiguous OCR association while the topology
    still contains an exact-radius construction. That radius is a geometric
    invariant of the chosen candidate, not another admitted OCR constraint.
    Incomplete or inconsistent history cannot introduce a new invariant.
    """
    invariants=[];ignored=[]
    for entity in entities:
        binding=entity.get("radius_binding")
        if not binding or entity["type"]!="ARC":continue
        reason=None;evidence=entity.get("radius_annotation_evidence")
        if (graph["units"]!="mm" or entity.get("radius_constructed") is not True or
                entity.get("parameter_source")!="multimodal_annotation_guided_arc_refit" or
                entity.get("radius_binding_status")!="constructed_unverified"):
            reason="not_a_declared_constructed_radius"
        elif not isinstance(binding,dict) or not isinstance(evidence,dict):
            reason="incomplete_construction_metadata"
        else:
            try:
                record=_identifier(binding.get("record_id"),"constructed record id")
                nominal=_number(binding.get("nominal"),"constructed nominal")
                target=_point(binding.get("target_source_px"),"constructed source target")
                evidence_nominal=_number(evidence.get("nominal"),"construction evidence nominal")
                evidence_target=_point(evidence.get("target_source_px"),"construction evidence target")
                fit_error=_number(entity.get("source_fit_error_px"),"constructed source fit error")
                if (nominal<=0 or fit_error<0 or binding.get("arrowhead_verified") is not True or
                        evidence.get("record_id")!=record or evidence_nominal!=nominal or
                        evidence_target!=target or evidence.get("arrowhead_verified") is not True):
                    reason="inconsistent_construction_metadata"
                elif entity["radius"]!=nominal:
                    reason="construction_radius_is_not_already_exact"
                elif max(abs(math.dist(entity["center"],entity[key])-nominal) for key in ("start","end"))>1e-8*max(1.,nominal):
                    reason="construction_circle_incidence_failed"
            except (ValueError,TypeError):
                reason="incomplete_or_invalid_construction_metadata"
        if reason:
            ignored.append({"entity_id":entity["id"],"reason":reason,"binding_verified":False})
        else:
            invariants.append({"entity_id":entity["id"],"historical_record_id":record,
                               "preserved_radius":entity["radius"],"binding_verified":False,
                               "source":"implemented_candidate_geometry","annotation_constraint_added":False})
    return invariants,ignored


def _constructed_radius_receipt(entities,invariants,ignored):
    indexed={entity["id"]:entity for entity in entities}
    checks=[{**row,"actual_radius":indexed[row["entity_id"]]["radius"],
             "passed":indexed[row["entity_id"]]["radius"]==row["preserved_radius"]} for row in invariants]
    return {"mode":"existing_geometry_parameter_elimination","required_count":len(checks),
            "satisfied":all(row["passed"] for row in checks),"checks":checks,"ignored":copy.deepcopy(ignored),
            "binding_verified":False,"annotation_constraint_count_added":0,
            "meaning":"Preserves already constructed numerical geometry only; fresh OCR target binding and annotation coverage remain unverified."}


def _finite_difference_jacobian(function,vector):
    # Independent of the scalar SLSQP gradient: rank must measure dimensional
    # constraints only, excluding the source prior and eliminated constants.
    values=function(vector);jacobian=np.empty((len(values),len(vector)))
    for index in range(len(vector)):
        delta=np.sqrt(np.finfo(float).eps)*max(1.,abs(vector[index]))
        after=vector.copy();after[index]+=delta
        before=vector.copy();before[index]-=delta
        jacobian[:,index]=(function(after)-function(before))/(2*delta)
    return jacobian


def _raw_source_prior(observation,graph,entities,scale,*,entity_weights=None):
    """Build bounded local, bidirectional residuals against immutable mask data.

    Source points may match the neighboring primitives while a shared joint
    moves. This permits correcting an initial partition without pretending
    that the old CAD sample parameterization was measured geometry.
    """
    if observation is None:return None,{"enabled":False}
    if (not isinstance(observation,dict) or observation.get("units")!=graph["units"] or
            observation.get("provenance")!="input_mask_boundary" or observation.get("reference_dxf_read") is not False):
        raise ValueError("Source observation must declare matching units and input-mask provenance without reference DXF")
    points=np.asarray(observation.get("points"),float)
    if (points.ndim!=2 or points.shape[1]!=2 or not 4<=len(points)<=30000 or
            not np.isfinite(points).all() or np.max(np.abs(points))>1e9 or
            np.linalg.norm(points[0]-points[-1])>max(1e-8,scale*1e-10)):
        raise ValueError("Source observation must be a bounded finite closed mask boundary")
    points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-12]]
    if len(points)<4:raise ValueError("Source observation mask boundary is degenerate")
    vertices=points[:-1];deltas=points[1:]-vertices
    squared=np.sum(deltas*deltas,axis=1);lengths=np.sqrt(squared)
    cumulative=np.r_[0.,np.cumsum(lengths)];perimeter=float(cumulative[-1])
    if perimeter<=1e-12:raise ValueError("Source observation mask boundary is degenerate")
    # Polygon vertex density is an encoding choice, not independent evidence.
    # Sample the immutable boundary uniformly in physical arclength so dense
    # raster stair steps and sparse long edges have the same influence.
    quadrature_count=1024
    locations=(np.arange(quadrature_count)+.5)*(perimeter/quadrature_count)
    segment_indices=np.minimum(np.searchsorted(cumulative,locations,side="right")-1,len(vertices)-1)
    fractions=(locations-cumulative[segment_indices])/lengths[segment_indices]
    quadrature=vertices[segment_indices]+fractions[:,None]*deltas[segment_indices]
    assigned=np.argmin(np.asarray([_primitive_distance(vertices,entity) for entity in entities]),axis=0)
    quadrature_assigned=np.argmin(np.asarray([_primitive_distance(quadrature,entity) for entity in entities]),axis=0)
    primitive_lengths=np.array([math.dist(entity["start"],entity["end"]) if entity["type"]=="LINE" else
                                entity["radius"]*_sweep(entity) for entity in entities])
    primitive_shares=primitive_lengths/max(float(primitive_lengths.sum()),1e-12)
    groups=[]
    weights=np.ones(len(entities)) if entity_weights is None else np.asarray(entity_weights,float)
    if weights.shape!=(len(entities),) or not np.isfinite(weights).all() or np.any(weights<0) or np.any(weights>1):
        raise ValueError('invalid_mask_localization_weights')
    for index in range(len(entities)):
        indices=np.flatnonzero(quadrature_assigned==index)
        selected=indices[np.unique(np.linspace(0,len(indices)-1,min(32,len(indices))).astype(int))] if len(indices) else indices
        neighborhood=[(index-1)%len(entities),index,(index+1)%len(entities)]
        local=np.flatnonzero(np.isin(assigned,neighborhood))
        local=np.unique(np.r_[local,(local-1)%len(vertices)])
        reverse_count=max(3,min(65,int(math.ceil(128*primitive_shares[index]))))
        groups.append((quadrature[selected],neighborhood,vertices[local],deltas[local],
                       squared[local],len(indices)/quadrature_count,
                       float(primitive_shares[index]),reverse_count))
    # Integrate per unit of normalized boundary length, instead of giving
    # every primitive an equal RMS weight. Splitting a supported line/arc must
    # not multiply its evidence or make tiny fillets dominate a long arc.
    weight=SOURCE_CURVE_PRIOR_WEIGHT*math.sqrt(perimeter/scale)
    def residuals(decoded):
        values=[]
        for index,(source,neighbors,segment_vertices,segment_deltas,segment_squared,
                   forward_share,reverse_share,reverse_count) in enumerate(groups):
            if len(source):
                source_distance=np.min(np.asarray([_primitive_distance(source,decoded[j]) for j in neighbors]),axis=0)
                values.append(source_distance*(weights[index]*weight*math.sqrt(forward_share/len(source)))/scale)
            if len(segment_vertices):
                samples=_sample_entity(decoded[index],reverse_count,midpoints=True)
                delta=samples[:,None,:]-segment_vertices[None,:,:]
                fraction=np.clip(np.sum(delta*segment_deltas[None,:,:],axis=2)/segment_squared[None,:],0.,1.)
                nearest=segment_vertices[None,:,:]+fraction[:,:,None]*segment_deltas[None,:,:]
                reverse=np.min(np.linalg.norm(samples[:,None,:]-nearest,axis=2),axis=1)
                values.append(reverse*(weights[index]*weight*math.sqrt(reverse_share/reverse_count))/scale)
        return np.concatenate(values)
    return residuals,{"enabled":True,"source":"immutable_input_mask_boundary","units":graph["units"],
                      "input_point_count":len(points),"source_quadrature_point_count":quadrature_count,
                      "source_quadrature":"uniform_physical_arclength_midpoints",
                      "maximum_forward_points_per_entity":32,"maximum_reverse_points_per_entity":65,
                      "matching":"local_entity_and_immediate_neighbors; exact_source_polyline_reverse_distance",
                      "weighting":"source_arclength_forward_and_primitive_arclength_reverse",
                      "source_perimeter":perimeter,"weight":SOURCE_CURVE_PRIOR_WEIGHT,"hard_constraint":False,
                      "entity_localization_weights":{e['id']:float(w) for e,w in zip(entities,weights)},
                      "oracle_mask_conditioned":observation.get("oracle_mask_conditioned") is True,
                      "reference_dxf_read":False,"prior_cad_curve_samples_replaced":True}


class _SourceBudgetExhausted(RuntimeError):
    pass


class _OptionalSourceRefinementFinished(RuntimeError):
    pass


class _NumericalBudgetExhausted(ArithmeticError):
    pass


class _NumericalBudget:
    """Bound optimization independently from any mask-fidelity equation."""
    def __init__(self, observation):
        spec=(observation or {}).get('optimization_budget')
        self.diagnostics={'enabled':spec is not None,'mask_fidelity_constraint':False}
        self.started=time.monotonic();self.count=0
        if spec is None:return
        self.seconds=_number(spec.get('max_wall_seconds',120.),'Optimization wall budget')
        self.maximum=spec.get('max_objective_evaluations',30000)
        if (not 0<self.seconds<=120 or type(self.maximum) is not int or not 1<=self.maximum<=30000):
            raise ValueError('Invalid bounded source optimization budget')
        self.diagnostics.update(maximum_wall_seconds=self.seconds,maximum_objective_evaluations=self.maximum)

    def guard(self):
        if not self.diagnostics['enabled']:return
        if time.monotonic()-self.started>=self.seconds or self.count>=self.maximum:
            self.diagnostics['exhausted']=True
            raise _NumericalBudgetExhausted('Bounded source optimization exhausted')
        self.count+=1

    def finish(self):
        self.diagnostics.update(objective_evaluations=self.count,elapsed_wall_seconds=time.monotonic()-self.started)


class _SourceBoundaryBudget:
    """Bounded witness generation with an independent complete-curve audit."""
    witnesses_per_direction=12
    def __init__(self,observation,graph):
        receipt=observation.get("boundary_error_budget") if observation else None
        self.enabled=receipt is not None;self.forward=[];self.reverse=[];self.seen=set()
        self.diagnostics={"enabled":self.enabled,"reference_dxf_read":False,"acceptance_thresholds_changed":False}
        if not self.enabled:return
        if (not isinstance(receipt,dict) or receipt.get("units")!=graph["units"] or
                receipt.get("source")!="initial_curve_fit_total_deviation_budget_px"):
            raise ValueError("Source boundary budget requires matching units and immutable initial-mask provenance")
        self.maximum=_number(receipt.get("maximum_deviation"),"Source boundary budget")
        self.step=_number(receipt.get("sampling_step"),"Source boundary sampling step")
        if self.maximum<=0 or not 0<self.step<self.maximum:
            raise ValueError("Source boundary budget must exceed its positive sampling step")
        self.wall_limit=_number(receipt.get("max_wall_seconds",120.),"Source boundary search wall limit")
        self.objective_limit=receipt.get("max_objective_evaluations",30000)
        if not 0<self.wall_limit<=120 or isinstance(self.objective_limit,bool) or not isinstance(self.objective_limit,int) or not 1<=self.objective_limit<=30000:
            raise ValueError("Source boundary search limits must be positive and cannot exceed 120 seconds or 30000 objective evaluations")
        self.started=time.monotonic();self.objective_evaluations=0;self.exhausted=False;self.search_active=True
        self.optional_refinement_deadline=None
        self.points=np.asarray(observation["points"],float)
        self.samples,_,_=_sample_entities(_line_entities(self.points),self.step,max_samples=100000)
        self.tree=cKDTree(self.samples)
        # The audit adds step/2; forward analytic primitive distances also need
        # a step/2 reserve for the candidate's nearest discrete audit sample.
        self.forward_limit=self.maximum-self.step
        self.reverse_limit=self.maximum-self.step/2
        self.diagnostics.update(units=graph["units"],maximum_deviation=self.maximum,sampling_step=self.step,
            audit_sampling_margin=self.step/2,forward_witness_limit=self.forward_limit,
            reverse_witness_limit=self.reverse_limit,maximum_rounds=SOURCE_BUDGET_SEARCH_ROUNDS,
            maximum_witnesses_per_round=2*self.witnesses_per_direction,rounds=[],infeasibility_proven=False,
            maximum_wall_seconds=self.wall_limit,maximum_objective_evaluations=self.objective_limit,
            method="bounded_worst_point_constraint_generation_then_RMS",
            source="immutable_input_mask_boundary")

    def guard(self,*,objective=False):
        elapsed=time.monotonic()-self.started
        if elapsed>=self.wall_limit or (objective and self.objective_evaluations>=self.objective_limit):
            self.exhausted=True
            self.diagnostics.update(exhausted=True,exhaustion_reason="wall_seconds" if elapsed>=self.wall_limit else "objective_evaluations",
                stop_reason="wall_time_budget_exhausted" if elapsed>=self.wall_limit else "objective_evaluation_budget_exhausted")
            raise _SourceBudgetExhausted("Bounded source-budget search exhausted its wall or objective-evaluation limit")
        if self.optional_refinement_deadline is not None and time.monotonic()>=self.optional_refinement_deadline:
            raise _OptionalSourceRefinementFinished("Optional RMS sub-budget ended; retain the certified feasible checkpoint")
        if objective:self.objective_evaluations+=1

    def progress(self):
        self.diagnostics.update(elapsed_wall_seconds=time.monotonic()-self.started,
                                objective_evaluations=self.objective_evaluations,exhausted=self.exhausted)

    def inspect(self,entities,*,add_witnesses=True):
        quality=assess_fit_quality(self.points,entities,max_step_px=self.step)
        upper=float(quality["source_boundary_deviation_px"]["conservative_upper_bound_px"])
        forward_distance=np.min(np.asarray([_primitive_distance(self.samples,e) for e in entities]),axis=0)
        update={}
        if add_witnesses:
            # Spatial separation avoids spending a round's finite witness
            # budget on adjacent samples of the same local displacement. Add
            # only actual violations, retaining every previous witness. A
            # plateau in the global maximum is not stagnation when new source
            # locations still reveal missing constraints.
            selected=[]
            forward_violations=np.flatnonzero(forward_distance>self.forward_limit)
            for index in forward_violations[np.argsort(forward_distance[forward_violations])[::-1]]:
                if ("forward",int(index)) in self.seen:continue
                if all(np.linalg.norm(self.samples[index]-self.samples[old])>self.step*3 for old in selected):
                    selected.append(int(index))
                if len(selected)>=self.witnesses_per_direction:break
            for index in selected:
                key=("forward",index)
                if key not in self.seen:self.seen.add(key);self.forward.append(self.samples[index].copy())
            reverse=[]
            for index,entity in enumerate(entities):
                length=math.dist(entity["start"],entity["end"]) if entity["type"]=="LINE" else entity["radius"]*_sweep(entity)
                count=max(2,int(math.ceil(length/self.step))+1)
                if count>100000:raise ValueError("Source boundary candidate sampling limit exceeded")
                samples=_sample_entity(entity,count);distances=self.tree.query(samples,workers=1)[0]
                violating=np.flatnonzero(distances>self.reverse_limit)
                for sample in violating:
                    reverse.append((float(distances[sample]),index,float(sample/(count-1))))
            unseen=[row for row in sorted(reverse,reverse=True) if ("reverse",row[1],round(row[2],8)) not in self.seen]
            selected_reverse=unseen[:self.witnesses_per_direction]
            for _,index,fraction in selected_reverse:
                key=("reverse",index,round(fraction,8))
                if key not in self.seen:self.seen.add(key);self.reverse.append((index,fraction))
            update={"new_forward_witnesses":len(selected),"new_reverse_witnesses":len(selected_reverse),
                    "new_witness_count":len(selected)+len(selected_reverse),
                    "violating_forward_sample_count":len(forward_violations),
                    "violating_reverse_sample_count":len(reverse),
                    "forward_witness_count":len(self.forward),"reverse_witness_count":len(self.reverse)}
        return {"passed":bool(quality["sampled_topology_valid"] and upper<=self.maximum),
                "conservative_max_deviation":upper,"sampling_uncertainty":quality["source_boundary_deviation_px"]["sampling_uncertainty_bound_px"],
                "source_to_curve_sample_max":float(forward_distance.max()),
                "geometry_valid":quality["sampled_topology_valid"],"reference_verified":False,
                **({"witness_update":update} if add_witnesses else {})}

    def witness_constraints(self,entities):
        values=[]
        if self.forward:
            samples=np.asarray(self.forward)
            distances=np.min(np.asarray([_primitive_distance(samples,e) for e in entities]),axis=0)
            values.extend(((self.forward_limit-distances)/self.maximum).tolist())
        for index,fraction in self.reverse:
            entity=entities[index];a=np.asarray(entity["start"]);b=np.asarray(entity["end"])
            if entity["type"]=="LINE":point=a+fraction*(b-a)
            else:
                center=np.asarray(entity["center"]);angle=math.atan2(a[1]-center[1],a[0]-center[0])
                angle+=(-1 if entity["clockwise"] else 1)*_sweep(entity)*fraction
                point=center+entity["radius"]*np.array([math.cos(angle),math.sin(angle)])
            values.append((self.reverse_limit-float(self.tree.query(point,workers=1)[0]))/self.maximum)
        return np.asarray(values)

    def safe_inspect(self,entities,*,add_witnesses=False):
        try:return self.inspect(entities,add_witnesses=add_witnesses)
        except (ValueError,ArithmeticError) as error:
            # A transient infeasible chord/invalid sampled polygon must not
            # abort the isolated search or be called evidence of infeasibility.
            return {"passed":False,"audit_error_type":type(error).__name__,"geometry_valid":False,
                    "reference_verified":False}


def numerical_problem_identity(graph, constraints):
    """Identify the immutable domain and complete equations of a continuation."""
    domain={k:graph.get(k) for k in ('units','coordinate_system','nodes','entities','source_sha256')}
    payload=json.dumps({'domain':domain,'constraints':constraints},sort_keys=True,
                       ensure_ascii=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(payload.encode('utf8')).hexdigest()


def solve_parametric(graph,constraints,*,output_dir=None,source_observation=None,seed_node_offsets=None,seed_entities=None,
                     source_warm_start=False):
    """Solve declared constraints jointly; preserve baseline on every rejection.

    ``accepted`` permits a validated partial dimensional candidate even when
    ``underconstrained`` remains true. Neither flag establishes all drawing
    dimensions, annotation bindings, reference accuracy or engineering approval.
    Invalid input/schema raises ValueError before any optimization or output.
    """
    if type(source_warm_start) is not bool:
        raise ValueError("source_warm_start must be boolean")
    nodes,baseline,prepared=_validate_inputs(graph,constraints)
    exact_radii={};conflicting_radii=set()
    for row in prepared:
        if row["kind"]!="radius":continue
        entity_id=row["entities"][0]
        if entity_id in exact_radii and exact_radii[entity_id]!=row["value"]:conflicting_radii.add(entity_id)
        else:exact_radii[entity_id]=row["value"]
    bound_radii=exact_radii.copy()
    preserved_radii,ignored_preservation=_constructed_radius_invariants(graph,baseline)
    preservation_conflicts=[]
    for row in preserved_radii:
        entity_id=row["entity_id"];radius=row["preserved_radius"]
        if entity_id in exact_radii and exact_radii[entity_id]!=radius:
            conflicting_radii.add(entity_id);preservation_conflicts.append(entity_id)
        else:exact_radii[entity_id]=radius
    unverified_preservation_count=sum(row["entity_id"] not in bound_radii for row in preserved_radii)
    strict_angles=_strict_ocr_angle_specs(baseline,prepared)
    strict_tangents=[row for row in prepared if row["kind"]=="tangent"]
    strict_measurements=[row for row in prepared if _strict_measurement_row(row)]
    strict_general_angles=_strict_general_angle_rows(prepared)
    strict_angle_setup_issues=[{"record_id":spec["row"]["record_id"],
                                "reason":spec["setup_issue"]} for spec in strict_angles if spec["setup_issue"]]
    initial,lower,upper,decode,scale,node_indices,arc_params,branches=_model(nodes,baseline,exact_radii)
    count=len(initial);normalizers=np.array([LINEAR_TOLERANCE_MM if row["kind"] in _DIMENSIONAL else ANGLE_TOLERANCE_DEG for row in prepared])
    _,initial_entities=decode(initial)
    if exact_radii:initial_entities=copy.deepcopy(baseline)
    from .source_stroke_prior import stroke_prior
    ink_prior,ink_prior_diagnostics=stroke_prior(
        (source_observation or {}).get("image_stroke_prior"),graph,initial_entities,scale)
    mask_weights=None
    if ((source_observation or {}).get('mask_role')=='initial_topology_and_soft_localization' and
            ink_prior_diagnostics.get('method')=='arclength_weighted_source_strip_centre_to_native_primitive'):
        fractions=ink_prior_diagnostics.get('entity_observed_fractions') or {}
        # Observed original ink determines unknown attributes. The mask remains
        # a soft fallback only where source strips are absent or ambiguous.
        # No source bounds, annotation equality or acceptance gate is changed.
        mask_weights=[(1.-fractions.get(e['id'],0.))**2 for e in initial_entities]
    raw_prior,raw_prior_diagnostics=_raw_source_prior(source_observation,graph,initial_entities,scale,
        entity_weights=mask_weights)
    source_budget=_SourceBoundaryBudget(source_observation,graph)
    numerical_budget=_NumericalBudget(source_observation)
    base_validation=_geometry_validation(initial_entities,branches,scale)
    result={"schema_version":"source-parametric-solver-v1","status":"no_constraints","accepted":False,"underconstrained":True,
            "units":graph["units"],"coordinate_system":copy.deepcopy(graph.get("coordinate_system")),
            "entities":copy.deepcopy(baseline),"nodes":copy.deepcopy(nodes),"baseline_entities":copy.deepcopy(baseline),
            "candidate_entities":None,"candidate_nodes":None,"constraints":[],"validation":{},
            "diagnostics":{"source_only":True,"ground_truth_used":False,"network_used":False,"source_coordinates_hard_locked":False,
                           "arc_parameterization":"shared endpoints + exact bound radius constants, preserved constructed geometry radii, or free chord-normal center offsets; fixed clockwise/minor-major branch",
                           "arc_branches":branches,"dimension_tolerance_mm":LINEAR_TOLERANCE_MM,"angle_tolerance_deg":ANGLE_TOLERANCE_DEG,
                           "strict_ocr_angle_certificate_tolerance_deg":STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG,
                           "strict_ocr_angle_record_count":len(strict_angles),
                           "strict_ocr_angle_equality_count":len({spec["entity_id"] for spec in strict_angles}),
                           "strict_tangent_certificate_tolerance_deg":STRICT_TANGENT_CERT_TOLERANCE_DEG,
                           "strict_tangent_constraint_count":len(strict_tangents),
                           "strict_measurement_constraint_count":len(strict_measurements),
                           "strict_general_angle_constraint_count":len(strict_general_angles),
                           "strict_measurement_certificate_tolerance_mm":STRICT_MEASUREMENT_CERT_TOLERANCE_MM,
                           "measurement_intersection_minimum_angle_deg":MIN_MEASUREMENT_INTERSECTION_ANGLE_DEG,
                           "radius_tolerance_mm":0.,"eliminated_exact_radius_constraint_count":len(bound_radii),
                           "eliminated_constructed_geometry_radius_count":unverified_preservation_count,
                           "preserved_geometry_is_dimensional_evidence":False,
                           "raw_source_prior":raw_prior_diagnostics,
                           "original_image_stroke_prior":ink_prior_diagnostics,
                           "source_budget_search":source_budget.diagnostics,
                           "optimization_budget":numerical_budget.diagnostics,
                           "reference_tolerance_changed":False,"independent_reference_accuracy_evaluated":False,
                           "prior_weight":.1,"variable_count":count,"max_displacement_fraction":MAX_DISPLACEMENT_FRACTION},
            "strict_radius_contract":_strict_radius_contract(baseline,prepared),
            "strict_ocr_angle_contract":_strict_ocr_angle_contract(baseline,strict_angles),
            "strict_source_nominal_angle_contract":_strict_general_angle_contract(
                {node["id"]:np.array([node["x"],node["y"]]) for node in nodes},baseline,strict_general_angles),
            "strict_tangent_contract":_strict_tangent_contract(baseline,strict_tangents),
            "strict_measurement_contract":_strict_measurement_contract(
                {node["id"]:np.array([node["x"],node["y"]]) for node in nodes},baseline,strict_measurements),
            "constructed_radius_preservation":_constructed_radius_receipt(baseline,preserved_radii,ignored_preservation),
            "all_dimensions_verified":False,"dimension_solve_success":False,"engineering_verified":False}
    if conflicting_radii:
        result.update(status="conflict",validation={"passed":False,"geometry_valid":base_validation["passed"],
                      "constraint_subset_satisfied":False,"strict_radius_satisfied":False,
                      "all_dimensions_verified":False,"dimensions_verified":False,
                      "issues":["An exact radius constraint conflicts with another constraint or a preserved constructed radius; baseline is retained."]})
        result["diagnostics"].update(converged=False,conflicting_radius_entity_ids=sorted(conflicting_radii),
                                     conflicting_constructed_radius_entity_ids=preservation_conflicts)
        return _persist(result,output_dir)
    if strict_angle_setup_issues:
        result.update(status="conflict",validation={"passed":False,"geometry_valid":base_validation["passed"],
                      "constraint_subset_satisfied":False,"strict_ocr_angle_satisfied":False,
                      "all_dimensions_verified":False,"dimensions_verified":False,
                      "issues":["A required OCR angle has no unique source-line direction quadrant or conflicts with another strict angle; baseline is retained."]})
        result["diagnostics"].update(converged=False,strict_ocr_angle_setup_issues=strict_angle_setup_issues)
        return _persist(result,output_dir)
    if not prepared:
        unchanged_validation=_geometry_validation(baseline,branches,scale)
        result["validation"]={**unchanged_validation,"passed":False,"geometry_valid":unchanged_validation["passed"],
                              "constraint_subset_satisfied":False,"all_dimensions_verified":False,"dimensions_verified":False,
                              "issues":[*unchanged_validation["issues"],"No explicit constraints were supplied; source geometry is retained."]}
        result["diagnostics"].update(constraint_rank=0,remaining_dof=count,
                                     remaining_shape_dof=max(0,count-3)+unverified_preservation_count,rigid_gauge_dof=3)
        return _persist(result,output_dir)

    strict_angle_by_id={spec["row"]["id"]:spec for spec in strict_angles}
    def constraint_residuals(vector):
        nmap,decoded=decode(vector);emap={entity["id"]:entity for entity in decoded}
        residuals=[]
        for row in prepared:
            spec=strict_angle_by_id.get(row["id"])
            if row["kind"]=="tangent":
                first,second=(emap[eid] for eid in row["entities"]);joint=row["nodes"][0]
                a=_tangent(first,first["end_node"]==joint)
                b=_tangent(second,second["end_node"]==joint)
                # Smooth through zero; directed forward orientation is a
                # separate optimizer inequality and final certificate.
                residuals.append(math.degrees(float(a[0]*b[1]-a[1]*b[0])))
            elif spec is None:
                residuals.append(_constraint_residual(row,_constraint_value(row,nmap,emap,allow_ill_conditioned=True)))
            else:
                # Signed, smooth endpoint-direction residual. The ordinary
                # unsigned atan2/abs has a cusp at exact 0°/90° and would
                # undercount this independent equality in the rank audit.
                direction=spec["direction"]
                normal=np.array([-direction[1],direction[0]])
                delta=nmap[spec["end_node"]]-nmap[spec["start_node"]]
                residuals.append(math.degrees(float(np.dot(normal,delta))/spec["initial_length"]))
        return np.asarray(residuals)/normalizers

    sparsity=lil_matrix((len(prepared)+count,count),dtype=int)
    for index,row in enumerate(prepared):
        dependencies=set(row["nodes"])
        dependencies.update(anchor["node_id"] for anchor in row.get("anchors",[]) if anchor["type"]=="node")
        for entity_id in row["entities"]:
            entity=next(entity for entity in baseline if entity["id"]==entity_id)
            dependencies.update((entity["start_node"],entity["end_node"]))
            if entity_id in arc_params:sparsity[index,arc_params[entity_id][0]]=1
        for node_id in dependencies:
            coordinate=2*node_indices[node_id];sparsity[index,coordinate:coordinate+2]=1
    for index in range(count):sparsity[len(prepared)+index,index]=1
    active=np.flatnonzero(np.asarray(sparsity[:len(prepared)].sum(axis=0)).ravel())
    supported_shapes=[index for index,entity in enumerate(initial_entities)
                      if isinstance(entity.get("source_fit_error_px"),Real) and
                      not isinstance(entity.get("source_fit_error_px"),bool) and
                      math.isfinite(entity["source_fit_error_px"]) and entity["source_fit_error_px"]>=0] if exact_radii else []
    if raw_prior is not None:supported_shapes=[]
    if supported_shapes or raw_prior is not None:
        # Curve priors couple shared endpoints to neighboring primitives and
        # their free center offsets. They are no longer separable coordinate
        # priors, so a variable absent from a dimension may still need to move.
        # Leaving those variables frozen at their old values would optimize a
        # different, artificially restricted shape-preservation objective.
        active=np.arange(count)
    prior_weights=np.full(count,.1)
    if raw_prior is not None:
        # Coordinate regularization is also a discretization of the source
        # curve. A newly inserted joint must not contribute another full
        # coordinate anchor and overpower the immutable boundary objective.
        # This remains a soft stabilizer; no source node becomes a lock.
        node_support=np.zeros(len(nodes))
        for entity in initial_entities:
            length=(math.dist(entity["start"],entity["end"]) if entity["type"]=="LINE" else
                    entity["radius"]*_sweep(entity))
            node_support[node_indices[entity["start_node"]]]+=length/2
            node_support[node_indices[entity["end_node"]]]+=length/2
            if entity["id"] in arc_params:
                prior_weights[arc_params[entity["id"]][0]]=.1*math.sqrt(length/scale)
        prior_weights[:2*len(nodes)]=np.repeat(.1*np.sqrt(node_support/scale),2)
    source_initialization_only = bool(
        (source_observation or {}).get('mask_role') == 'initial_topology_and_soft_localization'
        and ink_prior_diagnostics.get('enabled') is True)
    if source_initialization_only:
        # A fitted CAD vertex is a numerical starting point, not an observed
        # dimensional target. Original ink and the input mask's explicit soft
        # localization remain observations; do not also pull free contacts
        # back to old fitted vertex/center coordinates. Original bounds, arc
        # branches and all publication/equality gates are unchanged.
        prior_weights[:] = 0.
    result['diagnostics']['initial_fit_role'] = {
        'policy': 'initial_guess_only_with_original_source_observations' if source_initialization_only
                  else 'legacy_soft_coordinate_regularization',
        'coordinate_penalty_enabled': bool(np.any(prior_weights)),
        'source_bounds_recentered': False, 'known_constraints_changed': False,
        'input_mask_soft_localization_retained': raw_prior is not None,
        'original_image_objective_retained': ink_prior is not None}
    starting=initial.copy()
    if seed_entities is not None:
        # A previous constrained iterate is a starting point only. Preserve
        # the ORIGINAL graph, priors, source budget, topology and dimensions.
        # Rebuilding the graph from the iterate would silently reset all these
        # guards and accumulate source drift across iterations.
        if (not (source_budget.enabled or numerical_budget.diagnostics['enabled']) or
                not isinstance(seed_entities,list) or len(seed_entities)!=len(baseline)):
            raise ValueError("Feasible restart requires the same bounded source graph")
        seed_nodes={};seed_index={}
        for old,new in zip(baseline,seed_entities):
            if any(old.get(key)!=new.get(key) for key in ("id","type","start_node","end_node","clockwise")):
                raise ValueError("Feasible restart cannot change topology or traversal")
            seed_index[new["id"]]=new
            if new["type"]=="ARC" and (_sweep(new)>math.pi+1e-9)!=branches[new["id"]]["major"]:
                raise ValueError("Feasible restart cannot change the arc branch")
            for end in ("start","end"):
                point=np.asarray(_point(new[end],"Source seed endpoint"),float);node=new[end+"_node"]
                if node in seed_nodes and np.linalg.norm(seed_nodes[node]-point)>1e-8:
                    raise ValueError("Feasible restart shared endpoints disagree")
                seed_nodes[node]=point
        origin=np.array([[node["x"],node["y"]] for node in nodes]).mean(axis=0)
        for node_id,index in node_indices.items():starting[2*index:2*index+2]=(seed_nodes[node_id]-origin)/scale
        for eid,(index,sign) in arc_params.items():
            entity=seed_index[eid];a,b=np.asarray(entity["start"]),np.asarray(entity["end"])
            chord=b-a;normal=np.array([-chord[1],chord[0]])/max(np.linalg.norm(chord),1e-12)
            height=float(np.dot(np.asarray(_point(entity["center"],"Source seed center"))-(a+b)/2,normal))
            if height*sign<0:raise ValueError("Feasible restart circle center branch changed")
            starting[index]=abs(height)/scale
        if not np.isfinite(starting).all() or np.any(starting<lower) or np.any(starting>upper):
            raise ValueError("Feasible restart is outside original variable bounds")
        result["diagnostics"]["source_seed"]={"method":"previous_constrained_iterate_same_original_problem",
            "source_guard_origin_reset":False,"seed_is_accepted_solution":False,"ground_truth_used":False}
    if seed_node_offsets is not None:
        # A final-stage restart may move at most two source-graph joints by
        # two immutable mask sampling steps.  This only changes the numerical
        # starting point: dimensions, fixed-radius elimination, the source
        # budget, and every final acceptance check remain identical.
        if (not (source_budget.enabled or numerical_budget.diagnostics['enabled']) or
                not isinstance(seed_node_offsets,dict) or not 1<=len(seed_node_offsets)<=2):
            raise ValueError("Source restart requires a bounded input-mask observation and one or two node offsets")
        step=(source_budget.step if source_budget.enabled else _number(
            (source_observation.get('mask_representation_budget_diagnostic') or {}).get('sampling_step'),
            'Source restart sampling step'))
        if step<=0:raise ValueError('Source restart needs a positive immutable sampling step')
        maximum_seed_shift=min(2*step,.25 if graph["units"]=="mm" else 2.)
        for node_id,offset in seed_node_offsets.items():
            if node_id not in node_indices:
                raise ValueError("Source restart references an unknown graph node")
            shift=np.asarray(offset,float)
            if shift.shape!=(2,) or not np.isfinite(shift).all() or np.linalg.norm(shift)>maximum_seed_shift+1e-12:
                raise ValueError("Source restart offset exceeds its immutable-mask bound")
            first=2*node_indices[node_id]
            starting[first:first+2]+=shift/scale
        result["diagnostics"]["source_seed"]={"method":"bounded_source_joint_offset",
            "node_ids":sorted(seed_node_offsets),"maximum_offset":maximum_seed_shift,
            "units":graph["units"],"ground_truth_used":False}
    radius_initialization=[]
    radius_chords=[]
    for entity_id,target in exact_radii.items():
        entity=next(entity for entity in baseline if entity["id"]==entity_id)
        first=2*node_indices[entity["start_node"]];last=2*node_indices[entity["end_node"]]
        a=starting[first:first+2].copy();b=starting[last:last+2].copy()
        chord=float(np.linalg.norm(b-a))*scale;original_chord=chord
        if chord>2*target:
            # Closest endpoint-pair initialization on the radius feasibility
            # boundary, not an added constraint or a post-solve repair. All
            # endpoints remain free in the joint optimization and source guard.
            midpoint=(a+b)/2;half=(b-a)*(target/chord)
            starting[first:first+2]=midpoint-half;starting[last:last+2]=midpoint+half
            chord=2*target
        radius_chords.append((entity_id,first,last,2*target/scale))
        radius_initialization.append({"entity_id":entity_id,"target_radius":target,"original_chord":original_chord,
                                      "initialized_chord":chord,"endpoints_remain_free":True})
    def expand(reduced):
        vector=initial.copy();vector[active]=reduced;return vector
    # A fixed radius changes the arc interior even when its endpoints stay
    # put. Retain source-graph curve samples as an additional *soft* prior so
    # shared endpoints can move together to preserve the observed curve, not
    # just its chord. Nothing here reads a reference or an image. Equal sample
    # quadrature normalized by arclength avoids giving short split primitives
    # more influence than long source-supported arcs.
    shape_sample_count=9
    source_shapes={index:_sample_entity(initial_entities[index],shape_sample_count) for index in supported_shapes}
    source_lengths=np.array([math.dist(entity["start"],entity["end"]) if entity["type"]=="LINE" else
                             entity["radius"]*_sweep(entity) for entity in initial_entities])
    shape_weights=SOURCE_CURVE_PRIOR_WEIGHT*np.sqrt(len(nodes)*source_lengths/max(float(source_lengths.sum()),1e-12)/shape_sample_count)
    strict_relation_indices=[index for index,row in enumerate(prepared)
                             if row["kind"]=="tangent" or row["id"] in strict_angle_by_id or _strict_measurement_row(row) or
                             row in strict_general_angles]
    def objective_residuals(reduced):
        # Shared, pure evaluation also permits final certification of a saved
        # feasible point after the bounded search has ended. It never starts
        # another optimization or changes the objective/acceptance equations.
        vector=expand(reduced)
        dimensional_residuals=constraint_residuals(vector)
        # Exact relation rows are imposed by the constrained optimizer. Adding
        # their steep short-LINE derivatives to the soft objective as well
        # needlessly damages conditioning and can overwhelm source priors.
        dimensional_residuals[strict_relation_indices]=0.
        residuals=[dimensional_residuals,prior_weights[active]*(reduced-initial[active])]
        if supported_shapes:
            _,decoded=decode(vector)
            residuals.extend((shape_weights[index]*(_sample_entity(decoded[index],shape_sample_count)-source_shapes[index])/scale).ravel()
                             for index in supported_shapes)
        if raw_prior is not None:
            _,decoded=decode(vector)
            residuals.append(raw_prior(decoded))
        if ink_prior is not None:
            residuals.append(ink_prior(decode(vector)[1]))
        return np.concatenate(residuals)
    def objective(reduced):
        numerical_budget.guard()
        if source_budget.enabled and source_budget.search_active:source_budget.guard(objective=True)
        return objective_residuals(reduced)
    numerical_checkpoint=None
    continuation_checkpoint=None
    continuation_score=math.inf
    def retain_continuation(value):
        """Save a finite search seed, never assert that its equations pass."""
        nonlocal continuation_checkpoint,continuation_score
        if not numerical_budget.diagnostics['enabled']:return
        value=np.asarray(value,float)
        if (value.shape!=starting[active].shape or not np.isfinite(value).all() or
                np.any(value<lower[active]) or np.any(value>upper[active])):return
        vector=expand(value);nmap,decoded=decode(vector)
        geometry=_geometry_validation(decoded,branches,scale)
        if not geometry['passed'] or base_validation['signed_area']*geometry['signed_area']<=0:return
        limit=max(.25 if graph['units']=='mm' else 2.,scale*MAX_DISPLACEMENT_FRACTION)
        if any(float(np.linalg.norm(_sample_entity(new)-_sample_entity(old),axis=1).max())>limit
               for new,old in zip(decoded,initial_entities)):return
        residual=constraint_residuals(vector)
        score=float(np.linalg.norm(residual))
        if math.isfinite(score) and score<continuation_score:
            continuation_checkpoint=value.copy();continuation_score=score
    numerical_checkpoint_cost=math.inf
    numerical_checkpoint_selected=False
    numerical_checkpoint_iterations=0
    sparse_active=lil_matrix((len(prepared)+len(active),len(active)),dtype=int)
    sparse_active[:len(prepared)]=sparsity[:len(prepared),active]
    for index in range(len(active)):sparse_active[len(prepared)+index,index]=1
    try:
        linear_options={} if len(active)<=96 else {"jac_sparsity":sparse_active.tocsr(),"tr_solver":"lsmr",
                                                   "tr_options":{"atol":1e-12,"btol":1e-12,"maxiter":max(100,len(active)*5)}}
        if strict_tangents and not exact_radii and raw_prior is None and not source_budget.enabled:
            # With unconstrained ARC radii, a very short adjacent LINE can have
            # an angular derivative thousands of times larger than the arc.
            # A bounded least-squares seed approaches the relation manifold
            # before SLSQP enforces the exact equalities. This is initialization
            # only: no approximate seed is accepted or used as a certificate.
            def relation_seed_objective(reduced):
                return np.r_[constraint_residuals(expand(reduced)),
                             prior_weights[active]*(reduced-initial[active])]
            seed=least_squares(relation_seed_objective,starting[active],
                bounds=(lower[active],upper[active]),max_nfev=200,
                ftol=1e-11,xtol=1e-11,gtol=1e-9,x_scale=1.,**linear_options)
            if np.isfinite(seed.x).all():starting[active]=seed.x
            result["diagnostics"]["strict_relation_initialization"]={
                "method":"bounded_relation_least_squares_seed_then_equality_solve",
                "evaluations":int(seed.nfev),"converged":bool(seed.success),
                "seed_is_accepted_solution":False,"thresholds_changed":False}
        # Coordinates and free center offsets share one bounding-box-normalized
        # length unit. Jacobian column rescaling would distort their common
        # source-prior metric, especially for short lines next to large arcs.
        # The unannotated path retains bounded least squares. Exact radii need
        # a constrained optimizer because their endpoint chords must fit their
        # annotated diameters; neither branch relaxes the acceptance checks.
        if exact_radii or strict_angles or strict_general_angles or strict_tangents or strict_measurements or raw_prior is not None or source_budget.enabled:
            def scalar_objective(reduced):
                residuals=objective(reduced)
                return float(np.dot(residuals,residuals)/2)
            def scalar_gradient(reduced):
                return _finite_difference_jacobian(objective,reduced).T@objective(reduced)
            def chord_feasibility(reduced):
                vector=expand(reduced)
                return np.array([(diameter**2-float(np.dot(vector[b:b+2]-vector[a:a+2],vector[b:b+2]-vector[a:a+2])))
                                 /max(diameter**2,1e-20) for _,a,b,diameter in radius_chords])
            def chord_jacobian(reduced):
                vector=expand(reduced);jacobian=np.zeros((len(radius_chords),count))
                for index,(_,a,b,diameter) in enumerate(radius_chords):
                    delta=2*(vector[b:b+2]-vector[a:a+2])/max(diameter**2,1e-20)
                    jacobian[index,a:a+2]=delta;jacobian[index,b:b+2]=-delta
                return jacobian[:,active]
            optimizer_constraints=[{"type":"ineq","fun":chord_feasibility,"jac":chord_jacobian}] if exact_radii else []
            equality_functions=[];equality_jacobians=[];equality_ids=[]
            if strict_measurements:
                def measurement_equalities(reduced):
                    nmap,decoded=decode(expand(reduced));emap={entity["id"]:entity for entity in decoded}
                    return np.array([_constraint_residual(row,_constraint_value(
                        row,nmap,emap,allow_ill_conditioned=True))/scale for row in strict_measurements])
                def measurement_jacobian(reduced):
                    return _finite_difference_jacobian(measurement_equalities,reduced)
                equality_functions.append(measurement_equalities)
                equality_jacobians.append(measurement_jacobian)
                equality_ids.extend(row["id"] for row in strict_measurements)
                derived_terms={json.dumps(anchor,sort_keys=True):anchor
                    for row in strict_measurements for anchor in row.get("anchors",[])
                    if anchor["type"] in {"support_level","curve_extremum","curve_center","common_curve_center"}}
                level_terms=[anchor for anchor in derived_terms.values()
                             if anchor["type"] in {"support_level","common_curve_center"}]
                if level_terms:
                    def measurement_support_equalities(reduced):
                        _,decoded=decode(expand(reduced));emap={entity["id"]:entity for entity in decoded}
                        return np.array([value/scale for anchor in level_terms
                                         for value in derived_anchor_equalities(anchor,emap)])
                    equality_functions.append(measurement_support_equalities)
                    equality_jacobians.append(lambda value:_finite_difference_jacobian(measurement_support_equalities,value))
                    equality_ids.extend(anchor["type"]+"_"+anchor["entity_id"]+"_"+
                        anchor.get("other_entity_id", anchor["axis"])+"_"+str(index)
                        for anchor in level_terms
                        for index in range(2 if anchor["type"] == "common_curve_center" else 1))
                if derived_terms:
                    def measurement_support_domain(reduced):
                        _,decoded=decode(expand(reduced));emap={entity["id"]:entity for entity in decoded}
                        return np.array([value for anchor in derived_terms.values()
                                         for value in derived_anchor_domain_conditions(anchor,emap)])
                    optimizer_constraints.append({"type":"ineq","fun":measurement_support_domain})
                source_entities={entity["id"]:entity for entity in baseline};intersection_terms={}
                for row in strict_measurements:
                    for anchor in row.get("anchors",[]):
                        if anchor["type"]=="line_intersection":
                            condition,_=_measurement_intersection_condition(anchor,source_entities)
                            intersection_terms[tuple(anchor["entities"])]=(anchor,math.copysign(1.,condition))
                if intersection_terms:
                    def measurement_intersection_feasibility(reduced):
                        _,decoded=decode(expand(reduced));emap={entity["id"]:entity for entity in decoded}
                        minimum=math.sin(math.radians(MIN_MEASUREMENT_INTERSECTION_ANGLE_DEG))
                        rows=[]
                        for anchor,sign in intersection_terms.values():
                            condition,lengths=_measurement_intersection_condition(anchor,emap)
                            rows.extend([sign*condition-minimum,*[length/scale-1e-9/scale for length in lengths]])
                        return np.array(rows)
                    optimizer_constraints.append({"type":"ineq","fun":measurement_intersection_feasibility})
            if strict_general_angles:
                def nominal_angle_equalities(reduced):
                    nmap,decoded=decode(expand(reduced));emap={entity["id"]:entity for entity in decoded}
                    values=[]
                    for row in strict_general_angles:
                        if _strict_axis_relation(row):
                            # An axis is a linear endpoint equation. Using a
                            # wrapped atan2 here adds an unnecessary branch
                            # and poorly scaled derivatives near an exact axis.
                            a,b=(nmap[node] for node in row['nodes'])
                            values.append(float((b-a)[1 if row['kind']=='horizontal' else 0])/scale)
                        else:
                            values.append(math.radians(_constraint_residual(row,_constraint_value(row,nmap,emap))))
                    return np.array(values)
                equality_functions.append(nominal_angle_equalities)
                equality_jacobians.append(lambda value:_finite_difference_jacobian(nominal_angle_equalities,value))
                equality_ids.extend(row["id"] for row in strict_general_angles)
            if strict_angles:
                # One fixed, source-graph quadrant turns an acute unsigned
                # angle into a linear equation on the LINE's shared endpoints.
                # This avoids an abs(atan2(...)) equality's cusp and keeps the
                # constraint active in warm, feasibility, RMS and restart paths.
                angle_terms=[]
                seen_angle_entities=set()
                for spec in strict_angles:
                    if spec["entity_id"] in seen_angle_entities:continue
                    seen_angle_entities.add(spec["entity_id"])
                    direction=spec["direction"]
                    a=2*node_indices[spec["start_node"]];b=2*node_indices[spec["end_node"]]
                    normal=np.array([-direction[1],direction[0]])
                    length=max(spec["initial_length"]/scale,1e-12)
                    angle_terms.append((a,b,normal,direction,length))
                def angle_direction_equalities(reduced):
                    vector=expand(reduced)
                    return np.array([float(np.dot(normal,vector[b:b+2]-vector[a:a+2]))/length
                                     for a,b,normal,_,length in angle_terms])
                def angle_direction_jacobian(reduced):
                    jacobian=np.zeros((len(angle_terms),count))
                    for index,(a,b,normal,_,length) in enumerate(angle_terms):
                        jacobian[index,a:a+2]=-normal/length
                        jacobian[index,b:b+2]=normal/length
                    return jacobian[:,active]
                def angle_quadrant_feasibility(reduced):
                    vector=expand(reduced)
                    return np.array([float(np.dot(direction,vector[b:b+2]-vector[a:a+2]))/length-1e-10
                                     for a,b,_,direction,length in angle_terms])
                def angle_quadrant_jacobian(reduced):
                    jacobian=np.zeros((len(angle_terms),count))
                    for index,(a,b,_,direction,length) in enumerate(angle_terms):
                        jacobian[index,a:a+2]=-direction/length
                        jacobian[index,b:b+2]=direction/length
                    return jacobian[:,active]
                equality_functions.append(angle_direction_equalities)
                equality_jacobians.append(angle_direction_jacobian)
                equality_ids.extend(next(spec["row"]["id"] for spec in strict_angles
                                         if spec["entity_id"]==entity_id) for entity_id in
                                    dict.fromkeys(spec["entity_id"] for spec in strict_angles))
                optimizer_constraints.append(
                    {"type":"ineq","fun":angle_quadrant_feasibility,"jac":angle_quadrant_jacobian})
            if strict_tangents:
                def tangent_vectors(reduced):
                    _,decoded=decode(expand(reduced));indexed={entity["id"]:entity for entity in decoded}
                    return [tuple(_tangent(indexed[eid],indexed[eid]["end_node"]==row["nodes"][0])
                                  for eid in row["entities"]) for row in strict_tangents]
                def tangent_equalities(reduced):
                    return np.array([a[0]*b[1]-a[1]*b[0] for a,b in tangent_vectors(reduced)])
                def tangent_jacobian(reduced):
                    return _finite_difference_jacobian(tangent_equalities,reduced)
                def tangent_forward_feasibility(reduced):
                    return np.array([float(np.dot(a,b))-1e-8 for a,b in tangent_vectors(reduced)])
                equality_functions.append(tangent_equalities);equality_jacobians.append(tangent_jacobian)
                equality_ids.extend(row["id"] for row in strict_tangents)
                optimizer_constraints.append({"type":"ineq","fun":tangent_forward_feasibility})
            if equality_functions:
                def all_equalities(reduced):
                    return np.concatenate([function(reduced) for function in equality_functions])
                def all_equality_jacobians(reduced):
                    return np.vstack([function(reduced) for function in equality_jacobians])
                equality_basis,equality_rank_tolerance=_independent_equality_rows(
                    all_equality_jacobians(starting[active]))
                omitted=np.setdiff1d(np.arange(len(equality_ids)),equality_basis)
                if len(equality_basis):
                    optimizer_constraints.append({"type":"eq",
                        "fun":lambda value:all_equalities(value)[equality_basis],
                        "jac":lambda value:all_equality_jacobians(value)[equality_basis]})
                if len(omitted):
                    # Numerically dependent rows cannot all enter SLSQP's
                    # equality matrix. Keep them guarded well inside the same
                    # strict floating-point certificate and recheck ALL rows.
                    reserve=.25*min(math.sin(math.radians(min(STRICT_TANGENT_CERT_TOLERANCE_DEG,
                                                         STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG))),
                                   STRICT_MEASUREMENT_CERT_TOLERANCE_MM/scale if strict_measurements else math.inf)
                    def dependent_equality_guard(value):
                        residual=all_equalities(value)[omitted]
                        return np.r_[reserve-residual,reserve+residual]
                    def dependent_equality_jacobian(value):
                        jacobian=all_equality_jacobians(value)[omitted]
                        return np.vstack([-jacobian,jacobian])
                    optimizer_constraints.append({"type":"ineq","fun":dependent_equality_guard,
                                                  "jac":dependent_equality_jacobian})
                result["diagnostics"]["strict_equality_system"]={
                    "required_count":len(equality_ids),"optimizer_basis_count":len(equality_basis),
                    "basis_constraint_ids":[equality_ids[index] for index in equality_basis],
                    "dependent_constraint_ids":[equality_ids[index] for index in omitted],
                    "rank_tolerance":equality_rank_tolerance,
                    "dependent_rows_numerically_guarded":True,"all_original_rows_finally_certified":True}
            if source_budget.enabled:
                # Unlike RMS penalties, a dimensional inequality cannot trade
                # a correctly bound annotation against a better source fit.
                # The small inward reserve prevents numerical roundoff from
                # passing SLSQP while failing the unchanged final tolerance.
                def dimensional_feasibility(reduced):
                    source_budget.guard()
                    residual=constraint_residuals(expand(reduced))
                    return np.r_[1.-1e-5-residual,1.-1e-5+residual]
                def boundary_feasibility(reduced):
                    source_budget.guard()
                    return source_budget.witness_constraints(decode(expand(reduced))[1])
                movement_limit=max(.25 if graph["units"]=="mm" else 2.,scale*MAX_DISPLACEMENT_FRACTION)
                # The complete node-displacement guard already requires each
                # coordinate to stay inside this box. Expose that NECESSARY
                # bound to SLSQP instead of relying only on max(norm(...)),
                # whose derivative constrains a single worst-moving node.
                # The old generic +/-10 normalized box permitted trial jumps
                # hundreds of times farther than the unchanged movement guard;
                # an infeasible line search could then destroy arc incidence
                # and stop separation-oracle refinement prematurely.
                node_box_radius=movement_limit/scale
                node_variable_count=2*len(nodes)
                lower[:node_variable_count]=np.maximum(lower[:node_variable_count],
                    initial[:node_variable_count]-node_box_radius)
                upper[:node_variable_count]=np.minimum(upper[:node_variable_count],
                    initial[:node_variable_count]+node_box_radius)
                source_budget.diagnostics["node_coordinate_box"]={
                    "derived_from":"unchanged_complete_node_displacement_guard",
                    "maximum_coordinate_displacement":movement_limit,"units":graph["units"],
                    "guard_implied_bound_only":True,"radial_movement_guard_still_active":True,
                    "free_arc_parameter_bounds_changed":False}
                initial_curve_samples=np.asarray([_sample_entity(entity) for entity in initial_entities])
                def movement_feasibility(reduced):
                    source_budget.guard()
                    vector=expand(reduced)
                    node_distance=np.linalg.norm((vector[:2*len(nodes)]-initial[:2*len(nodes)]).reshape((-1,2)),axis=1)*scale
                    curves=np.asarray([_sample_entity(entity) for entity in decode(vector)[1]])
                    curve_distance=np.linalg.norm(curves-initial_curve_samples,axis=2)
                    # These are the existing final node/curve displacement
                    # guards, now supplied to the optimizer at the same limit.
                    return np.array([1.-float(node_distance.max())/movement_limit,
                                     1.-float(curve_distance.max())/movement_limit])
                warm_constraints=optimizer_constraints.copy()
                optimizer_constraints.extend([{"type":"ineq","fun":dimensional_feasibility},
                                              {"type":"ineq","fun":boundary_feasibility},
                                              {"type":"ineq","fun":movement_feasibility}])
                reduced=np.clip(starting[active],lower[active],upper[active])
                best_feasible=None;best_valid_unaccepted=None;total_evaluations=0;total_iterations=0
                source_budget.diagnostics.update(maximum_feasibility_iterations_per_round=120,
                    maximum_RMS_iterations_per_round=180,maximum_warm_RMS_iterations=160 if source_warm_start else 80,dimensional_tolerances_changed=False,
                    strict_ocr_angle_equalities_active=bool(strict_angles),
                    strict_source_nominal_angle_equalities_active=bool(strict_general_angles),
                    strict_tangent_equalities_active=bool(strict_tangents),
                    strict_measurement_equalities_active=bool(strict_measurements),
                    feasibility_objective="minimum_movement_regularizer; annotations, boundary and unchanged displacement guard are inequalities",
                    refinement_stop_policy="shared_time_or_objective_budget; no_new_violating_witnesses; finite_round_ceiling",
                    maximum_error_plateau_is_stop_condition=False)
                feasibility_target=starting[active].copy()
                def feasibility_objective(value):
                    source_budget.guard(objective=True)
                    return float(np.dot(value-feasibility_target,value-feasibility_target)/2)
                def feasibility_gradient(value):return value-feasibility_target
                fit=OptimizeResult(x=reduced.copy(),jac=np.zeros(len(reduced)),success=False,status=9,nfev=0,nit=0,
                                   message="Source-budget search has no converged checkpoint")
                def finite_checkpoint(value):
                    try: vector=np.asarray(value,dtype=float)
                    except (TypeError,ValueError):return False
                    return bool(vector.shape==reduced.shape and np.isfinite(vector).all() and
                                np.all(vector>=lower[active]) and np.all(vector<=upper[active]))
                def fixed_subset_passed(value):
                    return bool(np.max(np.abs(constraint_residuals(expand(value))))<=1. and
                                (not exact_radii or np.min(chord_feasibility(value))>=-1e-10) and
                                (not strict_angles or _strict_ocr_angle_contract(
                                    decode(expand(value))[1],strict_angles)["satisfied"]) and
                                (not strict_general_angles or _strict_general_angle_contract(
                                    *decode(expand(value)),strict_general_angles)["satisfied"]) and
                                (not strict_tangents or _strict_tangent_contract(
                                    decode(expand(value))[1],strict_tangents)["satisfied"]) and
                                (not strict_measurements or _strict_measurement_contract(
                                    *decode(expand(value)),strict_measurements,baseline)["satisfied"]))
                def complete_geometry_passed(value):
                    nmap,decoded=decode(expand(value));guard=_geometry_validation(decoded,branches,scale)
                    node_distance=max(float(np.linalg.norm(nmap[node["id"]]-np.array([node["x"],node["y"]]))) for node in nodes)
                    curves=np.asarray([_sample_entity(entity) for entity in decoded])
                    curve_distance=float(np.linalg.norm(curves-initial_curve_samples,axis=2).max())
                    return bool(guard["passed"] and base_validation["signed_area"]*guard["signed_area"]>0 and
                                max(node_distance,curve_distance)<=movement_limit)
                try:
                    initial_budget_audit=source_budget.safe_inspect(decode(expand(reduced))[1])
                    source_budget.diagnostics["initial_exact_radius_seed_audit"]=initial_budget_audit
                    # Spend the fixed budget on feasibility first when the
                    # exact-radius seed already violates the source boundary.
                    # An unconstrained RMS warm-up cannot certify that budget.
                    warm_audit=initial_budget_audit;warm_passed=False
                    if initial_budget_audit["passed"] or source_warm_start:
                        if initial_budget_audit["passed"] and fixed_subset_passed(reduced) and complete_geometry_passed(reduced):
                            best_feasible=copy.deepcopy(fit)
                            best_feasible.message="Original-problem seed independently revalidated before optional refinement"
                            checkpoint_key=("input_seed_feasible_checkpoint_revalidated" if seed_entities is not None
                                            else "initial_graph_feasible_checkpoint_revalidated")
                            source_budget.diagnostics[checkpoint_key]=True
                        # A source-owned topology repair may start outside the
                        # mask tube. Approach its exact annotation manifold
                        # before the separation-oracle search. This is only a
                        # numerical seed: the unchanged source budget, original
                        # graph displacement domain and every native equation
                        # still decide acceptance below. Both phases share the
                        # same wall/evaluation budget.
                        warm=minimize(scalar_objective,reduced,method="SLSQP",jac=scalar_gradient,
                            bounds=list(zip(lower[active],upper[active])),constraints=warm_constraints,
                            options={"maxiter":160 if source_warm_start else 80,"ftol":1e-12})
                        total_evaluations+=int(warm.nfev);total_iterations+=int(warm.nit)
                        fit=warm;reduced=warm.x.copy()
                        warm_finite=finite_checkpoint(reduced)
                        warm_audit=(source_budget.safe_inspect(decode(expand(reduced))[1]) if warm_finite else
                                    {"passed":False,"reason":"nonfinite_or_out_of_bounds_checkpoint"})
                        warm_passed=bool(warm_finite and warm_audit["passed"] and
                                         fixed_subset_passed(reduced) and complete_geometry_passed(reduced))
                        if warm_passed:
                            best_feasible=copy.deepcopy(warm)
                            source_budget.diagnostics["stop_reason"]="warm_source_feasible_checkpoint_found"
                        source_budget.diagnostics.update(warm_RMS_audit=warm_audit,warm_RMS_converged=bool(warm.success),
                                                         warm_RMS_reused=bool(warm_passed and warm.success),
                                                         warm_feasible_checkpoint_found=warm_passed)
                        if source_warm_start:
                            manifold_seed=bool(warm_finite and fixed_subset_passed(reduced) and
                                               complete_geometry_passed(reduced))
                            source_budget.diagnostics["annotation_manifold_warm_start"]={
                                "enabled":True,"maximum_iterations":160,
                                "strict_subset_and_original_domain_passed":manifold_seed,
                                "mask_budget_passed":bool(warm_audit.get("passed")),
                                "seed_is_publication_certificate":False,
                                "original_graph_and_source_budget_unchanged":True,
                                "shared_wall_and_objective_budget":True}
                            if manifold_seed:
                                feasibility_target=reduced.copy()
                                best_valid_unaccepted=(warm_audit.get("conservative_max_deviation",math.inf),copy.deepcopy(warm))
                    else:
                        source_budget.diagnostics.update(warm_RMS_skipped=True,
                            warm_RMS_skip_reason="initial_source_budget_not_satisfied",warm_RMS_reused=False)
                    if not warm_passed:
                        keep_manifold_seed=bool(source_warm_start and
                            (source_budget.diagnostics.get("annotation_manifold_warm_start") or {}).get(
                                "strict_subset_and_original_domain_passed"))
                        if not keep_manifold_seed and (not warm_audit.get("geometry_valid") or
                            initial_budget_audit.get("conservative_max_deviation",math.inf)<warm_audit.get("conservative_max_deviation",math.inf)):
                            reduced=starting[active].copy()
                            source_budget.diagnostics["initial_seed_retained_after_unconverged_warm_RMS"]=True
                        for round_index in range(SOURCE_BUDGET_SEARCH_ROUNDS):
                            source_budget.guard()
                            before=source_budget.safe_inspect(decode(expand(reduced))[1],add_witnesses=True)
                            source_budget.guard()
                            if before.get("audit_error_type"):
                                source_budget.diagnostics.update(stop_reason="source_boundary_audit_unavailable",
                                                                 stop_audit=before)
                                break
                            witness_update=before.get("witness_update",{})
                            if round_index>0 and witness_update.get("new_witness_count")==0:
                                # Repeating the same projection cannot discover
                                # a new source constraint. Do not confuse this
                                # with a maximum-error plateau while distinct
                                # violating witnesses are still being added.
                                source_budget.diagnostics.update(stop_reason="no_new_violating_witnesses",
                                    stop_audit=before,
                                    existing_witness_minimum_margin=float(np.min(boundary_feasibility(reduced)))
                                        if source_budget.forward or source_budget.reverse else None)
                                break
                            feasibility_fit=minimize(feasibility_objective,reduced,method="SLSQP",jac=feasibility_gradient,
                                bounds=list(zip(lower[active],upper[active])),constraints=optimizer_constraints,
                                options={"maxiter":120,"ftol":1e-12})
                            total_evaluations+=int(feasibility_fit.nfev);total_iterations+=int(feasibility_fit.nit)
                            reduced=feasibility_fit.x.copy()
                            candidate_finite=finite_checkpoint(reduced)
                            feasible_audit=(source_budget.safe_inspect(decode(expand(reduced))[1]) if candidate_finite else
                                            {"passed":False,"reason":"nonfinite_or_out_of_bounds_checkpoint"})
                            source_budget.guard()
                            subset_passed=bool(candidate_finite and fixed_subset_passed(reduced))
                            geometry_passed=bool(candidate_finite and complete_geometry_passed(reduced))
                            # SLSQP's iteration-limit status concerns optimality. A
                            # completed iterate can independently satisfy every
                            # unchanged constraint, geometry and source audit.
                            feasibility_passed=bool(candidate_finite and feasible_audit["passed"] and
                                                    subset_passed and geometry_passed)
                            row={"round":round_index+1,"before":before,"feasibility_audit":feasible_audit,
                                 "feasibility_converged":bool(feasibility_fit.success),"annotation_subset_and_chords_passed":subset_passed,
                                 "finite_and_within_bounds":candidate_finite,"complete_geometry_passed":geometry_passed,
                                 "feasibility_found":feasibility_passed,"forward_witness_count":len(source_budget.forward),
                                 "reverse_witness_count":len(source_budget.reverse),"feasibility_iterations":int(feasibility_fit.nit),
                                 "witness_update":witness_update,
                                 "elapsed_wall_seconds":time.monotonic()-source_budget.started}
                            source_budget.diagnostics["rounds"].append(row)
                            fit=feasibility_fit
                            if candidate_finite and subset_passed and geometry_passed and not feasible_audit.get("audit_error_type"):
                                deviation=feasible_audit.get("conservative_max_deviation",math.inf)
                                if best_valid_unaccepted is None or deviation<best_valid_unaccepted[0]:
                                    best_valid_unaccepted=(deviation,copy.deepcopy(feasibility_fit))
                            if (not candidate_finite or not geometry_passed or feasible_audit.get("audit_error_type")) and best_valid_unaccepted is not None:
                                # Preserve an earlier strict-equation/geometry
                                # checkpoint as the next caller's restart seed.
                                # It remains REJECTED until the original full
                                # source budget passes; an invalid later trial
                                # must not hide the actual source-only failure.
                                fit=copy.deepcopy(best_valid_unaccepted[1]);reduced=fit.x.copy()
                                source_budget.diagnostics.update(
                                    stop_reason="invalid_trial_retained_valid_source_search_checkpoint",
                                    invalid_trial_replaced_by_valid_checkpoint=True,
                                    retained_checkpoint_source_deviation=best_valid_unaccepted[0],
                                    retained_checkpoint_is_certified_solution=False)
                                break
                            if not candidate_finite:
                                source_budget.diagnostics["stop_reason"]="nonfinite_or_out_of_bounds_checkpoint"
                                break
                            if not feasibility_passed:continue
                            best_feasible=copy.deepcopy(feasibility_fit)
                            if output_dir is not None:
                                checkpoint=Path(output_dir);checkpoint.mkdir(parents=True,exist_ok=True)
                                (checkpoint/"source-budget-checkpoint.json").write_text(json.dumps({
                                    "schema_version":"source-budget-feasible-checkpoint-v1","units":graph["units"],
                                    "entities":decode(expand(reduced))[1],"source_boundary_audit":feasible_audit,
                                    "annotation_subset_and_chords_passed":True,"engineering_verified":False,
                                    "optimizer_converged":bool(feasibility_fit.success),"optimality_proven":False,
                                    "acceptance_basis":"independently_verified_feasible_checkpoint",
                                    "strict_ocr_angle_contract":_strict_ocr_angle_contract(
                                        decode(expand(reduced))[1],strict_angles),
                                    "strict_tangent_contract":_strict_tangent_contract(
                                        decode(expand(reduced))[1],strict_tangents),
                                    "source_image_gate_not_evaluated":True,"ground_truth_used":False,
                                    "round":round_index+1},ensure_ascii=False,indent=2),encoding="utf8")
                            if not feasibility_fit.success:
                                # Feasibility has been checked independently at
                                # this exact finite iterate. Optional RMS work
                                # cannot improve its feasibility certificate and
                                # must not consume the remaining topology budget.
                                fit=copy.deepcopy(best_feasible)
                                source_budget.diagnostics["RMS_refinement_skipped_reason"]="feasible_iteration_limit_checkpoint"
                                source_budget.diagnostics["stop_reason"]="source_feasible_checkpoint_found"
                                break
                            # Feasibility is already independently certified.
                            # Optional RMS optimality must not spend the whole
                            # shared request allowance and make that checkpoint
                            # too late for the caller's source/DXF publication
                            # audits. Its expiration is not global failure.
                            remaining=source_budget.wall_limit-(time.monotonic()-source_budget.started)
                            rms_seconds=min(5.,max(0.,remaining-5.))
                            row.update(RMS_wall_allowance_seconds=rms_seconds,publication_reserve_seconds=5.)
                            if rms_seconds<=0:
                                fit=copy.deepcopy(best_feasible)
                                source_budget.diagnostics.update(
                                    RMS_refinement_skipped_reason="shared_budget_reserved_for_publication",
                                    stop_reason="source_feasible_checkpoint_found")
                                break
                            rms_started=time.monotonic()
                            source_budget.optional_refinement_deadline=rms_started+rms_seconds
                            try:
                                rms_fit=minimize(scalar_objective,reduced,method="SLSQP",jac=scalar_gradient,
                                    bounds=list(zip(lower[active],upper[active])),constraints=optimizer_constraints,
                                    options={"maxiter":180,"ftol":1e-12})
                                total_evaluations+=int(rms_fit.nfev);total_iterations+=int(rms_fit.nit)
                                rms_finite=finite_checkpoint(rms_fit.x)
                                rms_audit=(source_budget.safe_inspect(decode(expand(rms_fit.x))[1]) if rms_finite else
                                           {"passed":False,"reason":"nonfinite_or_out_of_bounds_checkpoint"})
                                source_budget.guard()
                            except _OptionalSourceRefinementFinished:
                                fit=copy.deepcopy(best_feasible)
                                row.update(RMS_subbudget_exhausted=True,RMS_accepted=False,
                                           RMS_checkpoint="prior_independently_certified_feasible_point")
                                source_budget.diagnostics.update(
                                    RMS_refinement_skipped_reason="optional_RMS_wall_subbudget_exhausted",
                                    stop_reason="source_feasible_checkpoint_retained_after_optional_RMS_budget")
                                break
                            finally:
                                source_budget.optional_refinement_deadline=None
                                row["RMS_elapsed_wall_seconds"]=time.monotonic()-rms_started
                            rms_feasible=bool(rms_finite and rms_audit["passed"] and
                                              fixed_subset_passed(rms_fit.x) and complete_geometry_passed(rms_fit.x))
                            rms_passed=bool(rms_fit.success and rms_feasible)
                            row.update(RMS_audit=rms_audit,RMS_converged=bool(rms_fit.success),RMS_accepted=rms_passed,
                                       RMS_feasible_checkpoint_found=rms_feasible,
                                       RMS_iterations=int(rms_fit.nit))
                            if rms_feasible:best_feasible=copy.deepcopy(rms_fit)
                            if rms_passed:
                                fit=rms_fit
                                source_budget.diagnostics["stop_reason"]="source_feasible_RMS_checkpoint_found"
                                break
                            reduced=rms_fit.x.copy() if rms_finite else best_feasible.x.copy()
                        else:
                            if best_feasible is not None:fit=best_feasible
                            source_budget.diagnostics["stop_reason"]="maximum_refinement_rounds_reached"
                except _SourceBudgetExhausted:
                    source_budget.diagnostics.setdefault("stop_reason","source_budget_interrupted")
                    if best_feasible is not None:
                        fit=copy.deepcopy(best_feasible)
                        fit.message="Source-feasible checkpoint retained after RMS search budget exhaustion; no optimality claim"
                    else:
                        fit.success=False;fit.status=9;fit.message="Source-budget wall/evaluation limit exhausted; baseline retained"
                source_budget.search_active=False;source_budget.progress()
                source_budget.diagnostics.update(total_optimizer_evaluations=total_evaluations,
                    total_optimizer_iterations=total_iterations,feasible_checkpoint_found=best_feasible is not None,
                    completed_refinement_rounds=len(source_budget.diagnostics["rounds"]),
                    RMS_refinement_accepted=bool(source_budget.diagnostics.get("warm_RMS_reused") or
                                                (source_budget.diagnostics["rounds"] and source_budget.diagnostics["rounds"][-1].get("RMS_accepted"))),
                    fallback_meaning="Bounded search failure is not proof that the annotations and source budget are incompatible.")
            else:
                def retain_numerical_checkpoint(value):
                    """Certify a complete point, never a partially satisfied iterate."""
                    nonlocal numerical_checkpoint,numerical_checkpoint_cost,numerical_checkpoint_iterations
                    numerical_checkpoint_iterations+=1
                    if not numerical_budget.diagnostics['enabled']:return
                    value=np.asarray(value,float)
                    if (value.shape!=starting[active].shape or not np.isfinite(value).all() or
                            np.any(value<lower[active]) or np.any(value>upper[active])):return
                    vector=expand(value);nmap,decoded=decode(vector)
                    if (np.max(np.abs(constraint_residuals(vector)))>1. or
                            (exact_radii and np.min(chord_feasibility(value)) < -1e-10) or
                            not _strict_ocr_angle_contract(decoded,strict_angles)['satisfied'] or
                            not _strict_general_angle_contract(nmap,decoded,strict_general_angles)['satisfied'] or
                            not _strict_tangent_contract(decoded,strict_tangents)['satisfied'] or
                            not _strict_measurement_contract(nmap,decoded,strict_measurements,baseline)['satisfied']):return
                    validation=_geometry_validation(decoded,branches,scale)
                    limit=max(.25 if graph['units']=='mm' else 2.,scale*MAX_DISPLACEMENT_FRACTION)
                    node_shift=max(float(np.linalg.norm(nmap[node['id']]-[node['x'],node['y']])) for node in nodes)
                    curve_shift=max(float(np.linalg.norm(_sample_entity(after)-_sample_entity(before),axis=1).max())
                                    for before,after in zip(initial_entities,decoded))
                    if (not validation['passed'] or base_validation['signed_area']*validation['signed_area']<=0 or
                            max(node_shift,curve_shift)>limit):return
                    # Native contracts above check the complete source subset,
                    # including dependent equalities and finite anchor domains.
                    # Remaining optimizer inequalities also remain mandatory.
                    for row in optimizer_constraints:
                        if row['type']=='ineq' and np.any(np.asarray(row['fun'](value)) < -1e-10):return
                    residual=objective_residuals(value)
                    cost=float(residual@residual/2)
                    if math.isfinite(cost) and cost<numerical_checkpoint_cost:
                        numerical_checkpoint=value.copy();numerical_checkpoint_cost=cost
                # A certified input seed is rechecked against THIS complete
                # problem. A Boolean from an earlier solve is never trusted.
                def safe_retain_numerical_checkpoint(value):
                    try:
                        retain_continuation(value)
                        retain_numerical_checkpoint(value)
                    except (ValueError,FloatingPointError,ArithmeticError,np.linalg.LinAlgError):
                        # Infeasible intermediate branches are ordinary search
                        # states, not grounds to erase a previously valid point.
                        return
                safe_retain_numerical_checkpoint(starting[active])
                if (seed_entities is not None and numerical_budget.diagnostics['enabled']
                        and equality_functions and numerical_checkpoint is None):
                    # Restore feasibility before optimizing image agreement.
                    # A near-correct old solve otherwise spends its whole clock
                    # differentiating expensive ink costs while still missing
                    # an exact axis/measurement by a tiny amount. This phase
                    # uses all equalities and inequalities, the same bounds,
                    # and the SAME global budget. It is initialization only.
                    class ProjectionSubbudget(Exception):pass
                    projection_started=time.monotonic()
                    projection_limit=min(12.,.25*numerical_budget.seconds)
                    best_projection=starting[active].copy();best_projection_cost=math.inf
                    projection_origin=starting[active].copy()
                    def projection_residual(value):
                        nonlocal best_projection,best_projection_cost
                        if time.monotonic()-projection_started>projection_limit:raise ProjectionSubbudget()
                        numerical_budget.guard()
                        residual=np.concatenate([all_equalities(value),*[
                            np.minimum(0.,np.atleast_1d(row['fun'](value)))
                            for row in optimizer_constraints if row['type']=='ineq'],
                            1e-6*(value-projection_origin)])
                        cost=float(residual@residual)
                        if math.isfinite(cost) and cost<best_projection_cost:
                            best_projection=value.copy();best_projection_cost=cost
                        return residual
                    try:
                        projected=least_squares(projection_residual,starting[active],
                            bounds=(lower[active],upper[active]),max_nfev=120,
                            ftol=1e-13,xtol=1e-13,gtol=1e-13,x_scale=1.)
                    except ProjectionSubbudget:pass
                    except (ValueError,FloatingPointError,np.linalg.LinAlgError):pass
                    safe_retain_numerical_checkpoint(best_projection)
                    starting[active]=best_projection
                    result['diagnostics']['same_problem_feasibility_projection']={
                        'method':'bounded_all_known_equations_before_ink_objective',
                        'elapsed_seconds':time.monotonic()-projection_started,
                        'maximum_subbudget_seconds':projection_limit,
                        'local_initial_value_regularizer_weight':1e-6,
                        'residual_squared_norm':best_projection_cost if math.isfinite(best_projection_cost) else None,
                        'feasible_checkpoint_found':numerical_checkpoint is not None,
                        'all_known_equations_preserved':True,'publication_approved':False,
                        'original_domain_recentered':False,'thresholds_changed':False}
                try:
                    fit=minimize(scalar_objective,starting[active],method="SLSQP",
                                 jac=scalar_gradient,
                                 bounds=list(zip(lower[active],upper[active])),
                                 constraints=optimizer_constraints,
                                 callback=safe_retain_numerical_checkpoint,
                                 options={"maxiter":1200,"ftol":1e-12})
                except _NumericalBudgetExhausted:
                    if numerical_checkpoint is None:raise
                    numerical_checkpoint_selected=True
                    fit=OptimizeResult(x=numerical_checkpoint.copy(),jac=np.zeros(len(active)),
                        success=False,status=9,nfev=numerical_budget.count,nit=numerical_checkpoint_iterations,
                        message='Bounded optimization ended; independently revalidated feasible source subset retained, without optimality claim')
                if not fit.success and numerical_checkpoint is not None:
                    numerical_checkpoint_selected=True
                    fit=OptimizeResult(x=numerical_checkpoint.copy(),jac=np.zeros(len(active)),
                        success=False,status=int(fit.status),nfev=int(fit.nfev),nit=int(fit.nit),
                        message='Nonconverged optimization; independently revalidated feasible source subset retained, without optimality claim')
                result['diagnostics']['numerical_feasible_checkpoint']={
                    'enabled':numerical_budget.diagnostics['enabled'],
                    'found':numerical_checkpoint is not None,'selected':numerical_checkpoint_selected,
                    'complete_known_subset_rechecked':numerical_checkpoint is not None,
                    'original_domain_recentered':False,'original_source_publication_gate_still_required':True,
                    'reference_geometry_used':False,'optimality_proven':False}
            if not np.isfinite(np.asarray(fit.x,dtype=float)).all():
                raise FloatingPointError("Nonfinite optimization checkpoint")
            # The bounded search has already stopped. Certification of its
            # saved point must not re-enter the exhausted optimization guard.
            final_residual=objective_residuals(fit.x)
            fit.cost=float(final_residual@final_residual/2)
            fit.optimality=float(np.max(np.abs(fit.jac))) if len(fit.jac) else 0.
            fitted_jacobian=_finite_difference_jacobian(lambda reduced:constraint_residuals(expand(reduced)),fit.x)
        else:
            fit=least_squares(objective,starting[active],bounds=(lower[active],upper[active]),
                              max_nfev=1200,ftol=1e-11,xtol=1e-11,gtol=1e-9,x_scale=1.,**linear_options)
            fitted_jacobian=fit.jac[:len(prepared)]
        solved_vector=expand(fit.x)
        node_map,candidate=decode(solved_vector)
        if not np.isfinite(solved_vector).all():raise FloatingPointError("Nonfinite optimization result")
        entity_map={entity["id"]:entity for entity in candidate};checks=[]
        candidate_angle_contract=_strict_ocr_angle_contract(candidate,strict_angles)
        candidate_nominal_angle_contract=_strict_general_angle_contract(node_map,candidate,strict_general_angles)
        candidate_tangent_contract=_strict_tangent_contract(candidate,strict_tangents)
        candidate_measurement_contract=_strict_measurement_contract(node_map,candidate,strict_measurements,baseline)
        strict_measurement_checks={row["constraint_id"]:row for row in candidate_measurement_contract["checks"]}
        strict_tangent_checks={row["constraint_id"]:row for row in candidate_tangent_contract["checks"]}
        strict_angle_checks={row["constraint_id"]:row for row in candidate_angle_contract["checks"]}
        strict_nominal_angle_checks={row["constraint_id"]:row for row in candidate_nominal_angle_contract["checks"]}
        for index,row in enumerate(prepared):
            actual=_constraint_value(row,node_map,entity_map);residual=_constraint_residual(row,actual)
            strict_angle=strict_angle_checks.get(row["id"])
            strict_nominal_angle=strict_nominal_angle_checks.get(row["id"])
            strict_tangent=strict_tangent_checks.get(row["id"])
            strict_measurement=strict_measurement_checks.get(row["id"])
            tolerance=(0. if row["kind"]=="radius" else
                       STRICT_TANGENT_CERT_TOLERANCE_DEG if strict_tangent else
                       STRICT_MEASUREMENT_CERT_TOLERANCE_MM if strict_measurement else
                       STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG if strict_angle or strict_nominal_angle else float(normalizers[index]))
            checks.append({**row,"actual":float(actual),"signed_residual":float(residual),"absolute_residual":float(abs(residual)),
                           "tolerance":tolerance,"residual_unit":"mm" if row["kind"] in _DIMENSIONAL else "degree",
                           "enforcement":"exact" if row["kind"]=="radius" else
                                         "normalized_tangent_cross_equality" if strict_tangent else
                                         "derived_measurement_anchor_equality" if strict_measurement and "anchors" in row else
                                         "source_nominal_equality" if strict_measurement else
                                         "linear_endpoint_direction_equality" if strict_angle else
                                         "source_nominal_angle_equality" if strict_nominal_angle else "fixed_tolerance",
                           "passed":bool(abs(residual)<=tolerance and
                                         (strict_tangent is None or strict_tangent["passed"]) and
                                         (strict_measurement is None or strict_measurement["passed"]) and
                                         (strict_angle is None or strict_angle["passed"]) and
                                         (strict_nominal_angle is None or strict_nominal_angle["passed"])),
                           "binding_verified_by_solver":False})
        candidate_nodes=[{**node,"x":float(node_map[node["id"]][0]),"y":float(node_map[node["id"]][1])} for node in nodes]
        validation=_geometry_validation(candidate,branches,scale)
        displacement=max(float(np.linalg.norm(node_map[node["id"]]-np.array([node["x"],node["y"]]))) for node in nodes)
        shape_displacement=max(float(np.linalg.norm(_sample_entity(after)-_sample_entity(before),axis=1).max()) for before,after in zip(initial_entities,candidate))
        movement_limit=max(.25 if graph["units"]=="mm" else 2.,scale*MAX_DISPLACEMENT_FRACTION)
        same_winding=base_validation["signed_area"]*validation["signed_area"]>0
        moved_safely=max(displacement,shape_displacement)<=movement_limit
        if hasattr(fitted_jacobian,"toarray"):fitted_jacobian=fitted_jacobian.toarray()
        jacobian=np.zeros((len(prepared),count));jacobian[:,active]=fitted_jacobian
        # Support levels are relations of the measurement graph, not free
        # station coordinates. Include their independent equations in the
        # shape-DOF audit; exclude all soft priors and domain inequalities.
        support_levels={json.dumps(anchor,sort_keys=True):anchor for row in strict_measurements
                        for anchor in row.get("anchors",[]) if anchor["type"] in {"support_level","common_curve_center"}}
        if support_levels:
            def support_level_rank_rows(vector):
                _,decoded=decode(vector);emap={entity["id"]:entity for entity in decoded}
                return np.array([value/LINEAR_TOLERANCE_MM for anchor in support_levels.values()
                                 for value in derived_anchor_equalities(anchor,emap)])
            jacobian=np.vstack([jacobian,_finite_difference_jacobian(support_level_rank_rows,solved_vector)])
        singular=np.linalg.svd(jacobian,compute_uv=False);rank_tolerance=max(float(singular[0])*1e-6,1e-8) if len(singular) else 1e-8
        rank=int(np.count_nonzero(singular>rank_tolerance));remaining=count-rank
        gauges=[]
        for axis in range(2):
            vector=np.zeros(count);vector[axis:2*len(nodes):2]=1.;gauges.append(vector)
        rotation=np.zeros(count);xy=solved_vector[:2*len(nodes)].reshape((-1,2));rotation[:2*len(nodes)]=np.column_stack([-xy[:,1],xy[:,0]]).ravel();gauges.append(rotation)
        gauge_dof=sum(np.linalg.norm(jacobian@vector)<=max(1e-7,np.linalg.norm(jacobian)*np.linalg.norm(vector)*1e-7) for vector in gauges if np.linalg.norm(vector)>1e-10)
        numerical_shape_dof=max(0,remaining-int(gauge_dof))
        # Fixing candidate geometry is not dimension evidence. Retain these
        # unverified radius freedoms in the annotation-identifiability audit.
        shape_dof=numerical_shape_dof+unverified_preservation_count;underconstrained=shape_dof>0
        all_passed=all(check["passed"] for check in checks)
        chord_checks=[{"entity_id":entity_id,"chord_length":math.dist(entity_map[entity_id]["start"],entity_map[entity_id]["end"]),
                       "maximum_chord":2*radius,"numerical_tolerance":max(1e-10,scale*1e-12)}
                      for entity_id,radius in exact_radii.items()]
        for check in chord_checks:check["passed"]=check["chord_length"]<=check["maximum_chord"]+check["numerical_tolerance"]
        chords_feasible=all(check["passed"] for check in chord_checks)
        source_budget_audit=source_budget.safe_inspect(candidate) if source_budget.enabled else None
        source_budget_passed=source_budget_audit is None or source_budget_audit["passed"]
        if source_budget.enabled:source_budget.diagnostics.update(final_audit=source_budget_audit,passed=source_budget_passed)
        optimizer_converged=bool(fit.success)
        independently_verified_checkpoint=bool(numerical_checkpoint_selected or (
            source_budget.enabled and best_feasible is not None and
            np.array_equal(fit.x,best_feasible.x) and finite_checkpoint(fit.x) and
            fixed_subset_passed(fit.x) and complete_geometry_passed(fit.x) and
            source_budget_passed))
        accepted=bool((optimizer_converged or independently_verified_checkpoint) and all_passed and
                      validation["passed"] and chords_feasible and moved_safely and same_winding and source_budget_passed)
        acceptance_basis=("independently_verified_feasible_checkpoint" if accepted and not optimizer_converged else
                          "optimizer_converged_and_final_checks" if accepted else "rejected")
        status="accepted" if accepted else "source_budget_search_exhausted" if source_budget.enabled and source_budget.exhausted else "source_budget_search_failed" if source_budget.enabled and not source_budget_passed else "not_converged" if not fit.success else "conflict" if not all_passed else "invalid_geometry" if not validation["passed"] or not same_winding or not chords_feasible else "displacement_rejected"
        dimensional_checks=[check for check in checks if check["source"]!="source_geometry" and check["kind"] in _DIMENSIONAL|{"angle"}]
        issues=list(validation["issues"])
        if not all_passed:issues.append("Supplied constraints conflict or could not be jointly satisfied at the fixed numerical tolerances.")
        if not optimizer_converged:
            issues.append("Bounded nonlinear optimizer did not converge; this independently verified feasible checkpoint has no optimality claim." if accepted else
                          "Bounded nonlinear solve did not converge; baseline is retained.")
        if not moved_safely:issues.append("Candidate exceeds the declared source-displacement guard; baseline is retained.")
        if not same_winding:issues.append("Candidate reverses the input contour orientation; baseline is retained.")
        if not chords_feasible:issues.append("Candidate chord exceeds the diameter of an exact annotated radius; baseline is retained.")
        if not source_budget_passed:issues.append("Bounded source-boundary feasibility search did not find a certified solution; baseline is retained. This is not proof of mathematical infeasibility.")
        if source_budget.enabled and source_budget.exhausted:
            issues.append("RMS refinement exhausted its search budget; a source-feasible checkpoint passed complete final revalidation, without an optimality claim." if accepted else
                          "Source-budget wall/evaluation limit exhausted; baseline is retained. This is not proof of mathematical infeasibility.")
        if underconstrained:issues.append("Remaining shape freedoms are selected by a soft source prior; only the supplied constraint subset was solved.")
        validation.update(passed=accepted,geometry_valid=validation["passed"],constraint_subset_satisfied=all_passed,
                          strict_radius_satisfied=all(check["passed"] for check in checks if check["kind"]=="radius"),
                          strict_ocr_angle_satisfied=candidate_angle_contract["satisfied"],
                          strict_source_nominal_angle_satisfied=candidate_nominal_angle_contract["satisfied"],
                          strict_tangent_satisfied=candidate_tangent_contract["satisfied"],
                          strict_measurement_satisfied=candidate_measurement_contract["satisfied"],
                          exact_radius_chords_feasible=chords_feasible,exact_radius_chord_checks=chord_checks,
                          all_dimensions_verified=False,dimensions_verified=False,reference_verified=False,engineering_certified=False,
                          underconstrained=underconstrained,source_displacement_passed=moved_safely,winding_preserved=same_winding,issues=issues)
        if source_budget.enabled:validation.update(source_boundary_budget_passed=source_budget_passed,source_boundary_budget_audit=source_budget_audit)
        result.update(status=status,accepted=accepted,underconstrained=underconstrained,
                      entities=candidate if accepted else copy.deepcopy(baseline),nodes=candidate_nodes if accepted else copy.deepcopy(nodes),
                      candidate_entities=candidate,candidate_nodes=candidate_nodes,constraints=checks,validation=validation,
                      dimension_solve_success=bool(accepted and dimensional_checks and graph["units"]=="mm"))
        result["candidate_strict_radius_contract"]=_strict_radius_contract(candidate,prepared)
        result["strict_radius_contract"]=_strict_radius_contract(result["entities"],prepared)
        result["candidate_strict_ocr_angle_contract"]=candidate_angle_contract
        result["strict_ocr_angle_contract"]=_strict_ocr_angle_contract(result["entities"],strict_angles)
        result["candidate_strict_source_nominal_angle_contract"]=candidate_nominal_angle_contract
        result["strict_source_nominal_angle_contract"]=_strict_general_angle_contract(
            {node["id"]:np.array([node["x"],node["y"]]) for node in result["nodes"]},
            result["entities"],strict_general_angles)
        result["candidate_strict_tangent_contract"]=candidate_tangent_contract
        result["strict_tangent_contract"]=_strict_tangent_contract(result["entities"],strict_tangents)
        result["candidate_strict_measurement_contract"]=candidate_measurement_contract
        result["strict_measurement_contract"]=_strict_measurement_contract(
            {node["id"]:np.array([node["x"],node["y"]]) for node in result["nodes"]},
            result["entities"],strict_measurements,baseline)
        result["candidate_constructed_radius_preservation"]=_constructed_radius_receipt(candidate,preserved_radii,ignored_preservation)
        result["constructed_radius_preservation"]=_constructed_radius_receipt(result["entities"],preserved_radii,ignored_preservation)
        result["diagnostics"].update(converged=optimizer_converged,optimizer_converged=optimizer_converged,
                                     optimality_proven=False,acceptance_basis=acceptance_basis,
                                     independently_verified_feasible_checkpoint=independently_verified_checkpoint,
                                     optimizer_status=int(fit.status),evaluations=int(fit.nfev),
                                     optimizer_message=str(fit.message),
                                     optimizer_parameter_scaling="common_bbox_normalized_length_unit",
                                     cost=float(fit.cost),optimality=float(fit.optimality),constraint_rank=rank,rank_tolerance=rank_tolerance,
                                     optimality_measure="feasibility_movement_gradient_inf_norm_not_RMS_optimality_or_KKT" if source_budget.enabled and not source_budget.diagnostics.get("RMS_refinement_accepted") else "objective_gradient_inf_norm_not_KKT" if exact_radii or strict_angles or strict_general_angles or strict_tangents or strict_measurements or raw_prior is not None else "least_squares_first_order_optimality",
                                     singular_values=singular.tolist(),remaining_dof=remaining,rigid_gauge_dof=int(gauge_dof),remaining_shape_dof=shape_dof,
                                     numerical_remaining_shape_dof=numerical_shape_dof,
                                     rank_excludes_soft_prior=True,independent_dimension_record_count=len(dimensional_checks),
                                     optimizer_active_variable_count=len(active),inactive_variables_at_soft_prior_minimum=count-len(active),
                                     coordinate_prior_weighting="initial_primitive_arclength" if raw_prior is not None else "uniform_coordinate",
                                     radius_bound_arc_offsets_without_initial_h_penalty=sorted(bound_radii),
                                     constructed_radius_offsets_without_initial_h_penalty=sorted(row["entity_id"] for row in preserved_radii),
                                     radius_feasibility_initialization=radius_initialization,
                                     source_curve_prior={"enabled":bool(supported_shapes),"weight":SOURCE_CURVE_PRIOR_WEIGHT,
                                                         "entity_count":len(supported_shapes),
                                                         "samples_per_entity":shape_sample_count if supported_shapes else 0,
                                                         "weighting":"source_arclength_quadrature",
                                                         "admission":"finite_nonnegative_source_fit_error_receipt",
                                                         "source":"input_graph_entities","hard_constraint":False},
                                     optimizer="SLSQP_bounded_source_budget_feasibility_then_RMS" if source_budget.enabled else "SLSQP_strict_joint_equalities" if strict_tangents or strict_measurements or strict_general_angles else "SLSQP_exact_radius_chord_inequalities" if exact_radii else "SLSQP_strict_ocr_angle_equalities" if strict_angles else "SLSQP_source_observation" if raw_prior is not None else "bounded_least_squares",
                                     linear_subsolver="SLSQP" if exact_radii or strict_angles or strict_general_angles or strict_tangents or strict_measurements or raw_prior is not None else "exact_dense" if len(active)<=96 else "bounded_sparse_lsmr",
                                     maximum_node_displacement=displacement,maximum_parameterized_curve_displacement=shape_displacement,
                                     displacement_limit=movement_limit,displacement_unit=graph["units"],
                                     distance_axis_semantics="signed ordered nodes or derived anchors [1] minus [0]; x right, y up",
                                     angular_semantics="degree; angle defaults to unsigned 0..180; tangent compares directed traversal vectors at an explicit shared joint",
                                     acceptance_meaning="Valid source-preserving solution of supplied constraints only. Underconstraint, annotation binding correctness, all dimensions and reference accuracy remain separate.")
    except _NumericalBudgetExhausted:
        result.update(status="optimization_budget_exhausted",accepted=False,
            validation={"passed":False,"all_dimensions_verified":False,
                "issues":["Bounded optimization exhausted; baseline is retained. Infeasibility is not proven."]})
        result["diagnostics"].update(converged=False,infeasibility_proven=False)
        if continuation_checkpoint is not None:
            points,entities=decode(expand(continuation_checkpoint))
            result['candidate_entities']=entities
            result['candidate_nodes']=[{**node,'x':float(points[node['id']][0]),
                'y':float(points[node['id']][1])} for node in nodes]
            result['diagnostics']['numerical_continuation_checkpoint']={
                'available':True,'problem_sha256':numerical_problem_identity(graph,constraints),
                'constraint_residual_norm':continuation_score,
                'same_topology_and_original_domain':True,'known_equations_certified':False,
                'seed_only':True,'publication_approved':False,'reference_geometry_used':False}
    except (ValueError,FloatingPointError,ArithmeticError,np.linalg.LinAlgError) as error:
        result.update(status="failed",validation={"passed":False,"all_dimensions_verified":False,"issues":[f"Numerical solve failed ({type(error).__name__}); baseline is retained."]})
        result["diagnostics"].update(error_type=type(error).__name__,converged=False)
    numerical_budget.finish()
    return _persist(result,output_dir)

"""Joint source-only dimensional solving for an already declared CAD topology.

This module reads no images, references, training labels or network resources.
Shared nodes enforce closure; signed chord-normal arc centers preserve each
declared minor/major branch. Bound radius annotations are eliminated as exact
constants with explicit diameter feasibility. Source coordinates and supported
source-curve samples are soft priors, never locks or dimensional evidence.
"""
from __future__ import annotations

import copy
import json
import math
from numbers import Real
from pathlib import Path
import re
import time

import numpy as np
from scipy.optimize import OptimizeResult, least_squares, minimize
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from .topology import _primitive_distance
from .vectorize import _sample_entities, _line_entities, assess_fit_quality

LINEAR_TOLERANCE_MM = .05
ANGLE_TOLERANCE_DEG = .1
MAX_DISPLACEMENT_FRACTION = .10
MAX_ENTITIES = 256
MAX_CONSTRAINTS = 1024
SOURCE_CURVE_PRIOR_WEIGHT = .3
SOURCE_BUDGET_SEARCH_ROUNDS = 5
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
        value=row.get("value")
        if value is not None:value=_number(value,"constraint.value")
        row["value"]=value
        record=row.get("record_id")
        if source!="source_geometry":
            _identifier(record,"dimension record_id")
        elif record is not None:raise ValueError("source_geometry must not impersonate an OCR record binding")
        if kind in _DIMENSIONAL:
            if graph["units"]!="mm" or source=="source_geometry" or value is None:
                raise ValueError("Physical dimensions require an mm graph, OCR binding source and record_id/value")
            if kind=="radius" and (len(eids)!=1 or entity_map[eids[0]]["type"]!="ARC" or nids):
                raise ValueError("Radius binds exactly one ARC and no nodes")
            if kind!="radius" and len(nids)!=2:raise ValueError("Distances require two ordered nodes")
            if kind in {"radius","distance"} and value<=0:raise ValueError("Radius/distance must be positive")
        elif kind in {"horizontal","vertical"}:
            if value not in (None,0.):raise ValueError("Horizontal/vertical relation value must be null or zero")
            if len(nids)==2:
                pass
            elif not nids and len(eids)==1 and entity_map[eids[0]]["type"]=="LINE":
                row["nodes"]=[entity_map[eids[0]]["start_node"],entity_map[eids[0]]["end_node"]]
            else:raise ValueError("Axis relation needs two nodes or one LINE")
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
        if kind in {"horizontal","vertical","distance"}:canonical_nodes=tuple(sorted(canonical_nodes))
        if kind in {"distance_x","distance_y"} and canonical_nodes[0]>canonical_nodes[1]:
            canonical_nodes=canonical_nodes[::-1];canonical_value=-value
        if kind=="tangent":canonical_entities=tuple(sorted(canonical_entities))
        if kind=="angle" and row["angle_mode"]=="unsigned":
            canonical_entities=tuple(sorted(canonical_entities))
            if len(canonical_nodes)==3 and canonical_nodes[0]>canonical_nodes[2]:canonical_nodes=canonical_nodes[::-1]
        signature=(kind,canonical_entities,canonical_nodes,canonical_value,row.get("angle_mode"))
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


def _constraint_value(row,nodes,entities):
    kind=row["kind"];nids=row["nodes"];eids=row["entities"]
    if kind=="radius":return entities[eids[0]]["radius"]
    if kind.startswith("distance"):
        delta=nodes[nids[1]]-nodes[nids[0]]
        return float(delta[0] if kind=="distance_x" else delta[1] if kind=="distance_y" else np.linalg.norm(delta))
    if kind in {"horizontal","vertical"}:
        delta=nodes[nids[1]]-nodes[nids[0]]
        value=math.degrees(math.atan2(delta[1],delta[0]))-(90 if kind=="vertical" else 0)
        return (value+90)%180-90
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


def _raw_source_prior(observation,graph,entities,scale):
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
    for index in range(len(entities)):
        indices=np.flatnonzero(quadrature_assigned==index)
        selected=indices[np.unique(np.linspace(0,len(indices)-1,min(32,len(indices))).astype(int))] if len(indices) else indices
        neighborhood=[(index-1)%len(entities),index,(index+1)%len(entities)]
        local=np.flatnonzero(np.isin(assigned,neighborhood))
        local=np.unique(np.r_[local,(local-1)%len(vertices)])
        reverse_count=max(3,min(65,int(math.ceil(128*primitive_shares[index]))))
        groups.append((quadrature[selected],neighborhood,local,len(indices)/quadrature_count,
                       float(primitive_shares[index]),reverse_count))
    # Integrate per unit of normalized boundary length, instead of giving
    # every primitive an equal RMS weight. Splitting a supported line/arc must
    # not multiply its evidence or make tiny fillets dominate a long arc.
    weight=SOURCE_CURVE_PRIOR_WEIGHT*math.sqrt(perimeter/scale)
    def residuals(decoded):
        values=[]
        for index,(source,neighbors,segments,forward_share,reverse_share,reverse_count) in enumerate(groups):
            if len(source):
                source_distance=np.min(np.asarray([_primitive_distance(source,decoded[j]) for j in neighbors]),axis=0)
                values.append(source_distance*(weight*math.sqrt(forward_share/len(source)))/scale)
            if len(segments):
                samples=_sample_entity(decoded[index],reverse_count,midpoints=True)
                delta=samples[:,None,:]-vertices[segments][None,:,:]
                fraction=np.clip(np.sum(delta*deltas[segments][None,:,:],axis=2)/squared[segments][None,:],0.,1.)
                nearest=vertices[segments][None,:,:]+fraction[:,:,None]*deltas[segments][None,:,:]
                reverse=np.min(np.linalg.norm(samples[:,None,:]-nearest,axis=2),axis=1)
                values.append(reverse*(weight*math.sqrt(reverse_share/reverse_count))/scale)
        return np.concatenate(values)
    return residuals,{"enabled":True,"source":"immutable_input_mask_boundary","units":graph["units"],
                      "input_point_count":len(points),"source_quadrature_point_count":quadrature_count,
                      "source_quadrature":"uniform_physical_arclength_midpoints",
                      "maximum_forward_points_per_entity":32,"maximum_reverse_points_per_entity":65,
                      "matching":"local_entity_and_immediate_neighbors; exact_source_polyline_reverse_distance",
                      "weighting":"source_arclength_forward_and_primitive_arclength_reverse",
                      "source_perimeter":perimeter,"weight":SOURCE_CURVE_PRIOR_WEIGHT,"hard_constraint":False,
                      "oracle_mask_conditioned":observation.get("oracle_mask_conditioned") is True,
                      "reference_dxf_read":False,"prior_cad_curve_samples_replaced":True}


class _SourceBudgetExhausted(RuntimeError):
    pass


class _SourceBoundaryBudget:
    """Bounded witness generation with an independent complete-curve audit."""
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
            maximum_witnesses_per_round=24,rounds=[],infeasibility_proven=False,
            maximum_wall_seconds=self.wall_limit,maximum_objective_evaluations=self.objective_limit,
            method="bounded_worst_point_constraint_generation_then_RMS",
            source="immutable_input_mask_boundary")

    def guard(self,*,objective=False):
        elapsed=time.monotonic()-self.started
        if elapsed>=self.wall_limit or (objective and self.objective_evaluations>=self.objective_limit):
            self.exhausted=True
            self.diagnostics.update(exhausted=True,exhaustion_reason="wall_seconds" if elapsed>=self.wall_limit else "objective_evaluations")
            raise _SourceBudgetExhausted("Bounded source-budget search exhausted its wall or objective-evaluation limit")
        if objective:self.objective_evaluations+=1

    def progress(self):
        self.diagnostics.update(elapsed_wall_seconds=time.monotonic()-self.started,
                                objective_evaluations=self.objective_evaluations,exhausted=self.exhausted)

    def inspect(self,entities,*,add_witnesses=True):
        quality=assess_fit_quality(self.points,entities,max_step_px=self.step)
        upper=float(quality["source_boundary_deviation_px"]["conservative_upper_bound_px"])
        forward_distance=np.min(np.asarray([_primitive_distance(self.samples,e) for e in entities]),axis=0)
        if add_witnesses:
            # Spatial separation avoids spending a round's finite witness
            # budget on adjacent samples of the same local displacement.
            selected=[]
            for index in np.argsort(forward_distance)[::-1]:
                if ("forward",int(index)) in self.seen:continue
                if all(np.linalg.norm(self.samples[index]-self.samples[old])>self.step*3 for old in selected):
                    selected.append(int(index))
                if len(selected)>=12:break
            for index in selected:
                key=("forward",index)
                if key not in self.seen:self.seen.add(key);self.forward.append(self.samples[index].copy())
            reverse=[]
            for index,entity in enumerate(entities):
                length=math.dist(entity["start"],entity["end"]) if entity["type"]=="LINE" else entity["radius"]*_sweep(entity)
                count=max(2,int(math.ceil(length/self.step))+1)
                if count>100000:raise ValueError("Source boundary candidate sampling limit exceeded")
                samples=_sample_entity(entity,count);distances=self.tree.query(samples,workers=1)[0]
                for sample in np.argsort(distances)[-12:]:reverse.append((float(distances[sample]),index,float(sample/(count-1))))
            unseen=[row for row in sorted(reverse,reverse=True) if ("reverse",row[1],round(row[2],8)) not in self.seen]
            for _,index,fraction in unseen[:12]:
                key=("reverse",index,round(fraction,8))
                if key not in self.seen:self.seen.add(key);self.reverse.append((index,fraction))
        return {"passed":bool(quality["sampled_topology_valid"] and upper<=self.maximum),
                "conservative_max_deviation":upper,"sampling_uncertainty":quality["source_boundary_deviation_px"]["sampling_uncertainty_bound_px"],
                "source_to_curve_sample_max":float(forward_distance.max()),
                "geometry_valid":quality["sampled_topology_valid"],"reference_verified":False}

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


def solve_parametric(graph,constraints,*,output_dir=None,source_observation=None):
    """Solve declared constraints jointly; preserve baseline on every rejection.

    ``accepted`` permits a validated partial dimensional candidate even when
    ``underconstrained`` remains true. Neither flag establishes all drawing
    dimensions, annotation bindings, reference accuracy or engineering approval.
    Invalid input/schema raises ValueError before any optimization or output.
    """
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
    initial,lower,upper,decode,scale,node_indices,arc_params,branches=_model(nodes,baseline,exact_radii)
    count=len(initial);normalizers=np.array([LINEAR_TOLERANCE_MM if row["kind"] in _DIMENSIONAL else ANGLE_TOLERANCE_DEG for row in prepared])
    _,initial_entities=decode(initial)
    if exact_radii:initial_entities=copy.deepcopy(baseline)
    raw_prior,raw_prior_diagnostics=_raw_source_prior(source_observation,graph,initial_entities,scale)
    source_budget=_SourceBoundaryBudget(source_observation,graph)
    base_validation=_geometry_validation(initial_entities,branches,scale)
    result={"schema_version":"source-parametric-solver-v1","status":"no_constraints","accepted":False,"underconstrained":True,
            "units":graph["units"],"coordinate_system":copy.deepcopy(graph.get("coordinate_system")),
            "entities":copy.deepcopy(baseline),"nodes":copy.deepcopy(nodes),"baseline_entities":copy.deepcopy(baseline),
            "candidate_entities":None,"candidate_nodes":None,"constraints":[],"validation":{},
            "diagnostics":{"source_only":True,"ground_truth_used":False,"network_used":False,"source_coordinates_hard_locked":False,
                           "arc_parameterization":"shared endpoints + exact bound radius constants, preserved constructed geometry radii, or free chord-normal center offsets; fixed clockwise/minor-major branch",
                           "arc_branches":branches,"dimension_tolerance_mm":LINEAR_TOLERANCE_MM,"angle_tolerance_deg":ANGLE_TOLERANCE_DEG,
                           "radius_tolerance_mm":0.,"eliminated_exact_radius_constraint_count":len(bound_radii),
                           "eliminated_constructed_geometry_radius_count":unverified_preservation_count,
                           "preserved_geometry_is_dimensional_evidence":False,
                           "raw_source_prior":raw_prior_diagnostics,
                           "source_budget_search":source_budget.diagnostics,
                           "reference_tolerance_changed":False,"independent_reference_accuracy_evaluated":False,
                           "prior_weight":.1,"variable_count":count,"max_displacement_fraction":MAX_DISPLACEMENT_FRACTION},
            "strict_radius_contract":_strict_radius_contract(baseline,prepared),
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
    if not prepared:
        unchanged_validation=_geometry_validation(baseline,branches,scale)
        result["validation"]={**unchanged_validation,"passed":False,"geometry_valid":unchanged_validation["passed"],
                              "constraint_subset_satisfied":False,"all_dimensions_verified":False,"dimensions_verified":False,
                              "issues":[*unchanged_validation["issues"],"No explicit constraints were supplied; source geometry is retained."]}
        result["diagnostics"].update(constraint_rank=0,remaining_dof=count,
                                     remaining_shape_dof=max(0,count-3)+unverified_preservation_count,rigid_gauge_dof=3)
        return _persist(result,output_dir)

    def constraint_residuals(vector):
        nmap,decoded=decode(vector);emap={entity["id"]:entity for entity in decoded}
        return np.array([_constraint_residual(row,_constraint_value(row,nmap,emap)) for row in prepared])/normalizers

    sparsity=lil_matrix((len(prepared)+count,count),dtype=int)
    for index,row in enumerate(prepared):
        dependencies=set(row["nodes"])
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
    starting=initial.copy()
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
    def objective(reduced):
        if source_budget.enabled and source_budget.search_active:source_budget.guard(objective=True)
        vector=expand(reduced)
        residuals=[constraint_residuals(vector),prior_weights[active]*(reduced-initial[active])]
        if supported_shapes:
            _,decoded=decode(vector)
            residuals.extend((shape_weights[index]*(_sample_entity(decoded[index],shape_sample_count)-source_shapes[index])/scale).ravel()
                             for index in supported_shapes)
        if raw_prior is not None:
            _,decoded=decode(vector)
            residuals.append(raw_prior(decoded))
        return np.concatenate(residuals)
    sparse_active=lil_matrix((len(prepared)+len(active),len(active)),dtype=int)
    sparse_active[:len(prepared)]=sparsity[:len(prepared),active]
    for index in range(len(active)):sparse_active[len(prepared)+index,index]=1
    try:
        linear_options={} if len(active)<=96 else {"jac_sparsity":sparse_active.tocsr(),"tr_solver":"lsmr",
                                                   "tr_options":{"atol":1e-12,"btol":1e-12,"maxiter":max(100,len(active)*5)}}
        # Coordinates and free center offsets share one bounding-box-normalized
        # length unit. Jacobian column rescaling would distort their common
        # source-prior metric, especially for short lines next to large arcs.
        # The unannotated path retains bounded least squares. Exact radii need
        # a constrained optimizer because their endpoint chords must fit their
        # annotated diameters; neither branch relaxes the acceptance checks.
        if exact_radii or raw_prior is not None or source_budget.enabled:
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
                chord_constraints=optimizer_constraints.copy()
                optimizer_constraints.extend([{"type":"ineq","fun":dimensional_feasibility},
                                              {"type":"ineq","fun":boundary_feasibility},
                                              {"type":"ineq","fun":movement_feasibility}])
                reduced=starting[active].copy();best_feasible=None;total_evaluations=0;total_iterations=0
                source_budget.diagnostics.update(maximum_feasibility_iterations_per_round=120,
                    maximum_RMS_iterations_per_round=180,maximum_warm_RMS_iterations=80,dimensional_tolerances_changed=False,
                    feasibility_objective="minimum_movement_regularizer; annotations, boundary and unchanged displacement guard are inequalities")
                def feasibility_objective(value):
                    source_budget.guard(objective=True)
                    return float(np.dot(value-starting[active],value-starting[active])/2)
                def feasibility_gradient(value):return value-starting[active]
                fit=OptimizeResult(x=reduced.copy(),jac=np.zeros(len(reduced)),success=False,status=9,nfev=0,nit=0,
                                   message="Source-budget search has no converged checkpoint")
                def fixed_subset_passed(value):
                    return bool(np.max(np.abs(constraint_residuals(expand(value))))<=1. and
                                (not exact_radii or np.min(chord_feasibility(value))>=-1e-10))
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
                    # An already admissible ordinary RMS result needs no
                    # expensive constraint-generation rounds. It still faces
                    # the identical independent maximum-distance certificate.
                    warm=minimize(scalar_objective,reduced,method="SLSQP",jac=scalar_gradient,
                        bounds=list(zip(lower[active],upper[active])),constraints=chord_constraints,
                        options={"maxiter":80,"ftol":1e-12})
                    total_evaluations+=int(warm.nfev);total_iterations+=int(warm.nit)
                    fit=warm;reduced=warm.x.copy()
                    warm_audit=source_budget.safe_inspect(decode(expand(reduced))[1])
                    warm_passed=bool(warm.success and warm_audit["passed"] and fixed_subset_passed(reduced) and complete_geometry_passed(reduced))
                    source_budget.diagnostics.update(warm_RMS_audit=warm_audit,warm_RMS_converged=bool(warm.success),
                                                     warm_RMS_reused=warm_passed)
                    if not warm_passed:
                        if (not warm_audit.get("geometry_valid") or
                            initial_budget_audit.get("conservative_max_deviation",math.inf)<warm_audit.get("conservative_max_deviation",math.inf)):
                            reduced=starting[active].copy()
                            source_budget.diagnostics["initial_seed_retained_after_unconverged_warm_RMS"]=True
                        for round_index in range(SOURCE_BUDGET_SEARCH_ROUNDS):
                            source_budget.guard()
                            before=source_budget.safe_inspect(decode(expand(reduced))[1],add_witnesses=True)
                            feasibility_fit=minimize(feasibility_objective,reduced,method="SLSQP",jac=feasibility_gradient,
                                bounds=list(zip(lower[active],upper[active])),constraints=optimizer_constraints,
                                options={"maxiter":120,"ftol":1e-12})
                            total_evaluations+=int(feasibility_fit.nfev);total_iterations+=int(feasibility_fit.nit)
                            reduced=feasibility_fit.x.copy()
                            feasible_audit=source_budget.safe_inspect(decode(expand(reduced))[1])
                            subset_passed=fixed_subset_passed(reduced)
                            feasibility_passed=bool(feasibility_fit.success and feasible_audit["passed"] and subset_passed and complete_geometry_passed(reduced))
                            row={"round":round_index+1,"before":before,"feasibility_audit":feasible_audit,
                                 "feasibility_converged":bool(feasibility_fit.success),"annotation_subset_and_chords_passed":subset_passed,
                                 "feasibility_found":feasibility_passed,"forward_witness_count":len(source_budget.forward),
                                 "reverse_witness_count":len(source_budget.reverse),"feasibility_iterations":int(feasibility_fit.nit)}
                            source_budget.diagnostics["rounds"].append(row)
                            fit=feasibility_fit
                            if not feasibility_passed:continue
                            best_feasible=copy.deepcopy(feasibility_fit)
                            if output_dir is not None:
                                checkpoint=Path(output_dir);checkpoint.mkdir(parents=True,exist_ok=True)
                                (checkpoint/"source-budget-checkpoint.json").write_text(json.dumps({
                                    "schema_version":"source-budget-feasible-checkpoint-v1","units":graph["units"],
                                    "entities":decode(expand(reduced))[1],"source_boundary_audit":feasible_audit,
                                    "annotation_subset_and_chords_passed":True,"engineering_verified":False,
                                    "source_image_gate_not_evaluated":True,"ground_truth_used":False,
                                    "round":round_index+1},ensure_ascii=False,indent=2),encoding="utf8")
                            rms_fit=minimize(scalar_objective,reduced,method="SLSQP",jac=scalar_gradient,
                                bounds=list(zip(lower[active],upper[active])),constraints=optimizer_constraints,
                                options={"maxiter":180,"ftol":1e-12})
                            total_evaluations+=int(rms_fit.nfev);total_iterations+=int(rms_fit.nit)
                            rms_audit=source_budget.safe_inspect(decode(expand(rms_fit.x))[1])
                            rms_passed=bool(rms_fit.success and rms_audit["passed"] and fixed_subset_passed(rms_fit.x) and complete_geometry_passed(rms_fit.x))
                            row.update(RMS_audit=rms_audit,RMS_converged=bool(rms_fit.success),RMS_accepted=rms_passed,
                                       RMS_iterations=int(rms_fit.nit))
                            if rms_passed:fit=rms_fit;break
                            reduced=rms_fit.x.copy()
                        else:
                            if best_feasible is not None:fit=best_feasible
                except _SourceBudgetExhausted:
                    if best_feasible is not None:
                        fit=copy.deepcopy(best_feasible)
                        fit.message="Source-feasible checkpoint retained after RMS search budget exhaustion; no optimality claim"
                    else:
                        fit.success=False;fit.status=9;fit.message="Source-budget wall/evaluation limit exhausted; baseline retained"
                source_budget.search_active=False;source_budget.progress()
                source_budget.diagnostics.update(total_optimizer_evaluations=total_evaluations,
                    total_optimizer_iterations=total_iterations,feasible_checkpoint_found=best_feasible is not None,
                    RMS_refinement_accepted=bool(source_budget.diagnostics.get("warm_RMS_reused") or
                                                (source_budget.diagnostics["rounds"] and source_budget.diagnostics["rounds"][-1].get("RMS_accepted"))),
                    fallback_meaning="Bounded search failure is not proof that the annotations and source budget are incompatible.")
            else:
                fit=minimize(scalar_objective,starting[active],method="SLSQP",
                             jac=scalar_gradient,
                             bounds=list(zip(lower[active],upper[active])),
                             constraints=optimizer_constraints,
                             options={"maxiter":1200,"ftol":1e-12})
            fit.cost=scalar_objective(fit.x)
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
        for index,row in enumerate(prepared):
            actual=_constraint_value(row,node_map,entity_map);residual=_constraint_residual(row,actual)
            tolerance=0. if row["kind"]=="radius" else float(normalizers[index])
            checks.append({**row,"actual":float(actual),"signed_residual":float(residual),"absolute_residual":float(abs(residual)),
                           "tolerance":tolerance,"residual_unit":"mm" if row["kind"] in _DIMENSIONAL else "degree",
                           "enforcement":"exact" if row["kind"]=="radius" else "fixed_tolerance",
                           "passed":bool(abs(residual)<=tolerance),"binding_verified_by_solver":False})
        candidate_nodes=[{**node,"x":float(node_map[node["id"]][0]),"y":float(node_map[node["id"]][1])} for node in nodes]
        validation=_geometry_validation(candidate,branches,scale)
        displacement=max(float(np.linalg.norm(node_map[node["id"]]-np.array([node["x"],node["y"]]))) for node in nodes)
        shape_displacement=max(float(np.linalg.norm(_sample_entity(after)-_sample_entity(before),axis=1).max()) for before,after in zip(initial_entities,candidate))
        movement_limit=max(.25 if graph["units"]=="mm" else 2.,scale*MAX_DISPLACEMENT_FRACTION)
        same_winding=base_validation["signed_area"]*validation["signed_area"]>0
        moved_safely=max(displacement,shape_displacement)<=movement_limit
        if hasattr(fitted_jacobian,"toarray"):fitted_jacobian=fitted_jacobian.toarray()
        jacobian=np.zeros((len(prepared),count));jacobian[:,active]=fitted_jacobian
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
        accepted=bool(fit.success and all_passed and validation["passed"] and chords_feasible and moved_safely and same_winding and source_budget_passed)
        status="accepted" if accepted else "source_budget_search_exhausted" if source_budget.enabled and source_budget.exhausted else "source_budget_search_failed" if source_budget.enabled and not source_budget_passed else "not_converged" if not fit.success else "conflict" if not all_passed else "invalid_geometry" if not validation["passed"] or not same_winding or not chords_feasible else "displacement_rejected"
        dimensional_checks=[check for check in checks if check["source"]!="source_geometry" and check["kind"] in _DIMENSIONAL|{"angle"}]
        issues=list(validation["issues"])
        if not all_passed:issues.append("Supplied constraints conflict or could not be jointly satisfied at the fixed numerical tolerances.")
        if not fit.success:issues.append("Bounded nonlinear solve did not converge; baseline is retained.")
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
        result["candidate_constructed_radius_preservation"]=_constructed_radius_receipt(candidate,preserved_radii,ignored_preservation)
        result["constructed_radius_preservation"]=_constructed_radius_receipt(result["entities"],preserved_radii,ignored_preservation)
        result["diagnostics"].update(converged=bool(fit.success),optimizer_status=int(fit.status),evaluations=int(fit.nfev),
                                     optimizer_message=str(fit.message),
                                     optimizer_parameter_scaling="common_bbox_normalized_length_unit",
                                     cost=float(fit.cost),optimality=float(fit.optimality),constraint_rank=rank,rank_tolerance=rank_tolerance,
                                     optimality_measure="feasibility_movement_gradient_inf_norm_not_RMS_optimality_or_KKT" if source_budget.enabled and not source_budget.diagnostics.get("RMS_refinement_accepted") else "objective_gradient_inf_norm_not_KKT" if exact_radii or raw_prior is not None else "least_squares_first_order_optimality",
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
                                     optimizer="SLSQP_bounded_source_budget_feasibility_then_RMS" if source_budget.enabled else "SLSQP_exact_radius_chord_inequalities" if exact_radii else "SLSQP_source_observation" if raw_prior is not None else "bounded_least_squares",
                                     linear_subsolver="SLSQP" if exact_radii or raw_prior is not None else "exact_dense" if len(active)<=96 else "bounded_sparse_lsmr",
                                     maximum_node_displacement=displacement,maximum_parameterized_curve_displacement=shape_displacement,
                                     displacement_limit=movement_limit,displacement_unit=graph["units"],
                                     distance_axis_semantics="signed nodes[1] minus nodes[0]; x right, y up",
                                     angular_semantics="degree; angle defaults to unsigned 0..180; tangent compares directed traversal vectors at an explicit shared joint",
                                     acceptance_meaning="Valid source-preserving solution of supplied constraints only. Underconstraint, annotation binding correctness, all dimensions and reference accuracy remain separate.")
    except (ValueError,FloatingPointError,ArithmeticError,np.linalg.LinAlgError) as error:
        result.update(status="failed",validation={"passed":False,"all_dimensions_verified":False,"issues":[f"Numerical solve failed ({type(error).__name__}); baseline is retained."]})
        result["diagnostics"].update(error_type=type(error).__name__,converged=False)
    return _persist(result,output_dir)

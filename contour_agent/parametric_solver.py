"""Joint source-only dimensional solving for an already declared CAD topology.

This module reads no images, references, training labels or network resources.
Shared nodes enforce closure; signed chord-normal arc centers preserve each
declared minor/major branch. Source coordinates are a soft prior, never locks.
"""
from __future__ import annotations

import copy
import json
import math
from numbers import Real
from pathlib import Path
import re

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

LINEAR_TOLERANCE_MM = .05
ANGLE_TOLERANCE_DEG = .1
MAX_DISPLACEMENT_FRACTION = .10
MAX_ENTITIES = 256
MAX_CONSTRAINTS = 1024
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


def _model(nodes,entities):
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
                index,sign=arc_params[entity["id"]];chord=b-a;length=float(np.linalg.norm(chord))
                normal=np.array([-chord[1],chord[0]])/max(length,1e-12);h=sign*max(vector[index],0)*scale
                entity.update(center=((a+b)/2+h*normal).tolist(),radius=float(math.hypot(length/2,h)))
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


def _sample_entity(entity,count=17):
    a=np.array(entity["start"]);b=np.array(entity["end"]);t=np.linspace(0,1,count)
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


def solve_parametric(graph,constraints,*,output_dir=None):
    """Solve declared constraints jointly; preserve baseline on every rejection.

    ``accepted`` permits a validated partial dimensional candidate even when
    ``underconstrained`` remains true. Neither flag establishes all drawing
    dimensions, annotation bindings, reference accuracy or engineering approval.
    Invalid input/schema raises ValueError before any optimization or output.
    """
    nodes,baseline,prepared=_validate_inputs(graph,constraints)
    initial,lower,upper,decode,scale,node_indices,arc_params,branches=_model(nodes,baseline)
    count=len(initial);normalizers=np.array([LINEAR_TOLERANCE_MM if row["kind"] in _DIMENSIONAL else ANGLE_TOLERANCE_DEG for row in prepared])
    initial_nodes,initial_entities=decode(initial)
    base_validation=_geometry_validation(initial_entities,branches,scale)
    result={"schema_version":"source-parametric-solver-v1","status":"no_constraints","accepted":False,"underconstrained":True,
            "units":graph["units"],"coordinate_system":copy.deepcopy(graph.get("coordinate_system")),
            "entities":copy.deepcopy(baseline),"nodes":copy.deepcopy(nodes),"baseline_entities":copy.deepcopy(baseline),
            "candidate_entities":None,"candidate_nodes":None,"constraints":[],"validation":{},
            "diagnostics":{"source_only":True,"ground_truth_used":False,"network_used":False,"source_coordinates_hard_locked":False,
                           "arc_parameterization":"shared endpoints + fixed-sign chord-normal center offset; fixed clockwise/minor-major branch",
                           "arc_branches":branches,"dimension_tolerance_mm":LINEAR_TOLERANCE_MM,"angle_tolerance_deg":ANGLE_TOLERANCE_DEG,
                           "reference_tolerance_changed":False,"independent_reference_accuracy_evaluated":False,
                           "prior_weight":.1,"variable_count":count,"max_displacement_fraction":MAX_DISPLACEMENT_FRACTION},
            "all_dimensions_verified":False,"dimension_solve_success":False,"engineering_verified":False}
    if not prepared:
        unchanged_validation=_geometry_validation(baseline,branches,scale)
        result["validation"]={**unchanged_validation,"passed":False,"geometry_valid":unchanged_validation["passed"],
                              "constraint_subset_satisfied":False,"all_dimensions_verified":False,"dimensions_verified":False,
                              "issues":[*unchanged_validation["issues"],"No explicit constraints were supplied; source geometry is retained."]}
        result["diagnostics"].update(constraint_rank=0,remaining_dof=count,remaining_shape_dof=max(0,count-3),rigid_gauge_dof=3)
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
    prior_weights=np.full(count,.1)
    radius_bound_offsets=[]
    radius_targets={}
    for row in prepared:
        if row["kind"]=="radius":
            index,_=arc_params[row["entities"][0]]
            # An observed radius supersedes the unreliable fitted center offset.
            # Penalizing a large initial h here can prefer collapsing the chord
            # merely to keep h close to its old value. Endpoints retain their
            # source prior; the exact radius equation determines h instead.
            prior_weights[index]=0.;radius_bound_offsets.append(row["entities"][0])
            radius_targets.setdefault(row["entities"][0],[]).append(row["value"])
    starting=initial.copy()
    radius_initialization=[]
    for entity_id,targets in radius_targets.items():
        entity=next(entity for entity in baseline if entity["id"]==entity_id)
        first=2*node_indices[entity["start_node"]];last=2*node_indices[entity["end_node"]]
        a=starting[first:first+2].copy();b=starting[last:last+2].copy()
        chord=float(np.linalg.norm(b-a))*scale;target=float(np.mean(targets));original_chord=chord
        if chord>2*target:
            # Closest endpoint-pair initialization on the radius feasibility
            # boundary, not an added constraint or a post-solve repair. All
            # endpoints remain free in the joint optimization and source guard.
            midpoint=(a+b)/2;half=(b-a)*(target/chord)
            starting[first:first+2]=midpoint-half;starting[last:last+2]=midpoint+half
            chord=2*target
        index,_=arc_params[entity_id]
        # A source dimension supplies an analytic starting offset. If its
        # radius is smaller than half the chord, h=0 leaves the joint optimizer
        # to move endpoints; it never silently changes the radius target.
        starting[index]=min(upper[index],math.sqrt(max(0.,target*target-(chord/2)**2))/scale)
        radius_initialization.append({"entity_id":entity_id,"target_radius":target,"original_chord":original_chord,
                                      "initialized_chord":chord,"endpoints_remain_free":True})
    def expand(reduced):
        vector=initial.copy();vector[active]=reduced;return vector
    def objective(reduced):
        vector=expand(reduced)
        return np.concatenate([constraint_residuals(vector),prior_weights[active]*(reduced-initial[active])])
    sparse_active=lil_matrix((len(prepared)+len(active),len(active)),dtype=int)
    sparse_active[:len(prepared)]=sparsity[:len(prepared),active]
    for index in range(len(active)):sparse_active[len(prepared)+index,index]=1
    try:
        linear_options={} if len(active)<=96 else {"jac_sparsity":sparse_active.tocsr(),"tr_solver":"lsmr",
                                                   "tr_options":{"atol":1e-12,"btol":1e-12,"maxiter":max(100,len(active)*5)}}
        # _model already expresses BOTH coordinates and arc offsets in one
        # bounding-box-normalized length unit. Re-scaling each variable by its
        # Jacobian column destroys that common source-prior metric: a short
        # segment's angle has a much larger derivative than a long segment's,
        # and Jacobian scaling can stall feasible underconstrained chains while
        # the trust region continues optimizing their soft prior. Keep the
        # existing normalized unit; objective, tolerances and success gate stay
        # unchanged.
        # Radius feasibility can require moving both endpoints before the
        # trust-region step settles.  A connected, valid CL60 source graph
        # reaches the unchanged numerical convergence gate after 726 calls;
        # the former 400-call budget rejected that valid candidate.  Keep a
        # finite budget and the same tolerances / acceptance checks.
        fit=least_squares(objective,starting[active],bounds=(lower[active],upper[active]),
                          max_nfev=1200,ftol=1e-11,xtol=1e-11,gtol=1e-9,x_scale=1.,**linear_options)
        solved_vector=expand(fit.x)
        node_map,candidate=decode(solved_vector)
        if not np.isfinite(solved_vector).all():raise FloatingPointError("Nonfinite optimization result")
        entity_map={entity["id"]:entity for entity in candidate};checks=[]
        for index,row in enumerate(prepared):
            actual=_constraint_value(row,node_map,entity_map);residual=_constraint_residual(row,actual);tolerance=float(normalizers[index])
            checks.append({**row,"actual":float(actual),"signed_residual":float(residual),"absolute_residual":float(abs(residual)),
                           "tolerance":tolerance,"residual_unit":"mm" if row["kind"] in _DIMENSIONAL else "degree",
                           "passed":bool(abs(residual)<=tolerance),"binding_verified_by_solver":False})
        candidate_nodes=[{**node,"x":float(node_map[node["id"]][0]),"y":float(node_map[node["id"]][1])} for node in nodes]
        validation=_geometry_validation(candidate,branches,scale)
        displacement=max(float(np.linalg.norm(node_map[node["id"]]-np.array([node["x"],node["y"]]))) for node in nodes)
        shape_displacement=max(float(np.linalg.norm(_sample_entity(after)-_sample_entity(before),axis=1).max()) for before,after in zip(initial_entities,candidate))
        movement_limit=max(.25 if graph["units"]=="mm" else 2.,scale*MAX_DISPLACEMENT_FRACTION)
        same_winding=base_validation["signed_area"]*validation["signed_area"]>0
        moved_safely=max(displacement,shape_displacement)<=movement_limit
        fitted_jacobian=fit.jac[:len(prepared)]
        if hasattr(fitted_jacobian,"toarray"):fitted_jacobian=fitted_jacobian.toarray()
        jacobian=np.zeros((len(prepared),count));jacobian[:,active]=fitted_jacobian
        singular=np.linalg.svd(jacobian,compute_uv=False);rank_tolerance=max(float(singular[0])*1e-6,1e-8) if len(singular) else 1e-8
        rank=int(np.count_nonzero(singular>rank_tolerance));remaining=count-rank
        gauges=[]
        for axis in range(2):
            vector=np.zeros(count);vector[axis:2*len(nodes):2]=1.;gauges.append(vector)
        rotation=np.zeros(count);xy=solved_vector[:2*len(nodes)].reshape((-1,2));rotation[:2*len(nodes)]=np.column_stack([-xy[:,1],xy[:,0]]).ravel();gauges.append(rotation)
        gauge_dof=sum(np.linalg.norm(jacobian@vector)<=max(1e-7,np.linalg.norm(jacobian)*np.linalg.norm(vector)*1e-7) for vector in gauges if np.linalg.norm(vector)>1e-10)
        shape_dof=max(0,remaining-int(gauge_dof));underconstrained=shape_dof>0
        all_passed=all(check["passed"] for check in checks)
        accepted=bool(fit.success and all_passed and validation["passed"] and moved_safely and same_winding)
        status="accepted" if accepted else "not_converged" if not fit.success else "conflict" if not all_passed else "invalid_geometry" if not validation["passed"] or not same_winding else "displacement_rejected"
        dimensional_checks=[check for check in checks if check["source"]!="source_geometry" and check["kind"] in _DIMENSIONAL|{"angle"}]
        issues=list(validation["issues"])
        if not all_passed:issues.append("Supplied constraints conflict or could not be jointly satisfied at the fixed numerical tolerances.")
        if not fit.success:issues.append("Bounded nonlinear solve did not converge; baseline is retained.")
        if not moved_safely:issues.append("Candidate exceeds the declared source-displacement guard; baseline is retained.")
        if not same_winding:issues.append("Candidate reverses the input contour orientation; baseline is retained.")
        if underconstrained:issues.append("Remaining shape freedoms are selected by a soft source prior; only the supplied constraint subset was solved.")
        validation.update(passed=accepted,geometry_valid=validation["passed"],constraint_subset_satisfied=all_passed,
                          all_dimensions_verified=False,dimensions_verified=False,reference_verified=False,engineering_certified=False,
                          underconstrained=underconstrained,source_displacement_passed=moved_safely,winding_preserved=same_winding,issues=issues)
        result.update(status=status,accepted=accepted,underconstrained=underconstrained,
                      entities=candidate if accepted else copy.deepcopy(baseline),nodes=candidate_nodes if accepted else copy.deepcopy(nodes),
                      candidate_entities=candidate,candidate_nodes=candidate_nodes,constraints=checks,validation=validation,
                      dimension_solve_success=bool(accepted and dimensional_checks and graph["units"]=="mm"))
        result["diagnostics"].update(converged=bool(fit.success),optimizer_status=int(fit.status),evaluations=int(fit.nfev),
                                     optimizer_message=str(fit.message),
                                     optimizer_parameter_scaling="common_bbox_normalized_length_unit",
                                     cost=float(fit.cost),optimality=float(fit.optimality),constraint_rank=rank,rank_tolerance=rank_tolerance,
                                     singular_values=singular.tolist(),remaining_dof=remaining,rigid_gauge_dof=int(gauge_dof),remaining_shape_dof=shape_dof,
                                     rank_excludes_soft_prior=True,independent_dimension_record_count=len(dimensional_checks),
                                     optimizer_active_variable_count=len(active),inactive_variables_at_soft_prior_minimum=count-len(active),
                                     radius_bound_arc_offsets_without_initial_h_penalty=sorted(set(radius_bound_offsets)),
                                     radius_feasibility_initialization=radius_initialization,
                                     linear_subsolver="exact_dense" if len(active)<=96 else "bounded_sparse_lsmr",
                                     maximum_node_displacement=displacement,maximum_parameterized_curve_displacement=shape_displacement,
                                     displacement_limit=movement_limit,displacement_unit=graph["units"],
                                     distance_axis_semantics="signed nodes[1] minus nodes[0]; x right, y up",
                                     angular_semantics="degree; angle defaults to unsigned 0..180; tangent compares directed traversal vectors at an explicit shared joint",
                                     acceptance_meaning="Valid source-preserving solution of supplied constraints only. Underconstraint, annotation binding correctness, all dimensions and reference accuracy remain separate.")
    except (ValueError,FloatingPointError,ArithmeticError,np.linalg.LinAlgError) as error:
        result.update(status="failed",validation={"passed":False,"all_dimensions_verified":False,"issues":[f"Numerical solve failed ({type(error).__name__}); baseline is retained."]})
        result["diagnostics"].update(error_type=type(error).__name__,converged=False)
    return _persist(result,output_dir)

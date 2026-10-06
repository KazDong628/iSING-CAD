"""Bounded source-path partitioning for two or three verified radius targets.

Outer endpoints are fixed source vertices; internal joints can be refined
within bounded source-error checks. Annotation radii are constants;
failed fixed-radius fits never fall back to freely fitted circles. This module
reads no files, images, providers or reference geometry.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .topology import _primitive_distance
from .vectorize import _sample_entities


def _landmarks(points, targets):
    count = len(points)
    if count <= 128:
        return list(range(count))
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    stations = np.r_[0., np.cumsum(lengths)]
    chosen = {0, count-1, *targets}
    chosen.update(np.searchsorted(stations, np.linspace(0., stations[-1], 72)).tolist())
    # Curvature changes are candidate source joints, never mandatory joints.
    # Bounded non-maximum suppression keeps raster zigzags from consuming the
    # entire search grid while preserving smooth circle-to-circle transitions.
    vectors = np.diff(points, axis=0)
    turn = np.arctan2(vectors[:-1, 0]*vectors[1:, 1]-vectors[:-1, 1]*vectors[1:, 0],
                      np.sum(vectors[:-1]*vectors[1:], axis=1))
    curvature = turn/np.maximum((lengths[:-1]+lengths[1:])/2, 1e-9)
    changes = np.abs(np.diff(curvature))
    added = []
    for offset in np.argsort(changes)[::-1]:
        index = int(offset)+2
        if all(abs(index-old)>2 for old in added):
            chosen.add(index);added.append(index)
        if len(added) >= 40:
            break
    return sorted(chosen)


def _dense_path(points, step):
    parts = []
    total = 0
    for a, b in zip(points, points[1:]):
        count = max(1, int(math.ceil(np.linalg.norm(b-a)/step)))
        total += count
        if total > 50000:
            raise ValueError("annotated_chain_source_sampling_budget_exceeded")
        parts.append(a+(b-a)*np.linspace(0., 1., count, endpoint=False)[:, None])
    return np.vstack([*parts, points[-1]])


def _fixed_arc(points, annotation, tolerance):
    radius = annotation["radius_px"]
    a, b = points[0], points[-1]
    chord = b-a; length = float(np.linalg.norm(chord))
    if len(points)<4 or length<1e-8 or length>2*radius:
        return None
    normal = np.array([-chord[1], chord[0]])/length
    height = math.sqrt(max(0., radius*radius-(length/2)**2))
    choices = []
    for sign in (-1, 1):
        center = (a+b)/2+sign*height*normal
        angles = np.unwrap(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))
        sweep = float(angles[-1]-angles[0])
        if not .02 < abs(sweep) < 1.8*math.pi or np.any(np.diff(angles)*np.sign(sweep)<-.01):
            continue
        arc = {"type":"ARC", "start":a.tolist(), "end":b.tolist(),
               "center":center.tolist(), "radius":radius, "clockwise":sweep<0}
        errors = _primitive_distance(points, arc)
        target_error = float(_primitive_distance(np.asarray([annotation["target_source_px"]]), arc)[0])
        if float(errors.max())>tolerance or target_error>max(1., tolerance):
            continue
        step = max(.05, min(.5, tolerance/4))
        sampled = _sample_entities([arc], max_step_px=step)[0]
        source = _dense_path(points, step)
        reverse = float(cKDTree(source).query(sampled, workers=1)[0].max())
        forward = float(cKDTree(sampled).query(source, workers=1)[0].max())
        if max(forward, reverse)>tolerance:
            continue
        binding = copy.deepcopy(annotation.get("binding") or {})
        binding.update(record_id=annotation["record_id"], nominal=annotation.get("nominal"),
                       arrowhead_verified=True, target_source_px=list(annotation["target_source_px"]))
        arc.update(fit_error_px=max(forward, reverse), radius_binding=binding,
                   radius_annotation_evidence=copy.deepcopy(binding), radius_binding_status="applied",
                   parameter_source="source_verified_multi_radius_resegmentation",
                   source_support_vertex_count=len(points))
        choices.append((max(forward, reverse), float(np.mean(errors**2)), arc))
    return min(choices, key=lambda row:row[:2]) if choices else None


def _arc_geometry(a, b, source, row, sign):
    """A circle through two endpoints with its annotated R as a constant."""
    delta=b-a;length=float(np.linalg.norm(delta));radius=row["radius_px"]
    if length<=1e-8 or length>2*radius:return None
    center=(a+b)/2+sign*math.sqrt(max(0.,radius*radius-(length/2)**2))*np.array([-delta[1],delta[0]])/length
    angles=np.unwrap(np.arctan2(source[:,1]-center[1],source[:,0]-center[0]));sweep=float(angles[-1]-angles[0])
    if not .02<abs(sweep)<1.8*math.pi or np.any(np.diff(angles)*np.sign(sweep)<-.01):return None
    return {"type":"ARC","start":a.tolist(),"end":b.tolist(),"center":center.tolist(),
            "radius":radius,"clockwise":sweep<0}


def _joint_seeds(points, rows, landmarks, maximum):
    # Keep a small beam for each source station, so a middle radius does not
    # inherit the only cut preferred by the first arc's mask-only minimax fit.
    best={0:[(0.,0.,[0],[])]};cache={}
    for stage,row in enumerate(rows):
        following={}
        ends=([len(points)-1] if stage==len(rows)-1 else
              [i for i in landmarks if row["source_index"]<=i<=rows[stage+1]["source_index"]])
        for first,paths in best.items():
            for last in ends:
                if not first<=row["source_index"]<=last or last-first<3:continue
                key=(stage,first,last)
                if key not in cache:
                    source=points[first:last+1];alternatives=[]
                    for sign in (-1,1):
                        arc=_arc_geometry(source[0],source[-1],source,row,sign)
                        if arc is not None:
                            errors=_primitive_distance(source,arc)
                            alternatives.append((float(errors.max()),float(np.mean(errors**2)),sign))
                    cache[key]=min(alternatives) if alternatives else None
                candidate=cache[key]
                if candidate is None:continue
                error,cost,sign=candidate
                for previous,old_cost,cuts,signs in paths:
                    following.setdefault(last,[]).append((max(previous,error),old_cost+cost*(last-first),
                                                         [*cuts,last],[*signs,sign]))
        best={end:sorted(paths,key=lambda p:p[:2])[:maximum] for end,paths in following.items()}
        if not best:return []
    return sorted(best.get(len(points)-1,[]),key=lambda p:p[:2])[:maximum]


def _refined_partitions(points, rows, tolerance, landmarks, *, objectives=("minimax","least_squares"), maximum=6):
    """Jointly refine all internal cuts of a two/three-radius source chain.

    The minimax fit controls worst-case mask error. A separate constrained
    least-squares fit reduces systematic displacement along an entire arc,
    which can otherwise lose original-ink support despite a small max error.
    Every output is recertified against the complete unmodified source path.
    """
    seeds=_joint_seeds(points,rows,landmarks,maximum)
    choices=[];joint_count=len(rows)-1;dimension=joint_count*2;step=max(.05,min(.5,tolerance/4))
    for initial_error,_,cuts,signs in seeds:
        parts=[points[a:b+1] for a,b in zip(cuts,cuts[1:])]
        anchors=points[cuts[1:-1]]
        # Bounded optimization uses no more than 160 samples per arc; final
        # certification below includes every source vertex and dense segment.
        fit_parts=[source[np.unique(np.linspace(0,len(source)-1,min(160,len(source))).astype(int))]
                   for source in parts]
        error_count=sum(len(source)+1 for source in fit_parts)
        def construct(vector):
            joints=anchors+np.asarray(vector[:dimension]).reshape(joint_count,2)*tolerance
            nodes=[points[0],*joints,points[-1]]
            arcs=[_arc_geometry(nodes[i],nodes[i+1],source,row,sign)
                  for i,(source,row,sign) in enumerate(zip(fit_parts,rows,signs))]
            return None if any(arc is None for arc in arcs) else arcs
        def errors(vector):
            arcs=construct(vector)
            if arcs is None:return np.full(error_count,1e6)
            values=[]
            for source,row,arc in zip(fit_parts,rows,arcs):
                values.extend((_primitive_distance(source,arc)/tolerance).tolist())
                values.append(float(_primitive_distance(np.asarray([row["target_source_px"]]),arc)[0])/max(1.,tolerance))
            return np.asarray(values)
        minimax=None
        for objective in objectives:
            if objective=="minimax":
                initial=np.r_[np.zeros(dimension),max(1.,float(errors(np.zeros(dimension)).max()))]
                fit=minimize(lambda x:x[-1]+1e-9*float(x[:dimension]@x[:dimension]),initial,method="SLSQP",
                             bounds=[(-1.,1.)]*dimension+[(0.,max(8.,initial[-1]+1.))],
                             constraints=[{"type":"ineq","fun":lambda x:x[-1]-errors(x)}],
                             options={"maxiter":120,"ftol":1e-10})
                if fit.success:minimax=fit.x[:dimension].copy()
            else:
                initial=minimax if minimax is not None else np.zeros(dimension)
                fit=minimize(lambda x:float(np.mean(errors(x)**2))+1e-9*float(x@x),initial,method="SLSQP",
                             bounds=[(-1.,1.)]*dimension,
                             constraints=[{"type":"ineq","fun":lambda x:1.-errors(x)}],
                             options={"maxiter":120,"ftol":1e-10})
            arcs=construct(fit.x)
            if not fit.success or arcs is None or float(errors(fit.x).max())>1.+1e-10:continue
            maximum_error=0.
            for source,row,arc,sign in zip(parts,rows,arcs,signs):
                # _arc_geometry must also preserve traversal order on the full
                # source interval, not merely on optimization samples.
                full=_arc_geometry(np.asarray(arc["start"]),np.asarray(arc["end"]),source,row,sign)
                if full is None:break
                dense=_dense_path(source,step);sampled=_sample_entities([arc],max_step_px=step)[0]
                error=max(float(cKDTree(dense).query(sampled,workers=1)[0].max()),
                          float(cKDTree(sampled).query(dense,workers=1)[0].max()))
                target_error=float(_primitive_distance(np.asarray([row["target_source_px"]]),arc)[0])
                if error>tolerance or target_error>max(1.,tolerance):break
                maximum_error=max(maximum_error,error)
                binding=copy.deepcopy(row.get("binding") or {})
                binding.update(record_id=row["record_id"],nominal=row.get("nominal"),arrowhead_verified=True,
                               target_source_px=list(row["target_source_px"]))
                arc.update(fit_error_px=error,radius_binding=binding,radius_annotation_evidence=copy.deepcopy(binding),
                           radius_binding_status="applied",parameter_source="source_verified_multi_radius_resegmentation",
                           source_support_vertex_count=len(source))
            else:
                movements=np.linalg.norm(fit.x[:dimension].reshape(joint_count,2)*tolerance,axis=1)
                choices.append({"entities":arcs,"schema_version":"source-multi-radius-partition-v1",
                    "record_ids":[row["record_id"] for row in rows],"cut_source_indices":cuts,
                    "source_endpoints_fixed":True,"source_points_moved":False,"joint_on_original_source_vertex":False,
                    "joint_refinement":"bounded_shared_joint_"+objective,
                    "joint_displacement_from_source_vertex_px":float(movements.max()),
                    "joint_displacements_from_source_vertices_px":movements.tolist(),
                    "optimizer_iterations":int(fit.nit),"radius_enforcement":"exact",
                    "maximum_bidirectional_error_px":maximum_error,"source_tolerance_px":tolerance,
                    "candidate_evaluations":len(seeds)*len(objectives),"landmark_count":len(landmarks),
                    "ground_truth_used":False,"whole_contour_validation_required":True,
                    "binding_verification_recheck_required":True})
    return choices


def _refine_two_arc_joint(points, rows, tolerance, landmarks):
    """Retain the public diagnostic entry point for one shared-joint minimax."""
    if len(rows)!=2:return None
    choices=_refined_partitions(points,rows,tolerance,landmarks,objectives=("minimax",),maximum=4)
    return min(choices,key=lambda row:row["maximum_bidirectional_error_px"]) if choices else None


def resegment_annotated_arcs(points, annotations, tolerance, *, source_entity_count, candidate_scorer=None):
    """Return exact-radius arcs and an audit, or reject an unsupported chain.

    Optional ``candidate_scorer(entities)`` ranks already source-certified pixel
    geometries by original-image evidence; higher finite scores are preferred.
    It does not relax mask error, independently verify bindings or replace the
    downstream full-contour acceptance gate. No scorer means legacy minimax
    selection, with bounded joint refinement only if a discrete fit fails.
    """
    points = np.asarray(points, float)
    if (points.ndim!=2 or points.shape[1]!=2 or not 8<=len(points)<=10000 or
            not np.isfinite(points).all() or not 1<=source_entity_count<=8):
        raise ValueError("annotated_chain_outside_bounded_source_contract")
    if not isinstance(annotations, list) or not 2<=len(annotations)<=3:
        raise ValueError("annotated_chain_requires_two_or_three_verified_radii")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int,float)) or not math.isfinite(tolerance) or tolerance<=0:
        raise ValueError("annotated_chain_requires_positive_source_tolerance")
    if candidate_scorer is not None and not callable(candidate_scorer):
        raise ValueError("annotated_chain_requires_callable_source_scorer")
    rows=[];seen=set()
    for original in annotations:
        row=copy.deepcopy(original)
        target=np.asarray(row.get("target_source_px"), float)
        radius=row.get("radius_px");record=row.get("record_id")
        if (row.get("source_arrow_verified") is not True or not isinstance(record,str) or record in seen or
                isinstance(radius,bool) or not isinstance(radius,(int,float)) or not math.isfinite(radius) or radius<=0 or
                target.shape!=(2,) or not np.isfinite(target).all()):
            raise ValueError("annotated_chain_requires_independently_verified_targets")
        seen.add(record);row["radius_px"]=float(radius)
        distances=np.linalg.norm(points-target,axis=1);row["source_index"]=int(np.argmin(distances))
        if float(distances.min())>max(1.,tolerance):
            raise ValueError("annotated_chain_target_does_not_reach_source_path")
        rows.append(row)
    rows.sort(key=lambda row:row["source_index"])
    targets=[row["source_index"] for row in rows]
    if len(set(targets))!=len(targets):
        raise ValueError("annotated_chain_target_order_is_ambiguous")
    landmarks=_landmarks(points,targets)
    beam=12 if candidate_scorer is not None else 1
    best={0:[(0.,0.,[],[0])]};evaluations=0;cache={}
    for stage,row in enumerate(rows):
        following={};last_stage=stage==len(rows)-1
        ends=[len(points)-1] if last_stage else [index for index in landmarks if targets[stage]<=index<=targets[stage+1]]
        for first,paths in best.items():
            for last in ends:
                if not first<=targets[stage]<=last or last-first<3:
                    continue
                key=(stage,first,last)
                if key not in cache:
                    evaluations+=1
                    cache[key]=_fixed_arc(points[first:last+1],row,tolerance)
                candidate=cache[key]
                if candidate is None:
                    continue
                error,mean_squared,arc=candidate
                for maximum,cost,entities,cuts in paths:
                    score=(max(maximum,error),cost+mean_squared*(last-first))
                    following.setdefault(last,[]).append((*score,[*entities,arc],[*cuts,last]))
        best={end:sorted(paths,key=lambda p:p[:2])[:beam] for end,paths in following.items()}
        if not best:
            break
    proposals=[]
    for maximum,cost,entities,cuts in best.get(len(points)-1,[]):
        proposals.append({"entities":copy.deepcopy(entities),"schema_version":"source-multi-radius-partition-v1",
            "record_ids":[row["record_id"] for row in rows],"cut_source_indices":cuts,
            "source_endpoints_fixed":True,"source_points_moved":False,
            "radius_enforcement":"exact","maximum_bidirectional_error_px":maximum,
            "source_tolerance_px":tolerance,"candidate_evaluations":evaluations,
            "landmark_count":len(landmarks),"ground_truth_used":False,
            "whole_contour_validation_required":True,"binding_verification_recheck_required":True})
    if not proposals or candidate_scorer is not None:
        proposals.extend(_refined_partitions(points,rows,tolerance,landmarks))
    if not proposals:
        raise ValueError("source_does_not_support_exact_annotated_arc_partition")
    if candidate_scorer is None:
        return min(proposals,key=lambda p:p["maximum_bidirectional_error_px"])
    for proposal in proposals:
        score=candidate_scorer(copy.deepcopy(proposal["entities"]))
        if isinstance(score,bool) or not isinstance(score,(int,float,np.floating)) or not math.isfinite(score):
            raise ValueError("annotated_chain_source_score_must_be_finite")
        proposal["source_candidate_score"]=float(score)
    chosen=max(proposals,key=lambda p:(p["source_candidate_score"],-p["maximum_bidirectional_error_px"]))
    chosen.update(source_scoring_applied=True,source_scored_candidate_count=len(proposals),
                  source_score_is_acceptance_certificate=False)
    return chosen

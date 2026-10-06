"""Source-image topology proposals for subsequent dimension binding/solving.

This module reads source pixels/OCR and a segmentation contour only. It never
opens training labels or reference CAD. Its coarse graph is deliberately NOT
the final CAD artifact or a relaxation of the existing vectorization gate.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from .ocr import canonical_records
from .vectorize import _arc, _closed_ring, _line, _sample_entities, assess_fit_quality, fit_polyline


def _write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")


def stroke_support_fraction(metrics, *, against=None):
    """Use explicit v2 stroke coverage, retaining old receipts unchanged."""
    if not isinstance(metrics, dict):
        raise TypeError("Source support metrics must be an object")
    common_v2=("stroke_supported_fraction" in metrics and
               (against is None or (isinstance(against,dict) and "stroke_supported_fraction" in against and
                                    against.get("support_measurement_version")==metrics.get("support_measurement_version"))))
    value = metrics.get("stroke_supported_fraction") if common_v2 else metrics.get("edge_supported_fraction")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid source support fraction")
    return float(value)


def _sample_path(points, step, maximum=30000):
    points=np.asarray(points,float)
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    distance=np.r_[0.,np.cumsum(lengths)]
    if distance[-1]<=0:
        return points[:1]
    count=min(maximum,max(2,int(math.ceil(distance[-1]/step))+1))
    targets=np.linspace(0,distance[-1],count)
    return np.c_[np.interp(targets,distance,points[:,0]),np.interp(targets,distance,points[:,1])]


def _primitive_distance(points, entity):
    points=np.asarray(points,float);a=np.asarray(entity["start"]);b=np.asarray(entity["end"])
    if entity["type"]=="LINE":
        delta=b-a;denom=float(delta@delta)
        if denom<=0:return np.linalg.norm(points-a,axis=1)
        t=np.clip((points-a)@delta/denom,0,1)
        return np.linalg.norm(points-(a+t[:,None]*delta),axis=1)
    center=np.asarray(entity["center"]);relative=points-center
    first=math.atan2(a[1]-center[1],a[0]-center[0]);last=math.atan2(b[1]-center[1],b[0]-center[0])
    sign=-1 if entity["clockwise"] else 1
    sweep=((last-first)*sign)%(2*math.pi)
    angles=((np.arctan2(relative[:,1],relative[:,0])-first)*sign)%(2*math.pi)
    radial=np.abs(np.linalg.norm(relative,axis=1)-entity["radius"])
    endpoints=np.minimum(np.linalg.norm(points-a,axis=1),np.linalg.norm(points-b,axis=1))
    return np.where((angles<=sweep+1e-10)|(angles>=2*math.pi-1e-10),radial,endpoints)


class _StrokeEvidence:
    """Distance and direction support from source strokes with OCR text masked."""
    def __init__(self, gray, records, grid_pitch):
        self.scale=min(1.,2400/max(gray.shape))
        small=cv2.resize(gray,None,fx=self.scale,fy=self.scale,interpolation=cv2.INTER_AREA)
        smooth=cv2.GaussianBlur(small,(3,3),.7)
        edges=cv2.Canny(smooth,60,160)
        self.text_mask=np.zeros_like(edges)
        boxes=0
        for row in records:
            box=row.get("box")
            if not box:continue
            p=np.asarray(box,float)
            if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():continue
            cv2.fillPoly(self.text_mask,[np.rint(p*self.scale).astype(np.int32)],255)
            boxes+=1
        # A loose OCR box can overlap a real long boundary. Preserve observed
        # edge pixels on long coherent strokes, rather than erasing that whole
        # boundary together with the glyphs. This does not invent missing ink.
        coherent=np.zeros_like(edges)
        lines=cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=max(30,10*grid_pitch*self.scale),
                             maxLineGap=max(3,2*grid_pitch*self.scale))
        if lines is not None:
            for x0,y0,x1,y1 in lines[:1000,0]:
                cv2.line(coherent,(int(x0),int(y0)),(int(x1),int(y1)),255,3)
        edges[(self.text_mask!=0)&(coherent==0)]=0
        self.interior_text_mask=self.text_mask.copy()
        self.text_mask[(coherent!=0)&(edges!=0)]=0
        gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
        distance,indices=distance_transform_edt(edges==0,return_indices=True)
        self.distance=distance.astype(np.float32)
        ny,nx=indices
        self.gx=gx[ny,nx];self.gy=gy[ny,nx]
        self.source_gray=small
        self.edge_gx=gx;self.edge_gy=gy
        self.edges=edges;self.grid=grid_pitch
        self.near=max(1.5,.35*grid_pitch*self.scale)
        self.metadata={"working_image_size":{"width":edges.shape[1],"height":edges.shape[0]},
                       "working_scale":self.scale,"edge_nearness_original_px":self.near/self.scale,
                       "orientation_tolerance_degrees":45.,"masked_ocr_text_regions":boxes,
                       "method":"Canny_edges_and_bilateral_dark_stroke_interiors_with_local_tangent_agreement",
                       "support_measurement_version":"source-stroke-support-v2",
                       "legacy_method":"Canny_stroke_distance_and_local_tangent_agreement_excluding_OCR_text_preserving_long_observed_strokes",
                       "interior_evidence":{"maximum_half_width_working_px":2*self.near,
                                            "maximum_gray_value_exclusive":170,
                                            "opposing_gradient_dot_maximum":-.8,
                                            "ocr_boxes_excluded":True},
                       "limitation":"Source strokes also contain dimension, hatch and construction lines; support is evidence, not verified boundary identity."}

    def _query_evidence(self, points):
        points=np.asarray(points,float)
        p=np.rint(points*self.scale).astype(int)
        inside=(p[:,0]>=0)&(p[:,0]<self.edges.shape[1])&(p[:,1]>=0)&(p[:,1]<self.edges.shape[0])
        x=np.clip(p[:,0],0,self.edges.shape[1]-1);y=np.clip(p[:,1],0,self.edges.shape[0]-1)
        distances=self.distance[y,x]/self.scale
        # Tangents span approximately a segmentation cell, suppressing raster
        # staircase directions without changing the geometry being queried.
        if len(points)>1:
            spacing=max(1e-8,float(np.median(np.linalg.norm(np.diff(points,axis=0),axis=1))))
            window=max(1,min(len(points)//2,int(round(self.grid/spacing))))
            before=points[np.maximum(0,np.arange(len(points))-window)]
            after=points[np.minimum(len(points)-1,np.arange(len(points))+window)]
            tangent=after-before
        else:tangent=np.zeros_like(points)
        normals=np.c_[self.gx[y,x],self.gy[y,x]]
        denom=np.linalg.norm(tangent,axis=1)*np.linalg.norm(normals,axis=1)
        dot=np.abs(np.sum(tangent*normals,axis=1))/np.maximum(denom,1e-10)
        aligned=(dot<=math.sin(math.radians(45)))&(denom>1e-10)
        # Nearest edges already exclude glyphs. Testing the query pixel's OCR
        # box again would incorrectly reject points beside a preserved real
        # boundary that crosses that box.
        edge_supported=inside&(distances<=self.near/self.scale)&aligned
        # A thick drawn line has two Canny edges; its dark centre can be farther
        # from both than the unchanged nearness tolerance. Admit that centre
        # only when nearby opposite edges bracket contiguous actual ink, their
        # normals agree with this path, and the point is outside every OCR box.
        tangent_norm=np.linalg.norm(tangent,axis=1)
        normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(tangent_norm[:,None],1e-10)
        eligible=(inside&~edge_supported&(tangent_norm>1e-10)&
                  (self.source_gray[y,x]<170)&(self.interior_text_mask[y,x]==0))
        candidates=np.flatnonzero(eligible)
        interior=np.zeros(len(points),bool)
        if len(candidates):
            centers=p[candidates];directions=normal[candidates]
            side_gradients=[];side_valid=[]
            for sign in (-1.,1.):
                active=np.ones(len(candidates),bool)
                accepted=np.zeros(len(candidates),bool)
                gradients=np.zeros((len(candidates),2),float)
                for offset in np.arange(.5,2*self.near+.01,.5):
                    query=np.rint(centers+sign*offset*directions).astype(int)
                    sx=np.clip(query[:,0],0,self.edges.shape[1]-1)
                    sy=np.clip(query[:,1],0,self.edges.shape[0]-1)
                    valid=(query[:,0]>=0)&(query[:,0]<self.edges.shape[1])&(query[:,1]>=0)&(query[:,1]<self.edges.shape[0])
                    hit=active&valid&(self.edges[sy,sx]!=0)
                    gradient=np.c_[self.edge_gx[sy,sx],self.edge_gy[sy,sx]]
                    magnitude=np.linalg.norm(gradient,axis=1)
                    unit=gradient/np.maximum(magnitude[:,None],1e-10)
                    compatible=(magnitude>1e-10)&(np.abs(np.sum(unit*directions,axis=1))>=math.cos(math.radians(45)))
                    accepted|=hit&compatible
                    gradients[hit]=unit[hit]
                    active&=valid&~hit&(self.source_gray[sy,sx]<170)
                    if not active.any():break
                side_gradients.append(gradients);side_valid.append(accepted)
            opposed=np.sum(side_gradients[0]*side_gradients[1],axis=1)<=-.8
            interior[candidates]=side_valid[0]&side_valid[1]&opposed
        return distances,edge_supported,interior

    def query(self, points):
        distances,edge_supported,interior=self._query_evidence(points)
        return distances,edge_supported|interior

    def summarize(self, points):
        samples=_sample_path(points,max(.5,1/self.scale))
        distances,edge_supported,interior=self._query_evidence(samples)
        legacy={"sample_count":len(samples),"edge_supported_fraction":float(edge_supported.mean()),
                "mean_edge_distance_px":float(distances.mean()),"p90_edge_distance_px":float(np.quantile(distances,.9)),
                "max_edge_distance_px":float(distances.max())}
        return {**legacy,"support_measurement_version":"source-stroke-support-v2",
                "stroke_supported_fraction":float((edge_supported|interior).mean()),
                "ink_interior_supported_fraction":float(interior.mean()),
                "legacy_edge_supported_fraction":legacy["edge_supported_fraction"],
                "legacy_edge_only_metrics":dict(legacy)}


def _grid_pitch(model, width, height):
    extraction=model.get("extraction",{})
    # Deliberately whitelist one model metadata field. Training provenance can
    # contain annotation paths and is neither inspected nor copied here.
    size=extraction.get("model",{}).get("size")
    if isinstance(size,int) and not isinstance(size,bool) and 128<=size<=4096:
        return max(width,height)/size,{"kind":"model_input_grid","size":size}
    mask=extraction.get("evidence",{}).get("mask_size",{})
    if all(isinstance(mask.get(k),int) and mask[k]>0 for k in ("width","height")):
        return max(width/mask["width"],height/mask["height"]),{"kind":"declared_segmentation_mask_grid","size":mask}
    return 1.,{"kind":"original_pixel_fallback","reason":"No model or mask grid size declared; no inferred low-resolution uncertainty."}


def _repair_local_shortcuts(ring, evidence, grid):
    """Replace unsupported local detours only when a straight source stroke wins."""
    points=ring[:-1];count=len(points);tree=cKDTree(points)
    proposals={};checks=[];intervals=set();interval_budget=4000
    for multiple in (1.,2.,4.,8.,16.):
        anchors=cv2.approxPolyDP(points.astype(np.float32)[:,None,:],multiple*grid,True)[:,0,:]
        indices=sorted(set([0,*tree.query(anchors)[1].tolist()]))
        for pair in zip(indices,indices[1:]+[count]):
            if len(intervals)>=interval_budget:break
            intervals.add(pair)
    fine=cv2.approxPolyDP(points.astype(np.float32)[:,None,:],.5*grid,True)[:,0,:]
    fine_indices=sorted(set([0,*tree.query(fine)[1].tolist(),count]))
    for i,first in enumerate(fine_indices[:-1]):
        for last in fine_indices[i+2:i+7]:
            if len(intervals)>=interval_budget:break
            intervals.add((first,last))
        if len(intervals)>=interval_budget:break
    for first,last in sorted(intervals):
            if last-first<2:continue
            part=ring[first:last+1];chord=float(np.linalg.norm(part[-1]-part[0]))
            length=float(np.linalg.norm(np.diff(part,axis=0),axis=1).sum())
            if not 2*grid<=chord<=96*grid or length<=1.08*chord:continue
            replacement={"type":"LINE","start":part[0].tolist(),"end":part[-1].tolist()}
            departure=float(_primitive_distance(part,replacement).max())
            if departure>24*grid:continue
            before=evidence.summarize(part);after=evidence.summarize([part[0],part[-1]])
            gain=stroke_support_fraction(after)-stroke_support_fraction(before)
            accept=(stroke_support_fraction(after)>=.72 and gain>=.18 and
                    after["mean_edge_distance_px"]<before["mean_edge_distance_px"])
            record={"start_source_index":first,"end_source_index":last,"removed_vertex_count":last-first-1,
                    "source_bbox_px":[part.min(axis=0).tolist(),part.max(axis=0).tolist()],
                    "maximum_local_change_px":departure,"before":before,"after":after,
                    "accepted":accept,"reason":"source_supported_straight_shortcut" if accept else "insufficient_source_evidence_for_removal"}
            key=(first,last)
            checks.append(record)
            if accept:proposals[key]=(gain,record)
    # Highest evidence gain first; overlapping edits cannot accumulate into a
    # larger unreviewed change. Every edit must preserve a single simple ring.
    removed=set();accepted=[]
    for (first,last),(gain,record) in sorted(proposals.items(),key=lambda item:(-item[1][0],item[0])):
        interval=set(range(first+1,last))
        if interval&removed or first in removed or last in removed:continue
        trial=[p for i,p in enumerate(points) if i not in removed|interval]
        poly=Polygon(trial)
        if not poly.is_valid or poly.area<=0:
            record["accepted"]=False;record["reason"]="shortcut_would_break_polygon"
            continue
        removed.update(interval);accepted.append(record)
    corrected=_closed_ring([p for i,p in enumerate(points) if i not in removed])
    accepted_keys={(r["start_source_index"],r["end_source_index"]) for r in accepted}
    unique={}
    for record in checks:
        key=(record["start_source_index"],record["end_source_index"])
        if key not in accepted_keys and record["accepted"]:
            record["accepted"]=False;record["reason"]="overlaps_higher_evidence_correction"
        unique[key]=record
    return corrected,accepted,list(unique.values())


def _local_edge_path(part, evidence, grid):
    """Bounded source-edge shortest path near a glyph-occluded boundary."""
    low=part.min(axis=0)-12*grid;high=part.max(axis=0)+12*grid
    low=np.maximum(low,[0.,0.]);high=np.minimum(high,[evidence.edges.shape[1]/evidence.scale-1,evidence.edges.shape[0]/evidence.scale-1])
    step=max(1/evidence.scale,.5*grid,float(max(high-low))/220)
    xs=np.arange(low[0],high[0]+step,step);ys=np.arange(low[1],high[1]+step,step)
    xx,yy=np.meshgrid(xs,ys)
    pixels=np.c_[xx.ravel(),yy.ravel()]
    sx=np.clip(np.rint(pixels[:,0]*evidence.scale).astype(int),0,evidence.edges.shape[1]-1)
    sy=np.clip(np.rint(pixels[:,1]*evidence.scale).astype(int),0,evidence.edges.shape[0]-1)
    edge_distance=evidence.distance[sy,sx]/evidence.scale
    old_distance=cKDTree(_sample_path(part,step)).query(pixels)[0]
    # Original glyph-supported wiggles are not privileged. The weak distance
    # term only keeps the search local; actual observed source strokes dominate.
    cost=(1+4*np.minimum(edge_distance/(evidence.near/evidence.scale),8)**2+.03*(old_distance/grid)**2).reshape(xx.shape)
    cost[old_distance.reshape(xx.shape)>24*grid]=np.inf
    start=tuple(np.rint((part[0]-low)/step).astype(int)[::-1]);end=tuple(np.rint((part[-1]-low)/step).astype(int)[::-1])
    start=tuple(min(max(v,0),cost.shape[i]-1) for i,v in enumerate(start))
    end=tuple(min(max(v,0),cost.shape[i]-1) for i,v in enumerate(end))
    queue=[(0.,0.,start)];best={start:0.};previous={};expanded=0
    while queue and expanded<50000:
        _,distance,point=heapq.heappop(queue)
        if distance!=best.get(point):continue
        if point==end:break
        expanded+=1
        for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            other=(point[0]+dy,point[1]+dx)
            if not (0<=other[0]<cost.shape[0] and 0<=other[1]<cost.shape[1]):continue
            candidate=distance+.5*(cost[point]+cost[other])*math.hypot(dx,dy)
            if candidate<best.get(other,math.inf):
                best[other]=candidate;previous[other]=point
                heuristic=math.hypot(other[0]-end[0],other[1]-end[1])
                heapq.heappush(queue,(candidate+heuristic,candidate,other))
    if end not in best or (start!=end and end not in previous):return None,{"expanded_cells":expanded,"status":"no_bounded_path"}
    route=[end]
    while route[-1]!=start:route.append(previous[route[-1]])
    route=np.asarray(route[::-1],float)
    path=np.c_[low[0]+route[:,1]*step,low[1]+route[:,0]*step]
    path[0]=part[0];path[-1]=part[-1]
    path=cv2.approxPolyDP(path.astype(np.float32)[:,None,:],.3*grid,False)[:,0,:].astype(float)
    path[0]=part[0];path[-1]=part[-1]
    return path,{"expanded_cells":expanded,"status":"candidate","grid_step_original_px":step}


def _repair_text_occlusions(ring, records, evidence, grid):
    changes=[];attempts=[]
    for row in records:
        if len(attempts)>=32:break
        box=row.get("box")
        if not box:continue
        box=np.asarray(box,float)
        if box.ndim!=2 or box.shape[1]!=2 or not np.isfinite(box).all():continue
        low=box.min(axis=0)-2*grid;high=box.max(axis=0)+2*grid
        mask=np.all((ring[:-1]>=low)&(ring[:-1]<=high),axis=1)
        indices=np.flatnonzero(mask)
        if len(indices)<3:continue
        # Skip a box touching two remote arcs or the seam. Such an annotation
        # cannot safely establish a single local replacement interval.
        if np.any(np.diff(indices)>1) or indices[0]==0 or indices[-1]>=len(ring)-2:continue
        first=int(indices[0])-1;last=int(indices[-1])+1;part=ring[first:last+1]
        if np.linalg.norm(np.diff(part,axis=0),axis=1).sum()>120*grid:continue
        before=evidence.summarize(part)
        if stroke_support_fraction(before)>=.72:continue
        path,search=_local_edge_path(part,evidence,grid)
        record={"method":"source_edge_path_around_ocr_overlap","record_id":row.get("id"),
                "source_bbox_px":[part.min(axis=0).tolist(),part.max(axis=0).tolist()],
                "before":before,"search":search,"accepted":False}
        if path is None:
            record["reason"]="bounded_source_path_unavailable";attempts.append(record);continue
        after=evidence.summarize(path)
        before_length=float(np.linalg.norm(np.diff(part,axis=0),axis=1).sum())
        after_length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
        trial=_closed_ring(np.vstack([ring[:first],path,ring[last+1:]]))
        valid=Polygon(trial).is_valid and Polygon(trial).area>0
        accept=(valid and stroke_support_fraction(after)>=.75 and
                stroke_support_fraction(after)-stroke_support_fraction(before)>=.2 and
                after["mean_edge_distance_px"]<before["mean_edge_distance_px"] and after_length<=1.25*before_length)
        record.update(after=after,before_length_px=before_length,after_length_px=after_length,
                      accepted=accept,reason="source_supported_text_overlap_reroute" if accept else "insufficient_source_support_or_shape_validity",
                      removed_local_path_px=part.tolist(),replacement_local_path_px=path.tolist())
        attempts.append(record)
        if accept:ring=trial;changes.append(record)
    return ring,changes,attempts


def _protect_source_features(points, candidate, evidence, grid, tolerance, depth=0):
    """Split a coarse primitive if it removes a source-supported local feature."""
    samples=_sample_path(points,max(.5,1/evidence.scale))
    _,supported=evidence.query(samples)
    deviation=_primitive_distance(samples,candidate)
    protected=supported&(deviation>max(grid,evidence.near/evidence.scale))
    supported_length=float(np.linalg.norm(np.diff(samples,axis=0),axis=1).sum())*float(protected.mean())
    if supported_length<2*grid or len(points)<=2 or depth>=12:
        if protected.any() and depth>=12:
            return [_line(points[i:i+2],tolerance) for i in range(len(points)-1)],1
        return [candidate],0
    target=samples[int(np.argmax(np.where(protected,deviation,-1.)))]
    split=int(np.argmin(np.linalg.norm(points-target,axis=1)))
    split=max(1,min(len(points)-2,split))
    entities=[];protections=1
    for part in (points[:split+1],points[split:]):
        fit=_line(part,tolerance) or _arc(part,tolerance)
        if fit is None:
            # Use small source segments rather than inventing a smooth bridge.
            entities.extend(_line(part[i:i+2],tolerance) for i in range(len(part)-1))
        else:
            fitted,extra=_protect_source_features(part,fit,evidence,grid,tolerance,depth+1)
            entities.extend(fitted);protections+=extra
    return entities,protections


def _candidate(ring, evidence, grid, multiple):
    tolerance=grid*multiple
    initial=fit_polyline(ring,tolerance_px=tolerance)
    points=ring[:-1];tree=cKDTree(points);closed=np.vstack([points,points])
    entities=[];protections=0
    for entity in initial:
        first=int(tree.query(entity["start"])[1]);last=int(tree.query(entity["end"])[1])
        if last<=first:last+=len(points)
        part=closed[first:last+1]
        protected,number=_protect_source_features(part,entity,evidence,grid,tolerance)
        entities.extend(protected);protections+=number
    sampled,_,_=_sample_entities(entities,max_step_px=max(.5,1/evidence.scale))
    polygon=Polygon(sampled)
    valid=bool(polygon.is_valid and polygon.area>0 and
               all(math.dist(e["end"],entities[(i+1)%len(entities)]["start"])<1e-7 for i,e in enumerate(entities)))
    support=evidence.summarize(sampled)
    return entities,{"grid_multiple":multiple,"tolerance_px":tolerance,"entity_count":len(entities),
                     "line_count":sum(e["type"]=="LINE" for e in entities),
                     "arc_count":sum(e["type"]=="ARC" for e in entities),
                     "protected_source_feature_splits":protections,"connected_simple_closed":valid,
                     "source_stroke_support":support}


def _tangent(entity, at_end):
    if entity["type"]=="LINE":
        result=np.asarray(entity["end"])-np.asarray(entity["start"])
    else:
        radial=np.asarray(entity["end"] if at_end else entity["start"])-np.asarray(entity["center"])
        result=np.array([-radial[1],radial[0]])*(-1 if entity["clockwise"] else 1)
    return result/max(1e-12,float(np.linalg.norm(result)))


def _relations(entities):
    relations=[]
    for entity in entities:
        if entity["type"]!="LINE":continue
        tangent=np.abs(_tangent(entity,False))
        relation="horizontal" if tangent[1]<=math.sin(math.radians(5)) else "vertical" if tangent[0]<=math.sin(math.radians(5)) else None
        if relation:relations.append({"type":relation,"entities":[entity["id"]],"source":"geometry_hypothesis","required":False})
    for index,entity in enumerate(entities):
        following=entities[(index+1)%len(entities)]
        if entity["type"]==following["type"]=="LINE":continue
        angle=math.degrees(math.acos(float(np.clip(_tangent(entity,True)@_tangent(following,False),-1,1))))
        if angle<=12:
            relations.append({"type":"tangent","entities":[entity["id"],following["id"]],
                              "source":"geometry_hypothesis","required":False,"observed_deviation_degrees":angle})
    for index,relation in enumerate(relations):relation["id"]=f"rel{index:03d}"
    return relations


def build_topology(image_path, document, model, output_dir):
    """Build a source-supported closed LINE/ARC hypothesis, not solved CAD."""
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    image=cv2.imdecode(np.fromfile(str(image_path),np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError("Cannot read source image")
    height,width=image.shape[:2]
    if width*height>80_000_000:raise ValueError("Source image exceeds 80 million pixels")
    extraction=model.get("extraction",{})
    ring=_closed_ring(extraction.get("raw_polyline_px") or extraction.get("polyline_px") or model.get("polyline_px"))
    if len(ring)>30000:raise ValueError("Source contour exceeds 30000 vertices")
    if not Polygon(ring).is_valid or Polygon(ring).area<=0:raise ValueError("Source contour must be a valid closed simple polygon")
    # Canonical seam/order makes graph IDs stable for a fixed source contour.
    seam=int(np.lexsort((ring[:-1,1],ring[:-1,0]))[0]);ring=_closed_ring(np.roll(ring[:-1],-seam,axis=0))
    grid,grid_evidence=_grid_pitch(model,width,height)
    records=canonical_records(document)
    source=_StrokeEvidence(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),records,grid)
    baseline_support=source.summarize(ring)
    oracle_mask=model.get("oracle_mask_conditioned") is True
    if oracle_mask:
        # This experiment explicitly supplies a complete material mask. Ink
        # shortcuts repair uncertain segmentation, so they cannot rewrite this
        # fixed geometric observation merely by finding a nearby hatch stroke.
        corrected=ring.copy();accepted=[];corrections=[]
    else:
        corrected,accepted,corrections=_repair_local_shortcuts(ring,source,grid)
        corrected,text_changes,text_attempts=_repair_text_occlusions(corrected,records,source,grid)
        accepted.extend(text_changes);corrections.extend(text_attempts)
    corrected_support=source.summarize(corrected)
    candidates=[];logs=[]
    for multiple in (() if oracle_mask else (.75,1.5,2.5)):
        try:
            entities,log=_candidate(corrected,source,grid,multiple)
            # Source-based selection only. Coarser shape hypotheses may exceed
            # the old mask tracing error, but cannot buy compactness by losing
            # substantial original stroke support or invalidating connectivity.
            eligible=(log["connected_simple_closed"] and
                      stroke_support_fraction(log["source_stroke_support"])>=stroke_support_fraction(corrected_support)-.035 and
                      log["source_stroke_support"]["p90_edge_distance_px"]<=corrected_support["p90_edge_distance_px"]+grid)
            log["eligible"]=eligible
            if eligible:candidates.append((len(entities),log["source_stroke_support"]["mean_edge_distance_px"],entities,log))
            logs.append(log)
        except (ValueError,ArithmeticError) as error:
            logs.append({"grid_multiple":multiple,"eligible":False,"error":str(error)})
    if oracle_mask:
        # Preserve the already validated initial CAD as the exact rollback
        # candidate. Later annotation-guided edits remain free to change its
        # decomposition, within the original mask representation budget.
        system=model["coordinate_system"]
        initial_scale=float(model["scale"]["pixels_per_mm"]) if system["units"]=="mm" else 1.
        initial_origin=np.asarray(system["origin_source_px"],float)
        entities=[]
        for item in model["entities"]:
            entity={"type":item["type"],"fit_error_px":0.}
            for key in ("start","end","center"):
                if key in item:entity[key]=(np.asarray(item[key],float)*[initial_scale,-initial_scale]+initial_origin).tolist()
            if item["type"]=="ARC":
                entity.update(radius=float(item["radius"])*initial_scale,clockwise=not bool(item["clockwise"]))
            entities.append(entity)
        selected={"tolerance_px":float(model.get("curve_fit",{}).get("tolerance_px") or .25),
                  "grid_multiple":None,"entity_count":len(entities),
                  "source_stroke_support":source.summarize(_sample_entities(entities)[0])}
        selection="oracle_mask_preserved_initial_cad_geometry"
    elif candidates:
        _,_,entities,selected=min(candidates,key=lambda item:(item[0],item[1]))
        selection="fewest_source_supported_valid_primitives_then_stroke_distance"
    else:
        entities=fit_polyline(corrected,tolerance_px=max(.25,.35*grid))
        sampled,_,_=_sample_entities(entities)
        if not Polygon(sampled).is_valid:
            entities=[_line(corrected[i:i+2],grid) for i in range(len(corrected)-1)]
        selected={"tolerance_px":max(.25,.35*grid),"grid_multiple":.35,"entity_count":len(entities),
                  "source_stroke_support":source.summarize(_sample_entities(entities)[0])}
        selection="no_coarse_candidate_supported_retained_fine_source_hypothesis"
    source_deviation=assess_fit_quality(ring,entities)
    coord=dict(model.get("coordinate_system") or {})
    units=coord.get("units","pixel")
    if units not in {"mm","pixel"}:raise ValueError("Topology requires declared mm or pixel model coordinates")
    scale=model.get("scale",{})
    ppm=float(scale.get("pixels_per_mm") or 1.) if units=="mm" else 1.
    if units=="mm" and (scale.get("status")!="resolved" or not math.isfinite(ppm) or ppm<=0):
        raise ValueError("Millimetre topology requires a resolved positive source scale")
    origin=np.asarray(coord.get("origin_source_px",[0.,0.]),float)
    if origin.shape!=(2,) or not np.isfinite(origin).all():raise ValueError("Invalid source coordinate origin")
    coord.update(units=units,origin_source_px=origin.tolist(),x="image right",y="image up")
    def xy(point):return [(float(point[0])-origin[0])/ppm,(origin[1]-float(point[1]))/ppm]
    nodes=[];geometries=[]
    for index,entity in enumerate(entities):
        node=f"v{index:03d}";point=xy(entity["start"])
        nodes.append({"id":node,"x":point[0],"y":point[1],"source_px":list(map(float,entity["start"]))})
        item={"id":f"g{index:03d}","type":entity["type"],"start_node":node,
              "end_node":f"v{(index+1)%len(entities):03d}","start":point,"end":xy(entity["end"]),
              "parameter_source":"source_topology_hypothesis","dimension_bound":False}
        if entity["type"]=="ARC":item.update(center=xy(entity["center"]),radius=entity["radius"]/ppm,clockwise=not entity["clockwise"])
        geometries.append(item)
    correction_evidence={"method":"source-supported-topology-v1","scope":"Topology proposal only; neither dimensional solution nor final CAD precision.",
                         "grid":grid_evidence,"source_grid_pitch_px":grid,"stroke_evidence":source.metadata,
                         "original_boundary":ring.tolist(),"corrected_boundary":corrected.tolist(),
                         "baseline_support":baseline_support,"corrected_support":corrected_support,
                         "accepted_local_corrections":accepted,"local_correction_candidates":corrections,
                         "coarse_candidates":logs,"selection":selection,"selected_candidate":selected,
                         "source_deviation_diagnostic":source_deviation,"baseline_modified":False,
                         "ground_truth_used":False,"dimensions_solved":False,"engineering_verified":False,
                         "boundary_observation_policy":"fixed_oracle_mask" if oracle_mask else "source_supported_segmentation_repair",
                         "mask_boundary_repair_skipped":oracle_mask}
    graph={"schema_version":"source-topology-v1","status":"proposal","units":units,"coordinate_system":coord,
           "nodes":nodes,"entities":geometries,"relations":_relations(geometries),
           "source_grid_pitch_px":grid,"proposal_tolerance_px":selected["tolerance_px"],
           "proposal_tolerance_units":selected["tolerance_px"]/ppm,
           "source_evidence":{"baseline_stroke_support":baseline_support,"proposal_stroke_support":selected["source_stroke_support"],
                              "source_deviation":source_deviation,"local_corrections":len(accepted),"selection":selection},
           "validation":{"closed":True,"connected":True,"simple":bool(source_deviation["sampled_topology_valid"]),
                         "ordered_entity_cycle":True,"dimensions_solved":False,"engineering_verified":False},
           "source_sha256":hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
           "ground_truth_used":False,"baseline_modified":False,"requires_dimension_binding":True,
           "scope":"Approximate object types, order and optional relations for annotation binding; source geometry parameters are initial guesses, not solved dimensions.",
           "limitations":["Ink can belong to dimensions or hatching; source stroke support alone does not certify boundary identity.",
                          "Only the selected material exterior is proposed; unseen islands, holes, omitted features and full tread restoration are not inferred.",
                          "Coarse object count is not a prescribed design primitive count; all numerical geometry remains an initialization for dimension solving."],
           "artifacts":{"topology":"topology.json","overlay":"topology-overlay.png","corrections":"correction-evidence.json"}}
    _write(output/"topology.json",graph);_write(output/"correction-evidence.json",correction_evidence)
    display_scale=min(1.,2400/max(width,height));canvas=cv2.resize(image,None,fx=display_scale,fy=display_scale)
    cv2.polylines(canvas,[np.rint(ring*display_scale).astype(np.int32)],True,(195,140,80),1)
    label_boxes=[]
    for index,entity in enumerate(entities):
        sample,_,_=_sample_entities([entity],max_step_px=max(.5,1/display_scale))
        cv2.polylines(canvas,[np.rint(sample*display_scale).astype(np.int32)],False,(30,130,20),2)
        middle=np.rint(sample[len(sample)//2]*display_scale).astype(int)
        point=np.rint(np.asarray(entity["start"])*display_scale).astype(int)
        for text,at,color in ((f"g{index:03d}",middle,(20,105,10)),(f"v{index:03d}",point,(150,40,150))):
            anchor=tuple(map(int,at));chosen=None
            size,_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.40,1)
            for distance in (5,16,28,42,60,80):
                for dx,dy in ((1,-1),(1,1),(-1,-1),(-1,1),(0,-1),(0,1)):
                    x=max(1,min(canvas.shape[1]-size[0]-2,anchor[0]+dx*distance))
                    y=max(size[1]+2,min(canvas.shape[0]-3,anchor[1]+dy*distance))
                    box=(x-2,y-size[1]-2,x+size[0]+2,y+3)
                    if all(box[2]<b[0] or b[2]<box[0] or box[3]<b[1] or b[3]<box[1] for b in label_boxes):
                        chosen=(x,y,box);break
                if chosen:break
            if not chosen:chosen=(anchor[0]+4,anchor[1]-4,(anchor[0],anchor[1]-12,anchor[0]+40,anchor[1]))
            x,y,box=chosen;label_boxes.append(box);at=(x,y)
            if math.dist(anchor,at)>18:cv2.line(canvas,anchor,at,(165,165,165),1,cv2.LINE_AA)
            cv2.putText(canvas,text,at,cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),3,cv2.LINE_AA)
            cv2.putText(canvas,text,at,cv2.FONT_HERSHEY_SIMPLEX,.40,color,1,cv2.LINE_AA)
        cv2.circle(canvas,tuple(point),2,(150,40,150),-1)
    cv2.imencode(".png",canvas)[1].tofile(str(output/"topology-overlay.png"))
    return graph

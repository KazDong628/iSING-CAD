"""Bounded original-pixel hypotheses and verification for one-bend R leaders.

No radius, CAD centre, GT coordinate or contour type is used to find a path.
The same filled-arrow and per-segment ink thresholds as straight leaders apply.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def segment_ink(gray, start, end, size):
    start, end = np.asarray(start, float), np.asarray(end, float)
    delta = end-start; length = float(np.linalg.norm(delta))
    if length < 3. or length > 5000.:
        return {'verified': False}
    unit = delta/length; normal = np.array([-unit[1], unit[0]])
    points = np.linspace(start, end, max(8, int(math.ceil(length))+1))
    ij = np.rint(points[:, None, :]+np.arange(-2., 3.)[None, :, None]*normal).astype(int)
    x,y = ij[:,:,0],ij[:,:,1]
    valid=(x>=0)&(x<gray.shape[1])&(y>=0)&(y<gray.shape[0])
    ink=np.zeros(x.shape,bool);ink[valid]=gray[y[valid],x[valid]]<170
    support=ink.any(axis=1);longest=running=0
    for present in support:
        running=0 if present else running+1;longest=max(longest,running)
    gap=max(5.,min(12.,size*.10))
    return {'verified':bool(support.mean()>=.88 and longest<=gap),
            'supported_fraction':float(support.mean()),'sample_count':len(points),
            'maximum_unobserved_run_px':longest,'maximum_allowed_gap_px':gap}


def elbow_hypotheses(gray, record, proposal, *, audit=None):
    from .constraint_binding import _box, _label_ray_entry
    receipt=audit if audit is not None else {}
    receipt.update(method='bounded_original_ink_single_elbow_search',search_complete=True,
                   maximum_hypotheses=64,nominal_used_to_rank=False,ground_truth_used=False)
    box=_box(record)
    try: tip=np.asarray(proposal['tip_px'],float);shaft=np.asarray(proposal['shaft_px'],float)
    except (KeyError,TypeError,ValueError):return []
    if (gray is None or box is None or tip.shape!=(2,) or shaft.shape!=(2,) or
        not np.isfinite([tip,shaft]).all() or proposal.get('record_id',record.get('id'))!=record.get('id')):return []
    low,high=box.min(axis=0),box.max(axis=0);size=max(12.,float(np.linalg.norm(high-low)))
    pad=min(64.,max(16.,.4*size))
    left=np.maximum(0,np.floor(np.minimum(low,tip)-pad)).astype(int)
    right=np.minimum([gray.shape[1],gray.shape[0]],np.ceil(np.maximum(high,tip)+pad)).astype(int)
    if np.min(right-left)<=0 or np.prod(right-left)>2_000_000:return []
    crop=gray[left[1]:right[1],left[0]:right[0]].copy()
    cv2.fillPoly(crop,[np.rint(box-left).astype(np.int32)],255)
    lines=cv2.HoughLinesP(cv2.Canny(crop,70,180),1.,np.pi/1440,12,minLineLength=12,maxLineGap=5)
    if lines is None:return []
    if len(lines)>2048:
        receipt.update(search_complete=False,reason='source_path_detector_budget_exhausted');return []
    first=[];last=[];last_segment_count=0
    for seg in lines[:,0].reshape(-1,2,2).astype(float)+left:
        a,b=seg;length=float(np.linalg.norm(b-a))
        if length<12.:continue
        unit=(b-a)/length
        gaps=np.linalg.norm(np.maximum(low-seg,0)+np.minimum(high-seg,0),axis=1)
        if float(gaps.min())<=max(12.,.15*size):first.append(seg)
        along=float((tip-a)@unit);lateral=abs(float((tip-a)@[-unit[1],unit[0]]))
        # Hough often ends on the arrow body, before its tapered tip. This is
        # the same bounded prelocalization allowance as the straight detector;
        # the final taper/shaft verification thresholds remain unchanged.
        tip_allowance=max(16.,.65*max(7.,min(90.,size*.52)))
        maximum_shift=min(48.,max(16.,.35*size))
        # Seed error has a longitudinal component too. Hough can stop at the
        # arrow base, so require the measured tip (below), not the noisy seed,
        # to stay in the existing tip-shift budget.
        longitudinal_allowance=tip_allowance+maximum_shift
        if lateral<=maximum_shift and -longitudinal_allowance<=along<=length+longitudinal_allowance:
            last_segment_count+=1
            for target in [a+along*unit,a,b]:
                if np.linalg.norm(target-tip)<=maximum_shift:
                    last.append((seg,target))
    if len(first)>64 or last_segment_count>64:
        receipt.update(search_complete=False,reason='source_path_segment_budget_exhausted');return []
    result=[]
    for a,b in first:
        for (c,d),snapped_tip in last:
            matrix=np.column_stack([b-a,c-d])
            determinant=float(np.linalg.det(matrix))
            if abs(determinant)<1e-8:continue
            t,s=np.linalg.solve(matrix,c-a);bend=a+t*(b-a)
            if not (-.25<=t<=1.25 and -.25<=s<=1.25):continue
            if min(np.linalg.norm(bend-a),np.linalg.norm(bend-b))>12.:continue
            if min(np.linalg.norm(bend-c),np.linalg.norm(bend-d))>20.:continue
            far=max((a,b),key=lambda q:np.linalg.norm(q-bend));u=bend-far;length=float(np.linalg.norm(u))
            if length<12.:continue
            u/=length;v=snapped_tip-bend;final_length=float(np.linalg.norm(v))
            if final_length<12.:continue
            v/=final_length
            angle=math.degrees(math.acos(float(np.clip(u@v,-1.,1.))))
            if not 20.<=angle<=150.:continue
            gap=_label_ray_entry(far,-u,low-2.,high+2.,max(18.,.75*size))
            if gap is None:continue
            root=far-u*gap
            if np.linalg.norm(root-bend)>max(160.,2*size):continue
            if not segment_ink(gray,root,bend,size)['verified']:continue
            # Hypotheses are deduplicated geometrically, without using R.
            if any(np.linalg.norm(bend-np.asarray(old['proposal']['leader_path_px'][1]))<1.5 and
                   np.linalg.norm(snapped_tip-np.asarray(old['proposal']['tip_px']))<1.5 for old in result):continue
            prop={'record_id':record.get('id'),'tip_px':snapped_tip.tolist(),
                  'shaft_px':(bend+min(8.,final_length*.2)*v).tolist(),
                  'leader_path_px':[root.tolist(),bend.tolist(),snapped_tip.tolist()]}
            result.append({'proposal':prop,'localization':{'method':'original_ink_single_elbow_path',
                'tip_shift_px':float(np.linalg.norm(snapped_tip-tip)),
                'maximum_tip_shift_px':maximum_shift,'nominal_used_to_rank':False,'ground_truth_used':False}})
            if len(result)>64:
                receipt.update(search_complete=False,reason='source_path_hypothesis_budget_exhausted');return []
    receipt['hypothesis_count']=len(result)
    return result


def verify_elbow_ink(gray, record, proposal, band, *, target_points=None):
    from .constraint_binding import _box,_label_ray_entry,_arrowhead_evidence
    from .source_arrow_localization import source_label_shaft_ownership
    if gray is None or gray.ndim!=2:return None
    box=_box(record)
    try: path=np.asarray(proposal['leader_path_px'],float).copy();tip=np.asarray(proposal['tip_px'],float)
    except (KeyError,TypeError,ValueError):return None
    if (box is None or path.shape!=(3,2) or not np.isfinite(path).all() or
        tip.shape!=(2,) or not np.isfinite(tip).all() or
        proposal.get('record_id',record.get('id'))!=record.get('id') or
        np.linalg.norm(path[-1]-tip)>1e-6 or
        np.any(path<0) or np.any(path>=[gray.shape[1],gray.shape[0]])):return None
    low,high=box.min(axis=0),box.max(axis=0);size=max(12.,float(np.linalg.norm(high-low)))
    vectors=np.diff(path,axis=0);lengths=np.linalg.norm(vectors,axis=1)
    if min(lengths)<6. or max(lengths)>5000.:return None
    units=vectors/lengths[:,None]
    gap=_label_ray_entry(path[0],-units[0],low-2.,high+2.,max(18.,.75*size))
    if gap is None:return None
    path[0]-=gap*units[0]
    arrow=_arrowhead_evidence(gray,tip,units[1],size,band,target_points)
    if arrow is None:return None
    actual=np.asarray(arrow['tip_px'],float)
    if np.linalg.norm(actual-tip)>max(10.,.55*float(arrow['length_px'])):return None
    path[-1]=actual
    checks=[segment_ink(gray,a,b,size) for a,b in zip(path[:-1],path[1:])]
    if not all(row['verified'] for row in checks):return None
    ownership=source_label_shaft_ownership(gray,record,path[1],units[0])
    if ownership.get('repetitive_label_crossing'):return None
    return {'method':'original_ink_single_elbow_arrow_verified','segment_px':[path[1].tolist(),actual.tolist()],
        'label_gap_px':float(gap),'label_ray_intersection_gap_px':float(gap),
        'source_label_exit_px':path[0].tolist(),'source_leader_path_px':path.tolist(),
        'leader_path_evidence':{'method':'each_original_segment_remeasured','verified':True,'segments':checks},
        'shaft_evidence':{'method':'complete_polyline_source_shaft_pixel_support','verified':True,
                         'supported_fraction':min(row['supported_fraction'] for row in checks)},
        'arrowhead':arrow,'arrowhead_verified':True,'target_source_px':actual.tolist(),
        'source_label_association':ownership,'target_conditioned_detection':target_points is not None,
        'score':float(gap)*.2,'nominal_used_to_rank':False}


def path_visibility(path,contours,band):
    from .constraint_binding import _leader_contour_visibility
    path=np.asarray(path,float);lengths=np.linalg.norm(np.diff(path,axis=0),axis=1)
    checks=[];remaining=None;first=None
    for i,(a,b) in enumerate(zip(path[:-1],path[1:])):
        row=_leader_contour_visibility(a,b,contours,band);checks.append(row)
        if remaining is None and row['first_intersection_to_target_px'] is not None:
            remaining=row['first_intersection_to_target_px']+float(lengths[i+1:].sum())
            first=row['first_intersection_px']
    return {'method':'first_source_contour_intersection_along_polyline','verified':remaining is None or remaining<=max(10.,1.7*band),
        'first_intersection_to_target_px':remaining,'first_intersection_px':first,
        'label_exit_px':path[0].tolist(),'target_tip_px':path[-1].tolist(),'segment_checks':checks}

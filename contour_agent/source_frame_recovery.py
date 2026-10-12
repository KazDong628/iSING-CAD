"""Bounded image-frame translation proposals; no CAD attributes are refitted."""
from __future__ import annotations
import copy
import math
import time
import cv2
import numpy as np
from .ocr import canonical_records
from .topology import _StrokeEvidence
from .vectorize import _sample_entities


def source_frame_candidate(image_path,document,baseline,graph,entities,*,max_seconds=8.):
    """Observe a small rigid displacement from original ink, never from GT.

    This is only a proposal. The caller must recheck every native equation,
    the unchanged original-domain displacement guard and full source gates.
    """
    receipt=dict(method='bounded_original_ink_translation_v1',accepted=False,
        reference_geometry_used=False,known_attributes_refitted=False,
        optimization_domain_recentered=False,trial_count=0)
    image=cv2.imdecode(np.fromfile(str(image_path),np.uint8),0)
    if image is None:return None,{**receipt,'reason':'source_image_unavailable'}
    ppm=float(baseline['scale']['pixels_per_mm']) if graph['units']=='mm' else 1.
    origin=np.asarray(baseline['coordinate_system']['origin_source_px'],float)
    grid=float(graph.get('source_grid_pitch_px') or 1.)
    radius=min(3.,max(grid,float((baseline.get('curve_fit') or {}).get('total_deviation_budget_px') or grid)))
    evidence=_StrokeEvidence(image,canonical_records(document),grid)
    pixels=_sample_entities(entities,max_step_px=max(.25,grid/ppm/2))[0]*[ppm,-ppm]+origin
    current=evidence.summarize(pixels)
    def score(check,shift):
        return (check['stroke_supported_fraction'],-check['p90_edge_distance_px'],-float(np.linalg.norm(shift)))
    best=score(current,[0,0]);delta=None;chosen=current
    deadline=time.monotonic()+max(0.,min(8.,float(max_seconds)))
    offsets=np.linspace(-radius,radius,13)
    trials=sorted(((float(x),float(y)) for x in offsets for y in offsets),key=lambda d:math.hypot(*d))
    for shift in trials:
        if time.monotonic()>=deadline:break
        check=evidence.summarize(pixels+shift);receipt['trial_count']+=1
        rank=score(check,shift)
        if rank>best:best,delta,chosen=rank,shift,check
    receipt.update(before=current,after=chosen,search_radius_px=radius,
        maximum_trials=169,translation_px=delta,publication_approved=False)
    if delta is None or chosen['stroke_supported_fraction']<=current['stroke_supported_fraction']+1e-4:
        return None,{**receipt,'reason':'no_source_supported_translation_improvement'}
    translation=np.asarray(delta)/[ppm,-ppm]
    result=copy.deepcopy(entities)
    for entity in result:
        for key in ('start','end','center'):
            if key in entity:entity[key]=(np.asarray(entity[key],float)+translation).tolist()
    return result,{**receipt,'translation_model_units':translation.tolist(),'proposal_created':True}

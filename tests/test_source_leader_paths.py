from copy import deepcopy
import cv2
import numpy as np

from contour_agent.constraint_binding import _verify_source_arrow_ink
from contour_agent.source_leader_paths import elbow_hypotheses
from contour_agent.source_arrow_localization import (localize_source_arrow_candidates,
    source_arrow_label_attachment,resolve_source_arrow_ownership)


def source():
    gray=np.full((180,230),255,np.uint8)
    cv2.putText(gray,'R10',(20,45),cv2.FONT_HERSHEY_SIMPLEX,.65,0,1,cv2.LINE_AA)
    root,bend,tip=np.asarray([[70.,40.],[110.,40.],[180.,130.]])
    cv2.line(gray,tuple(root.astype(int)),tuple(bend.astype(int)),0,2)
    cv2.line(gray,tuple(bend.astype(int)),tuple(tip.astype(int)),0,2)
    direction=(tip-bend)/np.linalg.norm(tip-bend);normal=np.array([-direction[1],direction[0]])
    base=tip-18*direction
    cv2.fillConvexPoly(gray,np.rint([tip,base+6*normal,base-6*normal]).astype(np.int32),0)
    record={'id':'r000','text':'R10','parsed':{'kind':'radius','nominal':10.},
            'box':[[18,20],[69,20],[69,55],[18,55]]}
    proposal={'record_id':'r000','tip_px':tip.tolist(),'shaft_px':[80.,40.]}
    return gray,record,proposal


def verify(gray,record,proposal):
    return localize_source_arrow_candidates(gray,record,proposal,None,4.,
        verifier=lambda im,row,pr,*unused:_verify_source_arrow_ink(im,row,pr,4.))


def test_single_elbow_is_checked_from_original_ink_without_radius_or_cad():
    gray,record,proposal=source();before=gray.copy()
    assert _verify_source_arrow_ink(gray,record,proposal,4.) is None
    candidates=verify(gray,record,proposal)
    assert len(candidates)==1
    row=candidates[0];attachment=source_arrow_label_attachment(gray,record,row)
    assert row['leader_path_evidence']['verified'] and row['shaft_evidence']['verified']
    assert attachment['strong_text_adjacency'] and attachment['directed_ray_reaches_text']
    assert all(check['supported_fraction']>=.88 for check in row['leader_path_evidence']['segments'])
    assert np.array_equal(gray,before)
    other=deepcopy(record);other['parsed']['nominal']=99999.
    assert elbow_hypotheses(gray,other,proposal)==elbow_hypotheses(gray,record,proposal)


def test_broken_first_or_last_leg_and_missing_arrow_cannot_be_repaired():
    gray,record,proposal=source()
    for x0,y0,x1,y1 in [(82,35,102,46),(135,70,150,95),(164,108,192,145)]:
        broken=gray.copy();broken[y0:y1,x0:x1]=255
        assert not verify(broken,record,proposal)
    wrong=deepcopy(record);wrong['box']=[[18,70],[69,70],[69,105],[18,105]]
    assert not verify(gray,wrong,proposal)


def test_elbow_does_not_bypass_global_label_ambiguity():
    gray,record,proposal=source();verified=verify(gray,record,proposal)[0]
    other=deepcopy(record);other.update(id='r001',text='R20')
    observations=[{**deepcopy(verified),'record_id':rid,'target_candidates':[]}
                  for rid in ['r000','r001']]
    accepted,rejected,audit=resolve_source_arrow_ownership(gray,[record,other],observations,band=4.)
    assert not accepted
    assert len(rejected)==2 and audit['accepted_observation_count']==0


def test_invented_bend_on_blank_pixels_never_becomes_full_shaft_evidence():
    gray,record,proposal=source()
    invented={**proposal,'leader_path_px':[[70.,40.],[130.,30.],[180.,130.]]}
    assert _verify_source_arrow_ink(gray,record,invented,4.) is None


def test_noisy_tip_is_snapped_within_existing_source_search_window():
    gray,record,proposal=source()
    noisy={**proposal,'tip_px':[192.,130.]}
    candidates=verify(gray,record,noisy)
    assert len(candidates)==1
    assert source_arrow_label_attachment(gray,record,candidates[0])['strong_text_adjacency']
    distant={**proposal,'tip_px':[225.,130.]}
    assert not verify(gray,record,distant)

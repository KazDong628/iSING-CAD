import copy
import cv2
import numpy as np
from contour_agent.source_frame_recovery import source_frame_candidate
from test_source_stroke_prior import fixture_graph


def test_original_ink_can_localize_rigid_gauge_without_changing_object_attributes(tmp_path):
    graph=fixture_graph();before=copy.deepcopy(graph)
    gray=np.full((120,120),255,np.uint8)
    cv2.rectangle(gray,(23,18),(93,88),0,1)
    path=tmp_path/'source.png';cv2.imwrite(str(path),gray)
    baseline={'scale':{'pixels_per_mm':1.},'coordinate_system':{'origin_source_px':[0.,0.]},
        'curve_fit':{'total_deviation_budget_px':3.}}
    entities,receipt=source_frame_candidate(path,{},baseline,graph,graph['entities'])
    assert entities is not None and graph==before
    assert receipt['after']['stroke_supported_fraction']>receipt['before']['stroke_supported_fraction']
    assert receipt['trial_count']<=169 and not receipt['publication_approved']
    for old,new in zip(graph['entities'],entities):
        assert np.asarray(new['end'])-new['start']==__import__('pytest').approx(np.asarray(old['end'])-old['start'])
        assert old['id']==new['id'] and old['type']==new['type']


def test_empty_source_does_not_invent_a_frame_correction(tmp_path):
    gray=np.full((120,120),255,np.uint8);path=tmp_path/'source.png';cv2.imwrite(str(path),gray)
    baseline={'scale':{'pixels_per_mm':1.},'coordinate_system':{'origin_source_px':[0.,0.]}}
    result,receipt=source_frame_candidate(path,{},baseline,fixture_graph(),fixture_graph()['entities'])
    assert result is None and not receipt['accepted']

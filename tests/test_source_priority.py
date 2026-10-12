from copy import deepcopy
import hashlib

import cv2
import numpy as np

from contour_agent import parametric_pipeline as p
from contour_agent.reconstruction_priority import source_first_model
from test_parametric_solver import rectangle


def source_fixture(tmp_path):
    graph=rectangle(20.,10.)
    graph.update(coordinate_system={'units':'mm','origin_source_px':[20.,100.],
        'x':'image right','y':'image up'},source_grid_pitch_px=1.,proposal_tolerance_px=2.,ground_truth_used=False,
        validation=dict(closed=True,connected=True,simple=True,ordered_entity_cycle=True))
    for node in graph['nodes']:
        node['source_px']=[20.+node['x'],100.-node['y']]
    ring=[n['source_px'] for n in graph['nodes']]
    gray=np.full((140,140),255,np.uint8)
    cv2.polylines(gray,[np.array(ring,np.int32)],True,0,3)
    image=tmp_path/'source.png';cv2.imwrite(str(image),gray)
    graph['source_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
    graph['source_evidence']={'baseline_stroke_support':{'edge_supported_fraction':1.,'p90_edge_distance_px':0.},
        'proposal_stroke_support':{'edge_supported_fraction':1.,'p90_edge_distance_px':0.}}
    model=dict(coordinate_system=deepcopy(graph['coordinate_system']),
        scale=dict(status='resolved',pixels_per_mm=1.),oracle_mask_conditioned=True,
        extraction=dict(raw_polyline_px=ring),curve_fit=dict(total_deviation_budget_px=.8))
    return image,graph,model


def test_source_first_keeps_failed_mask_diagnostic_and_exact_original_threshold(tmp_path):
    image,graph,model=source_fixture(tmp_path)
    # A deliberately incompatible raster prior; actual image and CAD coincide.
    model['extraction']['raw_polyline_px']=(np.array(model['extraction']['raw_polyline_px'])+[6.,0.]).tolist()
    before=deepcopy(model)
    legacy=p._topology_source_validation(image,model,graph)
    priority=source_first_model(model)
    fresh=p._topology_source_validation(image,priority,graph)
    solved=p._solved_source_validation(image,{},priority,graph,graph['entities'])
    assert not legacy['passed'] and fresh['passed'] and solved['passed']
    for check in (fresh,solved):
        assert check['oracle_mask_validation']['passed'] is False
        assert check['oracle_mask_validation']['original_deviation_budget_px']==.8
        assert check['priority_resolution']['source_mask_conflict']
        assert not check['priority_resolution']['mask_measurement_or_threshold_changed']
    assert not fresh['mask_fidelity_gate_used'] and model==before
    observation=p._solver_source_observation(priority,graph)
    assert 'boundary_error_budget' not in observation
    assert observation['mask_representation_budget_diagnostic']['maximum_deviation']==.8
    assert 'boundary_error_budget' in p._solver_source_observation(model,graph)


def test_source_first_cannot_bypass_hash_coordinates_or_ink(tmp_path):
    image,graph,model=source_fixture(tmp_path);model=source_first_model(model)
    graph['source_sha256']='changed'
    assert not p._topology_source_validation(image,model,graph)['passed']
    graph['source_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
    graph['nodes'][0]['source_px'][0]+=3
    assert not p._topology_source_validation(image,model,graph)['passed']
    bad=deepcopy(graph['entities'])
    for entity in bad:
        for end in ['start','end']:entity[end][0]+=30
    assert not p._solved_source_validation(image,{},model,graph,bad)['passed']


def test_published_mask_receipt_records_actual_policy_and_does_not_rewrite_failure():
    check={'passed':True,'oracle_mask_validation':{'applicable':True,'passed':False,
        'original_deviation_budget_px':.8}}
    original=deepcopy(check)
    for baseline,expected in [({},True),(source_first_model({}),False)]:
        model={}
        p._record_published_source_validation(model,check,baseline)
        assert model['validation']['source_mask_fidelity_used_as_acceptance_gate'] is expected
        assert model['validation']['solved_source_validation']['oracle_mask_validation']['passed'] is False
    assert check==original

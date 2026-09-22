"""Source-only synthetic geometry and real-notch protection tests."""
import copy
import json
import math

import cv2
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from contour_agent.topology import _StrokeEvidence, _local_edge_path, build_topology


def _source(tmp_path, target, contour=None, *, size=256, units="pixel", grid_size=128):
    image=np.full((size,size,3),255,np.uint8)
    cv2.polylines(image,[np.asarray(target,np.int32)],True,(0,0,0),2)
    path=tmp_path/"source.png";cv2.imencode('.png',image)[1].tofile(str(path))
    contour=target if contour is None else contour
    raw=np.vstack([contour,contour[0]]).astype(float).tolist()
    model={"extraction":{"raw_polyline_px":raw,"polyline_px":raw,
                         "model":{"size":grid_size,"training_provenance":{"untrusted_reference_path":"must_not_be_opened"}}},
           "scale":{"status":"resolved" if units=="mm" else "unresolved","pixels_per_mm":2. if units=="mm" else None},
           "coordinate_system":{"units":units,"origin_source_px":[20.,220.],"x":"image right","y":"image up"}}
    return path,model


RECTANGLE=[[20,40],[220,40],[220,220],[20,220]]


def test_graph_is_closed_ordered_and_original_model_is_unchanged(tmp_path):
    path,model=_source(tmp_path,RECTANGLE,units="mm")
    before=copy.deepcopy(model)
    graph=build_topology(path,{},model,tmp_path/"out")
    assert model==before
    assert graph['units']=='mm' and len(graph['entities'])==4
    assert graph['status']=='proposal' and graph['requires_dimension_binding']
    assert graph['baseline_modified'] is False and graph['ground_truth_used'] is False
    assert graph['validation']['simple'] and not graph['validation']['dimensions_solved']
    for index,entity in enumerate(graph['entities']):
        assert entity['id']==f'g{index:03d}' and entity['start_node']==f'v{index:03d}'
        assert entity['end_node']==f'v{(index+1)%len(graph["nodes"]):03d}'
        assert entity['end']==graph['entities'][(index+1)%len(graph['entities'])]['start']
        node=graph['nodes'][index]
        assert entity['start']==pytest.approx([(node['source_px'][0]-20)/2,(220-node['source_px'][1])/2])
    assert all(r['source']=='geometry_hypothesis' and r['required'] is False for r in graph['relations'])
    assert {r['type'] for r in graph['relations']}=={'horizontal','vertical'}
    assert (tmp_path/'out/topology-overlay.png').is_file()
    saved=json.loads((tmp_path/'out/topology.json').read_text(encoding='utf8'))
    assert saved==graph and 'training_provenance' not in json.dumps(graph)


def test_actual_source_notch_is_not_erased_by_coarse_fit(tmp_path):
    target=[[20,40],[94,40],[94,57],[106,57],[106,40],[220,40],[220,220],[20,220]]
    path,model=_source(tmp_path,target)
    graph=build_topology(path,{},model,tmp_path/'out')
    # A real 17 px notch has visible side/bottom strokes. The proposal must
    # preserve its exclusion region, regardless of its total primitive count.
    source_points=[node['source_px'] for node in graph['nodes']]
    polygon=Polygon(source_points)
    assert not polygon.contains(Point(100,47))
    assert polygon.contains(Point(100,68))
    assert any(abs(p[1]-57)<2 for p in source_points)
    evidence=json.loads((tmp_path/'out/correction-evidence.json').read_text())
    assert not evidence['accepted_local_corrections']


def test_unsupported_segmentation_spike_can_be_removed_with_source_evidence(tmp_path):
    contour=[[20,40],[93,40],[94,25],[106,25],[107,40],[220,40],[220,220],[20,220]]
    path,model=_source(tmp_path,RECTANGLE,contour)
    graph=build_topology(path,{},model,tmp_path/'out')
    evidence=json.loads((tmp_path/'out/correction-evidence.json').read_text())
    assert evidence['accepted_local_corrections']
    assert all(c['after']['edge_supported_fraction']>c['before']['edge_supported_fraction'] for c in evidence['accepted_local_corrections'])
    assert min(n['source_px'][1] for n in graph['nodes'])>=39
    assert graph['validation']['simple']


def test_ocr_text_is_not_accepted_as_shape_evidence(tmp_path):
    path,model=_source(tmp_path,RECTANGLE)
    doc={'records':[{'id':'text','text':'(38)','box':[[50,20],[120,20],[120,36],[50,36]]}]}
    graph=build_topology(path,doc,model,tmp_path/'out')
    evidence=json.loads((tmp_path/'out/correction-evidence.json').read_text())
    assert evidence['stroke_evidence']['masked_ocr_text_regions']==1
    assert not graph['validation']['dimensions_solved']


def test_unknown_units_stay_pixels_and_bad_mm_scale_is_rejected(tmp_path):
    path,model=_source(tmp_path,RECTANGLE)
    graph=build_topology(path,{},model,tmp_path/'out')
    assert graph['units']=='pixel'
    assert graph['proposal_tolerance_units']==graph['proposal_tolerance_px']
    model['coordinate_system']['units']='mm'
    with pytest.raises(ValueError,match='resolved positive'):
        build_topology(path,{},model,tmp_path/'bad')


def test_self_crossing_source_is_rejected_without_invented_repair(tmp_path):
    path,model=_source(tmp_path,RECTANGLE,[[20,40],[220,220],[220,40],[20,220]])
    with pytest.raises(ValueError,match='valid closed'):
        build_topology(path,{},model,tmp_path/'out')


def test_grid_size_and_repeatability_are_source_driven(tmp_path):
    path,model=_source(tmp_path,RECTANGLE,size=512,grid_size=256)
    first=build_topology(path,{},model,tmp_path/'a')
    second=build_topology(path,{},model,tmp_path/'b')
    assert first==second and first['source_grid_pitch_px']==2.
    assert first['validation']['ordered_entity_cycle']


def test_bounded_edge_path_replaces_glyph_detour_with_observed_corner():
    gray=np.full((180,220),255,np.uint8)
    cv2.line(gray,(30,140),(30,40),0,2)
    cv2.line(gray,(30,40),(190,40),0,2)
    # A falsely segmented text-shaped detour stands well inside a true L edge.
    part=np.array([[30.,100.],[65.,100.],[65.,70.],[85.,70.],[85.,100.],[115.,100.],[115.,40.]])
    evidence=_StrokeEvidence(gray,[],5.)
    path,search=_local_edge_path(part,evidence,5.)
    assert path is not None and search['expanded_cells']<=50000
    assert np.allclose(path[0],part[0]) and np.allclose(path[-1],part[-1])
    before=evidence.summarize(part);after=evidence.summarize(path)
    assert after['edge_supported_fraction']>.8
    assert after['mean_edge_distance_px']<before['mean_edge_distance_px']/4
    assert LineString(path).distance(Point(30,40))<6


def test_real_notch_survives_an_adjacent_dimension_stroke(tmp_path):
    target=[[20,40],[94,40],[94,57],[106,57],[106,40],[220,40],[220,220],[20,220]]
    path,model=_source(tmp_path,target)
    image=cv2.imread(str(path))
    cv2.line(image,(80,32),(120,32),(0,0,0),1)
    cv2.imencode('.png',image)[1].tofile(str(path))
    graph=build_topology(path,{},model,tmp_path/'out')
    polygon=Polygon([node['source_px'] for node in graph['nodes']])
    assert not polygon.contains(Point(100,47)) and polygon.contains(Point(100,68))

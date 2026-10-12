import copy
import cv2
import numpy as np
import pytest

from contour_agent.source_stroke_prior import source_stroke_observation, stroke_prior
from contour_agent.parametric_solver import solve_parametric


def fixture_graph():
    points = [[20., -20.], [90., -20.], [90., -90.], [20., -90.]]
    nodes = [{"id": f"v{i}", "x": p[0], "y": p[1]} for i, p in enumerate(points)]
    entities = [{"id": f"g{i}", "type": "LINE", "start": p, "end": points[(i+1)%4],
                 "start_node": f"v{i}", "end_node": f"v{(i+1)%4}"} for i, p in enumerate(points)]
    return {"units": "mm", "nodes": nodes, "entities": entities, "source_grid_pitch_px": 1.}


def test_observation_uses_actual_ink_and_preserves_unobserved_closure(tmp_path):
    gray = np.full((120, 120), 255, np.uint8)
    cv2.line(gray, (20, 20), (90, 20), 0, 2)
    path = tmp_path/"source.png"
    cv2.imwrite(str(path), gray)
    graph = fixture_graph()
    baseline = {"scale": {"pixels_per_mm": 1.}, "coordinate_system": {"origin_source_px": [0., 0.]},
                "curve_fit": {"total_deviation_budget_px": 2.}}
    observation = source_stroke_observation(path, {"records": []}, baseline, graph)
    rows = observation["entities"]
    assert len(rows[0]["indices"]) > 20
    assert rows[2]["indices"] == []
    assert observation["reference_dxf_read"] is False
    residual, audit = stroke_prior(observation, graph, graph["entities"], 100.)
    before = np.linalg.norm(residual(graph["entities"]))
    moved = copy.deepcopy(graph["entities"])
    moved[0]["start"][1] -= 2.
    moved[0]["end"][1] -= 2.
    assert np.linalg.norm(residual(moved)) > before
    assert audit["hard_constraint"] is False


def test_normal_prior_does_not_lock_tangential_sample_positions():
    graph = fixture_graph()
    observation = {"units": "mm", "source": "original_image_direction_compatible_ink",
                   "reference_dxf_read": False, "hard_constraint": False,
                   "entities": [{"entity_id": "g0", "sample_count": 3, "indices": [1],
                                 "target_points": [[55., -20.]], "normals": [[0., 1.]]}]}
    residual, _ = stroke_prior(observation, graph, graph["entities"], 100.)
    moved = copy.deepcopy(graph["entities"])
    moved[0]["start"][0] += 5.
    moved[0]["end"][0] += 5.
    assert residual(moved) == pytest.approx([0.])
    for key, value in (("reference_dxf_read", True), ("hard_constraint", True), ("units", "px")):
        bad = copy.deepcopy(observation)
        bad[key] = value
        with pytest.raises(ValueError):
            stroke_prior(bad, graph, graph["entities"], 100.)


def test_refresh_from_current_candidate_reobserves_ink_without_resetting_domain(tmp_path):
    gray=np.full((120,120),255,np.uint8)
    cv2.line(gray,(20,22),(90,22),0,2)
    path=tmp_path/'source.png';cv2.imwrite(str(path),gray)
    graph=fixture_graph();before=copy.deepcopy(graph)
    baseline={'scale':{'pixels_per_mm':1.},'coordinate_system':{'origin_source_px':[0.,0.]},
              'curve_fit':{'total_deviation_budget_px':2.}}
    moved=copy.deepcopy(graph['entities'])
    moved[0]['start'][1]-=2.;moved[0]['end'][1]-=2.
    observation=source_stroke_observation(path,{},baseline,graph,probe_entities=moved)
    assert observation['probe_geometry']=='current_numerical_candidate'
    assert observation['optimization_domain_recentered'] is False
    assert observation['reference_dxf_read'] is False and graph==before
    assert len(observation['entities'][0]['indices'])>20
    moved[0]['id']='foreign'
    with pytest.raises(ValueError,match='preserve_source_topology'):
        source_stroke_observation(path,{},baseline,graph,probe_entities=moved)


def test_previous_iterate_seed_cannot_reset_original_source_budget():
    graph=fixture_graph()
    points=[e["start"] for e in graph["entities"]]+[graph["entities"][0]["start"]]
    observation={"units":"mm","points":points,"provenance":"input_mask_boundary","reference_dxf_read":False,
        "boundary_error_budget":{"units":"mm","source":"initial_curve_fit_total_deviation_budget_px",
            "maximum_deviation":.5,"sampling_step":.1,"max_wall_seconds":5.}}
    constraint={"id":"width","kind":"distance_x","record_id":"r0","nodes":["v0","v1"],
                "entities":[],"value":70.,"source":"ocr_api_binding","required":True,
                "enforcement":"exact","nominal_source":"source_ocr"}
    seed=copy.deepcopy(graph["entities"])
    for row in seed:
        row["start"][1]+=.1
        row["end"][1]+=.1
    before=copy.deepcopy(graph)
    solution=solve_parametric(graph,[constraint],source_observation=observation,seed_entities=seed)
    assert solution["accepted"]
    assert solution["baseline_entities"]==before["entities"] and graph==before
    assert solution["diagnostics"]["source_seed"]["source_guard_origin_reset"] is False
    assert solution["diagnostics"]["source_budget_search"]["maximum_deviation"]==.5
    seed[0]["id"]="different_graph"
    with pytest.raises(ValueError):
        solve_parametric(graph,[constraint],source_observation=observation,seed_entities=seed)


def test_source_first_observes_stroke_centre_instead_of_nearest_acceptable_white_pixel(tmp_path):
    from contour_agent.reconstruction_priority import source_first_model
    gray=np.full((120,120),255,np.uint8)
    cv2.line(gray,(20,23),(90,23),0,3)
    path=tmp_path/'source.png';cv2.imwrite(str(path),gray)
    baseline=source_first_model({'scale':{'pixels_per_mm':1.},
        'coordinate_system':{'origin_source_px':[0.,0.]},'curve_fit':{'total_deviation_budget_px':6.}})
    observation=source_stroke_observation(path,{},baseline,fixture_graph())
    row=observation['entities'][0]
    assert len(row['indices'])>20
    assert np.asarray(row['target_points'])[:,1]==pytest.approx(-23.,abs=.3)
    assert observation['matching']=='source_strip_centre_to_native_primitive'


def test_source_first_parallel_tracks_are_not_resolved_using_the_proposal(tmp_path):
    from contour_agent.reconstruction_priority import source_first_model
    gray=np.full((120,120),255,np.uint8)
    for y in (18,24):cv2.line(gray,(20,y),(90,y),0,1)
    path=tmp_path/'source.png';cv2.imwrite(str(path),gray)
    baseline=source_first_model({'scale':{'pixels_per_mm':1.},
        'coordinate_system':{'origin_source_px':[0.,0.]},'curve_fit':{'total_deviation_budget_px':6.}})
    row=source_stroke_observation(path,{},baseline,fixture_graph())['entities'][0]
    assert row['indices']==[]


def test_native_circle_ink_residual_does_not_lock_arc_sample_parameterization():
    # The same circle with changed contacts still supports the observed ink.
    arc={'id':'a','type':'ARC','start':[10.,0.],'end':[0.,10.],
         'center':[0.,0.],'radius':10.,'clockwise':False}
    graph={'units':'mm','nodes':[{},{}],'entities':[arc]}
    observation={'units':'mm','source':'original_image_direction_compatible_ink',
        'reference_dxf_read':False,'hard_constraint':False,
        'matching':'source_strip_centre_to_native_primitive',
        'entities':[{'entity_id':'a','sample_count':3,'indices':[1],
            'target_points':[[8.,6.]],'normals':[[.8,.6]]}]}
    residual,audit=stroke_prior(observation,graph,[arc],100.)
    moved=copy.deepcopy(arc);moved['end']=[-6.,8.]
    assert residual([moved])==pytest.approx([0.])
    moved['center']=[.5,0.]
    assert abs(residual([moved])[0])>0
    assert audit['entity_observed_fractions']['a']==pytest.approx(1/3)


def test_original_ink_sampling_scales_with_length_but_remains_bounded(tmp_path):
    gray=np.full((140,140),255,np.uint8)
    cv2.line(gray,(20,20),(130,20),0,2)
    path=tmp_path/'source.png';cv2.imwrite(str(path),gray)
    graph=fixture_graph()
    graph['entities'][0]['end']=[130.,-20.]
    baseline={'scale':{'pixels_per_mm':1.},'coordinate_system':{'origin_source_px':[0.,0.]},
        'curve_fit':{'total_deviation_budget_px':2.}}
    observation=source_stroke_observation(path,{},baseline,graph)
    count=observation['entities'][0]['sample_count']
    assert count==55
    graph['entities'][0]['end']=[10000.,-20.]
    assert source_stroke_observation(path,{},baseline,graph)['entities'][0]['sample_count']==65


@pytest.mark.parametrize('kind',['ARC','LINE'])
def test_native_ink_requires_finite_primitive_extent(kind):
    entity=({'id':'a','type':'ARC','start':[10.,0.],'end':[0.,10.],
        'center':[0.,0.],'radius':10.,'clockwise':False} if kind=='ARC' else
        {'id':'a','type':'LINE','start':[0.,0.],'end':[10.,0.]})
    point=[-8.,6.] if kind=='ARC' else [12.,0.]
    graph={'units':'mm','nodes':[{},{}],'entities':[entity]}
    observation={'units':'mm','source':'original_image_direction_compatible_ink',
        'reference_dxf_read':False,'hard_constraint':False,
        'matching':'source_strip_centre_to_native_primitive',
        'entities':[{'entity_id':'a','sample_count':3,'indices':[1],
            'target_points':[point],'normals':[[0.,1.]]}]}
    residual,audit=stroke_prior(observation,graph,[entity],100.)
    assert residual([entity])[0]>0
    assert audit['finite_primitive_matching']


def test_native_ink_allows_contact_migration_only_to_topological_neighbor():
    entities=[{'id':'a','type':'LINE','start':[0.,0.],'end':[10.,0.],
        'start_node':'v0','end_node':'v1'},
        {'id':'b','type':'LINE','start':[10.,0.],'end':[20.,0.],
        'start_node':'v1','end_node':'v2'},
        {'id':'other','type':'LINE','start':[30.,0.],'end':[40.,0.],
        'start_node':'x0','end_node':'x1'}]
    graph={'units':'mm','nodes':[{}, {}, {}],'entities':entities}
    observation={'units':'mm','source':'original_image_direction_compatible_ink',
        'reference_dxf_read':False,'hard_constraint':False,
        'matching':'source_strip_centre_to_native_primitive',
        'entities':[{'entity_id':'a','sample_count':3,'indices':[1],
            'target_points':[[9.,0.]],'normals':[[0.,1.]]}]}
    residual,_=stroke_prior(observation,graph,entities,100.)
    moved=copy.deepcopy(entities);moved[0]['end']=[8.,0.];moved[1]['start']=[8.,0.]
    assert residual(moved)==pytest.approx([0.])
    moved[1]['start']=[10.,0.];moved[2]['start']=[8.,0.]
    assert residual(moved)[0]>0


def test_thick_ink_aperture_observes_both_edges_without_increasing_centre_budget(tmp_path):
    from contour_agent.reconstruction_priority import source_first_model
    gray=np.full((120,120),255,np.uint8)
    cv2.line(gray,(20,21),(90,21),0,3)
    path=tmp_path/'source.png';cv2.imwrite(str(path),gray)
    baseline=source_first_model({'scale':{'pixels_per_mm':1.},
        'coordinate_system':{'origin_source_px':[0.,0.]},'curve_fit':{'total_deviation_budget_px':2.}})
    observation=source_stroke_observation(path,{},baseline,fixture_graph())
    assert observation['search_radius_px']==2.
    assert observation['measurement_aperture_radius_px']>2.
    assert len(observation['entities'][0]['indices'])>20
    assert np.asarray(observation['entities'][0]['target_points'])[:,1]==pytest.approx(-21.,abs=.3)

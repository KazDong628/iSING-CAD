from copy import deepcopy

from contour_agent.parametric_pipeline import _solver_source_observation
from contour_agent.parametric_solver import solve_parametric
from contour_agent.reconstruction_priority import source_first_model
from contour_agent.source_stroke_prior import source_stroke_observation
from test_source_priority import source_fixture


def test_source_observations_replace_fitted_coordinate_target_without_recentring_bounds(tmp_path):
    image,graph,model=source_fixture(tmp_path)
    model=source_first_model(model);before=deepcopy(graph)
    observation=_solver_source_observation(model,graph)
    observation['image_stroke_prior']=source_stroke_observation(image,{},model,graph)
    constraints=[{'id':'width','kind':'distance_x','nodes':['v0','v1'],'entities':[],
        'value':20.,'record_id':'r0','source':'ocr_api_binding','required':True,
        'enforcement':'exact','nominal_source':'source_ocr'}]
    result=solve_parametric(graph,constraints,source_observation=observation)
    assert result['accepted'] and all(row['passed'] for row in result['constraints'])
    role=result['diagnostics']['initial_fit_role']
    assert role['policy']=='initial_guess_only_with_original_source_observations'
    assert not role['coordinate_penalty_enabled']
    assert role['input_mask_soft_localization_retained'] and role['original_image_objective_retained']
    assert not role['source_bounds_recentered'] and graph==before


def test_missing_original_ink_retains_explicit_unresolved_fallback(tmp_path):
    _,graph,model=source_fixture(tmp_path)
    observation=_solver_source_observation(source_first_model(model),graph)
    result=solve_parametric(graph,[{'id':'h','kind':'horizontal','entities':['g0'],
        'nodes':[],'source':'source_geometry'}],source_observation=observation)
    role=result['diagnostics']['initial_fit_role']
    assert role['coordinate_penalty_enabled']
    assert role['policy']=='legacy_soft_coordinate_regularization'
    assert result['underconstrained']

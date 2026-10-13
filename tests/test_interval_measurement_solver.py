from copy import deepcopy
import math
import pytest
from contour_agent.dimension_policy import source_dimension_interval
from contour_agent.parametric_solver import solve_parametric,native_measurement_checks
from test_parametric_solver import rectangle


def interval_row(kind='distance_x',entities=None,nodes=None,nominal=20,upper=3):
    _,metadata=source_dimension_interval(dict(kind='length',nominal=nominal,lower_deviation=0,upper_deviation=upper))
    return dict(id='thickness',record_id='rthickness',kind=kind,entities=entities or [],nodes=nodes or [],
        value=float(nominal),source='ocr_api_binding',required=True,nominal_source='source_ocr',
        enforcement='source_tolerance_interval',source_tolerance=metadata)


def test_interval_solver_preserves_boundary_value_instead_of_midpoint():
    graph=rectangle(20,10);row=interval_row(nodes=['v0','v1'])
    result=solve_parametric(graph,[row])
    assert result['accepted'],result.get('validation')
    assert result['constraints'][0]['actual']==pytest.approx(20,abs=1e-7)
    assert result['constraints'][0]['enforcement']=='source_tolerance_interval'
    assert result['diagnostics']['tolerance_inequality_count']==1
    assert result['diagnostics']['intervals_counted_as_shape_equalities'] is False
    assert result['diagnostics']['constraint_rank']==0
    assert result['underconstrained']
    assert native_measurement_checks(result['entities'],[row])['passed']


@pytest.mark.parametrize('width,passed',[(20,True),(21.5,True),(23,True),(23.01,False)])
def test_native_interval_checks_bounds_not_nominal_equality(width,passed):
    graph=rectangle(width,10)
    assert native_measurement_checks(graph['entities'],[interval_row(nodes=['v0','v1'])])['passed'] is passed


def test_interval_projected_distance_preserves_source_direction():
    graph=rectangle(20,10)
    assert not native_measurement_checks(graph['entities'],[interval_row(nodes=['v1','v0'])])['passed']


def annular_sector():
    points=[[180,0],[0,180],[0,160],[160,0]]
    entities=[]
    for i in range(4):
        entity=dict(id='g'+str(i),type='ARC' if i%2==0 else 'LINE',start_node='v'+str(i),end_node='v'+str((i+1)%4),start=points[i],end=points[(i+1)%4])
        if i%2==0:entity.update(center=[0.,0.],radius=180. if i==0 else 160.,clockwise=i==2)
        entities.append(entity)
    return dict(units='mm',nodes=[dict(id='v'+str(i),x=p[0],y=p[1],source_px=p) for i,p in enumerate(points)],entities=entities)


def test_exact_radii_and_radial_interval_are_jointly_feasible():
    graph=annular_sector();row=interval_row('radial_gap',entities=['g0','g2'])
    radii=[dict(id='R'+str(i),record_id='rR'+str(i),kind='radius',entities=['g'+str(i)],nodes=[],value=r,source='ocr_api_binding') for i,r in [(0,180.),(2,160.)]]
    result=solve_parametric(graph,[*radii,row])
    assert result['accepted'],result.get('validation')
    assert [r['actual'] for r in result['constraints'][:2]]==[180.,160.]
    assert result['constraints'][2]['actual']==20.
    assert native_measurement_checks(result['entities'],[row])['passed']


def test_radial_difference_alone_does_not_certify_offset_centers():
    graph=annular_sector();graph['entities'][2]['center']=[.5,0.]
    for node in graph['nodes'][2:]:node['x']+=.5
    graph['entities'][1]['end'][0]+=.5
    graph['entities'][2]['start'][0]+=.5
    graph['entities'][2]['end'][0]+=.5
    graph['entities'][3]['start'][0]+=.5
    result=native_measurement_checks(graph['entities'],[interval_row('radial_gap',entities=['g0','g2'])])
    assert not result['passed']


def test_interval_cannot_be_silently_relabelled_as_exact():
    row=interval_row(nodes=['v0','v1']);row['enforcement']='exact'
    with pytest.raises(ValueError,match='inequality enforcement'):solve_parametric(rectangle(20,10),[row])

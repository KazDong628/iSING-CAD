import copy
import pytest
from contour_agent.parametric_solver import solve_parametric, native_axis_relation_checks
from test_parametric_solver import rectangle
from test_exact_source_nominal_solver import native


@pytest.mark.parametrize('kind,entity', [('horizontal','g0'),('vertical','g1')])
def test_source_admitted_axis_is_an_equality_and_survives_native_readback(kind,entity):
    graph=rectangle()
    # Start from a visibly plausible but non-axis fit.
    graph['nodes'][1]['y'] += .015
    graph['nodes'][1]['x'] += .015
    nodes={n['id']:[n['x'],n['y']] for n in graph['nodes']}
    for e in graph['entities']:
        e.update(start=nodes[e['start_node']],end=nodes[e['end_node']])
    before=copy.deepcopy(graph)
    row=dict(id='axis',kind=kind,record_id=None,entities=[entity],nodes=[],value=None,source='source_geometry')
    result=solve_parametric(graph,[row])
    assert result['accepted'],result['validation']
    assert result['constraints'][0]['absolute_residual'] < 1e-7
    assert result['constraints'][0]['enforcement']=='source_nominal_angle_equality'
    assert graph==before
    assert native_axis_relation_checks(result['entities'],[row],dxf_document=native(result['entities']))['passed']
    assert not native_axis_relation_checks(graph['entities'],[row],dxf_document=native(graph['entities']))['passed']


def test_node_pair_axis_is_exact_without_a_line_reference():
    row=dict(id='axis',kind='horizontal',record_id=None,entities=[],nodes=['v0','v2'],value=None,source='source_geometry')
    result=solve_parametric(rectangle(),[row])
    # This diagonal becoming horizontal degenerates the polygon: exactness
    # must not turn an invalid shape into a successful artifact.
    assert not result['accepted']


def test_native_axis_contract_accepts_legacy_relation_without_an_ocr_record():
    graph=rectangle()
    row=dict(id='axis',kind='horizontal',entities=['g0'],nodes=[],source='source_geometry')
    audit=native_axis_relation_checks(graph['entities'],[row],dxf_document=native(graph['entities']))
    assert audit['passed'] and audit['required_count']==1
    assert audit['checks'][0]['record_id'] is None

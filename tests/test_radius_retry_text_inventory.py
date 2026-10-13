from contour_agent.radius_target_provider import radius_retry_selection
from contour_agent.ocr import canonical_records


def test_malformed_radius_like_text_stays_in_coverage_but_not_numeric_retry():
    doc={'records':[{'text':t} for t in ['R44','R45°','R','-R15','R9.5','R0','R-3']]}
    coverage={'unresolved':[{'record_id':f'r{i:03}'} for i in range(7)]}
    selected,audit=radius_retry_selection(doc,coverage,supported_record_ids=['r000'])
    assert selected==['r004']
    assert [r['record_id'] for r in audit['text_review_required']]==['r001','r002','r003','r005','r006']
    assert len(coverage['unresolved'])==7
    assert audit['coverage_denominator_changed'] is False and audit['unknown_text_exempted'] is False


def test_retry_deduplicates_targets_and_requires_current_source_identity():
    doc={'records':[{'text':'R12'},{'text':'R20'}]}
    selected,audit=radius_retry_selection(doc,{'unresolved':[
        {'record_id':'r001'},{'record_id':'r001'},{'record_id':'r999'}]})
    assert selected==['r001'] and audit['text_review_required'][0]['record_id']=='r999'
    assert all(next(r for r in canonical_records(doc) if r['id']==rid)['parsed']['kind']=='radius'
        for rid in selected)


def test_repeated_source_timeouts_do_not_starve_joint_solving_or_exempt_radii():
    doc={'records':[{'text':'R12'},{'text':'R20'},{'text':'R6.4'}]}
    coverage={'unresolved':[{'record_id':f'r{i:03}'} for i in range(3)]}
    selected,audit=radius_retry_selection(doc,coverage,prior_attempts=[
        {'input_record_ids':['r000','r001'],'error_code':'timeout'},
        {'input_record_ids':['r000'],'error_code':'timeout'}])
    assert selected==['r001','r002']
    assert audit['deferred_record_ids']==['r000']
    assert len(coverage['unresolved'])==3
    assert not audit['coverage_denominator_changed'] and not audit['unknown_text_exempted']

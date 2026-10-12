"""A complete larger source inventory preserves competitors and all proof gates."""
from copy import deepcopy
import pytest
import contour_agent.source_arrow_localization as localization
from test_source_arrow_candidate_preservation import two_arrow_source


@pytest.mark.parametrize('complete,expected',[(True,2),(False,0)])
def test_one_bounded_expansion_requires_complete_inventory_and_preserves_real_arrows(monkeypatch,complete,expected):
    gray,row,boundary,proposal=two_arrow_source()
    actual=localization.localize_source_arrow_candidates(gray,row,proposal,boundary,4.)
    assert len(actual)==2
    calls=[]
    def enumerate_pixels(gray,record,proposal,*,limit=24,audit=None):
        calls.append(limit)
        audit.update(status='completed' if limit==64 and complete else 'hypothesis_budget_exhausted',
                     search_complete=limit==64 and complete,maximum_hypotheses=limit,
                     observed_choice_count=48,enumerated_hypothesis_count=2 if limit==64 else 25)
        return [{'proposal':{'choice':i},'localization':{'tip_shift_px':1.}} for i in range(2)] if limit==64 else []
    def verify(gray,record,proposed,*args):
        return deepcopy(actual[proposed['choice']]) if 'choice' in proposed else None
    monkeypatch.setattr(localization,'source_arrow_hypotheses',enumerate_pixels)
    audit={}
    result=localization.localize_source_arrow_candidates(gray,row,proposal,boundary,4.,verifier=verify,audit=audit)
    assert calls==[24,64] and len(result)==expected
    assert audit['bounded_search_expansion']['source_verification_thresholds_changed'] is False
    assert audit['search_complete'] is complete
    if complete:
        assert audit['physical_arrow_candidate_count']==2 and audit['distinct_source_candidates_preserved']
        assert localization.localize_source_arrow_proposal(gray,row,proposal,boundary,4.,verifier=verify) is None


def test_inventory_larger_than_expansion_budget_remains_unknown(monkeypatch):
    gray,row,boundary,proposal=two_arrow_source()
    calls=[]
    def enumerate_pixels(gray,record,proposal,*,limit=24,audit=None):
        calls.append(limit)
        audit.update(status='hypothesis_budget_exhausted',search_complete=False,observed_choice_count=65)
        return []
    monkeypatch.setattr(localization,'source_arrow_hypotheses',enumerate_pixels)
    audit={}
    assert localization.localize_source_arrow_candidates(gray,row,proposal,boundary,4.,audit=audit)==[]
    assert calls==[24] and audit['search_complete'] is False
    assert 'bounded_search_expansion' not in audit

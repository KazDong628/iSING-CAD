"""A missing repeat detection is different from positive ownership conflict."""
import copy
import math

import numpy as np
import pytest

from contour_agent.source_radius_hypotheses import _carried_radius_finite_ownership


def _arc(eid="a", *, center=(0., 0.), quadrant=0):
    first, last = quadrant * math.pi / 2, (quadrant + 1) * math.pi / 2
    return {"id": eid, "type": "ARC", "center": list(center), "radius": 30., "clockwise": False,
            "start": [center[0] + 30 * math.cos(first), center[1] + 30 * math.sin(first)],
            "end": [center[0] + 30 * math.cos(last), center[1] + 30 * math.sin(last)]}


def _leader():
    return {"arrowhead_verified": True,
            "arrowhead": {"tip_px": [30 / math.sqrt(2)] * 2, "direction_px": [1., 0.],
                          "length_px": 20., "verified": True}}


def _check(fresh=None, *, entities=None, leader=None):
    identity = lambda points: np.asarray(points, dtype=float)
    return _carried_radius_finite_ownership(leader or _leader(), "r001", "a",
        {"entities": entities or [_arc()]}, identity, identity, 4., fresh or {})


def test_missing_repeat_detector_does_not_revoke_unique_old_arrow():
    result = _check()
    assert result["passed"]
    assert result["numeric_binding_approved"] is False
    assert result["original_arrow_target_band_px"] == 10.
    assert [r["entity_id"] for r in result["current_target_candidates"]] == ["a"]


@pytest.mark.parametrize("reason", ["shared_source_arrow_ownership_ambiguous",
    "source_arrow_claim_owned_by_other_label", "outside_selected_boundary_or_scope_conflict"])
def test_positive_rejection_of_same_physical_arrow_revokes_carry(reason):
    rejected = {**_leader(), "record_id": "r001", "reason": reason, "selected_record_id": "r002"}
    result = _check({"radius_source_rejections": [rejected]})
    assert not result["passed"]
    assert result["reason"] == "same_original_arrow_has_fresh_source_contradiction"


def test_rejected_different_arrow_hypothesis_for_same_record_is_not_contradiction():
    rejected = {**_leader(), "record_id": "r001", "reason": "source_arrow_claim_owned_by_other_label"}
    rejected["arrowhead"]["tip_px"][0] += 40.
    assert _check({"radius_source_rejections": [rejected]})["passed"]


def test_nearby_different_shaft_direction_is_not_same_physical_arrow():
    rejected = {**_leader(), "record_id": "r001", "reason": "shared_source_arrow_ownership_ambiguous"}
    rejected["arrowhead"]["direction_px"] = [0., 1.]
    assert _check({"radius_source_rejections": [rejected]})["passed"]


def test_new_neighbor_inside_original_arrow_corridor_is_ambiguous():
    result = _check(entities=[_arc(), _arc("b", center=(4., 4.))])
    assert not result["passed"]
    assert result["reason"] == "original_arrow_current_finite_target_not_unique"
    assert {r["entity_id"] for r in result["current_target_candidates"]} == {"a", "b"}


def test_circle_extension_cannot_compete_with_finite_owned_interval():
    result = _check(entities=[_arc(), _arc("b", quadrant=2)])
    assert result["passed"]
    assert [r["entity_id"] for r in result["current_target_candidates"]] == ["a"]


def test_fresh_duplicate_target_ambiguity_not_laundered_as_nondetection():
    observation = {**_leader(), "record_id": "r001", "target_candidates": [
        {"entity_id": "a"}, {"entity_id": "b"}],
        "source_arrow_ownership": {"status": "globally_unique_source_claim",
                                   "duplicate_target_ambiguity_preserved": True}}
    result = _check({"radius_source_observations": [observation]})
    assert not result["passed"]
    assert result["fresh_contradictions"][0]["duplicate_target_ambiguity_preserved"] is True


def test_same_physical_arrow_freshly_owned_by_other_label_revokes_carry():
    observation = {**_leader(), "record_id": "r002", "target_candidates": [{"entity_id": "a"}],
        "source_arrow_ownership": {"status": "globally_unique_source_claim"}}
    result = _check({"radius_source_observations": [observation]})
    assert not result["passed"]
    assert result["fresh_contradictions"][0]["record_id"] == "r002"


def test_positive_conflict_in_fresh_candidate_leader_is_also_checked():
    leader = _leader()
    leader.update(target_candidates=[{"entity_id": "b"}],
                  source_arrow_ownership={"status": "globally_unique_source_claim"})
    assert not _check({"all_candidates": [{"record_id": "r001", "evidence": {"leader": leader}}]})["passed"]


def test_fresh_same_owned_arrow_retained_and_inputs_immutable():
    leader = _leader()
    leader.update(target_candidates=[{"entity_id": "a"}],
                  source_arrow_ownership={"status": "globally_unique_source_claim"})
    fresh = {"radius_source_observations": [{**leader, "record_id": "r001"}]}
    original = copy.deepcopy(fresh)
    assert _check(fresh)["passed"]
    assert fresh == original


def test_missing_or_nonfinite_current_geometry_fails_closed():
    bad = _arc()
    bad["radius"] = float("nan")
    assert not _check(entities=[bad])["passed"]
    bad = _arc()
    bad["id"] = "different"
    assert not _check(entities=[bad])["passed"]


def test_certified_radius_semantics_resolve_adjacent_coarse_line_but_retain_all_hypotheses():
    identity=lambda points:np.asarray(points,dtype=float)
    line={'id':'line','type':'LINE','start':[20.,18.],'end':[20.,26.]}
    graph={'entities':[_arc(),line]}
    interval={'checked':True,'passed':True,'entity_id':'a','nominal':30.}
    fresh={'radius_source_observations':[{**_leader(),'record_id':'r001',
        'target_candidates':[{'entity_id':'a'},{'entity_id':'line'}],
        'source_arrow_ownership':{'status':'globally_unique_source_claim',
                                'duplicate_target_ambiguity_preserved':True}}]}
    old=copy.deepcopy(fresh)
    assert not _check(fresh,entities=graph['entities'])['passed']
    result=_carried_radius_finite_ownership(_leader(),'r001','a',graph,identity,identity,4.,fresh,
        certified_radius=30.,verified_interval=interval)
    assert result['passed'] and len(result['current_target_candidates'])==2
    assert result['coarse_line_hypotheses'][0]['entity_id']=='line' and fresh==old
    graph['entities'].append(_arc('b',center=(4.,4.)))
    assert not _carried_radius_finite_ownership(_leader(),'r001','a',graph,identity,identity,4.,fresh,
        certified_radius=30.,verified_interval=interval)['passed']


@pytest.mark.parametrize('field,value',[('passed',False),('checked',False),('nominal',31.),('entity_id','other')])
def test_type_scoped_radius_ownership_requires_current_verified_interval(field,value):
    identity=lambda points:np.asarray(points,dtype=float)
    interval={'checked':True,'passed':True,'entity_id':'a','nominal':30.};interval[field]=value
    assert not _carried_radius_finite_ownership(_leader(),'r001','a',{'entities':[_arc()]},
        identity,identity,4.,{},certified_radius=30.,verified_interval=interval)['passed']

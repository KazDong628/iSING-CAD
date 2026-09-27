"""Topology arrows require the same directed source path as numeric bindings."""
import numpy as np

from contour_agent import topology_candidates as module


def rectangle_ring():
    corners = np.array([[100.,40.],[180.,40.],[180.,80.],[100.,80.]])
    edges = [np.linspace(a,b,81,endpoint=False) for a,b in zip(corners,np.roll(corners,-1,axis=0))]
    return np.vstack([*edges,corners[:1]])


def inventory(monkeypatch, box, lines, arrow=None):
    monkeypatch.setattr(module,"_leaders",lambda *args:lines)
    monkeypatch.setattr(module,"_arrowhead_evidence",arrow or
                        (lambda gray,endpoint,*args:{"tip_px":endpoint.tolist(),"verified":True}))
    row={"id":"r000","text":"R36","parsed":{"kind":"radius","nominal":36.},"box":box}
    values,_=module._annotation_inventory(np.full((140,300),255,np.uint8),[row],rectangle_ring(),2.)
    return values[0]


def test_topology_distant_arrow_crossing_another_boundary_remains_undirected(monkeypatch):
    item=inventory(monkeypatch,[[10.,50.],[30.,50.],[30.,70.],[10.,70.]],
                   [np.array([[35.,60.],[180.,60.]])])
    leader=item["leader"]
    assert item["leader_status"]=="undirected_leader_candidate"
    assert leader["arrowhead_verified"] is False
    assert leader["arrowhead"] is None
    assert leader["unverified_arrowhead_hypothesis"]["verified"]
    assert "earlier_source_contour_intersection" in leader["directed_verification_issues"]
    assert np.allclose(leader["contour_visibility"]["first_intersection_px"],[100.,60.])
    assert item["leader_hypotheses_requiring_review"]


def test_topology_shaft_beside_label_does_not_establish_directed_target(monkeypatch):
    item=inventory(monkeypatch,[[225.,15.],[250.,15.],[250.,45.],[225.,45.]],
                   [np.array([[225.,60.],[180.,60.]])])
    assert item["leader_status"]=="undirected_leader_candidate"
    assert item["leader"]["arrowhead_verified"] is False
    assert item["leader"]["directed_verification_issues"]==["label_ray_misses_source_box"]


def test_topology_shared_corner_is_a_valid_first_target(monkeypatch):
    item=inventory(monkeypatch,[[230.,30.],[265.,30.],[265.,50.],[230.,50.]],
                   [np.array([[229.,40.],[180.,40.]])])
    leader=item["leader"]
    assert item["leader_status"]=="directed_arrow_candidate"
    assert leader["arrowhead_verified"] is True
    assert leader["contour_visibility"]["verified"] is True
    assert np.allclose(leader["target_source_px"],[180.,40.])
    assert leader["directed_verification_issues"]==[]


def test_topology_arrow_tip_must_stay_in_existing_boundary_band(monkeypatch):
    item=inventory(monkeypatch,[[230.,50.],[265.,50.],[265.,70.],[230.,70.]],
                   [np.array([[229.,60.],[180.,60.]])],
                   arrow=lambda *args:{"tip_px":[160.,60.],"verified":True})
    assert item["leader"]["arrowhead_verified"] is False
    assert "arrow_tip_misses_material_boundary" in item["leader"]["directed_verification_issues"]


def test_topology_verified_arrow_takes_priority_over_closer_undirected_stroke(monkeypatch):
    box=[[230.,50.],[265.,50.],[265.,70.],[230.,70.]]
    item=inventory(monkeypatch,box,[np.array([[230.,72.5],[180.,72.5]]),
                                    np.array([[225.,60.],[180.,60.]])])
    assert item["leader"]["arrowhead_verified"] is True
    assert item["leader"]["leader_id"]=="line001"
    assert item["leader_hypotheses_requiring_review"][0]["leader_id"]=="line000"

"""A supported annotation line is a topology feature, not a fit preference."""

from contour_agent.planning_provider import evaluate_candidates
from contour_agent.topology_candidates import (_angular_line_witnesses,
    _preserved_angular_line_records, _base_source_entities)
from contour_agent.reconstruction_feedback import constraint_regression
import numpy as np


def candidate(name, *, preserved):
    graph={"ground_truth_used":False,
           "validation":{"closed":True,"connected":True,"simple":True,"ordered_entity_cycle":True},
           "angle_line_preservation":{"passed":preserved,"required_record_ids":["r_angle"],
                                      "supported_record_ids":["r_angle"] if preserved else [],
                                      "lost_record_ids":[] if preserved else ["r_angle"]},
           "nodes":[{"id":f"v{i}","source_px":point} for i,point in enumerate(((0,0),(10,0),(10,10)))],
           "entities":[{"id":f"g{i}","type":"LINE","start":list(a),"end":list(b)}
                       for i,(a,b) in enumerate((((0,0),(10,0)),((10,0),(10,10)),((10,10),(0,0))))],
           "relations":[],"annotation_support":[]}
    return {"id":name,"graph":graph,"ground_truth_used":False,
            "annotation_support":{"compatibility_fraction":1.},
            "source_stroke_support":{"edge_supported_fraction":.98},
            "unsupported_primitive_count":0,"binding_candidate_ids":[]}


def test_planner_cannot_select_a_smoother_arc_that_erases_a_verified_line():
    preserved=candidate("retained",preserved=True)
    erased=candidate("erased",preserved=False)
    result=evaluate_candidates([erased,preserved])
    rows={row["candidate_id"]:row for row in result["evaluated"]}
    assert rows["retained"]["admissible"]
    assert not rows["erased"]["admissible"]
    assert "source_annotated_line_support_lost" in rows["erased"]["rejection_reasons"]
    assert "erased" not in result["bounded_candidate_ids"]


def test_missing_independent_angle_evidence_does_not_invent_line_requirement():
    no_angle=candidate("unknown",preserved=True)
    no_angle["graph"]["angle_line_preservation"].update(required_record_ids=[],supported_record_ids=[])
    result=evaluate_candidates([no_angle])
    assert result["evaluated"][0]["admissible"]


def observation(*, target_type="LINE", span=79., shift=0., verified=True,
                second_line_span=None):
    targets=[{"entity_id":"g017", "entity_type":target_type,
              "whole_line_supported":target_type == "LINE", "supported_span_px":span},
             {"entity_id":"g016", "entity_type":"ARC", "whole_line_supported":False,
              "supported_span_px":89.}]
    if second_line_span is not None:
        targets.append({"entity_id":"g018", "entity_type":"LINE",
                        "whole_line_supported":True,"supported_span_px":second_line_span})
    return {"record_id":"r022", "reference_axis":"vertical", "verified":verified,
            "source_line":{"start_px":[2308.+shift,988.],"end_px":[2374.+shift,1225.]},
            "target_candidates":targets,
            "evidence":{"verified":verified,
                        "method":"source_angular_arrows_axis_and_straight_support_v1"}}


def test_ambiguous_arc_neighbor_does_not_cancel_straight_line_preservation():
    # A broad annotation band can contain the adjacent ARC and a fully
    # supported LINE. Numeric binding may remain ambiguous, while this
    # independent finite straight feature still has to survive topology edits.
    required=_angular_line_witnesses([observation()])
    retained=_angular_line_witnesses([observation(shift=2.)])
    erased=_angular_line_witnesses([observation(target_type="ARC")])
    assert required[0]["entity_ids"] == ["g017"]
    assert _preserved_angular_line_records(required, retained, 5.) == {"r022"}
    assert _preserved_angular_line_records(required, erased, 5.) == set()


def test_line_preservation_requires_same_finite_source_stroke_and_substantial_span():
    required=_angular_line_witnesses([observation()])
    displaced=_angular_line_witnesses([observation(shift=35.)])
    tiny=_angular_line_witnesses([observation(span=32.)])
    split=_angular_line_witnesses([observation(span=40., second_line_span=40.)])
    assert _preserved_angular_line_records(required, displaced, 5.) == set()
    assert _preserved_angular_line_records(required, tiny, 5.) == set()
    assert _preserved_angular_line_records(required, split, 5.) == {"r022"}


def test_unverified_angle_arrow_cannot_invent_line_protection():
    assert _angular_line_witnesses([observation(verified=False)]) == []


def test_original_graph_identity_survives_source_candidate_and_local_split():
    base = {"source_sha256": "f" * 64, "entities": [
        {"id": "g000", "type": "LINE", "start": [0., 0.], "end": [10., 0.]},
        {"id": "g001", "type": "LINE", "start": [10., 0.], "end": [10., 10.]},
    ]}
    source = _base_source_entities(base, lambda points: np.asarray(points, float), 1.)
    assert source[0]["stable_id"].startswith("e-")
    assert source[0]["parent_stable_ids"] == [source[0]["stable_id"]]
    assert "graph:g000" in source[0]["ancestor_stable_ids"]
    assert "graph:g001" in source[1]["ancestor_stable_ids"]
    previous = {"structural_constraints": [
        {"kind": "horizontal", "stable_ids": ["graph:g000"]},
        {"kind": "vertical", "stable_ids": ["graph:g001"]},
    ], "bound_record_ids": [], "constraint_count": 2, "solver_accepted": True}
    after = {"structural_constraints": [
        {"kind": "horizontal", "stable_ids": ["split-left"]},
        {"kind": "vertical", "stable_ids": ["unchanged-right"]},
    ], "entity_ancestry": {
        "split-left": [*source[0]["ancestor_stable_ids"], "split-left"],
        "unchanged-right": [*source[1]["ancestor_stable_ids"], "unchanged-right"],
    }, "bound_record_ids": [], "constraint_count": 2, "solver_status": "accepted",
       "solver_accepted": True, "source_validation": {"passed": True}}
    assert constraint_regression(previous, after)["passed"] is True

"""An OCR-directed small radius can repair a rough ARC without GT geometry."""

import copy
import math

import numpy as np
import pytest

from contour_agent import topology_editing as editing
from contour_agent.vectorize import _sample_entities


def _case():
    radius = 25.
    center_offset = math.sqrt(radius * radius - 200.) / math.sqrt(2.)
    old_arc = {"type": "ARC", "start": [100., 20.], "end": [120., 40.],
               "center": [110. - center_offset, 30. + center_offset],
               "radius": radius, "clockwise": False,
               "radius_binding_status": "unresolved_fixed_radius_fit_failed",
               "radius_annotation_evidence": {"record_id": "r001", "nominal": 20.}}
    old = [
        {"type": "LINE", "start": [20., 20.], "end": [100., 20.]},
        old_arc,
        {"type": "LINE", "start": [120., 40.], "end": [120., 110.]},
        {"type": "LINE", "start": [120., 110.], "end": [20., 110.]},
        {"type": "LINE", "start": [20., 110.], "end": [20., 20.]},
    ]
    target = [100. + 20. / math.sqrt(2.), 40. - 20. / math.sqrt(2.)]
    graph = {
        "units": "mm", "source_grid_pitch_px": 1.,
        "nodes": [{"id": f"v{i:03d}", "source_px": entity["start"]}
                  for i, entity in enumerate(old)],
        "entities": [dict(entity, id=f"g{i:03d}", start_node=f"v{i:03d}",
                          end_node=f"v{(i + 1) % len(old):03d}")
                     for i, entity in enumerate(old)],
        "annotation_support": [{
            "record_id": "r001", "kind": "radius", "status": "candidate_supported",
            "candidate_entity_id": "g001", "arrowhead_verified": True,
            "target_gap_px": .25, "adjacent_target_hypotheses": [],
            "source_evidence": {
                "target_source_px": target,
                "shaft_evidence": {"verified": True},
                "arrowhead": {"verified": True},
                "contour_visibility": {"verified": True,
                                       "first_intersection_px": target},
            },
        }],
    }
    inventory = [{"record_id": "r001", "kind": "radius", "nominal": 20.,
                  "text": "R20"}]
    exact = copy.deepcopy(old[:3])
    exact[1] = {"type": "ARC", "start": [100., 20.], "end": [120., 40.],
                "center": [100., 40.], "radius": 20., "clockwise": False}
    source = _sample_entities(exact, max_step_px=.25)[0]
    return graph, inventory, source, target


def _fillet(operations):
    return next((row for row in operations if row["action"] == "insert_annotated_fillet"
                 and row["entity_ids"] == ["g000", "g001", "g002"]), None)


def test_unique_source_claim_proposes_exact_finite_line_arc_line_reinsertion():
    graph, inventory, source, target = _case()
    original_graph, original_inventory = copy.deepcopy(graph), copy.deepcopy(inventory)
    operations = editing.propose_annotation_arc_edits(
        graph, inventory, limit=4, unresolved_radius_record_ids={"r001"})
    operation = _fillet(operations)
    assert operation is not None
    assert "unique_source_targeted_unbound_arc" in operation["evidence_tags"]
    assert editing._complete_fillet_support_scope(graph, inventory, operation) == operation["entity_ids"]
    _, radius_px, binding = editing._radius_edit_evidence(
        graph, operation, inventory, operation["entity_ids"],
        lambda points: np.asarray(points, float))
    assert radius_px == 20.
    assert binding["unique_unbound_arc_source_claim"] is True
    replacement = editing._replacement_chain(
        source, operation["action"], .5, graph["entities"][:3],
        annotated_radius_px=radius_px, radius_binding=binding)
    assert [row["type"] for row in replacement] == ["LINE", "ARC", "LINE"]
    assert replacement[1]["radius"] == 20.
    assert replacement[1]["fillet_construction"] == "existing_line_arc_line_tangent_reinsertion"
    assert replacement[0]["start"] == graph["entities"][0]["start"]
    assert replacement[-1]["end"] == graph["entities"][2]["end"]
    assert replacement[1]["radius_binding"]["record_id"] == "r001"
    assert graph == original_graph
    assert inventory == original_inventory


@pytest.mark.parametrize("defect", [
    "missing_shaft", "missing_first_hit", "weak_arrow", "competing_claim",
    "competing_arc", "other_binding", "constructed", "unknown_visibility",
])
def test_unbound_arc_reinsertion_rejects_unproven_or_conflicted_ownership(defect):
    graph, inventory, _, _ = _case()
    claim = graph["annotation_support"][0]
    arc = graph["entities"][1]
    if defect == "missing_shaft":
        claim["source_evidence"]["shaft_evidence"]["verified"] = False
    elif defect == "missing_first_hit":
        claim["source_evidence"]["contour_visibility"].pop("first_intersection_px")
    elif defect == "weak_arrow":
        claim["arrowhead_verified"] = False
    elif defect == "competing_claim":
        graph["annotation_support"].append(copy.deepcopy(claim))
    elif defect == "competing_arc":
        claim["adjacent_target_hypotheses"] = [{"entity_id": "g003", "entity_type": "ARC"}]
    elif defect == "other_binding":
        arc["radius_binding"] = {"record_id": "r002", "nominal": 10.}
    elif defect == "constructed":
        arc["radius_constructed"] = {"record_id": "r002"}
    else:
        claim["source_evidence"]["contour_visibility"]["verified"] = None
    assert not editing._unique_source_targeted_unbound_arc(graph, inventory, "g001", "r001")
    assert _fillet(editing.propose_annotation_arc_edits(graph, inventory, limit=4)) is None


def test_unbound_arc_kernel_requires_source_claim_and_rejects_conflicting_binding():
    graph, _, source, target = _case()
    prior = copy.deepcopy(graph["entities"][:3])
    base_binding = {"record_id": "r001", "arrowhead_verified": True,
                    "target_source_px": target}
    with pytest.raises(ValueError, match="fillet_unbound_arc_requires_unique_source_claim"):
        editing._replacement_chain(source, "insert_annotated_fillet", .5, prior,
                                   annotated_radius_px=20., radius_binding=base_binding)
    prior[1]["radius_binding"] = {"record_id": "r002", "nominal": 25.}
    with pytest.raises(ValueError, match="fillet_reinsertion_requires_same_bound_radius"):
        editing._replacement_chain(source, "insert_annotated_fillet", .5, prior,
                                   annotated_radius_px=20.,
                                   radius_binding={**base_binding,
                                                   "unique_unbound_arc_source_claim": True})


def test_unbound_arc_reinsertion_needs_two_finite_line_supports():
    graph, inventory, _, _ = _case()
    graph["entities"][2]["end"] = [120., 42.]
    assert _fillet(editing.propose_annotation_arc_edits(graph, inventory, limit=4)) is None

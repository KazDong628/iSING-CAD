import copy
import math

import numpy as np
import pytest

from contour_agent.annotated_arc_resegmentation import resegment_annotated_arcs
from contour_agent.vectorize import _sample_entities


def source_chain(radii=(40.,15.), sweeps=(.55,-1.1)):
    point=np.array([0.,0.]);heading=0.;entities=[];annotations=[]
    for index,(radius,sweep) in enumerate(zip(radii,sweeps)):
        sign=math.copysign(1.,sweep)
        center=point+sign*radius*np.array([-math.sin(heading),math.cos(heading)])
        angle=math.atan2(point[1]-center[1],point[0]-center[0])
        end=center+radius*np.array([math.cos(angle+sweep),math.sin(angle+sweep)])
        target=center+radius*np.array([math.cos(angle+sweep/2),math.sin(angle+sweep/2)])
        entities.append({"type":"ARC","start":point.tolist(),"end":end.tolist(),
                         "center":center.tolist(),"radius":radius,"clockwise":sweep<0})
        annotations.append({"record_id":f"r{index}","radius_px":radius,"nominal":radius,
                            "source_arrow_verified":True,"target_source_px":target.tolist()})
        point=end;heading+=sweep
    samples=_sample_entities(entities,max_step_px=.08)[0]
    return samples,annotations


@pytest.mark.parametrize("radii,sweeps", [((40.,15.),(.55,-1.1)),((40.,15.,25.),(.55,-1.1,.7))])
def test_multiple_exact_radii_repartition_original_source_path(radii,sweeps):
    points,annotations=source_chain(radii,sweeps)
    before=points.copy();original=copy.deepcopy(annotations)
    result=resegment_annotated_arcs(points,annotations,.15,source_entity_count=5)
    assert result["source_endpoints_fixed"] and not result["source_points_moved"]
    assert result["entities"][0]["start"]==points[0].tolist()
    assert result["entities"][-1]["end"]==points[-1].tolist()
    assert [e["radius"] for e in result["entities"]]==list(radii)
    assert result["maximum_bidirectional_error_px"]<=.15
    assert np.array_equal(points,before) and annotations==original
    for first,second in zip(result["entities"],result["entities"][1:]):
        assert first["end"]==second["start"]
        assert any(np.array_equal(point,first["end"]) for point in points)
    assert result["whole_contour_validation_required"]
    assert result["binding_verification_recheck_required"]


def test_source_target_order_controls_partition_not_incoming_record_order():
    points,annotations=source_chain()
    result=resegment_annotated_arcs(points,annotations[::-1],.15,source_entity_count=3)
    assert result["record_ids"]==["r0","r1"]
    assert [e["radius"] for e in result["entities"]]==[40.,15.]


@pytest.mark.parametrize("change",["unverified","outside_target","wrong_radius","duplicate_target","too_many_primitives"])
def test_unverified_or_infeasible_source_partition_is_rejected(change):
    points,annotations=source_chain();count=3
    if change=="unverified":annotations[1]["source_arrow_verified"]=False
    elif change=="outside_target":annotations[1]["target_source_px"]=[999.,999.]
    elif change=="wrong_radius":annotations[0]["radius_px"]=1.
    elif change=="duplicate_target":annotations[1]["target_source_px"]=annotations[0]["target_source_px"]
    else:count=9
    with pytest.raises(ValueError):
        resegment_annotated_arcs(points,annotations,.15,source_entity_count=count)


def test_existing_split_action_enforces_verified_radii_without_free_fit_fallback():
    from contour_agent.topology_editing import _replacement_chain
    points,annotations=source_chain()
    selected=[{"type":"ARC"} for _ in range(3)]
    parts=_replacement_chain(points,"split_chain_at_source_features",.15,selected,radius_targets=annotations)
    assert [row["radius"] for row in parts]==[40.,15.]
    assert all(row["source_partition_evidence"]["radius_enforcement"]=="exact" for row in parts)
    annotations[0]["radius_px"]=1.
    with pytest.raises(ValueError,match="source_does_not_support_exact_annotated_arc_partition"):
        _replacement_chain(points,"split_chain_at_source_features",.15,selected,radius_targets=annotations)


def test_multiple_verified_radii_reserve_a_source_partition_suggestion():
    from contour_agent.topology_editing import propose_annotation_arc_edits
    graph={"units":"mm","entities":[{"id":"g0","type":"LINE"},
            {"id":"g1","type":"ARC","radius":15.},{"id":"g2","type":"LINE"}],
           "annotation_support":[{"record_id":record,"kind":"radius","status":"candidate_supported",
                                  "candidate_entity_id":"g1","arrowhead_verified":True}
                                 for record in ("r0","r1")]}
    inventory=[{"record_id":"r0","kind":"radius","nominal":40.},
               {"record_id":"r1","kind":"radius","nominal":15.}]
    operation=propose_annotation_arc_edits(graph,inventory)[0]
    assert operation["action"]=="split_chain_at_source_features"
    assert operation["entity_ids"]==["g1"]
    assert operation["record_id"] is None


@pytest.mark.parametrize("explicit_unresolved",[None,{"r1"}])
def test_partition_budget_prioritizes_unresolved_junction_over_earlier_constructed_radius(explicit_unresolved):
    from contour_agent.topology_editing import propose_annotation_arc_edits
    # Three source arcs compete for one local partition slot. The first has
    # already been constructed from its annotation. The second label lands at
    # the joint to the third, so selecting only the first pair omits its support.
    radii=[170.,176.,40.]
    graph={"units":"mm","source_grid_pitch_px":1.,"nodes":[],"entities":[],"annotation_support":[]}
    inventory=[]
    for index,radius in enumerate(radii):
        graph["nodes"].append({"id":f"v{index}","source_px":[index*20.,0.]})
        graph["entities"].append({"id":f"g{index}","type":"ARC","radius":radius,
                                  "start_node":f"v{index}","end_node":f"v{index+1}",
                                  "start":[index*20.,0.],"end":[(index+1)*20.,0.],
                                  "center":[index*20.+10.,100.],"clockwise":False})
        target=[40.,0.] if index==1 else [index*20.+10.,0.]
        graph["annotation_support"].append({"record_id":f"r{index}","kind":"radius","status":"candidate_supported",
            "candidate_entity_id":f"g{index}","arrowhead_verified":True,"source_evidence":{"target_source_px":target}})
        inventory.append({"record_id":f"r{index}","kind":"radius","nominal":radius})
    graph["nodes"].append({"id":"v3","source_px":[60.,0.]})
    graph["entities"][0]["radius_binding"]={"record_id":"r0","nominal":170.}
    operations=propose_annotation_arc_edits(graph,inventory,limit=1,unresolved_radius_record_ids=explicit_unresolved)
    assert len(operations)==1 and operations[0]["entity_ids"]==["g1","g2"]
    assert operations[0]["action"]=="split_chain_at_source_features"


def test_continuous_joint_refinement_keeps_source_and_radii_independent():
    from contour_agent.annotated_arc_resegmentation import _refine_two_arc_joint,_landmarks
    points,annotations=source_chain()
    for row in annotations:
        row["source_index"]=int(np.linalg.norm(points-np.asarray(row["target_source_px"]),axis=1).argmin())
    original=points.copy()
    result=_refine_two_arc_joint(points,annotations,.15,_landmarks(points,[row["source_index"] for row in annotations]))
    assert result is not None and result["joint_refinement"]=="bounded_shared_joint_minimax"
    assert result["source_endpoints_fixed"] and not result["source_points_moved"]
    assert np.array_equal(points,original)
    assert [arc["radius"] for arc in result["entities"]]==[40.,15.]
    assert result["entities"][0]["start"]==points[0].tolist()
    assert result["entities"][-1]["end"]==points[-1].tolist()
    assert result["maximum_bidirectional_error_px"]<=.15
    assert result["binding_verification_recheck_required"]


def test_three_annotated_radii_refine_both_joints_without_moving_source():
    from contour_agent.annotated_arc_resegmentation import _refined_partitions,_landmarks
    points,annotations=source_chain((40.,15.,25.),(.55,-1.1,.7))
    original=points.copy()
    for row in annotations:
        row["source_index"]=int(np.linalg.norm(points-np.asarray(row["target_source_px"]),axis=1).argmin())
    choices=_refined_partitions(points,annotations,.15,
                                _landmarks(points,[row["source_index"] for row in annotations]),maximum=3)
    assert choices
    for result in choices:
        assert [arc["radius"] for arc in result["entities"]]==[40.,15.,25.]
        assert len(result["joint_displacements_from_source_vertices_px"])==2
        assert result["entities"][0]["start"]==points[0].tolist()
        assert result["entities"][-1]["end"]==points[-1].tolist()
        assert all(a["end"]==b["start"] for a,b in zip(result["entities"],result["entities"][1:]))
        assert result["maximum_bidirectional_error_px"]<=.15
        assert not result["ground_truth_used"] and not result["source_points_moved"]
    assert np.array_equal(points,original)


def test_original_image_scorer_ranks_certified_partitions_and_cannot_change_radii():
    points,annotations=source_chain()
    original=points.copy();scores=[]
    def scorer(arcs):
        # Stand in for independent source-ink support with a finite preferred
        # station score. A callback receives a copy and cannot alter exact R.
        value=float(arcs[0]["end"][0])
        scores.append(value)
        arcs[0]["radius"]=1.
        return value
    result=resegment_annotated_arcs(points,annotations,.4,source_entity_count=3,candidate_scorer=scorer)
    assert result["source_scored_candidate_count"]==len(scores)>1
    assert result["source_candidate_score"]==max(scores)
    assert result["source_scoring_applied"] and not result["source_score_is_acceptance_certificate"]
    assert [arc["radius"] for arc in result["entities"]]==[40.,15.]
    assert result["maximum_bidirectional_error_px"]<=.4
    assert result["whole_contour_validation_required"] and result["binding_verification_recheck_required"]
    assert np.array_equal(points,original)


def test_nonfinite_source_score_cannot_bypass_geometric_certification():
    points,annotations=source_chain()
    with pytest.raises(ValueError,match="annotated_chain_source_score_must_be_finite"):
        resegment_annotated_arcs(points,annotations,.4,source_entity_count=3,candidate_scorer=lambda arcs:math.nan)

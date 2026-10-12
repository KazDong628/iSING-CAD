"""Independent observations must not masquerade as distinct finite targets."""
import copy

from contour_agent.constraint_binding import _directed_radius_source_counts
from contour_agent.source_radius_hypotheses import radius_hypotheses
from test_source_radius_hypotheses import fixture


def observation(arrow, targets):
    return {"record_id": "r000", "source_arrow_ownership": {
        "physical_arrow_id": arrow, "status": "globally_unique_source_claim"},
        "target_candidates": [{"entity_id": target} for target in targets]}


def test_two_physical_arrows_to_one_object_allow_joint_radius_trial(tmp_path):
    _, _, graph, known, inventory = fixture(tmp_path)
    observations = [observation("arrow-a", ["g001"]), observation("arrow-b", ["g001"])]
    before = copy.deepcopy(observations)
    counts = _directed_radius_source_counts(observations)
    assert counts == {"directed_source_target_count": 1, "directed_source_target_ids": ["g001"],
        "directed_source_observation_count": 2, "directed_source_physical_arrow_count": 2,
        "directed_source_physical_arrow_ids": ["arrow-a", "arrow-b"]}
    inventory["all_candidates"][0]["evidence"].update(counts)
    assert len(radius_hypotheses(inventory, graph, [known])) == 1
    assert observations == before  # No physical-arrow collapse or target deletion.


def test_two_distinct_targets_still_block_trial_even_in_one_observation(tmp_path):
    _, _, graph, known, inventory = fixture(tmp_path)
    for observations in ([observation("a", ["g001"]), observation("b", ["g003"])],
                         [observation("a", ["g001", "g003"])]):
        counts = _directed_radius_source_counts(observations)
        assert counts["directed_source_target_count"] == 2
        inventory["all_candidates"][0]["evidence"].update(counts)
        assert radius_hypotheses(inventory, graph, [known]) == []


def test_unique_target_does_not_override_arrow_ownership_rejection(tmp_path):
    _, _, graph, known, inventory = fixture(tmp_path)
    evidence = inventory["all_candidates"][0]["evidence"]
    evidence.update(_directed_radius_source_counts([
        observation("a", ["g001"]), observation("b", ["g001"])]))
    evidence["leader"]["source_arrow_ownership"]["status"] = "source_arrow_claim_owned_by_other_label"
    assert radius_hypotheses(inventory, graph, [known]) == []
    evidence["leader"]["source_arrow_ownership"]["status"] = "globally_unique_source_claim"
    evidence["leader"]["source_arrow_ownership"]["duplicate_target_ambiguity_preserved"] = True
    assert radius_hypotheses(inventory, graph, [known]) == []


def test_observation_and_physical_identity_counts_are_independent():
    counts = _directed_radius_source_counts([
        observation("a", ["g001"]), observation("a", ["g001"]), observation(None, ["g001"])])
    assert counts["directed_source_observation_count"] == 3
    assert counts["directed_source_physical_arrow_count"] == 1
    assert counts["directed_source_target_count"] == 1
    assert _directed_radius_source_counts([])["directed_source_target_count"] == 0

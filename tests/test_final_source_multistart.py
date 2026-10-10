"""The optional final solve must be source-only and never replace a valid incumbent on regression."""

import json

import pytest

from contour_agent import parametric_pipeline as pipeline


def _solution(name, *, accepted=True, evaluations=10):
    return {"status":"accepted" if accepted else "failed", "accepted":accepted,
            "underconstrained":True,"entities":[{"id":name}],
            "constraints":[{"id":"c0","passed":True}],
            "diagnostics":{"source_budget_search":{"objective_evaluations":evaluations}}}


def _source_check(mask, support, p90=2., *, passed=True):
    return {"passed":passed,"before":{"stroke_supported_fraction":.8,
                "support_measurement_version":"source-stroke-support-v2"},
            "after":{"stroke_supported_fraction":support,"p90_edge_distance_px":p90,
                     "support_measurement_version":"source-stroke-support-v2"},
            "oracle_mask_validation":{"conservative_max_deviation_px":mask}}


def test_final_multistart_selects_only_pareto_improved_source_candidate(monkeypatch,tmp_path):
    observation={"boundary_error_budget":{"sampling_step":.05}}
    monkeypatch.setattr(pipeline,"_solver_source_observation",lambda *args:observation)
    offsets=[{"v0":[.1,0.]},{"v0":[-.1,0.]}]
    monkeypatch.setattr(pipeline,"_source_restart_offsets",lambda *args:(offsets,{"status":"prepared"}))
    solves=iter([_solution("canonical"),_solution("better"),_solution("ink-regressed")])
    budgets=[]
    def solve(*args,**kwargs):
        budgets.append((kwargs["budget_seconds"],kwargs["budget_evaluations"],kwargs.get("seed_node_offsets")))
        return next(solves)
    monkeypatch.setattr(pipeline,"_solve_with_source_observation",solve)
    scores={"canonical":_source_check(3.,.8),"better":_source_check(2.8,.81),
            "ink-regressed":_source_check(2.5,.79)}
    monkeypatch.setattr(pipeline,"_solved_source_validation",
                        lambda *args:scores[args[-1][0]["id"]])
    monkeypatch.setattr(pipeline,"export_parametric",lambda *args,**kwargs:
                        {"validation":{"exact_radius_validation":{"passed":True,"dxf_readback_performed":True}}})
    selected=pipeline._final_source_multistart("source.png",{}, {}, {},[{"id":"c0"}],tmp_path,solver=object())
    assert selected["entities"][0]["id"]=="better"
    receipt=json.loads((tmp_path/"source-multistart.json").read_text(encoding="utf8"))
    assert receipt["selected_seed"]=="source-normal-1" and receipt["ground_truth_used"] is False
    assert receipt["attempts"][2]["reason"]=="source_evidence_not_improved"
    assert len(budgets)==3 and [row[1] for row in budgets]==[30000,3000,3000]
    assert budgets[1][2]==offsets[0] and budgets[2][2]==offsets[1]
    assert budgets[0][0]<=120. and all(row[0]<=15. for row in budgets[1:])


@pytest.mark.parametrize("readback",[False,True])
def test_final_multistart_keeps_canonical_when_alternative_fails_a_gate(monkeypatch,tmp_path,readback):
    monkeypatch.setattr(pipeline,"_solver_source_observation",
                        lambda *args:{"boundary_error_budget":{"sampling_step":.05}})
    monkeypatch.setattr(pipeline,"_source_restart_offsets",
                        lambda *args:([{"v0":[.1,0.]}],{"status":"prepared"}))
    solves=iter([_solution("canonical"),_solution("alternative")])
    monkeypatch.setattr(pipeline,"_solve_with_source_observation",lambda *args,**kwargs:next(solves))
    checks={"canonical":_source_check(3.,.8),
            "alternative":_source_check(2.,.9,passed=readback)}
    monkeypatch.setattr(pipeline,"_solved_source_validation",
                        lambda *args:checks[args[-1][0]["id"]])
    monkeypatch.setattr(pipeline,"export_parametric",lambda *args,**kwargs:
                        {"validation":{"exact_radius_validation":{"passed":False,"dxf_readback_performed":True}}})
    selected=pipeline._final_source_multistart("source.png",{}, {}, {},[{"id":"c0"}],tmp_path,solver=object())
    assert selected["entities"][0]["id"]=="canonical"
    receipt=json.loads((tmp_path/"source-multistart.json").read_text(encoding="utf8"))
    assert receipt["selected_seed"]=="canonical"
    assert receipt["attempts"][1]["source_gate_passed"] is readback
    assert receipt["attempts"][1]["dxf_readback_passed"] is False

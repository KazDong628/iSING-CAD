"""Real cut generation, strict equations and unchanged complete-curve audits."""
from copy import deepcopy

import numpy as np
import pytest

from contour_agent import parametric_solver as solver
from test_feasibility_first import inputs


def polygon(points):
    count=len(points)
    return {"units":"mm", "nodes":[{"id":f"v{i}","x":p[0],"y":p[1],"source_px":p}
                                   for i,p in enumerate(points)],
            "entities":[{"id":f"g{i}","type":"LINE","start_node":f"v{i}",
                         "end_node":f"v{(i+1)%count}","start":p,"end":points[(i+1)%count]}
                        for i,p in enumerate(points)]}


def observation(points, maximum=.08, step=.02):
    return {"units":"mm","points":points+[points[0]],"provenance":"input_mask_boundary",
            "reference_dxf_read":False,"boundary_error_budget":{"units":"mm",
                "maximum_deviation":maximum,"sampling_step":step,
                "source":"initial_curve_fit_total_deviation_budget_px"}}


def width_constraint(width):
    return {"id":"width","kind":"distance_x","nodes":["v0","v1"],"entities":[],
            "value":width,"record_id":"r0","source":"ocr_local_binding","required":True,
            "enforcement":"exact","nominal_source":"source_ocr"}


def test_seventh_round_repairs_distinct_regions_despite_maximum_error_plateau(monkeypatch, tmp_path):
    # A deliberately small separation-oracle batch reveals one of seven peaks
    # at a time. Geometry/optimization/audits are real; no passed result or
    # source deviation is substituted. A fixed five-round loop cannot finish.
    monkeypatch.setattr(solver._SourceBoundaryBudget,"witnesses_per_direction",1)
    width=3.5
    points=[[0.,0.],[width,0.],[width,1.]]+[
        [i*.25,1.+(.2 if i%2 else 0.)] for i in range(13,-1,-1)]
    graph=polygon([[x,min(y,1.)] for x,y in points]);before=deepcopy(graph)
    raw=observation(points);unchanged=deepcopy(raw)
    actual_minimize=solver.minimize;clock=[0.]
    monkeypatch.setattr(solver.time,"monotonic",lambda:clock[0])
    def expire_optional_rms(fun,*args,**kwargs):
        if fun.__name__=="scalar_objective":clock[0]=6.
        return actual_minimize(fun,*args,**kwargs)
    monkeypatch.setattr(solver,"minimize",expire_optional_rms)
    result=solver.solve_parametric(graph,[width_constraint(width)],source_observation=raw,output_dir=tmp_path)
    search=result["diagnostics"]["source_budget_search"]
    errors=[row["feasibility_audit"]["conservative_max_deviation"] for row in search["rounds"]]
    assert result["accepted"] and len(errors)>=6
    assert min(errors[:5])>raw["boundary_error_budget"]["maximum_deviation"]
    assert errors[0]==pytest.approx(errors[1],abs=1e-7)
    assert errors[1]==pytest.approx(errors[2],abs=1e-7)
    assert all(row["witness_update"]["new_witness_count"]>0 for row in search["rounds"])
    assert search["final_audit"]["passed"] and search["feasible_checkpoint_found"]
    assert search["stop_reason"]=="source_feasible_checkpoint_retained_after_optional_RMS_budget"
    assert search["rounds"][-1]["RMS_subbudget_exhausted"] and not search["exhausted"]
    assert not search["RMS_refinement_accepted"] and not result["diagnostics"]["optimality_proven"]
    assert result["strict_measurement_contract"]["satisfied"]
    assert result["constraints"][0]["absolute_residual"]<=1e-7
    assert (tmp_path/"source-budget-checkpoint.json").is_file()
    assert raw==unchanged and graph==before and not search["acceptance_thresholds_changed"]


def test_objective_budget_expiration_without_a_feasible_checkpoint_cannot_pass():
    graph,constraints,raw=inputs()
    raw["boundary_error_budget"]["max_objective_evaluations"]=1
    result=solver.solve_parametric(graph,constraints,source_observation=raw)
    search=result["diagnostics"]["source_budget_search"]
    assert not result["accepted"] and result["entities"]==graph["entities"]
    assert search["stop_reason"]=="objective_evaluation_budget_exhausted"
    assert search["objective_evaluations"]==1 and not search["feasible_checkpoint_found"]
    assert not search["infeasibility_proven"]


def test_duplicate_violations_stop_repeated_projection_without_certifying_failure(monkeypatch):
    points=[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]
    graph=polygon(points);raw=observation([[x+.5,y] for x,y in points],.2,.1)
    budgets=[];original_init=solver._SourceBoundaryBudget.__init__
    def capture(self,*args,**kwargs):
        original_init(self,*args,**kwargs);budgets.append(self)
    monkeypatch.setattr(solver._SourceBoundaryBudget,"__init__",capture)
    def stalled(fun,x0,**kwargs):
        fun(x0)
        return solver.OptimizeResult(x=np.asarray(x0).copy(),jac=np.zeros_like(x0),
            success=False,status=8,nfev=1,nit=1,message="synthetic stalled projection")
    monkeypatch.setattr(solver,"minimize",stalled)
    result=solver.solve_parametric(graph,[width_constraint(1.)],source_observation=raw)
    search=result["diagnostics"]["source_budget_search"];budget=budgets[0]
    assert not result["accepted"] and result["entities"]==graph["entities"]
    assert search["stop_reason"]=="no_new_violating_witnesses"
    assert 0<len(search["rounds"])<search["maximum_rounds"]
    assert search["stop_audit"]["witness_update"]["new_witness_count"]==0
    assert search["existing_witness_minimum_margin"]<0
    assert len({tuple(row) for row in budget.forward})==len(budget.forward)
    assert len({(index,round(fraction,8)) for index,fraction in budget.reverse})==len(budget.reverse)
    assert not search["infeasibility_proven"] and not search["exhausted"]


@pytest.mark.parametrize("after_feasibility",[117.,121.])
def test_publication_reserve_and_late_candidate_are_separate_from_feasibility(monkeypatch,after_feasibility):
    graph,constraints,raw=inputs();actual_minimize=solver.minimize;clock=[0.];calls=[]
    monkeypatch.setattr(solver.time,"monotonic",lambda:clock[0])
    def finish_near_deadline(fun,*args,**kwargs):
        calls.append(fun.__name__)
        fitted=actual_minimize(fun,*args,**kwargs)
        if fun.__name__=="feasibility_objective":clock[0]=after_feasibility
        return fitted
    monkeypatch.setattr(solver,"minimize",finish_near_deadline)
    result=solver.solve_parametric(graph,constraints,source_observation=raw)
    search=result["diagnostics"]["source_budget_search"]
    assert calls==["feasibility_objective"]
    if after_feasibility<120.:
        assert result["accepted"] and search["feasible_checkpoint_found"]
        assert search["RMS_refinement_skipped_reason"]=="shared_budget_reserved_for_publication"
        assert not search["exhausted"]
    else:
        assert not result["accepted"] and not search["feasible_checkpoint_found"]
        assert search["stop_reason"]=="wall_time_budget_exhausted"
        assert result["entities"]==graph["entities"]

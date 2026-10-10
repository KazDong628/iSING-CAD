"""The editor must start from the graph that passed the publication source gate."""
import copy
from pathlib import Path

from contour_agent.parametric_pipeline import _plan_topology


def _graph(name, passed):
    return {"candidate_id":name,"source_gate_passed":passed,"entities":[{"type":"LINE"} for _ in range(3)]}


def _setup(monkeypatch,tmp_path,*,with_valid_alternative=True,base_valid=True,wrapper_matches_base=True):
    base=_graph("preserved-base",base_valid)
    wrapped=copy.deepcopy(base) if wrapper_matches_base else _graph("regenerated-wrapper",False)
    bad=_graph("bad-planner-choice",False)
    good=_graph("source-supported-alternative",True)
    candidates=[{"id":"base","graph":wrapped},{"id":"bad","graph":bad}]
    if with_valid_alternative:candidates.append({"id":"good","graph":good})
    for candidate in candidates:
        overlay=tmp_path/f"{candidate['id']}-candidate.png"
        overlay.write_bytes(candidate["id"].encode())
        candidate.update(overlay_path=str(overlay),entity_counts={"total":3,"LINE":3,"ARC":0})
    bundle={"candidates":candidates,"annotation_inventory":[{"leader":{"method":"source-evidence"}}]}
    scores={"bad":1.,"good":.99,"base":.98}
    local={"recommended_candidate_id":"bad","admissible_candidate_ids":[c["id"] for c in candidates],
           "evaluated":[{"candidate_id":c["id"],"admissible":True,"score":scores[c["id"]],
                         "metrics":{"entity_count":3,"unsupported_primitive_count":0}} for c in candidates]}
    monkeypatch.setattr("contour_agent.topology_candidates.generate_topology_candidates",lambda *a,**k:copy.deepcopy(bundle))
    monkeypatch.setattr("contour_agent.planning_provider.evaluate_candidates",lambda *a,**k:copy.deepcopy(local))
    def validate(image,baseline,graph):
        passed=graph["source_gate_passed"]
        return {"passed":passed,"reasons":[] if passed else ["source_stroke_support_degraded"]}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation",validate)
    monkeypatch.setattr("contour_agent.topology_candidates.materialize_selected_candidate",
                        lambda selected,*a,**k:{"status":"materialized","candidate_id":selected["id"]})
    (tmp_path/"topology-overlay.png").write_bytes(b"original-base-overlay")
    class Planner:
        def select(self,*args):
            return {"status":"succeeded","schema_success":True,"selected_candidate_id":"bad","network_requests":1}
    return base,Planner()


def test_source_invalid_planner_choice_is_replaced_before_editor_receives_graph(monkeypatch,tmp_path):
    base,planner=_setup(monkeypatch,tmp_path)
    observed=[]
    def edit(image,document,baseline,selected,*args,**kwargs):
        observed.append(copy.deepcopy(selected))
        return selected,{"accepted_round_count":0}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_loop",edit)
    graph,plan=_plan_topology(tmp_path/"source.png",{}, {},base,tmp_path,planner_provider=planner,use_api=True)
    assert len(observed)==1 and observed[0]["id"]=="good"
    assert observed[0]["graph"]["source_gate_passed"] is True
    assert graph==observed[0]["graph"]
    assert plan["source_validation_before_edit"]["status"]=="fallback_selected"
    assert plan["source_validation_before_edit"]["original_selected_candidate_id"]=="bad"
    assert plan["provider"]["selected_candidate_id"]=="bad"  # Keep the online receipt truthful.


def test_no_source_valid_candidate_skips_editor_and_retains_failed_receipts(monkeypatch,tmp_path):
    base,planner=_setup(monkeypatch,tmp_path,with_valid_alternative=False,base_valid=False)
    calls=[]
    def edit(*args,**kwargs):
        calls.append(True)
        raise AssertionError("No source-valid graph may enter the edit/solve feedback loop")
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_loop",edit)
    graph,plan=_plan_topology(tmp_path/"source.png",{}, {},base,tmp_path,planner_provider=planner,use_api=True)
    assert not calls and not graph["source_gate_passed"]
    assert plan["source_validation_before_edit"]["status"]=="no_source_valid_topology"
    assert plan["source_validation_selection"]["status"]=="no_source_valid_topology"
    assert all(not row["validation"]["passed"] for row in plan["source_validation_before_edit"]["attempts"])


def test_preserved_base_wrapper_uses_exact_validated_graph_and_matching_overlay(monkeypatch,tmp_path):
    base,planner=_setup(monkeypatch,tmp_path,with_valid_alternative=False,wrapper_matches_base=False)
    observed=[]
    def forbidden_materialization(*args,**kwargs):
        raise AssertionError("The preserved graph differs from its generated bundle member; preserve it directly")
    monkeypatch.setattr("contour_agent.topology_candidates.materialize_selected_candidate",forbidden_materialization)
    def edit(image,document,baseline,selected,*args,**kwargs):
        observed.append((copy.deepcopy(selected),Path(selected["overlay_path"]).read_bytes()))
        return selected,{"accepted_round_count":0}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_loop",edit)
    graph,plan=_plan_topology(tmp_path/"source.png",{}, {},base,tmp_path,planner_provider=planner,use_api=True)
    assert len(observed)==1 and observed[0][0]["id"]=="base"
    assert observed[0][0]["graph"]==base
    assert observed[0][1]==b"original-base-overlay"
    assert graph==base
    assert plan["source_validation_before_edit"]["selected_graph_source"]=="preserved_base_topology"


def test_source_validation_is_still_rechecked_after_an_editor_changes_the_graph(monkeypatch,tmp_path):
    base,planner=_setup(monkeypatch,tmp_path)
    def edit(image,document,baseline,selected,*args,**kwargs):
        changed=copy.deepcopy(selected)
        changed.update(id="edited-invalid",graph=_graph("invalid-edit",False))
        return changed,{"accepted_round_count":1}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_loop",edit)
    graph,plan=_plan_topology(tmp_path/"source.png",{}, {},base,tmp_path,planner_provider=planner,use_api=True)
    assert graph["source_gate_passed"]
    assert plan["source_validation_selection"]["status"]=="fallback_selected"
    assert plan["source_validation_selection"]["attempts"][0]["candidate_id"]=="edited-invalid"


def test_seed_promotion_is_reported_separately_from_a_local_edit(monkeypatch,tmp_path):
    base,planner=_setup(monkeypatch,tmp_path)
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation",
                        lambda image,baseline,graph:{"passed":True,"reasons":[]})
    def seed_loop(image,document,baseline,selected,bundle,output,**kwargs):
        assert selected["id"]=="bad"
        assert kwargs["initial_seed_ids"]==()
        return next(row for row in bundle["candidates"] if row["id"]=="good"),{
            "final_selection_origin":"initial_seed_promotion","accepted_round_count":1,
            "accepted_seed_promotion_count":1,"accepted_local_edit_count":0}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_loop",seed_loop)
    graph,plan=_plan_topology(tmp_path/"source.png",{}, {},base,tmp_path,
                              planner_provider=planner,use_api=True)
    assert graph["candidate_id"]=="source-supported-alternative"
    assert plan["selection_source"]=="bounded_source_seed_promotion"
    assert plan["materialization"]["candidate_id"]=="good"

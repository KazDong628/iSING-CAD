"""Source-only geometry fidelity checks; no dataset or reference files."""
import json
import math

import numpy as np
from PIL import Image
import pytest

from contour_agent.automatic import build_automatic
from contour_agent.vectorize import _line, _merge_primitive_runs, assess_fit_quality, fit_polyline, fit_polyline_with_diagnostics


RECTANGLE = [[10.,10.],[110.,10.],[110.,80.],[10.,80.],[10.,10.]]


def test_finite_line_rejects_collinear_overshoot_hidden_by_infinite_line():
    points=np.array([[0.,0.],[2.,0.],[1.4,0.],[1.2,0.],[1.,0.]])
    assert _line(points,.7) is None
    line=_line(np.array([[0.,0.],[.5,.2],[1.,0.]]),.3)
    assert line["fit_error_px"]==pytest.approx(.2)


def test_rectangle_fidelity_is_measured_without_claiming_dimensions():
    result=fit_polyline_with_diagnostics(RECTANGLE,tolerance_px=1.)
    q=result["quality"]
    assert q["passed"] and not q["fallback_used"]
    assert len(result["entities"])==4
    assert q["source_boundary_deviation_px"]["sampled_symmetric_hausdorff_px"]<1e-9
    assert q["source_boundary_deviation_px"]["conservative_upper_bound_px"]<=q["total_deviation_budget_px"]
    assert q["area_iou"]==pytest.approx(1.)
    assert not q["dimensions_verified"] and not q["reference_verified"] and not q["engineering_certified"]


@pytest.mark.parametrize("clockwise",[False,True])
def test_arc_fidelity_checks_reverse_direction_and_shared_endpoints(clockwise):
    sign=-1 if clockwise else 1
    angles=np.linspace(0,sign*math.pi,181)
    arc_points=np.c_[np.cos(angles)*40,np.sin(angles)*40]
    source=np.vstack([arc_points,arc_points[0]])
    entities=[{"type":"ARC","start":arc_points[0].tolist(),"end":arc_points[-1].tolist(),
               "center":[0.,0.],"radius":40.,"clockwise":clockwise},
              {"type":"LINE","start":arc_points[-1].tolist(),"end":arc_points[0].tolist()}]
    quality=assess_fit_quality(source,entities,max_step_px=.1)
    assert quality["sampled_topology_valid"]
    assert quality["source_boundary_deviation_px"]["conservative_upper_bound_px"]<.11
    assert quality["area_iou"]>.999


def test_overshooting_fit_falls_back_to_original_contour_without_moving_vertices(monkeypatch):
    bad=[{"type":"LINE","start":a,"end":b} for a,b in zip([[10.,10.],[180.,10.],[180.,80.],[10.,80.]],
                                                             [[180.,10.],[180.,80.],[10.,80.],[10.,10.]])]
    monkeypatch.setattr("contour_agent.vectorize.fit_polyline",lambda *a,**kw:bad)
    result=fit_polyline_with_diagnostics(RECTANGLE,tolerance_px=1.)
    assert result["quality"]["fallback_used"]
    assert result["quality"]["fallback_reason"]=="source_boundary_deviation_exceeds_pixel_budget"
    assert result["quality"]["attempted_fit"]["source_boundary_deviation_px"]["conservative_lower_bound_px"]>60
    assert [e["start"] for e in result["entities"]]+[result["entities"][-1]["end"]]==RECTANGLE
    assert result["quality"]["source_boundary_deviation_px"]["conservative_upper_bound_px"]==0.


def test_invalid_fitted_topology_falls_back_but_invalid_source_is_not_repaired(monkeypatch):
    crossing=[[10.,10.],[110.,80.],[110.,10.],[10.,80.],[10.,10.]]
    monkeypatch.setattr("contour_agent.vectorize.fit_polyline",lambda *a,**kw:[{"type":"LINE","start":a,"end":b} for a,b in zip(crossing[:-1],crossing[1:])])
    result=fit_polyline_with_diagnostics(RECTANGLE,tolerance_px=1.)
    assert result["quality"]["fallback_reason"]=="fitted_chain_or_sampled_topology_invalid"
    with pytest.raises(ValueError,match="valid nondegenerate"):
        fit_polyline_with_diagnostics(crossing,tolerance_px=1.)


def test_raw_mask_simplification_error_is_disclosed_separately():
    raw=[[10.,10.],[60.,9.5],[110.,10.],[110.,80.],[10.,80.],[10.,10.]]
    result=fit_polyline_with_diagnostics(RECTANGLE,source_polyline_px=raw,tolerance_px=1.)
    q=result["quality"]
    assert .5<=q["pre_fit_simplification_upper_bound_px"]<1.
    assert q["total_deviation_budget_px"]==pytest.approx(1.+q["pre_fit_simplification_upper_bound_px"])
    assert q["raw_source_vertex_count"]==5 and q["fit_input_vertex_count"]==4


def source_extraction(tmp_path,monkeypatch):
    path=tmp_path/"source.png"
    Image.new("RGB",(160,120),"white").save(path)
    extraction={"polyline_px":RECTANGLE,"raw_polyline_px":RECTANGLE,"image_size":{"width":160,"height":120},"issues":[]}
    monkeypatch.setattr("contour_agent.automatic.extract_main_profile",lambda *a,**kw:extraction)
    return path


def test_contour_evidence_survives_downstream_measurement_failure(tmp_path,monkeypatch):
    path=source_extraction(tmp_path,monkeypatch)
    def fail(*args,**kwargs):raise ValueError("synthetic measurement failure")
    monkeypatch.setattr("contour_agent.automatic.estimate_scale",fail)
    with pytest.raises(ValueError,match="measurement failure"):
        build_automatic(path,{},tmp_path/"output")
    assert (tmp_path/"output/contour-overlay.png").is_file()
    assert not (tmp_path/"output/drawing.dxf").exists()


def test_curve_evidence_is_persisted_and_identical_in_model_validation(tmp_path,monkeypatch):
    path=source_extraction(tmp_path,monkeypatch)
    monkeypatch.setattr("contour_agent.automatic.estimate_scale",lambda *a,**kw:{"status":"unresolved","axis":None,"pixels_per_mm":None,"issues":[]})
    result=build_automatic(path,{"records":[]},tmp_path/"output")
    evidence=json.loads((tmp_path/"output/curve-fit.json").read_text(encoding="utf8"))
    assert evidence==result["curve_fit"]==result["validation"]["curve_fit"]
    assert result["coordinate_system"]["units"]=="pixel"
    assert result["validation"]["passed"] and not result["validation"]["dimensions_verified"]
    assert (tmp_path/"output/contour-overlay.png").is_file()


def test_vectorization_error_has_failed_receipt_and_preserved_contour(tmp_path,monkeypatch):
    path=source_extraction(tmp_path,monkeypatch)
    monkeypatch.setattr("contour_agent.automatic.estimate_scale",lambda *a,**kw:{"status":"unresolved","axis":None,"pixels_per_mm":None,"issues":[]})
    def fail(*args,**kwargs):raise ValueError("synthetic fit failure")
    monkeypatch.setattr("contour_agent.automatic.fit_polyline_with_diagnostics",fail)
    with pytest.raises(ValueError,match="fit failure"):
        build_automatic(path,{"records":[]},tmp_path/"output")
    evidence=json.loads((tmp_path/"output/curve-fit.json").read_text(encoding="utf8"))
    assert not evidence["passed"] and not evidence["engineering_certified"]
    assert (tmp_path/"output/contour-overlay.png").is_file()


@pytest.mark.parametrize("samples",[101,701])
def test_dense_half_circle_merges_across_old_vertex_limit(samples):
    theta=np.linspace(0,math.pi,samples)
    ring=np.vstack([np.c_[100*np.cos(theta),100*np.sin(theta)],[[100.,0.]]])
    result=fit_polyline_with_diagnostics(ring,tolerance_px=.5)
    assert sorted(e["type"] for e in result["entities"])==["ARC","LINE"]
    arc=next(e for e in result["entities"] if e["type"]=="ARC")
    assert arc["radius"]==pytest.approx(100.,abs=1e-8)
    assert arc["center"]==pytest.approx([0.,0.],abs=1e-8)
    assert arc["source_support_vertex_count"]==samples
    q=result["quality"]
    assert not q["fallback_used"] and not q["optimization_rollback_used"]
    assert q["source_boundary_deviation_px"]["conservative_upper_bound_px"]<=.5
    assert q["sampled_topology_valid"]


def test_candidate_budget_bounds_pair_construction_and_retains_seed_path():
    pts=np.c_[np.arange(1201,dtype=float),np.zeros(1201)]
    runs=[(i,i+1,_line(pts[i:i+2],.1)) for i in range(1200)]
    merged,evidence=_merge_primitive_runs(pts,runs,.1,candidate_budget=17)
    assert evidence["candidate_evaluations"]==17
    assert evidence["candidate_budget_exhausted"]
    assert len(merged)<=len(runs)
    assert merged[0][0]==0 and merged[-1][1]==1200
    assert all(a[1]==b[0] for a,b in zip(merged[:-1],merged[1:]))


def test_rejected_compact_fit_retains_seed_only_after_same_fidelity_gate(monkeypatch):
    valid=[{"type":"LINE","start":a,"end":b} for a,b in zip(RECTANGLE[:-1],RECTANGLE[1:])]
    bad=[dict(e) for e in valid]
    bad[0]["end"]=[180.,10.];bad[1]["start"]=[180.,10.]
    bad[0]["fitting_optimization"]={"method":"source-supported-primitive-merge-dp-v2","seed_entity_count":4,"merged_entity_count":4}
    calls=[]
    def candidate(*args,**kwargs):
        calls.append(kwargs.get("optimize",True))
        return bad if kwargs.get("optimize",True) else valid
    monkeypatch.setattr("contour_agent.vectorize.fit_polyline",candidate)
    result=fit_polyline_with_diagnostics(RECTANGLE,tolerance_px=1.)
    q=result["quality"]
    assert calls==[True,False]
    assert not q["fallback_used"] and q["optimization_rollback_used"]
    assert q["total_deviation_budget_px"]==1.
    assert q["source_boundary_deviation_px"]["conservative_upper_bound_px"]<=1.
    assert q["rejected_optimization"]["reason"]=="source_boundary_deviation_exceeds_pixel_budget"
    assert not q["optimization_summary"]["candidate_accepted"]


def test_ambiguous_repeated_radius_annotation_reverts_both_snaps(monkeypatch):
    # Exercise binding arbitration without depending on a particular curved
    # seed partition. All synthetic arcs still originate in this source ring.
    theta=np.linspace(0,2*math.pi,301)
    ring=np.c_[50*np.cos(theta),50*np.sin(theta)]
    def binding(choice,*args):
        if choice["type"]=="ARC":
            choice["radius"]=999.
            choice["radius_binding"]={"record_id":"one-label","nominal":50.}
        return choice
    monkeypatch.setattr("contour_agent.vectorize._bind_radius",binding)
    entities=fit_polyline(ring,tolerance_px=.5,optimize=False)
    arcs=[e for e in entities if e["type"]=="ARC"]
    assert len(arcs)>1
    assert all("radius_binding" not in e and e["radius"]==pytest.approx(50.) for e in arcs)
    ambiguous=entities[0]["fitting_optimization"]["ambiguous_radius_bindings"]
    assert len(ambiguous)==1 and len(ambiguous[0]["entity_ids"])==len(arcs)

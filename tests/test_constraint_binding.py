from copy import deepcopy
import json

from PIL import Image, ImageDraw
import pytest

from contour_agent.constraint_binding import analyze_constraint_bindings, build_binding_candidates


def source(tmp_path):
    path=tmp_path/"source.png"
    image=Image.new("RGB",(500,360),"white")
    draw=ImageDraw.Draw(image)
    draw.rectangle((100,100,400,300),outline="black",width=2)
    draw.line((100,50,400,50),fill="black",width=2)
    draw.line((100,40,100,110),fill="black",width=2)
    draw.line((400,40,400,110),fill="black",width=2)
    image.save(path)
    document={"records":[{"text":"150","box":[[225,25],[260,25],[260,43],[225,43]]}]}
    coords=[(0,100),(150,100),(150,0),(0,0)]
    pixels=[(100,100),(400,100),(400,300),(100,300)]
    nodes=[{"id":f"v{i:03d}","x":x,"y":y,"source_px":list(p)} for i,((x,y),p) in enumerate(zip(coords,pixels))]
    entities=[{"id":f"g{i:03d}","type":"LINE","start_node":nodes[i]["id"],"end_node":nodes[(i+1)%4]["id"],
               "start":list(coords[i]),"end":list(coords[(i+1)%4])} for i in range(4)]
    graph={"units":"mm","nodes":nodes,"entities":entities,"relations":[],"proposal_tolerance_px":5,
           "coordinate_system":{"units":"mm","origin_source_px":[100,300]}}
    model={"scale":{"pixels_per_mm":2},"coordinate_system":graph["coordinate_system"]}
    return path,document,model,graph


def observed_coarse_radius_source(path, model, radius=90., center=(90., 180.)):
    """R45 source arc remains near the fixture's coarse R40 proposal."""
    import numpy as np
    theta = np.linspace(-np.pi/2, np.pi/2, 361)
    points = np.c_[center[0]+radius*np.cos(theta), center[1]+radius*np.sin(theta)]
    model["extraction"] = {"raw_polyline_px": points.tolist()}
    image = Image.open(path)
    ImageDraw.Draw(image).line([tuple(point) for point in points], fill="black", width=2)
    image.save(path)


def test_source_dimension_line_binding_and_persisted_topology(tmp_path):
    path,document,model,graph=source(tmp_path)
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"out")
    assert result["provider"]["network_requests"]==0
    assert result["counts"]["local_accepted"]==1
    constraint=result["constraints"][0]
    assert constraint["kind"]=="distance_x" and constraint["value"]==150
    assert constraint["source"]=="ocr_local_binding"
    nodes={n["id"]:n for n in graph["nodes"]}
    assert nodes[constraint["nodes"][1]]["x"]>nodes[constraint["nodes"][0]]["x"]
    inventory=json.loads((tmp_path/"out"/"binding-candidates.json").read_text(encoding="utf-8"))
    candidate=inventory["all_candidates"][0]
    assert candidate["evidence"]["extension_lines"] and not candidate["evidence"]["arrowhead_verified"]
    assert (tmp_path/"out"/"binding-topology.png").is_file()
    assert result["dimensions_verified"] is False and result["ground_truth_used"] is False


def test_unresolved_units_retain_candidate_but_do_not_apply_mm(tmp_path):
    path,document,model,graph=source(tmp_path)
    graph["units"]="pixel"
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"out")
    assert result["constraints"]==[]
    assert result["counts"]["all_candidates"]>0 and "units_unresolved" in result["issues"]


def test_grossly_incompatible_ocr_span_is_not_a_hard_constraint(tmp_path):
    path,document,model,graph=source(tmp_path)
    document["records"][0]["text"]="1"
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"out")
    assert result["counts"]["all_candidates"]>0 and result["constraints"]==[]
    assert "ambiguous_or_insufficient_independent_source_evidence" in result["issues"]


def stub_inventory():
    record={"id":"r000","text":"R40","parsed":{"kind":"radius","nominal":40.},"box":[[10,10],[20,20]]}
    candidate={"id":"c000","record_id":"r000","kind":"radius","entities":["g000"],"nodes":[],"value":40.,
               "local_reliable":True,"evidence":{"leader":{"segment_px":[[10,10],[30,30]],"arrowhead_verified":True}}}
    return {"units":"mm","records":[record],"all_records":[record],"candidates":[candidate],"all_candidates":[candidate],
            "relations":[],"counts":{"ocr_records":1,"recognized_dimensions":1,"all_candidates":1},
            "artifacts":{"inventory":"candidates.json","topology":"topology.png"}}


def analyze_stub(tmp_path,monkeypatch,inventory,bindings,relations=None,graph=None,transcribe=True,sent_ids=None):
    monkeypatch.setattr("contour_agent.constraint_binding.build_binding_candidates",lambda *a,**k:deepcopy(inventory))
    texts={row["id"]:row["text"] for row in inventory["all_records"]}
    if transcribe:
        bindings=[{**row,"observed_text":row.get("observed_text",texts.get(row["record_id"],"unknown"))} for row in bindings]
    class Provider:
        def select(self,*args):
            return {"status":"succeeded","network_requests":1,"http_success":True,"schema_success":True,
                    "bindings":bindings,"relations":relations or [],
                    **(sent_ids if sent_ids is not None else {
                        "input_record_ids":[row["id"] for row in inventory["records"]],
                        "input_candidate_ids":[row["id"] for row in inventory["candidates"]],
                        "input_relation_ids":[row["id"] for row in inventory["relations"]]})}
    graph=graph or {"nodes":[],"entities":[{"id":"g000","type":"ARC"}]}
    return analyze_constraint_bindings("unused",{}, {},graph,tmp_path,provider=Provider(),use_api=True)


def test_api_selection_is_separate_from_local_and_value_comes_from_ocr(tmp_path,monkeypatch):
    result=analyze_stub(tmp_path,monkeypatch,stub_inventory(),[{"record_id":"r000","candidate_id":"c000"}])
    assert result["provider"]["http_success"] and result["counts"]["api_selected"]==1
    assert result["counts"]["api_accepted"]==1 and result["counts"]["local_accepted"]==0
    assert result["constraints"][0]["value"]==40 and result["constraints"][0]["source"]=="ocr_api_binding"


@pytest.mark.parametrize("mutate,selection,reason",[
    (lambda inv:None,{"record_id":"r000","candidate_id":"missing"},"unknown_candidate_id"),
    (lambda inv:None,{"record_id":"unknown","candidate_id":"c000"},"unknown_record_id"),
    (lambda inv:inv["all_candidates"][0].update(record_id="r001"),{"record_id":"r000","candidate_id":"c000"},"record_candidate_mismatch"),
    (lambda inv:inv["all_records"][0]["parsed"].update(kind="length"),{"record_id":"r000","candidate_id":"c000"},"record_type_mismatch"),
    (lambda inv:inv["all_candidates"][0].update(value=41),{"record_id":"r000","candidate_id":"c000"},"source_nominal_mismatch"),
    (lambda inv:inv["all_candidates"][0].update(local_reliable=False),{"record_id":"r000","candidate_id":"c000"},"ambiguous_or_insufficient_independent_source_evidence"),
    (lambda inv:inv.update(units="pixel"),{"record_id":"r000","candidate_id":"c000"},"units_unresolved"),
])
def test_invalid_semantic_selections_are_not_applied(tmp_path,monkeypatch,mutate,selection,reason):
    inventory=stub_inventory();mutate(inventory)
    result=analyze_stub(tmp_path,monkeypatch,inventory,[selection])
    assert result["counts"]["api_accepted"]==0
    assert reason in result["issues"]
    assert result["provider"]["schema_success"]


def test_duplicate_record_does_not_choose_first(tmp_path,monkeypatch):
    selected={"record_id":"r000","candidate_id":"c000"}
    result=analyze_stub(tmp_path,monkeypatch,stub_inventory(),[selected,selected])
    assert result["constraints"]==[] and result["counts"]["api_selected"]==2
    assert result["issues"]==["duplicate_record_selection"]


def test_conflicting_different_source_records_reject_both(tmp_path,monkeypatch):
    inventory=stub_inventory()
    other_record=deepcopy(inventory["records"][0]);other_record.update(id="r001",text="R45");other_record["parsed"]["nominal"]=45.
    other_candidate=deepcopy(inventory["candidates"][0]);other_candidate.update(id="c001",record_id="r001",value=45.)
    inventory["records"].append(other_record);inventory["all_records"].append(other_record)
    inventory["candidates"].append(other_candidate);inventory["all_candidates"].append(other_candidate)
    inventory["counts"]["recognized_dimensions"]=2
    result=analyze_stub(tmp_path,monkeypatch,inventory,[{"record_id":"r000","candidate_id":"c000"},{"record_id":"r001","candidate_id":"c001"}])
    assert result["constraints"]==[] and result["counts"]["api_accepted"]==0
    assert result["issues"]==["conflicting_constraints"]


def test_provider_failure_preserves_local_candidate_and_constraint(tmp_path):
    path,document,model,graph=source(tmp_path)
    class Failed:
        def select(self,*args):
            return {"status":"failed","network_requests":1,"http_success":False,"schema_success":False}
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"out",provider=Failed(),use_api=True)
    assert result["counts"]["local_accepted"]==1 and result["counts"]["api_accepted"]==0
    assert result["provider"]["network_requests"]==1


def test_provider_cancellation_does_not_trigger_local_constraint_fallback(tmp_path):
    path,document,model,graph=source(tmp_path)
    class Cancelled:
        def select(self,*args):raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        analyze_constraint_bindings(path,document,model,graph,tmp_path/"out",provider=Cancelled(),use_api=True)
    assert (tmp_path/"out"/"binding-candidates.json").is_file()
    assert not (tmp_path/"out"/"constraint-bindings.json").exists()


def test_radius_proximity_alone_is_not_independent_binding(tmp_path):
    path,document,model,graph=source(tmp_path)
    document={"records":[{"text":"R40","box":[[200,120],[230,120],[230,140],[200,140]]}]}
    graph["entities"]=[{"id":"g000","type":"ARC","start":[0,100],"end":[0,20],"center":[0,60],"radius":40,"clockwise":True}]
    result=build_binding_candidates(path,document,model,graph,tmp_path/"out")
    assert result["all_candidates"]
    assert all(not c["local_reliable"] for c in result["all_candidates"])


def test_y_dimension_uses_cad_upward_node_order(tmp_path):
    path,document,model,graph=source(tmp_path)
    image=Image.open(path)
    draw=ImageDraw.Draw(image)
    draw.line((50,100,50,300),fill="black",width=2)
    draw.line((40,100,110,100),fill="black",width=2)
    draw.line((40,300,110,300),fill="black",width=2)
    image.save(path)
    document={"records":[{"text":"100","box":[[25,185],[43,185],[43,220],[25,220]]}]}
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"out")
    constraint=result["constraints"][0]
    assert constraint["kind"]=="distance_y" and constraint["value"]==100
    nodes={n["id"]:n for n in graph["nodes"]}
    assert nodes[constraint["nodes"][1]]["y"]>nodes[constraint["nodes"][0]]["y"]


def test_line_leader_can_bind_coarse_arc_without_old_nominal_fit_gate(tmp_path,monkeypatch):
    import numpy as np
    path,document,model,graph=source(tmp_path)
    observed_coarse_radius_source(path, model)
    document={"records":[{"text":"R45","box":[[240,180],[265,180],[265,200],[240,200]]}]}
    graph["entities"]=[{"id":"g000","type":"ARC","start":[0,100],"end":[0,20],"center":[0,60],"radius":40,"clockwise":True}]
    image=Image.open(path);draw=ImageDraw.Draw(image)
    draw.line((239,180,180,180),fill="black",width=2)
    draw.polygon([(180,180),(195,175),(195,185)],fill="black")
    image.save(path)
    # Source radial leader ends on the coarse arc at image x180,y180; the
    # annotation requests R45 rather than the fitted R40. Do not reject a valid
    # source binding merely because it must later move the coarse geometry.
    monkeypatch.setattr("contour_agent.constraint_binding._leaders",lambda *args:[np.array([[239.,180.],[180.,180.]])])
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"out")
    assert result["constraints"][0]["value"]==45
    assert result["constraints"][0]["entities"]==["g000"]


def test_competing_radius_leader_targets_cannot_be_disambiguated_by_api_alone(tmp_path,monkeypatch):
    import numpy as np
    path,document,model,graph=source(tmp_path)
    document={"records":[{"text":"R45","box":[[240,180],[265,180],[265,200],[240,200]]}]}
    arc={"id":"g000","type":"ARC","start":[0,100],"end":[0,20],"center":[0,60],"radius":40,"clockwise":True}
    graph["entities"]=[arc,{**arc,"id":"g001"}]
    image=Image.open(path);draw=ImageDraw.Draw(image)
    draw.line((239,180,180,180),fill="black",width=2)
    draw.polygon([(180,180),(195,175),(195,185)],fill="black")
    image.save(path)
    monkeypatch.setattr("contour_agent.constraint_binding._leaders",lambda *args:[np.array([[239.,180.],[180.,180.]])])
    class Selector:
        def select(self,image,topology,inventory):
            return {"schema_success":True,"http_success":True,"network_requests":1,
                    "bindings":[{"record_id":"r000","candidate_id":inventory["candidates"][0]["id"]}],"relations":[]}
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"out",provider=Selector(),use_api=True)
    assert result["constraints"]==[] and result["counts"]["api_selected"]==1 and result["counts"]["api_accepted"]==0
    assert "ambiguous_or_insufficient_independent_source_evidence" in result["issues"]


def test_contradictory_relation_selectors_reject_all_members(tmp_path,monkeypatch):
    inventory=stub_inventory()
    inventory["all_candidates"]=[];inventory["candidates"]=[]
    inventory["relations"]=[{"id":"rel000","type":"horizontal","entities":["g000"],"nodes":[]},
                            {"id":"rel001","type":"vertical","entities":["g000"],"nodes":[]}]
    result=analyze_stub(tmp_path,monkeypatch,inventory,[],[{"relation_id":"rel000"},{"relation_id":"rel001"}],
                        graph={"nodes":[],"entities":[{"id":"g000","type":"LINE"}]})
    assert result["constraints"]==[] and result["issues"]==["conflicting_relations"]


def test_successful_api_abstention_is_not_silently_replaced_by_local_binding(tmp_path,monkeypatch):
    result=analyze_stub(tmp_path,monkeypatch,stub_inventory(),[])
    assert result['constraints']==[] and result['counts']['local_accepted']==0
    assert 'provider_abstained_from_sent_record' in result['issues']


def test_provider_budget_omitted_record_is_not_a_model_abstention(tmp_path,monkeypatch):
    result=analyze_stub(tmp_path,monkeypatch,stub_inventory(),[],sent_ids={
        "input_record_ids":[],"input_candidate_ids":[],"input_relation_ids":[]})
    assert result["counts"]["local_accepted"]==1
    assert "provider_abstained_from_sent_record" not in result["issues"]
    assert result["provider"]["input_inventory_verified"] is True


def test_api_cannot_select_candidate_outside_its_actual_sent_packet(tmp_path,monkeypatch):
    result=analyze_stub(tmp_path,monkeypatch,stub_inventory(),[
        {"record_id":"r000","candidate_id":"c000"}],sent_ids={
        "input_record_ids":["r000"],"input_candidate_ids":[],"input_relation_ids":[]})
    assert result["counts"]["api_accepted"]==0
    assert "candidate_not_sent" in result["issues"]


def test_legacy_receipt_without_input_ids_does_not_claim_api_admission_or_abstention(tmp_path,monkeypatch):
    result=analyze_stub(tmp_path,monkeypatch,stub_inventory(),[
        {"record_id":"r000","candidate_id":"c000"}],sent_ids={})
    assert result["provider"]["input_inventory_verified"] is False
    assert result["counts"]["api_accepted"]==0
    assert result["counts"]["local_accepted"]==1
    assert result["constraints"][0]["source"]=="ocr_local_binding"
    assert "provider_abstained_from_sent_record" not in result["issues"]


def test_radius_local_reliability_cannot_override_missing_arrow_evidence(tmp_path,monkeypatch):
    inventory=stub_inventory();inventory['all_candidates'][0]['evidence']['leader']['arrowhead_verified']=False
    result=analyze_stub(tmp_path,monkeypatch,inventory,[{'record_id':'r000','candidate_id':'c000'}])
    assert result['constraints']==[] and 'source_arrowhead_not_verified' in result['issues']


def test_radius_rejects_long_vertical_edge_beside_label_even_if_radial():
    import numpy as np
    from contour_agent.constraint_binding import _leader_evidence
    # A nearby label box does not connect to the parallel long shape edge.
    box=np.array([[40.,10.],[80.,10.],[80.,50.],[40.,50.]])
    arc=np.array([[100.,249.],[100.,250.],[101.,251.]])
    line=np.array([[100.,30.],[100.,250.]])
    assert _leader_evidence(box,arc,np.array([100.,200.]),[line],5.,np.full((300,160),255,np.uint8)) is None


def test_radius_rejects_horizontal_dimension_line_below_label():
    import numpy as np
    from contour_agent.constraint_binding import _leader_evidence
    box=np.array([[30.,10.],[70.,10.],[70.,30.],[30.,30.]])
    arc=np.array([[180.,50.],[181.,50.]])
    line=np.array([[75.,50.],[180.,50.]])
    assert _leader_evidence(box,arc,np.array([120.,50.]),[line],5.,np.full((100,220),255,np.uint8)) is None


def test_radial_hatch_line_without_arrow_and_wrong_arrow_polarity_are_rejected():
    import cv2
    import numpy as np
    from contour_agent.constraint_binding import _leader_evidence
    box=np.array([[230.,45.],[265.,45.],[265.,75.],[230.,75.]])
    arc=np.c_[np.full(9,100.),np.linspace(58,62,9)]
    center=np.array([60.,60.]);line=np.array([[229.,60.],[100.,60.]])
    gray=np.full((120,290),255,np.uint8)
    cv2.line(gray,(100,60),(229,60),0,2)
    assert _leader_evidence(box,arc,center,[line],5.,gray) is None
    # The arrow is on the label end, pointing away from the proposed arc.
    cv2.fillConvexPoly(gray,np.array([[229,60],[210,54],[210,66]],np.int32),0)
    assert _leader_evidence(box,arc,center,[line],5.,gray) is None


def test_crossing_two_strokes_is_not_a_filled_arrowhead():
    import cv2
    import numpy as np
    from contour_agent.constraint_binding import _arrowhead_evidence
    gray=np.full((120,240),255,np.uint8)
    cv2.line(gray,(30,60),(210,60),0,2)
    cv2.line(gray,(80,30),(140,90),0,2)
    assert _arrowhead_evidence(gray,np.array([100.,60.]),np.array([-1.,0.]),50.,5.) is None


def test_radius_leader_cannot_bind_beyond_an_earlier_material_boundary(monkeypatch):
    import numpy as np
    import contour_agent.constraint_binding as module
    # Isolate visibility from arrow detection: even a locally verified tip must
    # not skip a material boundary between its label and the nominated arc.
    monkeypatch.setattr(module, "_arrowhead_evidence", lambda gray, endpoint, *args:
                        {"tip_px": endpoint.tolist(), "verified": True})
    box=np.array([[10.,50.],[30.,50.],[30.,70.],[10.,70.]])
    arc=np.array([[180.,55.],[180.,65.]])
    shaft=np.array([[35.,60.],[180.,60.]])
    near_boundary=np.array([[100.,40.],[100.,80.]])
    rejected=[]
    assert module._leader_evidence(box,arc,np.array([200.,60.]),[shaft],5.,
                                   contours=[arc,near_boundary],rejections=rejected) is None
    assert len(rejected)==1
    assert rejected[0]["reason"]=="earlier_source_contour_intersection"
    assert np.allclose(rejected[0]["first_intersection_px"],[100.,60.])
    assert rejected[0]["first_intersection_to_target_px"]==80.
    assert rejected[0]["verified"] is False


def test_radius_leader_preserves_neighboring_primitives_at_shared_target(monkeypatch):
    import numpy as np
    import contour_agent.constraint_binding as module
    monkeypatch.setattr(module, "_arrowhead_evidence", lambda gray, endpoint, *args:
                        {"tip_px": endpoint.tolist(), "verified": True})
    box=np.array([[10.,50.],[30.,50.],[30.,70.],[10.,70.]])
    arc=np.array([[180.,60.],[180.,65.]])
    shaft=np.array([[35.,60.],[180.,60.]])
    # A neighbor sharing the arrow's endpoint is not an occluding boundary.
    neighbor=np.array([[174.,55.],[180.,60.]])
    result=module._leader_evidence(box,arc,np.array([200.,60.]),[shaft],5.,
                                   contours=[arc,neighbor])
    assert result is not None
    assert result["contour_visibility"]["verified"] is True
    assert np.allclose(result["contour_visibility"]["first_intersection_px"],[180.,60.])


def test_radius_visibility_tolerates_only_the_existing_target_band():
    import numpy as np
    from contour_agent.constraint_binding import _leader_contour_visibility
    start=np.array([30.,60.]);end=np.array([180.,60.])
    # Coarse source geometry near the same junction must not change the band.
    near=np.array([[171.,55.],[171.,65.]])
    far=np.array([[169.,55.],[169.,65.]])
    assert _leader_contour_visibility(start,end,[near],5.)["verified"] is True
    result=_leader_contour_visibility(start,end,[far],5.)
    assert result["target_support_band_px"]==10.
    assert result["verified"] is False


def test_radius_visibility_handles_collinear_boundary_and_multiple_crossings():
    import numpy as np
    from contour_agent.constraint_binding import _leader_contour_visibility
    contours=[np.array([[160.,50.],[160.,70.]]),
              np.array([[80.,60.],[110.,60.]])]
    result=_leader_contour_visibility([30.,60.],[180.,60.],contours,5.)
    assert np.allclose(result["first_intersection_px"],[80.,60.])
    assert result["verified"] is False


def test_genuine_directed_arrow_remains_bound_with_contour_visibility():
    import cv2
    import numpy as np
    from contour_agent.constraint_binding import _leader_evidence, _arrowhead_evidence
    gray=np.full((140,280),255,np.uint8)
    cv2.line(gray,(100,60),(229,60),0,2)
    cv2.fillConvexPoly(gray,np.array([[100,60],[125,54],[125,66]],np.int32),0)
    arc=np.array([[100.,58.],[100.,60.],[100.,62.]])
    box=np.array([[230.,45.],[265.,45.],[265.,75.],[230.,75.]])
    result=_leader_evidence(box,arc,np.array([60.,60.]),[np.array([[229.,60.],[100.,60.]])],
                            5.,gray,contours=[arc])
    assert result is not None and result["arrowhead_verified"]
    assert result["contour_visibility"]["verified"]
    assert result["arrow_tip_to_arc_gap_px"]<=10.
    # Ranking arrow hypotheses cannot accept a taper away from the target band.
    assert _arrowhead_evidence(gray,np.array([100.,60.]),np.array([-1.,0.]),50.,5.,arc+[80.,0.]) is None


def test_extension_uses_measured_ink_width_without_relaxing_band():
    from contour_agent.constraint_binding import _extension_support
    line={"cross":100.5,"lo":20.,"hi":220.,"thickness":8,"span":201.}
    found=_extension_support([line],105.,25.,210.,4.)
    assert found is not None and found["intersection_gap_px"]==.5
    assert found["ink_station_interval_px"]==[96.5,104.5]
    assert _extension_support([line],109.,25.,210.,4.) is None
    assert _extension_support([line],105.,25.,230.,4.) is None


def test_nodes_without_extension_do_not_create_station_ambiguity(tmp_path):
    path,document,model,graph=source(tmp_path)
    graph["nodes"].append({"id":"noise","x":149.7,"y":-20.,"source_px":[399.4,340.]})
    result=build_binding_candidates(path,document,model,graph,tmp_path/"out")
    candidates=[c for c in result["all_candidates"] if c["kind"]=="distance_x"]
    assert len(candidates)==1 and candidates[0]["local_reliable"]
    groups=candidates[0]["evidence"]["observed_station_groups"]
    assert all("noise" not in group["member_nodes"] for group in groups)


def test_observed_station_groups_remove_noisy_corner_not_true_parallel_station():
    from contour_agent.constraint_binding import _observed_station_groups
    nodes=[{"id":"a","x":0.,"y":0.,"source_px":[98.,40.]},
           {"id":"b","x":0.,"y":100.,"source_px":[102.,140.]},
           {"id":"corner","x":8.,"y":110.,"source_px":[110.,150.]}]
    line={"cross":100.,"lo":20.,"hi":200.,"thickness":8,"span":181.}
    graph={"nodes":nodes,"entities":[{"id":"edge","type":"LINE","start_node":"a","end_node":"b","start":[0.,0.],"end":[0.,100.]}],"units":"pixel"}
    options=[(abs(n["source_px"][0]-100),n["source_px"][1]-20,n,line) for n in nodes]
    groups=_observed_station_groups(options,0,6.,graph)
    assert len(groups)==1
    assert set(groups[0][1]["member_nodes"])=={"a","b"}
    assert "corner" in groups[0][1]["excluded_coarse_neighbours"]
    other={**line,"cross":104.}
    parallel=_observed_station_groups([options[0],(2.,20.,nodes[1],other)],0,6.,graph)
    assert len(parallel)==2


def test_single_coarse_curve_projection_is_not_reliable_station():
    from contour_agent.constraint_binding import _observed_station_groups
    node={"id":"curve","x":0.,"y":0.,"source_px":[107.,60.]}
    line={"cross":100.,"lo":20.,"hi":100.,"thickness":2,"span":81.}
    groups=_observed_station_groups([(7.,40.,node,line)],0,8.,{"nodes":[node],"entities":[]})
    assert groups[0][1]["support_kind"]=="unsupported_projection"


@pytest.mark.parametrize("observed,reason",[(None,"source_observed_text_missing"),("Δ1","source_observed_text_mismatch"),("R41","source_observed_text_mismatch"),("40","source_observed_text_mismatch")])
def test_api_binding_requires_independent_matching_source_transcription(tmp_path,monkeypatch,observed,reason):
    selected={"record_id":"r000","candidate_id":"c000"}
    if observed is not None:selected["observed_text"]=observed
    result=analyze_stub(tmp_path,monkeypatch,stub_inventory(),[selected],transcribe=False)
    assert result["constraints"]==[] and reason in result["issues"]


def _synthetic_first_glyph(symbol, trailing=True):
    import cv2
    import numpy as np
    image=np.full((84,95),255,np.uint8)
    if symbol=="delta":
        cv2.polylines(image,[np.array([[28,9],[9,60],[48,60]],np.int32)],True,0,4)
    else:
        cv2.polylines(image,[np.array([[39,9],[9,43],[49,43]],np.int32)],False,0,4)
        cv2.line(image,(39,9),(39,64),0,4)
    if trailing:cv2.line(image,(69,9),(69,64),0,5)
    return image


def test_source_triangle_misread_as_leading_four_is_flagged_without_changing_ocr():
    from contour_agent.constraint_binding import _source_text_evidence
    from contour_agent.ocr import parse_dimension
    row={"text":"41","parsed":parse_dimension("41"),"box":[[0,0],[94,0],[94,83],[0,83]]}
    result=_source_text_evidence(_synthetic_first_glyph("delta"),row)
    assert result["symbol_confusion"] and result["requires_source_text_confirmation"]
    assert row["parsed"]["nominal"]==41 and result["text_confirmed"] is False


@pytest.mark.parametrize("text",["4","41","47"])
def test_true_four_descender_is_not_flagged_as_delta(text):
    from contour_agent.constraint_binding import _source_text_evidence
    from contour_agent.ocr import parse_dimension
    row={"text":text,"parsed":parse_dimension(text),"box":[[0,0],[94,0],[94,83],[0,83]]}
    result=_source_text_evidence(_synthetic_first_glyph("four",len(text)>1),row)
    assert result["checked"] and not result["symbol_confusion"]


def test_symbol_confusion_prevents_api_echo_from_overriding_source_pixels(tmp_path,monkeypatch):
    inventory=stub_inventory()
    inventory["all_candidates"][0]["evidence"]["source_text"]={"symbol_confusion":True}
    result=analyze_stub(tmp_path,monkeypatch,inventory,[{"record_id":"r000","candidate_id":"c000"}])
    assert result["constraints"]==[] and "source_symbol_confusion_requires_confirmation" in result["issues"]


def test_provider_failure_retains_only_source_verified_structural_constraints(tmp_path):
    path,document,model,graph=source(tmp_path)
    graph["relations"]=[{"id":"rel000","type":"horizontal","entities":["g000"]},
                        {"id":"rel001","type":"vertical","entities":["g001"]}]
    class Failed:
        def select(self,*args):return {"status":"failed","schema_success":False,"http_success":False,"network_requests":1}
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"out",provider=Failed(),use_api=True)
    assert result["counts"]["structural_local_accepted"]==2
    assert result["counts"]["local_accepted"]==1
    assert {r["kind"] for r in result["constraints"]}=={"distance_x","horizontal","vertical"}
    assert not result["provider"]["http_success"] and not result["dimensions_verified"]
    inventory=json.loads((tmp_path/"out"/"binding-candidates.json").read_text(encoding="utf-8"))
    assert all(r["evidence"]["verified"] for r in inventory["relations"])


def test_axis_geometry_without_source_stroke_is_never_enough_even_with_api(tmp_path):
    path,document,model,graph=source(tmp_path)
    Image.new("RGB",(500,360),"white").save(path)
    graph["relations"]=[{"id":"rel000","type":"horizontal","entities":["g000"]}]
    class Selector:
        def select(self,*args):
            return {"schema_success":True,"http_success":True,"bindings":[],"relations":[{"relation_id":"rel000"}],
                    "input_record_ids":[],"input_candidate_ids":[],"input_relation_ids":["rel000"]}
    result=analyze_constraint_bindings(path,{},model,graph,tmp_path/"out",provider=Selector(),use_api=True)
    assert result["constraints"]==[]
    assert "insufficient_independent_source_relation_evidence" in result["issues"]


def test_parallel_dimension_stroke_keeps_axis_fallback_ambiguous(tmp_path):
    path,_,model,graph=source(tmp_path)
    image=Image.open(path);ImageDraw.Draw(image).line((100,95,400,95),fill="black",width=1);image.save(path)
    graph["relations"]=[{"id":"rel000","type":"horizontal","entities":["g000"]}]
    result=build_binding_candidates(path,{},model,graph,tmp_path/"out")
    relation=result["relations"][0]
    assert not relation["local_reliable"]
    assert relation["evidence"]["reason"]=="multiple_parallel_source_strokes"


def test_loose_ocr_box_does_not_delete_observed_long_boundary(tmp_path):
    path,_,model,graph=source(tmp_path)
    graph["relations"]=[{"id":"rel000","type":"horizontal","entities":["g000"]}]
    document={"records":[{"text":"surface","box":[[210,90],[260,90],[260,115],[210,115]]}]}
    partial=build_binding_candidates(path,document,model,graph,tmp_path/"partial")
    assert partial["relations"][0]["local_reliable"]
    document["records"][0]["box"]=[[90,90],[410,90],[410,115],[90,115]]
    covered=build_binding_candidates(path,document,model,graph,tmp_path/"covered")
    assert not covered["relations"][0]["local_reliable"]


def test_successful_provider_relation_abstention_is_preserved(tmp_path):
    path,_,model,graph=source(tmp_path)
    graph["relations"]=[{"id":"rel000","type":"horizontal","entities":["g000"]}]
    class Abstained:
        def select(self,*args):return {"schema_success":True,"http_success":True,"bindings":[],"relations":[],
                                      "input_record_ids":[],"input_candidate_ids":[],"input_relation_ids":["rel000"]}
    result=analyze_constraint_bindings(path,{},model,graph,tmp_path/"out",provider=Abstained(),use_api=True)
    assert result["constraints"]==[] and "provider_abstained_from_sent_relation" in result["issues"]


def _tangent_source(tmp_path, kink=False):
    import cv2
    import numpy as np
    path=tmp_path/"tangent.png"
    gray=np.full((260,260),255,np.uint8)
    cv2.line(gray,(30,150),(110,150),0,2)
    theta=np.linspace(np.pi/2,0,181)
    arc=np.c_[110+65*np.cos(theta),85+65*np.sin(theta)]
    if kink:
        angle=np.deg2rad(15)
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        arc=(arc-[110,150])@rotation.T+[110,150]
    cv2.polylines(gray,[np.rint(arc).astype(np.int32)],False,0,2)
    Image.fromarray(gray).save(path)
    points=[[30.,150.],[110.,150.],[175.,85.]]
    design=[[x,260-y] for x,y in points]
    graph={"units":"mm","proposal_tolerance_px":4.,
           "nodes":[{"id":f"v{i:03d}","x":q[0],"y":q[1],"source_px":p} for i,(p,q) in enumerate(zip(points,design))],
           "entities":[{"id":"g000","type":"LINE","start_node":"v000","end_node":"v001","start":design[0],"end":design[1]},
                       {"id":"g001","type":"ARC","start_node":"v001","end_node":"v002","start":design[1],"end":design[2],
                        "center":[110.,175.],"radius":65.,"clockwise":False}],
           "relations":[{"id":"rel000","type":"tangent","entities":["g000","g001"]}]}
    return path,graph


def test_tangency_requires_two_sided_source_ink_measurements(tmp_path):
    path,graph=_tangent_source(tmp_path)
    result=analyze_constraint_bindings(path,{}, {},graph,tmp_path/"out")
    assert result["counts"]["structural_local_accepted"]==1
    relation=next(row for row in result["bindings"] if row.get("relation_id"))
    assert relation["evidence"]["observed_deviation_degrees"]<=3
    assert all(side["verified"] for side in relation["evidence"]["sides"])


def test_geometrically_tangent_proposal_does_not_override_visible_corner(tmp_path):
    path,graph=_tangent_source(tmp_path,kink=True)
    result=analyze_constraint_bindings(path,{}, {},graph,tmp_path/"out")
    assert result["constraints"]==[]
    assert result["counts"]["structural_local_accepted"]==0


def test_topology_leader_is_rechecked_against_current_entity_ids_and_pixels(tmp_path,monkeypatch):
    path,_,model,graph=source(tmp_path)
    observed_coarse_radius_source(path, model)
    document={"records":[{"text":"R45","box":[[240,180],[265,180],[265,200],[240,200]]}]}
    graph["entities"]=[{"id":"g009","type":"ARC","start":[0,100],"end":[0,20],"center":[0,60],"radius":40,"clockwise":True}]
    graph["annotation_support"]=[{"record_id":"r000","candidate_entity_id":"g000","arrowhead_verified":True,
                                  "source_evidence":{"segment_px":[[239.,180.],[180.,180.]],"arrowhead_verified":True}}]
    monkeypatch.setattr("contour_agent.constraint_binding._leaders",lambda *args:[])
    # An old graph's assertion is insufficient when the source arrow is absent.
    before=analyze_constraint_bindings(path,document,model,graph,tmp_path/"before")
    assert before["constraints"]==[]
    image=Image.open(path);draw=ImageDraw.Draw(image)
    draw.line((239,180,180,180),fill="black",width=2)
    draw.polygon([(180,180),(195,175),(195,185)],fill="black");image.save(path)
    after=analyze_constraint_bindings(path,document,model,graph,tmp_path/"after")
    assert after["constraints"][0]["entities"]==["g009"]
    assert after["constraints"][0]["value"]==45


def test_carried_radius_segment_is_rechecked_without_old_target_or_verdict(tmp_path, monkeypatch):
    path, _, model, graph = source(tmp_path)
    observed_coarse_radius_source(path, model)
    document = {"records": [{"text": "R45", "box": [[240,180],[265,180],[265,200],[240,200]]}]}
    graph["entities"] = [{"id": "g009", "type": "ARC", "start": [0,100], "end": [0,20],
                          "center": [0,60], "radius": 40, "clockwise": True}]
    graph["annotation_support"] = []
    graph["radius_source_segment_hypotheses"] = [{
        "record_id": "r000", "kind": "radius", "candidate_entity_id": "obsolete-g000",
        "arrowhead_verified": True,
        "source_evidence": {"segment_px": [[239.,180.],[180.,180.]],
                            "arrowhead_verified": True}}]
    monkeypatch.setattr("contour_agent.constraint_binding._leaders", lambda *args: [])
    # Old positive metadata and coordinates alone cannot establish a binding.
    absent = analyze_constraint_bindings(path, document, model, graph, tmp_path/"absent")
    assert absent["constraints"] == []
    image = Image.open(path); draw = ImageDraw.Draw(image)
    draw.line((239,180,180,180), fill="black", width=2)
    draw.polygon([(180,180),(195,175),(195,185)], fill="black"); image.save(path)
    present = analyze_constraint_bindings(path, document, model, graph, tmp_path/"present")
    assert len(present["constraints"]) == 1
    assert present["constraints"][0]["entities"] == ["g009"]
    assert present["constraints"][0]["value"] == 45
    # A false carried coordinate hypothesis still must fail the same pixel
    # verification even though a different real arrow exists in the source.
    graph["radius_source_segment_hypotheses"][0]["source_evidence"]["segment_px"] = [[239.,160.],[180.,160.]]
    false = analyze_constraint_bindings(path, document, model, graph, tmp_path/"false")
    assert false["constraints"] == []


def test_radius_edit_diagnostic_does_not_claim_nominal_applied(tmp_path):
    path,_,model,graph=source(tmp_path)
    document={"records":[{"text":"R3","box":[[200,120],[230,120],[230,140],[200,140]]}]}
    graph["entities"]=[{"id":"g000","type":"ARC","start":[0,100],"end":[0,20],"center":[0,60],"radius":40,"clockwise":True}]
    graph["annotation_support"]=[{"record_id":"r000","candidate_entity_id":"g000","kind":"radius"}]
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"out")
    diagnostic=result["annotation_diagnostics"][0]
    assert result["constraints"]==[] and not diagnostic["dimensions_verified"]
    assert diagnostic["reason"]=="radius_requires_joint_or_topology_edit"
    assert diagnostic["upstream_target_hypotheses"][0]["nominal_difference"]==37
    assert not diagnostic["upstream_target_hypotheses"][0]["association_verified"]


def test_constructed_fillet_metadata_cannot_force_radius_constraint(tmp_path,monkeypatch):
    path,_,model,graph=source(tmp_path)
    observed_coarse_radius_source(path, model, radius=80., center=(100., 180.))
    document={"records":[{"text":"R40","box":[[240,180],[265,180],[265,200],[240,200]]}]}
    graph["entities"]=[{"id":"g009","type":"ARC","start":[40,60],"end":[0,100],"center":[0,60],
                        "radius":40,"clockwise":False,"dimension_bound":True,
                        "radius_binding_status":"applied",
                        "radius_binding":{"record_id":"r000","nominal":40.,"arrowhead_verified":True,
                                          "target_source_px":[180.,180.]}}]
    graph["annotation_support"]=[{"record_id":"r000","candidate_entity_id":"g009","arrowhead_verified":True,
                                  "source_evidence":{"segment_px":[[239.,180.],[180.,180.]],"arrowhead_verified":True}}]
    original=deepcopy(graph)
    monkeypatch.setattr("contour_agent.constraint_binding._leaders",lambda *args:[])
    class Selector:
        def select(self,image,topology,inventory):
            return {"schema_success":True,"http_success":True,"bindings":[
                {"record_id":"r000","candidate_id":inventory["candidates"][0]["id"],"observed_text":"R40"}],"relations":[]}
    result=analyze_constraint_bindings(path,document,model,graph,tmp_path/"unverified",provider=Selector(),use_api=True)
    assert result["constraints"]==[] and result["counts"]["api_accepted"]==0
    prior=result["constructed_radius_priors"][0]
    assert prior["status"]=="constructed_geometry_prior_unverified"
    assert prior["exact_constructed_radius_preserved"] and prior["upstream_dimension_bound_claim_ignored"]
    assert not prior["source_binding_verified"] and not prior["radius_constraint_pending_solve"]
    assert graph==original
    # The same constructed geometry is eligible only after the actual original
    # image supplies a directed arrow, independently of its stored true flags.
    image=Image.open(path);draw=ImageDraw.Draw(image)
    draw.line((239,180,180,180),fill="black",width=2)
    draw.polygon([(180,180),(195,175),(195,185)],fill="black");image.save(path)
    checked=analyze_constraint_bindings(path,document,model,graph,tmp_path/"verified")
    assert len(checked["constraints"])==1 and checked["constraints"][0]["kind"]=="radius"
    prior=checked["constructed_radius_priors"][0]
    assert prior["status"]=="source_verified_constraint_pending_solve" and prior["source_binding_verified"]
    assert prior["radius_constraint_pending_solve"] and not prior["dimensions_verified"]


def test_constructed_radius_with_wrong_source_record_stays_reviewable(tmp_path):
    from contour_agent.constraint_binding import _constructed_radius_priors
    entity={"id":"g1","type":"ARC","radius":3.,"dimension_bound":True,
            "radius_binding":{"record_id":"r0","nominal":3.,"arrowhead_verified":True}}
    records=[{"id":"r0","parsed":{"kind":"radius","nominal":40.}}]
    prior=_constructed_radius_priors({"entities":[entity]},records)[0]
    assert prior["exact_constructed_radius_preserved"] and not prior["construction_matches_source_ocr"]
    assert prior["status"]=="construction_metadata_requires_review" and not prior["source_binding_verified"]


def test_radius_coverage_does_not_exempt_undetected_arrows_or_metadata(tmp_path, monkeypatch):
    path, _, model, graph = source(tmp_path)
    document = {"records": [{"text": "R40", "box": [[240,180],[265,180],[265,200],[240,200]]}]}
    graph["entities"] = [{"id": "g000", "type": "ARC", "start": [0,100], "end": [0,20],
                          "center": [0,60], "radius": 40., "clockwise": True,
                          "radius_binding": {"record_id": "r000", "nominal": 40., "arrowhead_verified": True}}]
    graph["annotation_support"] = [{"record_id": "r000", "candidate_entity_id": "g000",
                                    "arrowhead_verified": True,
                                    "source_evidence": {"segment_px": [[239.,180.],[180.,180.]]}}]
    monkeypatch.setattr("contour_agent.constraint_binding._leaders", lambda *args: [])
    result = analyze_constraint_bindings(path, document, model, graph, tmp_path/"out")
    coverage = result["radius_binding_coverage"]
    assert coverage["recognized_radius_records"] == ["r000"]
    assert coverage["unknown_arrow_records"] == ["r000"]
    assert coverage["verified_absent_arrow_records"] == []
    assert coverage["confirmed_arrow_records"] == []
    assert not coverage["all_radius_records_resolved"]
    assert coverage["unresolved"][0]["reason"] == "source_arrow_not_verified"


def test_verified_radius_arrow_targeting_line_remains_required(tmp_path, monkeypatch):
    import numpy as np
    path, _, model, graph = source(tmp_path)
    document = {"records": [{"text": "R45", "box": [[240,180],[265,180],[265,200],[240,200]]}]}
    graph["entities"] = [{"id": "g009", "type": "LINE", "start": [40,100], "end": [40,20]}]
    image = Image.open(path); draw = ImageDraw.Draw(image)
    draw.line((180,100,180,260), fill="black", width=2)
    draw.line((239,180,180,180), fill="black", width=2)
    draw.polygon([(180,180),(195,175),(195,185)], fill="black"); image.save(path)
    monkeypatch.setattr("contour_agent.constraint_binding._leaders",
                        lambda *args: [np.array([[239.,180.],[180.,180.]])])
    result = analyze_constraint_bindings(path, document, model, graph, tmp_path/"out")
    coverage = result["radius_binding_coverage"]
    assert result["constraints"] == []
    assert coverage["confirmed_arrow_records"] == ["r000"]
    assert coverage["required_count"] == 1 and coverage["bound_count"] == 0
    assert coverage["required_mappings"][0]["candidate_entity_ids"] == ["g009"]
    assert coverage["unresolved"][0]["reason"] == "radius_target_is_line_requires_topology_edit"
    assert not coverage["all_confirmed_arrows_bound"] and not coverage["all_radius_records_resolved"]


def test_radius_coverage_keeps_provider_abstention_in_required_denominator(tmp_path, monkeypatch):
    result = analyze_stub(tmp_path, monkeypatch, stub_inventory(), [])
    coverage = result["radius_binding_coverage"]
    assert coverage["confirmed_arrow_records"] == ["r000"]
    assert coverage["required_count"] == 1 and coverage["bound_count"] == 0
    assert coverage["unresolved"][0]["reason"] == "provider_abstained_from_sent_record"
    assert not coverage["all_radius_records_resolved"]


def test_radius_binding_coverage_is_not_numeric_satisfaction(tmp_path, monkeypatch):
    result = analyze_stub(tmp_path, monkeypatch, stub_inventory(), [{"record_id":"r000","candidate_id":"c000"}])
    constraint = result["constraints"][0]
    assert constraint["required"] and constraint["enforcement"] == "exact"
    assert constraint["nominal_source"] == "source_ocr" and constraint["source_arrow_verified"]
    coverage = result["radius_binding_coverage"]
    assert coverage["all_confirmed_arrows_bound"] and coverage["all_radius_records_resolved"]
    assert coverage["bound_mappings"][0]["constraint_id"] == constraint["id"]
    assert not coverage["numeric_satisfaction_verified"]


def test_radius_coverage_preserves_competing_source_targets(tmp_path, monkeypatch):
    inventory = stub_inventory()
    second = deepcopy(inventory["all_candidates"][0])
    second.update(id="c001", entities=["g001"], local_reliable=False)
    inventory["all_candidates"][0]["local_reliable"] = False
    inventory["all_candidates"].append(second)
    inventory["candidates"] = inventory["all_candidates"]
    result = analyze_stub(tmp_path, monkeypatch, inventory, [], graph={
        "nodes": [], "entities": [{"id":"g000","type":"ARC"},{"id":"g001","type":"ARC"}]})
    coverage = result["radius_binding_coverage"]
    assert coverage["ambiguous_count"] == 1
    assert coverage["ambiguous"][0]["candidate_entity_ids"] == ["g000", "g001"]
    assert coverage["required_count"] == 1 and not coverage["all_confirmed_arrows_bound"]


def test_explicit_radius_arrow_can_cross_contour_only_with_complete_source_shaft():
    import cv2
    import numpy as np
    from contour_agent.constraint_binding import _leader_evidence, verify_source_arrow_proposal
    gray = np.full((120,290), 255, np.uint8)
    cv2.line(gray, (100,60), (229,60), 0, 2)
    cv2.fillConvexPoly(gray, np.array([[100,60],[116,54],[116,66]], np.int32), 0)
    cv2.line(gray, (180,35), (180,85), 0, 2)
    arc = np.c_[np.full(9,100.), np.linspace(58,62,9)]
    crossing = np.array([[180.,35.],[180.,85.]])
    record = {"id":"r000", "parsed":{"kind":"radius","nominal":40.},
              "box":[[230.,45.],[265.,45.],[265.,75.],[230.,75.]]}
    proposal = {"record_id":"r000", "shaft_px":[229.,60.], "tip_px":[100.,60.]}
    # Proposal origin is not evidence: the same complete source proof applies
    # to an independently detected Hough segment and an online hypothesis.
    detected = _leader_evidence(np.asarray(record["box"]), arc, np.array([60.,60.]),
                               [np.array([proposal["shaft_px"], proposal["tip_px"]])], 5.,
                               gray, [arc,crossing])
    assert detected is not None and detected["shaft_evidence"]["verified"]
    assert detected["proposal_origin"] == "source_hough"
    assert not detected["contour_visibility"]["verified"]
    verified = verify_source_arrow_proposal(gray, record, proposal, arc, 5., [arc,crossing])
    assert verified is not None and verified["arrowhead_verified"]
    assert verified["crossing_source_contour"]
    assert verified["shaft_evidence"]["verified"]
    assert not verified["contour_visibility"]["verified"]
    damaged = gray.copy(); damaged[54:67,135:160] = 255
    assert verify_source_arrow_proposal(damaged, record, proposal, arc, 5., [arc,crossing]) is None
    assert _leader_evidence(np.asarray(record["box"]), arc, np.array([60.,60.]),
                            [np.array([proposal["shaft_px"], proposal["tip_px"]])], 5.,
                            damaged, [arc,crossing]) is None
    assert verify_source_arrow_proposal(gray, record, {**proposal,"record_id":"r001"}, arc, 5.) is None


def test_provider_arrow_proposal_requires_real_arrow_pixels():
    import cv2
    import numpy as np
    from contour_agent.constraint_binding import verify_source_arrow_proposal
    gray = np.full((120,290), 255, np.uint8)
    cv2.line(gray, (100,60), (229,60), 0, 2)
    record = {"id":"r000", "parsed":{"kind":"radius","nominal":40.},
              "box":[[230.,45.],[265.,45.],[265.,75.],[230.,75.]]}
    proposal = {"record_id":"r000", "shaft_px":[229.,60.], "tip_px":[100.,60.], "verified":True}
    arc = np.c_[np.full(9,100.), np.linspace(58,62,9)]
    assert verify_source_arrow_proposal(gray, record, proposal, arc, 5.) is None


def test_online_arrow_proposals_persist_json_with_competing_arc_scores(tmp_path, monkeypatch):
    """Online pixel arithmetic must not leak numpy.bool_ into the receipt."""
    import numpy as np
    from contour_agent.constraint_binding import _label_ray_entry
    path, _, model, graph = source(tmp_path)
    observed_coarse_radius_source(path, model)
    document = {"records": [{"text": "R45", "box": [[240,180],[265,180],[265,200],[240,200]],
                              "source_arrow_proposals": [{"record_id":"r000", "shaft_px":[239.,180.],
                                                          "tip_px":[180.,180.]}]}]}
    graph["entities"] = [
        {"id":"g000", "type":"ARC", "start":[0,100], "end":[0,20],
         "center":[0,60], "radius":40., "clockwise":True},
        {"id":"g001", "type":"ARC", "start":[0,40], "end":[0,-40],
         "center":[0,0], "radius":40., "clockwise":True}]
    image = Image.open(path); draw = ImageDraw.Draw(image)
    draw.line((239,180,180,180), fill="black", width=2)
    draw.polygon([(180,180),(195,175),(195,185)], fill="black"); image.save(path)
    monkeypatch.setattr("contour_agent.constraint_binding._leaders", lambda *args: [])
    result = analyze_constraint_bindings(path,document,model,graph,tmp_path/"online")
    inventory = json.loads((tmp_path/"online/binding-candidates.json").read_text(encoding="utf8"))
    assert len(inventory["all_candidates"]) == 2
    assert all(type(row["local_reliable"]) is bool for row in inventory["all_candidates"])
    assert result["constraints"][0]["entities"] == ["g000"]
    persisted = json.loads((tmp_path/"online/constraint-bindings.json").read_text(encoding="utf8"))
    assert persisted["radius_binding_coverage"]["bound_count"] == 1
    # Exercise the numpy ray-box arithmetic that seeded the invalid bool type.
    value = _label_ray_entry(np.array([229.,60.]),np.array([1.,0.]),
                             np.array([230.,45.]),np.array([265.,75.]),20.)
    assert type(value) is float

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


def analyze_stub(tmp_path,monkeypatch,inventory,bindings,relations=None,graph=None,transcribe=True):
    monkeypatch.setattr("contour_agent.constraint_binding.build_binding_candidates",lambda *a,**k:deepcopy(inventory))
    texts={row["id"]:row["text"] for row in inventory["all_records"]}
    if transcribe:
        bindings=[{**row,"observed_text":row.get("observed_text",texts.get(row["record_id"],"unknown"))} for row in bindings]
    class Provider:
        def select(self,*args):
            return {"status":"succeeded","network_requests":1,"http_success":True,"schema_success":True,
                    "bindings":bindings,"relations":relations or []}
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

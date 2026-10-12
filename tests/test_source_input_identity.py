"""Reviewed OCR and CAD publication identity, using synthetic source data only."""
from copy import deepcopy
import json

import pytest

from contour_agent.dataset import read_ocr
from contour_agent.parametric_pipeline import CORE, refine_parametric, source_input_identity
from contour_agent.service import _load_published_source_ocr
from contour_agent.source_ocr_review import (canonical_document_sha256, load_effective_ocr,
                                            review_source_ocr)
from contour_agent.store import JobStore
from test_parametric_pipeline import prepared, recover, seed_pending_job, supported_graph
from test_strict_publication_recovery import _certified_core, _save_model


class SourceTextProvider:
    def __init__(self, text):
        self.text=text

    def transcribe(self, image, rows, *, view, timeout):
        return {"status":"succeeded", "http_success":True, "schema_success":True,
                "network_requests":1,
                "transcriptions":[{"id":row["id"],"text":self.text} for row in rows]}


def reviewed_core(tmp_path, *, use_api=True):
    model,output=_certified_core(tmp_path)
    image=tmp_path/"source.png"
    raw=tmp_path/"source-ocr.json"
    raw.write_text(json.dumps({"meta":{"original_size":{"width":200,"height":200}},
        "records":[{"text":"19","box":[[10,10],[30,10],[30,30],[10,30]]}]}),"utf8")
    original=read_ocr(raw)
    effective,_=review_source_ocr(image,original,model,output,provider=SourceTextProvider("10"),
                                  use_api=use_api,raw_ocr_path=raw)
    model["source_input_identity"]=source_input_identity(image,effective)
    _save_model(output,model)
    job=seed_pending_job(tmp_path,model,output)
    job.update(source_image=str(image),source_ocr=str(raw))
    JobStore(tmp_path/"runtime").save(job)
    return image,raw,effective,model,output


@pytest.mark.parametrize("use_api",[True,False])
def test_valid_review_identity_recovers_current_native_certificate(tmp_path,monkeypatch,use_api):
    image,raw,effective,model,output=reviewed_core(tmp_path,use_api=use_api)
    assert model["source_input_identity"]["effective_ocr_sha256"]==canonical_document_sha256(effective)
    before={name:(output/name).read_bytes() for name in CORE}
    assert _load_published_source_ocr(image,raw,output,model)==effective
    service=recover(tmp_path,monkeypatch)
    try:
        restored=service.store.get("persisted")
        assert restored["status"]=="completed" and restored["parameterization"]["accepted"]
        assert restored["parameterization"]["recovery_certificate"]["complete"]
        assert before=={name:(output/name).read_bytes() for name in CORE}
    finally:
        service.close()


def test_other_valid_review_cannot_certify_old_core_or_enter_feedback(tmp_path,monkeypatch):
    image,raw,effective,model,output=reviewed_core(tmp_path)
    before={name:(output/name).read_bytes() for name in CORE}
    # Both reviews belong to the same source image and raw OCR; the second is
    # internally valid. Only the CORE's effective-document identity catches it.
    other,_=review_source_ocr(image,read_ocr(raw),model,output,provider=SourceTextProvider("11"),
                              use_api=True,raw_ocr_path=raw)
    assert other!=effective and load_effective_ocr(image,raw,output)==other
    with pytest.raises(ValueError,match="输入身份不一致"):
        _load_published_source_ocr(image,raw,output,model)
    store=JobStore(tmp_path/"runtime")
    journal=store.get("persisted")
    journal["parameterization"]=deepcopy(model["parameterization"])
    store.save(journal)
    service=recover(tmp_path,monkeypatch)
    try:
        restored=service.store.get("persisted")
        assert restored["status"]=="failed" and not restored["automatic_completion"]
        assert not restored["validation"]["recovery_integrity_verified"]
        assert not restored["parameterization"]["accepted"]
        assert not restored["parameterization"]["recovery_certificate"]["complete"]
        assert before=={name:(output/name).read_bytes() for name in CORE}
        assert restored["artifacts"]["dxf"]  # Preserve the unverified draft.
        def forbidden(*args,**kwargs):
            pytest.fail("Mixed OCR/CAD must stop before any feedback provider call")
        monkeypatch.setattr(service.feedback_provider,"select",forbidden)
        feedback={"id":"feedback","status":"queued","mode":"autonomous_image","events":[],
                  "updated_at":"2026-10-10T00:00:00+00:00",
                  "issues":[],"source_image":str(image),"source_ocr":str(raw),
                  "parent_artifact_directory":str(output),"use_api":True}
        service.store.save(feedback)
        service._feedback_revision("feedback")
        rejected=service.store.get("feedback")
        assert rejected["status"]=="failed"
        assert "输入身份不一致" in rejected["issues"][0]
    finally:
        service.close()


def test_valid_review_requires_core_identity(tmp_path,monkeypatch):
    image,raw,effective,model,output=reviewed_core(tmp_path)
    model.pop("source_input_identity")
    _save_model(output,model)
    with pytest.raises(ValueError,match="输入身份不一致"):
        _load_published_source_ocr(image,raw,output,model)
    service=recover(tmp_path,monkeypatch)
    try:
        assert service.store.get("persisted")["status"]=="failed"
    finally:
        service.close()


def test_legacy_core_without_review_keeps_existing_recovery_path(tmp_path,monkeypatch):
    model,output=_certified_core(tmp_path)
    seed_pending_job(tmp_path,model,output)
    service=recover(tmp_path,monkeypatch)
    try:
        assert service.store.get("persisted")["status"]=="completed"
    finally:
        service.close()


def test_pipeline_publication_binds_effective_document_without_relabeling_baseline(tmp_path,monkeypatch):
    image,baseline,output=prepared(tmp_path)
    document={"meta":{"box_coord_space":"original_image","original_size":{"width":200,"height":200}},
              "records":[]}
    original=deepcopy(document)
    original["meta"]["review_evidence"]={"status":"before"}
    baseline["source_input_identity"]=source_input_identity(image,original)
    _save_model(output,baseline)
    before={name:(output/name).read_bytes() for name in CORE}
    graph=supported_graph(image,baseline)
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",lambda *a,**k:{
        "constraints":[],"provider":{"status":"disabled","network_requests":0},"counts":{},"issues":[]})
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",lambda *a,**k:{
        "accepted":False,"status":"underconstrained","entities":deepcopy(graph["entities"]),
        "underconstrained":True,"constraints":[],"validation":{"passed":False}})
    result,stage=refine_parametric(image,document,baseline,output,use_api=False,frozen_topology=graph)
    published=json.loads((output/"model.json").read_text("utf8"))
    expected=source_input_identity(image,document)
    assert stage["topology_exported"] and not stage["accepted"]
    assert result["source_input_identity"]==published["source_input_identity"]==expected
    assert published["parameterization"]["source_input_identity"]==expected
    assert baseline["source_input_identity"]!=expected
    assert {name:(output/("baseline-"+name)).read_bytes() for name in CORE}==before
    provenance=json.loads((output/"workflow-provenance.json").read_text("utf8"))
    assert provenance["input_ocr_canonical_sha256"]==expected["effective_ocr_sha256"]
    assert provenance["published_source_input_identity"]==expected

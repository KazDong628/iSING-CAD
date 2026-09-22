import json
from pathlib import Path

import pytest

from contour_agent.config import Settings
from contour_agent import autonomous_qualification as qualification


def install_fake_service(monkeypatch, tmp_path, *, reference_match=True, receipts=None, manual=False, model_metadata=None, source_hashes=None):
    calls, scores, closes = [], [], []
    catalog = {"cases": [{"id": str(i), "split": "calibration" if i == 49 else "holdout", "supported_template": None, "gt_dxf": f"GT/{i}.dxf"} for i in range(50)]}
    gt=tmp_path/"dataset"/"GT";gt.mkdir(parents=True)
    for i in range(50): (gt/f"{i}.dxf").write_text(f"synthetic reference {i}")

    class FakeService:
        def __init__(self, settings):
            self.settings = settings
            self.catalog = catalog

        def create_auto_case(self, case_id, use_api, asynchronous, **options):
            assert asynchronous is False
            assert case_id in {case["id"] for case in catalog["cases"]}
            calls.append((case_id, use_api, self.settings.runtime_root))
            output = Path(self.settings.runtime_root) / "jobs" / str(len(calls))
            output.mkdir(parents=True)
            (output / "drawing.dxf").write_text("synthetic test artifact")
            receipt = receipts[len(calls) - 1] if receipts else {"status": "disabled", "network_requests": 0}
            return {"id": f"job-{len(calls)}", "status": "completed", "artifact_directory": str(output),
                    "validation": {"passed": True, "scaled_mm": True}, "automatic_completion": True,
                    "manual_intervention": manual, "provider": receipt, "issues": [],
                    "source":{"image_sha256":(source_hashes or {}).get(case_id)},
                    "extraction":{"model":model_metadata}}

        def close(self):
            closes.append(True)

    def score(prediction, reference):
        # Scoring must follow export, and the source runtime never receives a
        # reference path or any metric in its create_auto_case arguments.
        assert prediction.is_file()
        scores.append((prediction, reference))
        return {"status": "compared", "artifact_completed": True, "geometry_valid": True, "scaled_mm": True,
                "reference_compared": True, "reference_within_0_1mm": reference_match, "alignment_kind": "shape_diagnostic"}

    monkeypatch.setattr(qualification, "AgentService", FakeService)
    monkeypatch.setattr(qualification, "evaluate_autonomous_artifact", score)
    settings = Settings(dataset_root=tmp_path / "dataset", runtime_root=tmp_path / "runtime", api_key="")
    return settings, calls, scores, closes


def test_subset_keeps_fifty_denominator_and_artifact_is_not_accuracy(monkeypatch, tmp_path):
    settings, calls, scores, closes = install_fake_service(monkeypatch, tmp_path, reference_match=False)
    report, path = qualification.qualify_autonomous(settings, cases=["0", "1"])
    assert report["mode"] == "autonomous_image"
    assert report["status"] == "completed"
    assert report["summary"]["total"] == 50
    assert report["summary"]["attempted"] == 2
    assert report["summary"]["auto_generated"] == 2
    assert report["summary"]["reference_within_0_1mm"] == 0
    assert report["summary"]["not_attempted"] == 48
    assert not report["qualification"]["strict_requested_scope_passed"]
    assert len(report["trials"]) == 2 and len(scores) == 2
    assert all(root != settings.runtime_root for _, _, root in calls)
    assert len(closes) == 1
    assert json.loads(path.read_text(encoding="utf8"))["completed_trials"] == 2
    assert report["online_request_summary"]["network_attempts"]["total"] == 0


def test_failed_repeat_cannot_be_hidden_by_last_online_success(monkeypatch, tmp_path):
    receipts = [
        {"status": "failed", "http_success": False, "schema_success": False, "network_requests": 2, "verdict": "uncertain", "overlay_sent": True, "ground_truth_sent": False},
        {"status": "succeeded", "http_success": True, "schema_success": True, "network_requests": 1, "verdict": "match", "overlay_sent": True, "ground_truth_sent": False},
    ]
    settings, _, _, _ = install_fake_service(monkeypatch, tmp_path, receipts=receipts)
    report, _ = qualification.qualify_autonomous(settings, online=True, repeats=2, cases=["0"])
    assert report["trials"][1]["strict_scope_passed"]
    assert not report["trials"][0]["strict_scope_passed"]
    assert not report["qualification"]["strict_requested_scope_passed"]
    assert not report["cases"][0]["strict_scope_passed"]
    assert report["online_request_summary"]["logical_calls"]["total"] == 2
    assert report["online_request_summary"]["logical_calls"]["http_success_rate"] == .5
    assert report["online_request_summary"]["network_attempts"]["total"] == 3
    assert report["online_request_summary"]["network_attempts"]["retries"] == 1


def test_manual_intervention_cannot_pass_autonomous_gate(monkeypatch, tmp_path):
    settings, _, _, _ = install_fake_service(monkeypatch, tmp_path, manual=True)
    report, _ = qualification.qualify_autonomous(settings, cases=["0"])
    assert not report["qualification"]["strict_requested_scope_passed"]
    assert report["summary"]["manual_interventions"] == 1
    assert report["summary"]["auto_generated"] == 0
    assert report["summary"]["reference_within_0_1mm"] == 1
    assert report["summary"]["autonomous_reference_within_0_1mm"] == 0


def test_successful_subset_does_not_claim_whole_dataset(monkeypatch, tmp_path):
    settings, _, _, _ = install_fake_service(monkeypatch, tmp_path)
    report, _ = qualification.qualify_autonomous(settings, cases=["0"])
    assert report["qualification"]["strict_requested_scope_passed"]
    assert not report["qualification"]["strict_whole_dataset_passed"]
    assert report["summary"]["auto_generated_rate"] == .02


def test_same_clock_tick_runs_use_distinct_artifact_directories(monkeypatch, tmp_path):
    from datetime import datetime, timezone
    fixed = datetime(2026, 9, 20, tzinfo=timezone.utc)
    class FrozenClock:
        @staticmethod
        def now(tz=None):
            return fixed
    monkeypatch.setattr(qualification, "datetime", FrozenClock)
    settings, calls, _, _ = install_fake_service(monkeypatch, tmp_path)
    first, path1 = qualification.qualify_autonomous(settings, cases=["0"])
    second, path2 = qualification.qualify_autonomous(settings, cases=["0"])
    assert first["run_id"] != second["run_id"]
    assert path1.is_file() and path2.is_file() and path1 != path2
    assert calls[0][2] != calls[1][2]


def test_unknown_and_repeated_selection_are_rejected(monkeypatch, tmp_path):
    settings, calls, _, closes = install_fake_service(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        qualification.qualify_autonomous(settings, cases=["0", "0"])
    with pytest.raises(ValueError, match="Unknown"):
        qualification.qualify_autonomous(settings, cases=["missing"])
    assert not calls and closes


def test_reports_redact_provider_credential_echo(monkeypatch, tmp_path):
    receipt = {"status": "failed", "http_success": False, "schema_success": False, "network_requests": 1,
               "issues": ["unsafe sk-test-secret-123"]}
    settings, _, _, _ = install_fake_service(monkeypatch, tmp_path, receipts=[receipt])
    settings.api_key = "sk-test-secret-123"
    report, path = qualification.qualify_autonomous(settings, online=True, cases=["0"])
    assert settings.api_key not in path.read_text(encoding="utf8")
    assert settings.api_key not in json.dumps(report)


def test_scoring_failure_keeps_actual_calls_and_exported_artifact(monkeypatch, tmp_path):
    receipt = {"status": "succeeded", "http_success": True, "schema_success": True, "network_requests": 2,
               "verdict": "match", "overlay_sent": True, "ground_truth_sent": False}
    settings, _, _, _ = install_fake_service(monkeypatch, tmp_path, receipts=[receipt])

    def failed_score(*args):
        raise RuntimeError("untrusted scorer failure detail")

    monkeypatch.setattr(qualification, "evaluate_autonomous_artifact", failed_score)
    report, _ = qualification.qualify_autonomous(settings, online=True, cases=["0"])
    assert report["summary"]["artifact_completed"] == 1
    assert report["summary"]["reference_compared"] == 0
    assert report["trials"][0]["comparison"]["status"] == "evaluation_error"
    assert report["online_request_summary"]["logical_calls"]["total"] == 1
    assert report["online_request_summary"]["network_attempts"]["total"] == 2
    assert not report["qualification"]["strict_requested_scope_passed"]
    assert "untrusted scorer failure detail" not in json.dumps(report)


def test_learned_model_training_split_overrides_legacy_holdout_meaning(monkeypatch, tmp_path):
    identities=[{"id":str(index),"split":split,"source_image_sha256":str(index+1)*64}
                for index,split in enumerate(("train","val","test"))]
    metadata={"checkpoint_sha256":"a"*64,"training_provenance":{
        "manifest_sha256":"b"*64,"case_identities":identities,"consumed_artifacts":identities}}
    settings,_,_,_=install_fake_service(monkeypatch,tmp_path,model_metadata=metadata,
        source_hashes={row["id"]:row["source_image_sha256"] for row in identities})
    report,_=qualification.qualify_autonomous(settings,cases=["0","1","2"],use_segmentation=True)
    trials=report["trials"]
    assert all(trial["split"] == trial["legacy_catalog_split"] == "holdout" for trial in trials)
    assert [trial["model_exposure"]["split"] for trial in trials] == ["train","val","test"]
    assert [trial["model_exposure"]["role"] for trial in trials] == ["optimization_training","checkpoint_selection","development_test"]
    assert trials[0]["model_exposure"]["optimization_exposed"] is True
    assert trials[1]["model_exposure"]["checkpoint_selection_exposed"] is True
    assert trials[2]["model_exposure"]["optimization_exposed"] is False
    assert report["model_exposure_summary"]["trial_split_counts"] == {"train":1,"val":1,"test":1}
    assert all(trial["model_exposure"]["blind_test"] is False for trial in trials)
    assert report["blind_test"] is False and report["cases"][0]["model_exposure"] == trials[0]["model_exposure"]


def test_default_workflow_does_not_read_segmentation_checkpoint(monkeypatch,tmp_path):
    settings,_,_,_=install_fake_service(monkeypatch,tmp_path)
    settings.segmentation_checkpoint="a checkpoint not selected by this run"
    report,_=qualification.qualify_autonomous(settings,cases=["0"])
    exposure=report["trials"][0]["model_exposure"]
    assert exposure["split"] == "not_applicable"
    assert exposure["evidence_status"] == "not_applicable"
    assert exposure["optimization_exposed"] is None


def test_checkpoint_metadata_only_loading_and_unknown_source_is_not_unseen(monkeypatch,tmp_path):
    torch=pytest.importorskip("torch")
    checkpoint=tmp_path/"metadata.pt"
    torch.save({"provenance":{"manifest_sha256":"a"*64}},checkpoint)
    original=torch.load;calls=[]
    def load(path,**kwargs):
        calls.append(kwargs)
        return original(path,**kwargs)
    monkeypatch.setattr(torch,"load",load)
    settings=Settings(segmentation_checkpoint=str(checkpoint),api_key="")
    context=qualification._exposure_context(settings,True)
    result=qualification._model_exposure("0","b"*64,context)
    assert calls == [{"map_location":"cpu","weights_only":True}]
    assert result["split"] == "unknown" and result["optimization_exposed"] is None
    assert result["blind_test"] is False


def test_legacy_checkpoint_uses_only_hash_verified_training_manifest(tmp_path):
    import hashlib
    manifest=tmp_path/"manifest.json"
    identities=[{"id":"0","split":"train","source_image_sha256":"1"*64,"image":"not-opened.png","mask":"not-opened-label.png"},
                {"id":"1","split":"val","source_image_sha256":"2"*64,"trainable":False,"mask":None}]
    manifest.write_text(json.dumps({"cases":identities}))
    metadata={"checkpoint_sha256":"a"*64,"training_provenance":{"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest()}}
    settings=Settings(segmentation_manifest=str(manifest),api_key="")
    context=qualification._exposure_context(settings,True,metadata)
    training=qualification._model_exposure("0","1"*64,context)
    excluded=qualification._model_exposure("1","2"*64,context)
    assert training["evidence_source"] == "checkpoint_hash_verified_training_manifest"
    assert training["optimization_exposed"] is True
    assert excluded["split"] == "val" and excluded["role"] == "excluded_label_in_development_split"
    assert excluded["checkpoint_selection_exposed"] is False
    assert qualification._model_exposure("0","9"*64,context)["evidence_status"] == "source_hash_mismatch"
    manifest.write_text(manifest.read_text()+" ")
    changed=qualification._exposure_context(settings,True,metadata)
    assert qualification._model_exposure("0","1"*64,changed)["split"] == "unknown"
    assert changed["status"] == "training_manifest_hash_mismatch"


def test_case_assignment_alone_does_not_prove_consumption(tmp_path):
    metadata={"training_provenance":{"case_identities":[{"id":"0","split":"train","source_image_sha256":"1"*64}]}}
    context=qualification._exposure_context(Settings(api_key=""),True,metadata)
    result=qualification._model_exposure("0","1"*64,context)
    assert result["split"] == "train" and result["identity_verified"] is True
    assert result["optimization_exposed"] is None
    assert result["role"] == "declared_development_split_consumption_unknown"


def test_conflicting_consumption_hash_cannot_claim_training_exposure():
    identity={"id":"0","split":"train","source_image_sha256":"1"*64}
    metadata={"training_provenance":{"case_identities":[identity],"consumed_artifacts":[{**identity,"source_image_sha256":"2"*64}]}}
    context=qualification._exposure_context(Settings(api_key=""),True,metadata)
    result=qualification._model_exposure("0","1"*64,context)
    assert result["split"] == "unknown" and result["optimization_exposed"] is None
    assert not result["identity_verified"]


def test_archive_reference_is_read_only_after_export_and_exact_bytes_are_scored(monkeypatch,tmp_path):
    import hashlib
    import zipfile
    settings,calls,scores,_=install_fake_service(monkeypatch,tmp_path)
    case=qualification.AgentService(settings).catalog["cases"][0]
    archive=settings.dataset_root/"GT"/"package.zip"
    raw=b"original unmodified DXF bytes\r\n"
    with zipfile.ZipFile(archive,"w") as package: package.writestr("nested/main.dxf",raw)
    before=archive.read_bytes()
    case.update(gt_dxf=None,gt_archive={"path":"GT/package.zip","member":"nested/main.dxf"})
    original=qualification._reference_for_scoring
    def after_export(*args):
        assert len(calls) == 1 and not scores
        return original(*args)
    monkeypatch.setattr(qualification,"_reference_for_scoring",after_export)
    report,_=qualification.qualify_autonomous(settings,cases=["0"])
    trial=report["trials"][0]
    assert scores[0][1].read_bytes() == raw
    assert scores[0][1].parent.name == "reference-only"
    assert trial["reference_sha256"] == hashlib.sha256(raw).hexdigest()
    assert trial["reference_source"] == {"kind":"zip_member","path":"GT/package.zip","member":"nested/main.dxf"}
    assert archive.read_bytes() == before and not (settings.dataset_root/"nested").exists()
    assert report["cases"][0]["reference_sha256"] == trial["reference_sha256"]


@pytest.mark.parametrize("member",["../escape.dxf","/absolute.dxf","C:\\escape.dxf","nested/../../escape.dxf","notes.txt"])
def test_archive_reference_rejects_unsafe_member(tmp_path,member):
    import zipfile
    root=tmp_path/"dataset";root.mkdir()
    with zipfile.ZipFile(root/"package.zip","w") as package: package.writestr(member,b"fixture")
    settings=Settings(dataset_root=root,api_key="")
    with pytest.raises(ValueError):
        qualification._reference_for_scoring(settings,{"id":"safe","gt_archive":{"path":"package.zip","member":member}},tmp_path/"runtime")
    assert not (tmp_path/"runtime").exists()


def test_archive_limits_and_original_partial_scope_are_preserved(monkeypatch,tmp_path):
    import zipfile
    root=tmp_path/"dataset";root.mkdir()
    with zipfile.ZipFile(root/"package.zip","w") as package: package.writestr("nested/reference_scored_only.dxf",b"fixture bytes")
    settings=Settings(dataset_root=root,api_key="")
    case={"id":"safe","gt_archive":{"path":"package.zip","member":"nested/reference_scored_only.dxf"}}
    monkeypatch.setattr(qualification,"MAX_REFERENCE_BYTES",5)
    with pytest.raises(ValueError,match="oversized"):
        qualification._reference_for_scoring(settings,case,tmp_path/"runtime")
    monkeypatch.setattr(qualification,"MAX_REFERENCE_BYTES",100)
    path,_=qualification._reference_for_scoring(settings,case,tmp_path/"runtime")
    assert "scored_only" in path.stem
    case["gt_archive"]["path"]="../package.zip"
    with pytest.raises(ValueError):qualification._reference_for_scoring(settings,case,tmp_path/"runtime")

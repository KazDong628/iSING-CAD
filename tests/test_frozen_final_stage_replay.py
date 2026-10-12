import hashlib
import json

import pytest

from scripts.recheck_frozen_final_stage import validate_upstream
from scripts.report_source_generalization import source_review


def test_final_stage_refuses_unfrozen_or_changed_upstream(tmp_path):
    after = tmp_path / "after"
    after.mkdir()
    target = after / "topology.json"
    target.write_text('{"entities": []}', "utf8")
    manifest = {"frozen_artifacts": {target.name: hashlib.sha256(target.read_bytes()).hexdigest()}}
    path = tmp_path / "run-manifest.json"
    path.write_text(json.dumps(manifest), "utf8")
    with pytest.raises(ValueError, match="not_frozen"):
        validate_upstream(tmp_path)
    manifest["prediction_frozen_at"] = "2026-10-10T00:00:00Z"
    path.write_text(json.dumps(manifest), "utf8")
    assert validate_upstream(tmp_path) == manifest
    target.write_text('{"entities": [1]}', "utf8")
    with pytest.raises(ValueError, match="artifact_changed"):
        validate_upstream(tmp_path)


def test_final_stage_report_does_not_count_reused_ocr_as_new_requests(tmp_path):
    after = tmp_path / "after"
    after.mkdir()
    review = after / "source-ocr-review.json"
    review.write_text(json.dumps({"status": "completed_with_unresolved", "pages": [
        {"provider": {"network_requests": 2, "http_success": True, "schema_success": True}}]}), "utf8")
    effective = after / "effective-source-ocr.json"
    effective.write_text("{}", "utf8")
    manifest = {"execution_scope": "frozen_topology_online_final_stage_replay",
        "frozen_artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (review, effective)}}
    (tmp_path / "run-manifest.json").write_text(json.dumps(manifest), "utf8")
    result = source_review(review)
    assert result["requests"] == result["http_successful_pages"] == result["schema_successful_pages"] == 0
    assert result["upstream_evidence_requests"] == 2
    assert result["reused_from_upstream"] is True
    assert all(row["passed"] for row in result["frozen_evidence_checks"])


def test_replay_preserves_source_first_policy_but_never_uses_solved_geometry(tmp_path):
    from scripts.recheck_frozen_final_stage import replay_baseline
    from contour_agent.reconstruction_priority import SOURCE_FIRST,mask_is_hard
    after=tmp_path/'after';after.mkdir()
    baseline={'entities':[{'id':'original'}], 'extraction':{'raw_polyline_px':[[1,2],[3,4]]}}
    (after/'baseline-model.json').write_text(json.dumps(baseline),'utf8')
    (after/'model.json').write_text(json.dumps({'entities':[{'id':'solved'}],
        'reconstruction_priority':{'policy':SOURCE_FIRST}}),'utf8')
    result=replay_baseline(tmp_path)
    assert not mask_is_hard(result)
    assert result['entities']==baseline['entities']
    assert result['extraction']==baseline['extraction']
    (after/'model.json').write_text(json.dumps({'entities':[{'id':'solved'}]}),'utf8')
    assert replay_baseline(tmp_path)==baseline

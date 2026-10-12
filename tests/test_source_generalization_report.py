import hashlib
import json
from scripts.report_source_generalization import source_review


def test_unreadable_rows_remain_in_frozen_review_denominator(tmp_path):
    after=tmp_path/"after";after.mkdir()
    review=after/"source-ocr-review.json"
    review.write_text(json.dumps({"status":"completed_with_unresolved","pages":[],"decisions":[
        {"record_id":"r001","status":"unresolved_transcription_unreadable"},
        {"record_id":"r002","status":"unresolved_transcription_disagreement"},
        {"record_id":"r003","status":"consensus_unchanged"}]}),"utf8")
    effective=after/"effective-source-ocr.json";effective.write_text('{}',"utf8")
    manifest=tmp_path/"run-manifest.json"
    manifest.write_text(json.dumps({"frozen_artifacts":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [review,effective]}}),"utf8")
    result=source_review(review)
    assert [d["record_id"] for d in result["unresolved"]]==["r001","r002"]
    assert all(c["passed"] for c in result["frozen_evidence_checks"])
    # Internally well-formed but swapped evidence is not attributed to old CAD.
    effective.write_text('{"records":[]}',"utf8")
    result=source_review(review)
    assert result["status"]=="unfrozen_or_mixed_ocr_sidecars"
    assert result["requests"] is None and "changed" not in result


def test_missing_old_review_is_not_claimed_as_success(tmp_path):
    assert source_review(tmp_path/"source-ocr-review.json")["status"]=="not_implemented_in_this_run"

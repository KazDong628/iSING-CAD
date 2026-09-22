import ezdxf
import pytest

from contour_agent.evaluation import compare_dxf, summarize_evaluation, summarize_online_trials


def _profile(path, shift=0, radius=5, reverse=False, construction=False, duplicate=False):
    doc = ezdxf.new()
    doc.units = ezdxf.units.MM
    space = doc.modelspace()
    lines = [((shift, 0), (shift + 10, 0)), ((shift + 10, 0), (shift + 10, 10))]
    for start, end in reversed(lines) if reverse else lines:
        space.add_line(end if reverse else start, start if reverse else end, dxfattribs={"layer": "MAIN"})
    space.add_arc((shift, 10), radius, 0, 180, dxfattribs={"layer": "MAIN"})
    if construction:
        space.add_circle((9999, 9999), 1000, dxfattribs={"layer": "CONSTRUCTION_DEBUG"})
        space.add_line((2000, 2000), (4000, 4000), dxfattribs={"layer": "GT_MEASURE"})
        space.add_text("arbitrary annotation")
    if duplicate:
        space.add_line((shift, 0), (shift + 10, 0), dxfattribs={"layer": "MAIN"})
    doc.saveas(path)
    return path


def test_geometry_comparison_ignores_order_and_direction(tmp_path):
    ref = _profile(tmp_path / "reference.dxf")
    pred = _profile(tmp_path / "prediction.dxf", reverse=True)
    result = compare_dxf(pred, ref)
    assert result["passed"]
    assert result["symmetric_max_error_mm"] < 1e-10
    assert result["engineering_certified"] is False


@pytest.mark.parametrize("kwargs", [{"shift": 1}, {"radius": 6}])
def test_equal_entity_counts_do_not_hide_geometry_error(tmp_path, kwargs):
    ref = _profile(tmp_path / "reference.dxf")
    pred = _profile(tmp_path / "prediction.dxf", **kwargs)
    result = compare_dxf(pred, ref)
    assert result["selection"]["prediction"]["entities"] == result["selection"]["reference"]["entities"]
    assert not result["passed"]
    assert result["symmetric_max_error_mm"] > 0.5


def test_construction_and_measurement_are_not_contour(tmp_path):
    ref = _profile(tmp_path / "reference.dxf", construction=True)
    pred = _profile(tmp_path / "prediction.dxf")
    result = compare_dxf(pred, ref)
    assert result["passed"]
    assert result["selection"]["reference"]["excluded"]["layer:GT_MEASURE"] == 1


def test_duplicate_geometry_not_rewarded(tmp_path):
    ref = _profile(tmp_path / "reference.dxf")
    pred = _profile(tmp_path / "prediction.dxf", duplicate=True)
    result = compare_dxf(pred, ref)
    assert result["symmetric_max_error_mm"] < 1e-10
    assert not result["passed"]
    assert result["length_difference_mm"] == pytest.approx(10)


def test_partial_reference_cannot_be_full_profile_pass(tmp_path):
    ref = _profile(tmp_path / "reference_scored_only.dxf")
    result = compare_dxf(ref, ref)
    assert result["scope_geometry_match"]
    assert result["scope"] == "partial_reference"
    assert not result["passed"]


def test_unsupported_manual_and_missing_rows_remain_in_denominator():
    catalog = {"cases": [
        {"id": "cal", "split": "calibration", "supported_template": "a"},
        {"id": "unsupported", "split": "holdout", "supported_template": None},
        {"id": "missing", "split": "holdout", "supported_template": None},
    ]}
    rows = [{"case_id": "cal", "status": "completed", "provider": {"attempted": True, "transport_success": True}, "manual_confirmation": True, "automatic_success": True, "validation": {"passed": True}, "comparison": {"passed": True}}, {"case_id": "unsupported", "status": "unsupported"}]
    result = summarize_evaluation(catalog, rows)
    assert result["overall"]["total"] == 3
    assert result["overall"]["automatic_success_rate"] == 0
    assert result["overall"]["geometry_valid_rate"] == pytest.approx(1 / 3)
    assert result["overall"]["online_transport_success_rate"] == 1
    assert result["holdout"]["total"] == 2
    assert result["holdout"]["not_evaluated"] == 1
    assert result["holdout"]["reference_geometry_pass"] == 0


def test_repeated_results_cannot_inflate_denominator():
    catalog = {"cases": [{"id": "a", "split": "holdout", "supported_template": None}]}
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_evaluation(catalog, [{"case_id": "a"}, {"case_id": "a"}])


def test_all_online_trials_and_retries_remain_in_transport_totals():
    protocols = [
        {"status": "failed", "http_success": True, "schema_success": False, "network_requests": 1},
        {"status": "succeeded", "http_success": True, "schema_success": True, "network_requests": 2},
    ]
    runs = [{"review_snapshot": {"provider": provider}} for provider in (
        {"status": "failed", "http_success": False, "schema_success": False, "network_requests": 1},
        {"status": "succeeded", "http_success": True, "schema_success": True, "network_requests": 1},
    )]
    result = summarize_online_trials(protocols, runs, online_requested=True)
    assert result["logical_calls"]["total"] == 4
    assert result["logical_calls"]["protocol_trials"] == 2
    assert result["logical_calls"]["calibration_trials"] == 2
    assert result["logical_calls"]["http_success_rate"] == .75
    assert result["logical_calls"]["schema_success_rate"] == .5
    assert result["network_attempts"]["total"] == 5
    assert result["network_attempts"]["retries"] == 1
    assert result["network_attempts"]["logical_calls_with_attempts"] == 4


def test_missing_historical_receipts_are_not_assumed_success_or_zero_attempts():
    result = summarize_online_trials([
        {"status": "failed", "code": "invalid_json"},
        {"status": "succeeded", "http_success": True, "schema_success": True, "network_requests": 1},
    ], [], online_requested=True)
    assert result["logical_calls"]["total"] == 2
    assert result["logical_calls"]["http_unknown"] == 1
    assert result["logical_calls"]["http_success_rate"] is None
    assert result["logical_calls"]["schema_success_rate"] is None
    assert result["network_attempts"]["total"] is None
    assert result["network_attempts"]["known_total"] == 1
    assert result["network_attempts"]["receipts_missing_attempt_count"] == 1


def test_offline_and_pre_network_errors_do_not_inflate_network_attempts():
    calls = [{"status": "failed", "code": "not_configured", "http_success": False,
              "schema_success": False, "network_requests": 0}]
    result = summarize_online_trials(calls, [], online_requested=True)
    assert result["logical_calls"]["total"] == 1
    assert result["logical_calls"]["http_success_rate"] == 0
    assert result["network_attempts"]["total"] == 0
    assert result["network_attempts"]["logical_calls_with_attempts"] == 0
    offline = summarize_online_trials(calls, [{"review_snapshot": {"provider": {"status": "disabled"}}}], online_requested=False)
    assert offline["logical_calls"]["total"] == 0
    assert offline["logical_calls"]["http_success_rate"] is None
    assert offline["network_attempts"]["total"] == 0

"""Reproducible qualification with explicit calibration and assisted scope.

This module alone may read reference DXF. Neither the provider nor the runtime
is given reference geometry. A scripted confirmation is labeled as a fixture,
never counted as human-free or held-out reconstruction success.
"""
from __future__ import annotations
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from .config import Settings
from .dataset import resolve_inside
from .evaluation import compare_dxf, summarize_evaluation, summarize_online_trials
from .geometry import solve_profile, template_schema
from .provider import DimensionProvider, ProviderError
from .service import AgentService

# Text-only fixtures with independent, explicit expected interpretations.
# They contain no case topology, coordinates, or reference geometry.
PROTOCOL_FIXTURES = [
    ({"id": "q1", "text": "⌀274^{+5}_{0}"}, {"kind": "diameter", "nominal": 274, "upper_deviation": 5, "lower_deviation": 0}),
    ({"id": "q2", "text": "R62"}, {"kind": "radius", "nominal": 62, "upper_deviation": None, "lower_deviation": None}),
    ({"id": "q3", "text": "17.5±1"}, {"kind": "length", "nominal": 17.5, "upper_deviation": 1, "lower_deviation": -1}),
    ({"id": "q4", "text": "12°"}, {"kind": "angle", "nominal": 12, "upper_deviation": None, "lower_deviation": None}),
    ({"id": "q5", "text": "R80Ra6.3"}, {"kind": "unknown", "nominal": None, "upper_deviation": None, "lower_deviation": None}),
    ({"id": "q6", "text": "⌀710^{0}_{-10}"}, {"kind": "diameter", "nominal": 710, "upper_deviation": 0, "lower_deviation": -10}),
    ({"id": "q7", "text": "17178"}, {"kind": "length", "nominal": 17178, "upper_deviation": None, "lower_deviation": None}),
]

def qualify(settings: Settings, *, online=False, repeats=1, output: Path | None = None):
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    directory = settings.runtime_root / "qualification" / run_id
    directory.mkdir(parents=True)
    service = AgentService(replace(settings, runtime_root=directory / "agent-runtime"))
    schema = template_schema()
    defaults = {p["id"]: p["default"] for p in schema["parameters"]}
    protocols = []
    runs = []
    try:
        baseline = solve_profile(defaults, directory / "engine-baseline")
        perturbations = []
        for field in schema["parameters"]:
            parameters = dict(defaults)
            parameters[field["id"]] += 1
            result = solve_profile(parameters, directory / "perturbations" / field["id"])
            perturbations.append({"parameter": field["id"], "delta": 1, "passed": bool(result["validation"]["passed"]),
                                  "geometry_changed": result["entities"] != baseline["entities"]})
        if online:
            provider = DimensionProvider(settings)
            for _ in range(repeats):
                try:
                    response = provider.normalize([r for r, expected in PROTOCOL_FIXTURES])
                    by_id = {r["id"]: r for r in response["dimensions"]}
                    correct = sum(all(by_id[raw["id"]].get(k) == v for k, v in expected.items()) for raw, expected in PROTOCOL_FIXTURES)
                    protocols.append({**response, "correct_records": correct, "total_records": len(PROTOCOL_FIXTURES),
                                      "passed": correct == len(PROTOCOL_FIXTURES)})
                except ProviderError as error:
                    protocols.append({"status": "failed", "passed": False, "code": error.code, "message": error.message,
                                      "status_code": error.status_code, "http_success": error.http_success,
                                      "schema_success": False, "network_requests": error.network_requests})
        for _ in range(repeats):
            job = service.create_case(schema["calibration_case"], use_api=online, asynchronous=False)
            review_snapshot = {"status": job["status"], "missing_parameters": [p["id"] for p in job["parameters"] if p["value"] is None],
                               "needs_review": [p["id"] for p in job["parameters"] if p["needs_review"]],
                               "provider": job["provider"]}
            reviewed = [p for p in job["parameters"] if "provider_agrees_with_local" in p]
            review_snapshot["literal_ocr_parse_agreement"] = {
                "correct": sum(p["provider_agrees_with_local"] for p in reviewed), "total": len(reviewed),
                "passed": bool(reviewed) and all(p["provider_agrees_with_local"] for p in reviewed),
                "scope": "Literal OCR parsing agreement only; not verification of printed glyphs or geometry bindings."}
            if job["status"] != "needs_review":
                runs.append({"job_id": job["id"], "passed": False, "review_snapshot": review_snapshot})
                continue
            # Explicit, declared calibration fixture. No claims of OCR recovery for these values.
            confirmed = {p["id"]: p["value"] if p["value"] is not None else defaults[p["id"]] for p in job["parameters"]}
            job = service.confirm(job["id"], confirmed, [a["id"] for a in schema["assumptions"]], True,
                                  asynchronous=False, actor="scripted_calibration_fixture")
            comparison = {"passed": False, "status": "not_available"}
            case = service.cases[schema["calibration_case"]]
            if job.get("artifact_directory") and case.get("gt_dxf"):
                comparison = compare_dxf(Path(job["artifact_directory"]) / "drawing.dxf", resolve_inside(settings.dataset_root, case["gt_dxf"]))
            runs.append({"job_id": job["id"], "status": job["status"], "passed": job["status"] == "completed" and comparison.get("passed", False),
                         "validation": job.get("validation"), "comparison": comparison,
                         "review_snapshot": review_snapshot, "confirmation": job.get("manual_confirmation"),
                         "automatic_completion": False, "split": "calibration"})

        # Deterministically check failure fallback without a network request.
        empty_settings = Settings(dataset_root=settings.dataset_root, runtime_root=directory / "fallback", api_key="")
        fallback_service = AgentService(empty_settings)
        try:
            fallback = fallback_service.create_case(schema["calibration_case"], use_api=True, asynchronous=False)
            fallback_pass = fallback["status"] == "needs_review" and bool(fallback["parameters"]) and fallback["provider"].get("code") == "not_configured"
        finally:
            fallback_service.close()

        rows = []
        for case in service.catalog["cases"]:
            if case["id"] == schema["calibration_case"] and runs:
                last = runs[-1]
                rows.append({"case_id": case["id"], "status": last.get("status", "failed"), "split": "calibration",
                             "provider": {"attempted": bool(online), "transport_success": last["review_snapshot"]["provider"].get("http_success", False)},
                             "validation": last.get("validation", {}), "comparison": last.get("comparison", {}),
                             "manual_confirmation": True, "automatic_success": False})
            else:
                rows.append({"case_id": case["id"], "status": "unsupported", "split": "holdout", "automatic_success": False,
                             "reason": "No registered and validated topology template; not sent to provider."})
        summary = summarize_evaluation(service.catalog, rows)
        geometry_pass = bool(baseline["validation"]["passed"]) and all(p["passed"] and p["geometry_changed"] for p in perturbations)
        assisted_pass = bool(runs) and all(r["passed"] for r in runs)
        online_pass = bool(online and protocols and all(p["passed"] for p in protocols) and
                           all(r["review_snapshot"]["provider"].get("status") == "succeeded" and
                               r["review_snapshot"]["literal_ocr_parse_agreement"]["passed"] for r in runs))
        report = {
            "schema_version": "1.0", "run_id": run_id, "created_at": timestamp.isoformat(),
            "protocol": "template-assisted-qualification-v1", "online_requested": online, "repeats": repeats,
            "model": settings.model, "base_url": settings.base_url,
            "qualification": {"geometry_engine_passed": geometry_pass, "assisted_calibration_workflow_passed": assisted_pass,
                              "api_failure_fallback_passed": fallback_pass, "online_dimension_protocol_passed": online_pass if online else None,
                              "online_assisted_scope_passed": online_pass and geometry_pass and assisted_pass and fallback_pass if online else None,
                              "dataset_fully_automatic_passed": False, "held_out_reconstruction_validated": False},
            "coverage": {"total": len(rows), "supported": sum(bool(c.get("supported_template")) for c in service.catalog["cases"]),
                         "unsupported": sum(r["status"] == "unsupported" for r in rows), "automatic_completed": 0,
                         "assisted_calibration_completed": sum(r["status"] == "completed" for r in rows), "held_out_supported": 0},
            "summary": summary, "protocol_trials": protocols, "calibration_runs": runs, "parameter_perturbations": perturbations,
            "online_request_summary": summarize_online_trials(protocols, runs, online_requested=online),
            "cases": rows,
            "limitations": ["293 is the declared template calibration case, not a held-out accuracy measurement.",
                            "Missing OCR values and shape priors are supplied by a labeled scripted calibration fixture.",
                            "Seven tangent directions and one unlabeled transition radius are calibration priors; LM tread is simplified.",
                            "The 49 unsupported cases remain in the dataset denominator. No full-dataset automatic pass is claimed.",
                            "GT is read only by the independent scorer; no GT coordinates, overlays or hidden parameters enter provider input.",
                            "Numerical consistency does not certify manufacturing suitability or all annotated dimensions."]}
        destination = output or settings.runtime_root / "evaluations" / f"{run_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return report, destination
    finally:
        service.close()

"""Autonomous source-image batch generation followed by independent scoring.

No scripted parameter confirmations, template defaults, or reference feedback
are supplied to the generator. Every repeat and every catalog case is retained.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
import zipfile
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .autonomous_evaluation import evaluate_autonomous_artifact, summarize_autonomous_evaluation
from .dataset import resolve_inside
from .service import AgentService

MAX_REFERENCE_BYTES = 20_000_000


def _reference_for_scoring(settings, case, directory):
    """Called only after export. Copy exact archive bytes; never repair a DXF."""
    if case.get("gt_dxf"):
        path = resolve_inside(settings.dataset_root, case["gt_dxf"])
        if path.suffix.lower() != ".dxf" or path.stat().st_size > MAX_REFERENCE_BYTES:
            raise ValueError("Invalid or oversized reference DXF")
        with path.open("rb") as stream:
            raw = stream.read(MAX_REFERENCE_BYTES+1)
        source = {"kind":"direct_dxf","path":case["gt_dxf"],"member":None}
    elif isinstance(case.get("gt_archive"), dict):
        declared = case["gt_archive"]
        archive_path = resolve_inside(settings.dataset_root, declared["path"])
        member = declared["member"]
        if not isinstance(member,str) or "\x00" in member or ":" in member:
            raise ValueError("Unsafe reference archive member")
        normalized = PurePosixPath(member.replace("\\","/"))
        if normalized.is_absolute() or ".." in normalized.parts or normalized.suffix.lower() != ".dxf":
            raise ValueError("Unsafe reference archive member")
        with zipfile.ZipFile(archive_path) as package:
            matches = [item for item in package.infolist() if item.filename == member]
            if len(matches) != 1 or matches[0].is_dir() or matches[0].file_size > MAX_REFERENCE_BYTES:
                raise ValueError("Missing, ambiguous or oversized reference archive member")
            with package.open(matches[0]) as stream:
                raw = stream.read(MAX_REFERENCE_BYTES+1)
        case_id = case["id"]
        if not isinstance(case_id,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,179}",case_id):
            raise ValueError("Unsafe reference case id")
        # The unchanged scorer uses these original filename scope markers.
        # Retain them so extracting a partial reference cannot widen its claim.
        scope_suffix = "".join("__"+token for token in ("scored_only","strict_body") if token in normalized.stem.lower())
        path = resolve_inside(directory, f"reference-only/{case_id}{scope_suffix}.dxf")
        if len(raw) > MAX_REFERENCE_BYTES:
            raise ValueError("Reference archive member exceeds decompression limit")
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("Reference changed between repeated trials")
        if not path.exists(): path.write_bytes(raw)
        source = {"kind":"zip_member","path":declared["path"],"member":member}
    else:
        return None, {"reference_source":None,"reference_sha256":None}
    if len(raw) > MAX_REFERENCE_BYTES:
        raise ValueError("Reference exceeds byte limit")
    return path, {"reference_source":source,"reference_sha256":hashlib.sha256(raw).hexdigest(),
                  "reference_byte_count":len(raw),"reference_modified":False}


def _digest(value):
    return value.lower() if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) else None


def _exposure_context(settings, requested, metadata=None):
    """Read reporting evidence only; never instantiate a model or open labels."""
    context = {"requested": bool(requested), "rows": {}, "status": "not_applicable",
               "checkpoint_sha256": None, "training_manifest_sha256": None}
    if not requested:
        return context
    context["status"] = "unknown"
    try:
        if metadata is None:
            import torch
            checkpoint = Path(settings.segmentation_checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            metadata = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "training_provenance": state.get("provenance")}
            del state
        provenance = metadata.get("training_provenance")
        context["checkpoint_sha256"] = _digest(metadata.get("checkpoint_sha256"))
        if not isinstance(provenance, dict):
            return context
        manifest_hash = _digest(provenance.get("manifest_sha256"))
        context["training_manifest_sha256"] = manifest_hash
        consumed = provenance.get("consumed_artifacts", provenance.get("consumed_cases"))
        identities = provenance.get("case_identities")
        source = "checkpoint_provenance"
        records = identities if isinstance(identities, list) else consumed
        if not isinstance(records, list) or not records:
            manifest_path = getattr(settings, "segmentation_manifest", "")
            if not manifest_path or not manifest_hash:
                return context
            raw = Path(manifest_path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest_hash:
                context["status"] = "training_manifest_hash_mismatch"
                return context
            document = json.loads(raw)
            records = document if isinstance(document, list) else next((document[k] for k in ("cases", "rows", "samples") if isinstance(document.get(k), list)), None)
            consumed = [row for row in records or [] if isinstance(row, dict) and row.get("trainable") is not False and row.get("image") and row.get("mask")]
            source = "checkpoint_hash_verified_training_manifest"
        if not isinstance(records, list):
            return context
        consumed_ids = {row.get("id", row.get("case_id")) for row in consumed if isinstance(row, dict)} if isinstance(consumed, list) else None
        indexed = {}
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("Malformed training identity")
            case_id = row.get("id", row.get("case_id"))
            if not isinstance(case_id, str) or case_id in indexed or row.get("split") not in {"train", "val", "test"}:
                raise ValueError("Ambiguous training identity")
            indexed[case_id] = {"split": row["split"], "source_image_sha256": _digest(row.get("source_image_sha256", row.get("image_sha256"))),
                                "consumed": case_id in consumed_ids if consumed_ids is not None else None}
        if isinstance(consumed,list):
            seen_consumed=set()
            for row in consumed:
                if not isinstance(row,dict): raise ValueError("Malformed consumed identity")
                case_id=row.get("id",row.get("case_id")); declared=indexed.get(case_id)
                if not declared or case_id in seen_consumed or row.get("split") != declared["split"] or _digest(row.get("source_image_sha256",row.get("image_sha256"))) != declared["source_image_sha256"]:
                    raise ValueError("Conflicting consumed identity")
                seen_consumed.add(case_id)
        context.update(rows=indexed, status="available", evidence_source=source)
    except Exception:
        # Missing/legacy/corrupt reporting evidence is unknown, never unseen.
        context.update(rows={}, status="unavailable_checkpoint_or_provenance")
    return context


def _model_exposure(case_id, image_sha256, context):
    row = context["rows"].get(case_id)
    result = {"split": "unknown" if context["requested"] else "not_applicable", "role": "unknown" if context["requested"] else "segmentation_not_requested",
              "identity_verified": False, "optimization_exposed": None, "checkpoint_selection_exposed": None,
              "development_only": True, "blind_test": False, "catalog_split_is_model_split": False,
              "checkpoint_sha256": context["checkpoint_sha256"], "training_manifest_sha256": context["training_manifest_sha256"],
              "evidence_source": context.get("evidence_source"), "evidence_status": context["status"]}
    if not row:
        return result
    result["declared_split"] = row["split"]
    if not _digest(image_sha256) or not row["source_image_sha256"]:
        result["evidence_status"] = "source_hash_unavailable"
        return result
    if _digest(image_sha256) != row["source_image_sha256"]:
        result["evidence_status"] = "source_hash_mismatch"
        return result
    result.update(split=row["split"], identity_verified=True, usable_training_label=row["consumed"])
    if row["consumed"] is False:
        result.update(role="excluded_label_in_development_split", optimization_exposed=False, checkpoint_selection_exposed=False)
    elif row["consumed"] is None:
        result["role"] = "declared_development_split_consumption_unknown"
    else:
        result.update(role={"train": "optimization_training", "val": "checkpoint_selection", "test": "development_test"}[row["split"]],
                      optimization_exposed=row["split"] == "train", checkpoint_selection_exposed=row["split"] == "val")
    return result


def _source_digest(settings, case):
    try:
        return hashlib.sha256(resolve_inside(settings.dataset_root, case["image"]).read_bytes()).hexdigest()
    except (KeyError, OSError, ValueError, TypeError):
        return None


def _transport(trials, online):
    called = [trial.get("provider") or {} for trial in trials if online and trial.get("provider_called")]
    http_success = sum(p.get("http_success") is True for p in called)
    schema_success = sum(p.get("schema_success") is True for p in called)
    http_unknown = sum(not isinstance(p.get("http_success"), bool) for p in called)
    schema_unknown = sum(not isinstance(p.get("schema_success"), bool) for p in called)
    attempts = [p["network_requests"] for p in called if isinstance(p.get("network_requests"), int) and not isinstance(p["network_requests"], bool) and p["network_requests"] >= 0]
    missing = len(called) - len(attempts)
    return {
        "requested_pipeline_trials": len(trials) if online else 0,
        "logical_calls": {"total": len(called), "autonomous_trials": len(called),
            "http_successful": http_success, "http_unknown": http_unknown,
            "http_success_rate": http_success / len(called) if called and not http_unknown else None,
            "schema_successful": schema_success, "schema_unknown": schema_unknown,
            "schema_success_rate": schema_success / len(called) if called and not schema_unknown else None},
        "network_attempts": {"total": sum(attempts) if not missing else None,
            "known_total": sum(attempts), "receipts_missing_attempt_count": missing,
            "logical_calls_with_attempts": sum(n > 0 for n in attempts),
            "retries": sum(max(0, n - 1) for n in attempts) if not missing else None},
        "vision_verdicts": {verdict: sum(p.get("verdict") == verdict for p in called) for verdict in ("match", "mismatch", "uncertain")},
        "scope": "All actual vision-provider invocations across every selected case and repeat. Generation failures before provider invocation remain pipeline failures but are not invented network calls.",
        "attempt_policy": "HTTP/schema rates are per logical provider invocation. network_attempts counts actual requests including retries; unknown receipt counts remain null.",
    }


def _case_row(case, trials, expected_repeats, exposure):
    if not trials:
        return {"case_id": case["id"], "split": case.get("split", "holdout"),
                "status": "not_attempted", "attempted": False, "manual_intervention": False,
                "artifact_completed": False, "automatic_completion": False, "geometry_valid": False,
                "scaled_mm": False, "comparison": {}, "trial_count": 0,
                "legacy_calibration": case.get("split") == "calibration",
                "legacy_catalog_split": case.get("split", "holdout"), "model_exposure": exposure}
    last = trials[-1]
    complete_repeats = len(trials) == expected_repeats
    comparison = dict(last.get("comparison") or {})
    comparison["reference_compared"] = any(t.get("comparison", {}).get("reference_compared") is True for t in trials)
    comparison["reference_within_0_1mm"] = complete_repeats and all(t.get("comparison", {}).get("reference_within_0_1mm") is True for t in trials)
    return {
        "case_id": case["id"], "split": case.get("split", "holdout"),
        "legacy_catalog_split": case.get("split", "holdout"), "model_exposure": last["model_exposure"],
        "legacy_calibration": case.get("split") == "calibration", "attempted": True,
        "status": "completed" if complete_repeats and all(t["status"] == "completed" for t in trials) else "failed" if complete_repeats else "running",
        "artifact_completed": complete_repeats and all(t["artifact_completed"] for t in trials),
        "automatic_completion": complete_repeats and all(t["automatic_completion"] for t in trials),
        "geometry_valid": complete_repeats and all(t["geometry_valid"] for t in trials),
        "scaled_mm": complete_repeats and all(t["scaled_mm"] for t in trials),
        "manual_intervention": any(t["manual_intervention"] for t in trials),
        "strict_scope_passed": complete_repeats and all(t["strict_scope_passed"] for t in trials),
        "comparison": comparison, "comparison_repeat_policy": "Strict pass requires every planned repeat; detailed metrics below are from the last repeat, and all individual metrics remain in trials.",
        "provider": last.get("provider", {}), "validation": last.get("validation", {}),
        "reference_source": last.get("reference_source"), "reference_sha256": last.get("reference_sha256"),
        "artifacts": last.get("artifacts", {}), "job_id": last.get("job_id"),
        "trial_count": len(trials), "planned_repeats": expected_repeats,
        "issues": last.get("issues", []), "elapsed_seconds": round(sum(t["elapsed_seconds"] for t in trials), 3),
    }


def qualify_autonomous(settings, online=False, repeats=1, cases=None, *, use_segmentation=False):
    """Return ``(report, report_path)`` and persist after every completed trial.

    ``cases=None`` selects all catalog cases; an explicit case-id list selects a
    subset while the complete catalog remains in every report denominator.
    ``online=True`` authorizes the existing service's bounded vision inspection.
    The scorer is invoked only after each source-only prediction has completed.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 10:
        raise ValueError("repeats must be an integer between 1 and 10")
    if cases is not None and (not isinstance(cases, (list, tuple)) or not cases or any(not isinstance(c, str) for c in cases)):
        raise ValueError("cases must be a non-empty list of case ids, or None for all cases")
    if cases is not None and len(set(cases)) != len(cases):
        raise ValueError("Duplicate selected case ids are not allowed")
    started = time.monotonic()
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    directory = Path(settings.runtime_root) / "autonomous-qualification" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    destination = Path(settings.runtime_root) / "evaluations" / f"{run_id}_autonomous.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    service = AgentService(replace(settings, runtime_root=directory / "agent-runtime"))
    trials = []
    report = {}
    try:
        catalog = service.catalog
        ids = {case["id"] for case in catalog["cases"]}
        selected = list(cases) if cases is not None else [case["id"] for case in catalog["cases"]]
        if set(selected) - ids:
            raise ValueError("Unknown selected case ids: " + ", ".join(sorted(set(selected) - ids)))
        by_id = {case["id"]: case for case in catalog["cases"]}
        frozen = json.dumps(catalog, sort_keys=True, ensure_ascii=False).encode("utf8")
        exposure_context = _exposure_context(settings, use_segmentation)
        source_hashes = {case["id"]: _source_digest(settings, case) for case in catalog["cases"]} if use_segmentation else {}
        exposures = {case["id"]: _model_exposure(case["id"], source_hashes.get(case["id"]), exposure_context) for case in catalog["cases"]}

        def persist(run_status):
            rows = [_case_row(case, [t for t in trials if t["case_id"] == case["id"]], repeats, exposures[case["id"]]) for case in catalog["cases"]]
            all_trials_done = len(trials) == len(selected) * repeats
            strict_pass = all_trials_done and bool(trials) and all(t["strict_scope_passed"] for t in trials)
            payload = {
                "schema_version": "1.0", "run_id": run_id, "created_at": timestamp.isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(), "status": run_status,
                "mode": "autonomous_image", "protocol": "autonomous-image-qualification-v1",
                "online_requested": bool(online), "repeats": repeats, "model": settings.model,
                "segmentation_requested": bool(use_segmentation),
                "evaluation_scope": "previously_exposed_development_dataset", "blind_test": False,
                "legacy_catalog_split_meaning": "Compatibility-only template calibration/holdout allocation; never evidence that a learned model has not used the image.",
                "model_exposure_summary": {
                    "case_split_counts": dict(Counter(row["model_exposure"]["split"] for row in rows)),
                    "trial_split_counts": dict(Counter(trial["model_exposure"]["split"] for trial in trials)),
                    "trial_role_counts": dict(Counter(trial["model_exposure"]["role"] for trial in trials)),
                    "identity_verified_trials": sum(trial["model_exposure"]["identity_verified"] for trial in trials),
                    "development_only": True, "blind_test": False},
                "selected_cases": selected, "selected_count": len(selected),
                "catalog_count": len(catalog["cases"]), "catalog_sha256": hashlib.sha256(frozen).hexdigest(),
                "completed_trials": len(trials), "planned_trials": len(selected) * repeats,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "qualification": {
                    "strict_requested_scope_passed": strict_pass,
                    "strict_whole_dataset_passed": strict_pass and len(selected) == len(catalog["cases"]),
                    "requested_scope_complete": all_trials_done,
                    "engineering_verified": False,
                    "strict_gate": "Every selected case and planned repeat must have a completed autonomous artifact, valid geometry, declared physical scale, no manual intervention, and independent 0.1 mm reference agreement; online runs additionally require successful HTTP/schema vision receipt and a match verdict for the candidate overlay.",
                },
                "summary": summarize_autonomous_evaluation(catalog, rows), "cases": rows,
                "trials": trials, "online_request_summary": _transport(trials, online),
                "limitations": [
                    "Artifact export and geometric validity do not establish reference accuracy or compliance with all dimensions.",
                    "Independent default scoring searches D4 orientation plus translation without scale fitting; resulting alignment is a shape diagnostic, not engineering approval.",
                    "Reference files can contain simplified tread/closure and fitted portions; missing or partial references cannot establish a strict full-profile pass.",
                    "All 50 source images have been exposed during development. Legacy catalog holdout is a template label, not a learned-model holdout or a blind test; model_exposure records optimization/selection/development-test use independently.",
                    "All catalog cases remain in the denominator even for a selected subset. Repeats are not cherry-picked; every planned repeat must pass for per-case strict success.",
                    "Reference geometry is read by the scorer only after export and is never sent back to the generator or provider.",
                ],
            }
            serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            if settings.api_key:
                serialized = serialized.replace(settings.api_key, "[REDACTED]")
            serialized = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", serialized)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(serialized, encoding="utf8")
            temporary.replace(destination)
            (directory / "report.json").write_text(serialized, encoding="utf8")
            return json.loads(serialized)

        report = persist("running")
        for case_id in selected:
            case = by_id[case_id]
            for repeat in range(1, repeats + 1):
                trial_start = time.monotonic()
                trial = {"case_id": case_id, "repeat": repeat, "split": case.get("split", "holdout"),
                         "legacy_catalog_split": case.get("split", "holdout"), "model_exposure": exposures[case_id],
                         "legacy_calibration": case.get("split") == "calibration", "attempted": True,
                         "status": "failed", "artifact_completed": False, "automatic_completion": False,
                         "geometry_valid": False, "scaled_mm": False, "manual_intervention": False,
                         "provider_called": False, "provider": {"status": "not_invoked", "network_requests": 0},
                         "comparison": {}, "strict_scope_passed": False, "issues": []}
                try:
                    # No reference path, reference coordinate or earlier score is
                    # passed to this source-only runtime method.
                    options = {"use_segmentation": True} if use_segmentation else {}
                    job = service.create_auto_case(case_id, use_api=bool(online), asynchronous=False, **options)
                    validation = job.get("validation") or {}
                    provider = job.get("provider") or {}
                    manual = bool(job.get("manual_intervention") or job.get("manual_confirmation"))
                    artifact_dir = Path(job["artifact_directory"]) if job.get("artifact_directory") else None
                    prediction = artifact_dir / "drawing.dxf" if artifact_dir else None
                    artifact_ok = bool(prediction and prediction.is_file())
                    completed = job.get("status") == "completed" and artifact_ok and job.get("automatic_completion") is True and not manual
                    called = bool(online and provider.get("status") not in {None, "pending", "disabled", "not_invoked"})
                    artifacts = {}
                    if artifact_dir:
                        for name in ("drawing.dxf", "preview.svg", "model.json", "validation.json", "overlay.png", "dimension-evidence.json",
                                     "segmentation-mask.png", "segmentation-overlay.png", "segmentation.json", "segmentation-refinement.json",
                                     "raw-segmentation-mask.png", "raw-segmentation-overlay.png",
                                     "topology.json", "topology-overlay.png", "topology-candidates.json", "topology-plan.json",
                                     "correction-evidence.json", "constraint-bindings.json",
                                     "binding-candidates.json", "binding-topology.png", "parametric-stage.json", "parametric-solution.json",
                                     "baseline-drawing.dxf", "baseline-preview.svg", "baseline-overlay.png", "baseline-model.json"):
                            path = artifact_dir / name
                            if path.is_file():
                                artifacts[name] = str(path.resolve())
                    # Preserve generation and actual provider receipts before
                    # reference scoring: a scorer failure cannot erase calls.
                    trial.update(job_id=job.get("id"), status=job.get("status", "failed"),
                                 artifact_completed=artifact_ok, automatic_completion=completed,
                                 manual_intervention=manual, validation=validation, provider=provider,
                                 provider_called=called, artifacts=artifacts,
                                 parameterization=job.get("parameterization"), dimension_analysis=job.get("dimension_analysis"),
                                 issues=job.get("issues", []), scale=job.get("scale"), source=job.get("source"))
                    trial["segmentation_model"] = (job.get("extraction") or {}).get("model")
                    actual_context = _exposure_context(settings, True, trial["segmentation_model"]) if use_segmentation and isinstance(trial["segmentation_model"], dict) else exposure_context
                    actual_hash = _digest((job.get("source") or {}).get("image_sha256")) or source_hashes.get(case_id)
                    trial["model_exposure"] = _model_exposure(case_id, actual_hash, actual_context)
                    comparison = {}
                    if artifact_ok:
                        reference, reference_receipt = _reference_for_scoring(settings, case, directory)
                        trial.update(reference_receipt)
                        comparison = evaluate_autonomous_artifact(prediction, reference)
                    valid = validation.get("passed") is True and comparison.get("geometry_valid") is True
                    scaled = validation.get("scaled_mm") is True and comparison.get("scaled_mm") is True
                    vision_pass = bool(called and provider.get("http_success") is True and provider.get("schema_success") is True and provider.get("verdict") == "match" and provider.get("overlay_sent") is True and provider.get("ground_truth_sent") is False)
                    strict = completed and valid and scaled and comparison.get("reference_within_0_1mm") is True and (not online or vision_pass)
                    trial.update(geometry_valid=valid, scaled_mm=scaled,
                                 online_vision_passed=vision_pass if online else None,
                                 comparison=comparison, strict_scope_passed=bool(strict))
                except Exception as error:
                    # Never serialize arbitrary exception/provider echoes or keys.
                    trial["issues"] = [*trial.get("issues", []), f"Autonomous trial could not complete ({type(error).__name__}); earlier stage artifacts remain in its isolated runtime."]
                    if trial["artifact_completed"]:
                        trial["comparison"] = {"status": "evaluation_error", "reference_compared": False,
                                               "reference_within_0_1mm": False}
                trial["elapsed_seconds"] = round(time.monotonic() - trial_start, 3)
                trials.append(trial)
                report = persist("running")
        report = persist("completed")
        return report, destination
    finally:
        service.close()

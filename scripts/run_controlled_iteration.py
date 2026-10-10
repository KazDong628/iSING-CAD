"""Replay frozen source/OCR/oracle-mask through an explicitly selected code root.

No GT DXF is opened by prediction. Optional evaluation runs the preserved main
project comparator only after prediction bytes have been frozen. Both baseline
and candidate are exposed development experiments, never held-out evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf8"))
    if not isinstance(value, dict):
        raise ValueError("expected_json_object")
    return value


def code_snapshot(root):
    root = Path(root).resolve(strict=True)
    files = sorted((root / "contour_agent").rglob("*.py"))
    files += [root / "scripts" / "evaluate_oracle_mask_run.py"]
    files += [p for p in (root / "pyproject.toml", root / "requirements.txt") if p.is_file()]
    return {p.relative_to(root).as_posix(): digest(p) for p in files}


def select_code_root(root):
    root = Path(root).resolve(strict=True)
    if not (root / "contour_agent" / "automatic.py").is_file():
        raise ValueError("invalid_code_root")
    if any(name == "contour_agent" or name.startswith("contour_agent.") for name in sys.modules):
        raise RuntimeError("project_already_imported_use_fresh_process")
    sys.path.insert(0, str(root))
    module = importlib.import_module("contour_agent.automatic")
    if Path(module.__file__).resolve() != root / "contour_agent" / "automatic.py":
        raise RuntimeError("unexpected_import_root")
    return root


def imported_sources(root):
    result = {}
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("contour_agent"):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_relative_to(root):
            raise RuntimeError("mixed_project_import_roots")
        result[name] = {"path": str(path), "sha256": digest(path)}
    return result


def freeze_inputs(source_run, output):
    source = Path(source_run).resolve(strict=True) / "inputs"
    receipt = read(source / "input-manifest.json")
    if (receipt.get("schema_version") != "oracle-mask-input-v1" or
            receipt.get("oracle_mask") is not True or receipt.get("held_out") is not False or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", str(receipt.get("case_id", "")))):
        raise ValueError("invalid_frozen_source_receipt")
    output = Path(output)
    output.mkdir()
    paths = {}
    for field, hash_field, allowed in (
        ("source_image", "source_image_sha256", {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}),
        ("source_ocr", "source_ocr_sha256", {".json"}),
        ("mask", "mask_sha256", {".png"}),
    ):
        declared = Path(receipt[field])
        original = (source / declared.name).resolve(strict=True)
        expected = receipt.get(hash_field)
        if (original.parent != source or original.suffix.lower() not in allowed or
                not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or digest(original) != expected):
            raise ValueError("frozen_source_hash_or_path_mismatch")
        name = "source-image" + original.suffix.lower() if field == "source_image" else (
            "source-ocr.json" if field == "source_ocr" else "oracle-mask.png")
        target = output / name
        shutil.copyfile(original, target)
        if digest(target) != expected:
            raise ValueError("frozen_copy_hash_mismatch")
        paths[field] = target
    # Pixel checks do not use registration transforms or any GT geometry.
    from PIL import Image
    import numpy as np
    with Image.open(paths["source_image"]) as image:
        image_size = {"width": image.width, "height": image.height}
    with Image.open(paths["mask"]) as image:
        mask_size = {"width": image.width, "height": image.height}
        mask = np.asarray(image.convert("L"))
    if (image_size != receipt.get("original_image_size") or mask_size != receipt.get("mask_size") or
            not np.isin(mask, [0, 255]).all() or
            int(np.count_nonzero(mask)) != receipt.get("mask_geometry", {}).get("foreground_pixels")):
        raise ValueError("frozen_pixel_contract_mismatch")
    safe_keys = ("schema_version", "case_id", "oracle_mask", "held_out", "calibration_or_development",
                 "source_split_label", "source_image_sha256", "source_ocr_sha256", "mask_sha256",
                 "reference_dxf_sha256", "original_image_size", "mask_size", "mask_geometry")
    copied = {key: receipt[key] for key in safe_keys if key in receipt}
    copied.update({key: str(value) for key, value in paths.items()})
    copied["source_receipt_sha256"] = digest(source / "input-manifest.json")
    copied["ground_truth_geometry_sent_to_provider"] = False
    write(output / "input-manifest.json", copied)
    return paths, copied


def providers(config_root, online, profile_id, timeout):
    if not online:
        return {}, {"online_requested": False, "network_requests": 0}
    from contour_agent.config import Settings, load_local_env, provider_registry
    from contour_agent.provider import DimensionProvider
    from contour_agent.binding_provider import BindingProvider
    from contour_agent.planning_provider import PlanningProvider
    from contour_agent.topology_edit_provider import TopologyEditProvider, TopologyEvaluationProvider
    for name in (".env.local", ".env"):
        load_local_env(Path(config_root) / name)
    base = Settings(api_timeout=timeout)
    registry, default_id = provider_registry(base)
    chosen = profile_id or default_id
    if chosen not in registry:
        raise ValueError("unknown_provider_profile")
    profile = registry[chosen]
    settings = profile.settings(base)
    if not settings.api_key:
        raise ValueError("provider_not_configured")
    return {
        "dimension_provider": DimensionProvider(settings, max_attempts=1),
        "provider": BindingProvider(settings), "planner_provider": PlanningProvider(settings),
        "editor_provider": TopologyEditProvider(settings), "evaluator_provider": TopologyEvaluationProvider(settings),
    }, {"online_requested": True, "profile_id": chosen, "model": settings.model,
        "wire_api": settings.wire_api, "request_timeout_seconds": settings.api_timeout}


def local_export_audit(after):
    """Check current published bytes, never a stale candidate's success flag."""
    import ezdxf
    from contour_agent.dxf_comparison import audit_dxf
    from contour_agent.radius_contract import exact_radius_checks
    path = after / "drawing.dxf"
    if not path.is_file():
        return {"status": "missing_prediction", "radius_contract_complete": False}
    audit = audit_dxf(path)
    model = read(after / "model.json")
    published = model.get("parameterization") or {}
    # Published model carries the constraints actually used for its export;
    # parametric-stage may describe a later failed candidate and is not a ledger.
    constraints = published.get("constraints", [])
    exact = exact_radius_checks(model.get("entities", []), constraints, dxf_document=ezdxf.readfile(path))
    contract = (model.get("validation") or {}).get("annotation_radius_contract") or {}
    return {"status": "read_back", "prediction_sha256": audit["sha256"], "model_sha256": digest(after / "model.json"),
            "native_object_count": audit["raw_modelspace"]["count"], "native_types": audit["raw_modelspace"]["types"],
            "filtered_count": audit["filtered_profile"]["count"], "units": audit["source_units"],
            "profile_issues": audit["filtered_profile"]["issues"], "closed": audit["connections"]["closed"],
            "component_count": len(audit["connections"]["components"]), "exact_published_radii": exact,
            "exact_radius_count": sum(row["passed"] for row in exact["checks"]),
            "radius_contract_complete": bool(exact["required_count"] and exact["passed"] and contract.get("satisfied")),
            "published_constraints_count": len(constraints), "published_constraint_kinds": dict(Counter(c.get("kind") for c in constraints)),
            "reference_geometry_used": False,
            "scope": "Exact exported source-linked R subset; zero radius constraints is not complete annotation coverage."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--provider-profile", default="9ecode-gpt-5.6-sol")
    parser.add_argument("--api-timeout", type=float, default=180.)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    if not 5 <= args.api_timeout <= 600:
        parser.error("api-timeout must be between 5 and 600 seconds")
    run = args.output.resolve()
    main_root = args.config_root.resolve(strict=True)
    if args.evaluate:
        evaluation_root = (args.evaluation_root or main_root).resolve(strict=True)
        if not (evaluation_root / "__dataset").is_dir():
            parser.error("evaluation-root must contain __dataset before prediction starts")
        if not (evaluation_root / "scripts" / "evaluate_oracle_mask_run.py").is_file():
            parser.error("evaluation-root must contain scripts/evaluate_oracle_mask_run.py")
    if not run.is_relative_to(main_root / "runtime" / "cad-controlled-20261008"):
        parser.error("output must be inside config-root/runtime/cad-controlled-20261008")
    if run.exists():
        parser.error("choose a fresh output directory; old evidence is immutable")
    run.mkdir(parents=True)
    state = {"schema_version": "controlled-cad-iteration-v1", "label": args.label, "created_at": stamp(),
             "oracle_mask_conditioned": True, "held_out": False, "calibration_or_development": True,
             "ground_truth_coordinates_sent_to_provider": False, "ground_truth_dxf_used_as_prediction_geometry": False,
             "status": "initializing", "stage_history": [], "evaluation_requested": args.evaluate,
             "frozen_strict_evaluation_tolerance_mm": .1, "additional_evaluation_tolerance_mm": 1.0}
    def checkpoint(stage, message=""):
        state["last_stage"] = str(stage)
        state["stage_history"].append({"stage": str(stage), "at": stamp()})
        write(run / "run-manifest.json", state)
        print(str(stage), flush=True)
    checkpoint("initializing")
    try:
        before = code_snapshot(args.code_root)
        root = select_code_root(args.code_root)
        state["code"] = {"root": str(root), "sha256_before": before, "runner_path": str(Path(__file__).resolve()),
                         "runner_sha256": digest(__file__), "python_version": sys.version.split()[0]}
        inputs, receipt = freeze_inputs(args.source_run, run / "inputs")
        state.update(case_id=receipt["case_id"], source_gt_sha256=receipt.get("reference_dxf_sha256"),
                     source_image_sha256=receipt["source_image_sha256"], source_ocr_sha256=receipt["source_ocr_sha256"],
                     oracle_mask_sha256=receipt["mask_sha256"])
        checkpoint("inputs_frozen")
        bundle, settings_receipt = providers(main_root, args.online, args.provider_profile, args.api_timeout)
        state["provider"] = settings_receipt
        dimension_provider = bundle.pop("dimension_provider", None)
        from contour_agent.automatic import build_automatic
        from contour_agent.dataset import read_ocr
        from contour_agent.dimension_analysis import analyze_dimensions
        from contour_agent.parametric_pipeline import refine_parametric
        document = read_ocr(inputs["source_ocr"])
        after = run / "after"
        state["status"] = "running"
        checkpoint("initial_cad")
        model = build_automatic(inputs["source_image"], document, after, segmentation_mask=inputs["mask"],
                                segmentation_review={"status": "oracle_mask", "source": "frozen_gt_raster_only", "reviewed": False},
                                progress=checkpoint)
        checkpoint("dimension_analysis")
        dimensions = analyze_dimensions(document, model, provider=dimension_provider, use_api=args.online)
        write(after / "dimension-analysis.json", dimensions)
        checkpoint("topology_and_constraints")
        model, stage = refine_parametric(inputs["source_image"], document, model, after,
                                         use_api=args.online, progress=checkpoint, **bundle)
        state["parameterization_status"] = stage.get("status")
        state["parameterization_accepted"] = stage.get("accepted") is True
        checkpoint("current_export_audit")
        write(run / "current-export-audit.json", local_export_audit(after))
        state["prediction_sha256"] = digest(after / "drawing.dxf") if (after / "drawing.dxf").is_file() else None
        state["frozen_artifacts"] = {p.name: digest(p) for p in after.iterdir() if p.is_file()}
        state["code"]["imported_sources"] = imported_sources(root)
        state["code"]["sha256_after"] = code_snapshot(root)
        state["code"]["unchanged"] = state["code"]["sha256_after"] == before
        if not state["code"]["unchanged"]:
            raise RuntimeError("code_changed_during_prediction")
        state["prediction_frozen_at"] = stamp()
        state["status"] = "prediction_frozen"
        checkpoint("prediction_frozen")
        if args.evaluate:
            evaluation_root = (args.evaluation_root or main_root).resolve(strict=True)
            comparator = evaluation_root / "scripts" / "evaluate_oracle_mask_run.py"
            state["evaluation_code"] = {"root": str(evaluation_root), "sha256_before": code_snapshot(evaluation_root),
                                        "comparator_sha256": digest(comparator)}
            checkpoint("post_export_evaluation")
            completed = subprocess.run([sys.executable, "-X", "utf8", str(comparator), "--run-dir", str(run),
                                        "--dataset", str(evaluation_root / "__dataset"), "--case-id", state["case_id"],
                                        "--target-tolerance-mm", "1"], cwd=evaluation_root,
                                       capture_output=True, timeout=600, check=False)
            state["evaluation_exit_code"] = completed.returncode
            state["evaluation_code"]["unchanged"] = code_snapshot(evaluation_root) == state["evaluation_code"]["sha256_before"]
            state["prediction_unchanged_after_evaluation"] = digest(after / "drawing.dxf") == state["prediction_sha256"]
            if not state["prediction_unchanged_after_evaluation"] or not state["evaluation_code"]["unchanged"]:
                raise RuntimeError("frozen_prediction_or_evaluator_changed")
            state["evaluation_status"] = read(run / "evaluation" / "summary.json").get("status") if (run / "evaluation" / "summary.json").is_file() else "missing_evaluation"
            if completed.returncode:
                state["status"] = "evaluation_failed"
                checkpoint("evaluation_failed")
                return 1
        state["status"] = "completed" if state["parameterization_accepted"] else "completed_with_unresolved_attributes"
        checkpoint("finished")
        return 0
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     failure_stage=state.get("last_stage"), failure_type=type(error).__name__, failed_at=stamp())
        write(run / "run-manifest.json", state)
        print(json.dumps({"status": state["status"], "failure_stage": state["failure_stage"], "failure_type": state["failure_type"]}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

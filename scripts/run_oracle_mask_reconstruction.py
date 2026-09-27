"""Reconstruct CAD from a frozen DXF-GT *raster mask* and source annotations.

Prediction receives only the original image, sanitized OCR, and the oracle
material mask.  The GT DXF is not opened as geometry by this runner, and no GT
coordinates or registration transform are supplied to a provider.  Runs are
development experiments, including cases inherited from a ``test`` split.

Examples:
  python scripts/run_oracle_mask_reconstruction.py CL60-main --run-dir runtime/oracle-mask/cl60-local
  python scripts/run_oracle_mask_reconstruction.py CL60-main --run-dir runtime/oracle-mask/cl60-online --online --provider-profile 9ecode-gpt-5.6-sol --evaluate
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.oracle_mask_inputs import load_oracle_mask_case


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _providers(online: bool, provider_profile: str | None, *, anthropic_thinking: str | None = None) -> tuple[dict, dict]:
    if anthropic_thinking not in (None, "disabled"):
        raise ValueError("--anthropic-thinking accepts only disabled")
    if anthropic_thinking is not None and not online:
        raise ValueError("--anthropic-thinking requires --online with an Anthropic provider")
    if provider_profile and not online:
        raise ValueError("--provider-profile requires --online")
    if not online:
        return {}, {"id": None, "status": "disabled", "network_requests": 0}

    from contour_agent.config import load_local_env, Settings, provider_registry
    from contour_agent.binding_provider import BindingProvider
    from contour_agent.provider import DimensionProvider
    from contour_agent.planning_provider import PlanningProvider
    from contour_agent.topology_edit_provider import TopologyEditProvider, TopologyEvaluationProvider

    load_local_env()
    settings = Settings()
    registry, default_id = provider_registry(settings)
    selected = provider_profile or default_id
    if selected not in registry:
        raise ValueError("Unknown configured provider profile")
    profile = registry[selected]
    settings = profile.settings(settings)
    if anthropic_thinking is not None:
        if settings.wire_api != "anthropic_messages":
            raise ValueError("--anthropic-thinking requires an Anthropic provider")
        settings = replace(settings, anthropic_thinking_mode=anthropic_thinking)
    if not settings.api_key:
        raise ValueError("Selected provider is not configured")
    bundle = {"dimension_provider": DimensionProvider(settings, max_attempts=1),
              "provider": BindingProvider(settings),
              "planner_provider": PlanningProvider(settings),
              "editor_provider": TopologyEditProvider(settings),
              "evaluator_provider": TopologyEvaluationProvider(settings)}
    return bundle, {"id": selected, "status": "configured", "model": profile.model,
                    "wire_api": profile.wire_api, "timeout_seconds": settings.api_timeout,
                    "anthropic_thinking_mode_requested": settings.anthropic_thinking_mode
                    if settings.wire_api == "anthropic_messages" else None}


def run_oracle_mask_case(case_id: str, run_dir: str | Path, *,
                         manifest_path: str | Path = ROOT / "runtime/segmentation/gt-data-v3/manifest.json",
                         online: bool = False, provider_profile: str | None = None,
                         evaluate: bool = False, anthropic_thinking: str | None = None) -> dict:
    """Create a new immutable run and preserve stage/artifact evidence on failure."""
    if anthropic_thinking not in (None, "disabled"):
        raise ValueError("--anthropic-thinking accepts only disabled")
    if anthropic_thinking is not None and not online:
        raise ValueError("--anthropic-thinking requires --online with an Anthropic provider")
    run = Path(run_dir).resolve()
    if run.exists():
        raise ValueError("Choose a new run directory; existing evidence is never overwritten")
    if run.is_relative_to((ROOT / "__dataset").resolve()):
        raise ValueError("Oracle runs must not be written into the source dataset")
    run.mkdir(parents=True)
    run_manifest = run / "run-manifest.json"
    state = {"schema_version": "oracle-mask-reconstruction-v1", "created_at": _stamp(),
             "case_id": case_id, "oracle_mask_conditioned": True,
             "calibration_or_development": True, "held_out": False,
             "online_requested": bool(online), "post_export_evaluation_requested": bool(evaluate),
             "ground_truth_coordinates_sent_to_provider": False,
             "ground_truth_dxf_used_as_prediction_geometry": False,
             "status": "initializing", "last_stage": "initializing", "stage_history": []}

    def checkpoint(stage: str, message: str = "") -> None:
        state["last_stage"] = stage
        state["stage_history"].append({"stage": stage, "at": _stamp()})
        # Do not persist arbitrary exception or provider text in the public run
        # manifest. The normal stage receipts contain structured diagnostics.
        _write(run_manifest, state)
        if message:
            print(f"{stage}: {message}", flush=True)

    checkpoint("initializing")
    try:
        inputs = load_oracle_mask_case(manifest_path, case_id, run / "inputs")
        receipt = inputs["receipt"]
        state.update(source_gt_sha256=receipt["reference_dxf_sha256"],
                     oracle_mask_sha256=receipt["mask_sha256"],
                     source_image_sha256=receipt["source_image_sha256"],
                     source_ocr_sha256=receipt["source_ocr_sha256"],
                     input_manifest="inputs/input-manifest.json",
                     source_split_label=receipt["source_split_label"],
                     reference_components=receipt["reference_components"],
                     reference_holes=receipt["reference_holes"])
        checkpoint("oracle_inputs_validated", "GT 栅格掩膜已冻结；参考 DXF 坐标不进入预测。")
        providers, selected = (_providers(online, provider_profile) if anthropic_thinking is None
                               else _providers(online, provider_profile, anthropic_thinking=anthropic_thinking))
        state["provider"] = selected
        checkpoint("provider_ready")

        from contour_agent.automatic import build_automatic
        from contour_agent.dataset import read_ocr
        from contour_agent.dimension_analysis import analyze_dimensions
        from contour_agent.parametric_pipeline import refine_parametric

        image = inputs["source_image"]
        document = read_ocr(inputs["source_ocr"])
        after = run / "after"
        state["status"] = "running"
        checkpoint("initial_cad", "从 oracle 材料掩膜形成初始 LINE/ARC CAD。")
        model = build_automatic(image, document, after, segmentation_mask=inputs["mask"],
                                segmentation_review={"status": "oracle_mask",
                                                     "source": "registered_dxf_gt",
                                                     "reviewed": False},
                                progress=checkpoint)
        # Keep this text-only interpretation step separate from topology and
        # geometry. Its proposals are recorded for audit, never substituted
        # into source OCR. analyze_dimensions retains the local inventory on
        # transport/schema failure so a failed parser cannot discard the CAD.
        providers = dict(providers)
        dimension_provider = providers.pop("dimension_provider", None)
        state["dimension_analysis"] = {"artifact": "after/dimension-analysis.json",
                                       "status": "pending"}
        checkpoint("dimension_analysis", "解析原图尺寸文字，记录在线解释与本地证据的一致性。")
        dimensions = analyze_dimensions(document, model, provider=dimension_provider, use_api=online)
        _write(after / "dimension-analysis.json", dimensions)
        dimension_receipt = dimensions.get("provider") or {}
        state["dimension_analysis"] = {
            "artifact": "after/dimension-analysis.json", "status": dimensions.get("status"),
            "counts": dimensions.get("counts", {}),
            "geometry_updated_by_api": dimensions.get("geometry_updated_by_api", False),
            "provider": {key: dimension_receipt.get(key) for key in
                         ("status", "error_code", "http_status", "http_success", "schema_success",
                          "network_requests", "elapsed_seconds", "model", "protocol")},
        }
        checkpoint("dimension_analysis_finished")
        # The core model's source-only flag means no GT *coordinates* were fed
        # into geometry fitting. Record the exceptional oracle condition in the
        # run receipt without modifying the source-only pipeline's contracts.
        checkpoint("topology_and_constraints", "依据原图与标注编辑拓扑、绑定约束并求解。")
        model, stage = refine_parametric(image, document, model, after,
                                         use_api=online, progress=checkpoint, **providers)
        state["parameterization_status"] = stage.get("status")
        state["parameterization_accepted"] = bool(stage.get("accepted"))
        state["prediction_artifact"] = "after/drawing.dxf" if (after / "drawing.dxf").is_file() else None
        state["status"] = "completed" if stage.get("accepted") else "completed_with_retained_draft"
        state["prediction_completed_at"] = _stamp()
        checkpoint("prediction_finished", "预测产物已冻结；GT 比较只能在此后单独执行。")

        if evaluate:
            # This import is intentionally after all prediction/provider work.
            # The comparator may open GT for independent post-export scoring;
            # its result has no path back to any provider call in this run.
            from scripts.evaluate_oracle_mask_run import evaluate_oracle_mask_run
            evaluation = evaluate_oracle_mask_run(run, case_id=case_id)
            state["evaluation_status"] = evaluation.get("status")
            state["evaluation_summary"] = "evaluation/summary.json"
            checkpoint("post_export_evaluation_finished")
        return {"run_dir": run, "manifest": run_manifest,
                "prediction": after / "drawing.dxf", "state": state}
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["interrupted_at"] = _stamp()
        _write(run_manifest, state)
        raise
    except Exception as error:
        state["status"] = "failed"
        state["failure_stage"] = state["last_stage"]
        state["failure_type"] = type(error).__name__
        state["failed_at"] = _stamp()
        _write(run_manifest, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "runtime/segmentation/gt-data-v3/manifest.json")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--provider-profile")
    parser.add_argument("--anthropic-thinking", choices=("disabled",),
                        help="Request disabled thinking for an online Anthropic provider; adapter compliance is not assumed.")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    result = run_oracle_mask_case(args.case_id, args.run_dir, manifest_path=args.manifest,
                                  online=args.online, provider_profile=args.provider_profile,
                                  evaluate=args.evaluate, anthropic_thinking=args.anthropic_thinking)
    print(json.dumps({"run_dir": str(result["run_dir"]), "status": result["state"]["status"],
                      "prediction": str(result["prediction"]),
                      "evaluation_status": result["state"].get("evaluation_status")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Create a separate user-declared terminal-face simplification from a source run.

Generation consumes source image/OCR and the exported prediction only. Optional
GT evaluation starts after the scoped DXF has been frozen and hashed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_run(source, policy_path, output, *, evaluate=False):
    from contour_agent.profile_scope import export_terminal_scope
    from contour_agent.parametric_pipeline import _write
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runtime"):
        raise ValueError("use_a_fresh_runtime_output_directory")
    manifest = read(source / "run-manifest.json")
    parent_hashes = {name: digest(source / "after" / name) for name in
                     ("drawing.dxf", "model.json", "constraint-bindings.json", "validation.json")}
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    for p in (source / "inputs").iterdir():
        if p.is_file() and p.suffix.lower() in {".json", ".png", ".jpg", ".jpeg"}:
            shutil.copyfile(p, inputs / p.name)
    ocr = source / "after" / "effective-source-ocr.json"
    if (source/'after/source-ocr-review.json').is_file():
        from contour_agent.source_ocr_review import load_effective_ocr
        document=load_effective_ocr(inputs/'source-image.jpg',inputs/'source-ocr.json',source/'after')
    else:
        if not ocr.is_file():ocr=inputs/'source-ocr.json'
        document=read(ocr)
    model, receipt = export_terminal_scope(inputs / "source-image.jpg", document,
        read(source / "after" / "model.json"), read(source / "after" / "constraint-bindings.json"),
        read(Path(policy_path)), output / "after")
    _write(output/'after/effective-source-ocr.json',document)
    state = {"schema_version": "user-scoped-profile-run-v1", "case_id": manifest["case_id"],
        "status": "user_scoped_partial_profile", "upstream_run": str(source),
        "upstream_prediction_sha256": parent_hashes["drawing.dxf"],
        "upstream_artifacts_sha256": parent_hashes,
        "oracle_mask_conditioned": manifest.get("oracle_mask_conditioned", False),
        "source_gt_sha256": manifest.get("source_gt_sha256"),
        "execution_scope": "explicit_user_scope_projection_of_existing_online_prediction",
        "new_online_api_calls": 0, "fresh_end_to_end_test": False,
        "new_optimization_performed": False, "ground_truth_coordinates_used_for_prediction": False,
        "full_source_annotation_target_satisfied": False, "scope_audit": receipt,
        "prediction_sha256": digest(output / "after" / "drawing.dxf"),
        "original_evaluation_thresholds_changed": False}
    _write(output / "run-manifest.json", state)
    summary = None
    if evaluate:
        from scripts.evaluate_oracle_mask_run import evaluate_oracle_mask_run
        summary = evaluate_oracle_mask_run(output, target_tolerance_mm=1.5,
                                          comparison_tolerances_mm=(1.,))
    state["prediction_unchanged_after_evaluation"] = state["prediction_sha256"] == digest(output / "after" / "drawing.dxf")
    state["upstream_artifacts_unchanged"] = all(digest(source / "after" / name) == value
        for name, value in parent_hashes.items())
    if not state["prediction_unchanged_after_evaluation"] or not state["upstream_artifacts_unchanged"]:
        raise ValueError("scoped_projection_artifact_integrity_failed")
    _write(output / "run-manifest.json", state)
    preview = output / "evaluation" / "overlay.svg"
    if not preview.is_file():
        preview = output / "after" / "preview.svg"
    uri = "data:image/svg+xml;base64," + base64.b64encode(preview.read_bytes()).decode("ascii")
    esc = lambda s: html.escape(str(s), quote=True)
    deferred = "、".join(row["text"] for row in receipt["deferred_annotations"])
    native = receipt["retained_native_checks"]
    counts = {kind: sum(e["type"] == kind for e in model["entities"]) for kind in ("LINE", "ARC")}
    evaluation_text = "未进行新的 GT 比较。"
    measurements = ""
    if summary is not None:
        geometry = summary.get("geometry", {})
        registered = geometry.get("registered_shape_diagnostic", {}).get("conservative_max_error_mm")
        direct = geometry.get("native_coordinate_frame", {}).get("conservative_max_error_mm")
        if isinstance(registered, (int, float)) and isinstance(direct, (int, float)):
            measurements = (f"<p>GT 比较：配准后的保守最大形状误差 {registered:.6f} mm，"
                f"原坐标保守最大误差 {direct:.6f} mm。配准形状{'通过' if registered<=1.5 else '未通过'} 1.5 mm 诊断，"
                f"原坐标形状{'通过' if direct<=1.5 else '未通过'}该数值诊断（坐标约定仍需独立核实）；"
                "对象数量相同也不能证明圆心、端点、扫角及全部拓扑关系相同。</p>")
        evaluation_text = "独立 GT 评估保留原有检查与 1.5 mm 附加诊断：" + esc(json.dumps(
            {k: summary.get(k) for k in ("status", "checks", "target_acceptance")}, ensure_ascii=False))
    report = f'''<!doctype html><meta charset="utf-8"><title>044 暂不构建踏面</title>
<style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:20px;background:#f5f8f7;color:#183332}}img{{width:100%;background:white}}pre{{white-space:pre-wrap}}</style>
<h1>{esc(manifest['case_id'])}：按用户范围简化踏面</h1>
<p>{esc(deferred)} 暂缓构建，指定末端区域替换为一条严格{('竖直' if receipt['axis']=='vertical' else '水平')} LINE。
范围依据为用户明确要求及当前原图／预测对象；未读取 GT 坐标生成预测。</p>
<p>输出 {counts['LINE']} LINE + {counts['ARC']} ARC；范围内严格半径 {receipt['exact_active_radius_count']}/{receipt['active_radius_count']}，
严格角度 {native['angles']['satisfied_count']}/{native['angles']['required_count']}；原生回读、保留的半径、角度、尺寸和已绑定相切检查均通过。</p>
<p><strong>这不是全图通过：</strong>原有 13 个半径的完整源标注目标及历史失败仍保留；其余未绑定尺寸、未知接点、自由度和 GT 误差没有因此自动解决。
新末端 LINE 的位置沿用预测初始支撑，并非新求解的标注属性。此次为已有在线输出的范围投影，新增 API 调用 0 次、优化求解 0 次。</p>
<p><a href="after/drawing.dxf">下载简化范围 DXF</a> · <a href="after/scope-audit.json">范围与原生约束审计</a> ·
<a href="{esc((source/'after'/'drawing.dxf').as_uri())}">原有完整范围 DXF</a></p>
{measurements}<img src="{uri}" alt="原始独立评估 overlay 或实际输出预览"><pre>{evaluation_text}</pre>'''
    (output / "report.html").write_text(report, encoding="utf8")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    state = export_run(args.source_run, args.policy, args.output, evaluate=args.evaluate)
    print(json.dumps({"status": state["status"], "prediction_sha256": state["prediction_sha256"],
                      "new_online_api_calls": state["new_online_api_calls"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Portable, offline Chinese report of segmentation and CAD development results.

Only completed qualification receipts and allowlisted prediction artifacts are
read. Reference geometry never enters prediction; this tool presents existing
independent comparison metrics and does not run inference or any provider.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
from urllib.parse import quote

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
GLOBAL_COMPARISON = RUNTIME / "segmentation/connectivity-eval-v1/comparison.json"
TRAINING = RUNTIME / "segmentation/runs/unet-r18-topology-v1/training.json"


def checked(path, *, exists=True):
    result = Path(path).resolve()
    if not result.is_relative_to(RUNTIME.resolve()):
        raise ValueError("Report inputs and outputs must remain within this workspace runtime")
    if exists and not result.is_file():
        raise ValueError(f"Missing report input: {result.name}")
    return result


def read(path, *, optional=False):
    if optional and not Path(path).exists():
        return {}
    value = json.loads(checked(path).read_text(encoding="utf8"))
    if not isinstance(value, dict):
        raise ValueError("Report JSON inputs must be mappings")
    return value


def digest(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def selected(mapping, keys):
    return {key: mapping[key] for key in keys if key in mapping}


def fmt(value, places=3):
    if value is None:
        return "未评定"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.{places}f}" if math.isfinite(value) else "未评定"
    return str(value)


def esc(value):
    return html.escape(fmt(value), quote=True)


def table(headers, rows):
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(item)}</td>" for item in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def provider_receipt(value):
    return selected(value or {}, ("status", "network_requests", "http_success", "schema_success", "verdict", "elapsed_seconds", "ground_truth_sent"))


def copied_image(source, destination, relative, caption):
    source = checked(source)
    if source.stat().st_size > 30_000_000:
        raise ValueError("Report image exceeds 30 MB")
    with Image.open(source) as image:
        if image.format not in {"PNG", "JPEG"} or image.width * image.height > 80_000_000:
            raise ValueError("Only bounded PNG/JPEG prediction previews may be embedded")
        mime = "image/png" if image.format == "PNG" else "image/jpeg"
        image.verify()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    encoded = base64.b64encode(destination.read_bytes()).decode("ascii")
    block = f'<figure><figcaption>{esc(caption)}</figcaption><img src="data:{mime};base64,{encoded}" alt="{esc(caption)}" loading="lazy"></figure>'
    return block, {"file": relative, "sha256": digest(source)}


def baseline(case_id):
    if case_id == "solid-arrow-ping__img_000281":
        value = read(RUNTIME / "best-segmentation-test-20260921/test-result.json")
        types = value.get("prediction_entities", {})
        return {"source": "best-segmentation-test-20260921", "entity_count": sum(types.values()), "types": types,
                "units": value.get("units"), "reference_status": value.get("comparison", {}).get("status")}
    if case_id == "solid-arrow-ping__img_000182-main":
        value = read(RUNTIME / "segmentation/pilot/20260920T183327511944Z-ca29f264/pilot_summary.json")
        row = next(row for row in value["cases"] if row["case_id"] == case_id)
        # Historical report schema carries independent comparison under reference.
        comparison = row.get("reference", row.get("comparison", {}))
        model_path = RUNTIME / "segmentation/pilot/20260920T183327511944Z-ca29f264/cases" / case_id / "model.json"
        model = read(model_path)
        return {"source": value["run_id"], "entity_count": len(model["entities"]),
                "types": dict(Counter(item["type"] for item in model["entities"])),
                "units": model.get("coordinate_system", {}).get("units"),
                "reference_status": comparison.get("status", "GT_units_unspecified")}
    return {}


def build(qualification, output):
    qualification = checked(qualification)
    output = checked(output, exists=False)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh empty output directory for an immutable report")
    document = read(qualification)
    if document.get("status") not in {"completed", "failed"}:
        raise ValueError("Qualification must finish before building its report")
    trials = document.get("trials", [])
    if not trials:
        raise ValueError("Qualification contains no attempted trials")
    output.mkdir(parents=True, exist_ok=True)
    global_result = read(GLOBAL_COMPARISON)
    training = read(TRAINING)
    case_records, sections = [], []
    for trial in trials:
        case_id = trial["case_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,179}", case_id):
            raise ValueError("Unsafe case identifier")
        repeat = trial.get("repeat", 1)
        if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
            raise ValueError("Invalid repeat identifier")
        folder = f"{case_id}-repeat-{repeat}"
        artifacts = trial.get("artifacts", {})
        comparison = trial.get("comparison", {})
        model_path = checked(artifacts["model.json"]) if artifacts.get("model.json") else None
        model = read(model_path) if model_path else {}
        artifact_dir = model_path.parent if model_path else None
        # Validate every advertised artifact, even when it is not copied.
        for value in artifacts.values():
            checked(value)
        stage = read(artifact_dir / "parametric-stage.json", optional=True) if artifact_dir else {}
        bindings = read(artifact_dir / "constraint-bindings.json", optional=True) if artifact_dir else {}
        dimension = read(artifact_dir / "dimension-analysis.json", optional=True) if artifact_dir else {}
        refinement = read(artifact_dir / "learned-evidence/refinement.json", optional=True) if artifact_dir else {}
        solution = read(artifact_dir / "parametric-solution.json", optional=True) if artifact_dir else {}
        old = baseline(case_id)
        record = {"case_id": case_id, "repeat": repeat, "job_id": trial.get("job_id"), "status": trial.get("status"),
                  "model_exposure": selected(trial.get("model_exposure", {}), ("split", "role", "development_only", "blind_test", "optimization_exposed", "checkpoint_selection_exposed", "checkpoint_sha256")),
                  "baseline": old, "entity_count": len(model.get("entities", [])),
                  "entity_types": dict(Counter(item["type"] for item in model.get("entities", []))),
                  "units": model.get("coordinate_system", {}).get("units"),
                  "scale": selected(model.get("scale", {}), ("status", "pixels_per_mm", "method", "cross_evidence_consistency")),
                  "geometry_valid": trial.get("geometry_valid", False), "artifact_completed": trial.get("artifact_completed", False),
                  "binding_counts": bindings.get("counts", stage.get("binding_counts", {})),
                  "solver": {**selected(solution, ("status", "accepted", "underconstrained", "dimension_solve_success", "all_dimensions_verified")),
                             **selected(solution.get("diagnostics", {}), ("remaining_shape_dof", "converged", "independent_dimension_record_count"))},
                  "publication_accepted": stage.get("accepted", False),
                  "refinement": selected(refinement, ("status", "before", "after", "reasons", "changed_pixels", "forced_bridges", "existing_components_deleted", "detail_inference_size")),
                  "providers": {"尺寸解析": provider_receipt(dimension.get("provider", trial.get("dimension_analysis", {}).get("provider", {}))),
                                "标注绑定": provider_receipt(bindings.get("provider", stage.get("provider", {}))),
                                "图像复核": provider_receipt(trial.get("provider", {}))},
                  "comparison": selected(comparison, ("status", "reference_compared", "reference_within_0_1mm", "tolerance_mm", "alignment_kind", "registered_metrics", "direct_metrics", "issues")),
                  "reference_entity_info": selected(comparison.get("reference_info", {}), ("entities", "types", "source_units", "unit_assumption")),
                  "artifacts": {}}
        counts = record["binding_counts"]
        metrics = comparison.get("registered_metrics", {})
        before, after = refinement.get("before", {}), refinement.get("after", {})
        values = [("模型开发划分", record["model_exposure"].get("split")),
                  ("旧版本 → 本次图元数", f'{old.get("entity_count", "—")} → {record["entity_count"]}'),
                  ("本次 LINE / ARC", f'{record["entity_types"].get("LINE", 0)} / {record["entity_types"].get("ARC", 0)}'),
                  ("GT 图元数（仅评估）", record["reference_entity_info"].get("entities")),
                  ("旧版本 → 本次单位", f'{old.get("units", "—")} → {record["units"]}'),
                  ("源图像素 / 毫米", record["scale"].get("pixels_per_mm")),
                  ("原始 → 细化后 4 连通分量", f'{before.get("components_4", "—")} → {after.get("components_4", "—")}'),
                  ("已绑定 / 识别尺寸", f'{counts.get("bound_source_records", 0)} / {counts.get("recognized_dimensions", "—")}'),
                  ("参数解接受 / 所有尺寸验证", f'{fmt(record["publication_accepted"])} / {fmt(record["solver"].get("all_dimensions_verified", False))}'),
                  ("剩余形状自由度", record["solver"].get("remaining_shape_dof")),
                  ("几何有效", record["geometry_valid"]), ("独立参考比较状态", comparison.get("status")),
                  ("配准后最大 / RMS 误差 (mm)", f'{fmt(metrics.get("max_error_mm"))} / {fmt(metrics.get("rms_error_mm"))}'),
                  ("0.1 mm 独立参考验收通过", comparison.get("reference_within_0_1mm", False))]
        images = []
        sources = [("learned-evidence/raw-prediction-overlay.png", "原始模型分割"),
                   ("learned-evidence/prediction-overlay.png", "受约束细化后的分割"),
                   ("learned-evidence/raw-prediction-mask.png", "原始材料掩膜"),
                   ("learned-evidence/prediction-mask.png", "细化后材料掩膜"),
                   ("topology-overlay.png", "从源图拟合的图元与连接关系"),
                   ("overlay.png", "最终导出的主轮廓叠加图")]
        for name, caption in sources:
            path = artifact_dir / name if artifact_dir else None
            if path is None or not path.is_file():
                images.append(f'<p class="muted">{esc(caption)}：本次记录未提供。</p>')
                continue
            relative = folder + "/" + name.replace("/", "-")
            block, evidence = copied_image(path, output / relative, relative, caption)
            images.append(block)
            record["artifacts"][name] = evidence
        download = ""
        if artifacts.get("drawing.dxf"):
            path = checked(artifacts["drawing.dxf"])
            relative = folder + "/drawing.dxf"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            record["artifacts"]["drawing.dxf"] = {"file": relative, "sha256": digest(path)}
            api_download = f"/api/evaluations/{quote(qualification.name)}/cases/{quote(case_id)}/artifacts/drawing.dxf"
            record["artifacts"]["drawing.dxf"]["api_url"] = api_download
            download = f'<a class="button" href="{html.escape(api_download, quote=True)}" download>下载本次 DXF 草稿</a><p class="muted">离线副本：<code>{esc(relative)}</code></p>'
        api_rows = [(name, value.get("network_requests", 0), value.get("http_success"), value.get("schema_success"), value.get("verdict")) for name, value in record["providers"].items()]
        problems = "；".join(comparison.get("issues", []) + refinement.get("reasons", []))
        sections.append(f'<section><h2>{esc(case_id)}</h2>{download}{table(["检查项", "实际结果"], values)}<h3>在线调用与几何精度分别计数</h3>{table(["阶段", "请求次数", "HTTP 成功", "结构校验成功", "视觉结论"], api_rows)}<p class="note">{esc(problems or "数值求解仅约束已绑定尺寸；闭合和视觉一致不能证明所有参数正确。")}</p><div class="images">{"".join(images)}</div></section>')
        case_records.append(record)
    training_fields = selected(training, ("status", "epochs_planned", "best_epoch", "trained_checkpoint_selected", "automatic_promotion", "initial_checkpoint_sha256", "checkpoint_sha256", "loss", "elapsed_seconds"))
    training_fields["initial_validation"] = selected(training.get("initial_validation", {}), ("mean_iou", "mean_boundary_distance_px", "mean_topology_count_error"))
    training_fields["best_validation"] = selected(training.get("best_validation", {}), ("mean_iou", "mean_boundary_distance_px", "mean_topology_count_error"))
    attempted_ids = {row["case_id"] for row in trials}
    total = document.get("catalog_count", 50)
    summary = {"protocol": "connectivity-parametric-report-v1", "run_id": document["run_id"],
               "qualification_file": str(qualification), "qualification_sha256": digest(qualification),
               "dataset_total": total, "attempted_cases": len(attempted_ids), "not_attempted": total-len(attempted_ids),
               "development_only": True, "blind_test": False, "ground_truth_used_for_generation": False,
               "reference_tolerance_mm": .1, "cases": case_records,
               "segmentation_ablation": {"file": str(GLOBAL_COMPARISON), "sha256": digest(GLOBAL_COMPARISON),
                                          **selected(global_result, ("status", "count", "not_tested", "summary", "source_denominator", "scoring_grid"))},
               "topology_training": training_fields, "engineering_verified": False}
    passes = sum(all(row["comparison"].get("reference_within_0_1mm") is True
                     and row["comparison"].get("reference_compared") is True
                     and row["geometry_valid"] is True and row["artifact_completed"] is True and row["units"] == "mm"
                     for row in case_records if row["case_id"] == case_id) for case_id in attempted_ids)
    summary["reference_passed_cases"] = passes
    global_rows = []
    for row in global_result.get("cases", []):
        raw, new = row.get("raw", {}), row.get("refined", {})
        old_iou, new_iou = raw.get("metrics", {}).get("iou"), new.get("metrics", {}).get("iou")
        global_rows.append((row["id"], row.get("split"), f'{fmt(old_iou*100 if old_iou is not None else None, 2)}%', f'{fmt(new_iou*100 if new_iou is not None else None, 2)}%',
                            raw.get("connectivity_native", {}).get("components_4"), new.get("connectivity_native", {}).get("components_4"), row.get("refinement", {}).get("status")))
    training_text = "本次微调没有超过预先声明的验证选择规则，保留原模型；未自动替换线上模型。" if not training.get("trained_checkpoint_selected") else "微调得到候选模型；报告本身不执行线上模型替换，实际测试使用各样本记录的 checkpoint。"
    training_rows = [("完成轮数", len(training.get("history", []))), ("选择的轮次（0 表示原模型）", training.get("best_epoch")),
                     ("采用训练后权重", training.get("trained_checkpoint_selected")), ("自动上线", training.get("automatic_promotion")),
                     ("初始验证 IoU", training_fields["initial_validation"].get("mean_iou")), ("入选验证 IoU", training_fields["best_validation"].get("mean_iou"))]
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>连通分割与参数化重建测试</title><style>
body{{margin:0;background:#f2f2ed;color:#202f30;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1260px;margin:auto;padding:36px 24px}}h1{{font-size:32px;margin:6px 0}}h2{{font-size:23px;overflow-wrap:anywhere}}h3{{font-size:18px}}section{{background:#fff;padding:26px;margin:24px 0;border:1px solid #d7ddd7;border-radius:8px}}.lead{{max-width:950px}}.note{{padding:14px 18px;background:#fff2de;border-left:4px solid #b97534}}.muted{{color:#63716e}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;margin:14px 0}}th,td{{border-bottom:1px solid #dbe2dc;padding:10px;text-align:left;vertical-align:top}}th{{background:#edf2ed;font-weight:600}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{margin:0;border:1px solid #dbe2dc;background:#f6f7f3}}figcaption{{padding:10px 14px;font-weight:600}}img{{display:block;width:100%;height:auto;background:white}}a{{color:#126c60}}.button{{display:inline-block;padding:9px 16px;background:#186b60;color:white;text-decoration:none;border-radius:4px}}.metrics{{display:flex;gap:20px;flex-wrap:wrap}}.metrics strong{{font-size:28px;display:block}}.metrics div{{flex:1;min-width:170px;padding:18px;background:#e4eee7}}code{{overflow-wrap:anywhere}}@media(max-width:760px){{main{{padding:20px 12px}}section{{padding:16px}}.images{{grid-template-columns:1fr}}}}
</style><main><p class="muted">CONTOUR / 开发验证 · {esc(document['run_id'])}</p><h1>连通分割 → 参数化主轮廓</h1><p class="lead">本报告展示模型原始输出、受约束细化、图元连接与在线尺寸绑定后的真实结果。GT 仅用于已完成预测的独立比较，未进入轮廓生成或在线 API。</p><div class="metrics"><div><strong>{len(attempted_ids)} / {total}</strong>本轮重建图纸</div><div><strong>{total-len(attempted_ids)}</strong>本轮未测试，保留在分母</div><div><strong>{passes} / {len(attempted_ids)}</strong>通过 0.1 mm 参考验收</div></div><p class="note">全部属于开发样本。DXF 配准标签由程序生成，尚非人工逐像素确认的真值；图纸曾用于筛选、分析或模型开发。HTTP 成功、区域连通、数值约束满足与 GT 精度分别报告。</p>{''.join(sections)}<section><h2>完整 14 张开发分割对照</h2><p>固定 512 评估网格；连通分量在原生准备图像上计算。保留失败和拒绝细化的样本。整图多分辨率预测仅在边界邻域融合，当前并非局部切块推理。</p>{table(['图纸','划分','原始 IoU','细化 IoU','原始分量','细化分量','细化状态'],global_rows)}<p class="muted">分割评估覆盖 {global_result.get('count',len(global_rows))} 张，和上方本轮在线重建的 {len(attempted_ids)} 张是不同统计范围。</p></section><section><h2>边界与拓扑损失微调</h2><p>{training_text}</p>{table(['训练项','记录'],training_rows)}<p class="muted">新增边界与局部拓扑损失不能保证真值拓扑；只有验证结果满足既定规则才能成为候选权重。此页面不执行模型上线。</p></section><p><a href="summary.json">下载完整汇总 JSON</a> · 页面图片已内嵌，可离线查看；DXF 与 PNG 副本保存在同目录子文件夹。</p><p class="muted">保持原有 0.1 mm 独立参考阈值；参考坐标只参与比较。报告没有网络请求，未改变原图、模型权重或预测文件。</p></main></html>'''
    page = page.replace('href="summary.json"', 'href="/connectivity-results/summary.json"')
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    (output / "index.html").write_text(page, encoding="utf8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.qualification, args.output)
    print(json.dumps({"run_id": result["run_id"], "attempted_cases": result["attempted_cases"],
                      "dataset_total": result["dataset_total"], "report": str(args.output / "index.html")}, ensure_ascii=False))

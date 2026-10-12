"""Summarize frozen retained drafts separately from previous solved subsets."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    output = ROOT / "runtime/cad-output-v78-20261012"
    coverage = read(output / "coverage-v2/report.json")
    baseline = read(ROOT / "runtime/cad-controlled-20261008/source-measurement-v77-review.json")
    rows = []
    sections = []
    for slug in ("hdsa", "044"):
        name = "HDSA-65-main" if slug == "hdsa" else "044-main"
        run_name = "output-availability-v78-online-" + slug
        run = ROOT / "runtime/cad-controlled-20261008" / run_name
        state = read(run / "run-manifest.json")
        evaluation = read(run / "evaluation/summary.json")
        old = next(r for r in baseline["rows"] if r["version"] == "v77" and r["case_id"] == name)
        row = {"case_id": name, "execution": "Fresh original image + frozen OCR + GT raster mask; online OCR/arrow/planning attempted",
               "status": state["status"], "last_stage": state["last_stage"], "budget_seconds": 900,
               "dxf_availability": state["dxf_availability"], "code_unchanged": state["code"]["unchanged"],
               "prediction_unchanged_after_evaluation": state["prediction_unchanged_after_evaluation"],
               "prediction_frozen_at": state["prediction_frozen_at"], "objects": evaluation["objects"],
               "native_conservative_max_error_mm": evaluation["geometry"]["native_coordinate_frame"]["conservative_max_error_mm"],
               "target_checks": evaluation["target_acceptance"]["checks"], "full_target_passed": False,
               "v77_published": {"status": old["status"], "radius_exact": old["native"]["exact_radius_count"],
                                 "radius_recognized": old["native"]["recognized_radius_count"],
                                 "angles_exact": old["angles"]["strict_native_satisfied_count"],
                                 "angles_recognized": old["angles"]["recognized_count"],
                                 "remaining_shape_dof": old["attributes"]["remaining_shape_dof"],
                                 "native_conservative_max_error_mm": old["native_max_error_mm"]}}
        rows.append(row)
        relative = "../cad-controlled-20261008/" + run_name
        old_overlay = "../cad-controlled-20261008/" + old["run"] + "/evaluation/overlay.svg"
        objects = evaluation["objects"]
        sections.append(f'<section><h2>{name}：新在线任务未完成，保留初始草稿</h2>'
            f'<p>状态：{html.escape(state["status"])}；停止阶段：{html.escape(state["last_stage"])}；预算 900 秒。'
            '未完成最终标注绑定和优化，不能计为参数化通过。冻结后才进行 GT 比较，比较结果不用于选解。</p>'
            f'<p>草稿类型：{html.escape(str(objects["prediction_filtered_types"]))}；GT 主轮廓过滤类型：{html.escape(str(objects["reference_filtered_types"]))}。'
            f'原生坐标保守最大误差 {row["native_conservative_max_error_mm"]:.6f} mm。轮廓接近不代表半径、对象属性或相切满足。</p>'
            f'<p><a href="{relative}/after/drawing.dxf">下载当前初始 DXF</a> · <a href="{relative}/evaluation/overlay.svg">打开当前草稿 overlay</a> · '
            f'<a href="{relative}/run-manifest.json">运行记录</a> · <a href="{old_overlay}">此前 V77 已发布结果 overlay</a></p>'
            f'<img src="{relative}/evaluation/overlay.svg" alt="{name} retained initial draft against GT">'
            '<p>紫红色为 GT，青绿色为当前初始草稿，虚线为 GT 声明的假设层。当前图不是 V77 已完成部分约束求解的结果。</p></section>')
    summary = {"schema_version": "dxf-availability-and-retained-online-drafts-v1", "output_coverage": coverage,
               "new_online_attempts": 2, "complete_new_online_runs": 0, "new_online_full_target_passes": 0,
               "cases": rows, "regression": {"passed": 2233, "skipped": 3},
               "source_dataset_modified": False, "reference_thresholds_relaxed": False,
               "source_gt_coordinates_sent_to_provider": False}
    (output / "review.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    page = ('<!doctype html><meta charset="utf-8"><title>V78：DXF 输出与未通过状态</title>'
            '<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;line-height:1.6;color:#172d32}img{width:100%;border:1px solid #ddd}a{color:#087f72}section{margin:40px 0}</style>'
            '<h1>V78：有 DXF 输出与完整通过分开记录</h1>'
            '<p>50/50 张登记图纸已有非空、可回读的自动 DXF，并独立读回为闭合轮廓。39 张采用可准入的 GT 栅格掩膜，11 张单独进行原图提取。'
            '39 个产物为毫米单位、11 个为像素或未定单位。该覆盖批次没有逐图调用在线 API，也没有认证标注或 GT 精度。</p>'
            '<p><a href="coverage-v2/report.html">逐图输出清单</a> · <a href="coverage-v2/automatic-dxf-outputs.zip">全部 50 个 DXF 与逐图状态 ZIP</a> · '
            '<a href="review.json">完整状态记录</a></p>'
            '<p>另行进行的两个新在线任务均达到 900 秒预算，0/2 完成最终绑定与优化、0/2 完整通过。已有 DXF 保留。此前 V77 结果未被这些草稿覆盖。'
            '全量回归 2233 通过、3 跳过；未放宽严格检查，数据集未修改。</p>'
            + ''.join(sections))
    (output / "review.html").write_text(page, encoding="utf8")
    print(json.dumps({"available": coverage["exported_readable_dxf_cases"], "cases": len(coverage["cases"]),
                      "new_online_full_target_passes": 0}))


if __name__ == "__main__":
    main()

"""Build a self-contained UTF-8 development evaluation report from saved receipts.

Read-only with respect to datasets, checkpoints and measurements. No inference,
API request or GT-based prediction correction is performed here. Missing reports
and masks stay visibly pending instead of becoming successful or zero-valued.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contour_agent.segmentation import letterbox

RUNS = (
    ("unet-r18-weak-v1", "初始弱标签 · 512"),
    ("unet-r18-dxf-v1", "首轮 DXF 标签 · 512"),
    ("unet-r18-dxf-v2", "修正配准 DXF · 768"),
)


def esc(value):
    return html.escape(str(value), quote=True)


def number(value, precision=4):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.{precision}f}"
    return "待评估"


def read_report(path, sources, errors):
    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("JSON 顶层不是对象")
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        return value
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}：无法读取，{exc}")
        return None


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join(f"<th>{esc(h)}</th>" for h in headers)+"</tr></thead><tbody>"+''.join("<tr>"+''.join(f"<td>{cell}</td>" for cell in row)+"</tr>" for row in rows)+"</tbody></table></div>"


def pending(message="待评估"):
    return f'<span class="pending">{esc(message)}</span>'


def status(value):
    labels = {"completed": "已完成", "running": "进行中", "failed": "失败", "no_usable_labels": "无可用标签"}
    return esc(labels.get(value, value or "待评估"))


def encode_png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")


def checked_local(path, workspace):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(workspace.resolve()):
        raise ValueError("图像路径不在当前项目内")
    return resolved


def draw_boundary(source, mask, colour):
    result = source.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, colour, 1, cv2.LINE_AA)
    return result


def prediction_mask(directory, case_id, evaluation):
    if evaluation is None or evaluation.get("status") != "completed":
        return None, "待评估"
    if not any(r.get("id") == case_id for r in evaluation.get("cases", [])):
        return None, "本例尚未评分"
    path = directory/case_id/"prediction.png"
    if not path.is_file():
        return None, "预测图缺失"
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 127
    if mask.shape != (512, 512):
        return None, "预测图尺寸不符；未擅自缩放"
    return mask, None


def case_metrics(report, case_id):
    return next((r for r in (report or {}).get("cases", []) if r.get("id") == case_id), {})


def operational_section(report, title, path):
    if report is None:
        return f'<section><h2>{esc(title)}</h2><p>{pending()} · 尚未找到 {esc(path.name)}。</p></section>'
    if isinstance(report.get("summary"), dict) and "artifact_completed" in report["summary"]:
        summary = report["summary"]
        requested = report.get("selected_count", summary.get("attempted"))
        cells = [("实际尝试 / 本次请求",f"{summary.get('attempted', '?')} / {requested if requested is not None else '?'}"),
                 ("完整源数据分母",summary.get("total","未知")),
                 ("自动产物",summary.get("artifact_completed","未知")),
                 ("几何校验有效",summary.get("geometry_valid","未知")),
                 ("毫米定标",summary.get("scaled_mm","未知")),
                 ("与参考比较",summary.get("reference_compared","未知")),
                 ("独立 0.1 mm 范围内",summary.get("autonomous_reference_within_0_1mm","未知")),
                 ("生成前 / 生成中失败",summary["statuses"].get("failed",0) if isinstance(summary.get("statuses"),dict) else "未知")]
        requests = report.get("online_request_summary",{})
        logical = requests.get("logical_calls",{})
        network = requests.get("network_attempts",{})
        verdicts = requests.get("vision_verdicts",{})
        if report.get("online_requested") or requests:
            cells += [("实际逻辑 API 调用",logical.get("total","未知")),
                      ("HTTP 成功 / 实际调用",f"{logical.get('http_successful', '?')} / {logical.get('total', '?')}"),
                      ("协议成功 / 实际调用",f"{logical.get('schema_successful', '?')} / {logical.get('total', '?')}"),
                      ("实际网络请求（含重试）",network.get("total","未知")),
                      ("视觉 verdict：match / mismatch / uncertain",f"{verdicts.get('match', '?')} / {verdicts.get('mismatch', '?')} / {verdicts.get('uncertain', '?')}")]
        passed = report.get("qualification",{}).get("strict_requested_scope_passed")
        gate = "通过" if passed is True else "未通过" if passed is False else "待评估"
        failures = [row for row in report.get("cases",[]) if row.get("attempted") and
                    (row.get("status") == "failed" or row.get("artifact_completed") is False)]
        failure_text = '<p class="notice"><strong>本轮产物失败：</strong>'+ '、'.join(esc(row.get("case_id",row.get("id","未知案例"))) for row in failures)+'</p>' if failures else ''
        return f'<section><h2>{esc(title)}</h2><p>运行状态：{status(report.get("status"))}；本次范围的严格综合门槛：<strong>{gate}</strong>。</p>'+failure_text+table(["证据项","记录值"],[[esc(key),esc(value if value is not None else "未知")] for key,value in cells])+f'<p class="muted">API 成功率只以实际调用为分母；在 API 调用前失败的案例仍属于系统失败。match verdict 不能证明独立参考精度。来源：{esc(path.name)}</p></section>'
    # Do not dump provider replies, request bodies, environment or configuration.
    # Numeric/boolean operational receipts only; all free text is deliberately
    # excluded from this public HTML section except known top-level status.
    labels = {
        "total": "总样本", "attempted": "已尝试", "completed": "已完成", "failed": "失败",
        "passed": "通过", "http_success": "HTTP 成功", "network_requests": "实际网络请求",
        "logical_calls": "逻辑调用", "provider_transport_success": "传输成功", "online": "在线",
        "artifact_success": "产物成功", "artifacts_created": "产物数", "mm_scale_resolved": "毫米定标数",
        "scale_resolved": "定标成功数", "strict_passed": "严格精度通过", "strict_pass": "严格精度通过",
        "reference_comparable": "可与参考比较", "geometry_valid": "几何校验通过", "online_success": "在线成功",
        "case_count": "案例数", "count": "计数", "success": "成功", "transport_success": "传输成功",
        "supported": "支持数", "elapsed_seconds": "耗时（秒）", "all_passed": "全部通过",
        "artifact_count": "产物数", "generated": "生成数", "errors": "错误数", "status": "状态",
        "network_attempts": "网络尝试", "readback_valid": "DXF 回读有效", "api_success": "API 成功",
    }
    rows = []
    def walk(value, prefix=""):
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            if key in {"summary", "totals", "aggregate", "transport", "geometry", "artifacts", "coverage", "evaluation"} and isinstance(child, dict):
                walk(child, prefix+key+" / ")
            elif key in labels and isinstance(child, (bool, int, float)):
                rows.append([esc(prefix+labels[key]), esc("是" if child is True else "否" if child is False else child)])
    walk(report)
    contents = table(["证据项", "记录值"], rows) if rows else '<p class="notice">报告已存在，但没有识别到可显示的统计字段；不据此判断成功。</p>'
    return f'<section><h2>{esc(title)}</h2><p>状态：{status(report.get("status"))}。传输、产物、定标和严格几何精度分别报告；任何一项均不能替代其他项。</p>{contents}<p class="muted">来源：{esc(path.name)}</p></section>'


def cumulative_online(reports):
    unique = {}
    for report in reports:
        if report is None:
            continue
        identity = report.get("run_id") or hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
        unique.setdefault(identity,report)
    if not unique:
        return '<section><h2>在线调用累计（保留历史）</h2><p>'+pending()+'</p></section>'
    keys = (("logical_calls","total","逻辑 API 调用"), ("logical_calls","http_successful","HTTP 成功调用"),
            ("logical_calls","schema_successful","协议成功调用"), ("network_attempts","total","实际网络请求（含重试）"))
    rows=[]
    for section,key,title in keys:
        values=[r.get("online_request_summary",{}).get(section,{}).get(key) for r in unique.values()]
        known=[v for v in values if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)]
        amount=esc(sum(known)) if len(known)==len(values) else esc(f"已知 {sum(known)}；另有 {len(values)-len(known)} 轮计数未知")
        rows.append([esc(title),amount])
    return f'<section><h2>在线调用累计（保留历史）</h2><p>已保存的 {len(unique)} 个独立在线 run，按 run_id 去重。包含首轮与修复后重跑，未将首轮失败抹去；这不是全部账户的 API 消耗。</p>'+table(["累计证据项","记录值"],rows)+'</section>'


def build(workspace=ROOT, output=None):
    workspace = Path(workspace).resolve()
    base = workspace/"runtime"/"segmentation"
    output = Path(output).resolve() if output else base/"gt-comparison"/"index.html"
    sources, errors = [], []
    manifest_path = base/"gt-data-v3"/"manifest.json"
    manifest = read_report(manifest_path, sources, errors)
    if manifest is None:
        raise ValueError("gt-data-v3/manifest.json 缺失或无效，不能构建数据范围报告")
    cases = manifest.get("cases", [])
    if not isinstance(cases, list) or not all(isinstance(row, dict) for row in cases):
        raise ValueError("manifest cases 格式无效")
    all_splits = Counter(row.get("split") for row in cases)
    accepted = [row for row in cases if row.get("trainable") is True]
    accepted_splits = Counter(row.get("split") for row in accepted)
    training, evaluations = {}, {}
    for run_id, _ in RUNS:
        training[run_id] = read_report(base/"runs"/run_id/"training.json", sources, errors)
        evaluations[(run_id, "val")] = read_report(base/"runs"/run_id/"gt-v3-val-native"/"evaluation.json", sources, errors)
        if run_id != "unet-r18-dxf-v1":
            evaluations[(run_id, "test")] = read_report(base/"runs"/run_id/"gt-v3-test-native"/"evaluation.json", sources, errors)
    manifest_sha = next(source["sha256"] for source in sources if Path(source["path"]) == manifest_path)
    for key, evaluation in list(evaluations.items()):
        if evaluation is None:
            continue
        if evaluation.get("provenance",{}).get("manifest_sha256") != manifest_sha:
            errors.append(f"{key[0]} / {key[1]}：评估标签 manifest 摘要与当前 v3 不符；不展示为统一比较结果")
            evaluations[key] = None
        elif evaluation.get("split") != key[1] or evaluation.get("scoring_grid",{}).get("size") != 512:
            errors.append(f"{key[0]} / {key[1]}：评估划分或评分网格不符；不展示为统一比较结果")
            evaluations[key] = None
    online_path, full_path = base/"gt-online-latest.json", base/"gt-full50-latest.json"
    online, full = read_report(online_path, sources, errors), read_report(full_path, sources, errors)
    prior_online_path = base/"gt-online-before-cell-boundary-fix.json"
    prior_full_path = base/"gt-full50-before-cell-boundary-fix.json"
    prior_online = read_report(prior_online_path,sources,errors)
    prior_full = read_report(prior_full_path,sources,errors)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    final_test = evaluations.get(("unet-r18-dxf-v2","test"))
    initial_test = evaluations.get(("unet-r18-weak-v1","test"))
    test_outcome = ''
    if final_test is not None:
        low = [(record.get("id"), record.get("clean",{}).get("iou")) for record in final_test.get("cases",[])
               if isinstance(record.get("clean",{}).get("iou"),(int,float)) and record["clean"]["iou"] < .25]
        low_text = '；明显低 IoU 案例：'+ '、'.join(f'{esc(cid)}（{number(value)}）' for cid,value in low) if low else ''
        test_outcome = f'<div class="notice"><strong>测试结果仍有失败，不能宣称系统已通过。</strong><br>全部 {esc(final_test.get("count", "?"))} 张测试图：新模型平均 IoU {number(final_test.get("mean_clean_iou"))}；初始模型 {number((initial_test or {}).get("mean_clean_iou"))}{low_text}。此处以 IoU &lt; 0.25 标出严重偏差，未把该提示阈值用作工程合格标准。</div>'
    parts = [f'<header><p class="eyebrow">轮廓分割 · 开发评估</p><h1>从弱标签到 DXF 监督</h1><p>同一 v3 标注与固定评分网格下，对比初始模型、首轮 DXF 模型和修正配准后的 768 模型。</p><p class="muted">生成时间 {stamp} · 缺失数据明确保留为待评估</p></header>',
             '<div class="notice">这是开发集结果，不是盲测，也不代表制造精度。DXF 配准标签未经人工逐像素认证；API 连通、自动产物和参考精度是不同证据。</div>',
             test_outcome,
             '<section><h2>数据与标签范围</h2><div class="cards">'+''.join(f'<div class="card"><strong>{esc(v)}</strong><span>{esc(k)}</span></div>' for k,v in (("源样本", len(cases)),("可用 DXF 标签",len(accepted)),("训练 / 验证 / 测试",f"{accepted_splits['train']} / {accepted_splits['val']} / {accepted_splits['test']}"),("保留在总分母中的排除项",len(cases)-len(accepted))))+'</div>',
             f'<p>原始划分：{all_splits["train"]} / {all_splits["val"]} / {all_splits["test"]}；划分未重新分配。排除项只退出监督损失或标签评分，仍属于 {len(cases)} 张源图。</p>',
             '<ul><li>188：上部直线封口是简化 CAD 主体，可能包含踏面上方空白，不是完整材料边界。</li><li>269：只标左侧主区域；右侧同等显著区域未纳入标签。</li><li>275 / 276：跨训练与验证集的图形高度相似；结果维持 development-only。</li><li>其他图也可能简化踏面、忽略内孔或采用名义尺寸边界，不能直接推导完整工程精度。</li></ul></section>']
    train_rows = []
    for run_id, title in RUNS:
        report = training[run_id]
        if report is None:
            train_rows.append([esc(title)]+[pending()]*7)
            continue
        history = report.get("history", [])
        split = report.get("split_counts", {})
        last_epoch = history[-1].get("epoch") if history else None
        initial = report.get("initial_val_label_iou", report.get("initial_val_weak_iou"))
        best = report.get("best_val_label_iou", report.get("best_val_weak_iou"))
        train_rows.append([esc(title), status(report.get("status")),esc(f"{report.get('size', '?')} px"),
                           esc(f"{last_epoch or 0} / {report.get('epochs_planned', '?')}"),
                           esc(f"{split.get('train', '?')} / {split.get('val', '?')} / {split.get('test', '?')}"),
                           number(initial), number(best),esc(report.get("best_epoch", "待评估"))])
    parts += ['<section><h2>三次训练记录</h2>', table(["模型", "状态", "输入", "完成 / 计划 epoch", "训练 / 验证 / 测试", "初始化验证 IoU", "最佳验证 IoU", "最佳 epoch"], train_rows),
              '<p class="muted">本表验证 IoU 来自每次训练当时的标签和选模规则；标签版本不同，不能直接当作公平精度提升。下面的统一 v3 评估才用于模型比较。</p></section>']
    for split, split_title in (("val", "统一验证集"), ("test", "测试划分（开发用途）")):
        rows = []
        for run_id, title in RUNS:
            if split == "test" and run_id == "unet-r18-dxf-v1":
                continue
            report = evaluations[(run_id, split)]
            if report is None:
                rows.append([esc(title),pending()]+[pending()]*5)
            else:
                rows.append([esc(title),status(report.get("status")),esc(f"{report.get('count', '?')} / {report.get('total', all_splits[split])}"),
                             number(report.get("mean_clean_iou")),number(report.get("mean_clean_boundary_f1")),
                             number(report.get("mean_noisy_iou")),number(report.get("mean_noisy_boundary_f1"))])
        parts += [f'<section><h2>{split_title}</h2>',table(["模型", "状态", "已评分 / 原划分总数", "原图 IoU（主指标）", "原图边界 F1", "条件干扰 IoU*", "条件干扰边界 F1*"],rows),
                  '<p class="notice"><strong>* 条件合成诊断，不是真实抗干扰证据。</strong>附加剖面线使用 GT mask 限定区域，向图像额外提供了目标位置线索。条件干扰分数高于原图分数不能解释为抗干扰能力或泛化改善；以原图 IoU 为主指标。</p>',
                  '<p class="muted">区域 IoU 与边界 F1 均对完整目标计算；边界容差为评分网格 3 px。各模型在准备后的原尺寸图上推理，再将二值预测映射到相同 512 letterbox 网格；填充区不评分。条件干扰还使用固定种子添加尺寸线、箭头与短文字。</p></section>']
    parts += [cumulative_online([prior_online,online]),operational_section(online,"最新在线 API 与系统链路",online_path),operational_section(full,"最新全 50 张自动处理",full_path)]
    if prior_online is not None:
        parts.append(operational_section(prior_online,"首轮历史：在线 API 与系统链路（修复前）",prior_online_path))
    if prior_full is not None:
        parts.append(operational_section(prior_full,"首轮历史：全 50 张自动处理（修复前）",prior_full_path))
    excluded_rows = []
    for row in cases:
        if row.get("trainable") is True:
            continue
        ref = row.get("reference_status")
        reason = "缺少匹配的 GT DXF" if ref == "missing" else "GT 不能组成合格闭合主体（244 存在大缺口）" if ref == "invalid" else "配准筛选未通过"
        failed = row.get("registration", {}).get("failed_quality_checks", [])
        if failed:
            reason += "："+", ".join(map(str,failed))
        excluded_rows.append([esc(row.get("id")),esc(row.get("split")),esc(reason)])
    parts += ['<section><h2>排除项明细</h2>',table(["案例", "原划分", "原因"],excluded_rows),'</section>',
              '<section><h2>全部测试图：GT 与新旧预测</h2><p><span class="red">红：配准 GT</span>　<span class="orange">橙：初始弱标签模型</span>　<span class="green">绿：最终 768 模型</span>。每例三列采用相同评分坐标；模型预测缺失时不绘制替代结果。</p>']
    for row in cases:
        if row.get("split") != "test":
            continue
        cid = row.get("id", "")
        if not isinstance(cid,str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,140}",cid):
            errors.append("跳过无效案例 ID")
            continue
        parts.append(f'<article class="case"><h3>{esc(cid)}</h3>')
        try:
            with Image.open(checked_local(row["image"],workspace)) as source:
                rgb = np.asarray(source.convert("RGB"))
            with Image.open(checked_local(row["mask"],workspace)) as source:
                target = np.asarray(source.convert("L")) > 127
            if target.shape != rgb.shape[:2]:
                raise ValueError("准备图与标签尺寸不一致")
            boxed,target_box,_ = letterbox(rgb,512,target)
            base_gt = draw_boundary(boxed,target_box,(215,43,53))
            panels = [("配准 GT",encode_png(base_gt),None)]
            for run_id,title,colour in (("unet-r18-weak-v1","旧预测 + GT",(230,125,25)),("unet-r18-dxf-v2","新预测 + GT",(0,144,89))):
                report = evaluations[(run_id,"test")]
                mask,issue = prediction_mask(base/"runs"/run_id/"gt-v3-test-native",cid,report)
                metrics = case_metrics(report,cid).get("clean",{})
                caption = title+f" · IoU {number(metrics.get('iou'))} / F1 {number(metrics.get('boundary_f1'))}"
                panels.append((caption,None if mask is None else encode_png(draw_boundary(base_gt,mask,colour)),issue))
            parts.append('<div class="compare">'+''.join(f'<figure><figcaption>{esc(title)}</figcaption>'+ (f'<img src="{data}" alt="{esc(cid+title)}">' if data else f'<div class="missing-image">{pending(issue)}</div>')+'</figure>' for title,data,issue in panels)+'</div>')
        except (OSError,ValueError,KeyError) as exc:
            parts.append(f'<p class="notice">对比图待生成：{esc(exc)}</p>')
            errors.append(f"{cid}：{exc}")
        parts.append('</article>')
    parts.append('</section>')
    if errors:
        parts.append('<section><h2>读取问题</h2><ul>'+''.join(f'<li>{esc(error)}</li>' for error in errors)+'</ul></section>')
    provenance_rows=[]
    for source in sources:
        path=Path(source["path"])
        try: name=path.relative_to(workspace).as_posix()
        except ValueError: name=path.name
        provenance_rows.append([esc(name),f'<code>{source["sha256"]}</code>'])
    parts.append('<section><details><summary>输入报告与 SHA-256</summary>'+table(["来源文件", "内容校验值"],provenance_rows)+'</details><p class="muted">此页面只读取已有标签、训练日志、预测图和评估记录，不修改测量，不触发模型推理或外部 API。所有展示图片均嵌入本文件。</p></section>')
    css='''*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#182b3a;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:48px 30px}header{margin-bottom:26px}h1{font-size:40px;letter-spacing:-1px;margin:8px 0}h2{font-size:23px;margin:0 0 18px}h3{font-size:17px;overflow-wrap:anywhere}p{margin:10px 0}.eyebrow{color:#176b78;font-size:13px;font-weight:700;letter-spacing:2px}.muted{font-size:13px;color:#637482}.notice{padding:16px 20px;border-left:4px solid #d9942f;background:#fff5de;border-radius:5px;margin:18px 0}section{background:white;border:1px solid #e1e7ec;border-radius:12px;padding:26px;margin:22px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{padding:18px;background:#edf5f7;border-radius:8px}.card strong{display:block;font-size:28px;color:#165b68}.card span{font-size:13px;color:#56717a}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;text-align:left}th{background:#eff4f6;color:#425c6d;font-size:12px;font-weight:650;white-space:nowrap}td,th{padding:12px 14px;border-bottom:1px solid #e5ebef}td{vertical-align:top}tr:last-child td{border:0}.pending{color:#966015;font-weight:600}.case{border-top:1px solid #e5ebef;padding-top:15px;margin-top:24px}.compare{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;background:#f9fafb;border:1px solid #e0e6eb;border-radius:6px;overflow:hidden}figcaption{padding:10px;font-size:12px;min-height:52px}img{width:100%;height:auto;display:block;background:white}.missing-image{aspect-ratio:1;display:grid;place-items:center;background:#f1f4f6}.red{color:#d72b35}.orange{color:#bd6412}.green{color:#008650}code{font-size:11px;word-break:break-all}summary{cursor:pointer;font-weight:600}li{margin:5px 0}@media(max-width:850px){main{padding:24px 14px}h1{font-size:29px}section{padding:18px}.cards{grid-template-columns:repeat(2,1fr)}.compare{grid-template-columns:1fr}figcaption{min-height:0}td,th{padding:9px}}@media print{body{background:white}main{padding:0}section{break-inside:avoid;border-radius:0}.case{break-inside:avoid}.compare{grid-template-columns:repeat(3,1fr)}}'''
    document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>轮廓分割评估：弱标签与 DXF 监督</title><style>'+css+'</style></head><body><main>'+''.join(parts)+'</main></body></html>'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(document,encoding="utf-8")
    receipt={"generated_at":stamp,"output":str(output),"source_count":len(sources),"test_panels":sum(r.get("split")=="test" for r in cases),"read_errors":errors,"sources":sources,"network_requests":0,"inference_requests":0}
    (output.parent/"build.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=build(args.workspace,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!="sources"},ensure_ascii=True))

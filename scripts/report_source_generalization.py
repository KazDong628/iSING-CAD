"""Post-freeze comparison of source parsing, current CAD certificates and GT scores.

Reference metrics are read only from already completed evaluation artifacts.
This script cannot invoke a provider or change a prediction.
"""
from __future__ import annotations

import argparse
import html
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.report_strict_joint_iteration import write_report


def source_review(path):
    if not path.is_file():
        return {"status": "not_implemented_in_this_run", "requests": 0}
    manifest_path = path.parent.parent / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text("utf8")) if manifest_path.is_file() else {}
    checks = []
    for name in ("source-ocr-review.json", "effective-source-ocr.json"):
        target = path.parent / name
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        expected = manifest.get("frozen_artifacts", {}).get(name)
        checks.append({"file": name, "expected_sha256": expected, "actual_sha256": actual,
                       "passed": bool(expected and actual == expected)})
    if not all(c["passed"] for c in checks):
        return {"status": "unfrozen_or_mixed_ocr_sidecars", "requests": None,
                "frozen_evidence_checks": checks, "consensus_is_ground_truth": False}
    audit = json.loads(path.read_text("utf8"))
    pages = audit.get("pages", [])
    reused = manifest.get("execution_scope") == "frozen_topology_online_final_stage_replay"
    return {"status": "reused_upstream_" + str(audit.get("status")) if reused else audit.get("status"),
        "frozen_evidence_checks": checks, "reused_from_upstream": reused,
        "requests": 0 if reused else sum(
        p.get("provider", {}).get("network_requests") or 0 for p in pages),
        "upstream_evidence_requests": sum(p.get("provider", {}).get("network_requests") or 0 for p in pages) if reused else 0,
        "unknown_request_counts": 0 if reused else sum(p.get("provider", {}).get("network_requests") is None for p in pages),
        "http_successful_pages": 0 if reused else sum(p.get("provider", {}).get("http_success") is True for p in pages),
        "schema_successful_pages": 0 if reused else sum(p.get("provider", {}).get("schema_success") is True for p in pages),
        "withheld_record_ids": audit.get("withheld_record_ids", []),
        "changed": [{k: d.get(k) for k in ("record_id", "original_text", "effective_text", "status")}
                    for d in audit.get("decisions", []) if d.get("changed")],
        "unresolved": [{k: d.get(k) for k in ("record_id", "status")}
                       for d in audit.get("decisions", []) if d.get("status") in {
                           "review_unavailable", "unresolved_transcription_disagreement", "unresolved_transcription_unreadable", "record_budget_exhausted"}],
        "original_document_sha256": audit.get("original_document_sha256"),
        "effective_document_sha256": audit.get("effective_document_sha256"),
        "row_quarantines": [{"page": p.get("page"), "view": p.get("view"), **q}
                            for p in pages for q in p.get("provider", {}).get("row_quarantines", [])],
        "consensus_is_ground_truth": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--versions", nargs="+", default=["v11", "v17", "v18", "v19"])
    parser.add_argument("--output-prefix", default="source-generalization-review-20261010")
    args = parser.parse_args()
    report, json_path, html_path = write_report(args.root, args.output_prefix, versions=args.versions)
    reviews = []
    for version in args.versions:
        for case in ("hdsa", "044"):
            run = args.root / f"angle-semantics-{version}-{case}"
            reviews.append({"version": version, "case": case, "run_exists": run.is_dir(),
                            **source_review(run / "after" / "source-ocr-review.json")})
    report["source_ocr_review"] = reviews
    scopes = []
    for version in args.versions:
        for case in ("hdsa", "044"):
            manifest_path = args.root / f"angle-semantics-{version}-{case}" / "run-manifest.json"
            manifest = json.loads(manifest_path.read_text("utf8")) if manifest_path.is_file() else {}
            scopes.append({"version": version, "case": case,
                "execution_scope": manifest.get("execution_scope", "full_source_pipeline"),
                "upstream_run": manifest.get("upstream_run"),
                "upstream_reconstruction_rerun": manifest.get("upstream_reconstruction_rerun", True)})
    report["execution_scopes"] = scopes
    probes = []
    for name in ("ocr-reread-v18-isolated-probe-20261010.json", "source-ocr-unprimed-probe-20261010.json"):
        path = args.root / name
        if path.is_file():
            data = json.loads(path.read_text("utf8"))
            probes.append({"artifact": name, "status": data.get("status"), "scope": data.get("scope"),
                           "prior_probe": data.get("prior_probe"), "consensus": data.get("consensus"),
                           "applied_to_cad": False})
    report["isolated_ocr_probes"] = probes
    report["scope_notes"] = [
        "V11 completed was a radius-subset status, not full CAD reconstruction.",
        "The same two exposed development inputs recur across versions; runs are not independent held-out cases.",
        "Original images, supplied oracle raster masks and OCR are prediction inputs; GT DXF comparison occurs after freeze.",
        "Two image transcription requests form model consensus, not an independent truth certificate.",
        "V17/V18 prompts contained numeric format examples matching one reviewed value; retain and disclose this priming risk. Production removed numeric examples; a separate source-only verification is reported independently.",
        "V20 adds symbolic finite support levels and curve extrema. Interior inspection stations, datum ownership and diameter dimensions remain incomplete; helper support does not establish real-case binding coverage.",
        "044 source drawing extends beyond the selected main-profile mask; scope conflicts remain unresolved and in the denominator.",
        "Original 0.1 mm accuracy target and supplemental 1 mm checks are reported separately; no failed source gate was relaxed."]
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf8")
    rows = []
    for row in reviews:
        cells = [row["case"], row["version"], row["status"], row["requests"],
                 row.get("http_successful_pages", 0), row.get("schema_successful_pages", 0),
                 json.dumps(row.get("changed", []), ensure_ascii=False), len(row.get("unresolved", [])),
                 ", ".join(row.get("withheld_record_ids", []))]
        rows.append("<tr>"+"".join("<td>"+html.escape(str(v))+"</td>" for v in cells)+"</tr>")
    intro = """<section><h2>本轮诊断：输入文字、测量对象和重建范围</h2>
<p>HDSA V11 的 completed 只代表原半径合同通过。当时仍有20个未绑定尺寸、23个形状自由度；044 的主轮廓草稿也不等于失败无产物。不能用一次HTTP成功、对象总数一致或肉眼相似替代完整重建。</p>
<p>本轮增加原图裁剪重读、短尺寸双箭头区间、可求解的理论角点、导出后线性尺寸独立回读。原始OCR不覆盖，失败页及未决记录保留。测量截面/符号宽度、旋转轴与直径、曲线极值和完整范围分类仍待解决。</p>
<p>V19增加竖排文字两种旋转方向、语义有反证的旧OCR数值暂停硬绑定，以及发布模型与有效OCR身份绑定。语义未决仍计入分母；接口测试通过不等于完整图纸已经求解成功。</p>
<p>044 的原图外轮缘与当前输入主轮廓范围不同；既有参考评估也包含假设闭合线。该后验信息只解释评估范围，不能用于预测规则或删除困难标注。</p>
<p>V17/V18转写提示含具体数值示例，恰好与044某条待重读值一致，存在答案引导风险。冻结实验原样保留；后续代码已移除数值示例，独立复核另列，不能把本轮共识当作无偏识别准确率。</p>
<p>以下是同两幅开发图的重复实验。API文字重读请求单列；原有阶段请求见各运行明细。所有GT比较均发生在DXF冻结之后。</p>
<table><thead><tr><th>图纸</th><th>版本</th><th>重读状态</th><th>请求</th><th>HTTP成功页</th><th>结构成功页</th><th>原图转写变化</th><th>未决记录</th><th>因语义反证暂停硬绑定</th></tr></thead><tbody>"""+"".join(rows)+"</tbody></table></section>"
    intro += "<section><h2>独立转写诊断，未应用于冻结CAD</h2><pre>" + html.escape(json.dumps(probes, ensure_ascii=False, indent=2)) + "</pre></section>"
    replays = [row for row in scopes if row["execution_scope"] == "frozen_topology_online_final_stage_replay"]
    if replays:
        intro = "<section><h2>测试范围</h2><p>以下版本复用上游冻结拓扑、有效OCR和原图箭头提议，只重新运行在线最终绑定、联合求解及导出；不能视为重跑完整上游，也不能据此声称对象识别已改善。重用的OCR请求没有重复计费到本轮调用统计。</p><pre>" + html.escape(json.dumps(replays, ensure_ascii=False, indent=2)) + "</pre></section>" + intro
    if "v22" in args.versions:
        intro = """<section><h2>V22：保留有效源证据，准确呈现未完成状态</h2>
<p>修复局部搜索截断抹掉此前已验证箭头的问题。阳性物理观测和搜索是否完整分别记录；对未完成搜索产生的阳性，额外检查全部已识别半径文字的归属竞争，实际歧义仍禁止自动锁定。没有降低笔画、字形、目标关联或数值验收阈值。</p>
<p>预检异常保留已完成的绑定阶段，区分预算耗尽和求解异常；不再把未启动求解误报为数值失败。截图修正产生的纯拓扑草稿明确显示待核验，文件可下载不代表标注求解完成。</p></section>""" + intro
        intro = "<section><h2>V22：延续完整曲线可行性检查</h2><p>修复固定五轮过早停止：在原有共享时间及调用预算内，继续补充实际违反且尚未检查的轮廓见证点，最多24轮。可选RMS精修最多5秒，保留已核实可行解及后续核验时间。半径、角度、相切和源图门限均不放宽；可行解不等于全局最优或完整标注覆盖。</p></section>" + intro
    if any(version in {"v20", "v21"} for version in args.versions):
        intro = """<section><h2>当前迭代：严格标注属性、测量语义与搜索可达性</h2>
<p>V20将新准入的原图距离和一般角度作为硬等式，并在导出后重新读取原生DXF验证。135 mm不能再用134.990 mm冒充精确满足；历史未声明硬约束的结果仍按历史合同展示，不追溯更改原实验。</p>
<p>新增有限直线面位置、有限圆弧极值两类符号测量锚点。锚点随未知CAD属性一起求解，不保存拟合坐标作为答案。真实图上的测量截面、旋转轴与直径基准仍有缺口；支持一种数学类型并不等于已经成功识别并绑定这些图纸上的每条标注。</p>
<p>拓扑搜索跳过未改变实际对象结构的无效拆分，并优先安排真正实现标注修复的候选。V21另保留经源图校验的多个物理箭头供全局归属判断；只有一个候选标签也不能免除文字与引线归属证据。</p>
<p>V21为已满足硬约束、但未通过原图偏差预算的有限合法候选增加受限的重新初始化求解。重新求解不放松R、角度、G1、距离或源图门限，不把失败候选的数值证书计入正式导出。</p>
<p>以下按冻结正式产物报告。对象数量、类型、完整标注覆盖、严格属性、连接关系、自由度和独立GT精度需要分别通过；任一子集通过都不代表完整自动重建成功。</p></section>""" + intro
    content = html_path.read_text("utf8")
    body = content.find(">", content.find("<body"))+1
    html_path.write_text(content[:body]+intro+content[body:], "utf8")
    print(json.dumps({"json": str(json_path), "html": str(html_path), "coverage": report["coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

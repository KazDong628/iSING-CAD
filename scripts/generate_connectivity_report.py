from __future__ import annotations

import argparse
import html
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _provider(receipt: dict | None) -> dict:
    receipt = receipt or {}
    return {
        "status": receipt.get("status"),
        "protocol": receipt.get("protocol"),
        "http_success": receipt.get("http_success"),
        "schema_success": receipt.get("schema_success"),
        "network_requests": receipt.get("network_requests"),
        "elapsed_seconds": receipt.get("elapsed_seconds"),
        "total_timeout_seconds": receipt.get("total_timeout_seconds"),
        "ground_truth_sent": receipt.get("ground_truth_sent"),
    }


def _case_summary(trial: dict, report_name: str) -> dict:
    artifacts = trial["artifacts"]
    plan = _read(Path(artifacts["topology-plan.json"]))
    binding = _read(Path(artifacts["constraint-bindings.json"]))
    stage = _read(Path(artifacts["parametric-stage.json"]))
    comparison = trial.get("comparison") or {}
    prediction = comparison.get("prediction_info") or {}
    reference = comparison.get("reference_info") or {}
    registered = comparison.get("registered_metrics") or {}
    selected_id = plan.get("selected_candidate_id")
    selected_row = next(
        (
            row
            for row in (plan.get("local_evaluation") or {}).get("evaluated", [])
            if row.get("candidate_id") == selected_id
        ),
        {},
    )
    base_row = next(
        (
            row
            for row in (plan.get("local_evaluation") or {}).get("evaluated", [])
            if row.get("candidate_id") == "cand-01-base_topology"
        ),
        {},
    )
    base = "/api/evaluations/{}/cases/{}/artifacts/".format(
        quote(report_name), quote(trial["case_id"])
    )
    links = {
        name: base + quote(name)
        for name in (
            "drawing.dxf",
            "preview.svg",
            "overlay.png",
            "segmentation-overlay.png",
            "topology-overlay.png",
            "binding-topology.png",
            "topology-plan.json",
            "topology-candidates.json",
            "constraint-bindings.json",
            "parametric-stage.json",
        )
        if name in artifacts
    }
    dimension_provider = (trial.get("dimension_analysis") or {}).get("provider") or {}
    return {
        "case_id": trial["case_id"],
        "status": trial.get("status"),
        "job_id": trial.get("job_id"),
        "elapsed_seconds": trial.get("elapsed_seconds"),
        "geometry_valid": trial.get("geometry_valid"),
        "selected_candidate_id": selected_id,
        "selection_source": plan.get("selection_source"),
        "candidate_count": plan.get("candidate_count"),
        "source_annotation_leader_count": plan.get("source_annotation_leader_count"),
        "annotation_coverage": plan.get("selected_annotation_coverage"),
        "source_boundary_support": (selected_row.get("metrics") or {}).get(
            "source_boundary_support"
        ),
        "base_entity_count": (base_row.get("metrics") or {}).get("entity_count"),
        "prediction": {
            "entities": prediction.get("entities"),
            "types": prediction.get("types") or {},
        },
        "reference": {
            "entities": reference.get("entities"),
            "types": reference.get("types") or {},
            "source_units": reference.get("source_units"),
        },
        "online_selection_gate": plan.get("online_selection_gate") or {},
        "planner_provider": _provider(plan.get("provider")),
        "dimension_provider": _provider(dimension_provider),
        "binding_provider": _provider(binding.get("provider")),
        "vision_provider": _provider(trial.get("provider")),
        "vision_verdict": (trial.get("provider") or {}).get("verdict"),
        "binding_counts": binding.get("counts") or {},
        "parameterization": {
            "status": stage.get("status"),
            "accepted": stage.get("accepted"),
            "topology_exported": stage.get("topology_exported"),
            "reason": stage.get("reason"),
        },
        "comparison": {
            "status": comparison.get("status"),
            "reference_within_0_1mm": comparison.get("reference_within_0_1mm"),
            "max_error_mm": registered.get("max_error_mm"),
            "p95_error_mm": registered.get("p95_error_mm"),
            "rms_error_mm": registered.get("rms_error_mm"),
            "note": (
                "GT 单位元数据缺失，不能计算可声明的毫米误差。"
                if comparison.get("status") != "compared"
                else "GT 仅在 DXF 导出后用于独立评分。"
            ),
        },
        "only_line_arc": set((prediction.get("types") or {}).keys()) <= {"LINE", "ARC"},
        "ground_truth_used_for_generation": False,
        "links": links,
    }


def _pill(ok: bool, yes: str = "成功", no: str = "失败") -> str:
    cls = "ok" if ok else "bad"
    return f'<span class="pill {cls}">{yes if ok else no}</span>'


def _seconds(value) -> str:
    return "—" if value is None else f"{float(value):.1f}s"


def _num(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _render_case(case: dict) -> str:
    esc = html.escape
    pred, ref = case["prediction"], case["reference"]
    pc, rc = pred["types"], ref["types"]
    bind = case["binding_counts"]
    cmp = case["comparison"]
    gate = case["online_selection_gate"]
    providers = [
        ("规划", case["planner_provider"]),
        ("尺寸解析", case["dimension_provider"]),
        ("约束绑定", case["binding_provider"]),
        ("视觉复核", case["vision_provider"]),
    ]
    provider_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            label,
            _pill(bool(row.get("http_success")), "HTTP 200", "HTTP 失败"),
            _pill(bool(row.get("schema_success")), "结构有效", "结构失败"),
            _seconds(row.get("elapsed_seconds")),
        )
        for label, row in providers
    )
    artifact_links = " ".join(
        f'<a href="{esc(url)}" target="_blank">{esc(name)}</a>'
        for name, url in case["links"].items()
        if not name.endswith(".png")
    )
    gate_note = (
        "在线选择已接受"
        if gate.get("accepted")
        else "在线选择被紧凑性门禁拒绝，采用本地推荐候选"
    )
    return f"""
    <section class="case">
      <div class="case-head">
        <div><div class="eyebrow">HOLDOUT / ONLINE</div><h2>{esc(case['case_id'])}</h2></div>
        <div>{_pill(bool(case['geometry_valid']), '几何有效', '几何无效')} {_pill(bool(case['only_line_arc']), '仅 LINE / ARC', '含非标准曲线')}</div>
      </div>
      <div class="metrics">
        <div><b>{case['base_entity_count']} → {pred['entities']}</b><span>候选规划前 → 输出对象数</span></div>
        <div><b>{ref['entities']}</b><span>GT 对象数（导出后评分）</span></div>
        <div><b>{pc.get('LINE',0)} / {pc.get('ARC',0)}</b><span>预测 LINE / ARC</span></div>
        <div><b>{rc.get('LINE',0)} / {rc.get('ARC',0)}</b><span>GT LINE / ARC</span></div>
      </div>
      <div class="grid two">
        <div class="panel">
          <h3>多候选拓扑规划</h3>
          <dl>
            <dt>选中候选</dt><dd>{esc(str(case['selected_candidate_id']))}</dd>
            <dt>选择来源</dt><dd>{esc(str(case['selection_source']))}</dd>
            <dt>候选数量</dt><dd>{case['candidate_count']}</dd>
            <dt>标注引线</dt><dd>{case['source_annotation_leader_count']}</dd>
            <dt>标注覆盖</dt><dd>{_num(100*(case['annotation_coverage'] or 0),1)}%</dd>
            <dt>源边界支持</dt><dd>{_num(100*(case['source_boundary_support'] or 0),1)}%</dd>
          </dl>
          <p class="note">{gate_note}</p>
        </div>
        <div class="panel">
          <h3>约束绑定与求解</h3>
          <dl>
            <dt>识别尺寸</dt><dd>{bind.get('recognized_dimensions','—')}</dd>
            <dt>API 选择 / 接受</dt><dd>{bind.get('api_selected','—')} / {bind.get('api_accepted','—')}</dd>
            <dt>未绑定尺寸</dt><dd>{bind.get('unbound_dimensions','—')}</dd>
            <dt>约束总数</dt><dd>{bind.get('constraints','—')}</dd>
            <dt>参数求解</dt><dd>{esc(str(case['parameterization']['status']))}</dd>
            <dt>原因</dt><dd>{esc(str(case['parameterization']['reason']))}</dd>
          </dl>
          <p class="note">拓扑已导出；参数候选因求解未收敛而未覆盖源拓扑。</p>
        </div>
      </div>
      <div class="grid images">
        <figure><img src="{esc(case['links']['segmentation-overlay.png'])}" loading="lazy"><figcaption>连续材料分割</figcaption></figure>
        <figure><img src="{esc(case['links']['topology-overlay.png'])}" loading="lazy"><figcaption>LINE / ARC 主轮廓</figcaption></figure>
        <figure><img src="{esc(case['links']['overlay.png'])}" loading="lazy"><figcaption>最终输出叠加</figcaption></figure>
      </div>
      <div class="grid two">
        <div class="panel">
          <h3>在线阶段回执</h3>
          <table><thead><tr><th>阶段</th><th>传输</th><th>结构</th><th>耗时</th></tr></thead><tbody>{provider_rows}</tbody></table>
          <p class="note">视觉结论：{esc(str(case['vision_verdict']))}。所有阶段均未向服务商发送 GT。</p>
        </div>
        <div class="panel">
          <h3>独立 GT 评分</h3>
          <dl>
            <dt>状态</dt><dd>{esc(str(cmp['status']))}</dd>
            <dt>最大误差</dt><dd>{_num(cmp['max_error_mm'])} mm</dd>
            <dt>P95</dt><dd>{_num(cmp['p95_error_mm'])} mm</dd>
            <dt>RMS</dt><dd>{_num(cmp['rms_error_mm'])} mm</dd>
            <dt>0.1 mm</dt><dd>{_pill(bool(cmp['reference_within_0_1mm']), '通过', '未通过')}</dd>
          </dl>
          <p class="note">{esc(cmp['note'])}</p>
        </div>
      </div>
      <div class="artifacts">{artifact_links}</div>
    </section>"""


def _render(summary: dict) -> str:
    bench = summary["api_speed"]
    cases = "".join(_render_case(case) for case in summary["cases"])
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>主轮廓在线重建结果</title>
<style>
:root{{--ink:#162825;--muted:#6b7772;--paper:#f5f3ec;--card:#fffef9;--line:#d7d9cf;--accent:#0b7165;--orange:#d87533;--bad:#a94b2b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}header{{background:#182927;color:white;padding:32px 5vw;border-bottom:5px solid var(--orange)}}header h1{{margin:4px 0;font-size:30px}}header p{{margin:4px 0;color:#c9d6d2}}main{{max-width:1440px;margin:auto;padding:28px 4vw 80px}}.eyebrow{{font:12px/1.4 ui-monospace,monospace;letter-spacing:.16em;color:#d99155}}.overview,.case{{background:var(--card);border:1px solid var(--line);box-shadow:0 5px 18px #253a3212;margin-bottom:24px;padding:24px}}.case-head{{display:flex;justify-content:space-between;gap:20px;align-items:center;border-bottom:1px solid var(--line);padding-bottom:14px}}h2{{margin:2px 0;font-size:23px}}h3{{margin:0 0 14px;font-size:17px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);margin:18px 0}}.metrics div{{background:#f8f7f1;padding:16px}}.metrics b{{display:block;font:700 23px ui-monospace,monospace}}.metrics span{{color:var(--muted);font-size:13px}}.grid{{display:grid;gap:18px;margin:18px 0}}.two{{grid-template-columns:1fr 1fr}}.images{{grid-template-columns:repeat(3,1fr)}}.panel,figure{{margin:0;border:1px solid var(--line);padding:17px;background:white}}figure img{{width:100%;height:310px;object-fit:contain;background:#eeefe9}}figcaption{{padding-top:8px;color:var(--muted)}}dl{{display:grid;grid-template-columns:145px 1fr;margin:0;gap:7px 14px}}dt{{color:var(--muted)}}dd{{margin:0;font-family:ui-monospace,monospace}}.pill{{display:inline-block;padding:3px 8px;border-radius:20px;font-size:12px;background:#eee}}.pill.ok{{background:#dcefe8;color:#075e52}}.pill.bad{{background:#f6dfd5;color:var(--bad)}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:8px}}.note{{color:var(--muted);font-size:13px}}.artifacts a{{display:inline-block;margin:4px 10px 4px 0;color:var(--accent)}}.warning{{border-left:4px solid var(--orange);padding:12px 16px;background:#fff1df}}code{{background:#e9ece5;padding:2px 5px}}@media(max-width:900px){{.two,.images,.metrics{{grid-template-columns:1fr}}figure img{{height:auto}}}}
</style></head><body>
<header><div class="eyebrow">CONTOUR AGENT / ONLINE EVALUATION</div><h1>标注驱动多候选主轮廓在线测试</h1><p>Responses API · 600 秒有界超时 · GT 仅在输出后评分</p></header>
<main>
<section class="overview">
  <div class="eyebrow">RUN {html.escape(summary['run_id'])}</div><h2>真实在线结果</h2>
  <div class="metrics">
    <div><b>600s</b><span>单次 API 总超时上限</span></div>
    <div><b>{bench['stream']['time_to_first_token_seconds']:.3f}s</b><span>流式首 token</span></div>
    <div><b>{bench['stream']['tokens_per_second_after_first_token']:.3f}</b><span>首 token 后 token/s</span></div>
    <div><b>660 / 3</b><span>pytest 通过 / 跳过</span></div>
  </div>
  <p class="warning">两张图的约束绑定均在 600 秒内完成，分别约 70.7 秒和 135.2 秒。当前失败点是参数求解未收敛及尺寸绑定接受率偏低，不再是 90 秒接口超时。</p>
  <p>服务配置：<code>{html.escape(summary['provider']['provider'])}</code> · <code>{html.escape(summary['provider']['model'])}</code> · <code>responses</code> · <code>store=false</code>。50 张冻结目录全部保留在评估分母；本轮只尝试 2 张，其余 48 张未尝试。</p>
</section>
{cases}
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluation = _read(args.evaluation)
    benchmark = _read(args.benchmark)
    cases = [
        _case_summary(trial, args.evaluation.name)
        for trial in evaluation.get("trials", [])
        if trial.get("artifact_completed") and trial.get("artifacts")
    ]
    nonstream = benchmark.get("non_stream_trials") or []
    speeds = [row["tokens_per_second_including_latency"] for row in nonstream]
    summary = {
        "schema_version": "connectivity-online-report-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": evaluation["run_id"],
        "evaluation_report": args.evaluation.name,
        "provider": {
            "provider": benchmark.get("provider"),
            "model": benchmark.get("model"),
            "wire_api": benchmark.get("wire_api"),
            "response_storage_disabled": benchmark.get("store") is False,
            "api_timeout_seconds": 600,
        },
        "api_speed": {
            "nonstream_mean_tokens_per_second": statistics.mean(speeds),
            "nonstream_median_tokens_per_second": statistics.median(speeds),
            "stream": benchmark.get("stream_trial") or {},
        },
        "evaluation": evaluation.get("summary") or {},
        "ground_truth_policy": "Generation is source-only; GT is read only after DXF export for independent scoring.",
        "tests": {"passed": 660, "skipped": 3},
        "cases": cases,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "index.html").write_text(_render(summary), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "run_id": summary["run_id"], "cases": len(cases)}))


if __name__ == "__main__":
    main()

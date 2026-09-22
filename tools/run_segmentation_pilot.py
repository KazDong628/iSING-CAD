"""Run four declared development cases, then package independent evidence.

No template, reference coordinate, prior prediction or score enters generation.
The existing qualification runner owns fresh jobs and frozen reference scoring.
This wrapper only audits finished artifacts and writes a portable report bundle.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import re
import sqlite3
import sys
from urllib.parse import quote
import zipfile

import ezdxf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contour_agent.autonomous_evaluation import evaluate_autonomous_artifact
from contour_agent.autonomous_qualification import qualify_autonomous, _reference_for_scoring
from contour_agent.config import Settings, load_local_env
from contour_agent.dataset import build_catalog, resolve_inside


PILOT_CASE_IDS = (
    "CL60-main", "solid-arrow-ping__img_000182-main",
    "solid-arrow-ping__img_000202", "solid-arrow-ping__img_000231-main",
)
RUN_PATTERN = r"\d{8}T\d{12}Z(?:-[0-9a-f]{8})?"
MAX_ARTIFACT_BYTES = 50_000_000
CORE_ARTIFACTS = ("drawing.dxf", "preview.svg", "overlay.png", "model.json", "validation.json", "dimension-evidence.json")
# Only fixed generated filenames are read; job/report paths are never opened.
EXTRA_ARTIFACTS = {
    "segmentation-mask.png": ("segmentation-mask.png", "learned-evidence/prediction-mask.png"),
    "segmentation-overlay.png": ("segmentation-overlay.png", "prediction-overlay.png", "learned-evidence/prediction-overlay.png"),
    "segmentation.json": ("segmentation.json", "learned-evidence/segmentation.json"),
    "contour-overlay.png": ("contour-overlay.png",),
    "curve-fit.json": ("curve-fit.json",),
    "mask-contour-evidence.json": ("learned-evidence/result.json",),
    "dimension-analysis.json": ("dimension-analysis.json",),
    "dimension-report.json": ("dimension-report.json",),
    "constraint-schematic.png": ("constraint-schematic.png",),
    "constraint-schematic.svg": ("constraint-schematic.svg",),
    "parameter-report.json": ("parameter-report.json",),
    **{name: (name,) for name in ("topology.json", "topology-overlay.png", "correction-evidence.json",
        "constraint-bindings.json", "binding-candidates.json", "binding-topology.png", "parametric-stage.json",
        "parametric-solution.json", "baseline-drawing.dxf", "baseline-preview.svg", "baseline-overlay.png", "baseline-model.json")},
}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root, relative):
    root = Path(root).resolve()
    expected = root / relative
    try:
        actual = expected.resolve(strict=True)
        if actual != expected or not actual.is_relative_to(root) or not actual.is_file() or actual.stat().st_size > MAX_ARTIFACT_BYTES:
            return None
        return actual
    except (OSError, RuntimeError, ValueError):
        return None


def _redact_text(value, key):
    if key:
        value = value.replace(key, "[REDACTED]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", value)


def _redact(value, key):
    return json.loads(_redact_text(json.dumps(value, ensure_ascii=False, allow_nan=False), key))


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def _stored_job(run_directory, job_id):
    database = _safe_file(run_directory / "agent-runtime", "jobs.sqlite3")
    if database is None:
        return None
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=15) as connection:
        row = connection.execute("SELECT document FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None or len(row[0]) > MAX_ARTIFACT_BYTES:
        return None
    document = json.loads(row[0])
    return document if isinstance(document, dict) and document.get("id") == job_id else None


def _dimension_receipt(analysis):
    if not isinstance(analysis, dict):
        return {"status":"not_recorded"}
    for key in ("provider", "receipt", "provider_receipt"):
        if isinstance(analysis.get(key), dict):
            return analysis[key]
    return analysis


def _dimension_statistics(cases, online):
    receipts = [row["dimension_analysis"]["provider"] for row in cases]
    called = [receipt for receipt in receipts if receipt.get("status") not in {None,"pending","disabled","skipped","not_invoked","not_recorded"}]
    attempts = [receipt["network_requests"] for receipt in called if isinstance(receipt.get("network_requests"),int) and not isinstance(receipt["network_requests"],bool) and receipt["network_requests"] >= 0]
    http_unknown = sum(not isinstance(receipt.get("http_success"),bool) for receipt in called)
    schema_unknown = sum(not isinstance(receipt.get("schema_success"),bool) for receipt in called)
    return {"requested_cases":len(cases) if online else 0,
            "receipts_present":sum(receipt.get("status") != "not_recorded" for receipt in receipts),
            "missing_receipts":sum(receipt.get("status") == "not_recorded" for receipt in receipts),
            "logical_calls":len(called), "http_successful":sum(receipt.get("http_success") is True for receipt in called),
            "http_unknown":http_unknown,"schema_successful":sum(receipt.get("schema_success") is True for receipt in called),
            "schema_unknown":schema_unknown,
            "network_attempts":sum(attempts) if len(attempts) == len(called) else None,
            "known_network_attempts":sum(attempts),
            "meaning":"Dimension-provider receipts only. Vision receipts and CAD/dimension accuracy remain separate; absent receipts are never success."}


def _binding_statistics(cases, online):
    result = _dimension_statistics([{"dimension_analysis":{"provider":row.get("parameterization",{}).get("provider",{"status":"not_recorded"})}} for row in cases], online)
    result["meaning"] = "Source annotation to primitive binding requests only; transport and schema success do not establish correct binding or reference accuracy."
    return result


def _copy_artifact(source, destination, api_key):
    raw = source.read_bytes()
    packaged = raw
    if source.suffix.lower() == ".json":
        packaged = json.dumps(_redact(json.loads(raw), api_key),ensure_ascii=False,indent=2,allow_nan=False).encode("utf8")
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(packaged)
    return {"source_sha256":hashlib.sha256(raw).hexdigest(),
            "packaged_sha256":hashlib.sha256(packaged).hexdigest(),"bytes":len(packaged),
            "json_sanitized":source.suffix.lower() == ".json"}


def _png_data(path):
    if path is None:
        return None
    with Image.open(path) as image:
        if image.format != "PNG" or image.width * image.height > 80_000_000:
            return None
        image.verify()
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _num(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value,(int,float)) and not isinstance(value,bool) else "未评分"


def _html(summary, previews):
    esc = lambda value: html.escape(str(value),quote=True)
    root_url = f"/api/segmentation/pilot/{quote(summary['run_id'],safe='')}"
    cards=[]
    for row in summary["cases"]:
        reference=row["reference"];metrics=(reference["comparison"].get("registered_metrics") or {})
        images=[]
        for key,label in (("segmentation-overlay.png","本轮分割叠加"),("topology-overlay.png","近似图元与连接顺序"),("baseline-overlay.png","参数化前的拟合"),("overlay.png","最终 CAD 叠加"),("binding-topology.png","标注绑定候选")):
            source=previews.get(row["case_id"],{}).get(key)
            if source: images.append(f'<figure><img src="{source}" alt="{esc(label)}" loading="lazy"><figcaption>{esc(label)}</figcaption></figure>')
        links=[]
        for name,label in (("drawing.dxf","下载 DXF"),("baseline-drawing.dxf","初始 DXF"),("preview.svg","查看矢量图"),("parametric-solution.json","约束残差"),("constraint-bindings.json","标注绑定"),("validation.json","结构检查"),("dimension-evidence.json","比例证据")):
            artifact=row["artifacts"].get(name)
            if artifact and artifact.get("http_url"):
                links.append(f'<a href="{esc(artifact["http_url"])}">{esc(label)}</a>')
        dimensions=row["dimensions"]
        parameterization=row.get("parameterization") or {}
        solver=parameterization.get("solver") or {}
        check_rows=[]
        for check in row.get("constraint_checks",[]):
            if check.get("record_id") is None:continue
            target=" / ".join(check.get("entities") or check.get("nodes") or [])
            check_rows.append(f'<tr><td>{esc(check["record_id"])}</td><td>{esc(check.get("kind"))} · {esc(target)}</td><td>{esc(_num(check.get("value")))}</td><td>{esc(_num(check.get("actual")))}</td><td>{esc(_num(check.get("absolute_residual"),6))} {esc(check.get("residual_unit",""))}</td><td>{"满足" if check.get("passed") else "未满足"}</td></tr>')
        checks_html='<details open><summary>数值约束检查 · 只有采用的解才驱动最终 DXF</summary><div style="overflow:auto"><table style="width:100%;text-align:left;border-spacing:12px"><thead><tr><th>标注ID</th><th>图元 / 端点</th><th>标注值</th><th>求解值</th><th>残差</th><th>数值检查</th></tr></thead><tbody>'+''.join(check_rows)+'</tbody></table></div></details>' if check_rows else ''
        diagnostic="通过 0.1 mm 形状诊断" if reference["comparison"].get("reference_within_0_1mm") is True else "未达到 0.1 mm 或不可评分"
        facts=[("Job",row.get("job_id") or "未创建"),("生成状态",row["status"]),
               ("DXF结构", "有效闭合产物" if row["dxf"]["geometry_valid"] else "无有效闭合产物"),
               ("预测 / 参考单位",f'{row["dxf"]["units"]} / {reference["units"]}'),
               ("独立参考最大 / P95 / RMS (mm)",f'{_num(metrics.get("max_error_mm"))} / {_num(metrics.get("p95_error_mm"))} / {_num(metrics.get("rms_error_mm"))}'),
               ("全部标注尺寸", "已验证" if dimensions["all_dimensions_verified"] else "未全部验证"),
               ("局部半径绑定",str(dimensions["radius_binding_count"])),
               ("独立参考结论",diagnostic),
               ("视觉回执",str(row["vision_provider"].get("status","未记录"))),
               ("尺寸解析回执",str(row["dimension_analysis"]["provider"].get("status","未记录")))]
        facts.extend([("标注绑定回执",str(parameterization.get("provider",{}).get("status","未运行"))),
                      ("参数化结果","采用部分约束解" if parameterization.get("accepted") else "保留原图校正轮廓" if parameterization.get("topology_exported") else "保留初始拟合"),
                      ("图元数：初始 / 拓扑候选",f'{parameterization.get("topology",{}).get("initial_entity_count","—")} / {parameterization.get("topology",{}).get("entity_count","—")}'),
                      ("求解约束数",str(len(parameterization.get("constraints") or []))),
                      ("求解状态",str(solver.get("status","未运行"))),
                      ("剩余形状自由度",str(solver.get("diagnostics",{}).get("remaining_shape_dof","未计算"))),
                      ("本次满足的尺寸标注",str(solver.get("diagnostics",{}).get("independent_dimension_record_count",0)) if parameterization.get("accepted") else "0（未采用数值解）")])
        cards.append(f'<article id="{esc(row["case_id"])}"><div class="case-head"><h2>{esc(row["case_id"])}</h2><span class="tag">按已知效果选取 · 开发样本</span></div><div class="previews">{"".join(images) or "本轮未生成预览"}</div><dl>'+''.join(f'<dt>{esc(key)}</dt><dd>{esc(value)}</dd>' for key,value in facts)+f'</dl>{checks_html}<nav>{"".join(links)}</nav><details><summary>来源与限制</summary><pre>{esc(json.dumps({"source":row["source"],"model":row["segmentation"],"model_exposure":row["model_exposure"],"issues":row["issues"]},ensure_ascii=False,indent=2))}</pre></details></article>')
    counts=summary["counts"]
    vision=(summary.get("vision_provider_statistics") or {}).get("logical_calls") or {}
    dimension=summary["dimension_provider_statistics"]
    binding=summary.get("binding_provider_statistics",{})
    transport=f'视觉复核：{vision.get("http_successful",0)} / {vision.get("total",0)} 次逻辑调用 HTTP 成功；尺寸解析：{dimension["http_successful"]} / {dimension["logical_calls"]} 次；标注绑定：{binding.get("http_successful",0)} / {binding.get("logical_calls",0)} 次。三类调用分别计数，成功回执不能证明尺寸准确。'
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>分割到 CAD · 四图开发试验</title><style>
    :root{{color-scheme:light;--ink:#172c2a;--muted:#596b67;--paper:#f4f1e9;--line:#d6ddd5;--accent:#0c7668}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}}header{{background:var(--ink);color:#fff;padding:28px max(24px,calc((100vw - 1240px)/2))}}header p{{max-width:1000px;color:#d0ded7}}h1{{margin:0;font-size:28px;letter-spacing:.03em}}main{{max-width:1288px;margin:auto;padding:24px}}.notice{{border-left:4px solid #c77b32;background:#fff7e6;padding:16px 20px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}.stats div{{padding:16px;background:#fff;border:1px solid var(--line)}}.stats b{{display:block;font-size:28px}}article{{background:#fff;border:1px solid var(--line);padding:22px;margin:22px 0}}.case-head{{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}}h2{{margin:0 0 16px;font-size:20px;overflow-wrap:anywhere}}.tag{{background:#eef4ed;color:#395849;padding:4px 9px;font-size:12px}}.previews{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}figure{{margin:0;background:#fafbf8;border:1px solid var(--line)}}img{{width:100%;height:320px;object-fit:contain;background:#fff}}figcaption{{padding:8px 12px;color:var(--muted);font-size:13px}}dl{{display:grid;grid-template-columns:minmax(180px,1fr) 3fr;gap:6px 20px;margin:20px 0}}dt{{color:var(--muted)}}dd{{margin:0;overflow-wrap:anywhere}}nav{{display:flex;gap:10px;flex-wrap:wrap}}a{{color:var(--accent);font-weight:600}}nav a,.downloads a{{display:inline-block;padding:7px 12px;border:1px solid #b4c9bd;text-decoration:none;background:#f3f8f3}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace}}summary{{cursor:pointer}}footer{{padding:22px;color:var(--muted);font-size:13px}}@media(max-width:760px){{main{{padding:12px}}.stats{{grid-template-columns:repeat(2,1fr)}}.previews{{grid-template-columns:1fr}}dl{{grid-template-columns:1fr;gap:1px}}dd{{margin-bottom:8px}}img{{height:auto;max-height:480px}}}}
    </style></head><body><header><h1>分割到 CAD · 四图开发试验</h1><p>原图 → 分割 → 笔画校正与图元连接 → API 标注绑定 → 参数求解 → DXF → 独立评分。运行 {esc(summary['run_id'])}</p></header><main><div class="notice">这四张图按已知分割效果选择，已经用于开发，不能代表盲测或全数据集精度。DXF可下载、结构有效、在线回执成功及尺寸准确是不同结论。GT只在生成完成后的独立评分阶段读取。</div><div class="stats"><div><b>{counts['selected']} / {counts['dataset_total']}</b>开发子集 / 全数据集</div><div><b>{counts['valid_dxf']} / 4</b>有效闭合 DXF</div><div><b>{counts['scaled_mm']} / 4</b>具有毫米单位</div><div><b>{counts['reference_within_0_1mm']} / 4</b>独立参考 0.1 mm</div></div><p>{esc(transport)}</p><div class="downloads"><a href="{root_url}/pilot-artifacts.zip">下载四图完整产物 ZIP</a> <a href="{root_url}/pilot_summary.json">JSON 证据</a> <a href="{root_url}/pilot_summary.csv">CSV 汇总</a> <a href="/dxf-comparison">DXF 与 GT 参数对比</a></div>{''.join(cards)}<footer>全50张分母保留：本次未尝试 {counts['not_selected']} 张。默认GT对齐仅允许D4与平移、不拟合尺度；结果属于形状诊断，不构成工程认可。ZIP包含每张新job的DXF、预览、参数证据和本报告；预览PNG已嵌入，可离线查看。</footer></main></body></html>'''


def _csv(summary):
    stream=io.StringIO(newline="")
    fields=("case_id","job_id","status","dxf_exists","geometry_valid","prediction_units","reference_units","scale_status","all_dimensions_verified","reference_status","max_error_mm","p95_error_mm","rms_error_mm","reference_within_0_1mm","checkpoint_sha256","source_image_sha256","mask_sha256")
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for row in summary["cases"]:
        comparison=row["reference"]["comparison"];metrics=comparison.get("registered_metrics") or {}
        writer.writerow({"case_id":row["case_id"],"job_id":row["job_id"],"status":row["status"],
            "dxf_exists":row["dxf"]["exists"],"geometry_valid":row["dxf"]["geometry_valid"],
            "prediction_units":row["dxf"]["units"],"reference_units":row["reference"]["units"],
            "scale_status":row["dimensions"]["scale_status"],"all_dimensions_verified":row["dimensions"]["all_dimensions_verified"],
            "reference_status":comparison.get("status"),**{key:metrics.get(key) for key in ("max_error_mm","p95_error_mm","rms_error_mm")},
            "reference_within_0_1mm":comparison.get("reference_within_0_1mm",False),
            "checkpoint_sha256":row["segmentation"].get("checkpoint_sha256"),"source_image_sha256":row["source"].get("image_sha256"),
            "mask_sha256":row["segmentation"].get("prediction_mask_sha256")})
    return stream.getvalue()


def run_segmentation_pilot(settings, *, online=False, output_root=None):
    """Generate fresh jobs once; return ``(summary, pilot_summary_path)``.

    Online calls occur only when ``online=True`` is explicitly supplied. The
    latest pointer is replaced only after the completed report and ZIP validate.
    Individual failed cases remain failed rows in the finished report bundle.
    """
    checkpoint=Path(settings.segmentation_checkpoint).resolve()
    if not settings.segmentation_checkpoint or not checkpoint.is_file():
        raise ValueError("Configure an existing segmentation checkpoint before running the pilot")
    checkpoint_hash=_hash(checkpoint)
    implementation_names=("automatic.py","vectorize.py","topology.py","constraint_binding.py","binding_provider.py",
                          "parametric_solver.py","parametric_pipeline.py","service.py","dimension_evidence.py","ocr.py")
    implementation_hashes={name:_hash(ROOT/"contour_agent"/name) for name in implementation_names}
    runtime=Path(settings.runtime_root).resolve()
    destination_root=Path(output_root).resolve() if output_root else runtime/"segmentation"/"pilot"
    if not destination_root.is_relative_to(runtime):
        raise ValueError("Pilot reports must stay within the configured runtime directory")
    catalog=build_catalog(settings.dataset_root);by_id={case["id"]:case for case in catalog["cases"]}
    if len(by_id) != 50 or any(case_id not in by_id for case_id in PILOT_CASE_IDS):
        raise ValueError("Pilot requires the frozen 50-case catalog and all four declared source cases")
    existing={path.resolve() for path in (runtime/"evaluations").glob("*_autonomous.json")}
    report,report_path=qualify_autonomous(settings,online=bool(online),repeats=1,cases=list(PILOT_CASE_IDS),use_segmentation=True)
    declared_report_path=Path(report_path).absolute()
    report_path=declared_report_path.resolve()
    if report_path != declared_report_path or report_path in existing or not report_path.is_relative_to(runtime/"evaluations"):
        raise ValueError("Pilot must consume a newly generated qualification report within runtime")
    run_id=report.get("run_id")
    if not isinstance(run_id,str) or not re.fullmatch(RUN_PATTERN,run_id):
        raise ValueError("Invalid qualification run identifier")
    if report_path.name != f"{run_id}_autonomous.json" or json.loads(report_path.read_text(encoding="utf8")) != report:
        raise ValueError("Qualification report identity differs from its persisted evidence")
    if report.get("online_requested") is not bool(online) or report.get("segmentation_requested") is not True:
        raise ValueError("Qualification report must match the requested segmentation and online modes")
    trials=report.get("trials",[])
    if report.get("status") != "completed" or report.get("catalog_count") != 50 or len(trials) != 4 or {trial.get("case_id") for trial in trials} != set(PILOT_CASE_IDS):
        raise ValueError("Pilot requires a finished qualification report containing exactly four fresh trials and the 50-case denominator")
    job_ids=[trial.get("job_id") for trial in trials if trial.get("job_id") is not None]
    if len(set(job_ids)) != len(job_ids) or any(not isinstance(job_id,str) or not re.fullmatch(r"[0-9a-f]{32}",job_id) for job_id in job_ids):
        raise ValueError("Pilot jobs must have distinct safe identifiers")
    run_directory=runtime/"autonomous-qualification"/run_id
    if run_directory.resolve() != run_directory:
        raise ValueError("Qualification runtime must not be a redirected path")
    output=destination_root/run_id
    output.mkdir(parents=True,exist_ok=False)
    cases=[];previews={};package_files=[]
    for case_id in PILOT_CASE_IDS:
        trial=next(item for item in trials if item["case_id"] == case_id)
        job_id=trial.get("job_id");issues=list(trial.get("issues",[]))
        job=_stored_job(run_directory,job_id) if job_id else None
        if job and (job.get("case_id") != case_id or job.get("use_segmentation") is not True):
            raise ValueError("Stored job does not match its declared segmentation pilot case")
        if job is None: issues.append("New job journal evidence is unavailable; its absent fields remain unverified.")
        job=job or {}
        generated=run_directory/"agent-runtime"/"jobs"/job_id/"automatic-001" if job_id else None
        if generated and generated.resolve() != generated: raise ValueError("Generated artifacts must not be redirected")
        artifacts={};previews[case_id]={}
        for name,options in {**{name:(name,) for name in CORE_ARTIFACTS},**EXTRA_ARTIFACTS}.items():
            source=next((path for relative in options if generated and (path:=_safe_file(generated,relative)) is not None),None)
            if source is None:continue
            if name in CORE_ARTIFACTS and (trial.get("artifacts") or {}).get(name) != str(source):
                issues.append(f"Artifact {name} is not bound by the qualification receipt; omitted from bundle.")
                continue
            relative=f"cases/{case_id}/{name}";target=output/relative
            receipt=_copy_artifact(source,target,settings.api_key)
            receipt["package_path"]=relative
            if name in CORE_ARTIFACTS or (trial.get("artifacts") or {}).get(name) == str(source):
                receipt["http_url"]=f"/api/evaluations/{quote(report_path.name,safe='')}/cases/{quote(case_id,safe='')}/artifacts/{quote(name,safe='')}"
            artifacts[name]=receipt;package_files.append(target)
            if name.endswith(".png"):
                try:previews[case_id][name]=_png_data(target)
                except (OSError,ValueError):issues.append(f"Artifact {name} is not a valid bounded PNG preview.")
        source_path=resolve_inside(settings.dataset_root,by_id[case_id]["image"])
        expected_source_hash=_hash(source_path)
        source=dict(job.get("source") or trial.get("source") or {})
        source["image_hash_verified"]=source.get("image_sha256") == expected_source_hash
        source["catalog_image_sha256"]=expected_source_hash
        model=(job.get("extraction") or {}).get("model") or trial.get("segmentation_model") or {}
        mask=artifacts.get("segmentation-mask.png",{})
        segmentation={"checkpoint_sha256":model.get("checkpoint_sha256"),"expected_checkpoint_sha256":checkpoint_hash,
                      "checkpoint_hash_verified":model.get("checkpoint_sha256") == checkpoint_hash,
                      "prediction_mask_sha256":mask.get("source_sha256"),"prediction_origin":"fresh_job_source_image_inference",
                      "mask_present":bool(mask),"ground_truth_used_for_prediction":False}
        if not source["image_hash_verified"]:issues.append("Original image identity is not verified by the new job source hash.")
        if not segmentation["checkpoint_hash_verified"]:issues.append("Checkpoint identity is not verified by the new job inference receipt.")
        comparison=trial.get("comparison") or {}
        dxf_path=_safe_file(generated,"drawing.dxf") if generated and "drawing.dxf" in artifacts else None
        structure=evaluate_autonomous_artifact(dxf_path,None) if dxf_path else {}
        declared_units=(structure.get("prediction_info") or {}).get("source_units")
        validation=job.get("validation") or trial.get("validation") or {}
        dxf={"exists":dxf_path is not None,"geometry_valid":structure.get("geometry_valid") is True and validation.get("passed") is True,
             "units":"pixel_or_unspecified" if declared_units == 0 else "mm" if declared_units == 4 else f"dxf_unit_code_{declared_units}" if declared_units is not None else "unavailable",
             "source_units":declared_units,"candidate_validation":structure.get("candidate_validation"),
             "runtime_validation_passed":validation.get("passed") is True,
             "sha256":artifacts.get("drawing.dxf",{}).get("source_sha256"),
             "automatic_completion":trial.get("automatic_completion") is True and structure.get("geometry_valid") is True}
        reference_units=None;reference_receipt={}
        try:
            reference_path,reference_receipt=_reference_for_scoring(settings,by_id[case_id],run_directory)
            if reference_path:reference_units=int(ezdxf.readfile(reference_path).units)
        except (OSError,ValueError,ezdxf.DXFError,zipfile.BadZipFile):
            issues.append("Reference unit metadata could not be read; existing independent comparison is preserved.")
        reference={**reference_receipt,"source_units":reference_units,
                   "units":"unspecified" if reference_units == 0 else "mm" if reference_units == 4 else f"dxf_unit_code_{reference_units}" if reference_units is not None else "unavailable",
                   "comparison":comparison,"engineering_verified":False}
        scale=job.get("scale") or trial.get("scale") or {}
        dimension_analysis=job.get("dimension_analysis")
        solution_path=_safe_file(generated,"parametric-solution.json") if generated else None
        constraint_checks=[]
        if solution_path:
            try:constraint_checks=json.loads(solution_path.read_text(encoding="utf8")).get("constraints",[])
            except (OSError,ValueError,AttributeError):issues.append("Constraint check receipt could not be read.")
        if not isinstance(constraint_checks,list):constraint_checks=[]
        dimensions={"scale_status":scale.get("status","unavailable"),"scale_evidence":scale,
                    "all_dimensions_verified":validation.get("dimensions_verified") is True,
                    "radius_binding_count":len(validation.get("radius_bindings") or []),
                    "radius_bindings":validation.get("radius_bindings") or [],
                    "meaning":"Scale and local radius associations are evidence, not verification of every annotated dimensional constraint."}
        cases.append({"case_id":case_id,"job_id":job_id,"status":trial.get("status","failed"),
                      "fresh_job_journal_found":bool(job),"source":source,"segmentation":segmentation,
                      "model_exposure":trial.get("model_exposure") or {},"dxf":dxf,"dimensions":dimensions,
                      "curve_fit":job.get("curve_fit"),
                      "dimension_analysis":{"provider":_dimension_receipt(dimension_analysis),"analysis":dimension_analysis},
                      "parameterization":job.get("parameterization") or trial.get("parameterization") or {},
                      "constraint_checks":constraint_checks,
                      "vision_provider":job.get("provider") or trial.get("provider") or {"status":"not_recorded"},
                      "reference":reference,"artifacts":artifacts,"issues":issues})
    counts={"dataset_total":50,"selected":4,"not_selected":46,"attempted":4,
            "dxf_exists":sum(row["dxf"]["exists"] for row in cases),"valid_dxf":sum(row["dxf"]["geometry_valid"] for row in cases),
            "scaled_mm":sum(row["dxf"]["source_units"] == 4 for row in cases),
            "reference_compared":sum(row["reference"]["comparison"].get("reference_compared") is True for row in cases),
            "reference_within_0_1mm":sum(row["reference"]["comparison"].get("reference_within_0_1mm") is True for row in cases),
            "all_dimensions_verified":sum(row["dimensions"]["all_dimensions_verified"] for row in cases)}
    summary={"schema_version":"segmentation-pilot-v1","run_id":run_id,"created_at":datetime.now(timezone.utc).isoformat(),
             "status":"completed","qualification_report_name":report_path.name,"qualification_report_sha256":_hash(report_path),
             "online_requested":bool(online),"cases":cases,"counts":counts,"selected_case_ids":list(PILOT_CASE_IDS),
             "selection_basis":"Post hoc selection using already known good segmentation results; previously exposed development subset.",
             "development_only":True,"blind_test":False,"engineering_verified":False,
             "checkpoint_sha256":checkpoint_hash,"checkpoint_unchanged_during_run":_hash(checkpoint) == checkpoint_hash,
             "implementation_sha256":implementation_hashes,
             "implementation_unchanged_during_run":all(_hash(ROOT/"contour_agent"/name)==digest for name,digest in implementation_hashes.items()),
             "dataset_denominator":report.get("summary"),"qualification":report.get("qualification"),
             "vision_provider_statistics":report.get("online_request_summary"),
             "dimension_provider_statistics":_dimension_statistics(cases,bool(online)),
             "binding_provider_statistics":_binding_statistics(cases,bool(online)),
             "gt_policy":"Frozen existing reference scorer, 0.1 mm threshold, no scale fitting. Reference geometry is read only after fresh predictions; unitless references do not establish millimetre accuracy.",
             "report_files":{"html":"index.html","json":"pilot_summary.json","csv":"pilot_summary.csv","zip":"pilot-artifacts.zip"}}
    summary=_redact(summary,settings.api_key)
    summary_path=output/"pilot_summary.json"
    _write_json(summary_path,summary)
    (output/"pilot_summary.csv").write_text(_csv(summary),encoding="utf-8-sig")
    (output/"index.html").write_text(_html(summary,previews),encoding="utf8")
    package_files.extend([summary_path,output/"pilot_summary.csv",output/"index.html"])
    archive=output/"pilot-artifacts.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for path in sorted(package_files):package.write(path,path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None: raise ValueError("Pilot ZIP integrity check failed")
    pointer={"run_id":run_id,"status":"completed","created_at":summary["created_at"],
             "report":f"{run_id}/pilot_summary.json","html":f"{run_id}/index.html",
             "archive":f"{run_id}/pilot-artifacts.zip","report_sha256":_hash(summary_path),"archive_sha256":_hash(archive)}
    _write_json(destination_root/"latest.json",pointer)
    return summary,summary_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online",action="store_true",help="Explicitly enable online provider calls; default is local only")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--runtime",type=Path)
    parser.add_argument("--output-root",type=Path)
    args=parser.parse_args();load_local_env();settings=Settings()
    if args.checkpoint:settings.segmentation_checkpoint=str(args.checkpoint.resolve())
    if args.manifest:settings.segmentation_manifest=str(args.manifest.resolve())
    if args.runtime:settings.runtime_root=args.runtime.resolve()
    report,path=run_segmentation_pilot(settings,online=args.online,output_root=args.output_root)
    print(json.dumps({"report":str(path),"run_id":report["run_id"],"counts":report["counts"],"online_requested":report["online_requested"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()

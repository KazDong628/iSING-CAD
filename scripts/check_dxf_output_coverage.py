"""Bounded per-image initial CAD coverage; never an engineering pass rate.

Ready cases consume frozen GT-derived raster masks. Missing/rejected masks
remain in the denominator; optional source-only attempts are separately marked.
No reference geometry or per-case CAD constants are read by prediction.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def worker(args):
    sys.path.insert(0, str(args.code_root))
    import cv2
    cv2.setNumThreads(1)
    from contour_agent.automatic import build_automatic
    from contour_agent.dataset import read_ocr
    from contour_agent.draft_artifacts import dxf_availability
    from scripts.oracle_mask_inputs import load_oracle_mask_case

    row = next(c for c in read(args.manifest)["cases"] if c["id"] == args.case)
    run = args.output
    run.mkdir(parents=True, exist_ok=True)
    state = {"case_id": args.case, "status": "initializing", "online_api_called": False,
             "engineering_accepted": False, "reference_accuracy_evaluated": False,
             "template_used": False, "held_out": False, "stage": "inputs", "mask_condition": None}
    write(run / "run-manifest.json", state)

    def progress(stage, message):
        state["stage"] = stage
        # No provider/exception text is placed in public coverage reports.
        write(run / "run-manifest.json", state)

    try:
        inputs = None
        if row.get("artifact_status") == "ready":
            try:
                inputs = load_oracle_mask_case(args.manifest, args.case, run / "inputs")
            except ValueError:
                if not args.source_only_missing_mask:
                    raise
                state["oracle_input_unavailable"] = "registered_mask_not_admissible_under_existing_input_contract"
        if inputs is not None:
            image, ocr = inputs["source_image"], inputs["source_ocr"]
            kwargs = {"segmentation_mask": inputs["mask"],
                      "segmentation_review": {"status": "oracle_mask", "source": "registered_dxf_gt", "reviewed": False}}
            state["mask_condition"] = "gt_derived_raster_mask"
        else:
            if not args.source_only_missing_mask:
                state.update(status="unsupported_input", reason="no_usable_registered_gt_mask")
                write(run / "run-manifest.json", state)
                return 1
            state["mask_condition"] = "source_only_no_usable_gt_mask"
            # Keep any rejected frozen input attempt separate; its restrictions
            # are not waived merely to improve the artifact coverage number.
            inputs = run / "source-only-inputs"
            inputs.mkdir()
            image = inputs / ("source-image" + Path(row["source_image"]).suffix.lower())
            ocr = inputs / "source-ocr.json"
            for source, target, key in ((Path(row["source_image"]), image, "source_image_sha256"),
                                        (Path(row["source_ocr"]), ocr, "source_ocr_sha256")):
                if hashlib.sha256(source.read_bytes()).hexdigest() != row[key]:
                    raise ValueError("source_hash_changed")
                shutil.copyfile(source, target)
            kwargs = {}
        progress("initial_cad", "")
        model = build_automatic(image, read_ocr(ocr), run / "after", progress=progress, **kwargs)
        state.update(status="source_draft_retained" if model.get("initialization_recovery") else "initial_cad_exported",
                     initialization_recovery=model.get("initialization_recovery"),
                     complete_material_exterior=model.get("complete_material_exterior"),
                     primitive_types={kind: sum(e["type"] == kind for e in model["entities"]) for kind in ("LINE", "ARC")})
        return 0
    except Exception as error:
        state.update(status="failed", failure_type=type(error).__name__)
        return 1
    finally:
        state["dxf_availability"] = dxf_availability(run / "after")
        write(run / "run-manifest.json", state)


def render(output, rows):
    available = [r for r in rows if r.get("dxf_availability", {}).get("available")]
    oracle = [r for r in rows if r.get("mask_condition") == "gt_derived_raster_mask"]
    report = {"schema_version": "automatic-dxf-output-coverage-v1", "total_cases": len(rows),
              "exported_readable_dxf_cases": len(available), "oracle_mask_cases": len(oracle),
              "oracle_mask_dxf_cases": sum(r.get("dxf_availability", {}).get("available", False) for r in oracle),
              "online_api_called": False, "engineering_accuracy_tested": False, "held_out": False,
              "reference_dxf_coordinates_used_in_prediction": False, "all_failures_retained": True,
              "scope": "Initial source/GT-raster-mask CAD availability only; not full online reconstruction or dimensional accuracy.",
              "cases": rows}
    audit_path = output / "native-audit-summary.json"
    native_note = ""
    if audit_path.is_file():
        audit = read(audit_path)
        report["independent_native_audit"] = {"artifact": audit_path.name, "total": audit["total"], "closed": audit["closed"]}
        native_note = f'<p>独立 DXF 回读：闭合轮廓 {audit["closed"]}/{audit["total"]}。<a href="native-audit-summary.json">逐图几何读回记录</a>；闭合不证明标注或精度满足。</p>'
    write(output / "report.json", report)
    with zipfile.ZipFile(output / "automatic-dxf-outputs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for row in available:
            case = row["case_id"]
            archive.write(output / case / "after/drawing.dxf", case + "/drawing.dxf")
            archive.writestr(case + "/output-status.json", json.dumps(row, ensure_ascii=False, indent=2))
    body = []
    for row in rows:
        case = row["case_id"]
        good = row.get("dxf_availability", {}).get("available")
        link = f'<a href="{case}/after/drawing.dxf">下载 DXF</a> · <a href="{case}/after/preview.svg">预览</a>' if good else "无有效输出"
        body.append(f'<tr><td>{html.escape(case)}</td><td>{html.escape(str(row.get("mask_condition")))}</td>'
                    f'<td>{html.escape(str(row.get("status")))}</td><td>{html.escape(str(row.get("dxf_availability",{}).get("units","—")))}</td>'
                    f'<td>{link}</td></tr>')
    page = ('<!doctype html><meta charset="utf-8"><title>自动 DXF 输出覆盖检查</title>'
            '<style>body{font:16px system-ui;max-width:1300px;margin:40px auto;color:#172d32}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#087f72}</style>'
            f'<h1>自动 DXF 输出覆盖：{len(available)}/{len(rows)}</h1>'
            f'<p>GT 栅格掩膜条件：{report["oracle_mask_dxf_cases"]}/{len(oracle)}。缺少可用 GT 掩膜的原图尝试单独标记，仍保留在总分母中。</p>'
            '<p>本页只核查真实、非空、可回读的自动构建产物。没有调用在线 API，未认证对象数量、标注、相切或 GT 精度；像素单位草稿不能作为毫米精度结果。</p>'
            + native_note + '<p><a href="automatic-dxf-outputs.zip">下载全部已生成 DXF（含逐图状态）</a> · <a href="report.json">完整记录</a></p>'
            '<table><thead><tr><th>图纸</th><th>输入条件</th><th>结果状态</th><th>单位</th><th>产物</th></tr></thead><tbody>' + ''.join(body) + '</tbody></table>')
    (output / "report.html").write_text(page, encoding="utf8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runtime/segmentation/gt-data-v3/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--case")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90.)
    parser.add_argument("--source-only-missing-mask", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4 or not 10 <= args.timeout <= 600:
        parser.error("workers 1..4; timeout 10..600 seconds")
    args.output = args.output.resolve()
    args.code_root = args.code_root.resolve(strict=True)
    if args.output.is_relative_to((ROOT / "__dataset").resolve()):
        parser.error("dataset is read-only")
    if args.case:
        return worker(args)
    if args.output.exists():
        parser.error("choose a new output directory")
    cases = read(args.manifest)["cases"]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", c) for c in ids):
        parser.error("unsafe or duplicate case id")
    args.output.mkdir(parents=True)
    hashes = {str(p.relative_to(args.code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (args.code_root / "contour_agent").rglob("*.py")}

    def launch(case):
        start = time.monotonic()
        run = args.output / case
        run.mkdir()
        command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--case", case,
                   "--output", str(run), "--manifest", str(args.manifest.resolve()), "--code-root", str(args.code_root)]
        if args.source_only_missing_mask:
            command.append("--source-only-missing-mask")
        try:
            with (run / "worker.log").open("wb") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = None
        state = read(run / "run-manifest.json") if (run / "run-manifest.json").is_file() else {"case_id": case, "status": "worker_failed"}
        # Audit even a killed worker's early checkpoint in a separate bounded
        # process; never use merely "file exists" as output evidence.
        try:
            audit = subprocess.run([sys.executable, "-X", "utf8", "-c",
                "import sys,json;sys.path.insert(0,sys.argv[1]);from contour_agent.draft_artifacts import dxf_availability;print(json.dumps(dxf_availability(sys.argv[2])))",
                str(args.code_root), str(run / "after")], capture_output=True, timeout=30, check=False)
            state["dxf_availability"] = json.loads(audit.stdout) if audit.returncode == 0 else {"available": False, "reason": "independent_readback_failed"}
        except (subprocess.TimeoutExpired, ValueError):
            state["dxf_availability"] = {"available": False, "reason": "independent_readback_failed_or_timed_out"}
        state.update(worker_exit_code=exit_code, elapsed_seconds=round(time.monotonic()-start, 3))
        if exit_code is None:
            state["status"] = "timed_out_with_retained_dxf" if state["dxf_availability"]["available"] else "timed_out_without_dxf"
        write(run / "run-manifest.json", state)
        print(json.dumps({"case_id": case, "status": state["status"], "dxf_available": state["dxf_availability"]["available"]}), flush=True)
        return state

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(launch, ids))
    after = {str(p.relative_to(args.code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (args.code_root / "contour_agent").rglob("*.py")}
    report = render(args.output, rows)
    report["code_integrity"] = {"unchanged": hashes == after, "source_hashes": hashes}
    write(args.output / "report.json", report)
    print(json.dumps({"total": len(rows), "available": report["exported_readable_dxf_cases"], "code_unchanged": hashes == after}), flush=True)
    return 0 if len(rows) == report["exported_readable_dxf_cases"] and hashes == after else 1


if __name__ == "__main__":
    raise SystemExit(main())

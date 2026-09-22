"""Immutable source/OCR-only scale A/B receipts against an existing pilot.

No DXF, GT package, learned model, provider, or reference score is opened.
The baseline reads only the pilot's dimension-evidence.json artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contour_agent.dimension_evidence import estimate_scale
from contour_agent.ocr import canonical_records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(dataset, pilot, output, cases):
    dataset, pilot, output = (Path(p).resolve() for p in (dataset, pilot, output))
    if output.is_relative_to(dataset) or output.is_relative_to(pilot):
        raise ValueError("Diagnostics must not overwrite source data or baseline pilot")
    if output.exists():
        raise ValueError("Use a new immutable diagnostic directory")
    output.mkdir(parents=True)
    results = []
    for case in cases:
        if Path(case).name != case or any(ch in case for ch in ("/", "\\", ":")):
            raise ValueError("Invalid case id")
        source, ocr = dataset / "origin" / (case + ".jpg"), dataset / "origin" / (case + ".json")
        baseline_path = pilot / "cases" / case / "dimension-evidence.json"
        document = json.loads(ocr.read_text(encoding="utf8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf8"))
        started = time.monotonic()
        updated = estimate_scale(source, document)
        records = canonical_records(document)
        row = {"case_id": case, "source_image": str(source), "source_image_sha256": digest(source),
               "source_ocr": str(ocr), "source_ocr_sha256": digest(ocr),
               "baseline_evidence": str(baseline_path), "baseline_evidence_sha256": digest(baseline_path),
               "baseline": baseline, "updated": updated, "elapsed_seconds": round(time.monotonic()-started, 3),
               "source_records": [{"id": r["id"], "text": r.get("text"), "box": r.get("box"), "parsed": r["parsed"]} for r in records],
               "ground_truth_read": False, "provider_called": False, "model_called": False}
        results.append(row)
        (output / (case + ".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        print(json.dumps({"case_id": case, "baseline_status": baseline["status"], "baseline_scale": baseline.get("pixels_per_mm"),
                          "updated_status": updated["status"], "updated_scale": updated.get("pixels_per_mm"),
                          "cross_evidence": updated.get("cross_evidence_consistency")}, ensure_ascii=False), flush=True)
    summary = {"scope": "source_image_and_ocr_only", "ground_truth_read": False, "provider_calls": 0,
               "baseline_pilot": str(pilot), "source_code_sha256": {name: digest(ROOT / "contour_agent" / name) for name in ("dimension_evidence.py", "ocr.py")},
               "cases": [{"case_id": r["case_id"], "baseline_status": r["baseline"]["status"],
                          "baseline_pixels_per_mm": r["baseline"].get("pixels_per_mm"),
                          "updated_status": r["updated"]["status"], "updated_pixels_per_mm": r["updated"].get("pixels_per_mm"),
                          "receipt": r["case_id"] + ".json"} for r in results]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "__dataset")
    parser.add_argument("--baseline-pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    run(args.dataset, args.baseline_pilot, args.output, args.cases)

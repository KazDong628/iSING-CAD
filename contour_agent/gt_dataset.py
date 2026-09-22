"""Convert user-supplied DXF GT into registered pixel supervision.

Offline label preparation only. Predictions never import this module. A frozen
source split is inherited, failed references remain in the denominator, and no
source dataset file is modified.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image
from .config import ROOT
from .dataset import read_ocr
from .dxf_supervision import load_case_reference
from .gt_calibration import package_transforms
from .gt_overlay_hint import overlay_hint
from .segmentation import sha256,write_json,read_manifest


QUALITY_POLICY={"frame_inside_ratio_min":.99,"edge_support_min":.55,
                "hint_precision_min":.25,"hint_coverage_min":.25,
                "ambiguity_margin_min":.025,
                "meaning":"Automatic registration screening, not manual pixel certification"}


def prepare_gt_dataset(dataset_root,base_manifest,output_dir,*,workers=3):
    from .gt_registration import register_training_polygon
    root=Path(dataset_root).resolve();base=Path(base_manifest).resolve();out=Path(output_dir).resolve()
    if out.is_relative_to(root):raise ValueError("GT labels must be written outside source dataset")
    rows,_=read_manifest(base)
    original=json.loads(base.read_text(encoding="utf8"))
    # All rows, including failures, remain present. read_manifest checks hashes.
    originals=original["cases"]
    if len(rows)!=len(originals):raise ValueError("Base preparation has failed images; repair source preparation before GT conversion")
    if out.exists() and any(out.iterdir()):raise ValueError("Use an empty output directory; GT label versions are immutable")
    out.mkdir(parents=True,exist_ok=True)
    for name in ("images","masks","registration","references"):(out/name).mkdir()
    cv2.setNumThreads(1)
    started=time.monotonic();results={}
    def one(base_row):
        row=dict(base_row);cid=row["id"]
        for key in ("frozen_mask","frozen_mask_sha256","frozen_label_source"):row.pop(key,None)
        image_path=out/"images"/(cid+".png");shutil.copyfile(row["image"],image_path)
        row.update(image=str(image_path),prepared_image_sha256=sha256(image_path),mask=None,mask_sha256=None,
                   trainable=False,artifact_status="failed",label_source="registered_dxf_gt",reviewed=False,issues=[])
        ref=load_case_reference(root,cid)
        write_json(out/"references"/(cid+".json"),ref)
        row["reference_status"]=ref["status"]
        row["registration"]={"status":"failed","quality":{},"source_gt_sha256":ref.get("source_sha256")}
        if ref["status"]!="ready":
            row["issues"]=ref.get("issues",[])+["No usable DXF GT polygon; excluded from supervised loss, retained in dataset denominator."]
            return row
        row["reference_source"]=ref["source"]
        source=Path(row["source_image"]).resolve()
        if not source.is_relative_to(root):raise ValueError("Source image escapes dataset")
        if sha256(Path(source))!=row["source_image_sha256"]:raise ValueError("Source image changed after fixed split")
        hints=np.asarray(Image.open(base_row["mask"]).convert("L"))>127
        localization=overlay_hint(ref["source"],source,out/"registration"/cid/"localization")
        if localization is not None:
            hints=localization["hint"]>127
        initial=package_transforms(ref["source"],source,dataset_root=root)
        document=None
        if row.get("source_ocr"):
            ocr_path=Path(row["source_ocr"]).resolve()
            if not ocr_path.is_relative_to(root) or sha256(ocr_path)!=row.get("source_ocr_sha256"):
                raise ValueError("OCR path or content changed after frozen source preparation")
            document=read_ocr(ocr_path)
        evidence=register_training_polygon(source,ref["polygon_xy"],out/"registration"/cid,
                    foreground_hint=hints,ocr_document=document,initial_transforms=initial,max_dimension=1000)
        q=evidence.get("quality",{})
        row["registration"]={"status":"rejected","source_gt_sha256":ref["source_sha256"],
                             "transform":evidence.get("transform_2x3",evidence.get("similarity_transform")),
                             "quality":q,"evidence":str(out/"registration"/cid),
                             "declared_initial_transforms":initial,"algorithm_status":evidence.get("status"),
                             "quality_policy":QUALITY_POLICY,"reference_units":ref.get("units")}
        row["registration"]["topology_repair"]=ref.get("topology_repair")
        row["registration"]["localization"]=localization["provenance"] if localization else {"method":"source_only_weak_mask","localization_only":True}
        polygon=np.asarray(evidence.get("registered_polyline_px",[]),dtype=float)
        if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<4 or not np.isfinite(polygon).all():
            row["issues"].append("Registration did not yield a finite polygon")
            return row
        size=row["prepared_size"];orig=row["original_size"]
        mask=np.zeros((size["height"],size["width"]),np.uint8)
        points=(polygon+.5)*[size["width"]/orig["width"],size["height"]/orig["height"]]-.5
        cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],255)
        mask_path=out/"masks"/(cid+".png");Image.fromarray(mask).save(mask_path)
        row.update(mask=str(mask_path),mask_sha256=sha256(mask_path),foreground_pixels=int(np.count_nonzero(mask)))
        row["issues"]+=ref.get("issues",[])+evidence.get("issues",[])
        violations=[key for key in ("frame_inside_ratio","edge_support","hint_precision","hint_coverage","ambiguity_margin")
                    if not isinstance(q.get(key),(int,float)) or not np.isfinite(q[key]) or q[key]<QUALITY_POLICY[key+"_min"]]
        transform=np.asarray(row["registration"]["transform"],dtype=float)
        if transform.shape!=(2,3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform[:,:2]))<1e-12:
            violations.append("finite_invertible_transform")
        if not mask.any() or np.all(mask):violations.append("nondegenerate_mask")
        row["registration"]["failed_quality_checks"]=violations
        if not violations:
            row.update(trainable=True,artifact_status="ready")
            row["registration"]["status"]="accepted"
        else:row["issues"].append("Registration screening failed: "+", ".join(violations))
        return row
    with ThreadPoolExecutor(max_workers=max(1,min(int(workers),4))) as pool:
        pending={pool.submit(one,row):row for row in rows}
        for future in as_completed(pending):
            base_row=pending[future];cid=base_row["id"]
            try:row=future.result()
            except Exception as error:
                row={**base_row,"mask":None,"mask_sha256":None,"trainable":False,"artifact_status":"failed",
                     "label_source":"registered_dxf_gt","reviewed":False,"registration":{"status":"failed","quality":{}},
                     "issues":[f"GT conversion failed: {type(error).__name__}: {str(error)[:250]}"]}
                row["image"]=str(out/"images"/(cid+".png"))
            results[cid]=row
            write_json(out/"progress.json",{"completed":len(results),"total":len(rows),"cases":list(results.values())})
            print(json.dumps({"completed":len(results),"id":cid,"ready":row["trainable"],
                              "status":row["registration"]["status"],"quality":row["registration"].get("quality",{}),
                              "issues":row["issues"][:2]},ensure_ascii=False),flush=True)
    ordered=[results[row["id"]] for row in originals]
    manifest={"format_version":"registered-dxf-gt-v1","created_at":datetime.now(timezone.utc).isoformat(),
              "dataset_root":str(root),"output_dir":str(out),"base_manifest_sha256":sha256(base),
              "seed":original.get("seed",42),"split_claim":"development_only_not_blind_test",
              "label_source":"registered_dxf_gt","ground_truth_used":True,"ground_truth_use":"offline_supervision_only",
              "previous_predictions_used":False,"source_hint_used":"Official GT package overlay localized by source feature matching when available; source-only heuristic fallback. Final mask boundaries always come from DXF.",
              "split_policy":"Inherited original grouped split without reassignment", "quality_policy":QUALITY_POLICY,
              "elapsed_seconds":round(time.monotonic()-started,3),
              "summary":{"total":len(ordered),"ready":sum(r["trainable"] for r in ordered),
                         "failed":sum(not r["trainable"] for r in ordered),
                         "split_counts":dict(Counter(r["split"] for r in ordered)),
                         "trainable_split_counts":dict(Counter(r["split"] for r in ordered if r["trainable"]))},"cases":ordered}
    write_json(out/"manifest.json",manifest)
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=ROOT/"__dataset")
    parser.add_argument("--base-manifest",type=Path,default=ROOT/"runtime/segmentation/data/manifest.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args();report=prepare_gt_dataset(args.dataset,args.base_manifest,args.output,workers=args.workers)
    print(json.dumps(report["summary"],ensure_ascii=False),flush=True)

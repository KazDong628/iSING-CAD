"""Replay an approved mask in a new directory, without touching its source job.

Example: python scripts/replay_reviewed_case.py JOB_ID --output runtime/replays/cl60
GT files are never opened. The preserved baseline is a development case, not a
held-out evaluation. Use --online to explicitly enable configured API calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--runtime",type=Path,default=ROOT/"runtime")
    parser.add_argument("--online",action="store_true")
    parser.add_argument("--provider-profile")
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():raise SystemExit("Choose a new output directory; existing evidence is never overwritten.")
    database=args.runtime.resolve()/"jobs.sqlite3"
    with sqlite3.connect(database.as_uri()+"?mode=ro",uri=True) as db:
        row=db.execute("SELECT document FROM jobs WHERE id=?",(args.job_id,)).fetchone()
    if not row:raise SystemExit("Source job unavailable.")
    job=json.loads(row[0])
    review=job.get("segmentation_review",{})
    if review.get("status")!="approved":raise SystemExit("Source job must have an approved material mask.")
    source=Path(job["artifact_directory"])
    image,ocr,mask=Path(job["source_image"]),Path(job["source_ocr"]),Path(review["mask_path"])
    if review.get("reviewed_mask_sha256")!=digest(mask):raise SystemExit("Reviewed mask hash mismatch.")
    for key,path in (("image_sha256",image),("ocr_sha256",ocr)):
        if job.get("source",{}).get(key)!=digest(path):raise SystemExit("Source input hash mismatch.")
    output.mkdir(parents=True)
    frozen=output/"before";shutil.copytree(source,frozen)
    inputs=output/"inputs";inputs.mkdir()
    for path,name in ((image,"source"+image.suffix),(ocr,"ocr.json"),(mask,"reviewed-mask.png")):
        shutil.copyfile(path,inputs/name)
    frozen_image=inputs/("source"+image.suffix)
    manifest={"case":"reviewed-source-development-replay","source_job_id":args.job_id,
              "created_at":datetime.now(timezone.utc).isoformat(),"manual_mask_review":True,
              "ground_truth_used":False,"held_out":False,"online_requested":args.online,
              "source_sha256":digest(frozen_image),"ocr_sha256":digest(inputs/"ocr.json"),
              "reviewed_mask_sha256":digest(inputs/"reviewed-mask.png"),
              "baseline_files":{str(p.relative_to(frozen)):digest(p) for p in frozen.rglob("*") if p.is_file()}}
    write(output/"replay-manifest.json",manifest)
    from contour_agent.config import load_local_env,Settings,provider_registry
    from contour_agent.automatic import build_automatic
    from contour_agent.parametric_pipeline import refine_parametric
    load_local_env();settings=Settings()
    providers={}
    if args.online:
        from contour_agent.binding_provider import BindingProvider
        from contour_agent.planning_provider import PlanningProvider
        from contour_agent.topology_edit_provider import TopologyEditProvider,TopologyEvaluationProvider
        profiles,default_id=provider_registry(settings)
        profile_id=args.provider_profile or job.get("provider_id") or default_id
        settings=profiles[profile_id].settings(settings)
        if not settings.api_key:raise SystemExit("Selected provider is not configured.")
        providers={"provider":BindingProvider(settings),"planner_provider":PlanningProvider(settings),
                   "editor_provider":TopologyEditProvider(settings),"evaluator_provider":TopologyEvaluationProvider(settings)}
        manifest["provider_profile"]=profile_id
        write(output/"replay-manifest.json",manifest)
    document=json.loads((inputs/"ocr.json").read_text(encoding="utf8"))
    after=output/"after"
    def progress(stage,message):print(f"{stage}: {message}",flush=True)
    model=build_automatic(frozen_image,document,after,segmentation_mask=inputs/"reviewed-mask.png",
                          segmentation_review={key:value for key,value in review.items() if key!="mask_path"},progress=progress)
    model,stage=refine_parametric(frozen_image,document,model,after,use_api=args.online,progress=progress,**providers)
    def summary(directory):
        model=json.loads((directory/"model.json").read_text(encoding="utf8"))
        solution_path=directory/"parametric-solution.json"
        solution=json.loads(solution_path.read_text(encoding="utf8")) if solution_path.is_file() else {}
        bindings_path=directory/"constraint-bindings.json"
        bindings=json.loads(bindings_path.read_text(encoding="utf8")) if bindings_path.is_file() else {}
        validation=model.get("validation",{})
        return {"entities":dict(Counter(e["type"] for e in model["entities"])),
                "entity_count":len(model["entities"]),"geometry_valid":validation.get("passed"),
                "dxf_readback":validation.get("dxf_readback",{}).get("passed"),
                "constraint_count":len(solution.get("constraints",[])),
                "constraint_kinds":dict(Counter(c["kind"] for c in solution.get("constraints",[]))),
                "solver_accepted":solution.get("accepted"),"remaining_shape_dof":solution.get("diagnostics",{}).get("remaining_shape_dof"),
                "binding_counts":bindings.get("counts",{}),"dimensions_verified":validation.get("dimensions_verified"),
                "reference_verified":False,"curve_fit":model.get("curve_fit",{}),
                "completion_class":model.get("completion_class"),"primitive_diagnostics":validation.get("primitive_diagnostics")}
    report={"before":summary(frozen),"after":summary(after),"stage_status":stage.get("status"),
            "accepted":stage.get("accepted"),"reason":stage.get("reason"),"ground_truth_used":False,
            "manual_mask_review":True,"held_out":False,
            "scope":"Same frozen source and reviewed mask. Mask fit is not GT accuracy; rejected solver candidates are not published improvements."}
    write(output/"comparison.json",report)
    print(json.dumps({"output":str(output),"status":stage.get("status"),"accepted":stage.get("accepted")},ensure_ascii=False),flush=True)


if __name__=="__main__":main()

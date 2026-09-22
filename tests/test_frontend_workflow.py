"""Verify that the workbench shows actual stage artifacts, including partial failures."""
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_stage_artifacts_do_not_invent_success_or_use_training_images():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for frontend workflow regression")
    script = r"""
const assert=require('node:assert/strict'),w=require(process.argv[1]);
const segmentation={status:'failed',use_segmentation:true,artifacts:{segmentation_overlay:'/api/jobs/a/artifacts/segmentation-overlay.png',segmentation_mask:'/api/jobs/a/artifacts/segmentation-mask.png'}};
assert.equal(w.artifactForView(segmentation,'segmentation'),segmentation.artifacts.segmentation_overlay);
assert.equal(w.artifactForView(segmentation,'segmentation',true),segmentation.artifacts.segmentation_mask);
assert.deepEqual(w.stageEvidence(segmentation).map(s=>s.status),['ready','stopped','stopped']);
assert.equal(w.artifactForView(segmentation,'overlay'),null,'a mask is never substituted for CAD');
const cad={...segmentation,status:'completed',artifacts:{...segmentation.artifacts,contour_overlay:'/api/jobs/a/artifacts/contour-overlay.png',cad_overlay:'/api/jobs/a/artifacts/cad-overlay.png',overlay:'/api/jobs/a/artifacts/overlay.png',dxf:'/api/jobs/a/artifacts/drawing.dxf'},validation:{passed:false,scaled_mm:false}};
assert.equal(w.artifactForView(cad,'overlay'),cad.artifacts.cad_overlay);
assert.equal(w.artifactForView(cad,'contour'),cad.artifacts.contour_overlay);
assert.deepEqual(w.stageEvidence(cad).map(s=>s.status),['ready','ready','warning']);
assert.equal(w.artifactForView({artifacts:{overlay:'/api/jobs/a/artifacts/overlay.png'}},'overlay'),'/api/jobs/a/artifacts/overlay.png','legacy overlay remains readable');
assert.equal(w.artifactForView({reference:{overlay:'GT.png'},segmentation:{mask:'training.png'}},'segmentation'),null);
assert.deepEqual(w.stageEvidence({status:'completed',provider:{verdict:'match'},artifacts:{},use_segmentation:true}).map(s=>s.status),['pending','pending','pending'],'provider verdict cannot establish any local stage artifact');
assert.equal(w.stageEvidence({use_segmentation:false,status:'extracting',artifacts:{}})[0].status,'skipped');
assert.equal(w.pilotIds.length,4);assert.equal(new Set(w.pilotIds).size,4);
assert.equal(w.dimensionProviderText({status:'disabled',network_requests:0,http_success:false,schema_success:false}),'未启用 / 未调用','offline mode is not a failed API');
assert.equal(w.dimensionProviderText({status:'skipped',network_requests:0}),'未调用 / 未调用');
assert.equal(w.dimensionProviderText({status:'pending',network_requests:0}),'调用中 / 等待返回');
assert.equal(w.dimensionProviderText({status:'failed',network_requests:0,http_success:false}),'未发起 / 未调用');
assert.equal(w.dimensionProviderText({status:'interrupted',network_requests:null}),'调用中断 / 结果未知');
assert.equal(w.dimensionProviderText({status:'failed',network_requests:1,http_success:true,schema_success:false}),'成功 / 未取得','HTTP 200 invalid JSON remains distinct from transport failure');
assert.equal(w.dimensionProviderText({status:'failed',network_requests:1,http_success:false,schema_success:false}),'未成功 / 未取得');
assert.equal(w.dimensionProviderText({status:'succeeded',network_requests:1,http_success:true,schema_success:true}),'成功 / 有效');
"""
    result = subprocess.run([node, "-e", script, str(ROOT / "web/workflow.js")], capture_output=True, text=True, encoding="utf8", timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pilot_submission_is_serial_guarded_and_explicitly_requests_model():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for frontend workflow regression")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[1],'utf8');
const start=source.indexOf('async function runPilots()'),end=source.indexOf('\nfunction pollPilots()',start);
const requests=[],elements={'run-pilots':{disabled:false},'use-api':{checked:false}};
const state={pilotJobs:new Map(),pilotSubmitting:false,pilotPoll:null,epoch:0};
let release;
const first=new Promise(resolve=>{release=resolve;});
const sandbox={state,workflow:{pilotIds:['CL60-main','182','202','231']},$:(id)=>elements[id],clearTimeout(){},renderPilots(){},renderHeading(){},adoptJob(){},pollPilots(){},toast(){},isBusy:()=>false,shortCase:id=>id,
post:async(url,payload)=>{requests.push({url,payload});if(requests.length===1)await first;return {id:'job'+requests.length,case_id:payload.case_id,status:'queued'};}};
vm.createContext(sandbox);vm.runInContext(source.slice(start,end),sandbox);
(async()=>{const pending=sandbox.runPilots();await sandbox.runPilots();assert.equal(requests.length,1,'double click cannot duplicate first request');release();await pending;assert.equal(requests.length,4);assert.deepEqual(requests.map(r=>r.payload.case_id),['CL60-main','182','202','231']);assert.ok(requests.every(r=>r.url==='/api/jobs'&&r.payload.mode==='autonomous_image'&&r.payload.use_segmentation===true&&r.payload.use_api===false));assert.equal(state.pilotSubmitting,false);assert.equal(state.pilotJobs.size,4);})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run([node, "-e", script, str(ROOT / "web/app.js")], capture_output=True, text=True, encoding="utf8", timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr

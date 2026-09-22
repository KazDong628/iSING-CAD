"""Exercise the shipped review script with a tiny canvas/DOM test double."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_undo_baseline_and_visible_discard_keep_workflow():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the review-script regression")
    html = Path(__file__).resolve().parents[1] / "web/segmentation-review.html"
    harness = r"""
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const elements=new Map(),unload={},requests=[];
function context(){const value={data:new Uint8ClampedArray(8),getImageData(){return {data:this.data.slice()};},putImageData(image){this.data=image.data.slice();}};return new Proxy(value,{get:(target,key)=>key in target?target[key]:()=>{}});}
function element(){const ctx=context();return {width:2,height:1,hidden:true,value:'',style:{},listeners:{},options:[],classList:{toggle(){}},setAttribute(){},focus(){},getContext(){return ctx;},addEventListener(name,handler){this.listeners[name]=handler;},getBoundingClientRect(){return {left:0,top:0,width:2,height:1};},append(){},replaceChildren(){}};}
const sandbox={assert,Uint8Array,Uint8ClampedArray,Image:class {},ResizeObserver:class{observe(){}},
 document:{getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},createElement:element,addEventListener(){}},
 window:{addEventListener(name,handler){unload[name]=handler;},confirm(){throw Error('Native confirm must not be used');}},
 fetch:(url,options)=>{requests.push({url,options});return new Promise(()=>{});}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0],sandbox);
sandbox.unload=unload;sandbox.requests=requests;
vm.runInContext(`
state.loaded=true;state.current={id:'a',reviewed:false};state.cases=[state.current,{id:'b',reviewed:false}];mask.width=2;mask.height=1;
function pixels(a,b){return {data:new Uint8ClampedArray([255,255,255,a,255,255,255,b])};}
mctx.putImageData(pixels(0,0));rememberBaseline();assert.equal(state.dirty,false);
state.undo.push(mctx.getImageData());mctx.putImageData(pixels(255,0));updateDirty();assert.equal(state.dirty,true);
state.undo.push(mctx.getImageData());mctx.putImageData(pixels(255,255));updateDirty();
$('undo').onclick();assert.equal(state.dirty,true,'one changed pixel remains after partial undo');
$('undo').onclick();assert.equal(state.dirty,false,'undo to initial mask clears unsaved state');
let protectedUnload=false;unload.beforeunload({preventDefault(){protectedUnload=true;}});assert.equal(protectedUnload,false);
let opened=null;openCase=id=>{opened=id;};
$('case-select').value='b';$('case-select').onchange({target:$('case-select')});assert.equal(opened,'b');
mctx.putImageData(pixels(0,255));updateDirty();assert.equal(state.dirty,true);
unload.beforeunload({preventDefault(){protectedUnload=true;}});assert.equal(protectedUnload,true,'real edits retain unload protection');
opened=null;$('case-select').value='b';$('case-select').onchange({target:$('case-select')});
assert.equal(opened,null);assert.equal($('case-select').value,'a');assert.equal($('switch-notice').hidden,false);assert.equal(state.pendingCase,'b');
$('keep-editing').onclick();assert.equal($('switch-notice').hidden,true);assert.equal(state.pendingCase,null);assert.equal(state.dirty,true);assert.equal(binaryMask()[1],1);
$('case-select').value='b';$('case-select').onchange({target:$('case-select')});$('discard-switch').onclick();
assert.equal(opened,'b');assert.equal($('switch-notice').hidden,true);assert.equal($('case-select').value,'b');
rememberBaseline();assert.equal(state.dirty,false,'successful save can establish a new baseline');
mctx.putImageData(pixels(255,255));updateDirty();assert.equal(state.dirty,true);mctx.putImageData(pixels(0,255));updateDirty();assert.equal(state.dirty,false);
assert.equal(state.current.reviewed,false,'navigation and undo never mark labels reviewed');
assert.equal(requests.some(request=>request.options?.method==='PUT'),false);
`,sandbox);
"""
    result = subprocess.run([node, "-e", harness, str(html)], capture_output=True, text=True, encoding="utf8", timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr

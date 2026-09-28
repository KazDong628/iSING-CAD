"use strict";

const $ = id => document.getElementById(id);
const state = {
  conversations: [], conversation: null, currentJob: null, view: "overlay", panel: "preview",
  sourceImage: null, ocr: null, feedback: null, selectedCase: "", polling: null,
  detailCache: new Map(), transcriptCache: new Map(), submitting: false,
  stopping: false, deletingConversationId: null,
  providers: [], selectedProvider: "",
  welcome: null,
  zoom: {preview:{scale:1,offsetX:0,offsetY:0,key:""},review:{scale:1,offsetX:0,offsetY:0,key:""}},
  review: {jobId:null,loading:false,loaded:false,tool:"add",brush:24,history:[],baseline:null,source:null,mask:null,maskSource:"human_review"},
};
const busy = new Set(["queued", "extracting", "solving", "auditing"]);
const clamp = (value,minimum,maximum) => Math.min(maximum,Math.max(minimum,value));

function syncRunButton(){
  const button=$("send"),running=state.submitting||busy.has(state.currentJob?.status);
  button.disabled=running;button.classList.toggle("running",running);button.setAttribute("aria-busy",String(running));
  button.setAttribute("aria-label",running?"任务运行中":"运行任务");
  const stop=$("stop-job"),canStop=busy.has(state.currentJob?.status);
  stop.hidden=!canStop;stop.disabled=!canStop||state.stopping;
  stop.setAttribute("aria-busy",String(state.stopping));
}
function promptSuggestion(){
  if(state.feedback||state.currentJob)return "请根据局部截图和我的描述修正主轮廓，重新执行几何校验并生成 DXF。";
  if(state.selectedCase)return `请对 ${state.selectedCase} 自动提取主轮廓并生成 DXF。`;
  if(state.sourceImage||state.ocr)return "请从上传的工程图中提取主轮廓，完成拓扑编辑、参数绑定并生成 DXF。";
  return "请自动提取工程图的主轮廓，完成拓扑编辑、参数绑定并生成 DXF。";
}

function zoomElements(kind){
  return kind==="review"
    ? {viewport:document.querySelector(".review-canvas-wrap"),media:$("review-canvas"),output:$("review-zoom")}
    : {viewport:document.querySelector(".preview-stage"),media:$("preview-image"),output:$("preview-zoom")};
}
function intrinsicSize(kind,media){
  return kind==="review"?{width:media.width,height:media.height}:{width:media.naturalWidth,height:media.naturalHeight};
}
function zoomGeometry(kind){
  const {viewport,media}=zoomElements(kind),size=intrinsicSize(kind,media),width=viewport?.clientWidth||0,height=viewport?.clientHeight||0;
  if(!viewport||!media||!size.width||!size.height||!width||!height)return null;
  const fit=Math.min(width/size.width,height/size.height),baseWidth=size.width*fit,baseHeight=size.height*fit;
  return {viewport,media,width,height,baseWidth,baseHeight,baseLeft:(width-baseWidth)/2,baseTop:(height-baseHeight)/2};
}
function constrainZoom(kind,geometry){
  const zoom=state.zoom[kind],scaledWidth=geometry.baseWidth*zoom.scale,scaledHeight=geometry.baseHeight*zoom.scale;
  if(scaledWidth<=geometry.width)zoom.offsetX=0;
  else zoom.offsetX=clamp(zoom.offsetX,geometry.width-scaledWidth-geometry.baseLeft,-geometry.baseLeft);
  if(scaledHeight<=geometry.height)zoom.offsetY=0;
  else zoom.offsetY=clamp(zoom.offsetY,geometry.height-scaledHeight-geometry.baseTop,-geometry.baseTop);
}
function applyZoom(kind){
  const geometry=zoomGeometry(kind),zoom=state.zoom[kind],elements=zoomElements(kind);
  if(!geometry)return;
  constrainZoom(kind,geometry);
  const {media,baseWidth,baseHeight,baseLeft,baseTop}=geometry;
  media.style.width=`${baseWidth}px`;media.style.height=`${baseHeight}px`;
  media.style.left=`${baseLeft+zoom.offsetX}px`;media.style.top=`${baseTop+zoom.offsetY}px`;
  media.style.transform=`scale(${zoom.scale})`;
  geometry.viewport.classList.toggle("zoomed",zoom.scale>1.001);
  if(elements.output)elements.output.textContent=`${Math.round(zoom.scale*100)}%`;
}
function resetZoom(kind,key){
  state.zoom[kind]={scale:1,offsetX:0,offsetY:0,key:key??state.zoom[kind].key};applyZoom(kind);
}
function zoomAtPointer(kind,event){
  if(!event.ctrlKey)return;
  const geometry=zoomGeometry(kind);if(!geometry)return;
  event.preventDefault();
  const zoom=state.zoom[kind],rect=geometry.viewport.getBoundingClientRect();
  const mouseX=event.clientX-rect.left,mouseY=event.clientY-rect.top;
  const imageX=(mouseX-geometry.baseLeft-zoom.offsetX)/zoom.scale;
  const imageY=(mouseY-geometry.baseTop-zoom.offsetY)/zoom.scale;
  const next=clamp(zoom.scale*(event.deltaY<0?1.2:1/1.2),1,8);
  zoom.scale=next;zoom.offsetX=mouseX-geometry.baseLeft-imageX*next;zoom.offsetY=mouseY-geometry.baseTop-imageY*next;
  applyZoom(kind);
}

function safeUrl(value) {
  if (typeof value !== "string" || !value.startsWith("/api/") || value.startsWith("//")) return null;
  const url = new URL(value, location.origin);
  return url.origin === location.origin ? url.pathname + url.search : null;
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `请求失败（HTTP ${response.status}）`);
  return body;
}
function fmtTime(value) {
  const date = new Date(value); return Number.isFinite(date.getTime()) ? date.toLocaleTimeString("zh-CN", {hour12:false,hour:"2-digit",minute:"2-digit"}) : "—";
}
function toast(text, error=false) {
  const node=$("toast"); node.textContent=text; node.classList.toggle("error",error); node.hidden=false;
  clearTimeout(node.timer); node.timer=setTimeout(()=>node.hidden=true,5000);
}
function node(tag, className, text) {
  const el=document.createElement(tag); if(className)el.className=className; if(text!==undefined)el.textContent=String(text); return el;
}

async function loadBootstrap() {
  try {
    state.welcome=$("welcome");
    const config=await api("/api/config");
    $("connection-label").textContent="本地执行器在线";
    state.providers=config.providers||[];
    const providerSelect=$("provider-select");providerSelect.replaceChildren();
    for(const profile of state.providers){const option=document.createElement("option");option.value=profile.id;option.disabled=!profile.configured;option.textContent=`${profile.name}${profile.configured?"":" · 未配置"}`;providerSelect.append(option);}
    const saved=localStorage.getItem("contour-provider-id"),fallback=config.default_provider_id||state.providers.find(row=>row.configured)?.id||"";
    state.selectedProvider=state.providers.some(row=>row.id===saved&&row.configured)?saved:fallback;providerSelect.value=state.selectedProvider;
    updateProviderSelection();
    $("use-segmentation").disabled=!config.segmentation_available;
    const [catalog,rows]=await Promise.all([api("/api/catalog"),api("/api/conversations")]);
    const select=$("case-select");
    for(const item of catalog.cases.filter(row=>row.image&&row.ocr)){
      const option=document.createElement("option");option.value=item.id;option.textContent=item.id;select.append(option);
    }
    state.conversations=rows.conversations||[];
    if(!state.conversations.length){ await createConversation(); }
    else { renderConversationList(); await openConversation(state.conversations[0].id); }
  } catch(error) {
    $("connection-label").textContent="本地执行器不可用";
    if(!state.providers.length){
      $("provider-select").replaceChildren(new Option("模型加载失败，请刷新页面",""));
      $("use-api").checked=false;$("use-api").disabled=true;
    }
    toast(error.message,true);
  }
}
function updateProviderSelection(){
  const profile=state.providers.find(row=>row.id===$("provider-select").value)||state.providers.find(row=>row.id===state.selectedProvider);
  state.selectedProvider=profile?.id||"";if(state.selectedProvider)localStorage.setItem("contour-provider-id",state.selectedProvider);
  $("provider-label").textContent=profile?.configured?`${profile.model_provider} / ${profile.model}`:"本地模式";
  $("use-api").disabled=!profile?.configured;if(!profile?.configured)$("use-api").checked=false;
}

async function createConversation() {
  const value=await api("/api/conversations",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:"新建主轮廓任务"})});
  state.conversations.unshift(value); renderConversationList(); await openConversation(value.id); $("prompt").focus();
}
async function openConversation(id) {
  state.conversation=await api(`/api/conversations/${encodeURIComponent(id)}`);
  const last=state.conversation.memory?.last_job_id;
  state.currentJob=last ? state.conversation.jobs?.[last]||null : null;
  state.detailCache.clear(); renderAll(); startPolling();
}
function renderConversationList(){
  const box=$("conversation-list");box.replaceChildren();
  for(const row of state.conversations){
    const card=node("article",`conversation-card${row.id===state.conversation?.id?" active":""}`);
    const open=node("button","conversation-open");open.type="button";open.setAttribute("aria-label",`打开任务 ${row.title}`);
    open.append(node("span","icon","▧"));const copy=node("span");copy.append(node("b","",row.title));
    copy.append(node("small","",`${row.memory?.turn_count||0} 条消息 · ${fmtTime(row.updated_at)}`));open.append(copy);
    open.onclick=()=>openConversation(row.id).catch(error=>toast(error.message,true));
    const remove=node("button","conversation-delete","×");remove.type="button";remove.title="删除任务历史";remove.setAttribute("aria-label",`删除任务 ${row.title}`);
    remove.onclick=()=>openDeleteDialog(row);card.append(open,remove);box.append(card);
  }
}
function openDeleteDialog(row){
  state.deletingConversationId=row.id;$("delete-dialog-name").textContent=row.title;
  $("delete-confirm").disabled=false;$("delete-confirm").textContent="确认删除";$("delete-dialog").showModal();
}
async function deleteConversation(){
  const id=state.deletingConversationId;if(!id)return;
  const confirm=$("delete-confirm");confirm.disabled=true;confirm.textContent="正在删除…";
  try{
    await api(`/api/conversations/${encodeURIComponent(id)}`,{method:"DELETE"});
    const wasCurrent=state.conversation?.id===id;state.conversations=state.conversations.filter(row=>row.id!==id);
    $("delete-dialog").close();state.deletingConversationId=null;toast("任务历史及专属产物已删除。");
    if(wasCurrent){clearTimeout(state.polling);state.conversation=null;state.currentJob=null;state.detailCache.clear();state.transcriptCache.clear();if(state.conversations.length)await openConversation(state.conversations[0].id);else await createConversation();}
    else renderConversationList();
  }catch(error){confirm.disabled=false;confirm.textContent="确认删除";toast(error.message,true);}
}
function syncConversationSummary(){
  const conversation=state.conversation;if(!conversation)return;
  const summary={id:conversation.id,title:conversation.title,created_at:conversation.created_at,updated_at:conversation.updated_at,memory:conversation.memory};
  const index=state.conversations.findIndex(row=>row.id===conversation.id);
  if(index>=0)state.conversations[index]=summary;else state.conversations.unshift(summary);
  state.conversations.sort((left,right)=>String(right.updated_at||"").localeCompare(String(left.updated_at||"")));
}
function renderAll(){
  const conversation=state.conversation;if(!conversation)return;
  syncConversationSummary();renderConversationList();renderMessages();renderWorkbench();renderMemory();renderAttachments();
  $("conversation-title").textContent=conversation.title;
  $("turn-count").textContent=conversation.memory?.turn_count||0;
  $("job-count").textContent=conversation.memory?.job_ids?.length||0;
}
function renderMessages(){
  const box=$("messages"),conversation=state.conversation;box.replaceChildren();
  if(!conversation?.messages?.length){box.append(state.welcome);state.welcome.hidden=false;return;}
  for(const message of conversation.messages){
    const card=node("article",`message ${message.role}`),meta=node("div","message-meta");
    meta.append(node("span","",message.role==="user"?"YOU":"CONTOUR AGENT"),node("time","",fmtTime(message.time)));card.append(meta,node("div","message-body",message.text));
    if(message.attachments?.length){const files=node("div","message-attachments");for(const file of message.attachments)files.append(node("span","",`${file.kind==="ocr"?"JSON":"IMG"} · ${file.name}`));card.append(files);}
    if(message.job_id){
      const job=conversation.jobs?.[message.job_id],callout=node("div","job-callout");
      callout.append(node("span","",`${message.trace_kind==="revision"?"修订":"重建"}任务 ${message.job_id.slice(0,8)} · ${job?.status||"已创建"}`));
      const open=node("button","","查看结果 →");open.type="button";open.onclick=()=>{state.currentJob=conversation.jobs?.[message.job_id]||job;renderWorkbench();};callout.append(open);card.append(callout);
      if(job){const transcript=node("section","model-transcript loading","正在读取模型回执…");card.append(transcript);loadModelTranscript(job,transcript);}
    }
    box.append(card);
  }
  requestAnimationFrame(()=>box.scrollTop=box.scrollHeight);
}

function answerSummary(stage){
  const answer=stage.answer||{};
  if(stage.transport?.schema_success!==true){
    const labels={
      truncated_output:"模型输出达到长度上限，未使用不完整结果",
      timeout:"模型请求超时，已保留本地结果",
      transport_error:"无法连接模型服务，已保留本地结果",
      authentication:"模型服务鉴权失败",
      permission:"模型服务拒绝访问",
      rate_limit:"模型服务限流",
      invalid_envelope:"服务返回格式与所选协议不兼容",
      invalid_json:"模型未返回完整 JSON",
      schema_mismatch:"模型 JSON 未通过字段校验",
      schema_validation_failed:"模型响应未通过结构校验",
      transport_failure_without_code:"传输失败，但旧回执没有记录具体错误码",
      stage_receipt_incomplete:"旧任务的阶段回执不完整",
    };
    return labels[answer.error_code]||`响应未通过结构校验：${answer.error_code||"未知错误"}`;
  }
  if(stage.id==="planning")return answer.candidate_id?`选择候选 ${answer.candidate_id} · ${answer.confidence||"未标注置信度"}`:"模型选择弃权";
  if(stage.id==="topology_edit")return Array.isArray(answer.operations)&&answer.operations.length?`${answer.observation||"提出局部拓扑编辑"} · ${answer.operations.length} 项操作`:answer.observation||"编辑 Agent 选择弃权";
  if(stage.id==="topology_evaluate")return `${answer.decision||"未决策"}${answer.candidate_id?` · ${answer.candidate_id}`:""}${answer.observation?` · ${answer.observation}`:""}`;
  if(stage.id==="dimensions")return `返回 ${Array.isArray(answer.dimensions)?answer.dimensions.length:0} 条尺寸解释`;
  if(stage.id==="binding")return `返回 ${Array.isArray(answer.bindings)?answer.bindings.length:0} 条绑定、${Array.isArray(answer.relations)?answer.relations.length:0} 条关系`;
  if(stage.id==="vision")return `判定 ${answer.verdict||"未知"}${Array.isArray(answer.issues)&&answer.issues.length?` · ${answer.issues.join("；")}`:""}`;
  if(stage.id==="feedback")return answer.observation||answer.proposed_action||"已返回截图修订选择";
  return answer.error_code?`响应未通过结构校验：${answer.error_code}`:"已返回结构化结果";
}
function renderModelTranscript(host,value){
  host.replaceChildren();host.classList.remove("loading");const stages=value?.stages||[];
  if(!stages.length&&!value?.iterations&&!value?.parameterization){host.remove();return;}
  const heading=node("div","transcript-heading");heading.append(node("span","","MODEL RECEIPTS"),node("b","",`${stages.length} 个在线阶段`));host.append(heading);
  const table=node("div","receipt-table");for(const text of ["阶段","传输","结构","耗时"])table.append(node("b","",text));
  for(const stage of stages){const transport=stage.transport||{};table.append(node("span","",stage.label),node("code","",transport.http_status?`HTTP ${transport.http_status}`:transport.http_success===false?"HTTP 失败":"未记录"),node("code","",transport.schema_success?"结构有效":"结构无效"),node("code","",Number.isFinite(transport.elapsed_seconds)?`${transport.elapsed_seconds.toFixed(1)}s`:"—"));}host.append(table);
  const cards=node("div","receipt-cards");
  for(const stage of stages){
    const card=node("article",`receipt-card ${stage.transport?.schema_success?"valid":"invalid"}`),head=node("div","receipt-head");
    head.append(node("b","",stage.label),node("span","",stage.transport?.model||stage.transport?.protocol||"ONLINE MODEL"));card.append(head,node("p","receipt-summary",answerSummary(stage)));
    const details=node("details","receipt-answer");details.open=true;details.append(node("summary","",stage.answer_source==="validated_structured_fields"?"模型回答 · 已校验字段":"模型回答 · 脱敏失败回执"),node("pre","",JSON.stringify(stage.answer,null,2)));card.append(details);cards.append(card);
  }
  host.append(cards);
  if(value.iterations)host.append(renderIterationAudit(value.iterations));
  if(value.parameterization){
    const audit=value.parameterization,counts=audit.counts||{},diagnostics=audit.diagnostics||{},box=node("div","parameter-audit");
    box.append(node("b","","参数化求解 · 独立检查"),node("p","",`${audit.accepted?"已接受约束子集的解":"尚未接受参数解"} · ${audit.constraints?.length||0} 项已求解约束 · ${counts.bound_source_records??"—"} 条已绑定标注 / ${counts.recognized_dimensions??"—"} 条识别标注`),
      node("p","",`未绑定标注 ${counts.unbound_dimensions??"—"} 条 · 剩余形状自由度 ${diagnostics.remaining_shape_dof??"—"}。几何可导出不代表尺寸完整或已通过 GT 验证。`));
    const issues=audit.feedback?.issues||[];
    if(issues.length){const list=node("ul","audit-issues");for(const issue of issues.slice(0,12))list.append(node("li","",issueText(issue)));box.append(list);}
    const details=node("details","receipt-answer");details.append(node("summary","","查看约束残差与未解决项"),node("pre","",JSON.stringify(audit,null,2)));box.append(details);host.append(box);
  }
  host.append(node("p","transcript-notice",value.notice||"仅显示可审计的模型输出。"));
}
const editActionLabels={merge_chain_as_line:"合并为直线",merge_chain_as_arc:"合并为圆弧",merge_chain_best_fit:"合并冗余图元",refit_chain_as_annotated_arc:"按标注拟合圆弧",refit_entity_as_line:"将单个图元修正为直线",split_chain_at_source_features:"按原图特征拆分",insert_annotated_fillet:"插入标注圆角",apply_nonoverlapping_edits:"组合独立编辑"};
const auditReasonLabels={radius_target_is_line:"半径标注指向了直线",radius_value_unresolved:"标注半径尚未满足",constraint_residual_failed:"约束残差超限",conflicting_constraints:"尺寸约束冲突",conflicting_relations:"几何关系冲突",no_accepted_improvement:"没有通过检查的进一步改善，保留最后有效结果",round_budget_exhausted:"已达到本轮迭代预算",repeated_geometry:"检测到重复几何，已停止循环",evaluator_preserved_base:"评估保留原候选",evaluator_did_not_select_an_edit:"没有选中可采用的编辑",edited_candidate_failed_local_improvement_gate:"编辑未通过独立接受检查",unresolved_fixed_radius_fit_failed:"固定半径未能满足，仍待解决"};
function issueText(issue){return `${issue.record_id||issue.entity_id||issue.entity_ids?.join(" → ")||"当前轮廓"} · ${auditReasonLabels[issue.code]||issue.code||"待检查"}${issue.record_id&&issue.entity_id?` (${issue.entity_id})`:""}`;}
function renderIterationAudit(iterations){
  const box=node("section","iteration-audit"),rounds=iterations.rounds||[];
  box.append(node("b","audit-title",`局部拓扑迭代 · ${rounds.length}/${iterations.max_rounds||3} 轮`));
  for(const round of rounds){
    const accepted=round.acceptance_gate?.accepted===true,details=node("details",`iteration-round ${accepted?"accepted":"preserved"}`);details.open=true;
    details.append(node("summary","",`第 ${round.round} 轮 · ${accepted?"采用编辑":"保留原候选"} · ${round.base_candidate_id||"—"} → ${round.final_candidate_id||"—"}`));
    const operations=round.operations||[];
    if(operations.length){const list=node("ul","audit-operations");for(const operation of operations){
      const item=node("li"),title=`${editActionLabels[operation.action]||operation.action||"局部编辑"} · ${operation.entity_ids?.join(" → ")||"独立操作组合"}${operation.record_id?` · ${operation.record_id}`:""}`;
      item.append(node("span","",title));
      const candidatePassed=operation.status==="accepted_as_candidate";
      item.append(node("small",candidatePassed?"candidate-pass":"candidate-reject",candidatePassed?`候选通过本地构建：${operation.candidate_id||"—"}`:`未构建候选：${auditReasonLabels[operation.reason]||operation.reason||operation.status||"等待执行"}`));
      if(operation.radius_binding_applied===true)item.append(node("small","candidate-pass","标注半径已应用；全局约束仍需联合检查"));
      else if(operation.radius_binding_status==="unresolved_fixed_radius_fit_failed")item.append(node("small","candidate-reject","固定半径未满足：当前只是自由圆弧拟合"));
      list.append(item);
    }details.append(list);}else details.append(node("p","receipt-summary","本轮没有可执行的局部操作。"));
    const issues=round.feedback?.issues||[];
    if(issues.length)details.append(node("p","receipt-summary",`本轮输入待解决项：${issues.slice(0,8).map(issueText).join("；")}${issues.length>8?`；另 ${issues.length-8} 项`:""}`));
    if(round.acceptance_gate?.reason)details.append(node("p","receipt-summary",auditReasonLabels[round.acceptance_gate.reason]||round.acceptance_gate.reason));
    box.append(details);
  }
  box.append(node("p","iteration-stop",auditReasonLabels[iterations.stop_reason]||iterations.stop_reason||"正在执行有界迭代…"));
  return box;
}
async function loadModelTranscript(job,host){
  const key=`${job.id}:${job.updated_at||""}`;
  if(state.transcriptCache.has(key)){renderModelTranscript(host,state.transcriptCache.get(key));return;}
  try{const value=await api(`/api/jobs/${encodeURIComponent(job.id)}/model-transcript`);state.transcriptCache.set(key,value);if(host.isConnected)renderModelTranscript(host,value);}
  catch{if(host.isConnected)host.remove();}
}

function artifactFor(job,view){
  if(!job)return null;const a=job.artifacts||{};
  return {overlay:a.cad_overlay||a.overlay,source:job.source_image_url,segmentation:a.reviewed_segmentation_overlay||a.segmentation_overlay,topology:a.topology_overlay}[view]||null;
}

function loadReviewImage(url){
  return new Promise((resolve,reject)=>{
    const image=new Image();
    image.onload=()=>resolve(image);
    image.onerror=()=>reject(new Error("无法载入分割审核图像。"));
    image.src=url;
  });
}
function setReviewControls(disabled){
  for(const id of ["review-add","review-erase","review-dxf-file","review-brush","review-undo","review-reset","review-confirm"]){$(id).disabled=disabled;}
  if(!disabled)$("review-undo").disabled=!state.review.history.length;
}
function drawSegmentationReview(){
  const review=state.review,canvas=$("review-canvas");
  if(!review.loaded||!review.source||!review.mask)return;
  const context=canvas.getContext("2d");
  context.clearRect(0,0,canvas.width,canvas.height);
  context.fillStyle="#f4f5f1";context.fillRect(0,0,canvas.width,canvas.height);
  context.drawImage(review.source,0,0,canvas.width,canvas.height);
  context.save();context.globalAlpha=.48;context.drawImage(review.mask,0,0);context.restore();
}
async function initSegmentationEditor(job){
  const review=state.review;
  if(review.jobId===job.id&&(review.loaded||review.loading))return;
  const sourceUrl=safeUrl(job.source_image_url),maskUrl=safeUrl(job.artifacts?.segmentation_mask);
  const oracleUrl=job.oracle_mask_import?.mask_sha256?safeUrl(job.artifacts?.oracle_generated_mask):null;
  if(!sourceUrl||!maskUrl)throw new Error("任务缺少可编辑的分割图或原图。");
  review.jobId=job.id;review.loading=true;review.loaded=false;review.history=[];review.baseline=null;review.source=null;review.mask=null;review.maskSource="human_review";
  state.zoom.review={scale:1,offsetX:0,offsetY:0,key:job.id};
  $("review-loading").hidden=false;$("review-loading").textContent="正在载入分割图…";$("review-status").textContent="正在准备像素级编辑器…";setReviewControls(true);
  try{
    const version=encodeURIComponent(job.updated_at||Date.now()),[source,maskImage,oracleImage]=await Promise.all([
      loadReviewImage(`${sourceUrl}${sourceUrl.includes("?")?"&":"?"}v=${version}`),
      loadReviewImage(`${maskUrl}${maskUrl.includes("?")?"&":"?"}v=${version}`),
      oracleUrl?loadReviewImage(`${oracleUrl}${oracleUrl.includes("?")?"&":"?"}v=${version}`):Promise.resolve(null),
    ]);
    if(state.review.jobId!==job.id)return;
    const mask=document.createElement("canvas");mask.width=maskImage.naturalWidth;mask.height=maskImage.naturalHeight;
    const context=mask.getContext("2d",{willReadFrequently:true});context.drawImage(maskImage,0,0);
    const pixels=context.getImageData(0,0,mask.width,mask.height);
    for(let index=0;index<pixels.data.length;index+=4){const foreground=pixels.data[index]>=128||pixels.data[index+1]>=128||pixels.data[index+2]>=128;pixels.data[index]=27;pixels.data[index+1]=190;pixels.data[index+2]=142;pixels.data[index+3]=foreground?255:0;}
    context.putImageData(pixels,0,0);
    const canvas=$("review-canvas");canvas.width=mask.width;canvas.height=mask.height;
    review.source=source;review.mask=mask;review.baseline=context.getImageData(0,0,mask.width,mask.height);review.loaded=true;
    if(oracleImage){
      if(oracleImage.naturalWidth!==mask.width||oracleImage.naturalHeight!==mask.height)throw new Error("DXF派生掩膜尺寸与模型分割图不一致。");
      context.clearRect(0,0,mask.width,mask.height);context.drawImage(oracleImage,0,0);
      const oraclePixels=context.getImageData(0,0,mask.width,mask.height);
      for(let index=0;index<oraclePixels.data.length;index+=4){
        const foreground=oraclePixels.data[index]>=128;
        oraclePixels.data[index]=27;oraclePixels.data[index+1]=190;oraclePixels.data[index+2]=142;oraclePixels.data[index+3]=foreground?255:0;
      }
      context.putImageData(oraclePixels,0,0);review.maskSource="gt_oracle";
    }
    $("review-loading").hidden=true;$("review-status").textContent=oracleImage?
      "已从GT DXF生成掩膜；请检查绿色区域与原图是否对齐，确认后继续开发实验。":
      "绿色为材料。Ctrl + 滚轮可在指针位置缩放；擦除标注线、剖面线和多余块，补全缺失区域。";
    setReviewControls(false);drawSegmentationReview();resetZoom("review",job.id);
  }finally{review.loading=false;}
}
function setReviewTool(tool){
  state.review.tool=tool;$("review-add").classList.toggle("active",tool==="add");$("review-erase").classList.toggle("active",tool==="erase");
}
function reviewPoint(event){
  const canvas=$("review-canvas"),rect=canvas.getBoundingClientRect();
  return {x:(event.clientX-rect.left)*canvas.width/rect.width,y:(event.clientY-rect.top)*canvas.height/rect.height};
}
function paintReview(from,to){
  const review=state.review;if(!review.loaded||!review.mask)return;
  if(review.maskSource==="gt_oracle"){
    review.maskSource="gt_oracle_edited";
    $("review-status").textContent="GT派生掩膜已人工涂改；本次仍记录GT来源，并标记为人工修订。";
  }
  const context=review.mask.getContext("2d",{willReadFrequently:true});
  context.save();context.globalCompositeOperation=review.tool==="erase"?"destination-out":"source-over";context.strokeStyle="#1bbe8e";context.fillStyle="#1bbe8e";
  context.lineWidth=review.brush;context.lineCap="round";context.lineJoin="round";context.beginPath();context.moveTo(from.x,from.y);context.lineTo(to.x,to.y);context.stroke();
  if(from.x===to.x&&from.y===to.y){context.beginPath();context.arc(to.x,to.y,review.brush/2,0,Math.PI*2);context.fill();}
  context.restore();drawSegmentationReview();
}
function pushReviewHistory(){
  const review=state.review;if(!review.loaded||!review.mask)return;
  review.history.push(review.mask.getContext("2d",{willReadFrequently:true}).getImageData(0,0,review.mask.width,review.mask.height));
  if(review.history.length>10)review.history.shift();$("review-undo").disabled=false;
}
function undoReview(){
  const review=state.review,item=review.history.pop();if(!item||!review.mask)return;
  review.mask.getContext("2d",{willReadFrequently:true}).putImageData(item,0,0);$("review-undo").disabled=!review.history.length;drawSegmentationReview();
}
function resetReview(){
  const review=state.review;if(!review.baseline||!review.mask)return;pushReviewHistory();review.mask.getContext("2d",{willReadFrequently:true}).putImageData(review.baseline,0,0);review.maskSource="human_review";$("review-status").textContent="已恢复模型分割结果；可继续人工修正。";drawSegmentationReview();
}
async function importOracleReviewDxf(file){
  const review=state.review;if(!file||!review.loaded||!review.mask)return;
  if(!file.name.toLowerCase().endsWith(".dxf")||file.size>20_000_000){toast("请选择不超过20MB的GT DXF文件。",true);return;}
  const jobId=review.jobId;
  setReviewControls(true);$("review-status").textContent="正在读取GT DXF、与原图配准并生成材料掩膜…";
  try{
    const form=new FormData();form.append("dxf",file,file.name);
    const job=await api(`/api/jobs/${encodeURIComponent(jobId)}/oracle-mask-from-dxf`,{method:"POST",body:form});
    if(review.jobId!==jobId||!review.loaded)return;
    const maskUrl=safeUrl(job.artifacts?.oracle_generated_mask);
    if(!maskUrl)throw new Error("服务端未返回DXF派生掩膜。");
    const image=await loadReviewImage(`${maskUrl}?v=${encodeURIComponent(job.updated_at||Date.now())}`);
    if(image.naturalWidth!==review.mask.width||image.naturalHeight!==review.mask.height)throw new Error("生成的GT掩膜尺寸与当前分割图不一致。");
    const canvas=document.createElement("canvas");canvas.width=review.mask.width;canvas.height=review.mask.height;
    const context=canvas.getContext("2d",{willReadFrequently:true});context.drawImage(image,0,0);
    const pixels=context.getImageData(0,0,canvas.width,canvas.height);let foreground=0;
    for(let index=0;index<pixels.data.length;index+=4){
      const value=pixels.data[index];
      if(pixels.data[index+3]!==255||value!==pixels.data[index+1]||value!==pixels.data[index+2]||(value!==0&&value!==255))throw new Error("DXF生成的掩膜不是黑白二值图。");
      pixels.data[index]=27;pixels.data[index+1]=190;pixels.data[index+2]=142;pixels.data[index+3]=value===255?255:0;
      if(value===255)foreground++;
    }
    if(foreground<3||foreground>=canvas.width*canvas.height)throw new Error("DXF生成的掩膜没有有效的单一材料区域。");
    review.mask.getContext("2d",{willReadFrequently:true}).putImageData(pixels,0,0);
    review.history=[];review.maskSource="gt_oracle";$("review-undo").disabled=true;
    state.currentJob=job;if(state.conversation?.jobs)state.conversation.jobs[job.id]=job;
    $("review-status").textContent="已从GT DXF自动生成掩膜。请检查绿色区域与原图是否对齐；确认后作为开发实验输入继续构建。";
    drawSegmentationReview();toast("GT DXF 已转换为掩膜，请检查后确认。");
  }catch(error){$("review-status").textContent=error.message;toast(error.message,true);}
  finally{$("review-dxf-file").value="";if(review.jobId===jobId&&review.loaded)setReviewControls(false);}
}
async function submitSegmentationReview(){
  const review=state.review;if(!review.loaded||!review.mask||!review.jobId)return;
  const maskSource=review.maskSource;
  setReviewControls(true);$("review-status").textContent=maskSource==="gt_oracle"?"正在保存DXF派生GT掩膜并恢复建模流水线…":maskSource==="gt_oracle_edited"?"正在保存GT辅助人工修订并恢复建模流水线…":"正在保存人工修订，并恢复参数化建模流水线…";
  try{
    const output=document.createElement("canvas"),foreground=document.createElement("canvas");output.width=foreground.width=review.mask.width;output.height=foreground.height=review.mask.height;
    const white=foreground.getContext("2d");white.fillStyle="#fff";white.fillRect(0,0,foreground.width,foreground.height);white.globalCompositeOperation="destination-in";white.drawImage(review.mask,0,0);
    const context=output.getContext("2d");context.fillStyle="#000";context.fillRect(0,0,output.width,output.height);context.drawImage(foreground,0,0);
    const blob=await new Promise((resolve,reject)=>output.toBlob(value=>value?resolve(value):reject(new Error("无法编码审核掩膜。")),"image/png"));
    const form=new FormData();form.append("mask",blob,"reviewed-segmentation-mask.png");form.append("mask_source",maskSource);
    const job=await api(`/api/jobs/${encodeURIComponent(review.jobId)}/segmentation-review`,{method:"PUT",body:form});
    state.currentJob=job;if(state.conversation?.jobs)state.conversation.jobs[job.id]=job;
    review.loaded=false;review.jobId=null;renderAll();startPolling();toast(maskSource==="gt_oracle"?"GT掩膜实验输入已记录，正在继续拓扑与参数化构建。":maskSource==="gt_oracle_edited"?"GT辅助人工修订已记录，正在继续构建。":"分割已确认，正在继续拓扑与参数化构建。");
  }catch(error){$("review-status").textContent=error.message;setReviewControls(false);toast(error.message,true);}
}
function renderWorkbench(){
  const job=state.currentJob;$("job-badge").textContent=job?`${job.mode==="autonomous_revision"?"REV":"JOB"} ${job.id.slice(0,8)}`:"NO JOB";
  $("job-id").textContent=job?job.id:"NO RESULT";
  const reviewing=job?.status==="awaiting_segmentation_review",editor=$("segmentation-editor"),stage=document.querySelector(".preview-stage");
  editor.hidden=!reviewing;stage.hidden=reviewing;
  $("preview-zoom-controls").hidden=reviewing;$("preview-fit").hidden=reviewing;
  const url=safeUrl(artifactFor(job,state.view)),image=$("preview-image"),zoomKey=`${job?.id||"none"}:${state.view}:${url||"none"}`;
  image.hidden=!url;$("preview-empty").hidden=Boolean(url);
  if(state.zoom.preview.key!==zoomKey){state.zoom.preview={scale:1,offsetX:0,offsetY:0,key:zoomKey};}
  if(url){image.onload=()=>applyZoom("preview");image.src=url+`?v=${encodeURIComponent(job.updated_at||"")}`;}else $("preview-zoom").textContent="100%";
  if(reviewing)initSegmentationEditor(job).catch(error=>{$("review-status").textContent=error.message;toast(error.message,true);});
  const status=job?.status||"idle";$("publish-state").textContent=status==="completed"?(job.validation?.passed?"已导出并回读验证":"已结束，验证未通过"):busy.has(status)?"流水线运行中":status==="cancelled"?"已停止，阶段产物已保留":status==="needs_review"?"需要复核":"尚未运行";
  if(reviewing)$("publish-state").textContent="等待人工确认分割";
  $("stage-dot").className=`stage-dot ${status==="completed"?"done":reviewing?"reviewing":busy.has(status)?"running":status==="cancelled"?"cancelled":job?"error":""}`;
  $("stage-title").textContent=status==="completed"?"本轮已完成":reviewing?"等待分割检查":busy.has(status)?"Agent 正在运行":status==="cancelled"?"任务已停止":status==="needs_review"?"等待用户复核":"等待任务";
  $("stage-detail").textContent=job?.events?.at(-1)?.message||"拖入图纸或选择数据集样本开始";
  const steps=$("stage-progress").children, progress={queued:1,extracting:1,awaiting_segmentation_review:1,solving:2,auditing:3,completed:4,needs_review:3,failed:2,cancelled:2}[status]||0;
  [...steps].forEach((el,i)=>el.classList.toggle("on",i<progress));
  syncRunButton();renderArtifactLinks(job);renderTrace(job);renderEvidence(job);loadDetails(job);
}
function renderArtifactLinks(job){
  const box=$("artifact-links");box.replaceChildren();if(!job)return;
  const labels={dxf:"下载 DXF",svg:"打开矢量",model:"模型 JSON",validation:"验证报告",segmentation_review:"分割审核记录",feedback_plan:"修正计划",topology_plan:"拓扑规划",topology_iterations:"迭代记录",reconstruction_feedback:"未解决项"};
  for(const [key,label] of Object.entries(labels)){const url=safeUrl(job.artifacts?.[key]);if(!url)continue;const a=node("a","",label);a.href=url;a.target="_blank";box.append(a);}
}
function renderTrace(job){
  const box=$("trace-list");box.replaceChildren();if(!job?.events?.length){box.append(node("p","empty-copy","运行后显示阶段轨迹。"));return;}
  for(const event of job.events){const row=node("div","trace-item");row.append(node("span","trace-icon",event.stage.includes("failed")?"×":"↳"));const copy=node("div");copy.append(node("b","",event.stage),node("p","",event.message));row.append(copy,node("time","",fmtTime(event.time)));box.append(row);}
}
function evidenceCard(label,value,detail){const card=node("div","evidence-card");card.append(node("span","",label),node("b","",value??"—"),node("small","",detail));return card;}
function renderEvidence(job){
  const box=$("evidence-grid");box.replaceChildren();if(!job){box.append(node("p","empty-copy","当前没有工程证据。"));return;}
  const entities=job.geometry?.entities||[],types=entities.reduce((a,e)=>(a[e.type]=(a[e.type]||0)+1,a),{});
  box.append(evidenceCard("GEOMETRY",entities.length,"图元总数"),evidenceCard("LINE / ARC",`${types.LINE||0} / ${types.ARC||0}`,"不使用贝塞尔曲线"),
    evidenceCard("DXF READBACK",job.validation?.dxf_readback?.passed?"PASS":job.validation?.passed?"PASS":"—","独立回读"),
    evidenceCard("SCALE",job.validation?.scaled_mm?"mm":"pixel","物理单位状态"),
    evidenceCard("ONLINE",job.provider?.http_success?"HTTP 200":job.provider?.status||"—","接口传输"),
    evidenceCard("MODEL",job.provider_profile?.model||"—",job.provider_profile?.name||"本地任务"),
    evidenceCard("PARAMETRIC",job.parameterization?.status||"—",job.parameterization?.accepted?"参数解已接受":"与拓扑结果分开"));
}
async function loadDetails(job){
  if(!job)return;const signature=`${job.id}:${job.updated_at}`;if(state.detailCache.has(signature)){renderReasonCards(state.detailCache.get(signature));return;}
  const detail={};
  for(const [key,name] of [["feedback_plan","feedback"],["topology_plan","topology"],["constraint_bindings","bindings"]]){
    const url=safeUrl(job.artifacts?.[key]);if(!url)continue;try{const response=await fetch(url);if(response.ok)detail[name]=await response.json();}catch{}
  }
  try{detail.audit=state.transcriptCache.get(signature)||await api(`/api/jobs/${encodeURIComponent(job.id)}/model-transcript`);state.transcriptCache.set(signature,detail.audit);}catch{}
  detail.primitiveDiagnostics=job.validation?.primitive_diagnostics;
  state.detailCache.set(signature,detail);if(state.currentJob?.id===job.id)renderReasonCards(detail);
}
function renderReasonCards(detail){
  const box=$("reason-cards");box.replaceChildren();const feedback=detail.feedback,topology=detail.topology,bindings=detail.bindings;
  const cards=[];
  if(feedback){cards.push(["观察",feedback.observation||"未提供可采纳观察",`证据：${(feedback.evidence_tags||[]).join(" · ")||"无"}`],["修改决策",feedback.proposed_action||"保持当前轮廓",`${feedback.selected_candidate_id||"ABSTAIN"} · ${feedback.confidence||"—"}`]);}
  if(topology){cards.push(["候选规划",topology.selected_candidate_id||"—",`${topology.candidate_count||0} 个候选 · ${topology.selection_source||"—"}`]);}
  if(bindings){cards.push(["约束绑定",`${bindings.counts?.bound_source_records??"—"} 条标注绑定`,`${bindings.counts?.unbound_dimensions??"—"} 条仍未绑定 · ${bindings.counts?.structural_accepted||0} 条结构关系`]);}
  const iterations=detail.audit?.iterations,parameterization=detail.audit?.parameterization;
  if(iterations)cards.push(["迭代进度",`${iterations.rounds?.length||0}/${iterations.max_rounds||3} 轮`,auditReasonLabels[iterations.stop_reason]||iterations.stop_reason||"运行中"]);
  if(parameterization)cards.push(["参数化求解",`${parameterization.constraints?.length||0} 项已求解约束`,`剩余形状自由度 ${parameterization.diagnostics?.remaining_shape_dof??"—"} · ${parameterization.accepted?"已接受约束子集的解":"参数解未接受"}`]);
  const primitives=detail.primitiveDiagnostics?.primitives||[],jumps=primitives.map(row=>row.tangent_jump_deg).filter(Number.isFinite);
  if(jumps.length)cards.push(["接点诊断",`最大方向跳变 ${Math.max(...jumps).toFixed(2)}°`,"测量值不代表错误；是否应相切须由原图与标注判断。"]);
  if(!cards.length){box.append(node("p","empty-copy","等待规划与校验数据。"));return;}
  for(const [label,title,copy] of cards){const card=node("div","reason-card");card.append(node("small","",label.toUpperCase()),node("b","",title),node("p","",copy));box.append(card);}
}
function renderMemory(){
  const box=$("memory-list"),memory=state.conversation?.memory;box.replaceChildren();$("memory-summary").textContent=`${memory?.turn_count||0} turns`;
  const ids=memory?.job_ids||[];if(!ids.length){box.append(node("p","empty-copy","本对话尚未保存任务上下文。"));return;}
  for(const id of ids.slice().reverse()){const job=state.conversation.jobs?.[id],item=node("div","memory-item");item.append(node("b","",`${job?.mode==="autonomous_revision"?"截图修订":"主轮廓重建"} · ${id.slice(0,8)}`),node("p","",`${job?.case_id||"upload"} · ${job?.status||"unknown"} · ${job?.geometry?.entities?.length||0} objects`));box.append(item);}
}

function setFile(kind,file){state[kind]=file;renderAttachments();}
function renderAttachments(){
  const box=$("attachment-chips");box.replaceChildren();for(const [kind,label] of [["sourceImage","原图"],["ocr","OCR"],["feedback","反馈截图"]]){const file=state[kind];if(!file)continue;const chip=node("span","attachment-chip",`${label} · ${file.name}`),close=node("button","","×");close.type="button";close.onclick=()=>setFile(kind,null);chip.append(close);box.append(chip);}
}
function acceptSourceFiles(files){for(const file of files){if(file.name.toLowerCase().endsWith(".json"))setFile("ocr",file);else if(file.type.startsWith("image/"))setFile("sourceImage",file);}}
function acceptDropped(files){
  const list=[...files],json=list.find(file=>file.name.toLowerCase().endsWith(".json")),images=list.filter(file=>file.type.startsWith("image/"));
  if(json){setFile("ocr",json);if(images[0])setFile("sourceImage",images[0]);if(images[1])setFile("feedback",images[1]);}
  else if(images.length===1&&state.currentJob)setFile("feedback",images[0]);else if(images[0])setFile("sourceImage",images[0]);
}
function acceptPastedImage(event){
  const item=[...(event.clipboardData?.items||[])].find(row=>row.kind==="file"&&row.type.startsWith("image/"));
  if(!item)return;const source=item.getAsFile();if(!source)return;
  const extension=(source.type.split("/")[1]||"png").replace("jpeg","jpg");
  const file=new File([source],`clipboard-${new Date().toISOString().replace(/[:.]/g,"-")}.${extension}`,{type:source.type||"image/png"});
  setFile(state.currentJob?"feedback":"sourceImage",file);event.preventDefault();
  toast(state.currentJob?"已粘贴为局部反馈截图。":"已粘贴为工程图；请继续添加配对 OCR JSON。");
}

async function submitTurn(event){
  event.preventDefault();if(state.submitting||busy.has(state.currentJob?.status)||!state.conversation)return;
  const prompt=$("prompt").value.trim()||(state.selectedCase?"请自动提取主轮廓并生成 DXF。":"");
  if(!prompt){toast("请先描述任务或修改要求。",true);return;}
  if((state.sourceImage&&!state.ocr)||(!state.sourceImage&&state.ocr)){toast("首次上传需要工程图和 OCR JSON 成对提供。",true);return;}
  const data=new FormData();data.append("prompt",prompt);data.append("use_api",String($("use-api").checked));data.append("use_segmentation",String($("use-segmentation").checked));
  if(state.selectedProvider)data.append("provider_id",state.selectedProvider);
  if(state.selectedCase)data.append("case_id",state.selectedCase);if(state.currentJob)data.append("job_id",state.currentJob.id);
  if(state.sourceImage)data.append("image",state.sourceImage);if(state.ocr)data.append("ocr",state.ocr);if(state.feedback)data.append("screenshot",state.feedback);
  state.submitting=true;syncRunButton();$("composer-status").classList.remove("error");$("composer-status").textContent="正在创建真实后台任务…";
  try{
    state.conversation=await api(`/api/conversations/${state.conversation.id}/turns`,{method:"POST",body:data});
    const last=state.conversation.memory?.last_job_id;state.currentJob=last?state.conversation.jobs?.[last]:state.currentJob;
    state.sourceImage=state.ocr=state.feedback=null;state.selectedCase="";if($("case-select"))$("case-select").value="";$("prompt").value="";
    renderAll();startPolling();
  }catch(error){$("composer-status").textContent=error.message;$("composer-status").classList.add("error");toast(error.message,true);}
  finally{state.submitting=false;syncRunButton();if(!$("composer-status").classList.contains("error"))$("composer-status").textContent="任务已进入后台；关闭网页不会丢失上下文。";}
}
async function stopCurrentJob(){
  const job=state.currentJob;if(!job||!busy.has(job.status)||state.stopping)return;
  state.stopping=true;syncRunButton();$("composer-status").classList.remove("error");$("composer-status").textContent="正在发送停止信号…";
  try{
    const stopped=await api(`/api/jobs/${encodeURIComponent(job.id)}/cancel`,{method:"POST"});
    state.currentJob=stopped;if(state.conversation?.jobs)state.conversation.jobs[stopped.id]=stopped;
    renderAll();toast("任务已停止；当前阶段产物和运行记录已保留。");$("composer-status").textContent="任务已停止。正在进行的网络请求可能在超时前结束，但返回结果不会发布。";
  }catch(error){$("composer-status").textContent=error.message;$("composer-status").classList.add("error");toast(error.message,true);}
  finally{state.stopping=false;syncRunButton();}
}
function startPolling(){
  clearTimeout(state.polling);const id=state.conversation?.id;if(!id)return;
  const run=async()=>{try{const value=await api(`/api/conversations/${id}`);if(state.conversation?.id!==id)return;state.conversation=value;const last=value.memory?.last_job_id;if(last)state.currentJob=value.jobs?.[last]||state.currentJob;renderAll();if(state.currentJob&&busy.has(state.currentJob.status)){state.polling=setTimeout(run,1500);}}catch(error){state.polling=setTimeout(run,4000);}};
  if(state.currentJob&&busy.has(state.currentJob.status))state.polling=setTimeout(run,900);
}

$("new-conversation").onclick=()=>createConversation().catch(error=>toast(error.message,true));
$("stop-job").onclick=stopCurrentJob;
$("delete-cancel").onclick=()=>{$("delete-dialog").close();state.deletingConversationId=null;};
$("delete-confirm").onclick=deleteConversation;
$("delete-dialog").addEventListener("cancel",()=>{state.deletingConversationId=null;});
$("composer").addEventListener("submit",submitTurn);
$("prompt").addEventListener("keydown",event=>{if(event.key!=="Tab"||event.shiftKey||$("prompt").value.trim())return;event.preventDefault();const suggestion=promptSuggestion();$("prompt").value=suggestion;$("prompt").setSelectionRange(suggestion.length,suggestion.length);$("composer-status").classList.remove("error");$("composer-status").textContent="已按当前任务补全示例，可继续修改后运行。";});
$("source-files").addEventListener("change",event=>acceptSourceFiles(event.target.files));
$("feedback-file").addEventListener("change",event=>setFile("feedback",event.target.files[0]||null));
$("drop-zone").addEventListener("paste",acceptPastedImage);
$("case-select").addEventListener("change",event=>{state.selectedCase=event.target.value;if(state.selectedCase&&!$("prompt").value)$("prompt").value=`请对 ${state.selectedCase} 自动提取主轮廓并生成 DXF。`;});
$("provider-select").addEventListener("change",updateProviderSelection);
for(const eventName of ["dragenter","dragover"]){$("drop-zone").addEventListener(eventName,event=>{event.preventDefault();$("drop-zone").classList.add("dragging");});}
for(const eventName of ["dragleave","drop"]){$("drop-zone").addEventListener(eventName,event=>{$("drop-zone").classList.remove("dragging");if(eventName==="drop"){event.preventDefault();acceptDropped(event.dataTransfer.files);}});}
document.querySelectorAll(".work-tabs button").forEach(button=>button.onclick=()=>{state.panel=button.dataset.panel;document.querySelectorAll(".work-tabs button").forEach(b=>b.classList.toggle("active",b===button));document.querySelectorAll(".work-panel").forEach(p=>p.classList.toggle("active",p.id===`panel-${state.panel}`));});
document.querySelectorAll(".preview-switches button[data-view]").forEach(button=>button.onclick=()=>{state.view=button.dataset.view;document.querySelectorAll(".preview-switches button[data-view]").forEach(b=>b.classList.toggle("active",b===button));renderWorkbench();});
$("preview-fit").onclick=()=>resetZoom("preview");
$("review-fit").onclick=()=>resetZoom("review");
$("review-add").onclick=()=>setReviewTool("add");
$("review-erase").onclick=()=>setReviewTool("erase");
$("review-dxf-file").onchange=event=>importOracleReviewDxf(event.target.files[0]);
$("review-brush").oninput=event=>{state.review.brush=Number(event.target.value);$("review-brush-value").textContent=`${state.review.brush} px`;};
$("review-undo").onclick=undoReview;
$("review-reset").onclick=resetReview;
$("review-confirm").onclick=()=>submitSegmentationReview();
{
  const canvas=$("review-canvas");let painting=false,last=null;
  canvas.addEventListener("pointerdown",event=>{if(!state.review.loaded)return;event.preventDefault();painting=true;canvas.setPointerCapture(event.pointerId);pushReviewHistory();last=reviewPoint(event);paintReview(last,last);});
  canvas.addEventListener("pointermove",event=>{if(!painting||!state.review.loaded)return;event.preventDefault();const current=reviewPoint(event);paintReview(last,current);last=current;});
  const finish=event=>{if(!painting)return;painting=false;last=null;if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);};
  canvas.addEventListener("pointerup",finish);canvas.addEventListener("pointercancel",finish);
}
document.querySelector(".preview-stage").addEventListener("wheel",event=>zoomAtPointer("preview",event),{passive:false});
document.querySelector(".review-canvas-wrap").addEventListener("wheel",event=>zoomAtPointer("review",event),{passive:false});
window.addEventListener("resize",()=>{applyZoom("preview");applyZoom("review");});
loadBootstrap();

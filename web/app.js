"use strict";

const $ = (id) => document.getElementById(id);
const workflow = window.ContourWorkflow;
const state = {
  catalog: { cases: [], counts: {} }, templates: [], config: {}, selected: null,
  job: null, filter: "all", view: "source", zoom: 1, highlight: null,
  viewZoom: { source: 1, vector: 1, overlay: 1, segmentation: 1, contour: 1 }, segmentationMask: false,
  pilotJobs: new Map(), pilotSubmitting: false, pilotPoll: null, pilotPollErrors: 0,
  parameterNodes: new Map(), assumptionNodes: new Map(), touched: new Set(),
  assumptionTouched: new Set(), poll: null, pollErrors: 0, epoch: 0,
  submitting: false, eventSignature: "", toastTimer: null,
};
const statuses = {
  queued: ["等待处理", "review"], extracting: ["正在提取", "review"],
  needs_review: ["旧任务待确认", "review"], solving: ["正在定标与导出", "review"],
  auditing: ["已出图 · 视觉复核中", "review"],
  completed: ["自动出图", "good"], unsupported: ["旧流程未支持", ""],
  failed: ["处理失败", "bad"], cancelled: ["已取消", ""],
};
const busyStates = new Set(["queued", "extracting", "solving", "auditing"]);
const isAutomatic = (job = state.job) => !job || job.mode === "autonomous_image";
const hasArtifact = (job = state.job) => Boolean(job?.artifacts?.svg || job?.artifacts?.dxf);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
function safeUrl(value) {
  if (typeof value !== "string" || !value.startsWith("/api/") || value.startsWith("//")) return null;
  const url = new URL(value, window.location.origin);
  return url.origin === window.location.origin ? url.pathname + url.search : null;
}
async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = body.detail;
      const message = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map(x => x.msg || "输入无效").join("；") : `请求失败（HTTP ${response.status}）`;
      throw new Error(message);
    }
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("本地服务响应超时；已提交的任务可在任务记录中恢复。");
    throw error;
  } finally { clearTimeout(timeout); }
}
function post(path, data) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
}
function toast(message, error = false) {
  clearTimeout(state.toastTimer);
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").hidden = false;
  state.toastTimer = setTimeout(() => { $("toast").hidden = true; }, error ? 8500 : 5500);
}
function shortCase(id) {
  if (!id || id === "upload") return "上传图纸";
  const match = id.match(/img_0*(\d+)/);
  return match ? `SOLID ${match[1]}` : id.replace(/-main$/, "");
}
function setStatus(node, value, fallback = "待开始") {
  const [text, style] = statuses[value] || [fallback, ""];
  node.textContent = text;
  node.className = `status-badge ${style}`;
}
function readableTime(value, full = false) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "—";
  return full ? date.toLocaleString("zh-CN", { hour12: false }) : date.toLocaleTimeString("zh-CN", { hour12: false });
}
function isBusy() { return Boolean(state.job && busyStates.has(state.job.status)); }

function renderPilots() {
  const busy = state.pilotSubmitting || [...state.pilotJobs.values()].some(job => busyStates.has(job.status));
  const uncertain = [...state.pilotJobs.values()].some(job => job.status === "submit_error");
  const available = workflow.pilotIds.every(id => state.catalog.cases.some(c => c.id === id));
  $("run-pilots").disabled = busy || uncertain || state.submitting || isBusy() || !available || !state.config.segmentation_available;
  $("run-pilots").textContent = uncertain ? "提交待核对 · 见任务记录" : busy ? "4 张任务处理中…" : state.pilotJobs.size ? "重新运行这 4 张" : "分割并绘制这 4 张";
  $("pilot-cases").replaceChildren(...workflow.pilotIds.map(id => {
    const c = state.catalog.cases.find(item => item.id === id), job = state.pilotJobs.get(id);
    const button = element("button", "pilot-case"); button.type = "button";
    button.disabled = !c || state.pilotSubmitting;
    button.setAttribute("aria-pressed", String(state.selected?.id === id));
    button.append(element("strong", "", shortCase(id).replace("SOLID ", "")));
    button.append(element("small", "", job?.status === "submit_error" ? "提交未确认" : job ? (statuses[job.status]?.[0] || "已提交") : "选择原图"));
    button.addEventListener("click", async () => {
      if (!job?.id) { selectCase(c); if (!$("use-segmentation").disabled) $("use-segmentation").checked = true; renderHeading(); return; }
      button.disabled = true;
      try { const current = await api(`/api/jobs/${encodeURIComponent(job.id)}`); state.epoch += 1; adoptJob(current, { reset: true }); }
      catch (error) { toast(error.message, true); } finally { button.disabled = false; }
    });
    return button;
  }));
  const completed = [...state.pilotJobs.values()].filter(job => job.status === "completed").length;
  $("pilot-status").textContent = state.pilotSubmitting ? "逐个提交任务，运行完成后点击样本查看。" : state.pilotPollErrors ? "任务刷新暂时失败；已提交任务仍可从任务记录恢复。" : state.pilotJobs.size ? `本批 ${completed} / 4 已结束出图；完成不代表尺寸验证通过。` : state.config.segmentation_available ? "使用当前模型与复核选项。" : "需先配置分割模型。";
}

async function runPilots() {
  if ($("run-pilots").disabled || state.pilotSubmitting || state.submitting || isBusy()) return;
  state.pilotSubmitting = true; state.pilotPollErrors = 0; state.pilotJobs.clear();
  clearTimeout(state.pilotPoll); renderPilots(); renderHeading();
  const useApi = $("use-api").checked;
  try {
    for (const id of workflow.pilotIds) {
      try {
        const job = await post("/api/jobs", { case_id: id, mode: "autonomous_image", use_api: useApi, use_segmentation: true });
        state.pilotJobs.set(id, job);
        if (state.pilotJobs.size === 1) { state.epoch += 1; adoptJob(job, { reset: true }); }
      } catch (error) {
        state.pilotJobs.set(id, { case_id: id, status: "submit_error" });
        toast(`${shortCase(id)}：${error.message} 未自动重复提交，请先检查任务记录。`, true);
        break;
      }
      renderPilots();
    }
  } finally { state.pilotSubmitting = false; renderPilots(); renderHeading(); pollPilots(); }
}

function pollPilots() {
  clearTimeout(state.pilotPoll);
  const running = [...state.pilotJobs.values()].filter(job => job.id && busyStates.has(job.status));
  if (!running.length) return;
  state.pilotPoll = setTimeout(async () => {
    const results = await Promise.allSettled(running.map(job => api(`/api/jobs/${encodeURIComponent(job.id)}`)));
    let failures = 0;
    results.forEach(result => {
      if (result.status !== "fulfilled") { failures += 1; return; }
      const job = result.value; state.pilotJobs.set(job.case_id, job);
      if (state.job?.id === job.id) adoptJob(job);
    });
    state.pilotPollErrors = failures ? state.pilotPollErrors + 1 : 0;
    renderPilots(); pollPilots();
  }, state.pilotPollErrors ? 8000 : 2200);
}

function renderCatalog() {
  const counts = state.catalog.counts;
  $("dataset-count").textContent = String(counts.total ?? "—");
  $("all-count").textContent = String(counts.total ?? "—");
  $("supported-count").textContent = String(counts.paired ?? counts.total ?? "—");
  $("filter-all").classList.toggle("active", state.filter === "all");
  $("filter-supported").classList.toggle("active", state.filter === "supported");
  const search = $("case-search").value.trim().toLowerCase();
  const visible = state.catalog.cases.filter(c => (state.filter !== "supported" || c.image && c.ocr) && (!search || c.id.toLowerCase().includes(search) || shortCase(c.id).toLowerCase().includes(search)));
  const fragment = document.createDocumentFragment();
  visible.forEach((c) => {
    const button = element("button", "case-button");
    button.type = "button";
    button.classList.toggle("selected", c.id === state.selected?.id);
    button.classList.toggle("supported", Boolean(c.image && c.ocr));
    button.setAttribute("aria-pressed", String(c.id === state.selected?.id));
    button.title = c.id;
    const match = c.id.match(/img_0*(\d+)/);
    button.append(element("span", "case-number", match ? match[1] : "—"));
    const description = element("span");
    description.append(element("strong", "", shortCase(c.id)));
    description.append(element("small", "", c.image && c.ocr ? "自动识别 · 图像 + OCR" : "来源文件待补齐"));
    button.append(description, element("span", c.image && c.ocr ? "support-dot" : ""));
    button.addEventListener("click", () => selectCase(c));
    fragment.append(button);
  });
  if (!visible.length) fragment.append(element("p", "empty-small muted", "没有符合条件的图纸。"));
  $("case-list").replaceChildren(fragment);
  renderPilots();
}

function resetEditors() {
  state.parameterNodes.clear(); state.assumptionNodes.clear();
  state.touched.clear(); state.assumptionTouched.clear();
  state.eventSignature = ""; state.highlight = null;
  $("parameters").replaceChildren(); $("assumptions").replaceChildren();
  $("confirm-bindings").checked = false;
}
function selectCase(c) {
  clearTimeout(state.poll); state.epoch += 1; state.pollErrors = 0;
  state.selected = c; state.job = null; state.view = "source"; state.zoom = 1;
  state.viewZoom = { source: 1, vector: 1, overlay: 1, segmentation: 1, contour: 1 }; state.segmentationMask = false;
  resetEditors(); renderCatalog(); renderJob();
}

function renderHeading() {
  const job = state.job, selected = state.selected;
  const id = job?.case_id || selected?.id;
  $("drawing-title").textContent = shortCase(id);
  $("drawing-split").textContent = isAutomatic(job) ? "AUTONOMOUS" : "LEGACY TEMPLATE";
  $("drawing-subtitle").textContent = job?.uploaded ? "上传图纸 · 自动识别材料主边界" : id || "选择图纸开始";
  setStatus($("job-status"), job?.status);
  if (isAutomatic(job) && job?.status === "completed") {
    const issue = job.provider?.verdict === "mismatch" ? "图像核验不一致" : job.validation?.scaled_mm === false ? "比例未确定" : null;
    if (issue) { $("job-status").textContent = `自动出图 · ${issue}`; $("job-status").className = "status-badge review"; }
    if (job.automatic_completion === false) { $("job-status").textContent = "产物校验未通过"; $("job-status").className = "status-badge bad"; }
  } else if (job?.status === "completed") $("job-status").textContent = "旧模板结果";
  if (isAutomatic(job) && job?.completion_class === "incomplete_material_exterior_draft") {
    $("job-status").textContent = "分割未连通 · 局部草稿";
    $("job-status").className = "status-badge review";
  }
  $("start-job").disabled = state.submitting || state.pilotSubmitting || isBusy() || !selected;
  $("start-job").textContent = isBusy() ? "正在自动处理…" : job ? "重新自动绘制" : "▷ 自动绘制主轮廓";
  $("cancel-job").hidden = !isBusy();
  $("cancel-job").disabled = state.submitting;
  $("extract-stage-label").textContent = job ? job.use_segmentation ? "预测分割" : "图像提取" : $("use-segmentation").checked ? "预测分割" : "图像提取";
  const index = { queued: 0, extracting: 1, needs_review: 2, solving: 2, auditing: 3, completed: 4 }[job?.status] ?? -1;
  document.querySelectorAll(".pipeline li").forEach((node, i) => {
    node.classList.toggle("done", i < index);
    node.classList.toggle("active", i === index);
  });
  renderPilots();
}

function renderViewer() {
  const job = state.job;
  const vector = state.view === "vector";
  const overlay = state.view === "overlay";
  const segmentation = state.view === "segmentation", contour = state.view === "contour", topology = state.view === "topology";
  workflow.views.forEach(view => {
    $(`view-${view}`).classList.toggle("active", state.view === view);
    $(`view-${view}`).setAttribute("aria-selected", String(state.view === view));
  });
  $("view-overlay").hidden = !workflow.artifactForView(job, "overlay");
  $("view-contour").hidden = !workflow.artifactForView(job, "contour");
  $("view-topology").hidden = !workflow.artifactForView(job, "topology");
  $("view-segmentation").hidden = !(job?.artifacts?.segmentation_overlay || job?.artifacts?.segmentation_mask);
  if (!job?.artifacts?.segmentation_overlay && job?.artifacts?.segmentation_mask) state.segmentationMask = true;
  if (!job?.artifacts?.segmentation_mask) state.segmentationMask = false;
  $("segmentation-view-options").hidden = !segmentation;
  ["overlay", "mask"].forEach(kind => {
    const node = $(`segmentation-${kind}-option`), active = state.segmentationMask === (kind === "mask");
    node.disabled = !job?.artifacts?.[`segmentation_${kind}`];
    node.classList.toggle("active", active); node.setAttribute("aria-pressed", String(active));
  });
  $("vector-ready").hidden = !job?.artifacts?.svg;
  $("drawing-frame").classList.toggle("vector-frame", vector);
  $("zoom-value").textContent = `${Math.round(state.zoom * 100)}%`;
  let url = state.view === "source" ? job?.source_image_url || (state.selected?.image ? `/api/cases/${encodeURIComponent(state.selected.id)}/image` : null) : workflow.artifactForView(job, state.view, state.segmentationMask);
  url = safeUrl(url);
  if (url && vector) url += `${url.includes("?") ? "&" : "?"}revision=${Number(job?.solve_revision || 0)}`;
  $("drawing-frame").hidden = !url;
  $("drawing-image").hidden = !url;
  $("canvas-empty").hidden = Boolean(url);
  if (url && $("drawing-image").getAttribute("src") !== url) $("drawing-image").src = url;
  $("drawing-image").alt = topology ? "近似图元和共享连接点编号" : segmentation ? "当前任务真实模型分割预测" : contour ? "预测区域提取边界与原图叠加" : vector ? "拟合并导出的 CAD 主轮廓" : overlay ? "拟合 CAD 与原始图纸的叠加对照" : `${shortCase(job?.case_id || state.selected?.id)} 原始工程图纸`;
  $("canvas-caption").textContent = segmentation ? state.segmentationMask ? "白色为模型预测区域；本图为当前原图的推理结果。" : "色块为当前模型预测区域，可与原始线条直接对照。" : contour ? "从预测区域提取的主边界；尚未等同于拟合后的 CAD。" : overlay ? "拟合后的 LINE / ARC 与原图叠加；尺寸约束状态见右侧。" : vector ? isAutomatic(job) ? `自动轮廓 · ${job?.validation?.scaled_mm ? "毫米比例已估计" : "像素单位 / 比例未确定"} · 尺寸约束尚未全面验证` : "旧模板辅助结果 · 包含形状先验与简化踏面" : "原图保持原始坐标；点击识别证据可定位标注。";
  $("source-size").textContent = !vector && job?.source ? `${job.source.width} × ${job.source.height} px` : "";
  $("source-size").hidden = vector;
  if (topology) $("canvas-caption").textContent = "绿色为近似图元，紫色为共享连接点；编号用于标注绑定，候选参数尚未全部求解。";
  $("source-highlight").hidden = state.view !== "source" || !state.highlight;
  if (state.view === "source" && state.highlight) {
    const h = state.highlight;
    Object.assign($("source-highlight").style, { left: `${h.x}%`, top: `${h.y}%`, width: `${h.w}%`, height: `${h.h}%` });
    $("highlight-label").textContent = h.label;
  }
  requestAnimationFrame(fitViewer);
}

function fitViewer() {
  const image = $("drawing-image"), viewport = $("drawing-viewport"), frame = $("drawing-frame");
  if (frame.hidden || !image.complete || !image.naturalWidth || !image.naturalHeight) return;
  const style = getComputedStyle(viewport);
  const width = Math.max(1, viewport.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight) - 2);
  const height = Math.max(1, viewport.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom) - 2);
  const ratio = image.naturalWidth / image.naturalHeight;
  // 100% means fit the complete drawing in both axes. User zoom is a separate
  // multiplier and survives polling and resize; it is never silently reset.
  const fitWidth = Math.min(width, height * ratio);
  frame.style.width = `${Math.max(1, fitWidth * state.zoom)}px`;
  if (state.zoom <= 1) { viewport.scrollTop = 0; viewport.scrollLeft = 0; }
}

function switchView(view) {
  state.viewZoom[state.view] = state.zoom;
  state.view = view;
  state.zoom = state.viewZoom[view] || 1;
  renderViewer();
}

function showSource(row) {
  const source = state.job?.source;
  if (!Array.isArray(row.source_box) || !source?.width || !source?.height) {
    toast("此参数没有唯一的 OCR 来源，请直接核对原图或补充尺寸。"); return;
  }
  const xs = row.source_box.map(p => p[0]), ys = row.source_box.map(p => p[1]);
  state.highlight = {
    x: Math.max(0, Math.min(...xs) / source.width * 100),
    y: Math.max(0, Math.min(...ys) / source.height * 100),
    w: (Math.max(...xs) - Math.min(...xs)) / source.width * 100,
    h: (Math.max(...ys) - Math.min(...ys)) / source.height * 100,
    label: `${row.source_record_id} · ${row.raw_text || row.label}`,
  };
  state.viewZoom[state.view] = state.zoom;
  state.view = "source"; state.zoom = Math.max(1.4, state.viewZoom.source); renderViewer();
  requestAnimationFrame(() => $("source-highlight").scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" }));
}

function addParameter(row) {
  const wrapper = element("div", "parameter-row");
  const top = element("div", "parameter-top");
  const label = element("label", "parameter-label", row.label);
  label.htmlFor = `parameter-${row.id}`;
  label.append(element("span", "parameter-id", row.id));
  const inputWrap = element("div", "parameter-input-wrap");
  const input = element("input");
  input.id = `parameter-${row.id}`; input.type = "number"; input.step = "any";
  input.inputMode = "decimal"; input.required = true; input.placeholder = "待补齐";
  if (Number.isFinite(row.min)) input.min = row.min;
  if (Number.isFinite(row.max)) input.max = row.max;
  input.setAttribute("aria-label", `${row.label}，${row.unit === "deg" ? "度" : row.unit}`);
  input.addEventListener("input", () => { state.touched.add(row.id); input.classList.toggle("missing", input.value === ""); updateReadiness(); });
  inputWrap.append(input, element("span", "parameter-unit", row.unit === "deg" ? "°" : row.unit));
  top.append(label, inputWrap);
  const sourceLine = element("div", "source-line");
  const sourceButton = element("button", "source-button"); sourceButton.type = "button";
  sourceButton.addEventListener("click", () => showSource(state.job.parameters.find(r => r.id === row.id)));
  const sourceType = element("span", "source-type");
  const suggestion = element("button", "suggestion-button"); suggestion.type = "button";
  suggestion.addEventListener("click", () => {
    const current = state.job.parameters.find(r => r.id === row.id);
    const proposed = current.suggested_value;
    if (Number.isFinite(proposed)) {
      input.value = proposed; state.touched.add(row.id); input.classList.remove("missing"); updateReadiness();
      toast(`已填入 ${current.label} 的校准建议值；请核对其来源与适用性。`);
    }
  });
  sourceLine.append(sourceButton, sourceType, suggestion);
  const note = element("p", "parameter-note");
  wrapper.append(top, sourceLine, note); $("parameters").append(wrapper);
  state.parameterNodes.set(row.id, { input, sourceButton, sourceType, suggestion, note });
}

function metricRow(label, value) {
  const row = element("div", "evidence-metric");
  row.append(element("span", "", label), element("strong", "mono", value));
  return row;
}

function renderWorkflowEvidence() {
  const host = $("workflow-evidence"), job = state.job;
  host.replaceChildren(); host.hidden = !isAutomatic(job);
  if (!isAutomatic(job)) return;
  const labels = { ready: "已产出", running: "处理中", pending: "等待", stopped: "未完成", skipped: "未使用", warning: "待检查" };
  workflow.stageEvidence(job).forEach(stage => {
    const card = element("section", `workflow-stage ${stage.status}`), head = element("div", "workflow-stage-heading");
    head.append(element("h3", "", stage.title), element("span", "", labels[stage.status]));
    card.append(head, element("p", "", stage.detail));
    const view = { segmentation: "segmentation", contour: "contour", topology: "topology", cad: "overlay" }[stage.key];
    const available = workflow.artifactForView(job, view) || (view === "segmentation" && job?.artifacts?.segmentation_mask);
    if (available) {
      const button = element("button", "text-button", "在原图上查看 →"); button.type = "button";
      button.addEventListener("click", () => switchView(view)); card.append(button);
    }
    host.append(card);
  });
  const analysis = job?.dimension_analysis;
  if (analysis) {
    const section = element("section", "dimension-analysis evidence-section");
    const names = { pending: "等待解析", running: "解析中", completed: "已完成解析", succeeded: "已完成解析", failed: "解析未完成", disabled: "未启用", skipped: "未执行" };
    section.append(element("h3", "", `尺寸解析 · ${names[analysis.status] || analysis.status || "未取得"}`));
    section.append(element("p", "evidence-empty", "尺寸来源与对应关系单独记录；解析完成不表示所有几何约束已满足。"));
    const provider = analysis.provider || {};
    if (provider.status || provider.network_requests || provider.http_success !== undefined) section.append(metricRow("尺寸 API / 结构", workflow.dimensionProviderText(provider)));
    const counts = analysis.counts || {};
    Object.entries(counts).filter(([, value]) => typeof value === "number").slice(0, 7).forEach(([key, value]) => {
      const label = { ocr_records: "OCR 记录", recognized_dimensions: "识别的尺寸", bound_source_records: "已绑定来源", api_selected: "API 选择", api_agreed: "API 核对一致", api_conflicts: "API 核对冲突", unbound_dimensions: "未绑定尺寸" }[key] || key;
      section.append(metricRow(label, value));
    });
    const url = safeUrl(job?.artifacts?.dimension_analysis);
    if (url) { const link = element("a", "text-button", "查看尺寸解析记录 ↗"); link.href = url; link.target = "_blank"; link.rel = "noopener"; section.append(link); }
    host.append(section);
  }
  const parameterization = job?.parameterization;
  if (parameterization) {
    const section = element("section", "evidence-section"), solver = parameterization.solver || {};
    section.append(element("h3", "", "标注绑定与参数化"));
    section.append(metricRow("绑定 API / 结构", workflow.dimensionProviderText(parameterization.provider || {})));
    section.append(metricRow("尺寸与关系约束", (parameterization.constraints || []).length));
    section.append(metricRow("数值解", parameterization.accepted ? "已采用部分约束解" : parameterization.status === "running" ? "求解中" : parameterization.topology_exported ? "未采用，保留原图校正轮廓" : "未采用，保留初始拟合"));
    if (Number.isInteger(solver.diagnostics?.remaining_shape_dof)) section.append(metricRow("剩余形状自由度", solver.diagnostics.remaining_shape_dof));
    section.append(element("p", "evidence-empty", "连接顺序来自近似轮廓，约束数值来自原图标注。已满足的约束与未绑定标注分别记录；不代表全部尺寸通过。"));
    const url = safeUrl(job.artifacts?.parametric_solution);
    if (url) { const link = element("a", "text-button", "查看约束残差与剩余自由度 ↗"); link.href = url; link.target = "_blank"; link.rel = "noopener"; section.append(link); }
    host.append(section);
  }
}

function renderAutomaticEvidence() {
  const job = state.job, host = $("automatic-evidence");
  host.replaceChildren();
  if (!job) { $("parameter-count").textContent = "自动"; return; }
  const scale = job.scale || {}, provider = job.provider || {}, validation = job.validation || {};
  const bindings = Array.isArray(scale.bindings) ? scale.bindings : [];
  const crossChecks = scale.status === "resolved" ? (scale.linear_witnesses || []).filter(row => Math.abs(row.pixels_per_mm / scale.pixels_per_mm - 1) <= .035 && !bindings.some(binding => binding.record_id === row.record_id)) : [];
  const radii = Array.isArray(validation.radius_bindings) ? validation.radius_bindings : [];
  const entities = job.geometry?.entities || [];
  $("parameter-count").textContent = `${bindings.length + crossChecks.length + radii.length} 条证据`;
  const summary = element("section", "evidence-summary");
  const ratio = scale.resolved_pixels_per_mm ?? scale.pixels_per_mm;
  summary.append(metricRow("轮廓提取方式", job.use_segmentation ? (job.extraction?.model?.label_status === "registered_dxf_gt_experimental" ? "U-Net · DXF GT监督" : job.extraction?.model?.label_status === "weak_supervision_experimental" ? "U-Net · 弱监督试验" : "U-Net · 读取模型信息") : "局部图像算法"));
  summary.append(metricRow("输出单位", validation.scaled_mm === true ? "mm" : hasArtifact(job) ? "px · 未定标" : "计算中"));
  summary.append(metricRow("图像比例", scale.status === "resolved" && Number.isFinite(ratio) ? `${ratio.toFixed(4)} px/mm` : scale.status === "ambiguous" ? "存在多个解释" : "尚未确定"));
  summary.append(metricRow("轮廓实体", entities.length ? `${entities.length} / LINE + ARC` : "提取中"));
  const visualText = provider.schema_success ? { match: "一致", mismatch: "不一致", uncertain: "不确定" }[provider.verdict] || "不确定" : job.status === "auditing" ? "复核中" : provider.status === "failed" ? "未取得结果" : "未执行";
  summary.append(metricRow("在线图像核验", visualText));
  if (provider.network_requests) summary.append(metricRow("API 传输 / 结构", `${provider.http_success ? "成功" : "未成功"} / ${provider.schema_success ? "有效" : "未取得"}`));
  host.append(summary);
  const addBindings = (title, rows, kind) => {
    const section = element("section", "evidence-section");
    section.append(element("h3", "", `${title} · ${rows.length}`));
    if (!rows.length) section.append(element("p", "evidence-empty", kind === "scale" ? "尚未形成一致的标注定标证据。" : "尚未绑定到可验证的半径标注。"));
    rows.forEach(binding => {
      const row = element("div", "evidence-binding"), top = element("div", "evidence-binding-top");
      top.append(element("span", "mono", binding.record_id || "OCR"), element("strong", "", binding.text || `${binding.nominal ?? "—"}`));
      row.append(top);
      if (Array.isArray(binding.box)) {
        const button = element("button", "source-button", "定位原图标注 ↗"); button.type = "button";
        button.addEventListener("click", () => showSource({ source_box: binding.box, source_record_id: binding.record_id, raw_text: binding.text }));
        row.append(button);
      }
      row.append(element("p", "parameter-note", kind !== "radius" ? `标注名义值 ${binding.nominal ?? "—"} · 尺寸线关联` : `半径名义值 ${binding.nominal ?? "—"} · 局部圆弧拟合`));
      section.append(row);
    });
    host.append(section);
  };
  addBindings("比例来源", bindings, "scale");
  if (crossChecks.length) addBindings("线性尺寸交叉检查", crossChecks, "scale");
  addBindings("半径关联", radii, "radius");
  if (Array.isArray(provider.issues) && provider.issues.length) {
    const notes = element("section", "evidence-section");
    notes.append(element("h3", "", "视觉复核意见"), ...provider.issues.map(issue => element("p", "evidence-empty", issue)));
    host.append(notes);
  }
}

function renderParameters() {
  const job = state.job;
  const automatic = isAutomatic(job);
  $("automatic-evidence").hidden = !automatic;
  $("parameter-form").hidden = automatic;
  $("confirm-solve").hidden = automatic;
  $("inspector-title").textContent = automatic ? "自动识别证据" : "旧模板参数";
  $("inspector-subtitle").textContent = automatic ? "尺寸来源 · 比例估计 · 图像一致性" : "历史模板辅助流程 · 参数与假设";
  $("inspector-footnote").replaceChildren(element("span", "", automatic ? "自动产物、图像一致性与尺寸验证分别记录。" : "旧模板结果包含校准先验与简化踏面。"), element("br"), element("strong", "", "可编辑轮廓尚非制造认证结果"));
  if (automatic) {
    $("parameter-toolbar").hidden = true;
    $("assumptions-section").hidden = true;
    renderAutomaticEvidence();
    return;
  }
  const rows = job?.parameters || [];
  const usable = rows.length > 0;
  $("parameter-toolbar").hidden = !usable;
  $("assumptions-section").hidden = !usable;
  rows.forEach(row => {
    if (!state.parameterNodes.has(row.id)) addParameter(row);
    const nodes = state.parameterNodes.get(row.id);
    if (!state.touched.has(row.id)) nodes.input.value = Number.isFinite(row.value) ? String(row.value) : "";
    nodes.input.classList.toggle("missing", nodes.input.value === "");
    nodes.input.disabled = isBusy();
    nodes.sourceButton.textContent = row.source_record_id ? `${row.source_record_id} · ${row.raw_text || "查看标注"}` : "无唯一 OCR 来源";
    nodes.sourceButton.classList.toggle("unresolved", !row.source_record_id);
    nodes.sourceButton.title = row.source_record_id ? `定位原图：${row.raw_text || row.source_record_id}` : row.note;
    nodes.sourceType.textContent = row.confirmed_by ? "已确认" : row.provider_agrees_with_local === false ? "API 与规则有分歧" : row.source_kind === "ocr_registered_layout" ? "版式绑定" : "待核对";
    nodes.suggestion.hidden = !Number.isFinite(row.suggested_value);
    nodes.suggestion.textContent = `建议 ${row.suggested_value}`;
    nodes.suggestion.title = "采用已声明的校准建议值；仍需人工核对";
    nodes.suggestion.disabled = isBusy();
    nodes.note.textContent = row.note || "";
    nodes.note.classList.toggle("attention", Boolean(row.needs_review));
  });
  (job?.assumptions || []).forEach(assumption => {
    if (!state.assumptionNodes.has(assumption.id)) {
      const label = element("label", "assumption-item"), check = element("input");
      check.type = "checkbox"; check.value = assumption.id; check.required = Boolean(assumption.required);
      check.checked = assumption.confirmed === true;
      check.addEventListener("change", () => { state.assumptionTouched.add(assumption.id); updateReadiness(); });
      label.append(check, element("span", "", assumption.label)); $("assumptions").append(label);
      state.assumptionNodes.set(assumption.id, check);
    }
    const check = state.assumptionNodes.get(assumption.id);
    if (!state.assumptionTouched.has(assumption.id)) check.checked = assumption.confirmed === true;
    check.disabled = isBusy();
  });
  $("confirm-bindings").disabled = isBusy();
  $("fill-suggestions").disabled = isBusy();
  updateReadiness();
}

function updateReadiness() {
  if (isAutomatic()) { $("confirm-solve").disabled = true; return; }
  const rows = state.job?.parameters || [];
  const nodes = [...state.parameterNodes.values()];
  const filled = nodes.filter(n => n.input.value !== "" && Number.isFinite(Number(n.input.value)) && n.input.validity.valid).length;
  $("parameter-count").textContent = rows.length ? `${filled} / ${rows.length}` : "— / —";
  $("missing-count").textContent = rows.length === filled ? "参数已齐备，核对后确认" : `${rows.length - filled} 项待补齐或修正`;
  const checks = [...state.assumptionNodes.values()];
  $("assumptions-count").textContent = `${checks.filter(c => c.checked).length} / ${checks.length}`;
  const canConfirm = state.job && ["needs_review", "completed", "failed"].includes(state.job.status) && rows.length > 0;
  const ready = canConfirm && filled === rows.length && checks.every(c => !c.required || c.checked) && $("confirm-bindings").checked;
  $("confirm-solve").disabled = !ready || state.submitting;
  $("confirm-solve").textContent = state.job?.status === "solving" ? "正在重建与验证…" : "确认参数并重建 →";
}

function renderNotice() {
  const job = state.job;
  let title = "选择图纸，一键自动绘制", message = "自动处理图像材料边界与尺寸标注。全部图纸可运行，成功数以独立评估为准。", style = "";
  if (isAutomatic(job)) {
    if (job?.completion_class === "incomplete_material_exterior_draft") {
      title = "材料区域未连通，已保留局部轮廓草稿";
      message = "分割中仍有独立区域或仅角点相连的区域。闭合 DXF 只覆盖选定主体，不能计为完整重建；分割图和原始预测均已保留供对照。";
      style = "warn";
    } else if (job?.status === "completed") {
      title = job.automatic_completion ? "自动轮廓已导出" : "已生成候选，产物校验未通过";
      message = job.provider?.verdict === "mismatch" ? "图像复核指出候选边界存在不一致。产物与原图叠加结果均已保留，可直接查看差异。" : job.validation?.scaled_mm === false ? "主轮廓已输出。标注比例尚未唯一确定，文件明确使用像素单位。" : "已根据图像边界和标注比例生成可编辑轮廓。完整尺寸约束与参考准确率仍需单独验证。";
      style = job.automatic_completion && job.provider?.verdict !== "mismatch" && job.validation?.scaled_mm ? "good" : "warn";
    } else if (job?.status === "auditing") {
      title = "轮廓已出图，正在视觉复核"; message = "DXF、矢量预览与原图叠加已可查看。在线复核结果会单独记录，不阻塞本地产物下载。";
    } else if (job?.status === "failed") {
      title = "本次自动处理未完成"; message = "已保留阶段进度及失败原因，详见运行记录。"; style = "warn";
    } else if (isBusy()) {
      title = job.status === "solving" ? "正在关联标注与导出轮廓" : "正在自动提取材料边界";
      message = "系统自动识别、定标并生成 CAD 产物，无需填写模板参数。";
    }
  } else if (job?.status === "unsupported") {
    title = "需要新增适用模板"; message = "当前无法自动重建此拓扑。任务已记录为未支持，不计为成功。"; style = "warn";
  } else if (job?.status === "needs_review") {
    title = job.provider?.status === "failed" ? "接口失败，尺寸进度已保留" : "请核对尺寸与建模假设";
    message = job.provider?.status === "failed" ? `${job.provider.message || "API 未成功完成。"} 可补齐参数继续重建。` : "橙色标记需要特别核对。建议值来自模板校准，点击采用也不会自动确认假设。"; style = "warn";
  } else if (job?.status === "completed") {
    title = "已完成模板辅助重建"; message = "数值几何校验通过，产物可下载。结果包含确认的形状先验与简化踏面；未取得制造认可。"; style = "good";
  } else if (job?.status === "failed") {
    title = "输入或处理需要修正"; message = "请查看下方问题和运行记录；已提取的参数会保留。"; style = "warn";
  } else if (isBusy()) {
    title = job.status === "solving" ? "正在重新计算几何" : "正在建立尺寸来源";
    message = job.status === "solving" ? "按当前确认参数重新计算圆心、切点和线段，并回读 DXF 验证。" : "本地规则先提取尺寸。开启 API 后只提交短 OCR 文字，不提交参考 DXF。";
  }
  $("review-notice").className = `review-notice ${style}`;
  $("review-notice").replaceChildren(element("strong", "", title), element("p", "", message));
  const issues = job?.issues || [];
  $("issues").hidden = issues.length === 0;
  $("issues").replaceChildren(...issues.slice(0, 5).map(issue => element("p", "", typeof issue === "string" ? issue : JSON.stringify(issue))));
  if (issues.length > 5) $("issues").append(element("p", "", `另有 ${issues.length - 5} 项，见对应参数说明。`));
}

function renderEvents() {
  const events = state.job?.events || [];
  $("job-short-id").textContent = state.job ? `JOB ${state.job.id.slice(0, 10)}` : "尚未创建任务";
  $("event-count").textContent = String(events.length).padStart(2, "0");
  const signature = `${state.job?.id || ""}:${events.length}:${events.at(-1)?.time || ""}`;
  if (signature === state.eventSignature) return;
  state.eventSignature = signature;
  if (!events.length) {
    $("events").replaceChildren(element("p", "muted empty-small", "轮廓提取、定标和导出自动执行；在线复核结果单独记录。")); return;
  }
  $("events").replaceChildren(...events.slice(-30).map(event => {
    const row = element("div", "event-row");
    row.classList.toggle("failure", /failed|fallback|unsupported/.test(event.stage || ""));
    row.append(element("time", "", readableTime(event.time)), element("span", "event-dot"), element("span", "", event.message));
    return row;
  }));
  $("events").scrollTop = $("events").scrollHeight;
}

function renderArtifacts() {
  const job = state.job, artifacts = job?.artifacts || {};
  $("artifact-bar").hidden = !Object.values(artifacts).some(safeUrl);
  $("artifact-links").replaceChildren();
  const labels = { dxf: "下载 DXF ↓", svg: "SVG ↗", segmentation_evidence: "分割记录", curve_fit: "曲线拟合", dimension_analysis: "尺寸解析", model: "几何 JSON", validation: "验证报告" };
  for (const [key, label] of Object.entries(labels)) {
    const url = safeUrl(artifacts[key]);
    if (!url) continue;
    const link = element("a", "", label); link.href = url;
    if (["svg", "overlay"].includes(key)) { link.target = "_blank"; link.rel = "noopener"; } else link.download = "";
    $("artifact-links").append(link);
  }
  const marker = document.querySelector(".valid-mark");
  marker.textContent = isAutomatic(job) ? job?.automatic_completion ? "✓ 自动产物已生成" : "候选产物" : "旧模板产物";
  const validation = job?.validation;
  $("validation-panel").hidden = !validation;
  if (validation) {
    const title = element("h3", "", validation.passed ? "产物结构校验通过" : "产物结构校验未通过");
    const metrics = isAutomatic(job) ? [["实体数", job.geometry?.entities?.length ?? validation.entity_count], ["毫米比例", validation.scaled_mm ? "已估计" : "未确定"], ["完整尺寸约束", validation.dimensions_verified ? "已验证" : "未验证"], ["工程认证", validation.engineering_certified ? "已认证" : "未认证"]] : [["实体数", job.geometry?.entities?.length], ["最大连接间隙 / mm", validation.max_gap_mm], ["最大切向误差 / °", validation.max_tangent_error_deg]];
    $("validation-panel").replaceChildren(title, ...metrics.map(([label, value]) => {
      const row = element("div", "validation-row");
      const text = typeof value === "string" ? value : Number.isFinite(value) ? Math.abs(value) < .001 && value !== 0 ? value.toExponential(2) : Number(value.toFixed(6)).toString() : "—";
      row.append(element("span", "", label), element("span", "", text)); return row;
    }));
  }
}

function renderJob() {
  renderHeading(); renderViewer(); renderNotice(); renderWorkflowEvidence(); renderParameters(); renderEvents(); renderArtifacts();
}
function adoptJob(job, { reset = false } = {}) {
  const changed = state.job?.id !== job.id;
  if (changed || reset) {
    resetEditors(); state.zoom = 1; state.view = "source";
    state.viewZoom = { source: 1, vector: 1, overlay: 1, segmentation: 1, contour: 1 }; state.segmentationMask = false;
    $("confirm-bindings").checked = Boolean(job.manual_confirmation);
  }
  state.job = job;
  if (state.pilotJobs.get(job.case_id)?.id === job.id) state.pilotJobs.set(job.case_id, job);
  if (job.case_id !== "upload") state.selected = state.catalog.cases.find(c => c.id === job.case_id) || state.selected;
  else state.selected = null;
  renderCatalog(); renderJob();
  clearTimeout(state.poll);
  if (busyStates.has(job.status)) schedulePoll();
}
function schedulePoll() {
  const epoch = state.epoch, id = state.job?.id;
  if (!id) return;
  state.poll = setTimeout(async () => {
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
      if (state.epoch !== epoch || state.job?.id !== id) return;
      state.pollErrors = 0;
      const before = state.job.status, hadArtifact = hasArtifact(), hadSegmentation = Boolean(state.job.artifacts?.segmentation_overlay || state.job.artifacts?.segmentation_mask); adoptJob(job);
      if (!hadArtifact && hasArtifact(job) && state.view === "source") { state.view = workflow.artifactForView(job, "overlay") ? "overlay" : "vector"; state.zoom = 1; renderViewer(); }
      else if (!hadSegmentation && (job.artifacts?.segmentation_overlay || job.artifacts?.segmentation_mask) && state.view === "source") { switchView("segmentation"); }
      if (job.status === "completed" && before !== "completed") {
        toast(isAutomatic(job) ? job.automatic_completion ? "自动产物已生成；图像一致性和尺寸验证状态已分别记录。" : "本次产物校验未通过，请查看识别证据。" : "旧模板重建已完成。");
      }
    } catch (error) {
      if (state.epoch !== epoch) return;
      state.pollErrors += 1;
      if (state.pollErrors === 1) toast("暂时无法刷新任务。当前编辑内容已保留，将继续重试。", true);
      if (state.pollErrors < 5) schedulePoll();
      else toast("连续刷新失败；请检查本地服务，恢复后从任务记录重新打开。", true);
    }
  }, state.pollErrors ? 5000 : 1400);
}

async function startJob() {
  if (!state.selected || state.submitting || state.pilotSubmitting || isBusy()) return;
  const caseId = state.selected.id, epoch = ++state.epoch;
  state.submitting = true; renderHeading();
  try {
    const job = await post("/api/jobs", { case_id: caseId, mode: "autonomous_image", use_api: $("use-api").checked, use_segmentation: $("use-segmentation").checked });
    if (state.epoch === epoch) adoptJob(job);
  } catch (error) { toast(error.message, true); }
  finally { state.submitting = false; renderHeading(); updateReadiness(); }
}

async function confirmSolve(event) {
  event.preventDefault();
  if (!state.job || isAutomatic() || state.submitting) return;
  if (!$("parameter-form").reportValidity()) return;
  const parameters = {};
  for (const row of state.job.parameters) {
    const input = state.parameterNodes.get(row.id).input;
    if (input.value === "" || !Number.isFinite(Number(input.value))) { input.focus(); toast(`请补齐 ${row.label}。`, true); return; }
    parameters[row.id] = Number(input.value);
  }
  const assumptions = [...state.assumptionNodes.entries()].filter(([, input]) => input.checked).map(([id]) => id);
  const id = state.job.id, epoch = state.epoch;
  state.submitting = true; updateReadiness();
  try {
    const job = await post(`/api/jobs/${encodeURIComponent(id)}/confirm`, { parameters, confirmed_assumptions: assumptions, confirm_bindings: $("confirm-bindings").checked });
    if (state.epoch === epoch && state.job?.id === id) { state.touched.clear(); adoptJob(job); }
  } catch (error) { toast(error.message, true); }
  finally { state.submitting = false; renderHeading(); updateReadiness(); }
}

async function loadHistory() {
  $("history-dialog").showModal();
  $("history-list").replaceChildren(element("p", "muted", "正在读取已保存任务…"));
  try {
    const { jobs } = await api("/api/jobs");
    if (!jobs.length) { $("history-list").replaceChildren(element("p", "muted empty-small", "尚无已保存任务。选择图纸并开始提取即可创建。")); return; }
    $("history-list").replaceChildren(...jobs.map(job => {
      const row = element("div", "history-item"), description = element("div"), right = element("div", "history-right");
      description.append(element("strong", "", `${shortCase(job.case_id)} · ${isAutomatic(job) ? "自动识别" : "旧模板任务"}`), element("small", "", `${readableTime(job.updated_at || job.created_at, true)} · ${job.id.slice(0, 10)}`));
      const badge = element("span"); setStatus(badge, job.status);
      if (!isAutomatic(job) && job.status === "completed") badge.textContent = "旧模板结果";
      const button = element("button", "button secondary", "打开任务 →");
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          const complete = await api(`/api/jobs/${encodeURIComponent(job.id)}`);
          state.epoch += 1; adoptJob(complete, { reset: true }); $("history-dialog").close();
          if (complete.artifacts?.svg) { state.view = "vector"; renderViewer(); }
        } catch (error) { toast(error.message, true); } finally { button.disabled = false; }
      });
      right.append(badge, button); row.append(description, right); return row;
    }));
  } catch (error) { $("history-list").replaceChildren(element("p", "inline-error", error.message)); }
}

function percentage(value) { return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "未评分"; }
function verdict(value, empty = "未测试") { return value === true ? "通过" : value === false ? "未通过" : empty; }
function automaticReport(report) {
  const card = element("article", "report-card"), header = element("div", "report-title-row");
  const summary = report.summary?.overall || report.summary || {};
  const rows = report.cases || [];
  const called = rows.filter(row => row.provider?.network_requests > 0);
  const calls = report.online_request_summary?.logical_calls;
  const callCount = calls?.total ?? called.length;
  const http = calls?.http_successful ?? called.filter(row => row.provider.http_success).length;
  const schema = calls?.schema_successful ?? called.filter(row => row.provider.schema_success).length;
  header.append(element("h3", "", "自动主轮廓评估"), element("span", "status-badge", report.online_requested ? "ONLINE" : "LOCAL"));
  card.append(header, element("p", "report-meta", `${readableTime(report.created_at, true)} · ${report.model || "自动图像流程"} · ${report.status === "running" ? "评估进行中，重新打开查看进度" : "评估已完成"}`));
  card.append(element("p", "report-filename mono", report.name || report.run_id || "评估报告"));
  const metrics = element("div", "report-metrics");
  [["自动产物 / 全数据集", `${summary.auto_generated ?? "—"} / ${summary.total ?? "—"}`],
   ["参考误差 ≤ 0.1 mm / 已比较", `${summary.reference_within_0_1mm ?? "—"} / ${summary.reference_compared ?? "—"}`],
   ["在线 HTTP · 有效结构 / 全部调用", callCount ? `${http} · ${schema} / ${callCount}` : "未执行"]].forEach(([label, value]) => {
    const metric = element("div", "report-metric"); metric.append(element("span", "", label), element("strong", "", value)); metrics.append(metric);
  });
  card.append(metrics);
  card.append(element("p", "", `已运行 ${summary.attempted ?? "—"} / ${summary.total ?? "—"}；结构有效 ${summary.geometry_valid ?? "—"}；毫米定标 ${summary.scaled_mm ?? "—"}；人工干预 ${summary.manual_interventions ?? "—"}。`));
  const qualification = report.qualification || {};
  const strict = element("div", "report-verdict");
  strict.append(element("span", "", "当前请求范围的严格验证"), element("span", qualification.strict_requested_scope_passed === true ? "verdict-pass" : "verdict-fail", verdict(qualification.strict_requested_scope_passed, "未验证")));
  card.append(strict);
  if (rows.length) {
    const table = element("table", "result-table");
    table.setAttribute("aria-label", "自动任务逐图评估");
    const head = element("tr"); ["图纸", "自动产物", "单位", "图像复核", "参考 ≤ 0.1 mm", "产物检查"].forEach(label => head.append(element("th", "", label)));
    const thead = element("thead"); thead.append(head); table.append(thead);
    const body = element("tbody");
    rows.forEach(item => {
      const row = element("tr"), comparison = item.comparison || {}, provider = item.provider || {};
      const visual = provider.schema_success ? { match: "一致", mismatch: "不一致", uncertain: "不确定" }[provider.verdict] || "未确定" : provider.network_requests ? "未取得" : "未执行";
      [shortCase(item.case_id), item.attempted === false ? "未运行" : item.artifact_completed ? "已生成" : "未完成", item.artifact_completed ? item.scaled_mm ? "mm" : "像素" : "—", visual, comparison.reference_compared ? verdict(comparison.reference_within_0_1mm, "未比较") : "未比较"].forEach(value => row.append(element("td", "", value)));
      const downloads = element("td", "report-artifacts");
      if (report.name && item.case_id && item.attempted !== false) {
        [["overlay.png", "原图叠加"], ["drawing.dxf", "DXF"]].forEach(([filename, label]) => {
          if (!item.artifacts?.[filename]) return;
          const link = element("a", "", label);
          link.href = `/api/evaluations/${encodeURIComponent(report.name)}/cases/${encodeURIComponent(item.case_id)}/artifacts/${encodeURIComponent(filename)}`;
          link.target = "_blank"; link.rel = "noopener";
          if (filename === "drawing.dxf") link.download = `${item.case_id}.dxf`;
          downloads.append(link);
        });
      }
      if (!downloads.childElementCount) downloads.textContent = "—";
      row.append(downloads);
      body.append(row);
    });
    table.append(body); const scroll = element("div", "result-table-scroll"); scroll.append(table); card.append(scroll);
  }
  card.append(element("p", "report-warning", "自动产物已生成、视觉一致、尺寸正确是不同结论；未运行与未验证样本保留在全数据集分母。"));
  const details = element("details");
  details.append(element("summary", "", "查看完整评估数据与口径"), element("pre", "", JSON.stringify(report, null, 2))); card.append(details);
  return card;
}
async function loadEvaluations() {
  $("evaluation-dialog").showModal();
  const c = state.catalog.counts;
  const stats = [[String(c.total ?? "—"), "全部样本 / 分母"], [`${c.paired ?? c.total ?? "—"} / ${c.total ?? "—"}`, "可自动运行范围"], ["分别验证", "图像、尺寸与参考准确率"], [String(c.missing_gt ?? "—"), "缺少参考包"]];
  $("evaluation-overview").replaceChildren(...stats.map(([value, label]) => { const stat = element("div", "evaluation-stat"); stat.append(element("strong", "", value), element("span", "", label)); return stat; }));
  $("evaluation-reports").replaceChildren(element("p", "muted", "正在读取实际评估报告…"));
  try {
    const { reports } = await api("/api/evaluations");
    if (!reports.length) {
      $("evaluation-reports").replaceChildren(element("p", "dialog-intro", "暂无自动评估报告。全数据集均可运行；自动出图率与参考准确率尚未取得完整测量。")); return;
    }
    reports.sort((a, b) => Number(b.mode === "autonomous_image") - Number(a.mode === "autonomous_image") || String(b.created_at || b.name).localeCompare(String(a.created_at || a.name)));
    $("evaluation-reports").replaceChildren(...reports.map((report, index) => {
      if (report.mode === "autonomous_image") return automaticReport(report);
      const card = element("article", "report-card");
      const header = element("div", "report-title-row");
      const reportLabel = report.aggregation_only ? "旧流程统计补充（无新增调用）" : `旧模板评估 · ${report.online_requested ? "在线记录" : "离线记录"}`;
      header.append(element("h3", "", reportLabel));
      const onlineBadge = element("span", "status-badge", report.aggregation_only ? "AUDIT" : report.online_requested ? "ONLINE" : "OFFLINE");
      header.append(onlineBadge); card.append(header);
      card.append(element("p", "report-meta", `${readableTime(report.created_at, true)} · ${report.model || "模型未记录"} · ${report.repeats ?? "—"} 轮`));
      card.append(element("p", "report-filename mono", report.name || report.run_id || "评估报告"));
      if (report.aggregation_only) {
        const original = typeof report.derived_from === "string" ? report.derived_from : report.derived_from?.report;
        card.append(element("p", "", `沿用原报告：${original || "未记录"}；仅补充统计口径，没有重新运行在线测试。`));
      }
      const qualification = report.qualification || {};
      const verdicts = element("div", "report-verdicts");
      [
        ["在线辅助范围验收", qualification.online_assisted_scope_passed, report.online_requested ? "未测试" : "离线未测试"],
        ["参数化几何引擎", qualification.geometry_engine_passed, "未测试"],
        ["全数据集全自动", qualification.dataset_fully_automatic_passed, "未验证"],
        ["保留集重建泛化", qualification.held_out_reconstruction_validated, "未验证"],
      ].forEach(([label, value, empty]) => {
        const row = element("div", "report-verdict"), result = element("span", value === true ? "verdict-pass" : value === false ? "verdict-fail" : "muted", verdict(value, empty));
        row.append(element("span", "", label), result); verdicts.append(row);
      });
      card.append(verdicts);
      const summary = report.summary || report.metrics || report;
      const overall = summary.overall || {};
      const allCalls = report.online_request_summary?.logical_calls;
      const attempts = report.online_request_summary?.network_attempts;
      const metrics = element("div", "report-metrics");
      [[allCalls ? "全部调用 HTTP 成功率" : "样本末轮 HTTP 成功率", allCalls ? allCalls.http_success_rate : overall.online_transport_success_rate], ["参数准确率 / 已评分字段", overall.parameter_accuracy], ["全自动成功率 / 全集", overall.automatic_success_rate]].forEach(([label, value]) => {
        const metric = element("div", "report-metric"); metric.append(element("span", "", label), element("strong", "", percentage(value))); metrics.append(metric);
      });
      card.append(metrics);
      if (allCalls) {
        card.append(element("p", "", `逻辑调用 HTTP 成功 ${allCalls.http_successful ?? "未知"} / ${allCalls.total ?? "未知"}；实际网络尝试 ${attempts?.total ?? "未知"} 次（包含重试）。HTTP 成功率按逻辑调用统计。`));
      }
      const trials = report.protocol_trials || [];
      if (report.online_requested && trials.length) {
        const passed = trials.filter(trial => trial.passed === true).length;
        const elapsed = trials.map(trial => trial.elapsed_seconds).filter(Number.isFinite);
        const timings = elapsed.length ? `；单轮耗时 ${elapsed.map(value => Number(value).toFixed(1) + "s").join(" / ")}` : "；未记录耗时";
        card.append(element("p", "", `尺寸协议实测：${passed} / ${trials.length} 轮通过${timings}`));
      }
      const coverage = report.coverage;
      if (coverage) card.append(element("p", "", `模板覆盖 ${coverage.supported ?? "—"}/${coverage.total ?? "—"} · 全自动完成 ${coverage.automatic_completed ?? "—"} · 辅助校准完成 ${coverage.assisted_calibration_completed ?? "—"}`));
      if (summary.holdout) card.append(element("p", "", `保留集：${summary.holdout.total ?? "—"} 张 · 全自动成功 ${summary.holdout.automatic_success ?? "—"} · 未支持 ${summary.holdout.unsupported ?? "—"}`));
      card.append(element("p", "report-warning", "校准通过不代表保留集泛化；人工确认与简化轮廓不能计为全自动工程成功。"));
      const details = element("details"), pre = element("pre", "", JSON.stringify(report, null, 2));
      details.append(element("summary", "", "查看完整评估数据与口径"), pre); card.append(details); return card;
    }));
  } catch (error) { $("evaluation-reports").replaceChildren(element("p", "inline-error", error.message)); }
}

async function upload(event) {
  event.preventDefault();
  if (!$("upload-form").reportValidity()) return;
  const image = $("upload-image").files[0], ocr = $("upload-ocr").files[0];
  $("upload-error").hidden = true;
  if (image.size > 20000000 || ocr.size > 5000000) {
    $("upload-error").textContent = "原图最大 20 MB，OCR JSON 最大 5 MB。"; $("upload-error").hidden = false; return;
  }
  const data = new FormData(); data.append("image", image); data.append("ocr", ocr);
  data.append("use_segmentation", String($("upload-segmentation").checked));
  data.append("mode", "autonomous_image"); data.append("use_api", String($("upload-api").checked));
  $("upload-submit").disabled = true; $("upload-submit").textContent = "正在上传…";
  try {
    const job = await api("/api/uploads", { method: "POST", body: data });
    state.epoch += 1; adoptJob(job); $("upload-dialog").close(); toast("已创建自动绘制任务，识别证据和产物会陆续显示。");
  } catch (error) {
    $("upload-error").textContent = error.message; $("upload-error").hidden = false;
  } finally { $("upload-submit").disabled = false; $("upload-submit").textContent = "上传并自动绘制 →"; }
}

$("case-search").addEventListener("input", renderCatalog);
$("filter-all").addEventListener("click", () => { state.filter = "all"; renderCatalog(); });
$("filter-supported").addEventListener("click", () => { state.filter = "supported"; renderCatalog(); });
$("start-job").addEventListener("click", startJob);
$("run-pilots").addEventListener("click", runPilots);
$("use-segmentation").addEventListener("change", renderHeading);
$("parameter-form").addEventListener("submit", confirmSolve);
$("confirm-bindings").addEventListener("change", updateReadiness);
$("fill-suggestions").addEventListener("click", () => {
  let filled = 0;
  for (const row of state.job?.parameters || []) {
    const input = state.parameterNodes.get(row.id)?.input;
    if (input && input.value === "" && Number.isFinite(row.suggested_value)) {
      input.value = row.suggested_value; input.classList.remove("missing"); state.touched.add(row.id); filled += 1;
    }
  }
  updateReadiness(); toast(filled ? `已填入 ${filled} 个校准建议值。请核对原图与来源，再逐项确认假设。` : "没有空缺参数；已有编辑值保持不变。");
});
$("cancel-job").addEventListener("click", async () => {
  if (!state.job) return;
  const id = state.job.id;
  try { const job = await post(`/api/jobs/${encodeURIComponent(id)}/cancel`, {}); if (state.job?.id === id) adoptJob(job); }
  catch (error) { toast(error.message, true); }
});
$("view-source").addEventListener("click", () => switchView("source"));
$("view-vector").addEventListener("click", () => switchView("vector"));
$("view-overlay").addEventListener("click", () => switchView("overlay"));
$("view-segmentation").addEventListener("click", () => switchView("segmentation"));
$("view-contour").addEventListener("click", () => switchView("contour"));
$("view-topology").addEventListener("click", () => switchView("topology"));
$("segmentation-overlay-option").addEventListener("click", () => { state.segmentationMask = false; renderViewer(); });
$("segmentation-mask-option").addEventListener("click", () => { state.segmentationMask = true; renderViewer(); });
$("zoom-in").addEventListener("click", () => { state.zoom = Math.min(4, state.zoom + .25); renderViewer(); });
$("zoom-out").addEventListener("click", () => { state.zoom = Math.max(.5, state.zoom - .25); renderViewer(); });
$("zoom-fit").addEventListener("click", () => { state.zoom = 1; renderViewer(); });
$("upload-open").addEventListener("click", () => { $("upload-api").checked = $("use-api").checked; $("upload-segmentation").checked = $("use-segmentation").checked; $("upload-error").hidden = true; $("upload-dialog").showModal(); });
$("upload-form").addEventListener("submit", upload);
$("history-open").addEventListener("click", loadHistory);
$("evaluation-open").addEventListener("click", loadEvaluations);
document.querySelectorAll("[data-close]").forEach(button => button.addEventListener("click", () => $(button.dataset.close).close()));
$("drawing-image").addEventListener("error", () => { toast("图像预览加载失败，请检查本地服务或重新打开任务。", true); });
$("drawing-image").addEventListener("load", fitViewer);
new ResizeObserver(() => requestAnimationFrame(fitViewer)).observe($("drawing-viewport"));
workflow.views.forEach(view => $(`view-${view}`).addEventListener("keydown", event => {
  if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  const tabs = workflow.views.map(name => $(`view-${name}`)).filter(node => !node.hidden);
  const direction = event.key === "ArrowRight" ? 1 : -1;
  const next = tabs[(tabs.indexOf(event.currentTarget) + direction + tabs.length) % tabs.length];
  event.preventDefault(); next.focus(); next.click();
}));

async function initialize() {
  try {
    const [catalog, config] = await Promise.all([api("/api/catalog"), api("/api/config")]);
    state.catalog = catalog; state.config = config;
    $("provider-dot").classList.toggle("ready", Boolean(config.provider_configured));
    $("provider-label").textContent = config.provider_configured ? `${config.model} · 已配置` : "API 未配置 · 可本地运行";
    $("provider-label").title = config.provider_configured ? "服务端已配置密钥；是否连通以真实请求为准" : "在服务端配置 USTC_API_KEY 后可进行在线请求";
    $("use-api").checked = Boolean(config.provider_configured);
    $("use-segmentation").disabled = !config.segmentation_available;
    $("upload-segmentation").disabled = !config.segmentation_available;
    $("use-segmentation").checked = Boolean(config.segmentation_available);
    $("upload-segmentation").checked = Boolean(config.segmentation_available);
    $("upload-open").disabled = false;
    selectCase(catalog.cases[0] || null);
  } catch (error) {
    $("connection-error").textContent = `无法读取工作台数据：${error.message}。确认本地服务启动后刷新页面。`;
    $("connection-error").hidden = false;
    $("provider-label").textContent = "本地服务不可用";
    $("drawing-title").textContent = "连接未完成";
    $("case-list").replaceChildren(element("p", "empty-small muted", "数据集尚未载入。"));
  }
}
initialize();

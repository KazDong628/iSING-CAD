/* Presentation facts from this job only. Training/evaluation images are never viewer inputs. */
(function (root) {
  "use strict";
  const pilotIds = ["CL60-main", "solid-arrow-ping__img_000182-main", "solid-arrow-ping__img_000202", "solid-arrow-ping__img_000231-main"];
  const views = ["source", "segmentation", "contour", "topology", "overlay", "vector"];
  function artifactForView(job, view, mask = false) {
    const artifacts = job?.artifacts || {};
    return ({ segmentation: mask ? artifacts.segmentation_mask : artifacts.segmentation_overlay,
      contour: artifacts.contour_overlay, topology: artifacts.topology_overlay, overlay: artifacts.cad_overlay || artifacts.overlay,
      vector: artifacts.svg })[view] || null;
  }
  function stageEvidence(job) {
    const a = job?.artifacts || {}, terminal = ["failed", "cancelled"].includes(job?.status);
    const event = job?.events?.at(-1)?.stage || "";
    const row = (key, title, ready, active, detail, skipped = false) => ({ key, title, detail,
      status: skipped ? "skipped" : ready ? "ready" : terminal ? "stopped" : active ? "running" : "pending" });
    const segmented = Boolean(a.segmentation_overlay || a.segmentation_mask);
    const contour = Boolean(a.contour_overlay || a.dxf || a.svg);
    const cad = Boolean(a.dxf);
    const rows = [
      row("segmentation", "分割预测", segmented, job?.use_segmentation && job?.status === "extracting", segmented ? "模型预测区域已保存，可切换叠加图与二值 mask。" : job?.use_segmentation ? "只读取当前原图；等待模型产物。" : "当前使用图像算法提取边界。", !job?.use_segmentation && Boolean(job)),
      row("contour", "主轮廓提取", contour, /measure|contour|vectorize/.test(event), contour ? "边界已提取；与原图比较，不代表尺寸验证。" : "从预测区域提取闭合主边界。"),
      row("cad", "CAD 构建", cad, job?.status === "solving", cad ? job?.validation?.passed === false ? "候选 DXF 已保存，结构检查未通过。" : `LINE / ARC 已导出；${job?.validation?.scaled_mm ? "毫米比例已估计" : "比例未确定，使用像素单位"}。` : "拟合直线与圆弧，再导出可编辑 DXF。"),
    ];
    if (cad && job?.validation?.passed === false) rows[2].status = "warning";
    if (job?.parameterization) {
      const p = job.parameterization;
      rows.splice(2, 0, row("topology", "图元连接与参数求解", Boolean(a.topology_overlay), p.status === "running",
        p.accepted ? `已采用部分约束解；${p.topology?.entity_count ?? "—"} 个图元，未绑定尺寸仍待验证。` : p.status === "running" ? "从原图标注关联尺寸，求解并检查约束残差。" : p.topology_exported ? "保留原图笔画支持的校正轮廓；数值约束尚未通过。" : "保留初始 CAD，可查看拓扑候选和未通过原因。"));
    }
    return rows;
  }
  function dimensionProviderText(provider = {}) {
    if (provider.status === "disabled") return "未启用 / 未调用";
    if (provider.status === "skipped") return "未调用 / 未调用";
    if (["pending", "running"].includes(provider.status)) return "调用中 / 等待返回";
    if (provider.status === "interrupted") return "调用中断 / 结果未知";
    if (provider.network_requests === 0) return "未发起 / 未调用";
    const http = provider.http_success === true ? "成功" : provider.http_success === false ? "未成功" : "未知";
    return `${http} / ${provider.schema_success ? "有效" : "未取得"}`;
  }
  const api = { pilotIds, views, artifactForView, stageEvidence, dimensionProviderText };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ContourWorkflow = api;
})(typeof window !== "undefined" ? window : globalThis);

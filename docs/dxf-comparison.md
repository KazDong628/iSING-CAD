# DXF 独立实体与参数审计

`contour_agent.dxf_comparison.audit_dxf(path)` 返回原始 modelspace 对象与主轮廓过滤结果，`compare_dxf_entities(prediction, reference)` 只读取已完成的两份 DXF。该模块不供分割、比例求解、矢量化或 provider 调用，GT 仅用于本模块的评分和报告图。

原始对象按文件原单位记录 LINE/ARC/CIRCLE 参数、polyline 顶点、图层、句柄；过滤后的对象展开 polyline 并复用 `evaluation._curves` 的图层规则。已声明单位的过滤后长度/圆心/半径统一换算为 mm；INSUNITS=0 始终标为未知原绘图单位。原始对象数和过滤后 primitive 数分列，不能混为实体准确率。

每个 primitive 有本文件内的 ID、类型、图层、首尾点、长度、圆心/半径/弧角，以及端点连接节点。ID 仅用于查表，预测 p0001 不自动对应参考 r0001。节点记录连接类型、端点距离、内部切向夹角和相对于平滑延续的偏差。偏差非零可能是设计角点，报告不自动判为应相切。闭合关系只依据端点容差，不能代替自交或工程意图检查。

物理评分调用现有 `evaluate_autonomous_artifact`，保持 0.1 mm 阈值、D4 方向搜索与平移、禁止拟合尺度。参数对应只接受双向唯一、类型相同且完整几何参数符合 0.1 mm 局部门限的候选；不按索引匹配，不用优化分配强迫一一对应。ARC 采用有向 CCW 起止端点，互补半圆不会因相同圆心/半径/无向端点而误判一致。其他实体只列最近几何候选、可能碎片、歧义或未匹配状态，不计算其“已匹配参数误差”。

逐曲线误差定位沿用完整评分的采样间距，报告每条参考/预测曲线的最大、P95、RMS、最坏点与最近目标实体。参考 core 和命名 assumption/closure 图层的统计是**参考到预测的单向覆盖误差**，不能惩罚多余预测几何，也不能替代完整双向评分。弧半径/弧角分布及小于 1°/5° 的弧数仅为分段诊断。

## 试验报告工具

```powershell
python tools/compare_pilot_dxf.py
python tools/compare_pilot_dxf.py --baseline runtime/segmentation/pilot/旧run_id/pilot_summary.json --candidate runtime/segmentation/pilot/新run_id/pilot_summary.json
```

默认使用 pilot/latest.json 指定的完整报告及原文件哈希。工具不运行分割、绘图模型、比例算法或网络请求。产物在 `runtime/dxf-comparison/<candidate或baseline_run_id>/`：

- `comparison.json`：两侧 raw/filtered 全参数、连接、对应、原 0.1 mm 评分、逐曲线误差和前后差值。
- `comparison.csv`：每张计数、单位和完整物理误差摘要。
- `index.html`：嵌入 PNG，展开可见原始参数、连接和 core/closure 诊断。
- `cases/<baseline或candidate>/<case_id>/`：叠加 PNG、参数 CSV、连接 CSV、对应 CSV。
- `comparison-artifacts.zip`：上述可分享产物；不包括 GT DXF、模型、数据库或凭据。

ZIP 校验通过后原子更新 `runtime/dxf-comparison/latest.json`。HTTP 挂载建议为 `/dxf-comparison` 与 `/api/dxf-comparison/{run_id}/{artifact_name}`，后者只允许四个顶层报告文件。HTML 图像已内嵌，可离线浏览。

若预测或参考单位未定，毫米比较不执行。可视化分别按各自包围盒最长边归一化并显著标注，这只是看形状，不能解释为真实比例、毫米参数精度或改善。参考未闭合时在正文标明连通分量与未连接端点的最近邻距离及其真实单位，不补连参考。182 当前 GT 的单位为 0，数值约 45.65 的最近端点距离不得表述成 45.65 mm。

前后比较只有相同 GT 字节且两轮都有有效物理指标时才计算 max/P95/RMS 差值。`max_error_improved` 仅表示最大误差下降，绝不意味着所有指标改善；对象减少百分比单独报告，少画实体也不能证明精度提高。原始/过滤对象数均显示旧、新、GT 三列。全 50 张分母保留，这四张仍为按已知效果选择的已暴露开发子集。

验证：`python -m pytest -q tests/test_dxf_comparison.py tests/test_compare_pilot_dxf.py`。全部使用合成 DXF/临时 pilot 数据，无模型推理与 API。

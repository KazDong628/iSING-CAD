# 四图分割到 CAD 开发试验

入口为 `tools/run_segmentation_pilot.py`。固定选择 `CL60-main`、`solid-arrow-ping__img_000182-main`、`solid-arrow-ping__img_000202`、`solid-arrow-ping__img_000231-main`。这是按已知分割效果选择的开发子集，四图及其所属 50 图均已被开发查看，不能作为全新盲测或全数据集精度证据。

## 运行接口

```python
from tools.run_segmentation_pilot import run_segmentation_pilot
summary, path = run_segmentation_pilot(settings, online=False)
```

```powershell
python tools/run_segmentation_pilot.py --checkpoint runtime/segmentation/runs/unet-r18-dxf-v2/best.pt --manifest runtime/segmentation/gt-data-v3/manifest.json
```

参数化改造说明见 [source-parametric-pipeline.md](source-parametric-pipeline.md)。当前分割流程额外包含原图笔画校正、图元连接图、独立标注绑定API与数值约束求解；下文旧版本的两类调用描述属于改造前记录。

以上默认仅进行本地生成。只有显式添加 `--online` 才允许在线尺寸解析和视觉复核。应以真实存在的训练 manifest 路径为准；manifest 仅用于训练暴露声明，不作为生成轮廓的输入。凭据只来自进程环境或忽略的本地配置，不写入命令参数、报告或浏览器。

每次调用复用 `qualify_autonomous(settings, online=..., repeats=1, cases=固定四图, use_segmentation=True)`，创建新的隔离运行目录与四个新任务；不能将旧报告当作本轮测试。每图顺序是原图推理、局部主轮廓与比例求解、可选在线尺寸解析、可选视觉复核、完成后独立 GT 评分。GT 坐标不进入生成器或 API。

## 结果与范围

输出到 `runtime/segmentation/pilot/<run_id>/`：

- `pilot_summary.json`：每张新 job、原图 SHA256、实际 checkpoint SHA256、掩膜 SHA256、训练暴露来源、DXF 结构、单位、比例/半径证据、完整独立参考误差及两类 provider 回执。
- `pilot_summary.csv`：方便比对的关键字段。
- `index.html`：自包含 PNG 预览与安全 HTTP 下载链接。分割掩膜、主外边界与 CAD 拟合叠加分别呈现。
- `cases/<case_id>/`：固定白名单产物，包括 DXF、SVG、JSON 证据、分割掩膜、原图叠加和拟合证据。缺失文件保留为缺失，失败样本不从分母移除。
- `pilot-artifacts.zip`：上述报告及每张实际产物；不包括原始 GT、checkpoint、数据库或任意报告路径文件。PNG/DXF 保留原字节，JSON 在去除凭据后重新序列化；原始与包内 SHA256 分开记录。

全数据集分母始终是 50，实际选择/尝试 4，未尝试 46。报告 `status=completed` 仅表示证据包写完；`counts.valid_dxf` 要求独立闭合端点检查与任务运行时几何检查都通过。它不等价于标注尺寸准确。像素/未指定单位产物不能获得毫米准确性结论。

独立参考评分沿用现有冻结协议与 0.1 mm 阈值。默认仅搜索 D4 方向与平移，不拟合尺度；最大误差、P95、RMS 和长度误差保留原值，仍属形状诊断。参考 DXF 的单位为 0 时不能擅自解释成毫米。四图中 182 的当前原始 ZIP 参考没有声明物理单位；应披露不可评分，不能修改参考以获取通过。

视觉复核来自 `provider`，尺寸解析来自 `dimension_analysis.provider`，分别统计逻辑调用、HTTP/schema 成功及实际请求数。四个在线任务可能产生四次尺寸解析加四次视觉复核，不能报告成总计四次调用。缺失请求计数为 `null`，缺失回执不算成功。尺寸 API 当前只对解析结果和本地证据记录一致性，`geometry_updated_by_api=false`、`dimensions_verified=false` 不因 API 成功而改变。

`latest.json` 只在报告、ZIP 写完且 ZIP 完整性检查通过后原子更新。打包中途失败会保留旧指针；单张生成失败则作为失败行包含在新的完成包内。

## HTTP 挂载合同

父服务负责路由，脚本本身不开端口：

1. `/segmentation-pilot`：读取安全验证后的 `runtime/segmentation/pilot/latest.json`，返回对应 `index.html`。
2. `/api/segmentation/pilot/{run_id}/{artifact_name}`：仅允许 `index.html`、`pilot_summary.json`、`pilot_summary.csv`、`pilot-artifacts.zip`，解析后必须仍在配置的 pilot 目录，拒绝重定向路径与非真实文件。
3. 已有核心产物路由 `/api/evaluations/{report_name}/cases/{case_id}/artifacts/{artifact_name}`：HTML 中的 DXF/SVG/检查证据使用该路由；源文件必须与新 qualification 的固定任务目录和声明匹配。

报告 PNG 使用 data URL，因此解压后可离线查看；离线读取单张 DXF 请打开 ZIP 内 `cases/` 目录，页面中的 HTTP 下载链接需要服务。

## 验证

2026-09-21（本地时间）完成真实在线试验 `20260920T160413269947Z-b4e187d3`，四张均从原图重新推理并导出有效 DXF；模型未重训，GT 仅在导出后用于评分。

| 图纸 | CAD 实体数 | 输出单位 | 尺寸 API | 视觉复核 | 与参考最大误差 |
| --- | ---: | --- | --- | --- | --- |
| CL60 | 122 | mm | JSON 有效，16/16 与本地解析一致 | mismatch | 13.069 mm |
| 182 | 68 | pixel | JSON 有效，13/16 与本地解析一致 | match | 不可评分 |
| 202 | 41 | mm | HTTP 成功，返回 JSON 无效 | match | 24.030 mm |
| 231 | 56 | pixel | JSON 有效，16/16 与本地解析一致 | match | 不可评分 |

文本与视觉合计 8 次真实请求，HTTP 8/8 成功；文本 schema 3/4、视觉 schema 4/4，视觉 match 3/4。尺寸解析失败不影响本地出图。四图均未完成全部标注约束验证，0 张通过冻结的 0.1 mm 参考阈值。另有一次关闭 API 的 CL60 工作台浏览器联调，不计入这四图在线试验。

拟合 CAD 与各自选定分割边界的区域 IoU 为 99.21%、99.11%、99.43%、99.01%，四图均无需折线回退。这是转换保真度，不能解释为与 GT 的分割准确率。

```powershell
python -m pytest -q tests/test_segmentation_pilot.py
```

测试使用临时 50 行目录、合成 DXF/PNG、假 checkpoint 字节及伪造的新任务 journal，不做推理、训练或网络调用。覆盖新任务来源绑定、失败/像素单位分母、参考单位缺失、双 provider 计数与未知值、安全路径白名单、秘密去除、报告重用拒绝、checkpoint/原图哈希不一致及 ZIP 失败不更新 latest。

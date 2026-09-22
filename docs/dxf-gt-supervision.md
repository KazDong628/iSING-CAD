# DXF GT 监督：标签生成、训练与在线使用

本流程把用户提供的 DXF GT 配准到原图，生成像素监督，训练 U-Net / ResNet-18。DXF 决定标签边界；原图启发式 mask、OCR 和 GT 包自带的叠图只帮助定位。`__dataset/` 始终只读，所有派生文件写入 `runtime/segmentation/`。

本轮候选使用 `runtime/segmentation/gt-data-v3/`，训练输出为 `runtime/segmentation/runs/unet-r18-dxf-v2/`，配置为 768 像素、60 轮、batch size 2。标签准备已完成，训练已启动；最终训练、开发测试和在线结果以本轮实际报告为准，此处暂不填写尚未完成的数值。旧启发式弱标签实验保留在[历史基线文档](deep-segmentation.md)。

## 参考范围与现有缺口

| DXF 读取结果 | 数量 | 说明 |
| --- | ---: | --- |
| 可生成闭合主区域 | 44 | 包含对 `1-main`、182、188 的既有线网面修复 |
| 几何无效 | 1 | 244 存在真实的 38.413 mm 开口 |
| 未找到对应参考 | 5 | 245、267、280、283、285 |
| 全量分母 | 50 | 缺失、无效与配准失败均保留记录 |

读取清单见 [gt-dxf-inventory.json](../runtime/segmentation/gt-dxf-inventory.json)。244 的[专项诊断](../runtime/segmentation/gt-dxf-244-diagnostic.json)记录了 `GT_MAIN` 上的 9 条 LINE、11 条 ARC，未发现被过滤的闭合实体或 INSERT。未连接端点为 `(34.677231717, 425.310545492)` 与 `(0, 441.835260116)`，距离约 38.413234607 mm；直接 DXF 与 ZIP 中的版本一致。读取器没有擅自补线闭合该缺口。

部分参考只能从 ZIP 读取；读取不会向源目录解压。声明为英寸的几何会先转换为毫米。参考包含多个闭合区域时，当前监督选择最大主外环，同时记录其余候选区域。

## 实际标签生成流程

1. **读取并选主外环。** [dxf_supervision.py](../contour_agent/dxf_supervision.py)查找直接 DXF 与 ZIP 内的参考，按参考用途选择候选，处理单位、实体连接顺序、圆弧和 bulge 曲线离散化。构造线、尺寸和调试几何不作为主轮廓。
2. **只从已有线网恢复闭合面。** 在允许的微小端点误差内吸附端点；严格单路径重建失败时，对已有线段交点做平面分割，选择最大有界面的外环。修复记录 `topology_repaired=true`、面面积、悬挂线、吸附量与 `closure_edges_added=0`。源 DXF 不变；派生的像素监督不等于原始参考已获得工程认证。
3. **全局配准。** [gt_registration.py](../contour_agent/gt_registration.py)综合 GT 包已有的标定变换、OCR 提供的尺度证据、整图边缘匹配和区域定位提示，选择 DXF 到原图的整体变换。它不通过逐点变形让参考追随图像噪声。
4. **利用 GT 包官方叠图定位。** [gt_overlay_hint.py](../contour_agent/gt_overlay_hint.py)只读取所选 GT 包中的完整叠图，排除局部放大、尺寸、调试图及旧预测。SIFT 匹配叠图与原图共享的灰度背景，RANSAC 求叠图到原图的变换；至少 20 个内点、内点比例至少 0.5，并检查重投影误差和空间覆盖。彩色 GT 线形成定位提示，不能直接成为训练目标。无可信叠图时使用源图启发式区域作为提示。
5. **渲染 DXF 标签并筛查。** [gt_dataset.py](../contour_agent/gt_dataset.py)把配准后的 DXF 主外环填充为 0/255 mask，保存到与准备图像一致、最长边不超过 1536 像素的画布。最终边界来自 DXF。当前标签是单个主外环的填充区域，不包含完整的孔洞或所有分离零件。

自动筛查门槛如下，均用于配准质量筛查，不能替代毫米精度检验：

| 字段 | 最小值 |
| --- | ---: |
| `frame_inside_ratio` | 0.99 |
| `edge_support` | 0.55 |
| `hint_precision` | 0.25 |
| `hint_coverage` | 0.25 |
| `ambiguity_margin` | 0.025 |

此外要求变换有限且可逆、mask 非空且不占满整图。边缘支持是在配准工作图的像素尺度下计算，不代表 0.1 mm 几何误差。

## 本轮标签清单与边界含义

[gt-data-v3/manifest.json](../runtime/segmentation/gt-data-v3/manifest.json)保留全部 50 个样本，其中 43 个通过自动筛查，可用于监督；对应 train / val / test 为 **29 / 7 / 7**。其余 7 个为：226 的 `hint_precision` 未通过、244 的 DXF 开口，以及上述 5 个缺失参考。它们不参与监督损失，但仍在全量分母内。

固定划分继承 [data/manifest.json](../runtime/segmentation/data/manifest.json)：49 个去重组、原始 train / val / test 为 **35 / 8 / 7**，293 固定在训练组。完全相同的图、pHash 距离不超过 4 且宽高比差不超过 3% 的近重复图按组划分，不因标签失败重新分配。报告必须同时列出原始划分人数、可训练人数和排除原因。

全部 50 张此前已经被开发流程查看，`split_claim=development_only_not_blind_test`。这套划分可用于开发比较，不能称为未见图纸盲测。

43 张叠图的开发阶段视觉检查未见明显整体定位错配，但这不等于人工逐像素真值验收，清单仍为 `reviewed=false`。尤其需要保留以下范围限制：

- 188 属于开发测试组，其 GT 顶部直线封口包含原图中的非材料区域。
- 269 属于验证组，当前 GT 只定义左侧主区域，右侧区域在本监督任务中属于背景。
- 部分参考踏面采用简化或声明的封口几何；训练目标是所选 **GT 主轮廓区域**，不能等同于整张图中所有材料的精确分割。

因此，DXF 监督比启发式教师提供了更明确的边界来源，但自动配准误差、主外环选择和原始 GT 的简化仍会影响训练与评分。模型与这些 mask 的 IoU 不能直接解释为工程尺寸准确率。

## 冻结清单与可追溯文件

每个版本目录保存：

- `manifest.json`：全部案例的组、划分、状态、`label_source=registered_dxf_gt`、源图/OCR/准备图/mask 哈希、参考来源及配准质量。
- `references/<case_id>.json`：DXF 或 ZIP 成员来源、DXF 内容 SHA256、单位、主外环选择、拓扑修复及失败原因。
- `registration/<case_id>/`：变换矩阵、配准质量和检查图；若使用官方叠图，`localization/` 另存原图/叠图哈希、SIFT 内点与误差、定位变换及提示图。
- `images/`、`masks/`：成对训练文件；失败项可能保留诊断 mask，是否可训练以 `trainable` 为准。
- `progress.json`：逐案例准备进度，失败案例也会记录。

数据版本不能静默覆盖。准备器拒绝非空输出目录，训练读取器核对冻结内容、源图身份和分组信息。更新标签应生成新版本并明确变更；独立的人工审阅记录也需要保留来源，不能把自动通过改写为人工审阅。

## 准备、训练与评估命令

在项目根目录使用 `.venv-seg`。环境版本与安装方式见[历史基线文档的环境章节](deep-segmentation.md#实际环境)及 [requirements-segmentation.txt](../requirements-segmentation.txt)。下列命令描述本轮配置；已存在的 `gt-data-v3` 不应重复生成到同一目录，已开始的训练也不应重复启动。

```powershell
.venv-seg\Scripts\python.exe -m contour_agent.gt_dataset --dataset __dataset --base-manifest runtime/segmentation/data/manifest.json --output runtime/segmentation/gt-data-v3 --workers 3

.venv-seg\Scripts\python.exe -m contour_agent.segmentation train --manifest runtime/segmentation/gt-data-v3/manifest.json --output runtime/segmentation/runs/unet-r18-dxf-v2 --epochs 60 --size 768 --batch-size 2 --device cuda
```

训练完成后，用同一冻结清单评估验证组和开发测试组：

```powershell
.venv-seg\Scripts\python.exe -m contour_agent.segmentation evaluate --manifest runtime/segmentation/gt-data-v3/manifest.json --checkpoint runtime/segmentation/runs/unet-r18-dxf-v2/best.pt --output runtime/segmentation/runs/unet-r18-dxf-v2/gt-v3-val-native --split val

.venv-seg\Scripts\python.exe -m contour_agent.segmentation evaluate --manifest runtime/segmentation/gt-data-v3/manifest.json --checkpoint runtime/segmentation/runs/unet-r18-dxf-v2/best.pt --output runtime/segmentation/runs/unet-r18-dxf-v2/gt-v3-test-native --split test

.venv-seg\Scripts\python.exe -m contour_agent.segmentation predict --checkpoint runtime/segmentation/runs/unet-r18-dxf-v2/best.pt --image __dataset/origin/solid-arrow-ping__img_000293.jpg --output runtime/segmentation/predictions/unet-r18-dxf-v2-293
```

293 的预测仅用于训练案例演示。训练报告记录轮数、最优轮次、标签来源、划分与 checkpoint SHA256；评估报告记录已评分与被排除样本、原图和固定合成干扰图的区域/边界指标。默认核对 checkpoint 所用清单，跨标签版本比较需要显式声明并核对源图与划分身份。标签改变后的 IoU 变化不能单独归因于模型改进；合成干扰分数更高也不证明未见图泛化更好。

本轮完成两次实际 DXF 监督训练。初轮 `unet-r18-dxf-v1` 使用 v2 标签（27 train / 4 val），40 轮、512 px；修正配准后的 `unet-r18-dxf-v2` 使用 v3 标签（29 train / 7 val），60 轮、768 px，训练 400.92 秒，按验证集选择第 27 轮。以下使用相同 v3 标签、原 prepared 图输入、统一 512 评分网格对比：

| 模型 | 验证 IoU（7/8，1张缺GT） | 验证边界 F1 | 开发测试 IoU（7/7） | 开发测试边界 F1 |
|---|---:|---:|---:|---:|
| 旧弱标签模型 | 0.7745 | 0.7127 | 0.5502 | 0.5222 |
| 初轮 DXF 监督 | 0.8503 | 0.8355 | 未用于本轮测试 | 未用于本轮测试 |
| 修正配准 + 768 px | 0.8622 | 0.8640 | 0.6585 | 0.6559 |

最终选择记录为 `runtime/segmentation/model-selection.json`，在本轮测试前写入；`best.pt` SHA256 为 `3e8719ec1ee98f27ac24279106474e1d17be6849db1bf730f7eb133062c96667`。测试所有 7 张均计入，1-main 和 188 的 IoU 分别约 0.114、0.027，仍有严重缺漏或误选；没有根据测试成绩重新训练或挑选轮次。这是开发测试，不是盲测：所有图此前已被开发流程查看，275/276 等相似图形家族跨划分存在。

合成干扰分数仅为条件诊断：额外剖面线使用 GT mask 限定位置，会提供目标区域线索，不能用于证明真实抗干扰或泛化提升。原图指标是主要比较依据。完整逐图对比见工作台 `/segmentation-results`，原始分数在各运行的 `gt-v3-val-native` / `gt-v3-test-native` 目录。

## 在线使用与 GT 隔离

训练完成并检查候选后，可显式配置其 checkpoint 和配套审阅清单，再启动本地服务：

```powershell
$env:CONTOUR_SEGMENTATION_CHECKPOINT = (Resolve-Path runtime/segmentation/runs/unet-r18-dxf-v2/best.pt).Path
$env:CONTOUR_SEGMENTATION_MANIFEST = (Resolve-Path runtime/segmentation/gt-data-v3/manifest.json).Path
.venv-seg\Scripts\python.exe -m contour_agent serve --port 8769
```

配置候选不表示模型已验收，也不会自动提升为合格默认模型。`/segmentation-review` 可用于后续审阅；保存审阅不自动重新训练或发布模型。

已按项目说明通过进程环境配置在线视觉 API 后，可显式提交分割与在线复核任务：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8769/api/jobs -ContentType application/json -Body '{"case_id":"solid-arrow-ping__img_000293","mode":"autonomous_image","use_segmentation":true,"use_api":true}'
```

离线标签准备可以读取用户授权的 GT，训练读取派生 mask。运行时分割只接受待绘制原图，随后从预测轮廓结合源 OCR 定标并拟合 LINE/ARC；不会按案例读取 GT 坐标或训练 mask 来生成输出。独立评估和审阅保留单独的数据用途。

在线视觉 API 只接收原图与当前预测轮廓叠加图，不发送 GT DXF、GT 包叠图或监督 mask。API 单次有界调用、不自动重试；失败仍保留自动生成的 CAD 产物。HTTP 连通、JSON 结构、视觉 verdict、DXF 几何有效性、毫米定标、尺寸约束和独立参考精度分别报告。视觉 `match` 或分割 IoU 不能证明尺寸、公差、相切约束或 0.1 mm 精度。


## 本轮在线复测与导出修复

首轮 293 / CL60 / 281 在线流程中，CL60 因像素中心轨迹自接触而未能导出；另两例实际调用成功，视觉 verdict 分别为 match / mismatch。修复将这类无效轨迹改为原二值 mask 的精确像素单元边界，不改变前景像素、不填缝、不添加连接。CL60 新旧 prediction-mask.png 哈希完全相同，保留 160,518 个前景像素；修复后生成 122 个 CAD 实体，结构检查通过。

再次运行相同三例，出图、HTTP 和 JSON 协议均为 3/3；视觉结果仍为 293 match、CL60 mismatch、281 mismatch。两轮共 5 次真实 API 请求，HTTP/结构 5/5、match 2 次、mismatch 3 次；首轮 CL60 在调用前失败，仍计为系统失败。GT 从未发送给在线服务。所有首轮和复测报告均保留，不以最后一次结果覆盖历史失败。

另外修复了 Windows 并发评估共享时钟精度导致的目录名碰撞，运行 ID 在时间戳后加入随机后缀；该次本地启动失败尚未调用 API。工作台 `/segmentation-results` 同时显示历史与最新结果。


修复后的全量复测：50/50 自动出图且结构校验有效；38/50 毫米定标、12/50 像素轮廓；35 张满足参考比较前提，0 张达到 0.1 mm 门槛（完整分母仍为 50）。首轮为 49/50 出图，唯一导出失败 CL60 已通过相同 mask 的通用边界修复恢复。最后全量代码回归：429 passed、3 skipped（Windows 符号链接能力限制）。这不意味着轮廓精度合格。

最终已启动的工作台使用 `unet-r18-dxf-v2`，默认勾选深度分割。启动脚本 `start-segmentation.ps1` 同时配置配套 v3 标签清单；`/segmentation-review` 读取 DXF 派生标签，`/segmentation-results` 展示可复现报告。重建报告：`python tools/build_segmentation_report.py`。

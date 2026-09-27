# Oracle 掩膜实验的独立 DXF 对比

先使用 `scripts/run_oracle_mask_reconstruction.py` 完成预测并写出 `after/drawing.dxf`，再执行：

```powershell
python scripts/evaluate_oracle_mask_run.py --run-dir runtime/oracle-mask/某次运行
```

预测阶段依次执行初始 CAD、尺寸文字解析、拓扑编辑、约束绑定与求解。加 `--online` 时，尺寸解析使用选定 API 的 `DimensionProvider(max_attempts=1)`，最多提交 16 条原图 OCR 的 ID 和文字；`after/dimension-analysis.json` 保存本地尺寸、模型解释、一致性以及 HTTP/结构回执，`run-manifest.json` 保存该阶段摘要。解析失败仍保留初始 CAD 和本地尺寸，继续后续受控流程。模型解释不会覆盖原始 OCR 或直接修改几何，解析一致也不代表尺寸已经绑定或满足。

Anthropic Messages 兼容接口的规划、绑定和评估阶段采用 8192 tokens 有限输出预算，局部编辑为 12288；尺寸解析仍为 2200，其他协议保持原有预算。多模态回执记录实际 `request_max_output_tokens` 及白名单数值 token 用量，不保存私有推理内容。单请求总时限仍不超过 600 秒，截断结果继续拒绝，不增加隐式重试；HTTP 成功但输出截断会记录为结构失败。

对 H800 可显式追加 `--anthropic-thinking disabled`，请求 Messages 协议的 `thinking: {type: disabled}`。默认仍为 `provider_default`，不会发送这一控制字段；离线或其他协议使用该 CLI 开关会在请求前报错。服务也可通过 `CONTOUR_ANTHROPIC_THINKING_MODE` 配置，允许 `provider_default` 或 `disabled`。清单和回执仅记录 `anthropic_thinking_mode_requested`；服务器可能忽略或映射此开关，因此不能凭 HTTP 200 宣称推理已关闭。协议说明见 [vLLM Anthropic serving](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/anthropic/serving/)。

在本实验中，`oracle_mask_conditioned=true` 明确表示掩膜是给定的完整边界。拓扑初始化保留初始 CAD，不执行为修补普通分割而设计的笔画拉直、文字遮挡绕行或粗化。之后仍可提出标注引导的合并、改型和圆角候选，但发布与求解后都要相对**最初的 raw mask 边界**检查，沿用初始 `curve_fit.total_deviation_budget_px`；中间候选不能重置预算。源图笔画、闭合和几何有效性检查也继续生效。这是对输入掩膜的保形约束，不是读取 GT DXF 坐标，也不代表达到 0.1 mm 参考精度。超限时保存诊断并保留最后一套合格草稿。普通自主分割流程的边界修补策略不受此实验模式影响。

评估器先回读预测 DXF。缺少或无法读取预测时仅写 `evaluation/summary.json`，不打开 GT。完成预测回读后，它在只读数据集目录中查找对应 GT，并核对 GT 字节哈希与 `run-manifest.json` 中供掩膜构建的 GT 哈希。哈希不符时停止比较，不将不同版本 GT 混用。`evaluation/comparison.json` 包含既有 `compare_dxf_entities` 的完整对象、参数和误差审计；`summary.json` 列出核心结果；`overlay.svg` 显示独立比较的重叠图。

报告分开记录原始 modelspace 对象数量、过滤后的 LINE/ARC 图元数量、端点连通、毫米单位、原始坐标系误差、D4 方向搜索加平移后的形状误差、面积和图元参数对应。形状误差来自双向密集采样到精确曲线的距离，报告所称 Hausdorff 是**采样近似及其半步上界**。对图元参数，仅在同类型、完整参数满足 0.1 mm 门限且双方唯一时建立对应；同一 GT 弧被多段近似覆盖不会算作图元匹配。全部图元一一对应时，还对比其端点邻接关系。面积由单个闭合 LINE/ARC/CIRCLE 环的解析 Green 积分计算；多个环、未闭合或无物理单位时不计算面积。

`summary.json.stage_comparison` 还逐一比较已保存的初始 CAD、源图拓扑、参数候选和最终发布文件；缺失阶段明确记为 `not_available`。各阶段使用完全相同的独立门限，图表写入 `evaluation/stages/`，预测 DXF 保持不变。这个对比用于区分误差是在初始拟合、拓扑修改还是参数求解中产生；不会根据 GT 分数回选、替换或重新生成预测。

该实验的输入掩膜取自评分用 GT。即使注册后的形状误差、实体数量都达到门限，也只能说明**在 GT 边界已知的条件下**后续 CAD 重建的效果，不能证明自动分割或独立未见图纸上的泛化。D4 方向和平移在评分时借助 GT 搜索得到；只有 `native_coordinate_frame` 指标才描述输出文件在原坐标系中与 GT 的差异。DXF 闭合不保证无自交，面积相近不保证局部参数相同，GT 的假设或简化图层也应单独查看。

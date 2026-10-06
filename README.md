# iSING-CAD · 自动主轮廓工作台

默认流程为：**原图 + OCR → 材料掩膜与初始 CAD → 源标注解析及局部拓扑编辑 → 约束绑定与数值求解 → LINE/ARC 导出及独立检查**。在线模式由模型提出源证据和编辑方案，本地几何执行器与求解器构造实际 DXF。阶段失败会保留有效草稿，并明确标注未完成的阶段。

严格半径模式将已验证箭头指向的标注 R 作为常数求解，并回读 DXF 原生 ARC 作零容差检查。部分约束成功不等于所有半径已解决，也不等于 GT 精度通过；详见[严格标注半径与来源审计](docs/strict-annotation-radius.md)。GT 派生掩膜只用于明确声明的条件性重建实验。

当前数据集全部 50 张都可提交运行。**可运行范围不等于识别成功率，更不等于尺寸准确率。** 系统分别记录提取结果、有效产物、毫米比例和独立参考误差；本文不宣称 50 张均已通过，实际结果以批量评估报告为准。

最新实测与逐项失败说明见 [自动流程验收记录](docs/acceptance-autonomous-2026-09-20.md)。工作台的评估报告可逐图打开原图叠加并下载 DXF。

已接入 [DXF GT 监督训练](docs/dxf-gt-supervision.md)：从用户提供的 DXF 建立图像配准掩膜，50 张图中 43 张通过标签检查，29 / 7 / 7 张用于训练 / 验证 / 开发测试。运行 `./start-segmentation.ps1` 启动后默认启用深度分割，推理只读取原图与模型权重。网页“查看训练标签”可检查标签，“GT 实测结果”查看新旧模型对比、在线调用与失败明细。历史弱标签基线记录保留在 [旧实验文档](docs/deep-segmentation.md)。

新增四图构建试验：先用已有分割表现较好的 CL60、182、202、231 跑完整流程。工作台提供快捷选择、真实模型分割叠加与二值 mask、提取边界及 CAD 叠加视图；[四图试验报告](http://127.0.0.1:8769/segmentation-pilot)保存新运行的产物及独立评分。这是按已知表现选择的开发子集，不能代表全部 50 张或未见图纸的准确率。详见 [试验说明](docs/segmentation-pilot.md)。

输出是自动轮廓草稿。当前实现了图像边界拟合、全局比例估计和少量兼容半径标注绑定，尚未完成所有尺寸、相切关系、公差与复杂踏面的联合约束求解。成功导出或视觉返回 `match`，均不会将 `dimensions_verified`、`engineering_certified` 改为 `true`。

## 启动本机工作台

需要 Python 3.10 或更高版本。在项目目录运行：

```powershell
python -m pip install -e ".[test]"
python -m contour_agent serve --port 8769
```

浏览器打开 [本机工作台](http://127.0.0.1:8769)，也可运行 `start.ps1`。服务监听本机地址；这是本机浏览器工作台与远程 API 调用，尚未部署为公网服务。

`__dataset/`、训练权重和 `runtime/` 产物不随源码仓库发布。需要深度分割时，请将数据集放入本地 `__dataset/`，按 [DXF GT 监督训练说明](docs/dxf-gt-supervision.md)生成标签并训练，再将 `.env.example` 复制为 `.env.local`，填写 `CONTOUR_SEGMENTATION_CHECKPOINT` 与 `CONTOUR_SEGMENTATION_MANIFEST`。也可以直接运行 `./start-segmentation.ps1`，它会检查默认权重和清单是否存在。密钥只写入本地环境或被忽略的 `.env.local`。

也可以打开 [对话式 Agent 工作台](http://127.0.0.1:8769/agent)。该页面支持拖拽工程图与 OCR、持久化会话上下文，并在已有任务上上传局部截图发起二次拓扑修订。修订阶段只允许在线模型从已经通过本地来源门禁的候选中选择；DXF 与叠加图仍由本地几何内核导出和回读验证。页面展示的是观察摘要、候选决策、证据 ID 和校验轨迹，不保存或展示模型私有思维链。

分割检查阶段可选择“导入 GT DXF → 生成掩膜（仅实验）”。服务端只在本地临时解析上传的 DXF，将唯一闭合材料主轮廓与当前原图配准、栅格化成同尺寸二值掩膜，并在工作台叠加预览后等待确认。原始 DXF 不保存在任务产物中，也不发送给在线模型；生成掩膜的哈希和配准质量会记录在 `oracle-mask-import.json`。自动配准不合格时拒绝将其声明为 GT 掩膜。该模式属于使用 GT 的开发实验，不能计为自动分割或盲测结果；若继续手工涂改，记录为 GT 辅助人工修订。

自动拓扑阶段现采用三段式闭环：局部拓扑编辑 Agent 只能引用已有图元 ID 并提出“合并为直线/圆弧/由本地择优”的操作；几何执行器回到原始分割边界拟合坐标，同时生成单项与互不冲突操作的组合候选；独立评估 Agent 再比较原候选和编辑候选。编辑结果只有在本地闭合、连续、原图笔画支持、边界残差和对象数改进门禁全部通过后才会发布。具体协议与产物见 [局部拓扑编辑闭环](docs/topology-edit-agent-v2.md)。

1. 选择任意图纸，或上传原图与对应 OCR JSON。
2. 按需开启“在线视觉复核”，点击“自动绘制主轮廓”。
3. 查看轮廓、原图叠加图和“自动识别证据”，检查尺寸线绑定及比例结论。
4. 下载 DXF，并分别查看结构有效性、物理比例、视觉复核与独立参考比较结果。

上传目前仍需要**图像与 OCR JSON 配对**，不包含从新上传图像自动生成 OCR 的服务。JSON 使用项目的 `meta`、`records` 格式，文字框坐标必须对应上传原图。

任务和产物保存在 `runtime/`。关闭网页不会取消后台任务；在线复核开始前已导出的产物即可下载。服务重启时，完整且已通过结构验证的自动产物会保留；被中断的视觉请求单独记录为中断，不会自动重发，也不会转换为参数确认任务。

## API 配置

通过进程环境或忽略提交的 `.env` 设置。以下仅为占位示例：

```dotenv
USTC_API_KEY=replace-with-your-local-key
CONTOUR_API_BASE=https://api.llm.ustc.edu.cn/v1
CONTOUR_MODEL=qwen-chat
CONTOUR_API_TIMEOUT=600
CONTOUR_TRUST_ENV=false
```

复杂图纸的规划与约束绑定可能超过 90 秒；单次请求默认及最高总超时为 600 秒，仍可显式配置较短预算。配置变更后重启服务；已有本地配置的较短超时不会被自动覆盖。命令行在线评估也需要其自身进程可读取的密钥，不能仅依赖另一个已启动服务的环境变量。

视觉复核发送带候选边界叠加的原图；半径定位另发送原图及有坐标映射的局部标注图，拓扑编辑发送当前候选和源证据。传输协议由所选模型配置决定，支持 Chat Completions、Responses 和 Anthropic Messages。模型返回受结构校验的解释、源像素提案或编辑操作；它不负责直接生成 CAD 坐标，也不认证尺寸精度。

自动流程先保存本地产物，再进行有界在线阶段，各阶段独立限制输出及耗时。半径定位最多两轮，第二轮聚焦未核实记录；拓扑编辑最多三轮，结构失败只能使用剩余轮次重试，每轮回执和失败都保留。文本解析与源证据一致不等于几何约束满足；API 返回值不会直接覆盖尺寸或生成 CAD 坐标。TLS 证书校验始终开启，默认不继承系统代理；需要代理时可显式设置 `CONTOUR_TRUST_ENV=true`。各阶段的 HTTP、schema、绑定覆盖、几何检查和独立参考精度分别记录。超时、接口错误或视觉不一致会保留已经导出的 DXF，但该草稿不自动升级为尺寸已验证的结果。

密钥仅在服务端读取，不写入网页、任务数据或评估报告；不要将密钥放进 URL。未配置 API 时仍可运行本地自动流程。普通文本问候调用成功，只能证明文本请求可用，不能单独证明视觉识别或轮廓精度。

## 实际自动流程与边界

1. **提取主材料边界。** 启用深度分割时，使用当前 U-Net 从原图预测主体区域；未启用时，使用剖面线与区域证据提取，仅当首个提取器没有返回轮廓时才调用备用提取器。两者均只使用原图及其 OCR，不读取参考轮廓，也不按参考评分挑选结果。首个提取器返回了错误轮廓时，目前不会因此自动切换备用提取器。
2. **用标注定标。** 将直径文字与尺寸线关联，提出整直径跨度和半剖面径向跨度等比例候选，再用独立线性尺寸交叉验证，避免二倍比例歧义。直径证据不足时，尝试至少三条不同名义值、不同物理尺寸线的线性尺寸一致比例；比例跨度不超过3%，有竞争比例簇时拒绝唯一结论。证据不足或候选未能唯一确定时保留像素坐标，DXF 不标为毫米。已定标结果仍依赖无单位工程尺寸按毫米解释、扫描比例一致等假设。
3. **生成 LINE/ARC。** 对像素边界拟合相连的直线与圆弧；只有与局部已拟合圆弧兼容的 R 标注才参与半径绑定。其余标注不是已经满足的几何约束，不保证小槽口、微圆角或踏面细节完整。
4. **导出并回读。** 检查闭合、自交与退化，独立回读 DXF，逐实体比较类型、端点、圆心、半径及圆弧平面方向。结构有效说明产物可以作为相连 CAD 轮廓使用，不能说明选中了正确材料边界或满足全部尺寸。
5. **可选视觉复核。** 在线模型检查叠加边界与原图的视觉一致性，结果单列。`match` 不替代尺寸验证或独立参考评分。

预测阶段不读取 GT 坐标，不使用逐图模板或参考轮廓作为输入。弱剖面线、多区域、尺寸线粘连、局部槽口和复杂踏面等仍可能使提取或拟合失败。比例未确定时输出的是像素草稿，不应按毫米使用。

每个已完成导出的任务包含：

| 文件 | 内容 |
| --- | --- |
| `drawing.dxf` | 原生 LINE/ARC 轮廓；单位以验证报告为准 |
| `preview.svg` | CAD 轮廓预览 |
| `overlay.png` | 拟合轮廓叠加在原图上，便于检查选区和边界 |
| `model.json` | 提取来源、比例、实体、坐标系与适用范围 |
| `validation.json` | 闭合、自交、DXF 回读及独立状态标记 |
| `dimension-evidence.json` | 文字框、尺寸线绑定与定标证据 |
| `segmentation-overlay.png`、`segmentation-mask.png` | 当前模型从原图预测的叠加图与二值 mask；只有启用分割且文件真实生成时提供 |
| `contour-overlay.png` | 拟合前提取的主边界叠加 |
| `curve-fit.json` | CAD 拟合相对分割边界的像素偏差、拓扑检查和回退记录；不是 GT 精度 |
| `dimension-analysis.json` | 本地尺寸清单、已有比例/半径绑定，以及独立 API 解析回执 |

提取中间证据也保存在任务目录中。`automatic_completion` 只表示自动产物通过结构检查；`scaled_mm`、视觉结果和参考误差另行判断。

## 独立评估

```powershell
python -m pytest -q
python -m contour_agent catalog --output runtime/catalog.json
python -m contour_agent evaluate
python -m contour_agent evaluate --online --cases 044-main solid-arrow-ping__img_000293
python -m contour_agent evaluate --online --repeats 2
```

`evaluate` 默认使用 `--engine autonomous`，不带 `--cases` 时运行全数据集。`--cases` 接受目录中的完整样本 ID；它只限制本次运行范围，**报告仍保留全部 50 张作为总分母**。CLI 的 `--repeats` 范围为 1–5，重复运行的样本必须全部轮次满足条件，不能挑选最好的一轮。

报告逐轮持久化到 `runtime/evaluations/`，工作台“评估”页面可查看。评估任务产物位于独立的 `runtime/autonomous-qualification/` 目录，不修改工作台现有任务。预测完成后，评分器才读取参考文件进行比较，参考数据不会回流给提取、定标或 API。

报告分别列出：

- 全数据集数量、本次尝试数、自动产物数、结构有效数、毫米定标数与人工干预数。
- 有参考且实际完成比较的样本数，以及参考最大误差不超过 0.1 mm 的样本数。
- 每张图每轮的失败原因、产物和视觉请求回执；`online_request_summary` 汇总全部实际调用，包含重复轮次。
- 总体、独立保留样本和历史校准样本的分组统计；未运行、失败、缺少参考或比例未确定的样本均保留。

`qualification.strict_requested_scope_passed` 只有在本次请求的每张图、每一轮都完成有效自动产物、取得毫米比例、没有人工干预且通过 0.1 mm 独立参考比较时才可能通过。在线模式还要求真实图像请求、HTTP/schema 成功和视觉 `match`。**严格范围未通过时退出码为 2，即使部分或全部 DXF 已经生成。** 子集通过也不等于全 50 张通过。

参考比较允许有限的朝向变换与平移对齐，不拟合缩放来掩盖比例错误。参考包本身可能包含拟合或简化，0.1 mm 阈值衡量的是与该参考的一致性，不是制造精度认证。源数据审计见 [dataset-audit.md](docs/dataset-audit.md)。历史 [acceptance-2026-09-20.md](docs/acceptance-2026-09-20.md) 描述的是旧模板辅助验收，不能作为当前自动流程的批量通过证据；当前运行应查看 `mode=autonomous_image` 的报告。

## 局部编辑与约束闭环

审核后的材料掩膜进入最多三轮局部拓扑编辑、几何执行和独立评估；每轮复查源图约束，避免只优化像素拟合或图元数量。聊天显示逐轮操作、拒绝原因与求解诊断。新增重分段、单图元改直线、带源图证据的相切圆角及跨版本对象追踪。

能力边界、产物含义和冻结案例重放方法见 [源图约束驱动的局部重建](docs/source-constrained-reconstruction.md)。已构造圆角、已绑定尺寸、通过联合求解和独立 GT 精度分别记录。

## 完整掩膜条件下的重建实验

为单独验证“初始 CAD → 标注引导的拓扑编辑 → 绑定与求解”，可使用已配准的 GT **栅格掩膜**。这是显式的开发实验，不是默认自主模式，也不证明分割准确率或未见图纸的泛化。预测端只读取冻结的原图、OCR 和掩膜，不读取 GT DXF 图元坐标；导出后才单独执行 DXF 比较。

```powershell
python scripts/run_oracle_mask_reconstruction.py CL60-main --run-dir runtime/oracle-mask/cl60-run-001 --online --provider-profile 9ecode-gpt-5.6-sol --evaluate
```

每次使用新的输出目录；省略 `--online` 可运行本地几何流程。阶段、API 回执及失败均持久化；结果分开报告对象类型数量、连通性、尺寸约束子集和 0.1 mm 独立参考比较。逐阶段叠加图可定位拓扑编辑或求解造成的退化。详见 [Oracle 掩膜实验与独立 DXF 对比](docs/oracle-mask-dxf-evaluation.md)。

本轮已确认的退化路径、真实在线回执与未通过的精度结果见 [2026-09-27 迭代记录](docs/oracle-mask-iteration-2026-09-27.md)。

2026-10-06 的分段搜索、物理弧长求解权重、源路径相切核验和构造半径保存改进，以及两图在线复测和未解决问题，见 [本轮迭代分析](docs/cad-iteration-2026-10-06.md)。构造半径的数值保存不等于完成独立标注绑定，API 成功不等于 GT 参数一致。

对象分解、完整引线归属、有界拓扑分支搜索和误差预算内求解的后续迭代见 [对象与属性精度迭代](docs/cad-precision-2026-10-06.md)。可为新实验增加用户指定的 1 mm 参考目标，原 0.1 mm 成绩继续保留，标注 R 仍精确约束：

```powershell
python scripts/run_oracle_mask_reconstruction.py HDSA-65-main --run-dir runtime/oracle-mask/hdsa-new-run --online --provider-profile 9ecode-gpt-5.6-sol --evaluate --evaluation-target-mm 1
```

`evaluation/summary.json` 的 `target_acceptance` 分别报告轮廓、对象数、完整参数、邻接和当前 DXF 的精确半径合同。1 mm 轮廓通过不能单独表示完整自动重建成功。

## 历史模板流程

旧的 293 参数模板和确认界面保留用于历史任务回读与回归，不是默认自动流程。需要显式运行历史评估：

```powershell
python -m contour_agent evaluate --engine template
python -m contour_agent evaluate --engine template --online --repeats 2
```

该路径使用声明的校准先验、简化踏面及参数确认，在线辅助评估还包含显式脚本确认。它的 `online_assisted_scope_passed` 不能代表零干预自动识别，更不能增加全数据集自动准确率。`--cases` 不适用于模板引擎。

## 代码入口

- `contour_agent/raster.py`、`outline_fallback.py`：原图轮廓提取与空结果备用路径。
- `contour_agent/dimension_evidence.py`：尺寸线关联、比例候选与独立线性尺寸验证。
- `contour_agent/vectorize.py`、`automatic.py`：LINE/ARC 拟合、兼容半径绑定、导出与逐实体回读。
- `contour_agent/vision_provider.py`：单次有界图像复核请求。
- `contour_agent/service.py`、`store.py`：任务调度、持久化与重启恢复。
- `contour_agent/server.py`、`web/`：本机工作台和产物浏览。
- `contour_agent/autonomous_qualification.py`、`autonomous_evaluation.py`：全分母批量评估与独立参考比较。
- `contour_agent/templates/`、`geometry.py`、`provider.py`、`qualification.py`：历史模板及其文本辅助评估。

# iSING-CAD · 自动主轮廓工作台

默认流程为：**原图 + OCR → 材料掩膜与初始 CAD → 源标注解析及局部拓扑编辑 → 约束绑定与数值求解 → LINE/ARC 导出及独立检查**。在线模式由模型提出源证据和编辑方案，本地几何执行器与求解器构造实际 DXF。阶段失败会保留有效草稿，并明确标注未完成的阶段。

严格半径模式将已验证箭头指向的标注 R 作为常数求解，并回读 DXF 原生 ARC 作零容差检查。部分约束成功不等于所有半径已解决，也不等于 GT 精度通过；详见[严格标注半径与来源审计](docs/strict-annotation-radius.md)。GT 派生掩膜只用于明确声明的条件性重建实验。

新增[截面站位与公差区间约束](docs/source-sections-and-tolerance-intervals.md)：区分曲线极值、顶点和参考尺寸定义的截面交点；新截面及径向厚度公差按上下界求解，不强制取中点。已知约束跨阶段保留时同时核实公差政策及边界，斜向厚度箭头尚未核实时保留未确认状态。

当前数据集全部 50 张都可提交运行。**可运行范围不等于识别成功率，更不等于尺寸准确率。** 系统分别记录提取结果、有效产物、毫米比例和独立参考误差；本文不宣称 50 张均已通过，实际结果以批量评估报告为准。

最新实测与逐项失败说明见 [自动流程验收记录](docs/acceptance-autonomous-2026-09-20.md)。工作台的评估报告可逐图打开原图叠加并下载 DXF。

已接入 [DXF GT 监督训练](docs/dxf-gt-supervision.md)：从用户提供的 DXF 建立图像配准掩膜，50 张图中 43 张通过标签检查，29 / 7 / 7 张用于训练 / 验证 / 开发测试。运行 `./start-segmentation.ps1` 启动后默认启用深度分割，推理只读取原图与模型权重。网页“查看训练标签”可检查标签，“GT 实测结果”查看新旧模型对比、在线调用与失败明细。历史弱标签基线记录保留在 [旧实验文档](docs/deep-segmentation.md)。

新增四图构建试验：先用已有分割表现较好的 CL60、182、202、231 跑完整流程。工作台提供快捷选择、真实模型分割叠加与二值 mask、提取边界及 CAD 叠加视图；[四图试验报告](http://127.0.0.1:8769/segmentation-pilot)保存新运行的产物及独立评分。这是按已知表现选择的开发子集，不能代表全部 50 张或未见图纸的准确率。详见 [试验说明](docs/segmentation-pilot.md)。

输出是自动轮廓草稿。当前已实现源证据核验、精确半径与角度绑定、部分尺寸/公差和相切的联合求解，仍未完成所有标注、对象和复杂踏面的覆盖。成功导出或视觉返回 `match`，均不会将 `dimensions_verified`、`engineering_certified` 改为 `true`。

## GitHub 版本与更新记录

以下按仓库实际提交、分支和标签整理，日期使用北京时间。`v11`、`v43`、`v77`、`V92` 等名称主要表示开发实验轮次，不是一一对应的 Git 标签；没有独立提交的实验不能作为单独的 GitHub 版本恢复。历史提交说明依据相应代码差异和该提交中的文档，不表示它已经满足全部 GT 检查。

### 主分支备份

| 日期 | 版本 / 提交 | 对应更新 | 验证范围与限制 |
| --- | --- | --- | --- |
| 2026-10-14 | [backup-2026-10-14](https://github.com/KazDong628/iSING-CAD/tree/backup-2026-10-14)（本次备份） | 截面测量位置与公差区间；单引线圆心尺寸；设计接点复核；已知约束固定后的可行解细化；有限直线保护；半径请求分页与有效旧提案保护；折线引线与未知 LINE/ARC 过渡候选。 | 当前代码回归 **2633 通过、3 跳过**。186 在线阶段重放有 **7 个精确半径**，配准最大误差 **1.123 mm**，仍缺一条 ARC，部分半径及相切未绑定；不能视为完整 GT 通过或新一轮网页全流程测试。详见[本次备份说明](docs/version-backup-2026-10-14.md)。 |
| 2026-10-12 | [c4106fd · V92 实验备份](https://github.com/KazDong628/iSING-CAD/commit/c4106fdd729db12a5ec52e3ea3b8649d23109c3d) | 汇总后续源标注完成闭环：尺寸测量对象、OCR 多模态重读、源证据与状态检查点、保留已知约束的追加绑定、局部拓扑补全、角度与圆角构造、标注优先冲突处理、自动 DXF 输出及失败记录。 | `V92` 是提交说明中的实验标识。包含多轮 HDSA/044 开发结果与未解项；成功输出和约束子集通过不等于完整参考精度通过。见[初始化与可辨识性](docs/source-initialization-and-identifiability-20261012.md)、[自动输出覆盖](docs/automatic-dxf-output-coverage-20261012.md)。 |
| 2026-10-10 | [220ffc8 · 源证据指纹修复](https://github.com/KazDong628/iSING-CAD/commit/220ffc827628a1ad4e0cfc82f5692e228798b472) | 仅从源证据指纹的指定检测统计位置排除实际耗时，避免相同物理证据因运行时间不同而错误撤销产物；几何、标注、预算、截断和验证状态继续参与核验。 | 是独立的发布回归修复，不能据此宣称所有尺寸或轮廓误差已解决。 |
| 2026-10-10 | [9397a51 · 标注角度与圆角迭代](https://github.com/KazDong628/iSING-CAD/commit/9397a51bc229abb6993df6292a7a64465b5922fc) | 强化源角度解析与斜线恢复、精确标注半径、有限对象归属、圆角插入及多轮拓扑搜索；将构造、绑定、求解和发布检查分开。 | 从 `f76d290` 基础继续受控迭代，保留未通过案例；见[源标注控制重建](docs/annotation-controlled-reconstruction.md)。 |
| 2026-10-06 | [f76d290 · CAD 精度迭代备份](https://github.com/KazDong628/iSING-CAD/commit/f76d290e784c68e15f38830859c192cace025fc8) | 严格标注 R 与完整引线归属、多圆弧分段及接点搜索、物理弧长求解权重、源笔画/掩膜诊断、构造半径保存和独立对象参数比较。 | 用户指定过的回退基线。两图在线实验与未通过的精度结果保留，见[精度迭代记录](docs/cad-precision-2026-10-06.md)。 |
| 2026-09-29 | [c1d825f · GT DXF 掩膜导入与工作台修复](https://github.com/KazDong628/iSING-CAD/commit/c1d825f240e135b7e73155e35d9e99587926b6ca) | 网页导入 GT DXF，在本地配准并生成二值掩膜；完善导入检查、工作台脚本缓存和前端元素处理。 | GT 导入是明确声明的条件性实验；原始 GT DXF 不发送给在线模型，不能计入自动分割成绩。 |
| 2026-09-27 | [ca34b91 · 标注引导重建与 Oracle 评估](https://github.com/KazDong628/iSING-CAD/commit/ca34b91048d980b313210772fd6ad888ab826204) | 加入完整掩膜条件下的重建脚本、逐阶段独立 DXF 评估、标注引导局部编辑、重建反馈、API 输出预算及回执审计。 | 预测与 GT 评分分开；已生成 DXF、在线成功、参数约束与独立参考误差分别报告。见[2026-09-27 迭代记录](docs/oracle-mask-iteration-2026-09-27.md)。 |
| 2026-09-22 | [v2](https://github.com/KazDong628/iSING-CAD/tree/v2) / [85e2256](https://github.com/KazDong628/iSING-CAD/commit/85e225631f53be364008d3ff72ea76a43f56fdd9) | 首次源码上传；标签说明为“v2：增加了一个局部拓扑自适应修改的 Agent”。包含原图/OCR 工作台、分割、初始 CAD、局部拓扑编辑、数值求解与导出评估入口。 | 这是当前 Git 历史的起点，仓库没有可独立恢复的 `v1` 标签。旧模板流程作为显式历史诊断保留，不能作为自主重建成绩。 |

### 已上传的历史开发分支

分支 [`codex/controlled-cad-iteration-f76d290`](https://github.com/KazDong628/iSING-CAD/tree/codex/controlled-cad-iteration-f76d290) 上另有下列提交。它们记录受控实验过程，不是主分支各备份之间的连续发布版本；其中 `6f4707a` 与主分支 `220ffc8` 是对应的指纹修复。

| 日期 | 提交 | 更新 |
| --- | --- | --- |
| 2026-10-10 | [1f10bcd](https://github.com/KazDong628/iSING-CAD/commit/1f10bcdbc10d02f9b80314f659ad33e59330878c) | 记录严格 CAD 迭代失败、发布重新检查结果、V16 阶段重放和仍未满足的对象/尺寸要求。 |
| 2026-10-10 | [6f4707a](https://github.com/KazDong628/iSING-CAD/commit/6f4707accbec641310c9551ea120eadc88ba780f) | 开发分支的源证据指纹耗时修复；只忽略指定遥测字段，真实来源改变仍拒绝复用。 |
| 2026-10-10 | [5c93c90](https://github.com/KazDong628/iSING-CAD/commit/5c93c9059243f0b17873518dfbde36e5ebd8b3fe) | 保留多步修复搜索机会与当前 CAD 约束证书，避免一次中间候选不可行就耗尽后续结构探索预算。 |
| 2026-10-10 | [11c99be](https://github.com/KazDong628/iSING-CAD/commit/11c99be01771115bc42f81bd1bcedad93ffe2fe5) | 补充未解决的尺寸覆盖、测量对象缺口和独立自由度分析；区分精确 R/角度与完整轮廓可辨识性。 |
| 2026-10-10 | [a493bb2](https://github.com/KazDong628/iSING-CAD/commit/a493bb2af198b8f55cd3e175fab98b1495ab82d9) | 完善源约束联合求解、绑定分页、重建完整性合同和发布 CAD 的原生几何验证。 |

使用上述提交 SHA 或标签可恢复相应的**已提交源码**。`__dataset/`、训练权重、密钥、本机配置与 `runtime/` 报告未纳入 Git，恢复源码不会同时恢复这些本地文件；复现实验还需要匹配的输入、配置与运行记录。本次标签不覆盖或重写任何历史版本。

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

自动拓扑阶段采用候选提案、几何执行和复核闭环：局部拓扑编辑 Agent 引用已有图元 ID 提出合并、重分段、斜线恢复或标注圆角插入等操作；几何执行器依据源边界与标注证据生成单项或互不冲突的组合候选；后续重新绑定、求解与原生回读检查决定是否替换当前产物。闭合、原图笔画、对象类型、已知约束保存、掩膜诊断和独立参考精度分别记录，不能只靠对象数减少发布候选。最初协议见 [局部拓扑编辑闭环](docs/topology-edit-agent-v2.md)，后续更新见上述版本记录及相关实现文档。

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
3. **初始 CAD、标注绑定与联合求解。** 对边界拟合相连的 LINE/ARC，作为对象假设和数值初值。源证据支持时，局部编辑可恢复斜线、插入圆角或重分段；独立核验后将已知 R、角度、尺寸和关系送入求解器，固定已知约束并优化未知属性。拟合值不能替代尚未成功绑定的标注；当前仍不保证小槽口、微圆角、踏面及所有接点完整。
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

缺失的圆滑接点现在可由两次在线原图复核提出设计相切关系，再保留已知半径、角度和尺寸联合求解切点与圆心。关系假设、原生 DXF 方程验证和独立 GT 精度分开记录；详见 [设计接点补全与联合求解](docs/source-design-joint-completion.md)。竖排 OCR 扩大阅读方向的上下文，对已撤回的可疑标注执行有预算限制的单条双视图重读，保留完整的重启证据链。

圆心位置尺寸也支持单根已核实半径引线终止于明确尺寸延长线的情况。已知标注方程固定后，没有新增绑定但尚未收敛的可行解仍可继续有界优化未知属性；只在同一冻结原图目标改善且全部原生方程检查通过时替换。详见 [圆心测量位置与后续优化](docs/source-center-stations-and-refinement.md)。

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

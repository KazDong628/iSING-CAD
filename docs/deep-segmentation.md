# 历史弱标签基线：环境、训练与复现

**本页保留 `unet-r18-weak-v1` 的历史启发式弱标签实验。当前方向已转为使用用户提供的 DXF GT 生成像素监督；本轮 `gt-data-v3` / `unet-r18-dxf-v2` 的流程见 [DXF GT 监督说明](dxf-gt-supervision.md)。下面的旧指标不代表新标签或本轮训练的结果。**

本实验使用真实的 U-Net / ResNet-18 模型预测材料区域，再从预测 mask 提取主轮廓。模型推理只读取原图，不读取 OCR、训练 mask、GT、逐图模板或历史预测。接入 CAD 工作流后，OCR 定标、LINE/ARC 拟合及验证仍是另外的阶段；分割模型本身不满足工程尺寸、半径、相切或公差约束。

当前起始数据为 **50 张源图生成的弱 mask、49 个去重组，train / val / test 为 35 / 8 / 7**。准备清单中的 50 个标签均为 `label_source=source_heuristic`、`reviewed=false`，人工审阅标签数为 0。这表示弱标签准备成功，不表示 50 张轮廓正确。现有 50 张此前已被开发流程查看，因此 `test` 也只能称开发测试，不能称未见图纸的盲测。

20 轮训练已在 `runtime/segmentation/runs/unet-r18-weak-v1/` 完成，用时 115.5 秒；按验证集弱标签 IoU 选中了第 12 轮。当前弱标签中存在只覆盖部分零件的样本，模型会受到这些教师标签偏差的影响，因此本轮结果仍是未验收的弱监督基线，不是已经合格的主轮廓识别模型。本轮没有自动追加训练或提升模型为默认方案。

## 本次实测结果

| 测量 | 实际结果 |
| --- | ---: |
| 完成轮次 / 最优轮次 | 20 / 12 |
| 训练用时 | 115.5 秒 |
| 最优验证集弱标签 IoU，8 张 | 0.467715 |
| 开发测试集原图弱标签 IoU，7 张 | 0.453432 |
| 同 7 张合成干扰图弱标签 IoU | 0.671378 |
| train / val / test 人工审阅标签 | 0 / 0 / 0 |

结果来自 [training.json](../runtime/segmentation/runs/unet-r18-weak-v1/training.json)、[验证报告](../runtime/segmentation/runs/unet-r18-weak-v1/validation/evaluation.json) 和 [开发测试报告](../runtime/segmentation/runs/unet-r18-weak-v1/test/evaluation.json)。本次 `best.pt` 的 SHA256 为 `fc3dae351c2d32a1f2def96a8d2413c231240420cce3fb68c950aa3c329ba7a6`。

合成干扰组分数更高，只说明模型在这 7 张经过特定处理的图上更接近现有弱 mask，**不能据此证明抗干扰能力或未见图泛化有所提升**。人工目标边界尚未建立，且这些图源此前已被开发流程查看；评估仍明确记录 `true_accuracy_verified=false`。

## 实际环境

本机使用 Windows、Python 3.11.9，`.venv-seg` 以 `--system-site-packages` 创建，复用已有基础工作台依赖。2026-09-20 核对的关键版本如下：

| 组件 | 版本或来源 |
| --- | --- |
| PyTorch | `2.1.2+cu121` |
| torchvision | `0.16.2+cu121` |
| segmentation-models-pytorch | `0.5.0`，本地 `vendor/segmentation_models.pytorch` |
| timm | `1.0.15` |
| NumPy | `1.26.4` |
| huggingface-hub | `0.24.7` |
| safetensors / tqdm / Pillow | `0.4.2` / `4.66.5` / `11.0.0` |
| 本次训练设备 | NVIDIA GeForce RTX 3050 Laptop GPU |

上游为 [segmentation_models.pytorch 官方仓库](https://github.com/qubvel-org/segmentation_models.pytorch)，MIT 许可证，固定到 `v0.5.0` 对应 commit `420ce84b0c2df0286fa9bb2bd1499eea625c9b33`。本机 vendor 已克隆；新副本缺少该目录时才执行克隆：

```powershell
git clone --branch v0.5.0 https://github.com/qubvel-org/segmentation_models.pytorch.git vendor/segmentation_models.pytorch
git -C vendor/segmentation_models.pytorch checkout --detach 420ce84b0c2df0286fa9bb2bd1499eea625c9b33
git -C vendor/segmentation_models.pytorch rev-parse HEAD
```

以下命令均从项目根目录运行。先按 [主 README](../README.md) 准备基础工作台依赖；已有 `.venv-seg` 时跳过创建环境这一步：

```powershell
python -m venv --system-site-packages .venv-seg
.venv-seg\Scripts\python.exe -m pip install -r requirements-segmentation.txt
.venv-seg\Scripts\python.exe -c "import torch, torchvision, segmentation_models_pytorch as smp; print(torch.__version__, torchvision.__version__, smp.__version__); print('cuda_available=', torch.cuda.is_available())"
```

`requirements-segmentation.txt` 固定本次分割实验的关键依赖，并从已核对的 vendor 安装 SMP。它不是整个 Anaconda / 工作台环境的完整锁文件；`--system-site-packages` 会继承基础环境。复现时同时记录 `pip freeze`、Python 版本、GPU / 驱动和 vendor commit。固定随机种子可以复现划分与实验配置，但本实现没有承诺跨硬件、驱动的逐位一致训练结果。

## 准备弱标签与冻结划分

```powershell
.venv-seg\Scripts\python.exe -m contour_agent.segmentation prepare --dataset __dataset --output runtime/segmentation/data
```

`prepare_dataset(..., seed=42)` 只扫描 `__dataset/origin` 的图像和配对 OCR。先使用现有剖面区域提取器；仅在没有轮廓时调用备用提取器，再填充多边形生成 0/255 二值 mask。它不读取 `GT`、`vis` 或既有 `runtime` 预测。图像和 mask 保存为匹配尺寸的 PNG，最长边不超过 1536；清单保留原尺寸、各轴缩放比例、源图 / OCR / 产物哈希及失败原因。

相同源图 SHA256、或 pHash 汉明距离不超过 4 且宽高比相差不超过 3% 的近重复图按连通组绑定。整个组进入同一划分，293 及其重复组固定进入 train；其余组按固定种子分配，目标比例约为 70% / 15% / 15%。实际清单为：

| 清单项目 | 实际数量 |
| --- | ---: |
| 全部源图 / 可加载弱标签 | 50 / 50 |
| 去重组 | 49 |
| train / val / test | 35 / 8 / 7 |
| 初始人工审阅标签 | 0 |

所有行保存在 `manifest.json` 的 `cases` 中。提取失败也保留，标记 `trainable=false`，不得从原始样本分母删除。`split_claim=development_only_not_blind_test` 明确限定这些划分的意义。

若 `manifest.json` 已存在，准备命令核对种子、源图 / OCR 哈希和产物完整性后复用；发现变化会拒绝覆盖。需要新版本时使用新的输出目录。独立 `reviews.json` 和 `reviewed_masks/` 不会被 prepare 覆盖。

## 训练

本次运行目录是 `runtime/segmentation/runs/unet-r18-weak-v1/`。复现应选择一个新目录；已存在 `best.pt` 的目录会被拒绝覆盖。以下示例使用另一个目录名：

```powershell
.venv-seg\Scripts\python.exe -m contour_agent.segmentation train --manifest runtime/segmentation/data/manifest.json --output runtime/segmentation/runs/unet-r18-weak-reproduce-01 --epochs 20 --size 512 --batch-size 2 --device cuda
```

无 CUDA 设备时可明确使用 `--device cpu`；不传 `--device` 则自动选择可用设备。当前 CLI 的训练与数据准备种子均默认为 42，没有 `--seed` 参数；Python 接口提供 `seed`。训练图像按宽高比缩放并补白为 512×512，mask 使用最近邻变换，归一化采用 ImageNet 均值 / 标准差。

| 训练项 | 当前实现 |
| --- | --- |
| 网络 | U-Net，ResNet-18 编码器，1 个前景通道 |
| decoder channels | `[128, 64, 32, 16, 8]` |
| 初始化 | 官方 torchvision ResNet-18 ImageNet-1K V1 编码器权重 |
| 损失 | BCE with logits + Dice loss |
| 优化器 | AdamW；编码器学习率 `1e-4`，解码器 / 分割头 `1e-3`，weight decay `1e-4` |
| 调度 | CosineAnnealingLR；20 轮配置下前 2 轮冻结编码器 |
| 增强 | 水平翻转、小角度旋转 / 缩放、可复现的标注线 / 箭头 / 文字 / 剖面线干扰 |
| 选模型 | 仅按 val 弱标签 IoU 选 `best.pt`；test 不参与训练或选 checkpoint |

编码器权重从 [PyTorch 官方文件](https://download.pytorch.org/models/resnet18-f37072fd.pth) 下载到 `runtime/segmentation/weights/`，代码校验发布哈希前缀。本次已下载文件的 SHA256 为 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

每轮更新 `training.json`，包含损失、val 弱标签 IoU、耗时、训练 / 验证样本 ID、审阅数量、manifest / reviews 哈希和初始化来源；`best.pt` 保存模型参数、输入尺寸、选中轮次和来源信息。首次训练与复现运行的数值必须分别读取各自报告，不能因为脚本结束或 checkpoint 存在就宣布精度合格。

## 评估与单图预测

本次已完成运行的 checkpoint 可用于下列评估与预测命令；评估示例使用另一个输出目录，保留原始运行报告：

```powershell
.venv-seg\Scripts\python.exe -m contour_agent.segmentation evaluate --manifest runtime/segmentation/data/manifest.json --checkpoint runtime/segmentation/runs/unet-r18-weak-v1/best.pt --output runtime/segmentation/evaluations/unet-r18-weak-v1-test --split test
.venv-seg\Scripts\python.exe -m contour_agent.segmentation predict --checkpoint runtime/segmentation/runs/unet-r18-weak-v1/best.pt --image __dataset/origin/solid-arrow-ping__img_000293.jpg --output runtime/segmentation/predictions/unet-r18-weak-v1-293
```

评估使用固定 512 像素画布并排除补白区域，分别记录原图与人工合成标注干扰下的 IoU、Dice、像素精确率 / 召回率、边界 F1 和边界距离。默认边界容差为评估网格上的 3 像素，距离也以该网格像素计，不能解释成原图像素或毫米误差。干扰压力测试不等于真实新图纸泛化测试。

当前 val / test 的目标仍是自动生成的弱 mask，指标表示**与这些弱标签的一致性**，不表示人工真值准确率、尺寸准确率或 DXF 的 0.1 mm 参考通过率。报告的 `true_accuracy_verified` 仍为 `false`。已有标签来源审计见 [segmentation-label-audit.md](segmentation-label-audit.md)；本次重新准备的弱标签没有使用该审计所述的旧 GT 派生 mask。

单图 `predict` 生成 `prediction-mask.png`、`prediction-overlay.png`、`segmentation.json` 及轮廓提取证据。它本身不导出带尺寸的 DXF。293 位于训练组，因此该命令是推理与可视化演示，不能当作独立测试。

## 本机工作台接入与标签审阅

本机已有环境和第一版权重时可直接运行 `./start-segmentation.ps1`。打开工作台，勾选“深度分割（试验）”后点击自动绘制；上传窗口也有相同选项。模型不会默认启用。

明确选定本地产生的 checkpoint 后，用分割环境启动服务器：

```powershell
$env:CONTOUR_SEGMENTATION_CHECKPOINT = (Resolve-Path runtime/segmentation/runs/unet-r18-weak-v1/best.pt).Path
.venv-seg\Scripts\python.exe -m contour_agent serve --port 8769
```

配置环境变量只使模型可用，不会自动把它提升为默认算法。任务还需显式启用 `use_segmentation=true`。例如本机 API 的完整请求：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8769/api/jobs -ContentType application/json -Body '{"case_id":"solid-arrow-ping__img_000293","mode":"autonomous_image","use_api":false,"use_segmentation":true}'
```

选定模型后若未找到轮廓，任务保留该失败，不替换成启发式结果冒充模型输出。获得轮廓后才继续现有的 OCR 定标、LINE/ARC 导出和逐实体回读；在线视觉 API 仍是独立可选项。分割置信度、视觉 `match`、几何有效、尺寸约束和参考精度继续分别报告。

打开 [本机标签审阅页](http://127.0.0.1:8769/segmentation-review)，对照图像修正 mask，再明确保存。保存生成独立的 `reviewed_masks/<case_id>.png` 与 `reviews.json`，记录源图哈希、mask 哈希、时间和 `label_source=local_user_review`；不会改写冻结 manifest、原始数据或 train / val / test 划分。

后续训练会读取与源图哈希匹配的审阅 mask。审阅保存不会启动训练，不会更换 checkpoint，也不会自动认证 CAD 精度。每次重训使用新运行目录，同时保留本次的 manifest、审阅记录和 mask 版本；只保留 `best.pt` 不足以重现标签状态。即使这些开发样本之后经过人工修正，也不能因此改称全新盲测；需要另外收集未被开发流程查看的独立图源。

## 本次网页与 HTTP 联调

2026-09-20，已在 `http://127.0.0.1:8769/` 实际启用该 checkpoint。通过网页提交 293 的模型任务，通过本机 HTTP API 提交 044、1-main 的模型任务。三例都实际使用学习模型，没有启发式替代；均生成并成功下载 DXF、SVG、叠图和 JSON。044、293 属训练组，仅验证端到端链路；1-main 属开发 test 组，但导出成功也不是轮廓准确率。

293 任务 `98d9bd169f2440d89695435352df335f` 实际调用 qwen-chat：HTTP 与输出结构成功，视觉结论 `match`，生成 98 个 LINE/ARC 实体。人工目视该叠图仍可见左上方粗糙度标注附近被误收入轮廓，因此不能把该视觉结论作为消除干扰的证明。1-main 的分割还报告多个显著材料区域，当前导出的主分量可能不完整。

逐项 HTTP 结果在 `runtime/segmentation/http-smoke-results.json`，权重在 `runtime/segmentation/runs/unet-r18-weak-v1/best.pt`。已有弱标签与模型预测的诊断拼图在 `runtime/segmentation/validation-contact.png`：每行依次为输入、弱标签、模型预测，预测列包含等比例补白；该图用于发现部分对象标签与断裂问题，不是人工真值对比。

目前没有保存任何人工审核标签。模型能运行，**尚未达到可靠自动主轮廓提取的验收标准**。优先修正材料掩膜中的遗漏、错误并入区域及多区域选择，并保留独立审核后的开发验证数据，再启动下一版微调。

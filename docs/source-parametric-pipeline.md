# 原图标注驱动的主轮廓参数化

主流程把分割边界当作初始形状。它先输出原始 LINE/ARC 拟合，再使用原图笔画、OCR文字区域和模型采样间距提出较少图元的连接图。这个连接图提供对象类型、共享端点和环内顺序；坐标、半径只是初值。

## 输入与独立阶段

1. `topology.py`：检查文字重叠、局部绕行和原图线条支持，保存修正前后边界、每次局部修正的支持度及候选图元。真实缺口有线条支持时保留。模型掩膜差异单独记录，不把旧拟合门限改为通过。
2. `topology_edit_provider.py`：局部编辑 Agent 查看原图和带图元编号的当前候选，只能返回已有连续图元 ID 与有限操作类型，不返回坐标。`topology_editing.py` 在本地原始边界上重新拟合 LINE/ARC，生成单项候选和互不冲突编辑的组合候选。第二个独立评估请求只能在通过本地门禁的候选中选择；最终还需通过本地改进门禁。完整约束见 [局部拓扑编辑闭环](topology-edit-agent-v2.md)。
3. `constraint_binding.py`：从原始OCR规范化记录读取名义尺寸，构建到图元或共享顶点的候选。半径关联需要引线证据；尺寸跨度需要尺寸线、端点位置及延长线证据。相近的拟合半径不能证明对应关系。
4. `binding_provider.py`：在线模型同时查看原图、编号拓扑和最多四个局部放大面板，只能选择已有候选ID、OCR记录ID和关系ID，不能生成新的尺寸值或CAD坐标。一次请求最多24条记录、48个候选、600秒；HTTP成功、输出结构成功、绑定采用分别记账。
5. `parametric_solver.py`：联合调整共享顶点和圆弧参数，强制保持闭合和圆弧端点入射关系。支持半径、轴向/点间距离、角度、水平、垂直、相切。当前自动绑定器仅生成半径与横纵距离；直径和角度仍留在未绑定清单，不能算已验证。
6. `parametric_pipeline.py`：保存初始产物、局部编辑候选、评估结论、校正轮廓、求解候选及失败原因。原图校正和尺寸求解有不同的接受依据；可发布尺寸未解出的校正草图。通过数值检查的部分约束解才替换最终CAD。下载DXF后再次回读检查实体和单位。

## 如何理解通过

求解器报告每项实际值、残差、固定公差和是否满足，同时报告约束雅可比秩与剩余形状自由度。数值线性公差为0.05 mm，角度公差为0.1度；这是已绑定约束的数值收敛检查，不是独立GT误差。共享节点确保连接，几何检查另外拒绝自交、退化和过大位移。

欠约束时可以采用满足部分约束的有效草图，但 `all_dimensions_verified` 始终为假。标注对应关系也不能由“求解成功”反向证明。原图单位未确定时仅允许无量纲几何关系，不把像素伪装成毫米。

独立参考DXF仅由生成结束后的评分器读取。现有0.1 mm参考阈值、D4方向/平移对齐且不拟合比例的协议不变。任何GT、评估对齐或逐图参考参数都不进入拓扑、绑定API或求解器。

## 产物与复现

每个自动任务保存 `baseline-drawing.dxf`、`baseline-overlay.png`、`topology.json`、`topology-overlay.png`、`topology-candidates.json`、`topology-edit-proposals.json`、`correction-evidence.json`、`binding-candidates.json`、`binding-topology.png`、`constraint-bindings.json`、`parametric-stage.json`、`parametric-solution.json`，以及当前可下载的 `drawing.dxf`、`overlay.png` 和 `model.json`。阶段失败也保留已有产物。

使用现有四图入口 `tools/run_segmentation_pilot.py --online` 并配置checkpoint和manifest即可触发。凭据只放进程环境 `USTC_API_KEY`。文字尺寸规范化、图元标注绑定和最终视觉复核是三类独立请求；四图最多各一次即12次，诊断性额外调用另记。测试四图均为开发样本，全50图分母不变，不能作为盲测成绩。

回归命令：`python -m pytest -q`。运行服务：`python -m contour_agent serve`。工作台新增“图元连接”视图，结果报告分别显示初始拟合、拓扑候选和最终CAD。

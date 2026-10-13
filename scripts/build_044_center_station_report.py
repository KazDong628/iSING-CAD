"""Read frozen experiments and their independent comparisons; never edit CAD."""
import base64
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_source_joint_report import collect

BASE = ROOT/'runtime/cad-044-center-station-20261013'
def read(path): return json.loads(path.read_text('utf8'))
def esc(value): return html.escape(str(value), quote=True)


def main():
    cases = [collect('上轮：补齐相切及181', ROOT/'runtime/cad-source-joints-20261013/044-scoped-round4'),
             collect('本轮1：新增圆心高度78', BASE/'scoped-round1'),
             collect('本轮2：保留已知约束继续优化', BASE/'scoped-round2'),
             collect('本轮2：完整范围（含未完成踏面）', BASE/'round2')]
    runs = [read(BASE/name/'run-manifest.json') for name in ('round1', 'round2')]
    completion = [read(BASE/name/'source-measurements/completion.json') for name in ('round1', 'round2')]
    latest = read(BASE/'round2/after/parametric-solution.json')
    bindings = read(BASE/'round2/after/constraint-bindings.json')
    known = [c for c in bindings['constraints'] if c.get('record_id') == 'r037']
    height = [c for c in latest['constraints'] if c.get('record_id') == 'r037']
    baseline_model = read(ROOT/'runtime/cad-source-joints-20261013/044-round4/after/model.json')
    old = {e['id']: e for e in baseline_model['entities']}
    old_height = old['g020']['start'][1]-old['g015']['center'][1]
    comparison = read(BASE/'scoped-round2/evaluation/comparison.json')
    inventory = read(BASE/'round2/after/binding-candidates.json')
    source_records = {r['id']: r for r in read(BASE/'round2/after/effective-radius-source-document.json')['records']}
    measurement_gaps = [{**r, 'text': source_records[r['record_id']]['text']}
        for r in inventory['length_source_observations'] if r['record_id'] in {'r012','r025','r030','r037','r041'}]
    pred = {e['id']: e for e in comparison['aligned_prediction_entities']}
    refs = {e['id']: e for e in comparison['reference']['filtered_profile']['entities']}
    localized = []
    import math
    for row in comparison['supplemental_entity_correspondence'][-1]['rows']:
        entity = pred[row['prediction_id']]
        nearest = refs[row['nearest_candidates'][0]['reference_id']]
        localized.append(dict(prediction_id=entity['id'], type=entity['type'], radius=entity.get('radius'),
            matching_status=row['status'], nearest_reference_id=nearest['id'],
            full_parameter_match=row['status'] == 'unique_full_parameter_agreement',
            center_error_mm=math.dist(entity['center'], nearest['center']) if entity['type'] == nearest['type'] == 'ARC' else None,
            sweep_error_deg=abs(entity['sweep_deg']-nearest['sweep_deg']) if entity['type'] == nearest['type'] == 'ARC' else None,
            sampled_primitive_max_mm=row['nearest_candidates'][0]['sampled_symmetric_max_mm']))
    log = ROOT/'runtime/pytest-044-center-station-final-20261013.txt'
    tests = log.read_text('utf8', errors='replace') if log.exists() else ''
    test_result = next((line for line in reversed(tests.splitlines()) if ' passed' in line or ' failed' in line), '全量回归进行中')
    data = dict(schema_version='044-source-center-station-iteration-v1', cases=cases,
        source_height_before_mm=old_height, source_height_binding=known, source_height_after=height,
        completion_receipts=completion, native_diagnostics=latest['diagnostics'], localization=localized,
        measurement_candidate_gaps=measurement_gaps,
        online=[{k:r.get(k) for k in ('status','provider','code_unchanged','upstream_unchanged',
            'prediction_unchanged_after_evaluation','ground_truth_coordinates_sent_to_provider')} for r in runs],
        test_result=test_result, target_tolerance_mm=1.5, original_0_1mm_threshold_retained=True,
        fresh_end_to_end_test=False, oracle_mask_conditioned=True, reference_geometry_used_for_prediction=False)
    (BASE/'analysis.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf8')
    headers = ['实验', 'LINE / ARC', 'R /角度 /G1 /尺寸', '配准最大 mm', 'RMS mm', 'P95 mm', '长度差 mm', '1.5mm完整对象匹配']
    rows = ''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [c['label'],
        f"{c['object_counts'].get('LINE', 0)} / {c['object_counts'].get('ARC', 0)}",
        ' / '.join(str(c['native'][key]['required_count']) for key in ('radii','angles','tangencies','measurements')),
        f"{c['shape']['conservative_max_error_mm']:.6f}", f"{c['shape']['rms_error_mm']:.6f}",
        f"{c['shape']['p95_error_mm']:.6f}", f"{c['shape']['length_error_mm']:.6f}",
        f"{c['match']['matched_count']} / {sum(c['object_counts'].values())}"])+'</tr>' for c in cases)
    table = '<table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr>'+rows+'</table>'
    local = '<table><tr><th>预测对象</th><th>R</th><th>1.5mm匹配状态</th><th>近邻圆心差 mm</th><th>近邻扫角差 °</th></tr>'+''.join(
        '<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [r['prediction_id'], r['radius'], r['matching_status'],
            f"{r['center_error_mm']:.6f}" if r['center_error_mm'] is not None else '—',
            f"{r['sweep_error_deg']:.6f}" if r['sweep_error_deg'] is not None else '—'])+'</tr>'
        for r in localized if r['type'] == 'ARC' or not r['full_parameter_match'])+'</table>'
    galleries = ''
    for case in cases:
        path = Path(case['path'])
        url = 'data:image/svg+xml;base64,'+base64.b64encode((path/'evaluation/overlay.svg').read_bytes()).decode()
        galleries += f"<figure><figcaption>{esc(case['label'])}：洋红GT，青色预测，保留原始比较器图。</figcaption><img src='{url}'><p><a href='{esc((path/'after/drawing.dxf').as_uri())}'>DXF</a> · <a href='{esc((path/'evaluation/comparison.json').as_uri())}'>独立逐对象比较</a></p></figure>"
    refinement = completion[-1]['rounds'][0].get('known_subset_refinement', {})
    gaps = '<table><tr><th>标注</th><th>源图箭头对</th><th>候选数</th><th>当前候选阶段状态</th></tr>'+''.join(
        '<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['record_id']+' '+r['text'],
            r.get('arrow_pair_observed'), r['candidate_count'], r['status']])+'</tr>' for r in measurement_gaps)+'</table>'
    outcome = ('接受了继续优化的候选' if refinement.get('accepted') else '继续优化未达到源图目标改善条件，保留原有可行解')
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>044圆心定位及固定约束后续优化</title>
<style>body{{max-width:1300px;margin:30px auto;padding:20px;font:16px/1.7 system-ui;color:#203c38;background:#f4f7f5}}section{{padding:24px;background:white;margin:20px 0;border:1px solid #d3dfda}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border:1px solid #ccdcd4;text-align:left}}th{{background:#e3eee8}}img{{width:100%}}figure{{margin:25px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#087469}}</style>
<h1>044：圆心尺寸遗漏与可行解过早停止的迭代</h1>
<p>保留精确半径、角度、相切及已绑定尺寸，补入原图圆心高度78，再继续优化未知属性。结果尚未达到全部GT对象参数一致；本报告保留每轮结果和完整范围的失败。</p>
<section><h2>原因与修改</h2><ol>
<li>“78”标注的远端是R65的圆心，不是轮廓顶点。旧候选生成只识别两根半径引线共同终止的圆心，漏掉单根半径引线与明确尺寸延长线共同定义的圆心。这使正确数字没有任何可供API选择的候选。</li>
<li>补充单引线圆心候选：要求箭头到ARC的唯一所有权、原始长引线笔画、有限终止、不越过延长线继续、两端尺寸箭头及连续延长线。API仍需独立检查原图语义；坐标仅用于关联，不作为锁定值。</li>
<li>在线API将r037关联到g015圆心与g020水平底边。高度由{old_height:.9f}变为{height[0]['actual']:.9f} mm，误差{height[0]['absolute_residual']:.3e} mm。新增等式后原有41条方程保留，自由度8降到7。</li>
<li>求解曾在45秒预算到达时得到可行检查点，尚未收敛，但流程见可行便停止。现在没有新增标注时仍可继续固定方程求解；新增方程通过后也可用剩余预算优化未知属性。</li>
<li>后续优化使用同一批冻结的原图笔画观测和同一权重比较前后目标，必须改善源图残差，并重新通过原图、全部已知方程、自由度及五类原生DXF回读检查。失败保留原输出，GT分数不参与选择。本轮2：{outcome}。</li></ol></section>
<section><h2>独立比较</h2>{table}<p>所有数值来自预测冻结后的比较。1.5mm为并列目标诊断，原有0.1mm检查没有修改。简化范围仅沿用你授权的右侧踏面省略政策，保留左侧全部圆角与斜线。完整范围仍未通过，保留在表中。</p>
<p>在线请求传输、源标注覆盖、数值约束满足与GT精度是不同结论；模型已处理可发送的候选页不代表全部原图标注已绑定。这是既有GT栅格掩膜和初始拓扑的后续阶段开发复测，未重新从网页上传全流程测试。</p></section>
<section><h2>仍有偏差的对象</h2>{local}<p>未完整匹配的行使用比较器的空间近邻诊断，不宣称已建立真值对应。本轮1.5mm完整参数检查仍未匹配的是R180、R114、R65、R167和左下24°短斜线；其半径或角度已精确，差异在圆心、端点、扫角或长度。R114近邻圆心差约2.36mm，短圆弧仍可有接近的局部轮廓；近似重合的轮廓并不唯一确定圆心和扫角。</p>
<p>仍需解决26的截面端点语义、20的径向厚度与公差语义，以及其他位置约束。它们不能直接绑到最近节点；参考尺寸132/41也不能无条件作为独立硬等式。完整父求解仍有7个形状自由度；局部数值收敛并不证明唯一性或全局最优。</p>
<h3>具体停在哪个环节</h3>{gaps}<p>26已经观察到尺寸箭头和正确尺度跨度，但测量端点没有表示为CAD测量位置，所以API没有候选可选。20是斜向厚度，现有轴向检测提出了不相容跨度且未核实箭头，必须补充径向/斜向测量语义，不能把它当水平距离硬绑。78则由原先“端点未表示”变成了一个源图可验证候选，并实际进入了求解。</p>
<p>20的+3/0公差还涉及约束优先级：已锁定的同心R180、R160给出径向厚度20，落在[20,23]范围内。不能把21.5公差中点再强制为独立等式，否则会与精确半径冲突。该处仍需实现径向测量与公差区间校验，本轮没有用错误的轴向候选替代。</p></section>
<section id="overlay-gallery"><h2>每轮原始overlay</h2>{galleries}</section>
<section><h2>验证与边界</h2><p>{esc(test_result)}</p><p>GT DXF仅由冻结后的独立评估读取，未将GT CAD坐标发送到API或写入求解。原图、OCR和GT栅格掩膜作为输入的哈希，以及预测未被评估改写的证据保存在run-manifest.json中。</p>
<p><a href="analysis.json">完整数值和流程审计</a> · <a href="round1/run-manifest.json">首轮在线记录</a> · <a href="round2/run-manifest.json">后续优化在线记录</a></p></section></html>'''
    (BASE/'report.html').write_text(report, 'utf8')
    print(BASE/'report.html')


if __name__ == '__main__': main()

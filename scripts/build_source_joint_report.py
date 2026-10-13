"""Build a source/native/reference comparison without modifying predictions."""
import base64
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'runtime/cad-source-joints-20261013'


def read(path):return json.loads(path.read_text('utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def esc(value):return html.escape(str(value),quote=True)


def collect(label,path):
    summary=read(path/'evaluation/summary.json')
    model=read(path/'after/model.json')
    scoped=(path/'after/scope-audit.json').is_file()
    scope=read(path/'after/scope-audit.json') if scoped else None
    validation=model['validation']
    native=scope['retained_native_checks'] if scoped else {
        'radii':validation['exact_radius_validation'],'angles':validation['native_source_angle_validation'],
        'tangencies':validation['strict_relation_validation'],'measurements':validation['native_measurement_validation']}
    contract=validation.get('reconstruction_contract') or {}
    return dict(label=label,path=str(path),prediction_sha256=sha(path/'after/drawing.dxf'),
        summary_sha256=sha(path/'evaluation/summary.json'),scoped=scoped,
        native={k:{v:r.get(v) for v in ('passed','required_count','satisfied_count')} for k,r in native.items()},
        shape=summary['geometry']['registered_shape_diagnostic'],
        native_frame=summary['geometry']['native_coordinate_frame'],
        match=summary['target_acceptance']['primitive_matching'],
        target_checks=summary['target_acceptance']['checks'],
        all_requested_checks_passed=summary['target_acceptance']['all_requested_checks_passed'],
        object_counts=summary['objects']['prediction_filtered_types'],
        remaining_shape_dof=contract.get('remaining_shape_dof'),
        recognized_dimensions=contract.get('recognized_dimensions'),bound_records=contract.get('bound_source_records'),
        unresolved_joints=contract.get('unresolved_joint_count'))


def main():
    from contour_agent.relation_contract import _geometry,_joint_check
    from contour_agent.parametric_solver import _sweep
    import math
    old044=ROOT/'runtime/cad-user-scoped-tread-044-20261013'
    oldhdsa=ROOT/'runtime/cad-controlled-20261008/source-obligation-v92-online-hdsa'
    cases=[collect('044 原有简化范围',old044),collect('044 本轮完整范围',BASE/'044-round4'),
        collect('044 本轮简化范围',BASE/'044-scoped-round4'),collect('HDSA 原有完整范围',oldhdsa),
        collect('HDSA 本轮完整范围',BASE/'hdsa-round1')]
    before=read(ROOT/'runtime/cad-controlled-20261008/source-obligation-v90-online-044/after/model.json')
    after=read(BASE/'044-round4/after/model.json')
    joints=[]
    for a,b in [('g009','g010'),('g008','g009'),('g007','g008'),('g012','g013'),('g013','g014'),('g014','g015')]:
        result={'objects':[a,b]}
        for name,model in [('before',before),('after',after)]:
            lookup={e['id']:e for e in model['entities']};left,right=lookup[a],lookup[b]
            result[name]=_joint_check(_geometry(left),_geometry(right),left['end_node'])
        joints.append(result)
    sweep={e['id']:math.degrees(_sweep(e)) for e in after['entities'] if e['id'] in {'g009','g013'}}
    bindings=read(BASE/'044-round4/after/constraint-bindings.json')
    height=[c for c in bindings['constraints'] if c.get('record_id')=='r024']
    attempts=[]
    for name in ('044-round1','044-round2','044-round3','044-round4','hdsa-round1'):
        manifest=read(BASE/name/'run-manifest.json')
        attempts.append(dict(run=name,status=manifest['status'],
            source_contacts=(manifest.get('source_joint_completion') or {}).get('added_constraint_ids',[]),
            source_contacts_accepted=(manifest.get('source_joint_completion') or {}).get('accepted'),
            recovered_text_records=(manifest.get('isolated_ocr_retry') or {}).get('changed_record_ids',[]),
            code_unchanged=manifest.get('code_unchanged'),failure_type=manifest.get('failure_type')))
    test_log=ROOT/'runtime/pytest-source-joints-final-20261013.txt'
    tests=test_log.read_text('utf16' if test_log.read_bytes().startswith(b'\xff\xfe') else 'utf8',errors='replace')
    test_result=next((line for line in reversed(tests.splitlines()) if ' passed' in line or ' failed' in line),'全量回归进行中')
    result=dict(schema_version='source-joint-iteration-review-v1',cases=cases,left_joints=joints,
        fillet_sweep_deg=sweep,source_height_binding=height,attempts=attempts,test_result=test_result,
        oracle_mask_conditioned=True,fresh_end_to_end_test=False,reference_used_for_prediction=False,
        original_thresholds_changed=False)
    (BASE/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf8')
    def table(rows):
        return '<table><thead><tr>'+''.join('<th>'+esc(s)+'</th>' for s in
            ['结果','LINE/ARC','精确R /角度 /G1 /尺寸','配准最大误差 mm','原坐标误差 mm','长度差 mm','1.5mm完整参数匹配','全检查通过'])+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+esc(s)+'</td>' for s in [c['label'],
                f"{c['object_counts'].get('LINE',0)}/{c['object_counts'].get('ARC',0)}",
                '/'.join(str(c['native'][k]['required_count']) for k in ('radii','angles','tangencies','measurements')),
                f"{c['shape']['conservative_max_error_mm']:.6f}",f"{c['native_frame']['conservative_max_error_mm']:.6f}",
                f"{c['shape']['length_error_mm']:.6f}",f"{c['match']['matched_count']}/{sum(c['object_counts'].values())}",
                '通过' if c['all_requested_checks_passed'] else '未通过'])+'</tr>' for c in rows)+'</tbody></table>'
    joint_table='<table><tr><th>接点</th><th>之前角度残差 °</th><th>当前角度残差 °</th></tr>'+''.join(
        f"<tr><td>{esc(' — '.join(r['objects']))}</td><td>{r['before']['angle_residual_deg']:.6f}</td>"
        f"<td>{r['after']['angle_residual_deg']:.3e}</td></tr>" for r in joints)+'</table>'
    galleries=''
    for c in [cases[0],cases[2],cases[3],cases[4]]:
        p=Path(c['path']);svg=p/'evaluation/overlay.svg'
        url='data:image/svg+xml;base64,'+base64.b64encode(svg.read_bytes()).decode()
        galleries+=f"<figure><figcaption>{esc(c['label'])}：GT洋红，预测青色。原始比较器输出，未重画轮廓。</figcaption><img src='{url}' alt='{esc(c['label'])} overlay'><p><a href='{esc((p/'after/drawing.dxf').as_uri())}'>DXF</a> · <a href='{esc((p/'evaluation/summary.json').as_uri())}'>独立比较</a></p></figure>"
    audit='<table><tr><th>尝试</th><th>状态</th><th>新增相切约束</th><th>隔离OCR恢复</th></tr>'+''.join(
        f"<tr><td>{esc(r['run'])}</td><td>{esc(r['status'])}</td><td>{esc(', '.join(r['source_contacts']))}</td><td>{esc(', '.join(r['recovered_text_records']))}</td></tr>" for r in attempts)+'</table>'
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>源标注接点与联合求解迭代</title>
<style>body{{font:16px/1.7 system-ui;color:#203533;background:#f4f7f6;max-width:1280px;margin:32px auto;padding:20px}}h1,h2{{line-height:1.4}}section{{background:white;padding:24px;margin:20px 0;border:1px solid #d5e1dd}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border:1px solid #d5e1dd;text-align:left}}th{{background:#e6f0eb}}img{{width:100%;background:white}}figure{{margin:24px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#08796d}}.notice{{border-left:5px solid #c48b38;padding-left:16px}}</style>
<h1>044 / HDSA：源标注接点补全、OCR恢复与联合求解</h1>
<p class="notice">本轮修复了 044 左侧相切链，并在线恢复及绑定了 181±1。044 按用户允许简化踏面的范围达到约 1.13 mm 配准形状误差；仍未达到全部 GT 对象参数一致。HDSA 已补齐本轮缺失相切方程，但形状误差仍高于 1.5 mm。所有失败仍保留。</p>
<section><h2>实际改进</h2><ol><li>原图设计关系与拟合接点的局部导数分开：在线模型读取完整原图及两种范围的局部像素，判断 tangent / corner / unknown，并指出支持的标注编号；一致后才提出硬 G1 方程。</li>
<li>两条已确定方向的 LINE 与已知 R 圆角先解析计算切点作为初值，再在原始域内联合求解。所有已知 R、角度、尺寸都保留，切点、圆心和未确定属性允许移动。</li>
<li>扩大竖排文字裁剪，不在原图上画矩形遮挡笔画；可疑 OCR 单条隔离复核，两次都提供旋正像素但使用不同裁剪范围，不能靠 CAD、GT 或人工填入数值。</li>
<li>继承硬相切与新检测到的同一接点方程合并，避免求解通过但因重复关系不能导出；保留旧证据和新检测记录，不降级硬约束。</li>
<li>每个接点记录实际角残差，区分设计模型意见、数值解、原生 DXF 方程验证及独立 GT 精度。失败保留旧输出，不放宽验收阈值。</li></ol></section>
<section><h2>同一源图的独立比较</h2>{table(cases)}
<p>表中约束计数为当前原生 DXF 中已绑定并通过的方程数，不能代表所有标注已覆盖。配准只搜索 D4 方向和平移，不调整尺度；原坐标约定仍未独立证实。原有 0.1 mm 检查保留，1.5 mm 是并列诊断。</p>
<p>044 简化范围只排除用户指定右侧踏面的 R6/R16/R13，不排除左侧圆角、斜线及尺寸。当前简化输出为 10 LINE + 10 ARC，10 个有效半径、4 个斜线角度、14 条 G1、6 个线性尺寸、5 个源轴向约束均通过原生回读。</p>
<p>044 完整范围仍显示 2.44 mm 左右的最大误差，包含尚未完成的右侧踏面；该失败没有从报告中删除。HDSA 最大形状误差由约1.648增至约1.692 mm，新增严格关系并不自动保证 GT 更接近。</p></section>
<section><h2>044 左侧具体修复</h2>{joint_table}<p>g009 / g013 的 R 均严格为15，扫角分别为 {sweep['g009']:.9f}°、{sweep['g013']:.9f}°，符合与水平边、10°及24°斜线相切的几何关系。</p>
<p>r024 由线上两次原图转写恢复为181±1，再线上关联到 g010 与 g012 的水平支撑，求解固定两边高度差为181.0 mm；此数值来自原图转写，不是GT坐标。当前044识别标注绑定为20/36，剩余形状自由度8；HDSA仍有10个形状自由度。</p></section>
<section><h2>仍未解决</h2><ul><li>044 在1.5mm下还有5个GT对象未完整匹配，HDSA还有7个；视觉接近、R精确及G1精确，不能代替圆心、端点、扫角与长度的完整一致。</li>
<li>未绑定尺寸、参考尺寸与结构/公差语义仍需分清几何锚点和独立性，减少剩余位置自由度；不能把全部 OCR 数字直接当作独立等式。</li>
<li>HDSA需继续补充源尺寸的测量平面/切点/圆心关联并优化未知属性。其全对象类型数量匹配，不代表参数已正确。</li>
<li>完整044踏面仍未达标；当前用户简化范围是明确范围投影，末端LINE位置仍由父预测给出，不是新标注推导的踏面。</li></ul></section>
<section id="overlay-gallery"><h2>原始 overlay</h2>{galleries}</section>
<section><h2>在线尝试与验证记录</h2>{audit}<p>044-round3 中重读仍分歧，随后触发重复方程原生拒绝；它保持失败记录。修复后044-round4重新调用在线API、重新绑定、求解、导出并独立评估。</p><p>当前全量回归：{esc(test_result)}</p>
<p>这是冻结原图、GT栅格掩膜及既有初始拓扑的后续阶段开发复测，并非新上传到网页的全流程复测。在线模型提出源语义关系及标注关联，本地解析几何与优化算法产生DXF。GT坐标不进入生成和模型请求，独立比较在预测冻结后运行。不能据此证明真实分割质量或未见图纸泛化。</p>
<p><a href="analysis.json">数值、原生残差、预测哈希与全部尝试</a> · <a href="044-round4/run-manifest.json">044在线与冻结证据</a> · <a href="hdsa-round1/run-manifest.json">HDSA在线与冻结证据</a></p></section></html>'''
    (BASE/'report.html').write_text(report,'utf8')
    print(BASE/'report.html')


if __name__=='__main__':main()

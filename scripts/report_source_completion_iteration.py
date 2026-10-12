"""Report frozen source-completion runs, without reading reference CAD."""
from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import report_source_priority_iteration as base


def read(path):
    return json.loads(path.read_text('utf8')) if path.is_file() else {}


def main():
    project=Path(__file__).resolve().parents[1]
    root=project/'runtime/cad-controlled-20261008'
    output=project/'runtime/cad-source-completion-20261012';output.mkdir(exist_ok=True)
    base.SLOTS=[('fixed-v86','source-fixed-v86-online','前次冻结在线结果'),
        ('fixed-v87','source-fixed-v87-online','第一轮：在线绑定／联合求解／有界拓扑补全'),
        ('fixed-v88','source-fixed-v88-online','第二轮：精确方程可行性恢复／缺失类型修复优先；复用冻结 OCR 和初始轮廓')]
    report=base.assemble(root)
    report.update(schema_version='source-completion-review-v1',fresh_full_end_to_end_runs=0,
        source_inputs='Original image + frozen OCR + GT-derived raster mask; no reference CAD in prediction.',
        scope='Development cases. Online final-stage replay with optional source-driven structural completion.',
        full_test_log=str(project/'runtime/pytest-v88-full.txt'))
    esc=lambda v:html.escape(str(v),quote=True)
    sections=[];images=[]
    for row in report['rows']:
        directory=root/row['run'];after=directory/'after'
        row['source_constraint_completion']=read(after/'source-constraint-completion/completion.json')
        row['source_topology_completion']=read(after/'source-topology-completion/completion.json')
        row['numerical_search']=read(after/'source-multistart.json')
        row['radius_source_retry_transport']=[]
        for number in (1,2):
            retry=after/'source-obligation-radius-retry'/f'radius-targets-attempt-{number:02d}.json'
            if retry.is_file():
                value=read(retry)
                receipt={'stage':f'unresolved_source_radius_retry_{number}',
                    'receipt_scope':'current_run_network_invocation',
                    **{k:value.get(k) for k in ('status','network_requests','http_success','schema_success',
                        'model','elapsed_seconds','ground_truth_sent','cad_coordinates_sent')}}
                row['radius_source_retry_transport'].append(receipt)
        row['current_run_transport_evidence']=[*row.get('transport',[]),*row['radius_source_retry_transport']]
        manifest=read(directory/'run-manifest.json')
        row['input_and_code_integrity']={k:manifest.get(k) for k in (
            'upstream_receipts_unchanged','prediction_unchanged_after_evaluation','prediction_frozen_at')}
        row['input_and_code_integrity']['code_unchanged']=manifest.get('code',{}).get('unchanged')
        overlay=directory/'evaluation/overlay.svg'
        if overlay.is_file():
            data=overlay.read_bytes();uri='data:image/svg+xml;base64,'+base64.b64encode(data).decode('ascii')
            images.append(dict(run=row['run'],sha256=hashlib.sha256(data).hexdigest(),
                path=str(overlay),original_svg_embedded_unchanged=True))
            preview=f'<a href="{esc(overlay.as_uri())}"><img src="{uri}" alt="{esc(row["run"])} overlay"></a>'
        else:preview='<p>Overlay 尚未生成。</p>'
        drawing=after/'drawing.dxf'
        links=(f'<a href="{esc(drawing.as_uri())}">自动输出 DXF</a> · ' if drawing.is_file() else '')
        links+=f'<a href="{esc((directory/"run-manifest.json").as_uri())}">运行清单</a>'
        fields=['current_run_transport_evidence','online_binding','attributes','native','strict_relations','angles','curved_joins',
            'reference_target','source_constraint_completion','source_topology_completion','numerical_search',
            'input_and_code_integrity']
        sections.append(f'<section><h3>{esc(row["run"])}</h3><p>{links}</p>{preview}'
            f'<details><summary>逐项约束、求解、补全与完整性证据</summary><pre>'
            +esc(json.dumps({k:row.get(k) for k in fields},ensure_ascii=False,indent=2))+'</pre></details></section>')
    report['overlay_previews']=images
    latest=[row for row in report['rows'] if row['version']=='fixed-v88']
    report['latest_outcome']={
        'complete_user_target_pass_count':sum(row.get('full_user_target_passed') is True for row in latest),
        'evaluated_development_case_count':len(latest),
        'new_full_end_to_end_case_count':0,
        'no_reference_accuracy_improvement_claimed':True,
        'remaining_failure_scope':'Unresolved source binding, structural construction and relation coverage; numerical feasibility alone is insufficient.'}
    outcome_rows=[]
    for row in latest:
        native=row.get('native') or {};angles=row.get('angles') or {};joins=row.get('curved_joins') or {}
        reference=row.get('reference') or {}
        maximum=reference.get('native_max_mm')
        maximum=f'{maximum:.6f} mm' if isinstance(maximum,(int,float)) else '未评估'
        outcome_rows.append('<tr>'+''.join(f'<td>{esc(value)}</td>' for value in (
            row['case_id'],f'{native.get("exact_radius_count",0)}/{native.get("recognized_radius_count",0)}',
            f'{angles.get("strict_native_satisfied_count",0)}/{angles.get("recognized_count",0)}',
            f'{joins.get("admitted_passed_count",0)}；未确认 {joins.get("unknown_count",0)}',maximum,
            '通过' if row.get('full_user_target_passed') is True else '未完整通过'))+'</tr>')
    tests=project/'runtime/pytest-v88-full.txt'
    tail=tests.read_text('utf8',errors='replace').splitlines()[-3:] if tests.is_file() else []
    report['regression_log_tail']=tail
    extra='<h2 id="overlay-gallery">本轮改进与 overlay</h2><p>紫红色：GT；青绿色：生成或保留的 DXF。叠图来自冻结预测后的独立评估，未改变 SVG 内容。</p>'
    extra+='<p>第二轮完整目标通过数：'+esc(report['latest_outcome']['complete_user_target_pass_count'])+'/'+esc(len(latest))+'。修复了求解续算、旧检查点重新认证和局部补全控制；本轮没有取得明显的 GT 形状误差改善。</p>'
    extra+='<table><thead><tr><th>图纸</th><th>严格半径／已识别 R</th><th>严格角度／已识别角度</th><th>通过的已绑定相切；未确认曲线接点</th><th>GT 原生坐标最大误差</th><th>完整目标</th></tr></thead><tbody>'+''.join(outcome_rows)+'</tbody></table>'
    extra+='<p>HDSA 仍有一个 12°标注未绑定、三个曲线接点关系未确认；局部 LINE 修复不能在新绑定中保留所有既有源约束，因此保留前一部分解。044 的 R6 尚未获得完整认证，R16／R13 的箭头目标和局部对象仍未解决，七个曲线接点关系未确认；局部圆角／分段构造未通过原图支撑或顺序消歧。</p>'
    extra+='<p>修复：超时后保存未经认证的同问题初值并续算；已知约束继续硬锁；有效部分解后重新检查缺失 LINE／ARC，允许局部修改、重新绑定和联合求解；仅在原图与 DXF 回读都通过且既有约束未丢失时替换产物。</p>'
    extra+='<p>第二轮追加：旧检查点的轴向误差不能因历史认证直接通过；原图绑定复核后保留全部方程，以旧几何为初值重新求解。先在有界子预算中恢复方程可行性，再优化原图误差。拓扑名额优先留给未解决的 LINE／ARC 类型冲突，所有新增 API 调用单独记账。</p>'
    extra+='<p>补充 1.5 mm 列只读取实际按 1.5 mm 计算的评估项。旧运行只有 1 mm 数据时显示未评估；0.1 mm 严格评估及半径、角度、相切门槛均保留。</p>'
    extra+='<p>复测范围为两幅开发图，复用冻结 OCR 与初始轮廓，并非从上传开始的全新端到端运行。HTTP 成功不代表几何、参数或覆盖完整；完整通过状态以各项证据为准。</p>'
    extra+='<p>表中“精确 R”仅统计当前发布约束能认证并从 DXF 回读的半径。保留产物未获本轮认证时，该列为零不能解释为所有实际圆弧半径都错误。</p>'
    extra+='<p>完整回归日志：'+esc(' / '.join(tail))+'</p>'+''.join(sections)
    rendered=base.render(report).replace('</main>','</main>'+extra,1)
    rendered=rendered.replace('</style>','section img{display:block;max-width:100%;height:auto;background:#fff}'
        'section{margin:24px 0;padding:16px;background:white;border:1px solid #ccd8d3}'
        'pre{max-height:600px;overflow:auto}</style>',1)
    for row in report['rows']:
        rendered=rendered.replace('href="'+row['run']+'/', 'href="'+(root/row['run']).as_uri()+'/')
    (output/'report.html').write_text(rendered,'utf8')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(output/'report.html')


if __name__=='__main__':main()

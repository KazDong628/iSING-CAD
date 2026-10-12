"""Compare frozen source-obligation iterations and embed independent overlays."""
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
    output=project/'runtime/cad-source-obligations-v92-20261012'
    output.mkdir(exist_ok=True)
    base.SLOTS=[('v88','source-fixed-v88-online','前次冻结在线结果'),
        ('v89','source-obligation-v89-online','第一轮；包含角度定位接口接线失败，保留原始收据'),
        ('v90','source-obligation-v90-online','第二轮；修正角度列表接口并从已认证源部分解在线续算'),
        ('v91','source-obligation-v91-online','HDSA 定向在线复测；暴露关闭结构搜索同时误关角度定位'),
        ('v92','source-obligation-v92-online','HDSA 定向在线复测；角度定位独立使用主模型 API 配置')]
    report=base.assemble(root)
    # V91 intentionally reruns only HDSA's continuous source constraints. Do
    # not invent a failed 044 invocation; its actual latest result is V90.
    report['rows']=[row for row in report['rows'] if row['version'] not in {'v91','v92'} or row['case_id']=='HDSA-65-main']
    report['attempt_count']=len(report['rows'])
    report['scope_conflict']='044 的 R6/R16/R13 仍未完整绑定；原图右端 R16/R13 凸起与现有简化 GT 闭合边界存在范围冲突。不得删掉原图义务来制造 GT 通过。'
    esc=lambda value:html.escape(str(value),quote=True)
    sections=[];previews=[]
    for row in report['rows']:
        directory=root/row['run'];after=directory/'after'
        row['source_constraint_completion']=read(after/'source-constraint-completion/completion.json')
        row['source_topology_completion']=read(after/'source-topology-completion/completion.json')
        row['missing_angle_localization']=read(after/'source-obligation-angle-retry/angle-targets.json')
        row['radius_retry_transport']=[]
        for number in (1,2):
            path=after/'source-obligation-radius-retry'/f'radius-targets-attempt-{number:02d}.json'
            if path.is_file():
                value=read(path)
                row['radius_retry_transport'].append(dict(
                    stage=f'unresolved_radius_retry_{number}',receipt_scope='current_run_network_invocation',
                    **{key:value.get(key) for key in ('status','network_requests','http_success',
                        'schema_success','model','elapsed_seconds')}))
        row['all_current_transport']=[*row.get('transport',[]),*row['radius_retry_transport']]
        manifest=read(directory/'run-manifest.json')
        row['integrity']={key:manifest.get(key) for key in ('upstream_receipts_unchanged',
            'prediction_unchanged_after_evaluation','prediction_frozen_at')}
        row['integrity']['code_unchanged']=manifest.get('code',{}).get('unchanged')
        overlay=directory/'evaluation/overlay.svg'
        preview='<p>尚无独立 overlay；失败仍保留在分母。</p>'
        if overlay.is_file():
            raw=overlay.read_bytes()
            uri='data:image/svg+xml;base64,'+base64.b64encode(raw).decode('ascii')
            previews.append(dict(run=row['run'],path=str(overlay),sha256=hashlib.sha256(raw).hexdigest(),
                original_svg_embedded_unchanged=True))
            preview=f'<a href="{esc(overlay.as_uri())}"><img src="{uri}" alt="{esc(row["run"])}"></a>'
        drawing=after/'drawing.dxf'
        links=f'<a href="{esc(drawing.as_uri())}">自动输出 DXF</a> · ' if drawing.is_file() else ''
        links+=f'<a href="{esc((directory/"run-manifest.json").as_uri())}">运行清单</a>'
        keys=('native','angles','strict_relations','curved_joins','attributes','reference',
            'reference_target','solution_publication_link','all_current_transport','missing_angle_localization',
            'source_constraint_completion','source_topology_completion','integrity')
        sections.append(f'<section><h3>{esc(row["run"])}</h3><p>{links}</p>{preview}'
            '<details><summary>绑定、求解、失败原因和独立评估</summary><pre>'+
            esc(json.dumps({key:row.get(key) for key in keys},ensure_ascii=False,indent=2))+
            '</pre></details></section>')
    latest=[next(row for row in reversed(report['rows']) if row['case_id']==case)
        for case in ('HDSA-65-main','044-main')]
    report.update(schema_version='source-obligation-review-v92',overlay_previews=previews,
        latest_outcome={'complete_user_target_pass_count':sum(row.get('full_user_target_passed') is True for row in latest),
            'development_case_count':len(latest),'fresh_full_end_to_end_case_count':0},
        source_inputs='Original source image + frozen OCR + GT-derived raster mask; no reference DXF in prediction.',
        changes=['Explicit unresolved radius/angle goals; refreshed original-source angle ownership.',
            'Bounded multimodal relocalization of missing angles, including frozen upstream replays.',
            'New source-certified relations may advance even without a new OCR record ID.',
            'One different bounded retry after failure; preserve the last valid DXF.',
            'Exact-radius original-ink seed also supports LINE-ARC-LINE reinsertion.',
            'Resumed numerical initialization does not ratchet the immutable original-source comparator.',
            'Short sibling temporary names prevent Windows path overflow during atomic stage writes.',
            'Missing-angle localization uses the binding API profile even when structural editing is disabled.'],
        regression_log_tail=(project/'runtime/pytest-v92-full.txt').read_text('utf8',errors='replace').splitlines()[-3:])
    report['remaining_problem_summary']={
        'HDSA-65-main':'r035 已对 g002 严格固定为相对竖直轴 12°，41 条方程通过。仍有 13 个形状自由度、17 条未绑定尺寸记录及 3 处未确认曲线接点；已知 R 和角度正确尚未唯一确定全部位置、长度及相切关系。GT 原生最大误差由 V88 的 1.078269 mm 增至 1.623057 mm，仍不通过 1.5 mm。',
        '044-main':'R6/R16/R13 尚未完成严格绑定；7 处曲线接点关系未确认，仍有 16 个形状自由度。输出 10 LINE + 11 ARC，现有 GT 为 10 LINE + 10 ARC；原图右端凸起与简化 GT 的范围冲突继续保留，不能靠删除源标注制造通过。候选还受原图支撑不足、特征顺序歧义及未找到可切分特征阻碍。'}
    outcome=[]
    for row in latest:
        native=row.get('native') or {};angles=row.get('angles') or {};joins=row.get('curved_joins') or {}
        maximum=(row.get('reference') or {}).get('native_max_mm')
        outcome.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in (row['case_id'],
            f'{native.get("exact_radius_count",0)}/{native.get("recognized_radius_count",0)}',
            f'{angles.get("strict_native_satisfied_count",0)}/{angles.get("recognized_count",0)}',
            f'{joins.get("admitted_passed_count",0)}；未确认 {joins.get("unknown_count",0)}',
            f'{maximum:.6f} mm' if isinstance(maximum,(int,float)) else '未评估',
            '通过' if row.get('full_user_target_passed') is True else '未完整通过'))+'</tr>')
    intro='<h2 id="overlay-gallery">V89—V92 在线复测与 overlay</h2>'
    intro+='<p>紫红色为 GT，青绿色为实际输出。原始独立评估 SVG 完整嵌入，未重画或修饰。</p>'
    intro+='<p>修复缺失角度未进入规划、仅补相切关系的候选被阻止、失败后未换候选重试，以及 LINE—ARC—LINE 圆角缺少源图候选分支。已知半径、角度、原图和 DXF 回读门槛均保留。</p>'
    intro+='<p>第一轮暴露缺失角度编号以集合传入、定位接口只接受列表的接线错误，HDSA 该定位请求实际没有发出。第二轮改为规范列表并从经原图和严格方程复核的部分解继续。HDSA 接续 V88，044 接续 V89；原始优化范围未重设。已绑定标注的修改必须涉及待补全接点，不能独占候选名额。</p>'
    intro+='<p>V90 的 HDSA 已定位、绑定缺失 12°，41 条方程联合可行；但续算把已优化初值当成新的原图支撑基准，导致候选在两个最初固定基准都通过时仍被拒绝。V91 对续算只使用经哈希核验的最初域和原始父图作为比较基准，上一轮初值的比较单独保留为诊断，0.035 阈值没有改变。另已复现 256 字符目标路径追加 .tmp 后达到 260 字符的 Windows 写入异常，改用较短的同目录临时文件。V91 仅针对 HDSA 连续约束续算在线复测，未再次启用结构搜索；044 最新结果仍为 V90。</p>'
    intro+='<p>V91 又暴露接口耦合：关闭结构编辑后 editor_provider 为空，缺失角度定位因此也未调用。该轮没有定位请求，不能用它证明角度已补全。V92 解耦这两项能力：角度定位优先使用编辑器配置，未提供时使用主绑定模型配置；再次在线复测。V91 原始部分解和失败收据继续展示。</p>'
    intro+='<p>两幅开发图复用冻结 OCR 和初始轮廓；本轮在线调用绑定、缺失证据定位和局部拓扑流程，并非从上传开始的全新端到端测试。保留产物、有效部分解、完整成功分开报告。参考 GT 只在预测冻结后使用。1.5 mm 为附加诊断，原有 0.1 mm 检查保留。</p>'
    intro+='<table><thead><tr><th>图纸</th><th>严格 R／识别 R</th><th>严格角度／识别角度</th><th>通过相切；未知曲线接点</th><th>GT 原生最大误差</th><th>完整目标</th></tr></thead><tbody>'+''.join(outcome)+'</tbody></table>'
    intro+='<h3>仍未解决的问题</h3>'+''.join('<p><strong>'+esc(case)+'：</strong>'+esc(text)+'</p>'
        for case,text in report['remaining_problem_summary'].items())
    intro+='<p>完整回归：'+esc(' / '.join(report['regression_log_tail']))+'</p>'
    rendered=base.render(report).replace('</main>','</main>'+intro+''.join(sections),1)
    rendered=rendered.replace('</style>','section img{display:block;max-width:100%;height:auto;background:#fff}'
        'section{margin:24px 0;padding:16px;background:#fff;border:1px solid #ccd8d3}'
        'pre{max-height:600px;overflow:auto}</style>',1)
    for row in report['rows']:
        rendered=rendered.replace('href="'+row['run']+'/', 'href="'+(root/row['run']).as_uri()+'/')
    (output/'report.html').write_text(rendered,'utf8')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(output/'report.html')


if __name__=='__main__':main()

"""Read frozen prediction/evaluation receipts; never consume reference CAD."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from urllib.parse import quote

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.report_strict_joint_iteration import collect_slot

SLOTS=[('v52','source-obligations-v52','最终阶段重放／基线'),
       ('v53','source-obligations-v53','启动失败'),
       ('v53','source-obligations-v53b','最终阶段重放'),
       ('v53','full-source-v53','完整重跑'),
       ('v54','source-obligations-v54','最终阶段重放'),
       ('v55','full-source-v55','完整重跑'),
       ('v56','source-obligations-v56','最终阶段重放'),
       ('v57','source-obligations-v57','最终阶段重放'),
       ('v58','source-obligations-v58','最终阶段重放／最新修复')]


def reference_target(summary, tolerance=1.5):
    """Select only a genuinely evaluated threshold, never relabel old receipts."""
    choices=[*summary.get('tolerance_comparison',[]),summary.get('target_acceptance',{})]
    return next((row for row in choices if row.get('tolerance_mm')==tolerance),{})


def assemble(root):
    rows=[]
    for version,prefix,scope in SLOTS:
        for suffix,case in [('hdsa','HDSA-65-main'),('044','044-main')]:
            name=prefix+'-'+suffix
            row=collect_slot(root,version,suffix,case,run_name=name)
            row['declared_online_api_called']=row.get('online_api_called')
            # Older manifests omitted the flag even when fresh stage receipts
            # recorded requests. Keep that declaration separate from evidence.
            row['online_api_called']=any(
                receipt.get('receipt_scope')=='current_run_network_invocation'
                and (receipt.get('network_requests') or 0)>0
                for receipt in row.get('transport',[]))
            row['scope_label']=scope
            manifest_path=root/name/'run-manifest.json'
            if manifest_path.is_file():
                manifest=json.loads(manifest_path.read_text('utf8'))
                row['code_unchanged']=manifest.get('code',{}).get('unchanged')
                row['execution_scope']=manifest.get('execution_scope','full_source_pipeline')
            evaluation=root/name/'evaluation/summary.json'
            if evaluation.is_file():
                data=json.loads(evaluation.read_text('utf8'))
                row['reference_target']=reference_target(data)
                row['reference_target_tolerance_mm']=1.5
                row['reference_target_evaluated']=bool(row['reference_target'])
                row['native_max_error_mm']=data.get('geometry',{}).get('native_coordinate_frame',{}).get('conservative_max_error_mm')
            after=root/name/'after'
            if row.get('strict_audit_status')=='read_back':
                model=json.loads((after/'model.json').read_text('utf8'))
                validation=model.get('validation',{})
                source=validation.get('solved_source_validation',{})
                row['source_priority']=model.get('reconstruction_priority')
                row['mask_diagnostic']=source.get('oracle_mask_validation')
                row['mask_policy_receipt']=source.get('priority_resolution')
                row['mask_used_as_gate']=validation.get('source_mask_fidelity_used_as_acceptance_gate')
                inventory=json.loads((after/'binding-candidates.json').read_text('utf8'))
                bound=set(row.get('attributes',{}).get('bound_record_ids',[]))
                row['unbound_records']=[dict(record_id=r.get('id'),text=r.get('text'),parsed=r.get('parsed'))
                    for r in inventory.get('all_records',inventory.get('records',[]))
                    if r.get('id') not in bound and (r.get('parsed') or {}).get('kind') in
                    {'radius','length','diameter','angle'}]
                recheck=after/'final-binding-attempt/checkpoint-known-source-revalidation.json'
                if recheck.is_file():
                    row['final_known_source_revalidation']=json.loads(recheck.read_text('utf8'))
                    row['files']['known_source_revalidation']=f'{name}/after/final-binding-attempt/checkpoint-known-source-revalidation.json'
            a=row.get('attributes',{});n=row.get('native',{});g=row.get('strict_relations',{})
            angles=row.get('angles',{});joins=row.get('curved_joins',{})
            row['source_goal_complete']=bool(
                row.get('strict_audit_status')=='read_back' and a.get('recognized_dimensions',0)>0 and
                a.get('bound_source_records')==a.get('recognized_dimensions') and a.get('remaining_shape_dof')==0 and
                n.get('all_recognized_radii_exact') is True and g.get('passed') is True and
                angles.get('strict_native_satisfied_count')==angles.get('recognized_count') and
                joins.get('unknown_count')==0 and not joins.get('admitted_failed_count') and
                row.get('audit_inputs_unchanged') is True)
            rows.append(row)
    return dict(schema_version='source-priority-conflicts-v1',created_at=datetime.now(timezone.utc).isoformat(),
        development_case_count=2,held_out=False,rows=rows,attempt_count=len(rows),
        report_reads_gt_dxf=False,report_calls_provider=False,
        priority='Verified original-source nominal values and relations before raster-mask fidelity.',
        reference_thresholds_changed=False,all_failures_retained=True,
        scope_conflict='044 R6/R16/R13 are original-source obligations absent from the existing simplified reference closure. Source completion and reference agreement are separate.',
        regression_logs=['../cad-obligation-20261011/pytest-main-v53.txt',
            '../cad-obligation-20261011/pytest-main-v54.txt','../cad-obligation-20261011/pytest-main-v55.txt',
            '../cad-obligation-20261011/pytest-main-v56.txt','../cad-obligation-20261011/pytest-main-v57.txt',
            '../cad-obligation-20261011/pytest-main-v58.txt'])


def render(report):
    esc=lambda value:html.escape(str(value),quote=True)
    def number(value):return '—' if value is None else f'{value:.6f}' if isinstance(value,float) else str(value)
    def ratio(a,b):return number(a)+' / '+number(b)
    table=[];details=[]
    for row in report['rows']:
        n=row.get('native',{});a=row.get('attributes',{});g=row.get('strict_relations',{})
        angles=row.get('angles',{});joins=row.get('curved_joins',{})
        native=' + '.join(f'{count} {kind}' for kind,count in sorted(n.get('native_types',{}).items())) or '—'
        target=row.get('reference_target',{})
        matching=target.get('primitive_matching',{})
        cells=[row['run'],row['scope_label'],row['status'],native,
            ratio(n.get('exact_radius_count'),n.get('recognized_radius_count')),
            ratio(angles.get('strict_native_satisfied_count'),angles.get('recognized_count')),
            number(g.get('satisfied_count')),number(joins.get('unknown_count')),
            ratio(a.get('bound_source_records'),a.get('recognized_dimensions')),number(a.get('remaining_shape_dof')),
            number(row.get('native_max_error_mm')),
            ratio(matching.get('matched_count'),n.get('native_object_count')),
            '通过' if target.get('checks',{}).get('native_coordinate_shape_within_target') is True else '未通过／未评估',
            '通过' if row['source_goal_complete'] else '未完整通过',
            '通过' if row.get('full_user_target_passed') else '未完整通过']
        table.append('<tr>'+''.join('<td>'+esc(cell)+'</td>' for cell in cells)+'</tr>')
        links=' · '.join(f'<a href="{quote(path,safe="/")}">{esc(label)}</a>' for label,path in row.get('files',{}).items())
        if (Path(__file__).resolve().parents[1]/'runtime/cad-controlled-20261008'/row['run']/'evaluation/overlay.svg').is_file():
            links+=f' · <a href="{quote(row["run"]+"/evaluation/overlay.svg",safe="/")}">GT 对比叠图</a>'
        detail={key:row.get(key) for key in ['online_api_called','declared_online_api_called',
            'transport','online_binding','code_unchanged','solution_publication_link',
            'source_priority','mask_diagnostic','mask_policy_receipt','mask_used_as_gate','unbound_records',
            'angles','curved_joins','attempt_attributes','reference_target','failure_type','last_stage',
            'final_known_source_revalidation']}
        details.append(f'<details><summary>{esc(row["run"])}：证据、冲突和未完成项</summary><p>{links}</p><pre>{esc(json.dumps(detail,ensure_ascii=False,indent=2))}</pre></details>')
    headers=['运行','范围','状态','原生对象','精确R','精确角度','严格G1','未知接点','绑定尺寸','形状DOF','原坐标最大误差mm','1.5mm完整对象参数','1.5mm形状检查','完整源标注目标','原有完整GT目标']
    return ('<!doctype html><meta charset="utf-8"><title>标注优先：冲突修复与独立评估</title>'
        '<style>body{font:15px/1.65 system-ui;margin:32px;color:#183332;background:#f5f8f7}h1{font-size:26px}main{overflow:auto}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #ccd8d3;padding:8px;white-space:nowrap}th{background:#dcece5}details{background:white;padding:14px;margin:16px 0}pre{white-space:pre-wrap;max-height:480px;overflow:auto}a{color:#176d61}</style>'
        '<h1>标注优先：冲突修复与独立评估</h1>'
        '<p>输入仍为原图、OCR 和 GT 派生栅格掩膜。两张图是已曝光开发样例；重复尝试不是独立样本。预测不读取参考 DXF 坐标。</p>'
        '<p>已核实的半径、角度、尺寸和关系为硬约束；掩膜只参与初始拓扑和软定位。原图支撑、原生几何及精确约束仍独立核验。旧掩膜阈值和失败测量保留。</p>'
        '<p>044 原图右侧 R6、R16、R13 与现有简化 GT 闭合不一致；优先完成原图标注，不删除这些义务，也不修改参考数据。原有 0.1mm 检查和 1.5mm 补充检查继续保留。</p>'
        '<p>重放仅重跑局部修复、在线绑定和求解；完整重跑重新执行原图流水线。在线传输、当前导出属性和最后一次候选分开审计。失败启动也计入。</p>'
        '<main><table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join(table)+'</table></main>'
        +''.join(details)+'<p>生成时间：'+esc(report['created_at'])+'</p>')


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'runtime/cad-controlled-20261008'
    report=assemble(root)
    (root/'source-priority-v58-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    (root/'source-priority-v58-review.html').write_text(render(report),'utf8')
    print('Attempts:',report['attempt_count'],'complete source goals:',sum(r['source_goal_complete'] for r in report['rows']))

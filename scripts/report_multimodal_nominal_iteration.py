"""Summarize frozen runs and source-only API probes, never read reference CAD."""
from __future__ import annotations

import json
import html
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import report_source_priority_iteration as report_base


def offline_refinements(root):
    rows=[]
    for version in ('v63','v64','v64b','v64c'):
        for case in ('hdsa','044'):
            directory=root/f'multimodal-nominal-{version}-refine-{case}'
            path=directory/'refinement.json'
            if not path.is_file():continue
            state=json.loads(path.read_text('utf8'))
            row={key:state.get(key) for key in ('status','scope','new_network_requests','publication_approved',
                'selected_numerical_round','known_constraint_values_changed','upstream_binding_unchanged',
                'code_unchanged','optimization_domain_recentered','runner_sha256','continued_from',
                'continuation_selection','continuation_solution_sha256')}
            row.update(run=directory.name,case=case,rounds=state.get('rounds',[]))
            if version=='v63':
                row['diagnostic_limitation']='Helper used flattened domain as its source comparator; the final full-source check below remained independent and correctly rejected HDSA. V64 fixes the helper comparator.'
            evaluation=directory/'evaluation-run-b/evaluation/summary.json'
            if evaluation.is_file():
                value=json.loads(evaluation.read_text('utf8'))
                row['independent_post_frozen_evaluation']={
                    'status':value.get('status'),
                    'native_max_error_mm':value.get('geometry',{}).get('native_coordinate_frame',{}).get('conservative_max_error_mm'),
                    'target_acceptance':value.get('target_acceptance'),
                    'overlay':str(evaluation.parent/'overlay.svg')}
            failures=[]
            for name in ('evaluation-run',):
                failed=directory/name/'evaluation/summary.json'
                if failed.is_file():
                    value=json.loads(failed.read_text('utf8'))
                    failures.append({key:value.get(key) for key in ('status','reason','failure_type')})
            row['retained_evaluation_startup_failures']=failures
            rows.append(row)
    return rows


def main():
    root=Path(__file__).resolve().parents[1]/'runtime/cad-controlled-20261008'
    report_base.SLOTS=[*report_base.SLOTS,
        ('v59','multimodal-nominal-v59','新增多模态角度定位／最终阶段重放'),
        ('v60','multimodal-nominal-v60','有界连接重试／最终阶段重放'),
        ('v61','multimodal-nominal-v61','冻结代码缺少评估脚本／启动失败'),
        ('v61','multimodal-nominal-v61b','多尺度原图重读、已知约束保留／最终阶段重放'),
        ('v62','multimodal-nominal-v62','保留文字遮蔽、原图笔画参与已知属性联合求解／最终阶段重放')]
    report=report_base.assemble(root)
    probes=[]
    for suffix in ['hdsa','044']:
        directory=root/'multimodal-nominal-v59-probes'/suffix
        original=directory/'probe-audit.json'
        if not original.exists():continue
        receipt=json.loads(original.read_text('utf8'))
        final=directory/('fresh-short-line-recheck.json' if suffix=='hdsa' else 'finite-association-recheck.json')
        fresh=json.loads(final.read_text('utf8')) if final.exists() else {}
        probes.append({'case':suffix,'scope':'source-only online localization; no predicted DXF',
            'online':receipt.get('online'),'initial_observation_count':len(receipt.get('local_angle_observations',[])),
            'later_local_candidate_count':len(fresh.get('candidates',[])),
            'later_recheck_network_requests':fresh.get('new_network_requests'),
            'ground_truth_used':False})
    for row in report['rows']:
        directory=root/row['run']/'after/multimodal-angle-source'
        receipt=directory/'angle-targets.json'
        if receipt.exists():
            value=json.loads(receipt.read_text('utf8'))
            row['multimodal_angle_source']={key:value.get(key) for key in
                ['status','error_code','network_requests','http_success','schema_success','elapsed_seconds',
                 'unknown_record_ids','input_images','proposals','attempts','maximum_attempts','total_timeout_seconds']}
            row['multimodal_angle_source']['retained_verified_hint_record_ids']=value.get('retained_verified_hint_record_ids')
            row['multimodal_angle_source']['retained_existing_verified_hint_record_ids']=value.get('retained_existing_verified_hint_record_ids')
            row['files']['angle_source_receipt']=row['run']+'/after/multimodal-angle-source/angle-targets.json'
        attempt=root/row['run']/'after/final-binding-attempt'
        solution=attempt/'parametric-solution.json'
        if solution.exists():
            value=json.loads(solution.read_text('utf8'))
            angle=value.get('strict_ocr_angle_contract') or {}
            source=attempt/'solved-source-validation.json'
            check=json.loads(source.read_text('utf8')) if source.exists() else {}
            guard=root/row['run']/'after/binding-publication-guard.json'
            publication=json.loads(guard.read_text('utf8')) if guard.exists() else {}
            validation=attempt/'parametric-export/validation.json'
            native=json.loads(validation.read_text('utf8')) if validation.is_file() else {}
            row['last_numerical_candidate']={
                'scope':'Last attempted solution; published DXF metrics in the main row remain separate.',
                'numerically_accepted':value.get('accepted'),
                'strict_angle_contract':angle,
                'source_gate_passed':check.get('passed'),
                'source_gate_reasons':check.get('reasons'),
                'replacement_gate':publication.get('replacement_gate'),
                'dxf_native_readback_performed':all(native.get(key,{}).get('passed') is True and
                    native.get(key,{}).get('dxf_readback_performed') is True for key in
                    ('exact_radius_validation','strict_relation_validation','native_measurement_validation','native_source_angle_validation')),
            }
    report['source_only_online_probes']=probes
    report['offline_fixed_binding_refinements']=offline_refinements(root)
    corners=[]
    for name in ('multimodal-nominal-v63-source-corner','multimodal-nominal-v63-source-corner-b',
                 'multimodal-nominal-v63-source-corner-c','multimodal-nominal-v64-source-corner'):
        directory=root/name;entry={'run':name,'scope':'Local source-only topology/binding probe; no online call or published DXF.'}
        for filename in ('execution.json','known-obligations.json'):
            path=directory/filename
            if path.is_file():entry[filename]=json.loads(path.read_text('utf8'))
        binding=directory/'known-obligations/cand-edit-01-annotated-fillet/constraint-bindings.json'
        if binding.is_file():
            value=json.loads(binding.read_text('utf8'))
            entry['local_radius_records']=[{'record_id':r.get('record_id'),'value':r.get('value')} for r in value['constraints'] if r['kind']=='radius']
            entry['local_angle_records']=[{'record_id':r.get('record_id'),'value':r.get('value')} for r in value['constraints'] if r['kind']=='angle']
        if directory.exists():corners.append(entry)
    report['source_corner_probes']=corners
    report['regression_logs']+=['../cad-obligation-20261011/pytest-multimodal-v59-full.txt',
        '../cad-obligation-20261011/pytest-multimodal-v60-full.txt',
        '../cad-obligation-20261011/pytest-multimodal-v61-full.txt',
        '../cad-obligation-20261011/pytest-multimodal-v62-focused.txt',
        '../cad-obligation-20261011/pytest-multimodal-v63-full.txt',
        '../cad-obligation-20261011/pytest-multimodal-v64-focused.txt',
        '../cad-obligation-20261011/pytest-multimodal-v64-focused-b.txt',
        '../cad-obligation-20261011/pytest-multimodal-v64-focused-c.txt',
        '../cad-obligation-20261011/pytest-multimodal-v64-iteration.txt',
        '../cad-obligation-20261011/pytest-multimodal-v64-full.txt',
        '../cad-obligation-20261011/pytest-multimodal-v64-arrow-ownership.txt',
        '../cad-obligation-20261011/pytest-multimodal-v64-full-b.txt']
    title='半径与斜线角度优先：多模态定位、绑定与联合求解'
    page=report_base.render(report).replace('标注优先：冲突修复与独立评估',title)
    extra='<p>新阶段先检查半径和斜线角度。模型仅提出源图像素位置，局部笔画、双箭头、有限对象归属和最终原生方程分别核验。源图定位探测与正式 DXF 重放分别记录；重复尝试不是独立样本。</p>'
    extra+='<p>最新结果区分实际发布与未发布候选；后面的历史主表仅列正式 DXF 重放。离线优化和局部 R6 探测单列，不能算作新增在线或端到端成功。原图支撑失败及全部历史尝试继续保留。</p>'
    compact=[]
    for row in report['rows']:
        if not row['run'].startswith('multimodal-nominal-v62-'):continue
        case='hdsa' if row['run'].endswith('-hdsa') else '044'
        compact.append([case,'在线最终阶段重放／已发布',
            str(row.get('native',{}).get('exact_radius_count'))+'/'+str(row.get('native',{}).get('recognized_radius_count')),
            str(row.get('angles',{}).get('strict_native_satisfied_count'))+'/'+str(row.get('angles',{}).get('recognized_count')),
            row.get('native_max_error_mm'),'完整目标未通过'])
        drafts=[value for value in report['offline_fixed_binding_refinements'] if value['case']==case and
                value['status']=='finished' and value.get('selected_numerical_round') is not None]
        if drafts:
            def source_score(value):
                row=next(r for r in value['rounds'] if r['round']==value['selected_numerical_round'])
                return (bool(row['source_gate_passed']),row['stroke_supported_fraction'])
            draft=max(drafts,key=source_score)
            chosen=next(r for r in draft['rounds'] if r['round']==draft['selected_numerical_round'])
            compact.append([case,'固定在线绑定／离线未发布候选',
                str(chosen['native_radius_checks'].get('required_count'))+'/'+str(row.get('native',{}).get('recognized_radius_count')),
                str(chosen['native_angle_checks'].get('required_count'))+'/'+str(row.get('angles',{}).get('recognized_count')),
                draft.get('independent_post_frozen_evaluation',{}).get('native_max_error_mm'),
                '完整源图检查通过；完整目标未通过' if chosen['source_gate_passed'] else '完整源图检查未通过'])
    extra+='<p>最新结果：在线模型负责原图理解与绑定；数值继续优化由本地确定性求解器执行，没有本地 GPT 参与。</p><table><tr>'+''.join(
        '<th>'+html.escape(s)+'</th>' for s in ('案例','范围','精确半径','精确角度','原坐标最大误差 mm','状态'))+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(cell))+'</td>' for cell in row)+'</tr>' for row in compact)+'</table>'
    for key,label in [('offline_fixed_binding_refinements','固定在线绑定后的离线数值迭代'),('source_corner_probes','R6 局部补入与旧约束继承')]:
        for value in report[key]:
            extra+='<details><summary>'+html.escape(value['run'])+'：'+label+'</summary><pre>'+html.escape(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'
    for row in report['rows']:
        if row.get('multimodal_angle_source'):
            extra+='<details><summary>'+html.escape(row['run'])+'：多模态角度定位</summary><pre>'+html.escape(json.dumps(row['multimodal_angle_source'],ensure_ascii=False,indent=2))+'</pre></details>'
        if row.get('last_numerical_candidate'):
            extra+='<details><summary>'+html.escape(row['run'])+'：最后数值候选与最终发布分开核对</summary><pre>'+html.escape(json.dumps(row['last_numerical_candidate'],ensure_ascii=False,indent=2))+'</pre></details>'
    page=page.replace('<main>',extra+'<main>',1)
    (root/'multimodal-nominal-v64-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    (root/'multimodal-nominal-v64-review.html').write_text(page,'utf8')
    print('DXF attempt slots:',len(report['rows']),'source-only online probes:',len(probes))


if __name__=='__main__':main()

"""Audit frozen source-only solves and independent evaluations, without GT CAD."""
from __future__ import annotations

import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import report_source_priority_iteration as base


def read(path):
    return json.loads(path.read_text('utf8')) if path.is_file() else {}


def pipeline_gaps(directory):
    after=directory/'after'
    bindings=read(after/'constraint-bindings.json')
    inventory=read(after/'binding-candidates.json')
    solution=read(after/'parametric-solution.json')
    bound={row.get('record_id') for row in bindings.get('constraints',[])}
    length={row.get('record_id'):row for row in inventory.get('length_source_observations',[])}
    radius={row.get('record_id'):row for row in bindings.get('radius_binding_coverage',{}).get('unresolved',[])}
    gaps=[]
    for record in inventory.get('all_records',[]):
        kind=record.get('parsed',{}).get('kind');rid=record['id']
        if kind not in {'radius','angle','length','diameter'} or rid in bound:
            continue
        candidates=[row for row in inventory.get('all_candidates',[]) if row.get('record_id')==rid]
        decisions=[row for row in bindings.get('bindings',[]) if row.get('record_id')==rid]
        gaps.append(dict(record_id=rid,text=record.get('text'),kind=kind,
            local_candidate_count=len(candidates),locally_reliable_count=sum(c.get('local_reliable') is True for c in candidates),
            source_detection_status=length.get(rid,{}).get('status',radius.get(rid,{}).get('reason')),
            decisions=[dict(accepted=r.get('accepted'),reason=r.get('reason'),source=r.get('source')) for r in decisions],
            endpoint_station_candidates=length.get(rid,{}).get('endpoint_station_candidates')))
    required={tuple(row.get('entities',[])) for row in bindings.get('constraints',[]) if row['kind']=='tangent'}
    unknown=[]
    for relation in inventory.get('relations',[]):
        if relation.get('type')!='tangent' or tuple(relation.get('entities',[])) in required:
            continue
        evidence=relation.get('evidence',{})
        unknown.append(dict(entities=relation['entities'],reason=evidence.get('reason'),
            original_image_observed_deviation_degrees=evidence.get('observed_deviation_degrees'),
            source_verified=evidence.get('verified'),
            side_failure_reasons=[row.get('reason') for row in evidence.get('sides',[])],
            tangency_is_not_implicitly_assumed=True))
    attempts=[]
    for suffix in ('','final-binding-attempt'):
        folder=after/suffix
        sol=read(folder/'parametric-solution.json');check=read(folder/'solved-source-validation.json')
        if not sol:continue
        diag=sol.get('diagnostics',{})
        attempts.append(dict(stage=suffix or 'published',numerically_accepted=sol.get('accepted'),
            original_source_gate_passed=check.get('passed'),source_gate_reasons=check.get('reasons'),
            initial_fit_role=diag.get('initial_fit_role'),
            original_image_stroke_prior=diag.get('original_image_stroke_prior'),
            numerical_feasible_checkpoint=diag.get('numerical_feasible_checkpoint'),
            remaining_shape_dof=diag.get('remaining_shape_dof')))
    diag=solution.get('diagnostics',{})
    return dict(unbound_source_records=gaps,unadmitted_curved_joints=unknown,
        constraint_identifiability={key:diag.get(key) for key in ('variable_count','constraint_rank','remaining_dof',
            'rigid_gauge_dof','remaining_shape_dof','rank_excludes_soft_prior')},
        published_initial_fit_role=diag.get('initial_fit_role'),numerical_attempts=attempts,
        checkpoint_recheck=read(after/'trusted-preflight/certificate.json'),
        full_resumed_source_certificate=read(after/'resumed-known-source-certificate.json'),
        pending_resumed_source_repair=read(after/'resumed-pending-source-repair.json'),
        radius_hypothesis_repair=read(after/'source-radius-hypotheses/source-radius-hypotheses.json'),
        complete_after_source_constraint_stage=read(after/'source-constraint-completion/completion.json'),
        scope='Published attributes and last numerical attempt are separate; missing equations are not optimizer infeasibility.')


def main():
    root=Path(__file__).resolve().parents[1]/'runtime/cad-controlled-20261008'
    base.SLOTS=[('v69','multimodal-nominal-v69-online','既有基线／在线最终阶段重放'),
        ('v71','source-initialization-v71-online','去除旧拟合坐标惩罚／在线最终阶段重放'),
        ('v72','source-initialization-v72-online','补齐部分数值阶段原图观测／在线最终阶段重放'),
        ('v73','source-initialization-v73-online','各阶段原图观测＋保存可行点／从 V69 继续'),
        ('v74','source-initialization-v74-online','有限对象笔画距离＋圆角复核证据传递／从 V69 继续'),
        ('v75','source-initialization-v75-online','逐项恢复已验证约束＋圆角待修复状态继续求解／从 V69 继续')]
    report=base.assemble(root)
    report.update(schema_version='initial-fit-and-stage-consistency-review-v1',
        report_reads_gt_dxf=False,report_calls_provider=False,
        fresh_full_end_to_end_runs=0,cloud_gpt_equivalence_established=False,
        changes=['Original fitted CAD coordinates are initialization only when original ink is observed',
            'Original image observations propagate into topology preflight, line preflight, radius trial, radius checkpoint final solve and enlarged constraint solve',
            'Accepted radius trial seeds the next radius trial in the same immutable domain',
            'Two-scale source-circle derivative supplements polynomial tangent evidence; no radius binding from this measurement',
            'Bounded source-first optimization saves complete feasible points; exhausted search still needs all final native and original-image gates',
            'Radius carry and scoped rechecks receive original image pixels and canonical OCR records',
            'Native ink objective measures finite LINE and ARC extent; matching may migrate only across shared topological nodes',
            'One failed resumed obligation does not abort ordinary joint solving of individually revalidated constraints; complete checkpoint reuse stays forbidden',
            'Uniquely source-owned constructed fillet with observed local ink failure becomes a bounded fixed-R retry, never automatic acceptance'],
        historical_report='source-checkpoint-v69-review.html',new_attempt_slots=10,
        regression_logs=['pytest-v72-full.txt','pytest-v73-full.txt','pytest-v74-full.txt','pytest-v75-full.txt','pytest-v75-focused.txt'])
    report['offline_diagnostics']=[]
    for name in ('source-initialization-v72-refine-hdsa','source-initialization-v74-refine-044',
                 'source-initialization-v75-refine-hdsa','source-conditioning-v76-refine-hdsa',
                 'source-conditioning-v76-refine-044'):
        item=read(root/name/'refinement.json')
        if item:
            item['run']=name
            item['production_code']=not name.startswith('source-conditioning')
            item['selected_local_source_certificate']=read(root/name/'selected-source-certificate.json')
            item['post_freeze_evaluation']=read(root/name/'post-frozen-evaluation/evaluation/summary.json')
            report['offline_diagnostics'].append(item)
    for row in report['rows']:
        directory=root/row['run'];manifest=read(directory/'run-manifest.json')
        row['pipeline_gaps']=pipeline_gaps(directory)
        row['snapshot_integrity']={key:manifest.get(key) for key in ('upstream_receipts_unchanged',
            'prediction_unchanged_after_evaluation')}
        row['snapshot_integrity']['code_unchanged']=manifest.get('code',{}).get('unchanged')
        row['upstream_run']=manifest.get('upstream_run')
        row['online_angle_localization']={key:read(directory/'after/multimodal-angle-source/angle-targets.json').get(key)
            for key in ('network_requests','http_success','schema_success','status')}
        row['online_radius_localization']=[{**{key:read(path).get(key) for key in
            ('network_requests','http_success','schema_success','status','unknown_record_ids','omitted_record_ids',
             'request_max_output_tokens','usage','ground_truth_sent')},'receipt':path.relative_to(directory).as_posix()}
            for path in sorted((directory/'after/source-obligation-radius-retry').glob('radius-targets-attempt-*.json'))]
        row['original_radius_pixel_rechecks']=[]
        for path in sorted((directory/'after/source-obligation-radius-retry').glob('radius-targets-local-*.json')):
            local=read(path)
            row['original_radius_pixel_rechecks'].append(dict(receipt=path.relative_to(directory).as_posix(),
                verified_record_ids=local.get('verified_record_ids'),
                failed=[dict(record_id=r.get('record_id'),text=r.get('text'),status=r.get('leader_status'),
                    api_source_proposal_count=r.get('source_arrow_proposal_count'),
                    locally_verified_proposals=r.get('locally_verified_source_arrow_proposal_count'),
                    candidate_issues=[h.get('directed_verification_issues') for h in r.get('leader_hypotheses_requiring_review',[])])
                    for r in local.get('records',[]) if r.get('record_id') not in local.get('verified_record_ids',[])]))
    title='V71–V75：初始解角色、有限图元、恢复流程与未解约束'
    page=base.render(report).replace('标注优先：冲突修复与独立评估',title)
    extra='<p>本轮新测试为两张开发图的十次在线最终阶段重放，使用 GT 派生掩膜，复用冻结 OCR 与上游拓扑。它们不是十张独立样本，也不是完整网页端到端测试。基线 V69 仅作对照；V71/V72 从 V62 恢复，V73/V74/V75 从 V69 的状态继续，因此不能声称各行只差一个因素。另有五组离线数值诊断，未调用新 API，单独列出且不计作在线成功；其中两组来自未纳入生产代码的数值尺度实验。</p>'
    extra+='<p>已绑定半径、角度和相切方程保持精确验收。移除的是优化目标中的旧拟合坐标惩罚，不改变约束、变量范围、原图支撑门槛及参考误差检查。GT 只在预测冻结后由独立评估进程读取。</p>'
    extra+='<p>对象数量正确不等于约束充分。需同时确认循环顺序、连接方式、弧分支、尺寸测量位置与约束独立性；仍有形状自由度时，不能声称仅凭已知标注恢复唯一 GT。</p>'
    extra+='<p>原图圆形笔画的测量半径仅用于检查切向稳定性，不进入标注半径约束。真实案例是否补回关系，以逐项原图证据和原生 DXF 回读为准。</p>'
    extra+='<p><a href="source-checkpoint-v69-review.html">此前全部尝试和失败记录</a></p>'
    for item in report['offline_diagnostics']:
        state={k:item.get(k) for k in ('run','status','production_code','new_network_requests','ground_truth_used',
            'selected_numerical_round','known_constraint_values_changed','publication_approved','code_unchanged')}
        state['rounds']=[{k:r.get(k) for k in ('round','numerically_accepted','source_gate_passed',
            'stroke_supported_fraction','source_gate_reasons','solver_status')} for r in item.get('rounds',[])]
        state['selected_local_source_failures']=[{k:r.get(k) for k in ('record_id','kind','reason')}
            for r in item.get('selected_local_source_certificate',{}).get('checks',[]) if not r.get('passed')]
        state['post_frozen_reference_geometry']=item.get('post_freeze_evaluation',{}).get('geometry')
        extra+='<details><summary>'+html.escape(item['run'])+'：离线诊断，非在线完成</summary><pre>'+html.escape(json.dumps(state,ensure_ascii=False,indent=2))+'</pre></details>'
    for row in report['rows']:
        detail={key:row.get(key) for key in ('run','snapshot_integrity','online_angle_localization',
            'online_radius_localization','original_radius_pixel_rechecks','pipeline_gaps')}
        extra+='<details><summary>'+html.escape(row['run'])+'：具体流程缺口</summary><pre>'+html.escape(json.dumps(detail,ensure_ascii=False,indent=2))+'</pre></details>'
    page=page.replace('<main>',extra+'<main>',1)
    (root/'source-initialization-v75-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    (root/'source-initialization-v75-review.html').write_text(page,'utf8')
    print(json.dumps({'rows':len(report['rows']),'new_attempt_slots':10,
        'source_goals_complete':sum(row['source_goal_complete'] for row in report['rows'])}))


if __name__=='__main__':main()

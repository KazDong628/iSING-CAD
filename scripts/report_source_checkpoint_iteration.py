"""Report frozen online replays and separate offline candidates without reading GT CAD."""
from __future__ import annotations
import hashlib,html,json,sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import report_source_priority_iteration as base


def read(path):
    return json.loads(path.read_text('utf8')) if path.is_file() else {}


def assemble(root):
    base.SLOTS=[(f'v{v}',f'multimodal-nominal-v{v}-online',
        '在线最终阶段重放／复用上游 OCR 与拓扑') for v in (65,66,67,69)]
    report=base.assemble(root)
    prior=read(root/'multimodal-nominal-v64-review.json')
    report.update(schema_version='source-checkpoint-and-glyph-review-v1',
        historical_attempt_count=prior.get('attempt_count'),historical_report='multimodal-nominal-v64-review.html',
        fresh_full_end_to_end_runs=0,cloud_gpt_equivalence_established=False,
        current_changes=['Original ink strip centres and native primitive residuals',
            'Independent finite-window source tangent remeasurement',
            'SHA-verified source obligations retained only on untouched uniquely identified objects',
            'Original glyph atlas for all page records; no OCR value drawn into images'],
        experimental_change_not_promoted='V68 widened numerical ink association only; source support regressed and the continuation incumbent was retained.',
        regression_logs=['pytest-v67-full.txt','pytest-v69-full.txt','pytest-v69-atlas.txt','pytest-v69-atlas-b.txt',
                         'pytest-v67-focused.txt','pytest-v67-focused-b.txt','pytest-v67-focused-c.txt'])
    for row in report['rows']:
        directory=root/row['run'];after=directory/'after'
        manifest=read(directory/'run-manifest.json')
        row['manifest_finished']=manifest.get('status') in {'completed','completed_with_unresolved_attributes','failed'}
        row['snapshot_integrity']={k:manifest.get(k) for k in
            ('upstream_receipts_unchanged','prediction_unchanged_after_evaluation')}
        row['snapshot_integrity']['code_unchanged']=manifest.get('code',{}).get('unchanged')
        angle=read(after/'multimodal-angle-source/angle-targets.json')
        row['online_angle_localization']={k:angle.get(k) for k in
            ('status','network_requests','http_success','schema_success','elapsed_seconds','error_code')}
        row['glyph_tiles']=[]
        for bp in (after/'constraint-bindings.json',after/'final-binding-attempt/constraint-bindings.json'):
            value=read(bp)
            row['glyph_tiles']+=[{**tile,'binding_stage':str(bp.relative_to(directory))}
                                 for tile in value.get('provider',{}).get('source_glyph_tiles',[])]
        repairs=read(after/'source-object-repairs/known-obligations.json')
        row['local_edit_known_obligations']={k:{field:v.get(field) for field in
            ('passed','required_count','failures')} for k,v in repairs.items() if isinstance(v,dict)}
        attempt=after/'final-binding-attempt'
        value=read(attempt/'parametric-solution.json');source=read(attempt/'solved-source-validation.json')
        export=read(attempt/'parametric-export/validation.json')
        candidate=attempt/'parametric-export/drawing.dxf';published=after/'drawing.dxf'
        same_export=bool(candidate.is_file() and published.is_file() and
            hashlib.sha256(candidate.read_bytes()).digest()==hashlib.sha256(published.read_bytes()).digest())
        row['last_numerical_candidate']={
            'published_export_bytes_match':same_export,
            'scope':'Last numerical attempt; actual published DXF remains the main table row.',
            'numerically_accepted':value.get('accepted'),
            'strict_angle_contract':value.get('strict_ocr_angle_contract'),
            'source_gate_passed':source.get('passed'),'source_gate_reasons':source.get('reasons'),
            'native_readback_passed':bool(export) and all(export.get(k,{}).get('passed') is True and
                export.get(k,{}).get('dxf_readback_performed') is True for k in
                ('exact_radius_validation','strict_relation_validation','native_measurement_validation','native_source_angle_validation'))}
        value=read(after/'constraint-bindings.json')
        row['source_text_mismatches']=[v for v in value.get('bindings',[]) if v.get('reason')=='source_observed_text_mismatch']
        comparison=read(directory/'evaluation/comparison.json')
        localization=comparison.get('error_localization',{})
        row['post_export_error_localization']={
            'scope':'Already frozen registered GT comparison for diagnosis only; never supplied to prediction or provider.',
            'worst_reference_curves':[{k:v.get(k) for k in ('entity_id','type','layer','assumption_layer','max_error_mm')}
                for v in localization.get('worst_reference_curves',[])[:3]],
            'worst_prediction_curves':[{**{k:v.get(k) for k in ('entity_id','type','max_error_mm')},
                'native_radius':v.get('parameters',{}).get('radius')}
                for v in localization.get('worst_prediction_curves',[])[:3]]}
    offline=[]
    for version in (65,66,68,70):
        directory=root/f'multimodal-nominal-v{version}-refine-hdsa';state=read(directory/'refinement.json')
        if not state:continue
        selected=state.get('selected_numerical_round')
        item={k:state.get(k) for k in ('status','scope','new_network_requests','publication_approved',
             'selected_numerical_round','known_constraint_values_changed','upstream_binding_unchanged',
             'code_unchanged','optimization_domain_recentered','continued_from')}
        item['run']=directory.name
        item['promoted_to_current_package']=version in (65,66)
        item['experimental_association_scope']=('entirely_unobserved_entities_only' if version==70 else
            'all_entities_wider_search' if version==68 else 'normal_bounded_source_observations')
        item['rounds']=[{**{k:v.get(k) for k in ('round','numerically_accepted','source_gate_passed',
            'source_gate_reasons','stroke_supported_fraction','all_known_constraint_ids_preserved')},
            'native_radius_count':v.get('native_radius_checks',{}).get('required_count'),
            'native_angle_count':v.get('native_angle_checks',{}).get('required_count'),
            'native_tangent_count':v.get('native_relation_checks',{}).get('required_count')}
            for v in state.get('rounds',[])]
        evaluation=read(directory/'post-frozen-evaluation/evaluation/summary.json')
        item['independent_post_frozen_evaluation']={
            'performed':bool(evaluation),'native_max_error_mm':evaluation.get('geometry',{}).get(
                'native_coordinate_frame',{}).get('conservative_max_error_mm'),
            'target_acceptance':evaluation.get('target_acceptance'),
            'freeze_audit':read(directory/'post-frozen-evaluation/freeze-audit.json')}
        if type(selected) is int:
            item['selected_source_validation']=read(directory/f'round-{selected:02d}/solved-source-validation.json')
            # Preserve reasons/thresholds without duplicating dense per-point measurements.
            item['selected_source_validation']={k:v for k,v in item['selected_source_validation'].items()
                                                 if k in ('passed','reasons','stroke_support','baseline_stroke_support','thresholds')}
        offline.append(item)
    report['offline_refinements']=offline
    report['new_attempt_count']=len(report['rows'])
    report['total_formal_attempt_slots_including_history']=len(report['rows'])+(prior.get('attempt_count') or 0)
    return report


def render(report):
    esc=lambda x:html.escape(str(x),quote=True)
    title='V65–V69：源证据保留、文字放大与实际在线结果'
    page=base.render(report).replace('标注优先：冲突修复与独立评估',title)
    extra='<p><b>未证明达到云端 GPT 的完整处理效果。</b>本轮是两个已曝光开发案例的在线最终阶段重放；复用上游 OCR 与拓扑，不计作全新网页端到端测试。模型调用、正式 DXF、未发布数值候选分别记录。</p>'
    extra+='<p>局部编辑后只保留同一原图、同一 OCR、同一对象几何的旧证据；哈希失配、对象改变或新的正向矛盾会拒绝继承。旧证据不算本次 API 重新确认。新增字形图对本页全部标注给出原图像素放大，不用 OCR 文字重画图像。</p>'
    extra+='<p>数值求解保留精确半径、角度和已绑定相切方程。V68 扩大笔画关联的实验导致源图支撑下降，未合入；V70 只对完全缺少观测的对象单独试验，亦与当前代码及在线结果分开。GT 仍仅在输出冻结后独立评估，所有失败保留，阈值没有放宽。</p>'
    extra+=f'<p>本轮 {report["new_attempt_count"]} 个正式尝试槽位；历史 {esc(report["historical_attempt_count"])} 个，开发样本始终只有 2 张。<a href="multimodal-nominal-v64-review.html">历史完整报告</a></p>'
    for row in report['rows']:
        info={k:row.get(k) for k in ('run','status','snapshot_integrity','online_angle_localization','glyph_tiles',
            'local_edit_known_obligations','source_text_mismatches','last_numerical_candidate','post_export_error_localization')}
        extra+='<details><summary>'+esc(row['run'])+'：本轮改进核验</summary><pre>'+esc(json.dumps(info,ensure_ascii=False,indent=2))+'</pre></details>'
    extra+='<h2>离线数值继续迭代：不计作新增在线成功</h2>'
    for value in report['offline_refinements']:
        extra+='<details><summary>'+esc(value['run'])+'</summary><pre>'+esc(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'
    return page.replace('<main>',extra+'<main>',1)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'runtime/cad-controlled-20261008'
    report=assemble(root)
    (root/'source-checkpoint-v69-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    (root/'source-checkpoint-v69-review.html').write_text(render(report),'utf8')
    print(json.dumps({'new_attempts':len(report['rows']),
        'finished':sum(row['manifest_finished'] for row in report['rows']),
        'source_goals_complete':sum(row['source_goal_complete'] for row in report['rows'])}))

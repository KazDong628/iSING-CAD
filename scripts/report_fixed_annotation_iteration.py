"""Report frozen online tests and explicitly offline fixed-annotation solves."""
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


def overlay_gallery(report, root):
    """Embed existing evaluation SVG bytes; never recompute or transform CAD."""
    esc=lambda value:html.escape(str(value),quote=True)
    previews=[]
    groups=[]
    specs=[('最新在线复测 V86', [row for row in report['rows'] if row['version']=='fixed-v86']),
           ('冻结在线标注后的离线求解对照',report['offline_fixed_annotation_diagnostics']),
           ('V77 原结果与 V82 中间复测',[row for row in report['rows'] if row['version']!='fixed-v86'])]
    for title,rows in specs:
        cards=[]
        for row in rows:
            offline='drawing' in row
            path=(Path(row['overlay']) if offline else
                  root/row['run']/'evaluation/overlay.svg')
            label=(('HDSA' if row['case']=='hdsa' else '044')+' · 离线数值对照'
                   if offline else row['case_id']+' · '+row['version'])
            caption=('零新增 API 请求；仍为部分解' if offline else
                     '在线最终阶段复测；完整目标未通过' if row['version']!='v77' else
                     'V77 冻结发布结果；完整目标未通过')
            if not offline and row['version']=='fixed-v86' and row['case_id']=='HDSA-65-main':
                caption+='；本次求解被拒绝，图中为保留产物'
            error=row.get('native_max_error_mm')
            caption+='；原生坐标最大 GT 误差：'+(f'{error:.6f} mm' if isinstance(error,(int,float)) else '未评估')
            if not path.is_file():
                cards.append(f'<figure><figcaption><strong>{esc(label)}</strong><p>叠图未生成。</p></figcaption></figure>')
                continue
            data=path.read_bytes()
            source='data:image/svg+xml;base64,'+base64.b64encode(data).decode('ascii')
            cards.append(f'<figure><figcaption><strong>{esc(label)}</strong><p>{esc(caption)}</p></figcaption>'
                         f'<a href="{esc(path.as_uri())}" target="_blank" rel="noopener">'
                         f'<img src="{source}" alt="{esc(label)}：GT 与重建 DXF 的轮廓叠图" loading="lazy"></a>'
                         '<p class="overlay-open">点击图片查看原始 SVG，可放大检查局部。</p></figure>')
            previews.append(dict(label=label,scope='offline' if offline else 'online',
                path=str(path),sha256=hashlib.sha256(data).hexdigest(),
                native_max_error_mm=error,original_evaluation_svg_embedded_unchanged=True))
        groups.append(f'<h3>{esc(title)}</h3><div class="overlay-grid">'+''.join(cards)+'</div>')
    report['overlay_previews']=previews
    return ('<section id="overlay-gallery"><h2>GT 与生成 DXF 的 overlay 对比</h2>'
        '<p><span class="gt-key">紫红色：GT</span>；<span class="prediction-key">青绿色：生成／保留的 DXF</span>；虚线：GT 假设闭合层。</p>'
        '<p>叠图沿用独立评估中的 D4 方向／平移配准，仅用于观察形状。图注中的最大误差来自原生坐标比较；视觉重合不能证明标注、对象数量或相切全部满足。</p>'
        +''.join(groups)+'</section>')


def main():
    project=Path(__file__).resolve().parents[1]
    root=project/'runtime/cad-controlled-20261008'
    out=project/'runtime/cad-fixed-annotations-20261012';out.mkdir(exist_ok=True)
    base.SLOTS=[('v77','source-measurement-v77-online','原 V77 冻结发布结果'),
                ('fixed-v82','source-fixed-v82-online','中间修订在线复测；复用上游 OCR／拓扑'),
                ('fixed-v86','source-fixed-v86-online','最终 V85 核心在线复测；复用上游 OCR／拓扑')]
    report=base.assemble(root)
    report.update(report_reads_gt_dxf=False,report_calls_provider=False,
        fresh_full_end_to_end_runs=0,reference_thresholds_changed=False,
        final_code_snapshot=str(root/'source-fixed-v85-code'),
        full_test_log=str(project/'runtime/pytest-v85-full.txt'))
    for row in report['rows']:
        validation=read(root/row['run']/'after/validation.json')
        row['exact_axis_readback']=validation.get('native_axis_relation_validation')
    offline=[]
    for case,slot in [('hdsa','source-fixed-v85-offline-hdsa'),('044','source-fixed-v81-offline-044')]:
        directory=root/slot;state=read(directory/'refinement.json');number=state.get('selected_numerical_round')
        if type(number) is not int:continue
        export=directory/f'round-{number:02d}'/'export'
        model=read(export/'model.json');validation=model.get('validation',{})
        selected=next(r for r in state['rounds'] if r['round']==number)
        summary=read(directory/'post-frozen-evaluation/evaluation/summary.json')
        offline.append(dict(case=case,scope='离线求解对照：冻结的在线标注，零新增网络请求',
            source_gate_passed=selected.get('source_gate_passed'),publication_approved=False,
            drawing=str(export/'drawing.dxf'),
            overlay=str(directory/'post-frozen-evaluation/evaluation/overlay.svg'),
            native_max_error_mm=summary.get('geometry',{}).get('native_coordinate_frame',{}).get('conservative_max_error_mm'),
            radius=validation.get('exact_radius_validation'),angle=validation.get('native_source_angle_validation'),
            axis=validation.get('native_axis_relation_validation'),tangent=validation.get('strict_relation_validation'),
            source_frame_recovery=read(directory/f'round-{number:02d}/parametric-solution.json').get('diagnostics',{}).get('source_frame_recovery'),
            entities=model.get('entities'),known_constraint_values_changed=state.get('known_constraint_values_changed')))
    report['offline_fixed_annotation_diagnostics']=offline
    gallery=overlay_gallery(report,root)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    esc=lambda v:html.escape(str(v))
    def link(path,label):
        path=Path(path)
        return f'<a href="{esc(path.as_uri())}">{esc(label)}</a>' if path.is_file() else esc(label+'（未生成）')
    extra='<p>已知半径、斜线角度、已准入水平／竖直关系进入精确方程；未知属性继续联合求解。没有从 GT DXF 复制坐标或向 API 发送 GT CAD。GT 派生的栅格掩膜仍为开发输入。</p>'
    extra+='<p>本轮修复：轴向关系硬等式和独立 DXF 回读；补全失败后保留已知约束再重试；按图元长度增加有界原图采样；继承 V77 标注优先策略；从原图笔画恢复整体图像定位。原有验证门槛没有降低。</p>'
    extra+='<p>下面在线表与离线对照分开列出。图像定位恢复只调整整体平移，所有尺寸与相切方程仍须重新回读验证；它不能补出 R16／R13，也不能证明尚未核实的接点相切。两例仍不能宣称完整目标全部通过。</p>'
    extra+='<p>最终代码在线复测 V86：HDSA 初次求解在预算内未获得可认证候选，保留原有效产物；044 发布部分约束结果，后续补全的联合方程未求解通过。完整目标仍为 0/2。下方 HDSA 离线对照中的 4 个角度同时精确，不能当成本次在线运行的成功。</p>'
    extra+='<p>最终全量回归：2242 passed、3 skipped、2 warnings（381.37 秒）；详细日志 runtime/pytest-v85-full.txt。该测试计数不代表数据集重建通过数。</p>'
    extra+='<table><tr><th>离线对照</th><th>精确 R</th><th>精确斜角</th><th>精确轴向</th><th>已准入 G1</th><th>原图门槛</th><th>最大 GT 误差</th><th>产物</th></tr>'
    for row in offline:
        ratio=lambda key:f"{(row.get(key) or {}).get('satisfied_count','—')}/{(row.get(key) or {}).get('required_count','—')}"
        extra+='<tr>'+''.join(f'<td>{value}</td>' for value in [esc(row['case']),ratio('radius'),ratio('angle'),ratio('axis'),ratio('tangent'),esc(row['source_gate_passed']),esc(row['native_max_error_mm']),link(row['drawing'],'DXF')+' · '+link(row['overlay'],'overlay')])+'</tr>'
    extra+='</table><p>HDSA 仍有未绑定尺寸与未准入相切；044 仍有 R6 来源证书、R16／R13 缺失对象和左上 R15／R35 的接点问题。保持失败项在分母内。</p>'
    rendered=base.render(report)
    # base.render intentionally omits the optional closing body tag. Insert
    # after its summary table instead of silently discarding the supplement.
    rendered=rendered.replace('</main>','</main>'+extra+gallery,1)
    rendered=rendered.replace('</style>',
        '.overlay-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}'
        '.overlay-grid figure{margin:0;background:#fff;border:1px solid #ccd8d3;border-radius:8px;overflow:hidden}'
        '.overlay-grid figcaption{padding:14px 18px}.overlay-grid figcaption p{margin:6px 0 0;font-size:13px}'
        '.overlay-grid img{display:block;width:100%;height:auto;background:#fff}'
        '.overlay-open{padding:0 18px 12px;font-size:13px;color:#526962}'
        '.gt-key{color:#b33d62}.prediction-key{color:#087f78}'
        '#overlay-gallery{margin:28px 0}#overlay-gallery h3{margin-top:24px}'
        '@media(max-width:900px){.overlay-grid{grid-template-columns:1fr}}'
        '</style>',1)
    rendered=rendered.replace('<main>','<p><a href="#overlay-gallery">直接查看 8 张 overlay 对比图 ↓</a></p><main>',1)
    # The base renderer's links are relative to its controlled experiment root.
    for row in report['rows']:
        rendered=rendered.replace('href="'+row['run']+'/', 'href="'+(root/row['run']).as_uri()+'/')
        rendered=rendered.replace('src="'+row['run']+'/', 'src="'+(root/row['run']).as_uri()+'/')
    (out/'report.html').write_text(rendered,'utf8')
    print(out/'report.html')


if __name__=='__main__':main()

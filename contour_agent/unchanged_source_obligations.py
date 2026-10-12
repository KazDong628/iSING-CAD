"""Keep certified obligations on untouched objects across a local topology edit.

This does not admit new annotations or transfer a proof to a changed primitive.
The parent checkpoint, source pixels/OCR, typed anchors and native equation are
rechecked. Missing repeat detection is distinct from positive contradiction.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path


def attach_unchanged_parent(graph,parent):
    evidence=parent.get('source_evidence') or {}
    resume=evidence.get('source_state_resume') or {}
    path=Path(resume.get('artifact_directory',''))/'observed-topology.json'
    if resume.get('method')=='matched_certified_source_state_v1' and path.is_file():
        graph['source_evidence']['unchanged_source_parent']={'path':str(path.resolve()),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    elif evidence.get('unchanged_source_parent'):
        graph['source_evidence']['unchanged_source_parent']=copy.deepcopy(evidence['unchanged_source_parent'])


def retain_unchanged_obligations(image,document,baseline,graph,bindings,inventory):
    proof=(graph.get('source_evidence') or {}).get('unchanged_source_parent')
    if not proof:return bindings,{'status':'not_applicable','carried':[]}
    from .source_state import resumed_files
    from .source_radius_hypotheses import graph_from_solution,_same_solved_geometry,_same_record_sources
    from .parametric_pipeline import _constraint_signatures,_constraint_receipts_complete
    from .constraint_binding import _constraint
    from .parametric_solver import native_measurement_checks
    from .relation_contract import relation_checks
    receipt={'status':'checked','carried':[],'rejected':[],'ground_truth_used':False,
        'current_API_reconfirmed':False,'repeat_detector_reconfirmed':False}
    raw=Path(proof['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=proof.get('sha256'):raise ValueError('unchanged_source_parent_snapshot_changed')
    parent=json.loads(raw);files=resumed_files(parent)
    digest=hashlib.sha256(Path(image).read_bytes()).hexdigest()
    if (files is None or parent.get('source_sha256')!=digest or graph.get('source_sha256')!=digest or
            parent.get('units')!=graph.get('units') or parent.get('coordinate_system')!=graph.get('coordinate_system')):
        raise ValueError('unchanged_source_inputs_changed')
    old_constraints=files['bindings.json']['constraints'];old_inventory=files['inventory.json']
    if (old_inventory.get('source_image_sha256')!=digest or
            not _same_record_sources(document,old_inventory,[r.get('record_id') for r in old_constraints]) or
            not _constraint_receipts_complete(files['bindings.json'],files['solution.json'],files['domain.json']) or
            not _same_solved_geometry(graph_from_solution(files['domain.json'],baseline,files['solution.json']),parent)):
        raise ValueError('unchanged_parent_certificate_not_matched')
    old={e['id']:e for e in parent['entities']};new={e['id']:e for e in graph['entities']}
    mapping={};nodes={}
    def identical(a,b):
        if a['type']!=b['type'] or a.get('clockwise')!=b.get('clockwise'):return False
        for key in ('start','end','center'):
            if key in a and (key not in b or math.dist(a[key],b[key])>1e-7):return False
        return 'radius' not in a or abs(a['radius']-b.get('radius',float('inf')))<=1e-7
    for eid,e in old.items():
        matches=[f for f in new.values() if identical(e,f) and (
            eid in f.get('parent_entity_ids',[]) or e.get('stable_id') and e.get('stable_id')==f.get('stable_id'))]
        if len(matches)==1:
            f=matches[0];mapping[eid]=f['id']
            for side in ('start','end'):
                key=e[side+'_node'];value=f[side+'_node']
                nodes[key]=value if key not in nodes or nodes[key]==value else None
    def remap(value,key=None):
        if isinstance(value,list):return [remap(v,key) for v in value]
        if isinstance(value,dict):return {k:remap(v,k) for k,v in value.items()}
        if key in {'entities','entity_id'}:return mapping.get(value,'unmapped')
        if key in {'nodes','node_id'}:return nodes.get(value,'unmapped')
        return value
    result=copy.deepcopy(bindings);fresh=result['constraints']
    signatures=_constraint_signatures(graph,fresh)
    for row in old_constraints:
        # Radius/angle ownership is intentionally left to the existing fresh
        # arrow and finite-LINE machinery. No new radius is granted here.
        if row['kind'] not in {'distance','distance_x','distance_y','horizontal','vertical','tangent'}:continue
        mapped=remap(row)
        if 'unmapped' in json.dumps(mapped) or None in mapped.get('nodes',[]):continue
        signature=_constraint_signatures(graph,[mapped])
        if signature<=signatures:continue
        rid=row.get('record_id');reason=None
        if any(rid is not None and c.get('record_id')==rid and
               _constraint_signatures(graph,[c])!=signature for c in fresh):reason='fresh_positive_annotation_conflict'
        if any(rid is not None and d.get('record_id')==rid and d.get('reason') in {
            'source_observed_text_mismatch','source_symbol_confusion_requires_confirmation'} for d in result.get('bindings',[])):
            reason='fresh_source_text_contradiction'
        if row.get('source')=='source_geometry':
            original=[r for r in old_inventory.get('relations',[]) if r.get('type')==row['kind'] and
                set(r.get('entities',[]))==set(row.get('entities',[])) and r.get('local_reliable') is True and
                (r.get('evidence') or {}).get('verified') is True]
            if len(original)!=1:reason='previous_independent_relation_proof_missing'
            for r in inventory.get('relations',[]):
                e=r.get('evidence') or {}
                if (r.get('type')==row['kind'] and set(r.get('entities',[]))==set(mapped['entities']) and
                        e.get('reason')=='source_junction_is_not_tangent' and
                        e.get('observed_deviation_degrees') is not None and
                        e.get('source_ink_measurement_performed',True)):
                    reason='fresh_positive_original_ink_contradiction'
        else:
            original=[c for c in old_inventory.get('all_candidates',old_inventory.get('candidates',[]))
                if c.get('local_reliable') is True and c.get('record_id')==rid and
                _constraint_signatures(parent,[_constraint(c,'ocr_local_binding')])==_constraint_signatures(parent,[row])]
            if len(original)!=1:reason='previous_unique_source_dimension_proof_missing'
        native=(relation_checks(graph['entities'],[mapped]) if row['kind'] in {'tangent','horizontal','vertical'} else
                native_measurement_checks(graph['entities'],[mapped]))
        if native.get('passed') is not True:reason='unchanged_native_equation_failed'
        if reason:
            receipt['rejected'].append({'constraint_id':row['id'],'reason':reason});continue
        mapped.update(id='kunchanged_'+row['id'],admission_method='unchanged_source_checkpoint',
            current_API_reconfirmed=False,repeat_detector_reconfirmed=False)
        if mapped.get('source')!='source_geometry':mapped['source']='ocr_local_binding'
        fresh.append(mapped);signatures.update(signature)
        witness=remap(copy.deepcopy(original[0]))
        witness.update(id='unchanged_'+row['id'],admission_method='unchanged_source_checkpoint',
            current_API_reconfirmed=False,fresh_detector_reconfirmed=False)
        inventory.setdefault('relations' if row.get('source')=='source_geometry' else 'all_candidates',[]).append(witness)
        result.setdefault('bindings',[]).append({'record_id':rid,'accepted':True,
            'source':mapped['source'],'constraint_id':mapped['id'],
            'admission_method':'unchanged_source_checkpoint','current_API_reconfirmed':False,
            'repeat_detector_reconfirmed':False,'reason':None})
        receipt['carried'].append({'old_constraint_id':row['id'],'constraint_id':mapped['id'],
            'record_id':rid,'kind':row['kind'],'entities':mapped['entities'],
            'native_equation_rechecked':True,'source_objects_unchanged':True})
    bound={c.get('record_id') for c in fresh}-{None}
    result['counts'].update(constraints=len(fresh),bound_source_records=len(bound),
        unbound_dimensions=result['counts']['recognized_dimensions']-len(bound),
        unchanged_source_constraints_retained=len(receipt['carried']))
    result['unchanged_source_obligation_retention']=receipt
    return result,receipt

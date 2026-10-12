"""Close the source-obligation loop after an accepted, incomplete joint solve.

Reference CAD is never an input. Structural edits start from the original
source graph and mask, and must freshly retain the previous typed equations.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path


def complete_source_topology(image_path, document, baseline, graph, bindings,
                             solution, inventory, output, *, provider=None,
                             editor_provider=None, use_api=False, max_rounds=2):
    from .constraint_binding import analyze_constraint_bindings
    from .parametric_pipeline import (_write, _constraint_receipts_complete, _constraint_signatures,
        _final_source_multistart, _solved_source_validation, export_parametric)
    from .reconstruction_feedback import reconstruction_feedback,geometry_fingerprint
    from .source_topology_initial_guess import required_constraints_preserved
    from .topology_candidates import (_read_image, _affines, _closed_ring,
        _base_source_ring, _base_source_entities, _annotation_inventory, _json_hash,
        _render_overlay)
    from .topology_editing import (_fresh_edit_source_graph,
        propose_annotation_arc_edits, execute_topology_edits)
    from .ocr import canonical_records
    from .parametric_solver import solve_parametric
    import cv2

    if type(max_rounds) is not int or not 0 <= max_rounds <= 2:
        raise ValueError('Topology completion requires at most two bounded rounds')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    receipt=dict(schema_version='post-solve-source-topology-completion-v1',accepted=False,
        maximum_rounds=max_rounds,maximum_candidates_per_round=2,rounds=[],
        source_graph_initialization='original_graph_and_immutable_input_mask',
        reference_geometry_used=False,acceptance_thresholds_changed=False,
        publication_approved=False)
    current=(graph,bindings,solution,inventory)
    attempted=set()
    previous_operations=[]
    def finish(reason):
        receipt['reason']=reason
        _write(output/'completion.json',receipt)
        return (*current,receipt)
    if solution.get('accepted') is not True or not _constraint_receipts_complete(bindings,solution,graph):
        return finish('previous_source_subset_not_certified')
    for number in range(max_rounds):
        parent,bound,sol,inv=current
        parent_fingerprint=geometry_fingerprint(parent)
        feedback=reconstruction_feedback(parent,bound,sol)
        unresolved={r['record_id'] for r in (bound.get('radius_binding_coverage') or {}).get('unresolved',[])}
        bound_ids={c.get('record_id') for c in bound['constraints']}
        missing_angles={r['id'] for r in canonical_records(document) or inv.get('all_records',inv.get('records',[]))
                        if (r.get('parsed') or {}).get('kind')=='angle' and r['id'] not in bound_ids}
        # A verified radius does not certify either adjacent contact. Review
        # unknown curved joins, including small jumps; never force G1 from
        # their appearance. Fresh original-source binding remains mandatory.
        review=unconfirmed_curved_joins(parent,bound)
        if not unresolved and not missing_angles and not review:
            return finish('no_unresolved_annotation_or_contact_obligation')
        feedback.update(round=number+1,required_radius_record_ids=sorted(unresolved),
            required_angle_record_ids=sorted(missing_angles),
            source_contact_review_pairs=review,previous_operations=copy.deepcopy(previous_operations))
        feedback['issues'].extend(dict(code='angle_annotation_unbound',record_id=rid,
            binding_verified=False) for rid in sorted(missing_angles))
        directory=output/f'round-{number+1:02d}';directory.mkdir(parents=True,exist_ok=True)
        row=dict(round=number+1,status='source_proposals',accepted=False,
                 unresolved_radius_record_ids=sorted(unresolved),
                 unresolved_angle_record_ids=sorted(missing_angles),
                 source_contact_review_pairs=review,candidates=[])
        receipt['rounds'].append(row);_write(output/'completion.json',receipt)
        try:
            # Recompute leaders from original pixels. Do not edit a solved
            # graph and thereby reset the original geometry/displacement prior.
            image=_read_image(Path(image_path));gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            transform,_,orientation=_affines(parent,baseline)
            grid=float(parent.get('source_grid_pitch_px') or 1.)
            raw=(baseline.get('extraction') or {}).get('raw_polyline_px')
            ring=_closed_ring(raw) if raw is not None else _base_source_ring(parent,transform,grid)
            annotations,summary=_annotation_inventory(gray,canonical_records(document),ring,grid,
                [*(parent.get('annotation_support') or []),*(parent.get('radius_source_segment_hypotheses') or [])])
            working,refresh=_fresh_edit_source_graph(parent,
                _base_source_entities(parent,transform,orientation),annotations,grid)
            # Reobserve angle ownership against the CURRENT original graph.
            # A post-solve multimodal hint must not leave stale angle targets
            # on the structural planner's copy.
            from .constraint_binding import _angle_source_observations
            working['angle_source_observations']=_angle_source_observations(gray,
                canonical_records(document),working,transform,
                max(4.,float(working.get('proposal_tolerance_px',max(gray.shape)/500.))))
            _write(directory/'original-source-annotation-inventory.json',{'records':annotations,'summary':summary})
            _write(directory/'source-parent-topology.json',working)
            overlay=directory/'source-parent.png'
            _render_overlay(image,_base_source_entities(working,transform,orientation),
                            working.get('candidate_id','source-parent'),overlay)
            selected=dict(id=working.get('candidate_id','source-parent'),graph=working,overlay_path=str(overlay))
            bundle=dict(source_sha256=parent['source_sha256'],base_graph_sha256=_json_hash(parent),
                        annotation_inventory=annotations,annotation_inventory_summary=summary)
            operations=propose_annotation_arc_edits(working,annotations,limit=2,
                unresolved_radius_record_ids=unresolved,unresolved_angle_record_ids=missing_angles)
            if use_api and editor_provider is not None:
                try:
                    proposed=editor_provider.propose(image_path,overlay,selected,annotations,
                        feedback=feedback,request_timeout_seconds=60.)
                except InterruptedError:raise
                except Exception as error:
                    proposed=dict(schema_success=False,status='failed',failure_type=type(error).__name__)
                row['provider']=proposed
                if proposed.get('schema_success') is True:
                    # A bounded API proposal may locate a missing feature that
                    # has no existing binding candidate. The kernel still
                    # checks its source evidence and accepted action schema.
                    operations=[*proposed.get('operations',[])[:2],*operations]
            unique={}
            out_of_focus=[]
            for op in operations:
                key=(op.get('action'),tuple(op.get('entity_ids') or []),op.get('record_id'))
                if not proposal_targets_obligation(op,unresolved|missing_angles,review,bound_ids):
                    out_of_focus.append({**op,'reason':'operation_does_not_target_current_source_obligation'})
                    continue
                if (parent_fingerprint,key) not in attempted:unique.setdefault(key,op)
            operations=prioritize_structural_obligations(working,list(unique.values()),unresolved,
                missing_angles=missing_angles)[:2]
            attempted.update((parent_fingerprint,(op.get('action'),tuple(op.get('entity_ids') or []),op.get('record_id')))
                             for op in operations)
            row.update(source_refresh=refresh,operations=operations,out_of_focus_operations=out_of_focus)
            previous_operations.extend({**op,'status':'rejected'} for op in out_of_focus)
            _write(output/'completion.json',receipt)
            if not operations:
                row.update(status='retained',reason='no_new_source_supported_structural_proposal')
                continue
            candidates,audit=execute_topology_edits(image_path,document,baseline,selected,
                bundle,operations,directory/'edits')
            row['execution']=audit
            previous_operations.extend({**item.get('operation',{}),
                **{key:item[key] for key in ('status','reason') if key in item}}
                for item in audit.get('operations',[]) if isinstance(item,dict)
                and isinstance(item.get('operation'),dict))
            # Combined disjoint repairs, when present, deserve a preflight.
            candidates.sort(key=lambda c:0 if (c['graph'].get('source_evidence',{}).get(
                'topology_edit',{}).get('action')=='apply_nonoverlapping_edits') else 1)
            advanced=False
            for index,candidate in enumerate(candidates[:2]):
                candidate_dir=directory/f'candidate-{index+1:02d}'
                child=candidate['graph'];entry=dict(id=candidate['id'],accepted=False)
                row['candidates'].append(entry);_write(output/'completion.json',receipt)
                try:
                    fresh=analyze_constraint_bindings(image_path,document,baseline,child,candidate_dir,
                        provider=provider,use_api=use_api)
                except InterruptedError:raise
                except Exception as error:
                    entry.update(reason='candidate_binding_failed',failure_type=type(error).__name__)
                    _write(output/'completion.json',receipt)
                    continue
                fresh_inv=json.loads(Path(fresh['inventory_artifact']).read_text('utf8'))
                retain=required_constraints_preserved(parent,child,bound['constraints'],fresh['constraints'])
                entry['known_obligation_preservation']=retain
                _write(candidate_dir/'known-obligation-preservation.json',retain)
                if retain.get('passed') is not True:
                    entry['reason']='previous_typed_obligation_not_freshly_retained';continue
                before={c.get('record_id') for c in bound['constraints']} - {None}
                after={c.get('record_id') for c in fresh['constraints']} - {None}
                entry['newly_bound_source_record_ids']=sorted(after-before)
                # Preservation has already checked typed identity/anchors.
                # Admit a genuinely enlarged relation set even when it has
                # no OCR record ID (e.g. two freshly proven G1 contacts).
                known=_constraint_signatures(parent,bound['constraints'])
                proposed_signatures=_constraint_signatures(child,fresh['constraints'])
                entry['added_unique_constraint_count']=len(proposed_signatures)-len(known)
                if not before<=after or (not after-before and len(proposed_signatures)<=len(known)):
                    entry['reason']='no_new_certified_source_obligation';continue
                solved=_final_source_multistart(image_path,document,baseline,child,
                    fresh['constraints'],candidate_dir/'solve',solver=solve_parametric,
                    budget_seconds=90.)
                _write(candidate_dir/'parametric-solution.json',solved)
                entry['solver_status']=solved.get('status')
                if not solved.get('accepted') or not _constraint_receipts_complete(fresh,solved,child):
                    entry['reason']='enlarged_structural_problem_not_certified';continue
                before_dof=(sol.get('diagnostics') or {}).get('remaining_shape_dof')
                after_dof=(solved.get('diagnostics') or {}).get('remaining_shape_dof')
                entry.update(shape_dof_before=before_dof,shape_dof_after=after_dof)
                if (type(before_dof) in (int,float) and type(after_dof) in (int,float)
                        and after_dof>before_dof):
                    entry['reason']='physical_shape_dof_increased';continue
                check=_solved_source_validation(image_path,document,baseline,graph,solved['entities'])
                entry['original_source_validation']=check
                if not check.get('passed'):
                    entry['reason']='original_source_gate_failed';continue
                exported=export_parametric(image_path,baseline,
                    {**solved,'constraints':copy.deepcopy(fresh['constraints'])},candidate_dir/'export')
                required=('exact_radius_validation','strict_relation_validation','native_measurement_validation',
                          'native_source_angle_validation','native_axis_relation_validation')
                native=all(exported['validation'].get(k,{}).get('passed') is True and
                    exported['validation'][k].get('dxf_readback_performed') is True for k in required)
                entry['native_readback_passed']=native
                if not native:
                    entry['reason']='native_equation_readback_failed';continue
                current=(child,fresh,solved,fresh_inv)
                row.update(status='certified',accepted=True,selected_candidate=candidate['id'])
                entry.update(accepted=True,reason='new_source_obligation_all_gates_passed')
                receipt['accepted']=advanced=True
                _write(output/'selected-topology.json',child)
                break
            if not advanced:
                row['status']='retained'
                # Feed bounded failure evidence into one DIFFERENT proposal.
                # Never repeat an identical action on the unchanged parent.
                previous_operations.extend({**op,'status':'rejected',
                    'reason':next((c.get('reason') for c in row['candidates'] if c.get('reason')),
                                  'local_construction_not_certified')} for op in operations)
        except InterruptedError:
            row.update(status='interrupted');_write(output/'completion.json',receipt);raise
        except Exception as error:
            row.update(status='retained',failure_type=type(error).__name__)
            return finish('structural_completion_failed_previous_subset_retained')
        finally:
            _write(output/'completion.json',receipt)
    return finish('bounded_structural_completion_finished')


def prioritize_structural_obligations(graph, operations, unresolved, *, missing_angles=()):
    """A repeated numeric ARC refit cannot starve an unresolved LINE target."""
    entities={e['id']:e for e in graph.get('entities',[])}
    support={r.get('record_id'):r for r in graph.get('annotation_support',[])}
    def score(op):
        rid=op.get('record_id');witness=support.get(rid,{})
        target=entities.get(witness.get('candidate_entity_id'),{})
        conflict=(rid in unresolved and target.get('type')=='LINE'
                  and witness.get('kind')=='radius')
        structural=op.get('action') in {'insert_annotated_fillet','restore_annotated_line_support',
                                        'split_chain_at_source_features'}
        return (int(rid in missing_angles),int(conflict),int(rid in unresolved and structural),
                int(structural),int(rid in unresolved))
    return sorted(operations,key=score,reverse=True)


def unconfirmed_curved_joins(graph, bindings):
    """A source review queue, not inferred tangent constraints or GT evidence."""
    confirmed={frozenset(c.get('entities',[])) for c in bindings.get('constraints',[])
               if c.get('kind') in {'tangent','angle_between','perpendicular','parallel'}}
    entities=graph.get('entities',[])
    return [[a['id'],b['id']] for a,b in zip(entities,entities[1:]+entities[:1])
            if 'ARC' in {a.get('type'),b.get('type')} and
            frozenset((a['id'],b['id'])) not in confirmed][:24]


def proposal_targets_obligation(operation, missing_records, review_pairs, bound_records):
    """An already solved label alone is not an unresolved planning goal."""
    rid=operation.get('record_id')
    if rid is None:return True
    if rid in missing_records:return True
    if rid not in bound_records:return True
    return bool(set(operation.get('entity_ids') or []) & {eid for pair in review_pairs for eid in pair})

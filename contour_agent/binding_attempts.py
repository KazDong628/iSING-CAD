"""Binding invocation receipts, distinct from the selected publication snapshot."""
from __future__ import annotations
import copy
import json
from pathlib import Path


def _read(directory, name):
    try:
        path = Path(directory) / name
        if path.stat().st_size > 8_000_000:
            return {}
        value = json.loads(path.read_text(encoding='utf8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def collect_binding_attempts(directory, stage=None):
    """Recover bounded actual calls, using paths rather than receipt hashes.

    Repeated identical provider responses can be distinct network invocations.
    Selected/publication copies are aliases, not additional calls. Frozen older
    runs are read only and marked as artifact recovery; no receipt is rewritten.
    """
    if directory is None:
        return []
    root = Path(directory)
    stage = stage if isinstance(stage, dict) else _read(root, 'parametric-stage.json')
    trusted = (root / 'final-binding-attempt').is_dir()
    base = root / 'final-binding-attempt' if trusted else root
    certificate = stage.get('source_radius_seed_certification') or _read(base,
        'source-radius-certification/radius-seed-certification.json')
    formal = _read(base, 'source-radius-certification/full-source-binding/constraint-bindings.json')
    formal_provider = formal.get('provider') or certificate.get('provider')
    rows = []
    def append(phase, document, artifact, *, recovery=False):
        provider = document.get('provider') if isinstance(document, dict) else None
        if not isinstance(provider, dict) or not provider:
            return
        if provider.get('status') in {'pending','not_invoked'} or (
                not provider.get('network_requests') and provider.get('status') in {'disabled','skipped','not_configured'}):
            return
        rows.append({'attempt_id': 'binding_' + phase, 'phase': phase,
            'provider': copy.deepcopy(provider), 'receipt_artifact': artifact,
            'receipt_scope': 'current_run_network_invocation',
            'recovered_from_artifacts': recovery, 'carried_constraints_are_new_API_confirmation': False})
    if not trusted:
        initial = _read(base, 'binding-initial-attempt.json')
        artifact = 'binding-initial-attempt.json'
        recovery = not bool(initial)
        if not initial:
            initial = _read(base, 'initial-radius-constraint-bindings.json')
            artifact = 'initial-radius-constraint-bindings.json'
        if not initial and certificate.get('status') != 'accepted_seed':
            initial = _read(base, 'constraint-bindings.json')
            if not initial.get('provider'):
                initial = {'provider': stage.get('provider', {})}
            artifact = 'constraint-bindings.json'
        append('initial', initial, artifact, recovery=recovery)
    if formal_provider:
        append('radius_rebinding', {'provider': formal_provider},
            str((base / 'source-radius-certification/full-source-binding/constraint-bindings.json').relative_to(root)),
            recovery=not bool(_read(root, 'binding-attempts.json')))
    if trusted and certificate.get('status') != 'accepted_seed':
        phase = 'fallback' if formal_provider else 'initial'
        name = 'binding-fallback-attempt.json' if formal_provider else 'binding-initial-attempt.json'
        final = _read(base, name)
        recovery = not bool(final)
        if not final:
            final = _read(base, 'constraint-bindings.json')
            if not final.get('provider'):
                final = {'provider': stage.get('provider', {})}
            name = 'constraint-bindings.json'
        append(phase, final, str((base / name).relative_to(root)), recovery=recovery)
    post = root / 'post-solve-radius-retry/source-radius-certification/full-source-binding/constraint-bindings.json'
    if post.is_file():
        append('post_solve_radius_rebinding', _read(post.parent, post.name),
               str(post.relative_to(root)), recovery=not bool(_read(root, 'binding-attempts.json')))
    for number in (1,2):
        completion=root/'source-constraint-completion'/f'round-{number:02d}'/'constraint-bindings.json'
        if completion.is_file():
            append(f'source_completion_{number}',_read(completion.parent,completion.name),
                   str(completion.relative_to(root)),recovery=not bool(_read(root,'binding-attempts.json')))
        for candidate in (1,2):
            structural=(root/'source-topology-completion'/f'round-{number:02d}'/
                        f'candidate-{candidate:02d}'/'constraint-bindings.json')
            if structural.is_file():
                append(f'structural_completion_{number}_candidate_{candidate}',
                    _read(structural.parent,structural.name),str(structural.relative_to(root)),
                    recovery=not bool(_read(root,'binding-attempts.json')))
    return rows


def binding_attempt_summary(rows):
    providers = [row['provider'] for row in rows]
    return {'attempt_count': len(rows),
        'known_network_requests': sum(p.get('network_requests') for p in providers
                                      if type(p.get('network_requests')) is int),
        'unknown_request_count_attempts': sum(p.get('network_requests') is None for p in providers),
        'http_successful_attempts': sum(p.get('http_success') is True for p in providers),
        'http_failed_attempts': sum(p.get('http_success') is False for p in providers),
        'schema_successful_attempts': sum(p.get('schema_success') is True for p in providers),
        'scope': 'all_current_binding_invocations_including_failed_candidates; not geometry accuracy'}


def published_binding_sources(published_stage):
    """Never label a carried source certificate as a new API confirmation."""
    rows = []
    for constraint in (published_stage or {}).get('constraints', [])[:128]:
        if not isinstance(constraint, dict):
            continue
        carried = constraint.get('admission_method') == 'carried_source_binding_certificate'
        rows.append({key: constraint.get(key) for key in ('id','kind','record_id','source','admission_method','prior_binding_source')})
        rows[-1]['current_API_reconfirmed'] = bool(not carried and constraint.get('source') == 'ocr_api_binding'
            and (published_stage or {}).get('dimensions_updated_by_api') is True)
        rows[-1]['carried_source_certificate'] = carried
    return {'scope':'current_published_dxf_constraints', 'constraints':rows,
        'carried_constraint_count':sum(row['carried_source_certificate'] for row in rows),
        'current_API_confirmed_constraint_count':sum(row['current_API_reconfirmed'] for row in rows),
        'source_geometric_relations_are_separate_from_API_dimension_confirmation':True}

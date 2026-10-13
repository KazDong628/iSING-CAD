"""Isolated retries for withheld OCR, with a verifiable parent evidence chain."""
from copy import deepcopy
import json
from pathlib import Path
import time

from .source_ocr_review import (_hash,_object_hash,_semantic,_write,
    _valid_transcriptions,_receipt_succeeded,_review_texts)
from .ocr import canonical_records

RETRY_FILE='source-ocr-retry.json'
EFFECTIVE_FILE='effective-source-ocr-retry.json'
MAX_RECORDS=4


def _apply(parent,attempts):
    result=deepcopy(parent);records={r['id']:r for r in canonical_records(parent)}
    seen=set();changed=[]
    for attempt in attempts:
        rid=attempt['record_id']
        if rid in seen or rid not in records:raise ValueError('retry_record_identity')
        seen.add(rid);record=records[rid]
        admission=record.get('source_text_admission') or {}
        if admission.get('status')!='withheld' or admission.get('record_id')!=rid:
            raise ValueError('retry_requires_withheld_parent_record')
        views=attempt['views']
        if not isinstance(views,list) or len(views)!=2:raise ValueError('retry_view_count')
        for receipt in views:
            if _receipt_succeeded(receipt):
                validated=_valid_transcriptions({'transcriptions':receipt['transcriptions']},{rid})
                if validated!=receipt['transcriptions']:raise ValueError('retry_transcription_identity')
        a,b=_review_texts(rid,views)
        consensus=bool(a and b and _semantic(a)==_semantic(b))
        if attempt['consensus'] is not consensus:raise ValueError('retry_consensus_mismatch')
        if consensus:
            i=int(rid[1:]);result['records'][i]['text']=a
            result['records'][i].pop('source_text_admission',None);changed.append(rid)
    return result,changed


def retry_withheld_source_ocr(image,document,output,*,provider=None,use_api=False,
                              max_seconds=180.,progress=None):
    """Never reinterpret an uncontested or scale-locked source value here.

    Standard OCR review and its effective document remain immutable. Each
    withheld record receives its own pair of raw-image requests, never other
    OCR values or CAD. A fresh binding/solve is still necessary after recovery.
    """
    if not 0<float(max_seconds)<=180:raise ValueError('retry_time_budget')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[r for r in canonical_records(document) if (r.get('source_text_admission') or {}).get('status')=='withheld'][:MAX_RECORDS]
    receipt=dict(schema_version='source-ocr-isolated-retry-v1',source_image_sha256=_hash(image),
        parent_document_sha256=_object_hash(document),attempts=[],changed_record_ids=[],
        ground_truth_used=False,cad_geometry_used=False,consensus_is_ground_truth=False,
        maximum_records=MAX_RECORDS,maximum_network_requests=2*MAX_RECORDS,maximum_seconds=float(max_seconds),
        status='running' if use_api and provider is not None and rows else 'not_needed_or_unavailable')
    started=time.monotonic()
    def save():
        effective,changed=_apply(document,receipt['attempts'])
        receipt['changed_record_ids']=changed
        _write(output/EFFECTIVE_FILE,effective)
        receipt.update(effective_document_sha256=_object_hash(effective),effective_file_sha256=_hash(output/EFFECTIVE_FILE))
        _write(output/RETRY_FILE,receipt)
        return effective
    if not use_api or provider is None or not rows:return document,receipt
    # Preserve different crop extents but provide upright vertical glyphs in
    # both requests. This changes presentation only, never the source value.
    isolated=getattr(provider,'isolated_reader',None)
    if callable(isolated):provider=isolated()
    for row in rows:
        views=[]
        for view in (0,1):
            remaining=max_seconds-(time.monotonic()-started)
            try:
                if remaining<1:raise TimeoutError()
                if progress:progress('source_ocr_retry','隔离重读存在分歧的原图标注；两次一致后才恢复绑定资格。')
                raw=provider.transcribe(image,[{'id':row['id'],'box':deepcopy(row['box'])}],
                    view=view,timeout=min(90.,remaining))
                safe={k:raw[k] for k in ('status','http_success','schema_success','http_status',
                    'network_requests','model','protocol','crops','elapsed_seconds','image_sent',
                    'ground_truth_sent','cad_values_sent','original_ocr_values_sent') if k in raw}
                if not _receipt_succeeded(safe):raise ValueError('retry_request_unsuccessful')
                safe['transcriptions']=_valid_transcriptions({'transcriptions':raw.get('transcriptions')},{row['id']})
            except InterruptedError:raise
            except Exception as error:
                safe=dict(status='failed',http_success=getattr(error,'http_success',None),schema_success=False,
                          failure_type=type(error).__name__,network_requests=getattr(error,'network_requests',None))
            views.append(safe)
        a,b=_review_texts(row['id'],views)
        receipt['attempts'].append(dict(record_id=row['id'],views=views,
            consensus=bool(a and b and _semantic(a)==_semantic(b))))
        save()
    receipt.update(status='completed',elapsed_seconds=round(time.monotonic()-started,3))
    return save(),receipt


def load_retry_ocr(image,parent,output):
    """Replay only the recorded two-view changes to a validated parent."""
    output=Path(output);audit=output/RETRY_FILE;effective=output/EFFECTIVE_FILE
    if not audit.exists() and not effective.exists():return parent
    try:
        receipt=json.loads(audit.read_text('utf8'));stored=json.loads(effective.read_text('utf8'))
        if (receipt.get('schema_version')!='source-ocr-isolated-retry-v1' or
                receipt.get('source_image_sha256')!=_hash(image) or
                receipt.get('parent_document_sha256')!=_object_hash(parent) or
                len(receipt['attempts'])>MAX_RECORDS or
                receipt.get('effective_file_sha256')!=_hash(effective) or
                receipt.get('effective_document_sha256')!=_object_hash(stored)):
            raise ValueError('retry_chain_identity')
        rebuilt,changed=_apply(parent,receipt['attempts'])
        if rebuilt!=stored or changed!=receipt['changed_record_ids']:raise ValueError('retry_replay_mismatch')
        return rebuilt
    except (OSError,ValueError,KeyError,TypeError,IndexError):
        raise ValueError('OCR isolated retry does not match its validated parent/source evidence') from None

"""Array-loader correction; no change to the paired training experiment."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
from mtare_topo import governance_bidirectional_relation_resume_v1 as previous

SCHEMA='v3_bidirectional_relation_resume_card_v2'
SLUG='gse_bidirectional_relation_resume_v2'
POLICY=deepcopy(previous.POLICY)
PARENT=previous.PARENT
ABORTED='results/gate3_semantics/gate3_20260909_gse_bidirectional_relation_resume_v1_seed0'
ABORTED_SEAL='f37f6f323be58c621cd18f8a40421f929bc00ede77eb7e0c422aa7a5018a630b'


def lineage(root):
    result=previous.lineage(root)
    seal=Path(root)/ABORTED/'artifacts/evidence_sha256.txt'
    if hashlib.sha256(seal.read_bytes()).hexdigest()!=ABORTED_SEAL:raise ValueError('loader-failure seal drift')
    entries=dict((p,h) for h,p in (s.split('  ',1) for s in seal.read_text().splitlines()))
    p=ABORTED+'/metrics/summary.json';raw=(Path(root)/p).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entries[p]:raise ValueError('loader-failure summary drift')
    summary=json.loads(raw)
    if summary['status']!='FAILED' or summary['updates']!={'c0':0,'c1':0} or summary['phase']!='load860':
        raise ValueError('loader failure must have zero new updates')
    result['pinned'].update({ABORTED+'/artifacts/evidence_sha256.txt':ABORTED_SEAL,p:entries[p]})
    result['loader_failure_run']=ABORTED
    return result


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    if not isinstance(card,dict):return ValidationReport(False,('card object required',))
    errors=[]
    if (card.get('schema_version'),card.get('card_id'))!=(SCHEMA,SLUG):errors.append('v2 identity mismatch')
    try:
        root=Path(__file__).resolve().parents[2]
        if card.get('lineage')!=lineage(root):errors.append('v2 lineage drift')
        translated=deepcopy(card)
        translated.update(schema_version=previous.SCHEMA,card_id=previous.SLUG,lineage=previous.lineage(root))
        errors.extend(previous.validate_card(translated).errors)
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))

"""Finalize only the externally interrupted, unsealed V8 comparison run.

The tool session 18559 exited 130 after explicit Ctrl-C. This does not resume
the experiment or rewrite any produced observation/configuration.
"""
import hashlib
import json
from pathlib import Path
from _bootstrap import PROJECT_ROOT

RUN = PROJECT_ROOT/'results/gate3_semantics/gate3_20260908_gse_v8_original_ten_probe_v1r_seed20260906'


def main():
    if (RUN/'artifacts/evidence_sha256.txt').exists() or (RUN/'metrics/summary.json').exists():
        raise FileExistsError('already finalized; no overwrite')
    if json.loads((RUN/'RUN_STATE.json').read_text())['state'] != 'RUNNING':
        raise ValueError('unexpected state')
    observations=[json.loads(line) for line in (RUN/'logs/observations.jsonl').read_text().splitlines()]
    if len(observations) != 4:
        raise ValueError('interrupted four-observation prefix changed')
    reason='Externally stopped after source inspection found implicit .01m distance fields in terminal-positive V5 and terminal-exclusion V7, despite V8 source contract .025m. Outputs invalid for whole-chain source-parameter qualification. No retry authorized by this finalizer.'
    summary=dict(status='ABORTED_SOURCE_PARAMETER_CONTRACT_INVALID',error=reason,
        tool_session=18559,tool_exit_code=130,completed_observations=len(observations),
        observations=observations,active_source=json.loads((RUN/'artifacts/active_observation.json').read_text()),
        last_completed_elapsed_s=observations[-1]['elapsed_s'],total_elapsed_s=None,peak_rss_bytes=None,
        finalization='External explicit finalization after terminated handle; not normal executor completion.',
        scientific_gate_pass=False,full_label_qualification=False,formal_optimizer_steps=0)
    for path,value in [('metrics/summary.json',summary),('RUN_STATE.json',dict(state='FAILED',run_id=RUN.name,error=reason))]:
        (RUN/path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n')
    (RUN/'logs/abort_reason.txt').write_text(reason+'\n')
    seal=RUN/'artifacts/evidence_sha256.txt'
    lines=[]
    for p in sorted(RUN.rglob('*')):
        if p.is_file() and p!=seal:
            lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(PROJECT_ROOT))+'\n')
    with seal.open('x') as stream:stream.writelines(lines)
    print(json.dumps(dict(state='FAILED',observations=4,sealed_files=len(lines),seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest())))


if __name__=='__main__':main()

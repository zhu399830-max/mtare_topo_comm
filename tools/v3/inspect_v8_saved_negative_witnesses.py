"""Read-only fixed four-observation audit; no labels/geometry rerender/writes."""
import gzip
import json
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_v8_probe_reader import V8ProbeReader
from mtare_topo.evaluation.gse_saved_negative_witness_audit import audit_saved_negative_witnesses
from mtare_topo.evaluation.gse_reference_chain_audit import audit_reference_chains

RUN='results/gate3_semantics/gate3_20260908_gse_v8_original_ten_probe_v1r_seed20260906'
SEAL_SHA='81e5dd95eb086f4bf6b5c9185dc8081bf8e0cf79013b7823027f2ff370cb0676'


def main():
    root=PROJECT_ROOT
    seal=read_pinned(root,RUN+'/artifacts/evidence_sha256.txt',SEAL_SHA)
    hashes={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    def read(path):return read_pinned(root,path,hashes[path])
    card=json.loads(read(RUN+'/config/data_card.json'))
    summary=json.loads(read(RUN+'/metrics/summary.json'))
    if summary['completed_observations']!=4 or summary['status']!='ABORTED_SOURCE_PARAMETER_CONTRACT_INVALID':
        raise ValueError('only the original four failed-run prefix allowed')
    reader=V8ProbeReader(root,card['scope'])
    total=0
    for item in summary['observations']:
        source=item['source'];task=source['task'];sequence=source['source_sequence_id']
        target=json.loads(gzip.decompress(read(RUN+'/artifacts/'+task+'_'+str(sequence)+'.json.gz')))['produced_targets']
        bundle,raw,old=reader.read_observation(task,sequence)
        report=audit_saved_negative_witnesses(target,bundle)
        report['independent_reference_chain_checks']=audit_reference_chains(target,bundle['construction_teacher_only'])
        report['reference_continuity_independently_rechecked']=True
        total+=report['checked_negative_candidates']
        print(json.dumps(report),flush=True)
    if total!=5:raise ValueError('five original candidates required')
    print('FIVE_SAVED_CANDIDATES_SOURCE_CHECK_COMPLETE_NOT_LABEL_QUALIFICATION',flush=True)


if __name__=='__main__':main()

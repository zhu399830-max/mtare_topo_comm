"""Metadata-only existing target inventory; never dereference raw-ray pointers."""
from _bootstrap import PROJECT_ROOT as ROOT
import gzip
import hashlib
import json
from collections import Counter,defaultdict
from check_local_pair_window_coverage import sha,write
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.gse_construction_continuations_v1 import construction_reference_continuations

RUN='results/gate3_semantics/gate3_20260908_gse_supplement_joint_v3_seed20260906'
CARD='configs/v3/gate3/data_cards/gse_supplement_joint_v3.json'
OUT='docs/figures/gse_local_pair_support_pilot_v1/existing_continuation_opportunities.json'


def main():
    if (ROOT/OUT).exists():raise FileExistsError('no overwrite')
    seal=RUN+'/artifacts/evidence_sha256.txt';pins={p:h for h,p in (l.split('  ',1) for l in (ROOT/seal).read_text().splitlines())}
    opened={seal:sha(ROOT/seal)}
    def read(p,h):
        raw=(ROOT/p).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('source drift '+p)
        opened[p]=h;return raw
    # Bind current source card to the immutable copy before reading source docs.
    archived=json.loads(read(RUN+'/config/data_card.json',pins[RUN+'/config/data_card.json']))
    if json.loads((ROOT/CARD).read_text())!=archived:raise ValueError('source card differs from archived run')
    scope=archived['scope'];counts=Counter();candidates=[];all_parents=defaultdict(set)
    expected=sum(len(e['observations']) for e in scope['entries'])
    if expected!=2676 or len(scope['entries'])!=210:raise ValueError('exact historical population required')
    for entry in scope['entries']:
        parent=entry['observations'][0]['parent_id'];split=entry['observations'][0]['split']
        if parent.rsplit('_',1)[-1] not in {'C01','C02','C03','C04','C05','C06','C07'}:raise ValueError('protected source')
        doc=json.loads(read(entry['construction_path'],scope['file_sha256'][entry['construction_path']]))
        references=construction_reference_continuations(doc,expected_document_sha256=canonical_sha(doc))
        components={s:c for c in references['continuations'] for s in c.source_ids}
        all_parents[split].add(parent)
        for source in entry['observations']:
            if source['split']!=split or source['parent_id']!=parent:raise ValueError('mixed source task')
            p=RUN+'/artifacts/'+source['task']+'_'+str(source['source_sequence_id'])+'.json.gz'
            packed=read(p,pins[p])
            raw=gzip.decompress(packed)
            if len(raw)>64*1024**2:raise ValueError('single target metadata exceeds64MiB')
            t=json.loads(raw)['produced_targets'];v=t['teacher_provenance'];record=t['record']
            if canonical_sha(record)!=t['target_record_sha256'] or record['source_frame_indices']!=source['frame_rows']:
                raise ValueError('target/frame binding')
            counts[split+':observations']+=1
            for ti,terminal in enumerate(v['terminals']):
                comp=components[terminal['endpoint_key_teacher_only'][0]]
                for oi,opening in enumerate(v['openings']):
                    counts[split+':terminal_opening_pairs']+=1
                    other=components[opening['primitive_id_teacher_only']]
                    if comp!=other:continue
                    if comp.unresolved_degree_two_nodes or comp.closed_reference_cycle or len(comp.structural_boundaries)!=2:
                        counts[split+':unresolved_continuation_pairs']+=1;continue
                    same=terminal['endpoint_key_teacher_only'][0]==opening['primitive_id_teacher_only']
                    candidates.append(dict(source=source,terminal_node=terminal['node_id_teacher_only'],
                        terminal_source=terminal['endpoint_key_teacher_only'][0],opening_source=opening['primitive_id_teacher_only'],
                        opening_index=oi,terminal_index=ti,source_ids=list(comp.source_ids),same_primitive=same,
                        original_membership=record['membership'][oi][v['terminal_anchor_start']+ti],evidence_path=p,evidence_sha256=pins[p]))
            del raw,t,v,record,packed
    summary={}
    for split in ('fit','calibration','development'):
        rows=[r for r in candidates if r['source']['split']==split]
        summary[split]=dict(observations=counts[split+':observations'],parents_read=len(all_parents[split]),
            terminal_opening_pairs=counts[split+':terminal_opening_pairs'],candidate_pairs=len(rows),
            candidate_windows=len({(r['source']['task'],r['source']['source_sequence_id']) for r in rows}),
            candidate_parents=len({r['source']['parent_id'] for r in rows}),
            independent_terminal_references=len({(r['source']['parent_id'],r['terminal_node']) for r in rows}),
            cross_primitive_pairs=sum(not r['same_primitive'] for r in rows),
            original_positive_pairs=sum(r['original_membership'] is True for r in rows),
            unknown_pairs=sum(r['original_membership'] is None for r in rows))
    result=dict(status='METADATA_CANDIDATES_NOT_OBSERVATION_QUALIFICATION',summary=summary,candidates=candidates,
        source_sha256=opened,targets_checked=expected,teacher_calls=0,ray_payload_reads=0,labels_changed=0,training_steps=0,
        limitations=['historical V6 targets, not V8 or trained predictions','source continuity is not same local window component or observed origin containment',
            'selection conditioned on teacher reference presence; not unbiased evaluation','no negative promotion from different source components'])
    write(ROOT/OUT,result)
    print(json.dumps(dict(summary=summary,files_verified=len(opened),target_records_checked=expected,teacher_calls=0)))


if __name__=='__main__':main()

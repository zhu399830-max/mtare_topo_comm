"""Join saved source intersections to saved lateral claims; no new labels."""
from _bootstrap import PROJECT_ROOT as ROOT
import collections, gzip, hashlib, json

RUN = 'results/gate3_semantics/gate3_20260911_gse_lateral_witness_recovery_single_thread_v1_seed0'
OUT = 'docs/figures/gse_supervision_acquisition_pilot_v1/lateral_source_bindings.json'


def read_bound(path, expected):
    raw = (ROOT / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('input drift: ' + path)
    return json.loads(gzip.decompress(raw))


def main():
    scope = json.loads((ROOT / 'configs/v3/gate3/gse_lateral_witness_recovery_scope_v1.json').read_text())
    pins = {p: h for h, p in (line.split('  ', 1) for line in
            (ROOT / RUN / 'artifacts/evidence_sha256.txt').read_text().splitlines())}
    rows = []
    for case, e in enumerate(scope['entries']):
        path = RUN + f'/artifacts/case_{case}.json.gz'
        recovered = read_bound(path, pins[path])
        reference = read_bound(e['reference']['path'], e['reference']['sha256'])
        if recovered['queried_rays'] != e['original_ray_indices']:
            raise ValueError('population drift')
        interfaces = {x['interface_id_teacher_only']: x for x in reference['raw_interfaces']['interfaces_teacher_only']}
        entries = recovered['entries']
        by_ray = collections.defaultdict(list)
        for index, record in enumerate(entries):
            by_ray[record['ray_index']].append(index)
        claims = collections.defaultdict(lambda: {'enter': [], 'depart': []})
        for anchor in reference['produced_targets']['teacher_provenance']['anchors']:
            for slot, interface in enumerate(anchor['interface_ids']):
                for role, key in [('enter', 'surface_entry_witness_ray_indices'), ('depart', 'surface_departure_witness_ray_indices')]:
                    for ray in anchor[key][slot]:
                        claims[ray][role].append({'interface': interface,
                            'node': anchor['node_id_teacher_only'],
                            'source': interfaces[interface]['endpoint_key_teacher_only'][0]})
        records = []; missing = 0; matched = 0; multiple = 0
        for ray in e['original_ray_indices']:
            claim = claims[ray]; joins = []
            for enter in claim['enter']:
                candidates = [j for j in by_ray[ray] if entries[j]['source_id_teacher_only'] == enter['source']]
                departing = [x for x in claim['depart'] if x['node'] == enter['node'] and x['source'] != enter['source']]
                missing += not bool(candidates); matched += bool(candidates); multiple += len(candidates) > 1
                joins.append(dict(enter=enter, saved_entry_indices=candidates, departure_claims=departing))
            records.append(dict(ray_index=ray, joins=joins, all_saved_entry_indices=by_ray[ray],
                                saved_claims=claim))
        row = dict(case=case, task=e['source']['task'], input_path=path, input_sha256=pins[path],
                   reference=e['reference'], matched_entry_claims=matched,
                   missing_entry_claims=missing, multi_hit_same_source_claims=multiple,
                   rays_without_enter_claim=sum(not r['joins'] for r in records), records=records)
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('records','reference')}))
    output = dict(status='SAVED_SOURCE_IDENTITY_JOIN_ONLY', rows=rows,
        not_revalidated=['unique origin containment', 'origin-source containment at entry',
                         'unique node/interface pair', 'direction dot products', 'node ambiguity'],
        interpretation='Source identity match is necessary, not sufficient for geometry or label qualification. Multiple entries and unknowns retained.',
        geometry_calls=0, training_steps=0, new_labels=0)
    with (ROOT / OUT).open('x') as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))


if __name__ == '__main__':
    main()

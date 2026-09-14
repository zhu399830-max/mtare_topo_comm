"""Read existing supplement provenance only; emit inventory, never labels."""
import collections
import gzip
import hashlib
import json
from pathlib import Path


def inspect(root):
    run = root / 'results/gate3_semantics/gate3_20260908_gse_supplement_joint_v3_seed20260906'
    card = root / 'configs/v3/gate3/data_cards/gse_supplement_joint_v3.json'
    scope = json.loads(card.read_text())['scope']
    sources = {}
    for entry in scope['entries']:
        for row in entry['observations']:
            key = (row['task'], row['source_sequence_id'])
            if key in sources:
                raise ValueError('duplicate source identity')
            sources[key] = row
    parents = {}
    rows = []
    digest = hashlib.sha256()
    kinds = ('entering_witness_ray_indices', 'interior_witness_ray_indices',
             'surface_entry_witness_ray_indices', 'surface_departure_witness_ray_indices')
    for task, sequence in sorted(sources):
        source = sources[(task, sequence)]
        parent = source['parent_id']
        if not parent.endswith(tuple('_C0' + str(i) for i in range(1, 8))):
            raise ValueError('out of development scope')
        path = run / 'artifacts' / f'{task}_{sequence}.json.gz'
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        data = json.loads(gzip.decompress(raw))
        target = data['produced_targets']
        binding = target['source_binding']['source']
        if binding != {k: source[k] for k in ('task', 'source_sequence_id', 'frame_rows')}:
            raise ValueError('source binding mismatch')
        provenance = target['teacher_provenance']
        junctions = provenance['anchors']
        terminals = provenance['terminals']
        if (provenance['terminal_anchor_start'] != len(junctions)
                or len(target['record']['anchors']) != len(junctions) + len(terminals)):
            raise ValueError('mixed anchor provenance alignment mismatch')
        stat = parents.setdefault(parent, dict(split=source['split'], counts=collections.Counter(),
                                              junctions=set(), terminals=set()))
        if stat['split'] != source['split']:
            raise ValueError('parent crosses split')
        counts = collections.Counter(observations=1, junction_occurrences=len(junctions),
                                     terminal_occurrences=len(terminals))
        ids = set()
        for anchor in junctions:
            stat['junctions'].add(anchor['node_id_teacher_only'])
            interface_ids = anchor['interface_ids']
            witnesses = anchor['witness_ray_indices']
            if len(interface_ids) != len(witnesses):
                raise ValueError('witness/interface length mismatch')
            for key in kinds:
                if key in anchor and len(anchor[key]) != len(interface_ids):
                    raise ValueError('witness kind alignment mismatch')
            for i, interface in enumerate(interface_ids):
                if interface in ids:
                    raise ValueError('duplicate interface across anchors')
                ids.add(interface)
                rays = witnesses[i]
                if rays != sorted(set(rays)) or any(type(v) is not int or not 0 <= v < 57600 for v in rays):
                    raise ValueError('invalid saved witness indices')
                counts['branch_occurrences'] += 1
                counts['branches_with_witness'] += bool(rays)
                for key in kinds:
                    if key in anchor:
                        if not set(anchor[key][i]).issubset(rays):
                            raise ValueError('kind outside combined evidence')
                        counts[key + '_nonempty_branches'] += bool(anchor[key][i])
        for terminal in terminals:
            stat['terminals'].add(terminal['node_id_teacher_only'])
            counts['terminals_with_witness'] += bool(terminal['witnesses'])
        counts['records_with_junction'] += bool(junctions)
        counts['records_with_terminal'] += bool(terminals)
        counts['mixed_anchor_records'] += bool(junctions and terminals)
        stat['counts'].update(counts)
        rows.append(dict(path=str(path.relative_to(root)), sha256=sha, parent=parent,
                         split=source['split'], source=binding, counts=dict(counts),
                         raw_interfaces_reference=data['raw_interfaces_reference']))
    # Same filename-sorted digest as the previous full record inventory.
    for row in sorted(rows, key=lambda r: Path(r['path']).name):
        digest.update((Path(row['path']).name + ' ' + row['sha256'] + '\n').encode())
    expected = '585b7ce46dcf60427919b4f2ecf1563aaccc477f6334043636f933ed7303cf98'
    if len(rows) != 2676 or digest.hexdigest() != expected:
        raise ValueError('saved population drift')
    splits = {}
    for parent, stat in parents.items():
        stat['independent_junctions'] = len(stat.pop('junctions'))
        stat['independent_terminals'] = len(stat.pop('terminals'))
        out = splits.setdefault(stat['split'], collections.Counter())
        out.update(stat['counts'])
        out.update(parents=1, independent_junctions=stat['independent_junctions'],
                   independent_terminals=stat['independent_terminals'])
    return dict(status='SAVED_PROVENANCE_INVENTORY_NOT_LABEL_QUALIFICATION',
                record_inventory_sha256=expected, by_split=splits, by_parent=parents,
                rows=rows, new_teacher_execution=False, training_eligible=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--details', action='store_true', help='include per-parent and per-record metadata')
    args = parser.parse_args()
    result = inspect(Path(__file__).resolve().parents[2])
    if not args.details:
        result.pop('rows')
        result.pop('by_parent')
    print(json.dumps(result, ensure_ascii=False, indent=2))

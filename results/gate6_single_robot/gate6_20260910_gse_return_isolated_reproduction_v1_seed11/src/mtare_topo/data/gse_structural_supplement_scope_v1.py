"""Compile exact supplementary pose permissions, without reading pose chunks.

Reads only C01-C07 sealed identity/construction metadata and XYZ headers.
No scan, model, label, or protected world payload may be opened here.
"""
from collections import Counter
from pathlib import Path

from mtare_topo.governance_identity_inventory import P1A
from mtare_topo.governance_surface_selection import INVENTORY, PARENTS, SEAL_SHA256
from mtare_topo.governance_surface_input import SEALS, expected_tasks
from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths
from .gse_structural_supplement_v1 import nominate_parent
from .gse_surface_input_scope_v1 import checked_read, _object


def plan_xyz_rows(header, rows, count):
    if (type(count) is not int or count < 1 or header.get('shape') != [count, 3]
            or header.get('chunks') != [min(4096, count), 3]
            or header.get('dtype') != '<f8' or header.get('zarr_format') != 2
            or header.get('order') != 'C' or header.get('dimension_separator', '.') != '.'):
        raise ValueError('original float64 XYZ array required')
    if (type(rows) is not list or not rows or rows != sorted(set(rows))
            or any(type(r) is not int or not 0 <= r < count for r in rows)):
        raise ValueError('sorted unique in-bounds nonempty pose rows required')
    chunk = header['chunks'][0]
    indices = sorted({r // chunk for r in rows})
    return dict(selected_rows=rows, chunk_keys=[f'{i}.0' for i in indices],
        decoded_rows_including_collateral=sum(min(count, (i+1)*chunk)-i*chunk for i in indices),
        decoded_padded_bytes=len(indices)*chunk*24, shape=[count, 3], chunks=[chunk, 3], dtype='<f8')


def _index(raw, allowed):
    output = {}
    for line in raw.decode().splitlines():
        sha, path = line.split(None, 1)
        if not allowed(path):
            continue
        if path in output:
            raise ValueError('duplicate selected source in seal')
        output[path] = sha
    return output


def compile_supplement_scope(root):
    root = Path(root).resolve(strict=True)
    reads = {}
    identity_paths = {INVENTORY+'/artifacts/'+p+'_identity_intervals.json' for p in PARENTS}
    split_path = INVENTORY+'/artifacts/parent_split.json'
    identity_paths.add(split_path)
    index = _index(checked_read(root, INVENTORY+'/artifacts/evidence_sha256.txt',
                              SEAL_SHA256, reads), lambda p: p in identity_paths)
    if set(index) != identity_paths:
        raise ValueError('exact parent inventory and split missing')
    split = _object(checked_read(root, split_path, index[split_path], reads))
    if set(split) != set(PARENTS) or Counter(split.values()) != {'fit': 60, 'calibration': 5, 'development': 5}:
        raise ValueError('original sealed parent split required')
    tasks = expected_tasks()
    constructions = {t: P1A+'/artifacts/constructions/'+s['partition']+'/'+t+'.json' for t,s in tasks.items()}
    prefixes = {s['sensor']+'/sensor_xyz_m/' for s in tasks.values()}
    construction_set = set(constructions.values())
    seal = SEALS['sensor']
    source_index = _index(checked_read(root, seal['path'], seal['sha256'], reads),
                         lambda p: p in construction_set or any(p.startswith(x) for x in prefixes))
    files, plans, parents, entries = {}, {}, [], []
    for parent in PARENTS:
        path = INVENTORY+'/artifacts/'+parent+'_identity_intervals.json'
        report = _object(checked_read(root, path, index[path], reads))
        if report['parent_id'] != parent:
            raise ValueError('identity report path mismatch')
        groups = {}
        parent_tasks = {t:s for t,s in tasks.items() if s['parent_id'] == parent}
        for task, source in sorted(parent_tasks.items()):
            path = constructions[task]
            document = _object(checked_read(root, path, source_index[path], reads))
            if document.get('parent_id') != parent or document.get('geometry_realization') != source['variant']:
                raise ValueError('construction parent/variant binding mismatch')
            groups[source['variant']] = construction_incident_paths(document)
        nominations = nominate_parent(report, split[parent], groups)
        parents.append(dict(parent_id=parent, split=split[parent], nominations=nominations))
        for task, source in sorted(parent_tasks.items()):
            variant = source['variant']
            rows = sorted({frame[-1] for n in nominations for r in n['variant_records']
                           if r['variant'] == variant for frame in r['frame_rows']})
            if not rows:
                continue  # No substitute history; nominations retain the gap.
            prefix = source['sensor']+'/sensor_xyz_m'
            header_path = prefix+'/.zarray'
            header = _object(checked_read(root, header_path, source_index[header_path], reads))
            plan = plan_xyz_rows(header, rows, report['counts']['variants'][variant]['frames'])
            plans[prefix] = plan
            for path in [header_path, *(prefix+'/'+k for k in plan['chunk_keys'])]:
                files[path] = source_index[path]
            entries.append(dict(task=task, parent_id=parent, variant=variant, xyz_prefix=prefix,
                anchor_world_m_by_node={g['node_id_teacher_only']:g['anchor_world_m'] for g in groups[variant]}))
    counts = Counter((p['split'], n['kind']) for p in parents for n in p['nominations'])
    # Recheck metadata after compilation to detect concurrent source drift.
    for path, sha in list(reads.items()):
        checked_read(root, path, sha, {})
    return dict(schema='gse_structural_supplement_scope_v1',
        status='PREPARATION_NOT_EXECUTION_AUTHORITY', parents=parents, entries=entries,
        file_sha256=files, array_access=plans, metadata_reads_sha256=reads,
        counts=dict(parents=len(parents), tasks=len(entries),
            entity_counts={s:{k:counts[s,k] for k in ('junction','terminal')} for s in ('fit','calibration','development')},
            candidate_traversals=len({n['traversal_id'] for p in parents for n in p['nominations'] if n['traversal_id']}),
            candidate_variant_decisions=sum(len(p['selected_rows']) for p in plans.values()),
            pose_chunk_files=sum(len(p['chunk_keys']) for p in plans.values()),
            decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in plans.values()),
            pose_payload_reads=0, scan_reads=0, observations_selected=0, labels=0),
        restrictions=['C01_C07_only', 'no_scan_model_or_teacher_labels', 'fixed_original_parent_split',
                      'pose_proximity_is_not_visibility', 'retain_missing_and_far', 'no_training'])

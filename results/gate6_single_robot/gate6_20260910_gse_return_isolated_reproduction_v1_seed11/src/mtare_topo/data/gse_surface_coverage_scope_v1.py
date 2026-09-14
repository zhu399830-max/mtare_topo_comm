"""Fixed3360 geometric population scope: construction and current XYZ only.

Compilation reads selection, sealed indexes and XYZ array headers; it does not
read construction bodies, pose chunks, scans or labels. Nearby is not visible.
"""
from pathlib import Path

from mtare_topo.governance_identity_inventory import P1A
from mtare_topo.governance_surface_input import SELECTION, SELECTION_SEAL, SEALS, expected_tasks
from .gse_surface_input_scope_v1 import checked_read, _object, selected_rows


def plan_current_xyz(header, decision_rows, count):
    if (type(count) is not int or count < 1 or header.get('shape') != [count, 3]
            or header.get('chunks') != [min(4096, count), 3]
            or header.get('dtype') != '<f8' or header.get('zarr_format') != 2
            or header.get('order') != 'C' or header.get('dimension_separator', '.') != '.'):
        raise ValueError('original float64 sensor XYZ header required')
    if (type(decision_rows) is not list or len(decision_rows) != 16
            or any(type(r) is not int or not 0 <= r < count for r in decision_rows)
            or len(set(decision_rows)) != 16):
        raise ValueError('sixteen unique fixed decision rows required')
    chunk = header['chunks'][0]
    indices = sorted({r // chunk for r in decision_rows})
    return dict(selected_rows=decision_rows, chunk_keys=[f'{i}.0' for i in indices],
                decoded_rows_including_collateral=sum(min(count, (i+1)*chunk)-i*chunk for i in indices),
                decoded_padded_bytes=len(indices)*chunk*3*8, shape=[count, 3], chunks=[chunk, 3], dtype='<f8')


def compile_coverage_scope(root):
    root = Path(root).resolve(strict=True)
    reads = {}
    selection_path = SELECTION+'/artifacts/selection_manifest.json'
    seal = checked_read(root, SELECTION+'/artifacts/evidence_sha256.txt', SELECTION_SEAL, reads)
    hits = [line.split('  ', 1)[0] for line in seal.decode().splitlines()
            if line.endswith('  '+selection_path)]
    if len(hits) != 1:
        raise ValueError('unique sealed selection required')
    manifest = _object(checked_read(root, selection_path, hits[0], reads))
    tasks = expected_tasks()
    rows = selected_rows(manifest, tasks)
    source_seal = SEALS['sensor']
    index = checked_read(root, source_seal['path'], source_seal['sha256'], reads)
    construction_paths = {t: P1A+'/artifacts/constructions/'+s['partition']+'/'+t+'.json'
                          for t, s in tasks.items()}
    prefixes = {s['sensor']+'/sensor_xyz_m/' for s in tasks.values()}
    wanted_json = set(construction_paths.values())
    candidates = {}
    for line in index.decode().splitlines():
        h, p = line.split(None, 1)
        if p not in wanted_json and not any(p.startswith(prefix) for prefix in prefixes):
            continue
        if p in candidates:
            raise ValueError('duplicate selected source hash')
        candidates[p] = h
    plans = {}; files = {}; entries = []
    for task, source in sorted(tasks.items()):
        construction = construction_paths[task]
        if construction not in candidates:
            raise ValueError('sealed construction missing')
        files[construction] = candidates[construction]
        prefix = source['sensor']+'/sensor_xyz_m'
        path = prefix+'/.zarray'
        header = _object(checked_read(root, path, candidates[path], reads))
        count_values = {r['source_frame_count'] for r in rows[task]}
        if len(count_values) != 1:
            raise ValueError('task frame count mismatch')
        decisions = [r['frame_rows'][-1] for r in rows[task]]
        plan = plan_current_xyz(header, decisions, count_values.pop())
        plans[prefix] = plan
        for p in [path, *(prefix+'/'+key for key in plan['chunk_keys'])]:
            if p not in candidates:
                raise ValueError('selected pose chunk missing')
            files[p] = candidates[p]
        entries.append(dict(task=task, construction_path=construction, xyz_prefix=prefix,
                            observations=rows[task]))
    return dict(schema='gse_surface_coverage_scope_v1', status='PREPARATION_NOT_EXECUTION_AUTHORITY',
                entries=entries, file_sha256=files, array_access=plans,
                metadata_reads_sha256=reads, source_seal=source_seal,
                counts=dict(parents=70, tasks=210, physical_edge_units=1120, observations=3360,
                            selected_current_xyz_rows=3360, construction_files=210, xyz_headers=210,
                            pose_chunk_files=sum(len(p['chunk_keys']) for p in plans.values()),
                            decoded_xyz_rows_including_collateral=sum(p['decoded_rows_including_collateral'] for p in plans.values()),
                            decoded_padded_xyz_bytes=sum(p['decoded_padded_bytes'] for p in plans.values()),
                            actual_payload_reads=0, scans=0, labels=0),
                restrictions=['no_C08_C10_payload', 'no_scans_or_membership_or_models',
                              'no_sample_reselection', 'nearby_not_observed', 'no_training'],
                radius_m=10., spacing='Original fixed selected decisions; no new spatial or temporal spacing claim.')

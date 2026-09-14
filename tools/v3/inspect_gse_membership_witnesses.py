"""Read-only, fixed ten-observation diagnostic. Does not produce labels.

Run with PYTHONPATH=src and the existing Cano sidecar. Only sealed C01/C03
development artifacts are read; no scan projection, model, or output writes.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import zarr

from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.gse_construction_continuations_v1 import construction_reference_continuations
from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths
from mtare_topo.teacher.gse_directed_interface_binding_v1 import bind_axes
from mtare_topo.data.gse_surface_input_export_v1 import _ExactStore
from mtare_topo.teacher.gse_directed_interface_binding_v1 import interpret_bound_result
from mtare_topo.teacher.gse_branch_transition_witness_v1 import branch_transition_witness
from mtare_topo.teacher.gse_caster_cap_replay_v1 import pack_caster_inputs
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions


def main():
    root = Path(__file__).resolve().parents[2]
    run = Path('results/gate3_semantics/gate3_20260908_gse_supplement_joint_v3_seed20260906')
    seal_bytes = (root / run / 'artifacts/evidence_sha256.txt').read_bytes()
    assert hashlib.sha256(seal_bytes).hexdigest() == 'c5c5f5fd8cfa48a10ee2c9726c1c0b1ac9785fd0b696db8a07403ac9bb71c8af'
    seal = {p: h for h, p in (line.split('  ', 1) for line in seal_bytes.decode().splitlines())}

    def read(path, sha):
        p = root / path
        assert p.resolve().is_relative_to(root)
        data = p.read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha, path
        return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)

    reads_path = str(run / 'artifacts/source_reads_sha256.json')
    reads = read(reads_path, seal[reads_path])
    variants = ('c1_mixed', 'ellipse', 'rounded_rectangle')
    selected = {
        **{f'S04_3d_unicyclic_small_C01__{v}': [36237] for v in variants},
        'S07_flat_loop_rich_C03__rounded_rectangle': [97000],
        **{f'S10_3d_complex_C01__{v}': [178898, 178901] for v in variants},
    }
    for task, sequences in selected.items():
        paths = [p for p in reads if '/constructions/' in p and Path(p).stem == task]
        assert len(paths) == 1
        document = read(paths[0], reads[paths[0]])
        reference = construction_reference_continuations(document, expected_document_sha256=canonical_sha(document))
        components = {source: c for c in reference['continuations'] for source in c.source_ids}
        for sequence in sequences:
            path = str(run / 'artifacts' / f'{task}_{sequence}.json.gz')
            result = read(path, seal[path])
            target = result['produced_targets']
            provenance = target['teacher_provenance']
            bound = target['source_binding']['source']
            pin = result['raw_interfaces_reference']
            original = read(pin['path'], pin['sha256'])
            if 'raw_interfaces' not in original:
                pin = original['raw_interfaces_reference']
                original = read(pin['path'], pin['sha256'])
            raw = original['raw_interfaces']
            assert set(bound) == {'task', 'source_sequence_id', 'frame_rows'}
            assert {key: raw['source'][key] for key in bound} == bound
            assert bound['task'] == task and bound['source_sequence_id'] == sequence
            assert len(provenance['anchors']) == len(provenance['terminals']) == 1
            junction = provenance['anchors'][0]
            terminal = provenance['terminals'][0]
            caps = {w['ray_index']: w['stored_t'] for w in terminal['witnesses']}
            witness = dict(zip(junction['interface_ids'], map(set, junction['witness_ray_indices'])))
            cap_cross = {hit['ray_index'] for hit in raw['raw_interface_intersections']
                         if hit['ray_index'] in caps and hit['inside_roi']
                         and 0 <= hit['t'] < caps[hit['ray_index']]
                         and hit['ray_index'] in witness.get(hit['interface_id_teacher_only'], set())}
            interior = set().union(*map(set, junction['interior_witness_ray_indices']))
            terminal_component = components[terminal['endpoint_key_teacher_only'][0]]
            direction_detail = None
            if task.startswith('S10_3d_complex_C01__'):
                # Original sealed pose chunks only; no scanner invocation.
                headers = [p for p in reads if f'/{task}.zarr/sensor_xyz_m/.zarray' in p]
                assert len(headers) == 1
                prefix = headers[0].removesuffix('/.zarray')
                store = _ExactStore(root, prefix, reads, {})
                store.allowed = {'.zarray'}
                header = json.loads(store['.zarray'])
                chunks = {'.'.join(map(str, (row // header['chunks'][0], 0)))
                          for row in bound['frame_rows']}
                store.allowed = {'.zarray', *chunks}
                origins = np.asarray(zarr.Array(store=store, read_only=True).oindex[bound['frame_rows']])
                axes = bind_axes(construction_incident_paths(document), raw['interfaces_teacher_only'])
                terminal_source = terminal['endpoint_key_teacher_only'][0]
                interface = [a for a in axes if a['node_id_teacher_only'] == junction['node_id_teacher_only']
                             and a['source_key_teacher_only'] == terminal_source]
                assert len(interface) == 1
                interface = interface[0]
                identifier = interface['interface_id_teacher_only']
                raw_interface = next(a for a in raw['interfaces_teacher_only']
                                     if a['interface_id_teacher_only'] == identifier)
                offset = origins - np.asarray(raw_interface['anchor_world_m'])
                hits = [h for h in raw['raw_interface_intersections']
                        if h['interface_id_teacher_only'] == identifier and h['ray_index'] in caps]
                outgoing = {}
                for hit in raw['raw_interface_intersections']:
                    if hit['interface_id_teacher_only'] == identifier and hit['inside_roi'] and hit['t'] > 0:
                        # Sign from the original float32 caster origin and
                        # archived intersection, not a newly projected ray.
                        slot = hit['ray_index'] // 11520
                        displacement = np.asarray(hit['intersection_world_m']) - origins[slot].astype(np.float32).astype(np.float64)
                        if displacement @ np.asarray(interface['inward_direction']) < 0:
                            outgoing.setdefault(hit['ray_index'], []).append(hit['t'])
                ordered = set()
                for hit in raw['raw_interface_intersections']:
                    ray = hit['ray_index']
                    if (hit['inside_roi'] and ray in witness.get(hit['interface_id_teacher_only'], set())
                            and any(0 < t < hit['t'] for t in outgoing.get(ray, []))):
                        ordered.add(ray)
                direction_detail = {
                    'sensor_signed_distance_along_terminal_branch_m': (offset @ np.asarray(interface['inward_direction'])).tolist(),
                    'terminal_branch_interface_in_supported_junction': identifier in witness,
                    'cap_rays_intersecting_terminal_branch_interface': len({h['ray_index'] for h in hits}),
                    'cap_rays_intersecting_before_return_inside_roi': len({h['ray_index'] for h in hits
                        if h['inside_roi'] and 0 <= h['t'] < caps[h['ray_index']]}),
                    'pose_chunk_keys_read': sorted(chunks),
                    'opening_rays_leaving_terminal_then_entering_supported_junction': [
                        len(ordered & set(p.get('ray_indices', []))) for p in provenance['relations']],
                }
                sensor = {'sensor_xyz_m': origins}
                for field in ('yaw_deg', 'primitive_membership_code'):
                    field_prefix = prefix.rsplit('/', 1)[0] + '/' + field
                    field_store = _ExactStore(root, field_prefix, reads, {})
                    field_store.allowed = {'.zarray'}
                    field_header = json.loads(field_store['.zarray'])
                    field_store.allowed |= {
                        '.'.join(map(str, (row // field_header['chunks'][0],) + (0,) * (len(field_header['shape']) - 1)))
                        for row in bound['frame_rows']}
                    sensor[field] = np.asarray(zarr.Array(store=field_store, read_only=True).oindex[bound['frame_rows']])
                input_paths = [p for p in reads if p.endswith('/' + task + '.npz')]
                assert len(input_paths) == 1
                input_bytes = (root / input_paths[0]).read_bytes()
                assert hashlib.sha256(input_bytes).hexdigest() == reads[input_paths[0]]
                with np.load(io.BytesIO(input_bytes), allow_pickle=False) as archive:
                    indices = np.flatnonzero(archive['source_sequence_ids'] == sequence)
                    assert len(indices) == 1
                    student = {k: archive[k][indices[0]].copy() for k in archive.files}
                assert student['frame_rows'].tolist() == bound['frame_rows']
                books = [p for p in reads if '/codebooks/' in p and Path(p).stem == task]
                assert len(books) == 1
                book = read(books[0], reads[books[0]])
                assert canonical_sha(document) == target['source_binding']['construction_sha256']
                bundle = dict(sensor_teacher_only=sensor, student=student,
                    construction_teacher_only=document, codebook_teacher_only=book, source=raw['source'])
                # Checks raw intersection XYZ, original ROI, poses, frame IDs,
                # return-source code alignment and first-return validity.
                interpret_bound_result(bundle, raw)
                local = lidar_local_directions().reshape(-1, 3).astype(np.float64)
                directions = np.concatenate([world_directions(local, float(y)) for y in sensor['yaw_deg']])
                _, packed = pack_caster_inputs(np.empty((0, 3)), np.repeat(origins, 11520, axis=0), directions)
                owners = [[book['primitive_ids'][i] for i in book['source_sets'][int(code)]]
                          for code in sensor['primitive_membership_code'].reshape(-1)]
                compatible = set()
                for other in junction['interface_ids']:
                    if other == identifier:
                        continue
                    transition = branch_transition_witness(raw['raw_interface_intersections'], axes,
                        leaving_interface=identifier, entering_interface=other,
                        directions=packed[:, 3:].astype(np.float64), first_return=student['ranges_m'].reshape(-1),
                        valid=student['valid_mask'].reshape(-1).astype(bool), return_sources=owners)
                    compatible.update(transition['ray_indices'])
                direction_detail['source_bound_direction_compatible_opening_counts'] = [
                    len(compatible & set(p.get('ray_indices', []))) for p in provenance['relations']]
            print(json.dumps({
                'task': task, 'sequence': sequence,
                'cap_rays': len(caps), 'cap_cross_junction_witness_before_return': len(cap_cross),
                'cap_intersects_junction_interior_witness': len(set(caps) & interior),
                'terminal_continuation_reaches_junction_reference': any(
                    node == junction['node_id_teacher_only']
                    for node, _, _ in terminal_component.structural_boundaries),
                'terminal_continuation_sources': len(terminal_component.source_ids),
                'terminal_continuation_unresolved': list(terminal_component.unresolved_degree_two_nodes),
                'opening_junction_witness_counts': [len(p.get('ray_indices', [])) for p in provenance['relations']],
                'negative_labels_produced': 0,
                'S10_direction_diagnostic': direction_detail,
                'interpretation': 'Necessary witness diagnostics only; no physical separation or membership certificate.',
            }), flush=True)


if __name__ == '__main__':
    main()

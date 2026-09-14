"""Compile only the previously diagnosed ten observations; no new labels.

Reads sealed observation metadata and Zarr headers, not scan chunks or
construction payloads. This is preparation, not experiment authority.
"""
import gzip
import json
from pathlib import Path

from mtare_topo.governance_surface_material import read_pinned
from .gse_surface_teacher_scope_v1 import FIELDS, plan_field

RUN = 'results/gate3_semantics/gate3_20260908_gse_supplement_joint_v3_seed20260906'
SEAL_SHA = 'c5c5f5fd8cfa48a10ee2c9726c1c0b1ac9785fd0b696db8a07403ac9bb71c8af'
VARIANTS = ('c1_mixed', 'ellipse', 'rounded_rectangle')
SELECTED = {
    **{f'S04_3d_unicyclic_small_C01__{v}': [36237] for v in VARIANTS},
    'S07_flat_loop_rich_C03__rounded_rectangle': [97000],
    **{f'S10_3d_complex_C01__{v}': [178898, 178901] for v in VARIANTS},
}


def compile_scope(root):
    root = Path(root).resolve(strict=True)
    opened = {}

    def read(path, sha):
        data = read_pinned(root, path, sha)
        opened[path] = sha
        return data

    seal_path = RUN + '/artifacts/evidence_sha256.txt'
    seal = {}
    for line in read(seal_path, SEAL_SHA).decode().splitlines():
        h, p = line.split('  ', 1)
        if p in seal:
            raise ValueError('duplicate seal path')
        seal[p] = h
    index_path = RUN + '/artifacts/source_reads_sha256.json'
    index = json.loads(read(index_path, seal[index_path]))
    card_path = RUN + '/config/data_card.json'
    card = json.loads(read(card_path, seal[card_path]))
    containers = {r['task']: r for r in card['scope']['entries'] if r['task'] in SELECTED}
    if set(containers) != set(SELECTED):
        raise ValueError('missing sealed input container population')
    files = {}; entries = []; frames = set()

    def unique(predicate):
        paths = [p for p in index if predicate(p)]
        if len(paths) != 1:
            raise ValueError('source path missing or ambiguous')
        files[paths[0]] = index[paths[0]]
        return paths[0]

    for task, sequences in SELECTED.items():
        construction = unique(lambda p: '/constructions/fit/' in p and Path(p).stem == task)
        codebook = unique(lambda p: '/codebooks/fit/' in p and Path(p).stem == task)
        inputs = unique(lambda p: p.endswith('/' + task + '.npz'))
        for sequence in sequences:
            path = RUN + f'/artifacts/{task}_{sequence}.json.gz'
            result = json.loads(gzip.decompress(read(path, seal[path])))
            references = [{'path': path, 'sha256': seal[path]}]
            current = result
            while 'raw_interfaces' not in current:
                if len(references) >= 3 or 'raw_interfaces_reference' not in current:
                    raise ValueError('unexpected raw-interface reference chain')
                pin = current['raw_interfaces_reference']
                if pin['path'] in {p['path'] for p in references}:
                    raise ValueError('cyclic raw-interface reference')
                current = json.loads(gzip.decompress(read(pin['path'], pin['sha256'])))
                references.append(pin)
            source = current['raw_interfaces']['source']
            if source['task'] != task or source['source_sequence_id'] != sequence:
                raise ValueError('raw observation identity mismatch')
            bound = result['produced_targets']['source_binding']['source']
            if {k: source[k] for k in bound} != bound:
                raise ValueError('old target / raw source mismatch')
            plans = {}
            for field in FIELDS:
                header_path = unique(lambda p: f'/{task}.zarr/{field}/.zarray' in p)
                header = json.loads(read(header_path, index[header_path]))
                plan = plan_field(header, field, source['frame_rows'], source['source_frame_count'])
                prefix = header_path.removesuffix('/.zarray')
                for chunk in plan['chunk_keys']:
                    p = prefix + '/' + chunk
                    if p not in index:
                        raise ValueError('required chunk absent from sealed read index')
                    files[p] = index[p]
                plans[field] = dict(prefix=prefix, **plan)
            frames.update((task, row) for row in source['frame_rows'])
            entries.append(dict(source=source, references=references, arrays=plans,
                construction_path=construction, codebook_path=codebook, input_path=inputs,
                container_observations=len(containers[task]['observations'])))
    return dict(schema='gse_v8_fixed_ten_probe_scope_v1', entries=entries,
        file_sha256=files, metadata_reads_sha256=opened,
        geometry_settings=dict(axial_spacing_m=.05, angular_segments=64, field_spacing_m=.025),
        counts=dict(parents=3, tasks=7, observations=len(entries), frame_occurrences=5*len(entries),
                    unique_variant_frames=len(frames), newly_generated_labels=0,
                    physical_traversals=len({r['source']['traversal_id'] for r in entries}),
                    container_observations=sum(len(r['observations']) for r in containers.values())),
        status='PREPARATION_ONLY_NOT_LABEL_QUALIFICATION',
        sampling='All original ten multi-anchor failures; not representative prevalence or independent ten places.',
        spacing='Original five causal frame rows retained; no newly asserted temporal/metric spacing.',
        collateral='NPZ containers include the reported container population; Zarr collateral rows/bytes are recorded per array plan. Only the ten selected observations may be evaluated.')

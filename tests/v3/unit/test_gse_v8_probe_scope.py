"""Metadata compilation never opens scan chunks, inputs, or constructions."""
import gzip
import hashlib
import json

import pytest

from mtare_topo.data import gse_v8_probe_scope as probe


def fixture(monkeypatch, *, source_drift=False, missing_chunk=False):
    data = {}; seal = {}; index = {}; rows = []

    def put(path, obj, *, compressed=False, sealed=False):
        raw = json.dumps(obj).encode()
        raw = gzip.compress(raw) if compressed else raw
        data[path] = raw
        h = hashlib.sha256(raw).hexdigest()
        if sealed:
            seal[path] = h
        return h

    for task, sequences in probe.SELECTED.items():
        for role in ('constructions', 'codebooks'):
            index[f'original/{role}/fit/{task}.json'] = 'a'*64
        index[f'inputs/{task}.npz'] = 'b'*64
        sources = []
        for sequence in sequences:
            source = dict(task=task, source_sequence_id=sequence, frame_rows=[0,1,2,3,4],
                          source_frame_count=32, traversal_id=task.split('__')[0]+':edge')
            sources.append(source)
            path = probe.RUN + f'/artifacts/{task}_{sequence}.json.gz'
            bound = dict(source, task='WRONG') if source_drift else source
            put(path, dict(raw_interfaces={'source': source}, produced_targets={
                'source_binding': {'source': bound}}), compressed=True, sealed=True)
        rows.append(dict(task=task, observations=sources))
        for field, (tail, dtype, _) in probe.FIELDS.items():
            prefix = f'original/dataset/fit/{task}.zarr/{field}'
            chunks = [16 if field == 'primitive_membership_code' else 32] + tail
            index[prefix+'/.zarray'] = put(prefix+'/.zarray', dict(shape=[32]+tail,
                chunks=chunks, dtype=dtype, zarr_format=2, order='C'))
            if not missing_chunk:
                index[prefix+'/' + '.'.join(['0']*len(chunks))] = 'c'*64
    put(probe.RUN+'/config/data_card.json', {'scope': {'entries': rows}}, sealed=True)
    put(probe.RUN+'/artifacts/source_reads_sha256.json', index, sealed=True)
    seal_path = probe.RUN+'/artifacts/evidence_sha256.txt'
    data[seal_path] = ''.join(h+'  '+p+'\n' for p,h in seal.items()).encode()
    monkeypatch.setattr(probe, 'SEAL_SHA', hashlib.sha256(data[seal_path]).hexdigest())
    reads = []

    def read(root, path, sha):
        reads.append(path)
        raw = data[path]  # no payload keys exist, so accidental reads fail
        assert hashlib.sha256(raw).hexdigest() == sha
        return raw

    monkeypatch.setattr(probe, 'read_pinned', read)
    return reads


def test_exact_metadata_only_scope(tmp_path, monkeypatch):
    reads = fixture(monkeypatch)
    scope = probe.compile_scope(tmp_path)
    assert scope['counts']['observations'] == 10
    assert scope['counts']['physical_traversals'] == 3
    assert scope['counts']['container_observations'] == 10
    assert scope['geometry_settings']['field_spacing_m'] == .025
    assert all(not p.endswith('.npz') and '/constructions/' not in p for p in reads)
    assert probe.compile_scope(tmp_path) == scope


@pytest.mark.parametrize('defect', ['source_drift', 'missing_chunk'])
def test_mismatched_source_or_missing_chunk_fails(tmp_path, monkeypatch, defect):
    fixture(monkeypatch, **{defect: True})
    with pytest.raises(ValueError):
        probe.compile_scope(tmp_path)

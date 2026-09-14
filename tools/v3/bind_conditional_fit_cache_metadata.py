"""Bind sealed existing fit cache metadata; no NPZ decode or label generation."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json

BASE = 'results/gate3_semantics/gate3_20260907_gse_surface_input_export_v1r_seed20260906'
POP = 'docs/figures/gse_conditional_geometry_fit_v1/fit_population_metadata.json'
OUT = 'docs/figures/gse_conditional_geometry_fit_v1/fit_cache_binding.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pop = json.loads((ROOT / POP).read_text())
    seal = ROOT / BASE / 'artifacts/evidence_sha256.txt'
    sealed = {p: h for h, p in (line.split('  ', 1) for line in seal.read_text().splitlines())}
    manifest_path = BASE + '/artifacts/input_manifest.json'
    assert digest(ROOT / manifest_path) == sealed[manifest_path]
    manifest = json.loads((ROOT / manifest_path).read_text())
    expected = []
    for path, sha in pop['source_sha256'].items():
        assert digest(ROOT / path) == sha
        expected.extend(json.loads((ROOT / path).read_text())['observations'])
    observed = [r for r in manifest['observations'] if r['split'] == 'fit']
    key = lambda r: (r['task'], r['source_sequence_id'])
    assert sorted(observed, key=key) == sorted(expected, key=key)
    shards = {s['task']: s for s in manifest['task_shards'] if s['parent_id'] in pop['parent_ids']}
    assert len(shards) == 180
    entries = []; total_bytes = 0; valid = 0
    for task, shard in sorted(shards.items()):
        rows = [r for r in observed if r['task'] == task]
        report = shard['read_report']
        assert len(rows) == 16 and report['selected_sequence_rows'] == [r['sequence_row'] for r in rows]
        path = BASE + '/' + shard['path']
        assert shard['sha256'] == sealed[path]
        # Stat only: data integrity is rechecked by the actual authorized reader.
        size = (ROOT / path).stat().st_size
        total_bytes += size; valid += shard['valid_returns']
        assert set(shard['array_sha256']) == set(manifest['student_fields'] + manifest['provenance_only_fields'])
        for index, row in enumerate(rows):
            entries.append(dict(row, student_path=path, student_sha256=shard['sha256'],
                                row_index=index, layout='saved_task_batch', decoded_observations=16))
    assert len(entries) == 2880
    result = dict(status='CACHE_METADATA_BOUND_PAYLOAD_NOT_REVERIFIED', observations=2880,
                  parents=60, task_shards=180, cache_compressed_bytes=total_bytes,
                  historical_export_valid_returns=valid,
                  student_fields=manifest['student_fields'], provenance_only_fields=manifest['provenance_only_fields'],
                  student_teacher_separation='No absolute poses or construction/source codes in cache',
                  entries=entries, payload_reads=0, labels_generated=0, training_steps=0,
                  input_sha256={POP: digest(ROOT / POP), manifest_path: sealed[manifest_path],
                                str(seal.relative_to(ROOT)): digest(seal)},
                  reuse_contract='Existing batched reader checks SHA, frame rows, sequence ID and four-field shapes/dtypes; discard identities before forward',
                  io_requirement='Read each task once when exporting features; do not decompress its 16-window archive 16 times',
                  missing_reference_inputs=['source codes and codebook', 'absolute original sensor pose', 'original construction surface'],
                  next='Freeze fit feature-export card with exact180batch hashes; separately plan missing reference sources; no training authority')
    out = ROOT / OUT
    if out.exists():
        assert json.loads(out.read_text()) == result
    else:
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('entries', 'input_sha256')}, indent=2))


if __name__ == '__main__':
    main()

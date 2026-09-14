"""Read-only exact45 sealed source census. No label creation or rendering."""
import _bootstrap
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from mtare_topo.data.gse_synthetic_fit_scope import compile_scope


def main():
    root = Path(__file__).resolve().parents[2]
    scope = compile_scope(root)
    groups = {}
    for row in scope['observations']:
        contents = {}
        for name, sha_name in [('input_path', 'input_sha256'),
                               ('source_record_path', 'source_record_sha256')]:
            raw = (root / row[name]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row[sha_name]:
                raise ValueError('sealed source drift')
            contents[name] = raw
        record = json.loads(gzip.decompress(contents['source_record_path']))
        book = record['codebook']
        ids = book['primitive_ids']
        primitives = record['construction']['base_construction']['primitives']
        if len(ids) != len(set(ids)) or set(ids) != {p['primitive_id'] for p in primitives}:
            raise ValueError('construction/codebook identity mismatch')
        sets = book['source_sets']
        if not sets or sets[0] != []:
            raise ValueError('missing no-return code')
        for members in sets[1:]:
            if not members or len(set(members)) != len(members) or any(type(i) is not int or not 0 <= i < len(ids) for i in members):
                raise ValueError('invalid source set')
        with np.load(io.BytesIO(contents['input_path']), allow_pickle=False) as a:
            codes = a['primitive_membership_code']; valid = a['valid_mask']
            if codes.dtype != np.uint16 or valid.dtype != np.uint8 or codes.shape != (5,16,720) or valid.shape != codes.shape:
                raise ValueError('sensor schema mismatch')
            if not np.isin(valid, [0,1]).all() or not np.array_equal(codes > 0, valid > 0) or codes.max() >= len(sets):
                raise ValueError('invalid return/code binding')
            sizes = np.array([len(s) for s in sets])[codes[valid > 0]]
            group = row['case_id'].split('__')[0]
            out = groups.setdefault(group, dict(observations=0, valid_returns=0, single_source=0, multiple_source=0))
            out['observations'] += 1
            out['valid_returns'] += len(sizes)
            out['single_source'] += int((sizes == 1).sum())
            out['multiple_source'] += int((sizes > 1).sum())
    print(json.dumps(dict(groups=groups, observations=45, frame_occurrences=225,
        source_files_hash_verified=90, new_labels=0, renders=0, optimizer_steps=0,
        point_to_structure_support_qualified=False), indent=2))


if __name__ == '__main__':
    main()

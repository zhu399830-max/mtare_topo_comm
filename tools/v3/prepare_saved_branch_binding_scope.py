"""Freeze existing reference paths, not labels or a training authorization."""
import gzip
import hashlib
import json
from pathlib import Path
from inspect_saved_branch_population import inspect


def compile_scope(root):
    inventory = inspect(root)
    card_path = 'configs/v3/gate3/data_cards/gse_supplement_joint_v3.json'
    card_bytes = (root / card_path).read_bytes()
    old = json.loads(card_bytes)['scope']
    tasks = {e['task']: e for e in old['entries']}
    selected = [r for r in inventory['rows'] if r['counts']['junction_occurrences']]
    files = {card_path: hashlib.sha256(card_bytes).hexdigest()}
    rows = []
    arrays = {}
    for item in selected:
        source = item['source']
        task = tasks[source['task']]
        ref = dict(path=item['path'], sha256=item['sha256'])
        chain = []
        while True:
            name = ref['path']
            if (not name.startswith('results/gate3_semantics/gate3_')
                    or '..' in Path(name).parts or name in chain or len(chain) >= 4):
                raise ValueError('unexpected reference chain')
            raw = (root / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != ref['sha256']:
                raise ValueError('archived reference drift')
            files[name] = ref['sha256']
            chain.append(name)
            data = json.loads(gzip.decompress(raw))
            if 'raw_interfaces' in data:
                original = data['raw_interfaces']
                if any(original['source'][k] != source[k] for k in source):
                    raise ValueError('raw interface source mismatch')
                break
            ref = data['raw_interfaces_reference']
        construction = task['construction_path']
        files[construction] = old['file_sha256'][construction]
        yaw = task['sensor_prefix'] + '/yaw_deg'
        access = old['array_access'][yaw]
        frame = source['frame_rows'][-1]
        if frame not in access['selected_rows'] or len(access['chunks']) != 1:
            raise ValueError('yaw row outside old scope')
        array = arrays.setdefault(yaw, dict(shape=access['shape'], chunks=access['chunks'],
                                           dtype=access['dtype'], selected_rows=set(), chunk_keys=set()))
        array['selected_rows'].add(frame)
        chunk = str(frame // access['chunks'][0])
        array['chunk_keys'].add(chunk)
        for suffix in ('.zarray', chunk):
            path = yaw + '/' + suffix
            files[path] = old['file_sha256'][path]
        rows.append(dict(source=source, split=item['split'], parent=item['parent'],
                         target_path=item['path'], reference_chain=chain,
                         construction_path=construction, yaw_array=yaw, yaw_row=frame,
                         branch_occurrences=item['counts']['branch_occurrences']))
    if len(rows) != 307 or sum(r['branch_occurrences'] for r in rows) != 951:
        raise ValueError('branch population differs')
    for value in arrays.values():
        value['selected_rows'] = sorted(value['selected_rows'])
        value['chunk_keys'] = sorted(value['chunk_keys'])
        value['decoded_padded_bytes'] = len(value['chunk_keys']) * value['chunks'][0] * 8
    return dict(schema='saved_branch_binding_scope_v1', status='PREPARATION_NOT_EXECUTION_AUTHORITY',
                observations=307, branch_occurrences=951, rows=rows, file_sha256=files,
                arrays=arrays, original_record_inventory_sha256=inventory['record_inventory_sha256'],
                verified_reference_chains=307, yaw_payload_read=False, construction_payload_read=False,
                training_eligible=False, new_teacher_execution=False)


if __name__ == '__main__':
    import base64
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compressed', action='store_true', help='lossless transport for large manifests')
    args = parser.parse_args()
    value = json.dumps(compile_scope(Path(__file__).resolve().parents[2]), separators=(',', ':'))
    print(base64.b64encode(gzip.compress(value.encode(), mtime=0)).decode() if args.compressed else value)

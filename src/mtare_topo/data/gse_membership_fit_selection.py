"""Fixed small fitting cohort from saved partial evidence, never model scores.

This selects windows, not teacher candidates for the forward pass. Full original
partial targets must be loaded by a separately bound reader; unknowns stay unknown.
The two source versions remain explicit. Selection is NOT label qualification,
an export, an independent evaluation, or permission to restart a retired model.
"""
from collections import defaultdict
import hashlib
import json


def _identity(source):
    if source['split'] != 'fit':
        raise ValueError('fitting sources only')
    frames = source['frame_rows']
    if len(frames) != 5 or any(type(x) is not int or x < 0 for x in frames):
        raise ValueError('five explicit source rows required')
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError('causal ordered frames required')
    return (source['parent_id'], source['task'], source['source_sequence_id'], tuple(frames))


def select_membership_fit_windows(positive_candidates, negative_windows, origin_rechecks,
                                  *, parents=8, seed=20260906):
    if type(parents) is not int or parents < 1 or type(seed) is not int:
        raise ValueError('explicit positive parent budget and integer seed required')

    def rank(value):
        return hashlib.sha256(json.dumps([seed, value], sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()

    retained = set()
    for row in origin_rechecks:
        if row['source']['split'] != 'fit':
            continue
        if (row['status'] == 'ORIGIN_PREREQUISITE_RETAINED'
                and row['retained_witness_frames']
                and row['original_witness_frames'] == row['retained_witness_frames']):
            retained.add((_identity(row['source']), row['terminal_node'], row['opening_source']))
    pools = {'positive': {}, 'negative': {}}
    for row in positive_candidates:
        if row['source']['split'] != 'fit' or row['original_membership'] is not True:
            continue
        key = _identity(row['source'])
        if (key, row['terminal_node'], row['opening_source']) not in retained:
            continue
        value = {'source': row['source'], 'evidence_path': row['evidence_path'],
                 'evidence_sha256': row['evidence_sha256'], 'source_version': 'saved_v6',
                 'selection_stratum': 'contains_original_terminal_positive'}
        previous = pools['positive'].get(key)
        if previous and (previous['evidence_path'], previous['evidence_sha256']) != (
                value['evidence_path'], value['evidence_sha256']):
            raise ValueError('conflicting positive evidence for one observation')
        pools['positive'][key] = value
    for row in negative_windows:
        if row['source']['split'] != 'fit' or row['negative_memberships'] < 1:
            continue
        key = _identity(row['source'])
        value = {'source': row['source'], 'evidence_file': row['evidence_file'],
                 'evidence_sha256': row['storage']['compressed_sha256'],
                 'source_version': 'saved_v8',
                 'selection_stratum': 'contains_original_terminal_negative'}
        if key in pools['negative'] and pools['negative'][key] != value:
            raise ValueError('conflicting negative evidence for one observation')
        pools['negative'][key] = value

    tasks = defaultdict(lambda: defaultdict(lambda: {'positive': [], 'negative': []}))
    for label, pool in pools.items():
        for key in pool:
            tasks[key[0]][key[1]][label].append(key)
    eligible = {parent: [task for task, labels in values.items()
                         if labels['positive'] and labels['negative']
                         and any(p != n for p in labels['positive'] for n in labels['negative'])]
                for parent, values in tasks.items()}
    eligible = {p: ts for p, ts in eligible.items() if ts}
    if len(eligible) < parents:
        raise ValueError(f'only {len(eligible)} paired parents, required {parents}; no padding')
    selected = []
    for parent in sorted(eligible, key=lambda p: (rank(p), p))[:parents]:
        task = min(eligible[parent], key=lambda t: (rank(t), t))
        labels = tasks[parent][task]
        pairs = [(p, n) for p in labels['positive'] for n in labels['negative'] if p != n]
        positive, negative = min(pairs, key=lambda pair: (rank(pair), pair))
        selected.extend([pools['positive'][positive], pools['negative'][negative]])
    return {'schema_version': 'gse_membership_fit_selection_v1', 'seed': seed,
            'parents': parents, 'windows': len(selected), 'eligible_parents': len(eligible),
            'records': selected, 'label_changes': 0, 'training_started': False,
            'scope': 'partial-reference fitting only; not independent method comparison'}

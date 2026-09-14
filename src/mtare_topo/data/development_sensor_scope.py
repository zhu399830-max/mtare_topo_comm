"""Metadata-only307 full sensor binding; no teacher/checkpoint/payload reads."""
from collections import Counter
from .development_partition_scope import compile_scope as compact_scope
from .gse_supplement_feature_scope_v1 import compile_feature_scope


def join_sensor_scope(compact, features):
    index = {}
    for task in features['tasks']:
        for row in task['observations']:
            key = (task['task'], row['source_sequence_id'], tuple(row['frame_rows']))
            if key in index:
                raise ValueError('duplicate sensor identity')
            index[key] = (task, row)
    rows, seen = [], set()
    for entry in compact['observations']:
        source = entry['source']
        key = (source['task'], source['source_sequence_id'], tuple(source['frame_rows']))
        if key in seen or key not in index:
            raise ValueError('missing or duplicate selected input')
        seen.add(key)
        task, row = index[key]
        provenance = entry['feature_entry']['source']
        if (row['split'] != entry['split'] or provenance['input_row'] != row['input_row']
                or provenance['input_file_sha256'] != task['input_sha256']
                or provenance['task'] != source['task']
                or provenance['source_sequence_id'] != source['source_sequence_id']
                or provenance['frame_rows'] != source['frame_rows']
                or provenance['coordinate_frame'] != 'current_sensor'):
            raise ValueError('encoder sensor source or split mismatch')
        rows.append(dict(source=source, split=entry['split'], parent_id=row['parent_id'],
            input_path=task['input_path'], input_sha256=task['input_sha256'],
            input_row=row['input_row'], observation_count=task['observation_count'],
            cache_path=entry['cache_path'], cache_sha256=entry['cache_sha256']))
    tasks = {r['source']['task']: r for r in rows}
    frames = {(r['source']['task'], f) for r in rows for f in r['source']['frame_rows']}
    parents = {r['parent_id'] for r in rows}
    counts = dict(observations=len(rows), unique_variant_frames=len(frames),
                  parents=len(parents), tasks=len(tasks), splits=dict(Counter(r['split'] for r in rows)),
                  task_container_observations=sum(r['observation_count'] for r in tasks.values()))
    if counts['observations'] != 307 or counts['unique_variant_frames'] != 1529 or counts['splits'] != dict(fit=250, calibration=27, development=30):
        raise ValueError('exact307 population drift')
    metadata = dict(compact['manifests_sha256'])
    for path, sha in features['metadata_sha256'].items():
        if path in metadata and metadata[path] != sha:
            raise ValueError('metadata hash conflict')
        metadata[path] = sha
    return dict(schema='development_full_sensor_scope_v1', observations=rows, counts=counts,
        metadata_sha256=metadata, sensory_input='original_full50m_fiveframe_range_valid_and_relative_motion',
        local_geometry_roi_m=10, teacher_payload_reads=0, sensor_payload_reads=0,
        restrictions=['metadata_only_not_run_authorization', 'no_change_to_frozen_learning_inputs',
                      'C01_C07_only', 'no_teacher_in_forward', 'not_full_structure_benchmark'])


def compile_scope(root):
    return join_sensor_scope(compact_scope(root), compile_feature_scope(root))

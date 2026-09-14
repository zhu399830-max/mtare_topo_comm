"""Stream every scoped observation through the historical teacher unchanged.

No training selection: zero-anchor windows and all unknown evidence survive.
Caller owns the immutable run, worker lifetime and process resource limits.
"""
import json
from pathlib import Path
import numpy as np
from .gse_lossless_evidence_v1 import encode_evidence
from .gse_structure_review_v1 import canonical_sha


def summarize_target(source, produced, *, expected_version='joint_partial_reference_v6_reference_competition'):
    record = produced['record']
    if (produced['producer_version'] != expected_version
            or record['coordinate_frame'] != 'current_sensor_m'
            or record['source_frame_indices'] != source['frame_rows']
            or canonical_sha(record) != produced['target_record_sha256']):
        raise ValueError('historical target version/frame/hash mismatch')
    anchors, openings = record['anchors'], record['openings']
    if len(anchors) > 32 or len(openings) > 64:
        raise ValueError('historical target capacity exceeded; no truncation')
    membership = record['membership']
    if len(membership) != len(openings) or any(len(row) != len(anchors) for row in membership):
        raise ValueError('membership shape mismatch')
    if any(value is not None and type(value) is not bool for row in membership for value in row):
        raise ValueError('membership must retain true/false/unknown')
    xyz = np.asarray([a['position_m'] for a in anchors], dtype=float).reshape(-1, 3)
    if not np.isfinite(xyz).all():
        raise ValueError('nonfinite anchor')
    start = produced['teacher_provenance']['terminal_anchor_start']
    if not 0 <= start <= len(anchors) or start != len(produced['teacher_provenance']['anchors']):
        raise ValueError('junction/terminal provenance mismatch')
    return dict(source=source, anchors=len(anchors), openings=len(openings),
                junction_positions_m=xyz[:start].tolist(), terminal_positions_m=xyz[start:].tolist(),
                positive_memberships=sum(v is True for row in membership for v in row),
                negative_memberships=sum(v is False for row in membership for v in row),
                unknown_memberships=sum(v is None for row in membership for v in row),
                unknown_candidates=produced['unknown_candidates'],
                target_record_sha256=produced['target_record_sha256'],
                supervision_status=produced['supervision_status'],
                full_training_gate_eligible=produced['full_training_gate_eligible'])


def export_population(run, scope, reader, client, check_resources, *, output_cap_bytes,
                      target_summarizer=summarize_target):
    """Dependency-injected for synthetic orchestration tests; no retry/cache swap."""
    run = Path(run)
    entries = scope['entries']
    expected = [(entry['task'], row['source_sequence_id'])
                for entry in entries for row in entry['observations']]
    if (len(expected) != scope['counts']['observations'] or len(set(expected)) != len(expected)
            or len(entries) != scope['counts']['tasks']
            or len({e['task'] for e in entries}) != len(entries)
            or any(e['split'] not in ('fit', 'calibration') for e in entries)):
        raise ValueError('exact unique fit/calibration population required')
    rows = []
    stored_bytes = 0
    with (run/'logs/observations.jsonl').open('x') as log:
        for entry in entries:
            check_resources()
            bundles = reader.read_task(entry['task'])
            if len(bundles) != len(entry['observations']):
                raise ValueError('reader omitted or added a window')
            for source, bundle in zip(entry['observations'], bundles, strict=True):
                if bundle['source'] != source:
                    raise ValueError('reader reordered or changed source')
                check_resources()
                # Log the active source before the potentially expensive call.
                log.write(json.dumps(dict(state='STARTED', source=source))+'\n'); log.flush()
                response = client.request(bundle)
                if response['raw_interfaces']['source'] != source:
                    raise ValueError('raw teacher source mismatch')
                row = target_summarizer(source, response['produced_targets'])
                packed, storage = encode_evidence(response)
                check_resources()
                used = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
                if used + len(packed) > output_cap_bytes:
                    raise OSError('full evidence output cap; no witness deletion')
                name = f'observation_{len(rows):05d}.json.gz'
                with (run/'artifacts'/name).open('xb') as stream:
                    stream.write(packed)
                row.update(split=entry['split'], evidence_file=name, storage=storage)
                rows.append(row); stored_bytes += len(packed)
                log.write(json.dumps(dict(state='COMPLETED', index=len(rows)-1, source=source,
                    anchors=row['anchors'], openings=row['openings'], evidence_file=name))+'\n'); log.flush()
                del response, packed
                check_resources()
            del bundles
    if [(r['source']['task'], r['source']['source_sequence_id']) for r in rows] != expected:
        raise ValueError('output population mismatch')
    summaries = {}
    for split in ('fit', 'calibration'):
        selected = [r for r in rows if r['split'] == split]
        positions = np.asarray([p for r in selected for p in r['junction_positions_m']], dtype=float).reshape(-1, 3)
        summaries[split] = dict(observations=len(selected),
            zero_anchor_windows=sum(r['anchors'] == 0 for r in selected),
            zero_opening_windows=sum(r['openings'] == 0 for r in selected),
            junction_targets=len(positions),
            junction_x_quantiles=np.quantile(positions[:, 0], [0,.05,.25,.5,.75,.95,1]).tolist() if len(positions) else None,
            junctions_behind_sensor=int((positions[:,0] < 0).sum()),
            unknown_memberships=sum(r['unknown_memberships'] for r in selected))
    return dict(status='HISTORICAL_PARTIAL_TARGET_EXPORT_COMPLETE_NOT_QUALIFIED',
                observations=rows, summaries=summaries, stored_evidence_bytes=stored_bytes,
                zero_anchor_is_negative=False, scientific_gate_pass=False, optimizer_steps=0)

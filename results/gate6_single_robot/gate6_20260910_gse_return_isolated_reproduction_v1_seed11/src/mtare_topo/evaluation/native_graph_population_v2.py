"""Execute an already authorized fixed population; no sampling or teacher access.

The caller must validate the frozen card/spec and initialize the material run.
This module never creates a run, resumes a partial export, or selects observations.
"""
import hashlib
import json
from pathlib import Path
import time

from mtare_topo.data.gse_supplement_feature_input_v1 import bind_feature_input
from mtare_topo.data.native_voxblox_packet import encode_observation
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from .native_graph_process_v2 import execute_native


def export_population(root, run, scope, *, command, environment, wall_cap_s=43200,
                      disk_cap_bytes=12*1024**3):
    root, run = Path(root), Path(run)
    rows = scope['observations']
    if len(rows) != 307 or scope['counts']['observations'] != 307:
        raise ValueError('exact 307 selected observations required')
    identities = [digest(r['source']) for r in rows]
    if len(set(identities)) != 307:
        raise ValueError('duplicate observation identity')
    output = run/'artifacts/native_graphs'
    output.mkdir()  # exclusive: partial populations cannot be resumed/overwritten
    started = time.monotonic()
    reads, entries = {}, []
    # Keep at most one authenticated container in memory; never retain feature arrays.
    cached_path, cached_hash, raw = None, None, None
    with (run/'logs/native_observations.jsonl').open('x') as log:
        for row, identity in zip(rows, identities):
            if time.monotonic()-started >= wall_cap_s:
                raise TimeoutError('population wall limit')
            path, expected = row['input_path'], row['input_sha256']
            if (path, expected) != (cached_path, cached_hash):
                raw = read_pinned(root, path, expected)
                cached_path, cached_hash = path, expected
            if path in reads and reads[path] != expected:
                raise ValueError('conflicting container binding')
            reads[path] = expected
            source = row['source']
            bound = bind_feature_input(raw, expected_sha256=expected,
                task=source['task'], row=row['input_row'],
                source_sequence_id=source['source_sequence_id'],
                frame_rows=source['frame_rows'], observation_count=row['observation_count'])
            packet = encode_observation(bound.range_valid, bound.translation_m, bound.yaw_deg)
            graph_path, log_path = output/(identity+'.json'), output/(identity+'.stderr.log')
            # Flush the active identity BEFORE starting native code: crashes remain attributable.
            log.write(json.dumps(dict(event='START', source=source, input_path=path,
                input_sha256=expected, input_row=row['input_row']))+'\n'); log.flush()
            graph, runtime = execute_native(command, packet, stdout_path=graph_path,
                stderr_path=log_path, environment=environment)
            entry = dict(source=source, split=row['split'], parent_id=row['parent_id'],
                input_path=path, input_sha256=expected, input_row=row['input_row'],
                input_binding_sha256=bound.input_binding_sha256,
                packet_sha256=hashlib.sha256(packet).hexdigest(),
                graph_file=str(graph_path.relative_to(run)),
                graph_sha256=hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                stderr_file=str(log_path.relative_to(run)),
                stderr_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
                nodes=graph['nodes'], edges=graph['edges'], rays=graph['rays'],
                self_loop_edges=sum(a['endpoints'][0]==a['endpoints'][1] for a in graph['edge_audit']),
                unknown_sample_edges=sum(a['unknown_samples']>0 for a in graph['edge_audit']),
                low_clearance_sample_edges=sum(a['below_0_4m_samples']>0 for a in graph['edge_audit']),
                runtime=runtime)
            entries.append(entry)
            log.write(json.dumps(dict(event='COMPLETE', **entry))+'\n'); log.flush()
            print(json.dumps(dict(completed=len(entries), total=307,
                elapsed_s=time.monotonic()-started, nodes=entry['nodes'], edges=entry['edges'])), flush=True)
            if sum(p.stat().st_size for p in run.rglob('*') if p.is_file()) > disk_cap_bytes:
                raise RuntimeError('population evidence disk cap')
        # A completed manifest is emitted only after all sources remain hash-consistent.
        for path, expected in reads.items():
            read_pinned(root, path, expected)
        manifest = dict(status='CANDIDATE_GRAPH_EXPORT_COMPLETE_NOT_SEMANTIC_OR_SAFETY_PASS',
            observations=entries, source_reads_sha256=reads, counts=scope['counts'],
            independent_observation_graphs=True, cross_observation_edges=0, optimizer_steps=0)
        with (run/'artifacts/native_graph_manifest.json').open('x') as f:
            json.dump(manifest, f, sort_keys=True, allow_nan=False)
            f.write('\n')
    return manifest


"""Replay original lateral predicates over saved intersections, no raycaster."""
import argparse, collections, gzip, hashlib, importlib, io, json, resource
from pathlib import Path
from recover_lateral_witness_worker import load_archive, write_gzip_json


def run(root, scope, index, output):
    import numpy as np
    e = scope['entries'][index]
    modules = load_archive(root, scope)
    files = {}
    for p, h in e['files'].items():
        raw = (root / p).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h: raise ValueError('input drift ' + p)
        if p.endswith('.npz'):
            with np.load(io.BytesIO(raw), allow_pickle=False) as z: files[p] = {k:z[k].copy() for k in z.files}
        else: files[p] = json.loads(raw)
    def bound_gzip(binding):
        raw = (root / binding['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding['sha256']: raise ValueError('bound gzip drift')
        return json.loads(gzip.decompress(raw))
    saved = bound_gzip(e['recovered']); reference = bound_gzip(e['reference'])
    if saved['queried_rays'] != e['original_ray_indices']: raise ValueError('ray population drift')
    sensor = next(v for p,v in files.items() if '/source_evidence/' in p and p.endswith('.npz'))
    construction = next(v for p,v in files.items() if p.endswith('_constructions.json'))
    _, primitives = modules['construction'].load_p1a_realized_construction(construction)
    field_module = importlib.import_module('mtare_topo.teacher.swept_superellipse_field')
    binding = importlib.import_module('mtare_topo.teacher.gse_directed_interface_binding_v1')
    paths = importlib.import_module('mtare_topo.teacher.gse_construction_paths_v3')
    ids = [p.primitive_id for p in primitives]; lookup = {s:i for i,s in enumerate(ids)}
    field = field_module.SweptSuperellipseProvenanceField(primitives, spacing_m=scope['geometry_settings']['field_spacing_m'])
    axes = binding.bind_axes(paths.construction_incident_paths(construction), reference['raw_interfaces']['interfaces_teacher_only'])
    local = modules['sensor'].lidar_local_directions().reshape(-1,3).astype(np.float64)
    directions = np.concatenate([modules['sensor'].world_directions(local,float(y)) for y in sensor['yaw_deg']])
    _, packed = modules['pack'].pack_caster_inputs(np.empty((0,3)),np.repeat(sensor['sensor_xyz_m'],11520,axis=0),directions)
    origins = field.operand_signed_distances_sparse(sensor['sensor_xyz_m'])
    candidates = collections.defaultdict(list); checks = []; reasons = collections.Counter()
    for i, entry in enumerate(saved['entries']):
        ray=entry['ray_index']; inside=np.flatnonzero(origins[ray//11520] < 0)
        row=dict(entry_index=i,ray_index=ray,origin_operand_indices=inside.tolist()); reason=None
        if len(inside)!=1:reason='origin_not_unique'
        else:
            source=ids[inside[0]];target=entry['source_id_teacher_only'];row.update(source=source,target=target)
            if source==target:reason='same_source'
            else:
                distance=float(field.operand_signed_distances_sparse(np.asarray([entry['intersection_world_m']]))[0][lookup[source]])
                row['entry_origin_source_distance']=distance
                if distance>=0:reason='entry_outside_origin'
                else:
                    pairs=[(a,b) for a in axes for b in axes if a['source_key_teacher_only']==source and b['source_key_teacher_only']==target and a['node_id_teacher_only']==b['node_id_teacher_only']]
                    row['interface_pair_count']=len(pairs)
                    if len(pairs)!=1:reason='interface_pair_not_unique'
                    else:
                        a,b=pairs[0];d=packed[ray,3:].astype(np.float64);guard=64*np.finfo(np.float64).eps*np.linalg.norm(d)
                        left=float(d@a['inward_direction']);right=float(d@b['inward_direction'])
                        row.update(left_dot=left,right_dot=right,guard=float(guard))
                        if left>=-guard or right<=guard:reason='direction_rejected'
                        else:candidates[ray].append((a,b))
        row['condition']=reason or 'candidate';reasons[row['condition']]+=1;checks.append(row)
    actual_entry=collections.defaultdict(set);actual_depart=collections.defaultdict(set);ambiguous=[]
    for ray,pairs in candidates.items():
        if len({a['node_id_teacher_only'] for a,b in pairs})!=1:ambiguous.append(ray);continue
        for a,b in pairs:
            actual_entry[b['interface_id_teacher_only']].add(ray);actual_depart[a['interface_id_teacher_only']].add(ray)
    comparisons=[]
    for anchor in reference['produced_targets']['teacher_provenance']['anchors']:
        for slot,interface in enumerate(anchor['interface_ids']):
            for role,key,actual in [('enter','surface_entry_witness_ray_indices',actual_entry),('depart','surface_departure_witness_ray_indices',actual_depart)]:
                expected=set(anchor[key][slot]);observed=actual[interface]
                comparisons.append(dict(interface=interface,role=role,expected=len(expected),recovered=len(observed),missing=sorted(expected-observed),extra=sorted(observed-expected)))
    agrees=all(not x['missing'] and not x['extra'] for x in comparisons)
    response=dict(status='ORIGINAL_CONDITIONS_REPRODUCED' if agrees else 'CONDITION_MISMATCH',case=index,
        queried_rays=saved['queried_rays'],archive_sha256=scope['archive']['sha256'],entries=checks,
        missing_entry_ray_indices=sorted(set(saved['queried_rays'])-set(candidates)),
        comparisons=comparisons,ambiguous_ray_indices=ambiguous,reasons=dict(reasons),
        agrees_with_saved_claims=agrees,new_labels=0,teacher_target_calls=0,intersection_calls=0,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_gzip_json(output,response)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scope',type=Path,required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    if a.index not in range(3):raise ValueError('exact three cases only')
    run(a.root,json.loads(a.scope.read_text()),a.index,a.output)

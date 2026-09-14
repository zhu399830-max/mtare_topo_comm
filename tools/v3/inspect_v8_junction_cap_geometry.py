"""Read-only source-interval audit of saved junction cap witnesses.

No target producer, new labels, lateral rerender, or population extension.
Checks only the four sealed prefix observations, not physical connectivity.
"""
import gzip
import json
import numpy as np
from _bootstrap import PROJECT_ROOT
from inspect_v8_saved_negative_witnesses import RUN, SEAL_SHA
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_v8_probe_reader import V8ProbeReader
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.gse_caster_cap_replay_v1 import pack_caster_inputs
from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths
from mtare_topo.teacher.gse_directed_interface_binding_v1 import bind_axes
from mtare_topo.evaluation.gse_cap_branch_precision import cap_branch_precision, summarize_cap_directions
from mtare_topo.evaluation.gse_ray_triangle_audit import ray_triangle_audit


def main():
    seal = read_pinned(PROJECT_ROOT, RUN+'/artifacts/evidence_sha256.txt', SEAL_SHA)
    hashes = {p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    def read(path): return read_pinned(PROJECT_ROOT,path,hashes[path])
    card = json.loads(read(RUN+'/config/data_card.json'))
    summary = json.loads(read(RUN+'/metrics/summary.json'))
    if summary['completed_observations'] != 4 or summary['status'] != 'ABORTED_SOURCE_PARAMETER_CONTRACT_INVALID':
        raise ValueError('only sealed four-observation failed prefix allowed')
    reader = V8ProbeReader(PROJECT_ROOT,card['scope'])
    for item in summary['observations']:
        task = item['source']['task']; seq = item['source']['source_sequence_id']
        target = json.loads(gzip.decompress(read(RUN+'/artifacts/'+task+'_'+str(seq)+'.json.gz')))['produced_targets']
        bundle,raw,_ = reader.read_observation(task,seq)
        _,primitives = load_p1a_realized_construction(bundle['construction_teacher_only'])
        byid = {p.primitive_id:p for p in primitives}
        interfaces = {i['interface_id_teacher_only']:i for i in raw['interfaces_teacher_only']}
        axes = {a['interface_id_teacher_only']:a for a in bind_axes(
            construction_incident_paths(bundle['construction_teacher_only']),raw['interfaces_teacher_only'])}
        sensor = bundle['sensor_teacher_only']
        local = lidar_local_directions().reshape(-1,3).astype(np.float64)
        directions = np.concatenate([world_directions(local,float(y)) for y in sensor['yaw_deg']])
        normalized_directions = directions / np.linalg.norm(directions,axis=1)[:,None]
        origins = np.repeat(sensor['sensor_xyz_m'],len(local),axis=0)
        ranges = bundle['student']['ranges_m'].reshape(-1)
        for ai,anchor in enumerate(target['teacher_provenance']['anchors']):
            reports = []; stable_all = set(); precision_reports = []
            for key,wanted_list in zip(anchor['interface_ids'],anchor['entering_witness_ray_indices'],strict=True):
                interface = interfaces[key]
                if interface['node_id_teacher_only'] != anchor['node_id_teacher_only']:
                    raise ValueError('junction/interface identity mismatch')
                sid,side = interface['endpoint_key_teacher_only']; wanted = set(wanted_list)
                mesh = mesh_swept_superellipse(byid[sid],axial_spacing_m=.05,angular_segments=64)
                hits = [h for h in raw['raw_interface_intersections'] if h['interface_id_teacher_only']==key
                    and h['ray_index'] in wanted and h['inside_roi'] and 0<=h['t']<float(ranges[h['ray_index']])]
                if {h['ray_index'] for h in hits} != wanted:
                    raise ValueError('saved branch entrance lacks original pre-return intersection')
                cap_start = len(mesh.triangle_vertex_indices)-128
                if any(h['triangle_index']<cap_start or (h['triangle_index']-cap_start)%2!=side for h in hits):
                    raise ValueError('claimed entrance does not touch declared interface cap')
                if not hits:
                    reports.append(dict(interface=key,saved_rays=0,stable_entering_rays=0)); continue
                ri = np.asarray([h['ray_index'] for h in hits]); fi = np.asarray([h['triangle_index'] for h in hits])
                precision = cap_branch_precision(mesh.vertices_xyz_m[mesh.triangle_vertex_indices[fi]],
                    origins[ri],normalized_directions[ri],ri,axes[key]['inward_direction'])
                precision_reports.append(precision)
                vertices,rays = pack_caster_inputs(mesh.vertices_xyz_m,origins,directions)
                triangle_report = ray_triangle_audit(vertices,mesh.triangle_vertex_indices,rays,ri,fi,[h['t'] for h in hits])
                stable = set(precision['stable_entering_ray_indices']); stable_all.update(stable)
                reports.append(dict(interface=key,saved_rays=len(wanted),stable_entering_rays=len(stable),
                    unknown_hits=precision['unknown_hit_count'],triangle_check=triangle_report))
            relations = [dict(opening=r['opening_index'],saved_rays=len(r['ray_indices']),
                rays_with_stable_junction_cap=len(set(r['ray_indices']) & stable_all))
                for r in target['teacher_provenance']['relations'] if r.get('anchor_index')==ai]
            print(json.dumps(dict(task=task,sequence=seq,node=anchor['node_id_teacher_only'],branches=reports,
                **summarize_cap_directions(precision_reports),relations=relations,
                )),flush=True)


if __name__ == '__main__': main()

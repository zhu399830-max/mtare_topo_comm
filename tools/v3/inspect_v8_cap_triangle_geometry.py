"""Same sealed four observations: independent cap triangle geometry only."""
import gzip,json
import numpy as np
from _bootstrap import PROJECT_ROOT
from inspect_v8_saved_negative_witnesses import RUN,SEAL_SHA
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_v8_probe_reader import V8ProbeReader
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions,world_directions
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.gse_caster_cap_replay_v1 import pack_caster_inputs
from mtare_topo.evaluation.gse_ray_triangle_audit import ray_triangle_audit
from mtare_topo.evaluation.gse_source_plane_intervals import source_plane_sign_bounds


def main():
    root=PROJECT_ROOT
    seal=read_pinned(root,RUN+'/artifacts/evidence_sha256.txt',SEAL_SHA)
    hashes={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    def read(p):return read_pinned(root,p,hashes[p])
    card=json.loads(read(RUN+'/config/data_card.json'));summary=json.loads(read(RUN+'/metrics/summary.json'))
    if summary['completed_observations']!=4:raise ValueError('fixed prefix required')
    reader=V8ProbeReader(root,card['scope'])
    for item in summary['observations']:
        source=item['source'];task=source['task'];seq=source['source_sequence_id']
        bundle,raw,_=reader.read_observation(task,seq)
        target=json.loads(gzip.decompress(read(RUN+'/artifacts/'+task+'_'+str(seq)+'.json.gz')))['produced_targets']
        prov=target['teacher_provenance'];_,primitives=load_p1a_realized_construction(bundle['construction_teacher_only'])
        byid={p.primitive_id:p for p in primitives};sensor=bundle['sensor_teacher_only']
        local=lidar_local_directions().reshape(-1,3).astype(np.float64)
        directions=np.concatenate([world_directions(local,float(y)) for y in sensor['yaw_deg']])
        for ti,terminal in enumerate(prov['terminals']):
            proofs=[p for p in prov['terminal_nonmembership'] if p['anchor_index']==prov['terminal_anchor_start']+ti]
            if not proofs:continue
            sid,side=terminal['endpoint_key_teacher_only']
            mesh=mesh_swept_superellipse(byid[sid],axial_spacing_m=.05,angular_segments=64)
            vertices,rays=pack_caster_inputs(mesh.vertices_xyz_m,np.repeat(sensor['sensor_xyz_m'],11520,axis=0),directions)
            cap_start=len(mesh.triangle_vertex_indices)-128
            witness=terminal['witnesses']
            if any(w['triangle_index']<cap_start or (w['triangle_index']-cap_start)%2!=side for w in witness):
                raise ValueError('claimed terminal witness not on declared endpoint cap')
            def check(hits):
                return ray_triangle_audit(vertices,mesh.triangle_vertex_indices,rays,
                    [h['ray_index'] for h in hits],[h['triangle_index'] for h in hits],
                    [h.get('stored_t',h.get('t')) for h in hits])
            cap_report=check(witness)
            for proof in proofs:
                keys=[i['interface_id_teacher_only'] for i in raw['interfaces_teacher_only']
                    if i['node_id_teacher_only']==proof['junction_node_teacher_only'] and i['endpoint_key_teacher_only'][0]==sid]
                if len(keys)!=1:raise ValueError('unique terminal-side interface required')
                wanted=set(proof['cap_entering_ray_indices'])
                hits=[h for h in raw['raw_interface_intersections'] if h['interface_id_teacher_only']==keys[0]
                    and h['ray_index'] in wanted and h['inside_roi'] and 0<=h['t']<float(bundle['student']['ranges_m'].reshape(-1)[h['ray_index']])]
                if {h['ray_index'] for h in hits}!=wanted:raise ValueError('missing original cap entrance before first return')
                if any(h['triangle_index']<cap_start or (h['triangle_index']-cap_start)%2==side for h in hits):
                    raise ValueError('entrance witness not opposite endpoint cap')
                ri=np.array([h['ray_index'] for h in hits]);fi=np.array([h['triangle_index'] for h in hits])
                tri=mesh.vertices_xyz_m[mesh.triangle_vertex_indices[fi]]
                normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                original_origins=np.repeat(sensor['sensor_xyz_m'],11520,axis=0)[ri]
                plane_t=np.sum((tri[:,0]-original_origins)*normal,axis=1)/np.sum(directions[ri]*normal,axis=1)
                original_directions=directions[ri]/np.linalg.norm(directions[ri],axis=1)[:,None]
                bounds=source_plane_sign_bounds(tri,original_origins,original_directions)
                print(json.dumps(dict(task=task,sequence=seq,opening=proof['opening_index'],
                    terminal_cap=cap_report,opposite_cap_entrance=check(hits),
                    original_float64_plane_t_min_m=float(plane_t.min()),
                    original_float64_plane_t_max_m=float(plane_t.max()),
                    original_float64_positive_plane_hits=int(np.sum(plane_t>0)),
                    quantized_positive_but_original_nonpositive_hits=int(np.sum(plane_t<=0)),
                    interval_stable_entering_hits=int(bounds['entering'].sum()),
                    interval_unknown_hits=int(bounds['unknown'].sum()),
                    independent_backend='numpy_float64_linear_solve_not_open3d',
                    continuity_or_membership_certified=False)),flush=True)


if __name__=='__main__':main()

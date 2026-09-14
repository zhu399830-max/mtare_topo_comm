"""Versioned full-chain comparison. Original failing diagnostic is untouched."""
import json
from check_interval_v2_native import box
from check_ordered_exit_adversarial_native import shells
from mtare_topo.teacher.bound_ray_diagnostic import BoundRayDiagnostic


def main():
    b=box('main',-1.,2.)
    cases=[('gap',[b,box('branch',2.00675564,30.)],[1,0],2.),
           ('overlap',[b,box('branch',1.5,30.)],[1,0],30.),
           ('separated',[b,box('branch',3.,30.)],[1,0],2.),
           ('contact',[b,box('branch',2.,30.)],[1,0],30.),
           ('coincident_sources',[b,box('other',-1.,2.)],[1,1],2.),
           ('internal_shell',[shells([b,box('nested',.5,1.)])],[1],2.),
           ('duplicate_shell',[shells([b,b])],[1],2.),
           ('surface_origin',[b],[1],None)]
    rows=[]
    for name,meshes,inside,expected in cases:
        diagnostic=BoundRayDiagnostic(meshes)
        origin=[0,1.,1.] if name=='surface_origin' else [0,.13,.17]
        result=diagnostic.compare(origin,[1,0,0],inside)
        distance=None if result.reference is None else result.reference.distance_m
        assert result.candidate_matches_reference is not False,(name,result)
        assert (distance is None if expected is None else distance is not None and abs(distance-expected)<1e-5)
        if name in ('contact','internal_shell','duplicate_shell','surface_origin'):
            assert result.candidate.status=='needs_reference'
        # External mutation cannot change the scene, origin cache or reference.
        for m in meshes:m.vertices_xyz_m[:]+=100.
        assert diagnostic.compare(origin,[1,0,0],inside)==result
        for m in meshes:m.vertices_xyz_m[:]-=100.
        rows.append({'case':name,'candidate_status':result.candidate.status,'reason':result.candidate.reason,
                     'reference_m':distance,'candidate_match':result.candidate_matches_reference})
    print(json.dumps({'cases':rows,'scope':'software_diagnostic_not_export'},indent=2))


if __name__=='__main__':main()

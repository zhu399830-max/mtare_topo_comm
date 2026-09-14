"""Same finite observations, distinct hidden section centers: no gold label."""
import numpy as np
from mtare_topo.teacher.gse_portal_ray_evidence import CausalRaySegments
from mtare_topo.teacher.gse_window_opening_proposals_v1 import propose_window_openings


def source_tube(ylo, yhi, shift):
    ring=[(ylo,-1.1),(yhi,-1.1),(yhi,1.1),(ylo,1.1)]
    vertices=np.array([[x,y,z] for x in (-2.,12.) for y,z in ring])+shift
    triangles=[]
    for a in range(4):
        b=(a+1)%4
        triangles.extend([[a,b,b+4],[a,b+4,a+4]])
    triangles.extend([[0,2,1],[0,3,2],[4,5,6],[4,6,7]])
    return vertices,np.array(triangles)


def test_identical_visible_roof_and_free_ray_do_not_identify_hidden_axis_center():
    shift=np.array([.13,.17,.19])
    origins=np.tile(np.array([7.,0.,0.])+shift,(2,1))
    delta=np.array([[1.,0.,1.1],[5.,0.,0.]])
    ranges=np.linalg.norm(delta,axis=1)
    rays=CausalRaySegments(origins,delta/ranges[:,None],ranges,
        np.ones(2,dtype=bool),np.array([0,4]),4,0.)
    proposals=[]
    for ylo,yhi,axis_y in [(-13.,1.,-6.),(-1.,13.,6.)]:
        vertices,triangles=source_tube(ylo,yhi,shift)
        # Independently verify that both closed boxes produce the supplied
        # first returns, rather than merely claiming identical observations.
        lo=vertices.min(axis=0);hi=vertices.max(axis=0)
        assert np.all(origins>lo) and np.all(origins<hi)
        exit_t=np.full_like(rays.directions,np.inf)
        np.divide(hi-origins,rays.directions,out=exit_t,where=rays.directions>0)
        np.divide(lo-origins,rays.directions,out=exit_t,where=rays.directions<0)
        np.testing.assert_allclose(exit_t.min(axis=1),ranges,rtol=0,atol=4e-15)
        result=propose_window_openings(vertices,triangles,
            section_center_m=np.array([8.,axis_y,0.])+shift,
            outward_direction=[1.,0.,0.],roi_center_m=shift,rays=rays,
            unique_return_source_index=np.zeros(2,dtype=int),source_index=0)
        p=result['proposals'][0]
        assert p['proposal_supported']
        assert p['outward_crossing_ray_indices']==[1]
        assert p['surface_return_ray_indices']==[0]
        assert not p['training_eligible'] and result['qualified_labels']==0
        proposals.append(p)
    assert np.linalg.norm(np.array(proposals[0]['reference_position_m'])-
                          proposals[1]['reference_position_m'])==12.

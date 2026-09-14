"""Same eight native box geometries; only the origin precondition is tested."""
import json
from check_interval_v2_native import box
from check_ordered_exit_adversarial_native import shells
from mtare_topo.teacher.mesh_origin_check import PreparedOriginCheck


def main():
    base=box('main',-1.,2.)
    rows=[('gap',[base,box('branch',2.00675564,30.)],[0,.13,.17],True),
          ('overlap',[base,box('branch',1.5,30.)],[0,.13,.17],True),
          ('separated',[base,box('branch',3.,30.)],[0,.13,.17],True),
          ('contact',[base,box('branch',2.,30.)],[0,.13,.17],True),
          ('coincident_sources',[base,box('other',-1.,2.)],[0,.13,.17],True),
          ('internal_shell',[shells([base,box('nested',.5,1.)])],[0,.13,.17],True),
          ('duplicate_shell',[shells([base,base])],[0,.13,.17],False),
          ('surface_origin',[base],[0,1.,1.],False)]
    results=[]
    for name,meshes,origin,expected in rows:
        prepared=PreparedOriginCheck(meshes);answer=prepared.check(origin)
        assert (answer.status=='origin_checked')==expected,(name,answer)
        assert prepared.check(origin) is answer
        results.append({'case':name,'status':answer.status,'inside':answer.inside,'reason':answer.reason})
    print(json.dumps({'cases':results,'scope':'origin_only_not_full_ray_qualification'},indent=2))


if __name__=='__main__':main()

"""Same four observations, exact declared-order restoration, separate run."""
import _bootstrap
import argparse,json
from pathlib import Path
import synthetic_four_corrective as common
from mtare_topo.governance_synthetic_corrective import validate_card_v2

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path)
    a=p.parse_args()
    if a.freeze:common.freeze(slug='gse_synthetic_corrective_v2',schema='v3_gse_synthetic_corrective_card_v2',
        validator=validate_card_v2,entry='tools/v3/synthetic_four_corrective_v2.py')
    elif a.spec and a.run_dir:raise SystemExit(common.guard.execute(json.loads(a.spec.read_text()),a.run_dir,validator=validate_card_v2,execute_fn=common.execute_four))
    else:p.error('--freeze or --spec/--run-dir required')

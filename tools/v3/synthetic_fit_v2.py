"""New immutable corrective run, reusing unchanged sealed v1 feature tensors."""
import synthetic_fit_v1 as base
from mtare_topo.governance_synthetic_fit_v2 import SCHEMA,SLUG,validate_card
from mtare_topo.data.gse_synthetic_fit_cache import compile_scope

base.SCHEMA=SCHEMA;base.SLUG=SLUG;base.validate_card=validate_card;base.compile_scope=compile_scope
base.CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';base.SPEC='configs/v3/gate3/'+SLUG+'.json'
base.RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed0';base.ENTRY='tools/v3/synthetic_fit_v2.py'

if __name__=='__main__':
    p=base.argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=base.Path);p.add_argument('--run-dir',type=base.Path)
    a=p.parse_args()
    if a.freeze:base.freeze()
    elif a.spec and a.run_dir:raise SystemExit(base.execute(base.load_json(a.spec),a.run_dir))
    else:p.error('freeze or spec/run-dir required')

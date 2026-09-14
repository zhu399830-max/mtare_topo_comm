"""Same executor, exactly one initialization change after sealed fit failure."""
import argparse
from pathlib import Path
import grouping_center_fit_v1 as executor
from mtare_topo.governance_grouping_fit_v1r import SCHEMA,SLUG,POLICY,compile_scope,validate_card
from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model


def configure():
    executor.SCHEMA=SCHEMA;executor.SLUG=SLUG;executor.POLICY=POLICY
    executor.compile_scope=compile_scope;executor.validate_card=validate_card;executor.build_model=build_model
    executor.CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
    executor.SPEC='configs/v3/gate3/'+SLUG+'.json'
    executor.RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed0'
    executor.SCRIPT='tools/v3/grouping_center_fit_v1r.py'


if __name__=='__main__':
    configure()
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.freeze:executor.freeze()
    elif a.spec and a.run_dir:raise SystemExit(executor.execute(executor.load_json(a.spec),a.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')

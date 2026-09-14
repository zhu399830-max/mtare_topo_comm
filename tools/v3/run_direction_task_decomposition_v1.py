"""One predeclared observation-only component probe, not a learned method win."""
import argparse
from pathlib import Path
import run_direction_task_witness_replay as runner
from mtare_topo.semantics.observed_direction_components import range_proposals

runner.NAME='gse_direction_task_decomposition_v1'
runner.SPEC=f'configs/v3/gate3/{runner.NAME}.json'
runner.CARD=f'configs/v3/gate3/data_cards/{runner.NAME}.json'
runner.RUN=f'results/gate3_semantics/gate3_20260913_{runner.NAME}_seed0'
runner.ENTRYPOINT=Path(__file__).resolve()
runner.SUPPORT_POLICY='four_neighbor_ray_components_reaching_existing10m_boundary_no_pruning_residuals_preserved'
runner.CONTROL_RUN='results/gate3_semantics/gate3_20260913_gse_direction_task_witness_replay_v2_seed0'
runner.range_proposals=range_proposals

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:runner.freeze()
    else:raise SystemExit(runner.execute())

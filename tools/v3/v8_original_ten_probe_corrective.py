#!/usr/bin/env python3
"""Single comparison-only correction. Original failed run stays immutable."""
import argparse
from pathlib import Path
import v8_original_ten_probe as runner
from mtare_topo.governance_v8_probe_corrective import SCHEMA,SLUG,validate_card
from mtare_topo.data.gse_v8_probe_comparison import compare

# Reuse the unchanged executor, reader, teacher and source settings.
runner.SCHEMA=SCHEMA;runner.SLUG=SLUG;runner.validate_card=validate_card;runner.compare=compare
runner.CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
runner.SPEC='configs/v3/gate3/'+SLUG+'.json'
runner.RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed20260906'
runner.ENTRYPOINT='tools/v3/v8_original_ten_probe_corrective.py'

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--spec',type=Path);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.freeze:
        runner.freeze()
    elif args.spec and args.run_dir:
        raise SystemExit(runner.execute(runner.load_json(args.spec),args.run_dir))
    else:parser.error('--freeze or --spec/--run-dir required')

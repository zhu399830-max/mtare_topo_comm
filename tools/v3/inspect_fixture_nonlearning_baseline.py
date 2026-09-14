"""Same sealed45 scans, unchanged nonlearning defaults, read-only diagnostics."""
import _bootstrap
from collections import Counter,defaultdict
import io,json
from pathlib import Path
import numpy as np
from mtare_topo.data.gse_synthetic_fit_scope import compile_scope
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.semantics.nonlearning_geometry_observation import nonlearning_geometry_observation


def main():
    root=Path(__file__).resolve().parents[2];scope=compile_scope(root)
    groups=defaultdict(Counter)
    for row in scope['observations']:
        with np.load(io.BytesIO(read_pinned(root,row['input_path'],row['input_sha256'])),allow_pickle=False) as p:
            observation=nonlearning_geometry_observation(p['ranges_m'],p['valid_mask'])
        event=observation.event.value;headings=[e.heading_robot_deg for e in observation.exit_tokens]
        groups[row['case_id'].split('__')[0]][(event,len(headings))]+=1
        print(json.dumps(dict(case_id=row['case_id'],event=event,headings_deg=headings,has_3d_anchor=False)),flush=True)
    print(json.dumps(dict(observations=45,groups={k:{str(key):n for key,n in v.items()} for k,v in groups.items()},
        default_thresholds_unchanged=True,no_truth_positions_injected=True,new_labels=0,optimizer_steps=0,
        interpretation='Event/heading diagnostic only; no 3D node F1 or graph qualification.')),flush=True)


if __name__=='__main__':main()

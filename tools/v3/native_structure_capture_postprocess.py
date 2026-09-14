"""ROS bag decoding stays in its pinned environment after native execution."""
from _bootstrap import PROJECT_ROOT
import argparse
import json
import os
from pathlib import Path
from export_native_structure_windows import export


def main():
    p=argparse.ArgumentParser();p.add_argument('--case-dir',type=Path,required=True)
    p.add_argument('--logical-output-relative',required=True)
    p.add_argument('--host-uid',type=int,required=True);p.add_argument('--host-gid',type=int,required=True)
    a=p.parse_args();out=Path('/evidence/model_inputs')
    metrics=json.loads((a.case_dir/'evidence/metrics.json').read_text())
    try:
        export(a.case_dir/'raw.bag',out,project_root=PROJECT_ROOT,
            start_ns=round(metrics['start_sim_sec']*1e9),end_ns=round(metrics['end_sim_sec']*1e9),
            logical_output_relative=a.logical_output_relative)
    finally:
        # Only this newly created owned evidence subtree, never arbitrary paths.
        if out.is_dir():
            for path in [out,*out.rglob('*')]:
                if path.is_symlink():
                    raise ValueError('evidence handoff rejects symlinks')
                os.chown(path,a.host_uid,a.host_gid)


if __name__=='__main__':
    main()

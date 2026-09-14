"""Reuse V6 transport loop with an explicitly bound V8 archive/producer.

Executed only in a fresh worker. No changes to the frozen research source.
"""
import argparse
from pathlib import Path
import historical_teacher_worker_v1 as transport
from v8_teacher_snapshot_probe_v1 import load_teacher


def main(echo=False,memory_bytes=None):
    # Load inside transport.main, after its memory cap and stdout isolation.
    def load():
        diagnose,produce,modules,manifest=load_teacher()
        transport.ZIP=Path(__file__).resolve().parents[2]/manifest['archive_path']
        transport.SHA=manifest['archive_sha256']
        return diagnose,produce,modules
    transport.load_teacher=load
    return transport.main(echo,memory_bytes)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--echo',action='store_true')
    parser.add_argument('--memory-bytes',type=int)
    args=parser.parse_args()
    raise SystemExit(main(args.echo,args.memory_bytes))

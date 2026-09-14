#!/usr/bin/env python3
"""Launch the original native controller plus a non-control advice sidecar.

The case runner owns system/recorder/runtime; this process owns only its two
children and tears both down on failure or shutdown. No model runs here.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from _bootstrap import PROJECT_ROOT


def commands(*, planner_seed, world, output, cache_manifest=None, cache_sha256=None):
    if planner_seed != 11 or world != 'tunnel':
        raise ValueError('This bounded development launcher is frozen to tunnel/seed11')
    planner = ['roslaunch', str(PROJECT_ROOT / 'integration/native_structure_bridge/explore_native_shadow.launch'),
        'scenario:=' + world, 'planner_seed:=' + str(planner_seed)]
    sidecar = [sys.executable, str(PROJECT_ROOT / 'tools/v3/native_structure_advice_node.py'),
        '--output', str(output / 'advice_sidecar')]
    if bool(cache_manifest) != bool(cache_sha256):
        raise ValueError('Cache manifest and SHA256 must be supplied together')
    if cache_manifest:
        sidecar += ['--cache-manifest', str(cache_manifest), '--cache-manifest-sha256', cache_sha256]
    return planner, sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner-seed', required=True, type=int)
    parser.add_argument('--world', default='tunnel')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache-manifest', type=Path)
    parser.add_argument('--cache-manifest-sha256')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    planner_command, sidecar_command = commands(planner_seed=args.planner_seed, world=args.world, output=output,
        cache_manifest=args.cache_manifest, cache_sha256=args.cache_manifest_sha256)
    with (output / 'native_method_commands.json').open('x') as stream:
        json.dump(dict(planner=planner_command, sidecar=sidecar_command, sole_waypoint_publisher='tare_planner_node'), stream, indent=2)
    stopping = False
    def on_signal(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    processes, streams = [], []
    failed = None
    try:
        for name, command in [('sidecar', sidecar_command), ('planner', planner_command)]:
            log = (output / (name + '.log')).open('xb')
            streams.append(log)
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((name, child))
            if name == 'sidecar':
                deadline = time.monotonic() + 15
                while not (output / 'advice_sidecar/ready.json').is_file():
                    if child.poll() is not None or time.monotonic() >= deadline or stopping:
                        raise RuntimeError('Sidecar did not become ready before native planner start')
                    time.sleep(.05)
        while not stopping:
            for name, child in processes:
                if child.poll() is not None:
                    raise RuntimeError(name + ' exited unexpectedly: ' + str(child.returncode))
            time.sleep(.1)
    except Exception as error:
        failed = repr(error)
    finally:
        for _, child in reversed(processes):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for _, child in reversed(processes):
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
        for stream in streams:
            stream.close()
        with (output / 'native_method_summary.json').open('x') as stream:
            json.dump(dict(failure=failed, received_shutdown=stopping,
                child_exit_codes={name: child.returncode for name, child in processes},
                model_executed=False, new_control_publisher=False), stream, indent=2)
    if failed:
        print(failed, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

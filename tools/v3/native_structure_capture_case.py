"""Thin case-runner adaptation: native control, instrumented shadow only."""
import shlex
import run_mtare_single_robot_case_v1 as original


def shadow_command(args, planner_output):
    if args.world != 'tunnel' or args.environment_seed != 11 or args.runtime_sec != 120:
        raise ValueError('frozen tunnel/seed11/120-second capture only')
    if args.method_family != 'original_mtare':
        raise ValueError('only original M-TARE may control this capture')
    return shlex.join(['python3', '/workspace/tools/v3/launch_native_structure_shadow.py',
        '--planner-seed', '11', '--world', 'tunnel', '--output', str(planner_output)])


if __name__ == '__main__':
    original.method_command = shadow_command
    raise SystemExit(original.main())

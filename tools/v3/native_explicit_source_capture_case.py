"""Same development case, explicitly enable same-callback source tracing."""
from pathlib import Path
import run_mtare_single_robot_case_v1 as original
from native_structure_capture_case import shadow_command

OLD='/workspace/configs/v3/gate5/roslaunch/system_seeded.launch'
NEW='/workspace/configs/v3/gate6/roslaunch/system_explicit_scan_source.launch'


def source_system_command(command):
    if command.startswith('roslaunch '+OLD+' '):
        return command.replace(OLD,NEW,1)
    return command


if __name__=='__main__':
    base_start=original.start
    def start(command,log_path):
        return base_start(source_system_command(command),log_path)
    original.start=start
    original.method_command=shadow_command
    raise SystemExit(original.main())

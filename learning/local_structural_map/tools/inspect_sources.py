import argparse
import platform
import subprocess
from pathlib import Path

from learning.local_structural_map.datasets.common import bag_topic_summary
from learning.local_structural_map.datasets.subt_mrs import SubTMRSDatasetAdapter
from learning.local_structural_map.tools.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--subt-dir", default="/mnt/nas_znfy/Shared-2/dataset/SLAM/subt/SuperOdometry/subt/SubT_MRS_Final_Challenge_UGV2/Final_Challenge_UGV2_Rosbag")
    parser.add_argument("--subt-extrinsics", default="/mnt/nas_znfy/Shared-2/dataset/SLAM/subt/SuperOdometry/ex/SubT_MRS_Final_Challenge_UGV2_Extrinsics.yaml")
    parser.add_argument("--lamp-scan-bag", default="/mnt/nas_znfy/Shared-2/dataset/SLAM/subt/LAMP/tunnel/rosbag/husky3.bag")
    parser.add_argument("--lamp-odom-bag", default="/mnt/nas_znfy/Shared-2/dataset/SLAM/subt/LAMP/tunnel/ground_truth/husky3_odom.bag")
    parser.add_argument("--mtare-bag", default="")
    args = parser.parse_args()
    git_status = subprocess.run(["git", "status", "--short"], cwd=Path.cwd(), text=True, capture_output=True)
    data = {
        "host": platform.node(),
        "python": platform.python_version(),
        "git_status": git_status.stdout if git_status.returncode == 0 else "UNAVAILABLE: workspace is not a git repository",
        "subt_mrs": SubTMRSDatasetAdapter(args.subt_dir, args.subt_extrinsics).audit(),
        "lamp": bag_topic_summary([Path(args.lamp_scan_bag), Path(args.lamp_odom_bag)]),
    }
    if args.mtare_bag:
        data["mtare"] = bag_topic_summary([Path(args.mtare_bag)])
    write_json(args.output, data)


if __name__ == "__main__":
    main()

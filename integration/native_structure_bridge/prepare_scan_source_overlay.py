"""Add opt-in source tracing to the pinned scan callback; no geometry changes."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = 'ed906e39762379074c2fbc4ac8a597403181315626d993fcfa43b6eb4f3c028e'


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('PINNED_SCAN_SOURCE_ANCHOR_CHANGED')
    return text.replace(old, new, 1)


def transform(text):
    if hashlib.sha256(text.encode()).hexdigest() != EXPECTED:
        raise ValueError('PINNED_VEHICLE_SIMULATOR_SOURCE_CHANGED')
    text = once(text, '#include <std_msgs/Bool.h>\n',
        '#include <std_msgs/Bool.h>\n#include <std_msgs/String.h>\n#include <sstream>\n#include <cstdint>\n')
    text = once(text, 'bool use_gazebo_time = false;\n', '''bool use_gazebo_time = false;
// Default disabled: observations only; never a transform, scan or motion source.
bool recordNativeScanSources = false;
ros::Publisher pubNativeScanSources;
std::string nativeScanSourceSession;
std::uint64_t nativeRegisteredSequence = 0;
''')
    text = once(text, '  nhPrivate.getParam("use_gazebo_time", use_gazebo_time);\n', '''  nhPrivate.getParam("use_gazebo_time", use_gazebo_time);
  nhPrivate.param<bool>("record_native_scan_sources", recordNativeScanSources, false);
  if (recordNativeScanSources) {
    nativeScanSourceSession = std::to_string(ros::WallTime::now().toNSec());
    pubNativeScanSources = nh.advertise<std_msgs::String>("/native_structure/scan_sources", 100);
  }
''')
    text = once(text, '  pubScanPointer->publish(scanData2);\n', r'''  if (recordNativeScanSources) {
    // Both headers come from THIS callback, not a timestamp-nearest lookup.
    // Echo precedes the original publish; output receipt must still be checked.
    std::ostringstream out;
    out << "{\"schema_version\":\"native_scan_source_v1\",\"session_id\":\""
        << nativeScanSourceSession << "\",\"source_seq\":" << scanIn->header.seq
        << ",\"source_stamp_ns\":" << scanIn->header.stamp.toNSec()
        << ",\"registered_seq\":" << nativeRegisteredSequence
        << ",\"registered_stamp_ns\":" << scanData2.header.stamp.toNSec()
        << ",\"raw_topic\":\"/velodyne_points\",\"registered_topic\":\"/registered_scan\""
        << ",\"source_frame\":\"velodyne\",\"registered_frame\":\"map\""
        << ",\"raw_frame_matches\":" << (scanIn->header.frame_id == "velodyne" || scanIn->header.frame_id == "/velodyne" ? "true" : "false")
        << ",\"physical_pose_verified\":false}";
    std_msgs::String evidence; evidence.data = out.str();
    pubNativeScanSources.publish(evidence);
  }
  pubScanPointer->publish(scanData2);
  ++nativeRegisteredSequence;
''')
    return text


def prepare(source, output):
    if output.exists():
        raise ValueError('NON_OVERWRITING_SOURCE_OVERLAY_REQUIRED')
    original = source.read_text()
    changed = transform(original)
    output.mkdir(parents=True, exist_ok=False)
    # Mechanical frozen-source transform into a fresh build overlay only.
    (output/'vehicleSimulator.cpp').write_text(changed)
    manifest = dict(original_sha256=EXPECTED,
        output_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        default_enabled=False, scan_arithmetic_changed=False, original_publish_preserved=True,
        new_topic='/native_structure/scan_sources', purpose='explicit_callback_source_identity_not_registration_accuracy')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.output)))

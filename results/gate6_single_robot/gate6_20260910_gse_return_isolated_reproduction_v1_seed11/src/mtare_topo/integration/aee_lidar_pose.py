"""AEE-specific commanded scan pose from original vehicle odometry.

Source: frozen runnable image vehicleSimulator.cpp 372-373, 374, 402-441;
VLP-16.urdf.xacro base_scan_joint z=.0377. This is NOT a generic extrinsic
or a measurement of actual Gazebo link pose. Physics/transport lag remains
unmeasured. It uses localization only, never world geometry or structural IDs.
"""
import math
import numpy as np


def commanded_lidar_pose(vehicle_xyz, vehicle_rpy, yaw_rate):
    xyz=np.asarray(vehicle_xyz,float);rpy=np.asarray(vehicle_rpy,float)
    if xyz.shape!=(3,) or rpy.shape!=(3,) or not np.isfinite(xyz).all() or not np.isfinite(rpy).all() or not math.isfinite(yaw_rate):
        raise ValueError('finite original odometry position, Euler angles and yaw rate required')
    roll,pitch,yaw=rpy
    # Original simulator updates vehicleYaw AFTER deriving vehicleRoll/Pitch.
    previous_yaw=yaw-.005*yaw_rate
    terrain_roll=roll*math.cos(previous_yaw)-pitch*math.sin(previous_yaw)
    terrain_pitch=roll*math.sin(previous_yaw)+pitch*math.cos(previous_yaw)
    cr,sr=math.cos(terrain_roll),math.sin(terrain_roll)
    cp,sp=math.cos(terrain_pitch),math.sin(terrain_pitch)
    rotation=np.array([[cp,sp*sr,sp*cr],[0,cr,-sr],[-sp,cp*sr,cp*cr]])
    pose=np.eye(4);pose[:3,:3]=rotation
    pose[:3,3]=xyz+rotation@np.array([0,0,.0377])
    return pose

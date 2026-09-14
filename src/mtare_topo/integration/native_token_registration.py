"""Source-bound sensor-frame ICP for recorded native retrieval candidates.

Uses original current-frame returns, not resampled model image points. Sensor
poses remain explicitly commanded poses, not measured physical link poses.
Successful local ICP is evidence only: no task identity, direction or merge.
"""
from dataclasses import asdict, dataclass
import hashlib
import io

import numpy as np

from mtare_topo.topology.gse_registration import RegistrationConfig, _se3, register_local_clouds


@dataclass(frozen=True)
class SensorRegistrationInput:
    epoch: str
    segment: str
    order: int
    stamp_ns: int
    points_sensor_m: np.ndarray
    world_from_sensor: np.ndarray
    raw_return_indices: tuple[int, ...]
    input_sha256: str
    source_key: str
    physical_pose_verified: bool = False

    def __post_init__(self):
        points = np.asarray(self.points_sensor_m, dtype='<f8')
        pose = _se3(self.world_from_sensor)
        if (not self.epoch or not self.segment or type(self.order) is not int or self.order < 0
                or type(self.stamp_ns) is not int or self.stamp_ns < 0
                or points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
                or len(self.raw_return_indices) != len(points)
                or len(set(self.raw_return_indices)) != len(points)
                or self.physical_pose_verified is not False):
            raise ValueError('invalid commanded sensor-frame registration input')
        object.__setattr__(self, 'points_sensor_m', np.frombuffer(points.tobytes(), dtype='<f8').reshape(points.shape))
        object.__setattr__(self, 'world_from_sensor', np.frombuffer(pose.tobytes(), dtype=pose.dtype).reshape(4, 4))

    @property
    def points_sha256(self):
        return hashlib.sha256(self.points_sensor_m.tobytes()).hexdigest()


def original_sensor_cloud(archive, record):
    """Call after archive.records authenticates this record's source metadata.

    Only finite original returns within the existing 10m sensor-local domain.
    No voxel resampling, learned scores, teacher IDs or extrapolated surfaces.
    """
    entry = archive.index[record.order]
    if entry['epoch'] != record.record_id or entry['raw_source_frame_keys'] != list(record.source_refs):
        raise ValueError('archive/record mismatch')
    reference = entry['input_ref']
    with np.load(io.BytesIO(archive._read(reference['path'], reference['sha256'])), allow_pickle=False) as data:
        raw = data['raw_pointcloud_bytes'][-1]
        pose = data['world_from_sensor'][-1]
    # This is the original AEE 16x350, 22-byte PointCloud2 contract authenticated
    # by the exporter and model input loader; do not treat arbitrary buffers as it.
    row = archive._json(archive.run + f'/artifacts/frozen_tokens/record_{record.order:06d}.json')
    layout = row['source']['frames'][-1]['pointcloud_layout']
    offsets = {f['name']:(f['offset'],f['datatype'],f['count']) for f in layout['fields']}
    if (layout['point_step'] != 22 or layout['is_bigendian'] is not False
            or (layout['height'],layout['width'],layout['row_step']) != (350,16,352)
            or any(offsets.get(k) != (offset,7,1) for k,offset in (('x',0),('y',4),('z',8)))
            or raw.dtype != np.uint8 or raw.shape != (123200,)):
        raise ValueError('original organized raw layout mismatch')
    dtype = np.dtype({'names':['x','y','z'], 'formats':['<f4']*3, 'offsets':[0,4,8], 'itemsize':22})
    data = np.frombuffer(raw.tobytes(), dtype=dtype)
    points = np.column_stack([data[k] for k in ('x','y','z')]).astype(np.float64)
    finite = np.isfinite(points).all(axis=1)
    ids = np.flatnonzero(finite & (np.linalg.norm(points,axis=1) > 0) & (np.linalg.norm(points,axis=1) <= 10.0))
    return SensorRegistrationInput(record.record_id, record.segment, record.order,
        record.source_frames[-1].stamp_ns, points[ids], pose, tuple(int(i) for i in ids),
        reference['sha256'], record.source_refs[-1])


def verify_native_candidate(current, historical, config):
    if not isinstance(current, SensorRegistrationInput) or not isinstance(historical, SensorRegistrationInput):
        raise ValueError('typed sensor clouds required')
    if not isinstance(config, RegistrationConfig):
        raise ValueError('explicit registration configuration required')
    if (current.segment != historical.segment or historical.order >= current.order
            or historical.stamp_ns >= current.stamp_ns or current.epoch == historical.epoch):
        raise ValueError('strictly past same-segment candidate required')
    initial = np.linalg.inv(historical.world_from_sensor) @ current.world_from_sensor
    common = dict(current_epoch=current.epoch, historical_epoch=historical.epoch,
        current_source_key=current.source_key, historical_source_key=historical.source_key,
        current_input_sha256=current.input_sha256, historical_input_sha256=historical.input_sha256,
        current_points_sha256=current.points_sha256, historical_points_sha256=historical.points_sha256,
        source_points=len(current.points_sensor_m), target_points=len(historical.points_sensor_m),
        initial_historical_sensor_from_current_sensor=initial.tolist(),
        coordinate_frame='current_sensor_to_historical_sensor', initial_pose_source='original_commanded_sensor_poses',
        physical_pose_verified=False, task_identity_verified=False, direction_verified=False,
        graph_mutated=False, control_changed=False)
    if min(len(current.points_sensor_m),len(historical.points_sensor_m)) < 3:
        return dict(**common, registration=None, accepted=False, rejection_reasons=['insufficient_original_returns'])
    evidence = register_local_clouds(current.points_sensor_m,historical.points_sensor_m,initial,config)
    return dict(**common, registration=asdict(evidence), accepted=evidence.accepted,
        rejection_reasons=list(evidence.rejection_reasons))

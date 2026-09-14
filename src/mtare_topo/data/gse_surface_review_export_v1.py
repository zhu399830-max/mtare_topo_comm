"""Project sealed observed-material arrays into the blind review schema.

Caller owns exact source scope and byte-hash verification. No file IO, new
projection, teacher candidate filtering, downsampling, or label generation.
"""
import numpy as np

from mtare_topo.data.gse_surface_review_v1 import validate_bundle


def observed_material_bundle(arrays, *, observation_id, source_frame_indices):
    points=arrays["points_current_sensor_m"]
    valid=arrays["first_return_valid"]
    slots=arrays["history_slot"]
    if type(points) is not np.ndarray or points.shape!=(57600,3) or points.dtype!=np.float32:
        raise ValueError("original full float32 five-frame point array required")
    if type(valid) is not np.ndarray or valid.shape!=(57600,) or valid.dtype!=np.bool_:
        raise ValueError("original return mask required")
    if type(slots) is not np.ndarray or slots.dtype.kind not in 'iu' or not np.array_equal(slots,np.repeat(np.arange(5),11520)):
        raise ValueError("original five-frame history ordering required")
    if not np.isfinite(points[valid]).all():raise ValueError("valid return coordinates must be finite")
    # Invalid rays are not points. All valid returns survive, even outside the
    # display ROI. The browser applies its stated 10m display mask, not labels.
    return validate_bundle(dict(schema="gse_surface_review_bundle_v1",observation_id=observation_id,
        coordinate_frame="current_sensor_m",source_frame_indices=list(source_frame_indices),
        points_xyz_m=points[valid].astype(float).tolist(),point_history_slots=slots[valid].tolist()))

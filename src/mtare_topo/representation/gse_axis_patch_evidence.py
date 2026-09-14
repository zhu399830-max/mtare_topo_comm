"""Axis/observed-patch intersection features, never whole-tunnel closure.

Bounding-box inclusion is only a necessary local support proxy. A patch can
have holes; neither a plane intersection nor its bounding box proves a cap.
Unresolved pairs retain masks and NaN positions, never fabricated origins.
"""
from dataclasses import dataclass
import numpy as np
from .gse_surface_patches_v1 import SurfacePatches


@dataclass(frozen=True)
class AxisPatchEvidence:
    coordinate_frame: str
    position_m: np.ndarray              # S,M,3; undefined is NaN
    unique_intersection: np.ndarray     # S,M
    within_patch_bounds: np.ndarray    # necessary proxy, not surface coverage
    normal_axis_abs_dot: np.ndarray
    extrapolation_m: np.ndarray
    distance_to_patch_bounds_m: np.ndarray  # continuous evidence; not a cutoff
    whole_tunnel_closure_verified: bool = False


def axis_patch_evidence(axes_m, axis_valid, patches, *, coordinate_frame):
    if (not isinstance(axes_m, np.ndarray) or axes_m.ndim != 3
            or axes_m.shape[1:] != (3, 3) or len(axes_m) > 32
            or axes_m.dtype not in (np.dtype('float32'), np.dtype('float64'))):
        raise ValueError('at most32 floating three-control axes required')
    if (not isinstance(axis_valid, np.ndarray) or axis_valid.dtype != np.bool_
            or axis_valid.shape != (len(axes_m),) or not np.isfinite(axes_m[axis_valid]).all()):
        raise ValueError('finite valid axes and explicit mask required')
    if not isinstance(patches, SurfacePatches):
        raise ValueError('typed observed patches required')
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError('explicit common frame required; caller binds provenance')
    m = len(patches.centers_m)
    if m > 4096:
        raise ValueError('patch capacity exceeded')
    for name in ('centers_m', 'normals', 'bounds_min_m', 'bounds_max_m'):
        x = getattr(patches, name)
        if not isinstance(x, np.ndarray) or x.shape != (m, 3) or not np.isfinite(x).all():
            raise ValueError('invalid patch geometry')
    if patches.normal_valid.dtype != np.bool_ or patches.normal_valid.shape != (m,):
        raise ValueError('explicit patch normal mask required')
    if (patches.bounds_min_m > patches.bounds_max_m).any():
        raise ValueError('inverted patch bounds')
    n = patches.normals.astype(np.float64)
    if not np.allclose(np.linalg.norm(n[patches.normal_valid], axis=1), 1., atol=1e-6, rtol=0):
        raise ValueError('supported unit normals required')
    axes = np.where(axis_valid[:, None, None], axes_m, 0.).astype(np.float64)
    d = axes[:, 2]-axes[:, 0]
    length = np.linalg.norm(d, axis=1)
    tol = 64*np.finfo(axes_m.dtype).eps
    supported = axis_valid & (length > tol)
    u = np.divide(d, length[:, None], out=np.zeros_like(d), where=supported[:, None])
    dot = u @ n.T
    unique = supported[:, None] & patches.normal_valid[None] & (np.abs(dot) > tol)
    numerator = np.einsum('smc,mc->sm', patches.centers_m[None]-axes[:, 0, None], n)
    t = np.divide(numerator, dot, out=np.full(dot.shape, np.nan), where=unique)
    xyz = axes[:, 0, None]+t[..., None]*u[:, None]
    if not np.isfinite(xyz[unique]).all():
        raise ValueError('nonfinite resolved intersection')
    # Roundoff padding only, no learned/tuned support dilation.
    pad = 64*np.finfo(np.float64).eps*np.maximum(1., np.max(np.abs(patches.centers_m), axis=1))
    inside = unique & np.all((xyz >= patches.bounds_min_m[None]-pad[None, :, None])
                            & (xyz <= patches.bounds_max_m[None]+pad[None, :, None]), axis=-1)
    extrapolation = np.maximum(np.maximum(-t, t-length[:, None]), 0.)
    distance = np.linalg.norm(np.maximum(np.maximum(patches.bounds_min_m[None]-xyz,
                                                   xyz-patches.bounds_max_m[None]), 0.), axis=-1)
    absolute_dot = np.where(supported[:, None] & patches.normal_valid[None], np.abs(dot), np.nan)
    for value in (xyz, unique, inside, absolute_dot, extrapolation, distance):
        value.setflags(write=False)
    return AxisPatchEvidence(coordinate_frame, xyz, unique, inside, absolute_dot, extrapolation, distance)

"""Observed material for labeling; no hidden map, labels, weights or decisions."""
from dataclasses import fields

import numpy as np
import torch

from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points
from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches, build_patch_neighbors
from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid, query_patch_gaps


def build_observed_material(ranges_m, valid_mask, translation_m, yaw_deg):
    """Use exactly the student's projection, including float32 normalization.

    All current and past returns survive in the numerical artifact; only the
    surface-patch extractor restricts surface membership to the fixed10m ball.
    A ray which returns outside the ball is not converted into a surface there.
    No observation ID or teacher can be passed as a model/geometric input.
    """
    expected = ((ranges_m, (5,16,720), "float32"), (valid_mask, (5,16,720), "uint8"),
                (translation_m, (5,3), "float32"), (yaw_deg, (5,), "float32"))
    for value, shape, dtype in expected:
        if type(value) is not np.ndarray or value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError("exact fiveframe raw input shapes and dtypes required")
        if not np.isfinite(value).all():
            raise ValueError("nonfinite observed input")
    if (np.any(ranges_m < 0) or np.any(ranges_m > 50) or np.any(valid_mask > 1)
            or np.any(translation_m[-1] != 0) or yaw_deg[-1] != 0):
        raise ValueError("frozen sensor units/validity/current-origin contract drift")
    image = np.stack((ranges_m / np.float32(50), valid_mask.astype(np.float32)), axis=1)
    with torch.no_grad():
        xyz, mask = register_causal_lidar_points(torch.from_numpy(image[None]),
            torch.from_numpy(translation_m.copy()[None]), torch.from_numpy(yaw_deg.copy()[None]))
    points = xyz.numpy().reshape(-1,3).copy(); valid = mask.numpy().reshape(-1).copy()
    frame = np.repeat(np.arange(5, dtype=np.int64), 16 * 720)
    origins = np.broadcast_to(translation_m[:,None,None,:], (5,16,720,3)).reshape(-1,3).copy()
    patches = extract_surface_patches(points, valid, frame, ray_origins_m=origins)
    relations = build_patch_neighbors(patches)
    grid = build_surface_ray_grid(origins, points, valid, frame)
    gaps = query_patch_gaps(grid, patches.centers_m, relations.neighbor_index, relations.valid)
    arrays = {"points_current_sensor_m": points, "first_return_valid": valid,
              "ray_origins_current_sensor_m": origins, "history_slot": frame}
    for prefix, obj in (("patch", patches), ("relation", relations), ("grid", grid), ("gap", gaps)):
        for field in fields(obj):
            value = getattr(obj, field.name)
            if isinstance(value, np.ndarray):
                arrays[prefix + "_" + field.name] = value
    evidence = gaps.fractions.copy()
    evidence[~gaps.gap_defined] = [0.,0.,1.]
    arrays["observed_relation_fractions"] = evidence
    metrics = {"input_frames": 5, "input_rays": 57600, "first_returns": int(valid.sum()),
        "roi_surface_returns": int((patches.point_patch_index >= 0).sum()),
        "surface_patches": len(patches.centers_m), "normal_supported_patches": int(patches.normal_valid.sum()),
        "message_neighbors": int(relations.valid.sum()), "defined_observed_gaps": int(gaps.gap_defined.sum()),
        "free_voxels": int((grid.state == 1).sum()), "occupied_voxels": int((grid.state == 2).sum()),
        "unknown_voxels": int((grid.state == 0).sum()),
        "grid_source_sha256": grid.source_geometry_sha256, "grid_content_sha256": grid.content_sha256,
        "surface_voxel_m": .5, "evidence_voxel_m": .25, "surface_radius_m": 10.,
        "coordinate_frame": "current_sensor", "physical_connectivity": False,
        "anchor_labels": 0, "opening_labels": 0, "human_reviewed": False, "training_eligible": False}
    for array in arrays.values():
        array.setflags(write=False)
    return arrays, metrics


def draw_material(arrays, metrics, destination, title):
    """All ROI points shown; center slices are named, never max-projection maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = [font_manager.FontProperties(fname=font_path).get_name()]
    xyz = arrays["points_current_sensor_m"]
    valid = arrays["first_return_valid"] & (np.linalg.norm(xyz, axis=1) <= 10.)
    centers = arrays["patch_centers_m"]
    fig, axes = plt.subplots(2, 3, figsize=(15,9), layout="constrained")
    for row, (a,b,view) in enumerate(((0,1,"XY俯视"),(0,2,"XZ侧视"))):
        axes[row,0].scatter(xyz[valid,a], xyz[valid,b], c=arrays["history_slot"][valid], s=.3,
                            cmap="viridis", vmin=0,vmax=4, rasterized=True)
        axes[row,0].set_title(view + "：五帧实际回波（颜色为历史帧）")
        if len(centers):
            axes[row,1].scatter(centers[:,a], centers[:,b], c=arrays["patch_normal_valid"], s=5,
                                cmap="coolwarm",vmin=0,vmax=1,rasterized=True)
        axes[row,1].set_title(view + "：面片中心（红色有法向支持）")
        # Cell40 is[0,.25), not a collapsed 2D map: preserve this limitation.
        plane = arrays["grid_state"][:,:,40] if row == 0 else arrays["grid_state"][:,40,:]
        axes[row,2].imshow(plane.T, origin="lower", extent=(-10,10,-10,10),
                           interpolation="nearest", cmap=ListedColormap(["#d8d8d8","#75b5a3","#303844"]),vmin=0,vmax=2)
        axes[row,2].set_title(("z" if row == 0 else "y") + "∈[0,0.25)m切片：灰未知/绿射线穿过/深色回波")
        for ax in axes[row]:
            ax.set(xlim=(-10,10),ylim=(-10,10),xlabel="x（米）",ylabel=("y" if row==0 else "z")+"（米）")
            ax.set_aspect("equal"); ax.plot(0,0,"r+",markersize=7)
    fig.suptitle(title + f"\n{metrics['surface_patches']}面片；仅观测材料，无结构标签、无物理可达证明",fontsize=13)
    fig.savefig(destination, dpi=120); plt.close(fig)

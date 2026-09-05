"""Analytic counterexample: a visible opening need not expose its endpoint.

This is a 2-D horizontal slice, not a dataset-performance experiment and not
proof that a particular 3-D scan supports an entire junction label.
"""
import numpy as np


def test_t_junction_open_branch_visible_but_construction_endpoint_has_no_wall_support():
    # Two rectangular free-space passages: [-6,6]x[-1,1] and
    # [-1,1]x[0,6]. The upper branch's construction endpoint is at (0,0).
    # A horizontal LiDAR at (0,0) sees the union boundary, not internal seams.
    theta = np.arange(720) * (2 * np.pi / 720)
    directions = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    dx, dy = directions.T
    with np.errstate(divide="ignore", invalid="ignore"):
        horizontal_exit = np.minimum(6 / np.abs(dx), 1 / np.abs(dy))
        branch_exit = np.where(dy > 0, np.minimum(1 / np.abs(dx), 6 / dy), 0)
    ranges = np.maximum(horizontal_exit, branch_exit)
    returns = ranges[:, None] * directions
    branch_surface = branch_exit > horizontal_exit + 1e-10
    branch_returns = returns[branch_surface]
    assert len(branch_returns) > 0
    # Every ray to these walls traverses known free space into the branch.
    assert ranges[180] == 6  # 90 degrees, unoccluded upper passage
    observed_axial_start = np.min(branch_returns[:, 1])
    assert observed_axial_start > 1
    assert observed_axial_start > .25  # old physical-endpoint support band
    # Therefore that band is not a necessary condition for seeing a port.

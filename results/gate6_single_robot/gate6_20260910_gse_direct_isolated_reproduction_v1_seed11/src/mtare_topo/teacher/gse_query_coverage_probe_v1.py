"""Fixed real-observation scoring probe, not model predictions or calibration."""
import gzip
import itertools
import json
import numpy as np
import torch
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points
from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.evaluation.gse_reference_covered_counts_v1 import (
    score_bound_anchor_references, score_bound_opening_references)

PATH = 'results/gate3_semantics/gate3_20260908_gse_terminal_pair_probe_v1_seed20260906/artifacts/S01_flat_tree_small_C01__c1_mixed.json.gz'
SHA = 'eae1594bc126dc3b60a542e85cada824335bf01788117a2ab06206875fdb8d5a'


def fixed_queries():
    # Cell-interior lattice fixed before reading the observation. All 64 kept,
    # including those outside the scoring sphere. No target-centred selection.
    return np.asarray(list(itertools.product((-7.125, -2.125, 2.125, 7.125), repeat=3)))


def inspect_coverage(bundle, root, opened):
    source = bundle['source']
    if (source['task'], source['source_sequence_id'], source['frame_rows']) != (
            'S01_flat_tree_small_C01__c1_mixed', 225, [289,290,291,292,293]):
        raise ValueError('exact frozen observation required')
    archived = json.loads(gzip.decompress(read_pinned(root, PATH, SHA)))
    opened[PATH] = SHA
    rows = archived['observations']
    if len(rows) != 1 or rows[0]['source'] != source:
        raise ValueError('sealed target source mismatch')
    target = rows[0]['new']
    manifest = {k: target[k] for k in ('source_binding','target_record_sha256')}
    student = bundle['student']
    ranges, mask = student['ranges_m'], student['valid_mask']
    translation = student['relative_translation_current_sensor_m']
    yaw = student['relative_yaw_current_sensor_deg']
    image = np.stack((ranges/np.float32(50), mask.astype(np.float32)),axis=1)
    with torch.no_grad():
        points, valid = register_causal_lidar_points(torch.from_numpy(image[None]),
            torch.from_numpy(translation.copy()[None]), torch.from_numpy(yaw.copy()[None]))
    points = points.numpy().reshape(-1,3); valid = valid.numpy().reshape(-1)
    origins = np.broadcast_to(translation[:,None,None,:], (5,16,720,3)).reshape(-1,3)
    grid = build_surface_ray_grid(origins, points, valid, np.repeat(np.arange(5),11520))
    q = fixed_queries()
    common = dict(bundle=bundle, grid=grid, produced_targets=target, manifest_row=manifest,
        predicted_xyz_m=q, confidence=np.ones(len(q)), threshold=.5)
    anchors = score_bound_anchor_references(**common)
    # All scales diagnostic; none selects an opening acceptance threshold.
    openings = {str(radius): score_bound_opening_references(**common, maximum_error_m=radius)
                for radius in (1.,2.,4.)}
    return dict(source=source, sealed_targets=dict(path=PATH,sha256=SHA),
        query_xyz_m=q.tolist(), query_role='fixed_lattice_diagnostic_not_model_predictions',
        grid_source_sha256=grid.source_geometry_sha256, grid_content_sha256=grid.content_sha256,
        anchors=anchors, openings_diagnostic_scales=openings, query_count=len(q),
        qualified_complete_labels=0, optimizer_steps=0, scientific_gate_pass=False,
        full_f1=None, threshold_selection=False)

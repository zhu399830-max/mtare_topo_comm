from pathlib import Path
from mtare_topo.data.gse_synthetic_fit_scope import compile_scope, CHECKPOINT
from mtare_topo.governance_surface_selection import digest

SCHEMA = 'v3_axis_pair_diagnostic_card_v1'
SLUG = 'gse_axis_pair_diagnostic_v1'


def scope(root, include_patches=False, patch_centers=False):
    old = compile_scope(root)
    keys = ('observations', 'primary_observations', 'frame_occurrences',
            'unique_archive_frame_keys', 'independent_program_types', 'section_variants',
            'real_worlds', 'ray_positions', 'history_spacing_m', 'view_offsets_m',
            'sampling', 'teacher', 'model_inputs', 'forbidden_forward_inputs')
    result = {k: old[k] for k in keys}
    result.update(input_sha256={p: h for p, h in old['input_sha256'].items() if p != CHECKPOINT},
                  split='Synthetic-only diagnostic, no training or held-out claim',
                  checkpoint_reads=0, optimizer_steps=0, thresholds_m=[1., 2., 4.],
                  method='Default RobustPrimitiveRelationBaseline axes, all chord pairs; no attachment input or filtering')
    if include_patches:
        result.update(patch_parameters=dict(voxel_size_m=.5,roi_radius_m=10.,max_patches=4096),
            method='Default observation axes plus all axis-patch unique intersections inside measured patch bounds; all axis-pair hypotheses retained; no closure claim',
            patch_candidate_selection='Unique intersection and observed bounding-box inclusion, no score or GT filter',
            acceptance='All27 visible anchors must have a1m candidate to consider full-node filtering; expose all surplus candidates, no research pass')
    if patch_centers:
        if not include_patches: raise ValueError('patches required')
        result.update(method='All observed fixed voxel patch centers plus unchanged axis-pair hypotheses; no normal, score or GT selection',
                      patch_candidate_selection='all_observed_patch_centers',
                      acceptance='All27 anchors covered at1m is a necessary input-coverage condition only; every surplus candidate scored, not detection success')
    return result


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    try:
        centers = card['card_id'] == 'gse_patch_center_diagnostic_v1'
        patches = centers or card['card_id'] == 'gse_axis_patch_diagnostic_v1'
        s = scope(Path(__file__).resolve().parents[2], patches, centers)
        assert card['schema_version'] == SCHEMA and card['card_id'] in (SLUG,'gse_axis_patch_diagnostic_v1','gse_patch_center_diagnostic_v1')
        assert card['operation'] == 'audit' and card['scope'] == s
        assert card['scope_sha256'] == digest(s)
        a = card['approval']
        assert a['status'] == 'APPROVED' and a['scope_sha256'] == digest(s)
        assert a['authorized_operations'] == ['audit'] and a['authorized_gates'] == [3]
        assert a['confirmation_reference']
    except (KeyError, ValueError, TypeError, AssertionError):
        return ValidationReport(False, ('Exact synthetic45 axis-pair audit scope required',))
    return ValidationReport(True, ())

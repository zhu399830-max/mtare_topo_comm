"""Source-bound conditional anchor negatives; no branch-negative invention."""
import numpy as np
import torch
from torch.nn import functional as F
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.gse_reference_exclusion_binding_v1 import bound_reference_exclusion
from .anchor_branch_loss import anchor_branch_loss


def conditional_anchor_branch_loss(prediction, target, *, bundle, grid,
                                  produced_targets, frozen_manifest):
    """Bind positive positions and exact query-wise background to one source.

    The explicit frozen reference indices permit partial targets without
    treating other references (including terminals) as absent. Branch target
    provenance is still the responsibility of the authenticated target reader.
    """
    if set(frozen_manifest) != {'source_binding', 'target_record_sha256', 'target_reference_indices'}:
        raise ValueError('closed independent target manifest required')
    record = produced_targets['record']
    if (produced_targets['source_binding'] != frozen_manifest['source_binding']
            or produced_targets['target_record_sha256'] != frozen_manifest['target_record_sha256']
            or canonical_sha(record) != frozen_manifest['target_record_sha256']
            or record['source_frame_indices'] != frozen_manifest['source_binding']['source']['frame_rows']):
        raise ValueError('frozen positive source or content mismatch')
    indices = frozen_manifest['target_reference_indices']
    if (not isinstance(indices, tuple) or len(indices) != len(set(indices))
            or any(type(i) is not int or not 0 <= i < len(record['anchors']) for i in indices)):
        raise ValueError('explicit unique reference target indices required')
    positions = np.asarray([record['anchors'][i]['position_m'] for i in indices], dtype=float).reshape(-1,3)
    expected = torch.as_tensor(positions, dtype=target.position_m.dtype, device=target.position_m.device)
    if not torch.equal(expected, target.position_m):
        raise ValueError('partial positive positions differ from frozen references')
    if target.anchors_complete:
        raise ValueError('conditional and complete-region supervision cannot be mixed')
    base = anchor_branch_loss(prediction, target)
    queries = prediction.position_m.detach().cpu().double().numpy()
    evidence = bound_reference_exclusion(bundle, grid, queries,
        expected_binding=frozen_manifest['source_binding'], matching_radius_m=4.)
    mask = torch.tensor(evidence['reference_negative_mask'], device=prediction.position_m.device, dtype=torch.bool)
    # A matched positive may start far from its reference; regression must be
    # allowed to move it. Never supervise the same query positive and negative.
    mask[base['anchor_assignment']] = False
    logits = prediction.presence_logits[mask]
    counts = dict(base['counts']); terms = dict(base['terms'])
    if logits.numel():
        numerator = terms['anchor_presence'] * counts['anchor_presence'] + F.softplus(logits).sum()
        counts['anchor_presence'] += logits.numel()
        terms['anchor_presence'] = numerator / counts['anchor_presence']
    evidence['used_negative_query_indices'] = torch.nonzero(mask).flatten().cpu().tolist()
    evidence['query_xyz_m'] = queries.tolist()
    return dict(base, total=sum(terms.values()), terms=terms, counts=counts,
                has_supervision=any(counts.values()), conditional_anchor_evidence=evidence)

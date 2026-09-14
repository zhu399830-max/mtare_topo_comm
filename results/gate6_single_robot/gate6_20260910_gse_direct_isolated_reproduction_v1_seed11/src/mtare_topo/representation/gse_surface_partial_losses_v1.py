"""Source-bound conditional anchor background for partial structure targets.

Diagnostic integration, not formal training qualification. No fixed-grid result
is interpolated to prediction positions. Opening background is unchanged unless
an explicit opening matching radius enables its separate reference inventory.
The caller must bind positive targets to the same frozen observation manifest.
"""
from dataclasses import replace
import torch
from torch.nn import functional as F
from .gse_surface_losses_v1 import _validate, surface_relation_losses
from mtare_topo.teacher.gse_reference_exclusion_binding_v1 import bound_reference_exclusion


def bound_partial_surface_losses(prediction, *, produced_targets, manifest_rows, bundles, grids,
                                 opening_matching_radius_m=None):
    """Validate frozen positive records and share identity with negative queries.

    Manifest rows contain source_binding and target_record_sha256 from the
    independently frozen export, not values selected from the current record.
    This verifies provenance, not the scientific validity of exported labels.
    """
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets
    n=len(prediction.anchor_position_m)
    if any(len(x)!=n for x in (produced_targets,manifest_rows,bundles,grids)):
        raise ValueError('one positive record and frozen manifest row per observation required')
    records=[];bindings=[]
    for produced,manifest in zip(produced_targets,manifest_rows,strict=True):
        if set(manifest)!={'source_binding','target_record_sha256'}:
            raise ValueError('closed frozen positive manifest row required')
        if (produced.get('source_binding')!=manifest['source_binding']
            or produced.get('target_record_sha256')!=manifest['target_record_sha256']
            or canonical_sha(produced['record'])!=manifest['target_record_sha256']):
            raise ValueError('positive record or source differs from frozen manifest')
        if produced['record']['source_frame_indices']!=manifest['source_binding']['source']['frame_rows']:
            raise ValueError('positive and negative source frames differ')
        records.append(produced['record']);bindings.append(manifest['source_binding'])
    targets=observed_targets(records,device=prediction.anchor_position_m.device)
    return partial_surface_losses(prediction,targets,bundles=bundles,grids=grids,expected_bindings=bindings,
                                  opening_matching_radius_m=opening_matching_radius_m)


def partial_surface_losses(prediction, targets, *, bundles, grids, expected_bindings,
                           opening_matching_radius_m=None):
    """Query exact detached predicted centers; combine one presence denominator.

    Complete-region supervision is a distinct contract: reject mixing it with
    conditional reference negatives. A matched query is supervised positively
    even when its current position is far from the target (regression must be
    able to move it). Unknown unmatched queries receive no presence gradient.
    """
    _validate(prediction, targets)
    n=len(prediction.anchor_position_m)
    if any(len(x)!=n for x in (bundles,grids,expected_bindings)):
        raise ValueError('one frozen source context per batch observation required')
    if bool(targets.anchor_region_complete.any()):
        raise ValueError('do not mix complete-region and conditional reference contracts')
    if opening_matching_radius_m is not None and bool(targets.opening_region_complete.any()):
        raise ValueError('do not mix complete opening region and conditional reference contracts')
    base=surface_relation_losses(prediction,targets)
    numerator=base.terms['anchor_presence']*base.denominators['anchor_presence']
    count=base.denominators['anchor_presence']; evidence=[]
    for b in range(n):
        q=prediction.anchor_position_m[b].detach().cpu().double().numpy()
        result=bound_reference_exclusion(bundles[b],grids[b],q,
            expected_binding=expected_bindings[b],matching_radius_m=4.)
        negative=torch.tensor(result['reference_negative_mask'],dtype=torch.bool,
                              device=prediction.anchor_position_m.device)
        inside=torch.linalg.vector_norm(prediction.anchor_position_m[b].detach().double()
            -targets.score_region_center_m[b].double(),dim=-1)<=targets.score_region_radius_m[b].double()
        negative &= inside
        result['outside_declared_score_region_query_indices']=torch.nonzero(~inside).flatten().cpu().tolist()
        assigned=base.assignments['anchor'][b]
        negative[assigned[assigned>=0]]=False
        if not bool(prediction.observation_supported[b]):negative[:]=False
        logits=prediction.anchor_presence_logits[b,negative]
        if logits.numel():
            numerator=numerator+F.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits),reduction='sum')
            count+=logits.numel()
        result['used_unmatched_negative_query_indices']=torch.nonzero(negative).flatten().cpu().tolist()
        result['query_xyz_m']=q.tolist()
        evidence.append(result)
    terms=dict(base.terms);denominators=dict(base.denominators)
    terms['anchor_presence']=numerator/max(1,count)
    denominators['anchor_presence']=count
    if opening_matching_radius_m is not None:
        from mtare_topo.teacher.gse_opening_reference_exclusion_v1 import bound_opening_reference_exclusion
        count=base.denominators['opening_presence']
        numerator=base.terms['opening_presence']*count
        for b in range(n):
            q=prediction.opening_position_m[b].detach().cpu().double().numpy()
            result=bound_opening_reference_exclusion(bundles[b],grids[b],q,
                expected_binding=expected_bindings[b],matching_radius_m=opening_matching_radius_m)
            negative=torch.tensor(result['reference_negative_mask'],dtype=torch.bool,
                                  device=prediction.opening_position_m.device)
            inside=torch.linalg.vector_norm(prediction.opening_position_m[b].detach().double()
                -targets.score_region_center_m[b].double(),dim=-1)<=targets.score_region_radius_m[b].double()
            negative &= inside
            result['outside_declared_score_region_query_indices']=torch.nonzero(~inside).flatten().cpu().tolist()
            assigned=base.assignments['opening'][b]
            negative[assigned[assigned>=0]]=False
            if not bool(prediction.observation_supported[b]):negative[:]=False
            logits=prediction.opening_presence_logits[b,negative]
            if logits.numel():
                numerator=numerator+F.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits),reduction='sum')
                count+=logits.numel()
            result['used_unmatched_negative_query_indices']=torch.nonzero(negative).flatten().cpu().tolist()
            result['query_xyz_m']=q.tolist()
            evidence[b]['opening_exclusion']=result
        terms['opening_presence']=numerator/max(1,count)
        denominators['opening_presence']=count
    return replace(base,total=sum(terms.values()),terms=terms,denominators=denominators,
                   has_supervision=any(denominators.values())), evidence

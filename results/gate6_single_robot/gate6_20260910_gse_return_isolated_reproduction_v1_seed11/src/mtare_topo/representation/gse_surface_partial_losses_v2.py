"""Versioned confirmed-reference duplicate supervision; no label qualification."""
from dataclasses import replace
import torch
from torch.nn import functional as F
from .gse_surface_partial_losses_v1 import bound_partial_surface_losses as previous
from mtare_topo.teacher.gse_reference_query_coverage_v2 import (
    bound_anchor_query_coverage, bound_opening_query_coverage)


def bound_partial_surface_losses(prediction, *, produced_targets, manifest_rows, bundles, grids,
                                 opening_matching_radius_m=None):
    base, evidence = previous(prediction,produced_targets=produced_targets,
        manifest_rows=manifest_rows,bundles=bundles,grids=grids,
        opening_matching_radius_m=opening_matching_radius_m)
    terms=dict(base.terms); denominators=dict(base.denominators)
    kinds=[('anchor',4.,bound_anchor_query_coverage)]
    if opening_matching_radius_m is not None:
        kinds.append(('opening',opening_matching_radius_m,bound_opening_query_coverage))
    for kind,radius,coverage_function in kinds:
        term=kind+'_presence'; count=denominators[term]; numerator=terms[term]*count
        for b in range(len(bundles)):
            positions=getattr(prediction,kind+'_position_m')[b].detach().cpu().double().numpy()
            coverage=coverage_function(bundles[b],grids[b],positions,
                produced_targets=produced_targets[b],manifest_row=manifest_rows[b],matching_radius_m=radius)
            added=torch.tensor(coverage['confirmed_structure_coverage_mask'],dtype=torch.bool,
                device=prediction.anchor_position_m.device)
            assigned=base.assignments[kind][b]; added[assigned[assigned>=0]]=False
            prior=evidence[b] if kind=='anchor' else evidence[b]['opening_exclusion']
            added[prior['used_unmatched_negative_query_indices']]=False
            if not bool(prediction.observation_supported[b]): added[:]=False
            logits=getattr(prediction,kind+'_presence_logits')[b,added]
            if logits.numel():
                numerator=numerator+F.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits),reduction='sum')
                count+=logits.numel()
            coverage['added_duplicate_negative_query_indices']=torch.nonzero(added).flatten().cpu().tolist()
            prior['confirmed_structure_duplicate_evidence_v2']=coverage
        terms[term]=numerator/max(1,count);denominators[term]=count
    return replace(base,total=sum(terms.values()),terms=terms,denominators=denominators,
        has_supervision=any(denominators.values())),evidence

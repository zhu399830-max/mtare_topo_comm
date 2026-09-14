"""Loss-only view of existing partial structure references; never model input.

This does not create supervision. References locate physical structures and
10m-window sections, not globally unique places or physical aperture bounds.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PartialStructureTargets:
    anchor_positions_m: np.ndarray
    window_section_positions_m: np.ndarray
    relation_values: np.ndarray
    relation_known: np.ndarray
    source_frame_indices: tuple
    full_detection_precision_supported: bool = False
    unmatched_is_background: bool = False
    aperture_dimensions_supported: bool = False


def bind_partial_targets(record):
    if record.get('schema')!='gse_surface_observed_targets_v1' or record.get('coordinate_frame')!='current_sensor_m':
        raise ValueError('original versioned partial current-sensor reference required')
    region=record['score_region']
    if region['anchors_complete'] is not False or region['openings_complete'] is not False or region['radius_m']!=10 or region['center_m']!=[0.,0.,0.]:
        raise ValueError('this interface supports only original partial10m contract')
    frames=record['source_frame_indices']
    if len(frames)!=5 or any(type(f) is not int for f in frames) or frames!=sorted(set(frames)):
        raise ValueError('five distinct ordered original frames required')
    anchors=np.asarray([a['position_m'] for a in record['anchors']],dtype=np.float64).reshape(-1,3)
    openings=np.asarray([a['position_m'] for a in record['openings']],dtype=np.float64).reshape(-1,3)
    if not np.isfinite(anchors).all() or not np.isfinite(openings).all():raise ValueError('finite reference positions required')
    if any(a.get(k) is not None for a in record['openings'] for k in ('width_m','height_m')):
        raise ValueError('dimension supervision is a separate task contract')
    relations=record['membership']
    if len(relations)!=len(openings) or any(len(r)!=len(anchors) for r in relations):raise ValueError('relation shape drift')
    values=np.zeros((len(openings),len(anchors)),dtype=np.float32)
    known=np.zeros(values.shape,dtype=bool)
    for o,row in enumerate(relations):
        for a,label in enumerate(row):
            if label is None:continue
            if type(label) is not bool:raise ValueError('explicit bool or unknown required')
            known[o,a]=True;values[o,a]=float(label)
    for v in (anchors,openings,values,known):v.setflags(write=False)
    return PartialStructureTargets(anchors,openings,values,known,tuple(frames))


def known_relation_loss(reference_order_logits, targets):
    """Loss AFTER independent correspondence, not GT-proposed model queries.

    Unknown zero placeholders must be masked. Empty known sets return an exact
    differentiable zero. Normalization matches the existing known-pair mean.
    """
    import torch
    from torch.nn import functional as F
    if tuple(reference_order_logits.shape)!=targets.relation_known.shape:raise ValueError('reference-order relation shape required')
    mask=torch.as_tensor(targets.relation_known.copy(),device=reference_order_logits.device)
    selected=reference_order_logits[mask]
    if not len(selected):return selected.sum()
    if not torch.isfinite(selected).all():raise ValueError('nonfinite known relation logits')
    values=torch.as_tensor(targets.relation_values.copy(),device=reference_order_logits.device,dtype=reference_order_logits.dtype)
    return F.binary_cross_entropy_with_logits(selected,values[mask])

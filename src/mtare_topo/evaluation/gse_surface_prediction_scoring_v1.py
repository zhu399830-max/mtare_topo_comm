"""Shared A/B/C snapshot scoring; no calibration, label creation or training."""
from mtare_topo.representation.gse_surface_losses_v1 import _validate
from .gse_surface_detection_metrics_v1 import detection_counts


def score_surface_prediction(prediction, targets, *, anchor_threshold,
                             opening_threshold, opening_maximum_error_m):
    """Thresholds must come from the separately frozen calibration contract.

    Returns conditional counts even when full F1 is unavailable. Completeness
    is supplied by labels, never inferred from successful target matches.
    The 4m anchor main contract and 1m/2m sensitivities are always reported.
    """
    _validate(prediction,targets)
    def array(t):
        return t.detach().cpu().numpy()
    rows=[]
    for b in range(len(prediction.anchor_position_m)):
        def score(kind, threshold, radius):
            return detection_counts(
                predicted_xyz_m=array(getattr(prediction,kind+'_position_m')[b]),
                confidence=array(getattr(prediction,kind+'_presence_logits')[b].sigmoid()),
                target_xyz_m=array(getattr(targets,kind+'_position_m')[b][getattr(targets,kind+'_valid')[b]]),
                threshold=threshold, maximum_error_m=radius,
                score_center_m=array(targets.score_region_center_m[b]),
                score_radius_m=float(targets.score_region_radius_m[b]),
                region_complete=bool(getattr(targets,kind+'_region_complete')[b]))
        anchors={str(r):score('anchor',anchor_threshold,r) for r in (1.,2.,4.)}
        openings=score('opening',opening_threshold,opening_maximum_error_m)
        rows.append(dict(batch_index=b,anchor_main_4m=anchors['4.0'],
            anchor_sensitivity_1m=anchors['1.0'],anchor_sensitivity_2m=anchors['2.0'],
            openings=openings,observation_supported=bool(prediction.observation_supported[b]),
            full_detection_scoring_available=anchors['4.0']['full_scoring_available'] and openings['full_scoring_available']))
    return dict(observations=rows,thresholds=dict(anchor=anchor_threshold,opening=opening_threshold),
        opening_maximum_error_m=opening_maximum_error_m,
        label_completeness_verified=False,calibration_verified=False,scientific_gate_pass=False)

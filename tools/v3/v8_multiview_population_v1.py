"""Use original lossless exporter, with a strict version-specific validator."""
from mtare_topo.data.multiview_teacher_population_v1 import export_population as original_export
from mtare_topo.data.multiview_teacher_population_v1 import summarize_target as original_summary

VERSION='joint_partial_reference_v8_lateral_terminal_exclusion_source_precision'


def summarize_target(source,produced):
    settings=produced.get('geometry_evidence_settings',{})
    expected=dict(axial_spacing_m=.05,angular_segments=64,field_spacing_m=.025,
        intersections='original_float32_rays_complete_operand_surfaces',recast_first_returns=False,
        cap_precision_policy='source_float64_float32_interval_v1')
    if settings!=expected:raise ValueError('original V8 source geometry policy mismatch')
    return original_summary(source,produced,expected_version=VERSION)


def export_population(*args,**kwargs):
    return original_export(*args,**kwargs,target_summarizer=summarize_target)

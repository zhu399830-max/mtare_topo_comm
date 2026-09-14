"""Bind existing common inputs to unchanged partial references, metadata only."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
from mtare_topo.representation.gse_partial_structure_contract import bind_partial_targets

COMMON='results/gate3_semantics/gate3_20260911_gse_common_observation_export_v1_seed0'
REF='results/gate3_semantics/gate3_20260911_gse_membership_fit_v1r1_seed0'


def main():
    def index(run):
        raw=(ROOT/run/'artifacts/evidence_sha256.txt').read_bytes()
        return dict((p,h) for h,p in (line.split('  ',1) for line in raw.decode().splitlines()))
    common=index(COMMON);ref=index(REF)
    card_path=COMMON+'/config/data_card.json';raw=(ROOT/card_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=common[card_path]:raise ValueError('common card drift')
    entries=json.loads(raw)['scope']['binding']['entries'];rows=[]
    counts=dict(anchors=0,window_sections=0,positive_relations=0,negative_relations=0,unknown_relations=0,
                complete_background_windows=0,dimension_targets=0)
    for i,e in enumerate(entries):
        rp=f'artifacts/reference_{i:02d}.json';raw=(ROOT/REF/rp).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ref[rp]:raise ValueError('reference drift')
        target=json.loads(raw);record=target['record'];t=bind_partial_targets(record)
        if tuple(e['frame_rows'])!=t.source_frame_indices or target['source_binding']['source']['task']!=e['source']['task']:
            raise ValueError('input/reference binding mismatch')
        fp=COMMON+f'/artifacts/window_{i:02d}.npz'
        if hashlib.sha256((ROOT/fp).read_bytes()).hexdigest()!=common[fp]:raise ValueError('feature drift')
        pos=int(t.relation_values[t.relation_known].sum());known=int(t.relation_known.sum())
        counts['anchors']+=len(t.anchor_positions_m);counts['window_sections']+=len(t.window_section_positions_m)
        counts['positive_relations']+=pos;counts['negative_relations']+=known-pos;counts['unknown_relations']+=int((~t.relation_known).sum())
        rows.append(dict(observation=i,source=e['source'],feature_path=fp,feature_sha256=common[fp],
            reference_path=REF+'/'+rp,reference_sha256=ref[rp],known_positive=pos,known_negative=known-pos,
            unknown_relations=int((~t.relation_known).sum()),reference_status=target['supervision_status'],
            existing_missing_tasks=target['missing_tasks'],full_training_gate_eligible=target['full_training_gate_eligible']))
    result=dict(schema_version='gse_common_partial_structure_task_v1',status='INPUT_TARGET_BINDING_NOT_TRAINING_RUN',
        observations=16,independent_parents=8,split='fit_only',counts=counts,entries=rows,
        model_input=['full_sensor_context_and_validity','observed_local_ray_segments_with_measured_vs_crop_endpoint_flag','local_surface_patches'],
        supervision_only=['partial_physical_structure_positions','partial_window_section_positions','known_structure_section_relations'],
        unknown_policy='No negative from unmatched, missing dimensions, incomplete background, source ambiguity or missing local surface',
        comparison_contract=dict(common_cache=True,common_source_order=True,common_supervision=True,
            common_initialization_required=True,common_update_budget_required=True,
            observation_baseline='full context and observed ray representation',
            unary_reference='same observation representation plus surface attributes',
            relation_method='same unary representation plus explicit geometric relations'),
        unsupported_claims=['calibrated_full_detection_precision','physical_aperture_dimensions','physical_root_reachability','global_loop_identity','topology_advantage'],
        training_authorized_by_this_file=False,new_labels=0,teacher_in_forward=False,
        next='Implement and freeze a consumer of the common observation with explicit prediction/matching contract; do not resume closed heads or claim fit is independent validation')
    out=ROOT/'configs/v3/gate3/gse_common_partial_structure_task_v1.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(counts))


if __name__=='__main__':main()

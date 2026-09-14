"""Bind existing selection metadata and model hashes; no new row selection.

This is a metadata manifest, not permission to read sensor/teacher payloads.
An exact payload card/spec is still required before feature/target export.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
from ai_junction_pilot import sha,write

SOURCE='results/gate3_semantics/gate3_20260907_gse_surface_identity_selection_v1_seed20260906'
PARENTS=('S03_flat_unicyclic_small_C07','S04_3d_unicyclic_small_C07','S05_flat_branch_medium_C07','S06_3d_branch_medium_C07','S10_3d_complex_C07')
MODEL_INDEX='configs/v3/gate3/gse_conditional_frozen_models_v1.json'
FIT_CARD='configs/v3/gate3/data_cards/gse_new12_features_v1.json'
ENCODER_CARD='results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0/config/data_card.json'
OUT='configs/v3/gate3/gse_conditional_development_manifest_v1.json'


def main():
    seal=ROOT/SOURCE/'artifacts/evidence_sha256.txt'
    sealed={p:h for h,p in (line.split('  ',1) for line in seal.read_text().splitlines())}
    fit=json.loads((ROOT/FIT_CARD).read_text())['scope'];models=json.loads((ROOT/MODEL_INDEX).read_text())
    old=json.loads((ROOT/ENCODER_CARD).read_text())
    assert 'All C01-C06' in old['split']['fit'] and 'All C07 parents select' in old['split']['selection']
    fit_parents={e['parent'] for e in fit['entries']}
    assert not fit_parents.intersection(PARENTS)
    pins={str(seal.relative_to(ROOT)):sha(seal),FIT_CARD:sha(ROOT/FIT_CARD),MODEL_INDEX:sha(ROOT/MODEL_INDEX),ENCODER_CARD:sha(ROOT/ENCODER_CARD)}
    entries=[];parent_counts={};frames=set();edges=set();tasks=set()
    for parent in PARENTS:
        path=f'{SOURCE}/artifacts/{parent}_selection.json'
        assert sha(ROOT/path)==sealed[path];pins[path]=sealed[path]
        data=json.loads((ROOT/path).read_text())
        assert data['parent_id']==parent and data['split']=='development' and data['selected_physical_edges']==16
        rows=data['observations'];assert len(rows)==48
        grouped={}
        for row in rows:
            assert row['parent_id']==parent and row['split']=='development' and '_C07__' in row['task']
            assert len(row['frame_rows'])==5 and row['frame_rows']==sorted(set(row['frame_rows']))
            assert row['frame_rows'][-1]==row['decision_frame_row'] and not row['continuous_route_evidence']
            grouped.setdefault(row['physical_edge_id'],[]).append(row)
            frames.update((row['task'],i) for i in row['frame_rows']);edges.add(row['physical_edge_id']);tasks.add(row['task'])
            entries.append(dict(case=len(entries),source_metadata=path,**row))
        assert len(grouped)==16 and all({r['variant'] for r in v}=={'ellipse','rounded_rectangle','c1_mixed'} and len(v)==3 for v in grouped.values())
        parent_counts[parent]=dict(observations=len(rows),physical_edges=len(grouped))
    for variant in 'ABC':
        record=models[variant];assert sha(ROOT/record['path'])==record['sha256'];pins[record['path']]=record['sha256']
    assert len(entries)==240 and len(edges)==80 and len(tasks)==15
    result=dict(schema_version='conditional_development_metadata_manifest_v1',status='EXISTING_IDENTITIES_BOUND_NO_PAYLOAD_EXPORT',
        parent_ids=list(PARENTS),parents=5,tasks=15,physical_edges=80,observations=240,frame_exposures=1200,
        unique_variant_source_frames=len(frames),parent_counts=parent_counts,entries=entries,
        sampling='Reuse all original16-edge selections and three variants per development parent; no model-score selection or reranking',
        temporal_scope='five causal source frame indices per observation; no continuous route or measured clock inferred',
        actual_support_counts=dict(valid_returns=None,patches=None,known_relations=None,cross_component_pairs=None),
        split_audit=dict(conditional_head_fit_parent_overlap=[],encoder_gradient_parents='C01-C06',
            encoder_selection_parents='all C07; includes these five parents',strict_unseen=False,
            interpretation='parent-held-out conditional-head development evaluation, not unseen whole-system test',
            historical_scene_development=True,read_c08_c09_c10=False),
        frozen_models=models,encoder_checkpoint=fit['checkpoint'],input_sha256=pins,
        next='Bind exact cached or original sensor/source/pose arrays and chunks in a new export card/spec; no model or threshold adaptation',
        labels_generated=0,model_inference_frames=0,training_steps=0)
    write(ROOT/OUT,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('entries','input_sha256','frozen_models','encoder_checkpoint')},indent=2))


if __name__=='__main__':main()

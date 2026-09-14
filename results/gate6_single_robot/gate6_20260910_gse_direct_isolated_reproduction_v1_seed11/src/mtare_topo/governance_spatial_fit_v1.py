"""One added SPATIAL diagnostic; reuse sealed PRIMITIVE without retraining."""
from pathlib import Path
import ast
import json
import zipfile
from mtare_topo.governance_grouping_supervision_v2 import POLICY as P_POLICY
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_spatial_fit_card_v1'
SLUG='gse_spatial_center_fit_v1'
PARENT='results/gate3_semantics/gate3_20260909_gse_grouping_center_supervision_v2_seed0'
PARENT_SEAL='a62d0798e3bfe2ee15d88d2f336ecc854dbd8d5416af4b97feb7175d14f5acba'
PRELIM='docs/figures/gse_graph/primitive_localization_ceiling_20260909.json'
POLICY=dict(P_POLICY,method='SPATIAL',parent_run=PARENT,parent_seal_sha256=PARENT_SEAL,
    question='On the same16 and corrected shared detector, is the localization/selection failure shared or grouping-dependent?',
    baseline='Reuse sealed PRIMITIVE1000 corrected-supervision run; exactly one SPATIAL1000 fit diagnostic, no main comparison claim.',
    authorization='20260909 latest user: keep initialization/supervision fixes and4m unknown protection; saved-checkpoint localization ceiling first; then exactly one same16 same-init same-supervision same1000 SPATIAL fit; reuse matching PRIMITIVE; no expansion/branches/maps.',
    extra_sources=['tests/v3/unit/test_grouping_localization_ceiling_v1.py',
        'tests/v3/unit/test_spatial_fit_routing_v1.py','docs/GSE_SPATIAL_FIT_DIAGNOSTIC_SPEC_20260909.md',PRELIM],
    mask_gradient_reaudit=False,primitive_new_updates=0)


def common_implementation(root,parent_spec):
    common={p:h for p,h in parent_spec['source_sha256'].items() if p.startswith((
        'src/mtare_topo/representation/','src/mtare_topo/teacher/','src/mtare_topo/data/'))
        or p in ('src/mtare_topo/evaluation/grouping_center_scoring_v1.py',
            'tools/v3/bidirectional_paired_reader_v1.py','tools/v3/bidirectional_paired_scope_v1.py','tools/v3/grouping_fit_scope_v1.py')}
    for p,h in common.items():read_pinned(root,p,h)
    # Compare the real training loop, not just matching policy strings. The
    # sole executable loop difference is the selected grouping argument.
    with zipfile.ZipFile(root/PARENT/'artifacts/source_snapshot.zip') as archive:
        old=ast.parse(archive.read('tools/v3/grouping_center_fit_v1.py'))
    new=ast.parse((root/'tools/v3/grouping_center_fit_v1.py').read_text())
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,node):return ast.copy_location(ast.Constant('PRIMITIVE'),node) if node.id=='GROUPING' else node
    def loop(tree):
        return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
            and [getattr(e,'id',None) for e in n.target.elts]==['step','batch'])
    if ast.dump(loop(old))!=ast.dump(Normalize().visit(loop(new))):raise ValueError('training loop changed beyond grouping')
    def named(tree,name):return next(n for n in tree.body if getattr(n,'name',None)==name)
    if ast.dump(named(old,'SelectedReader'))!=ast.dump(named(new,'SelectedReader')):raise ValueError('reader changed')
    return dict(source_sha256=common,training_loop_only_grouping_changed=True,reader_ast_equal=True)


def compile_scope(root):
    from surface_features_v1 import sha
    from grouping_fit_scope_v1 import compile_scope as original
    root=Path(root);scope=original(root)
    read_pinned(root,PARENT+'/artifacts/evidence_sha256.txt',PARENT_SEAL)
    index=dict((p,h) for h,p in (s.split('  ',1) for s in (root/PARENT/'artifacts/evidence_sha256.txt').read_text().splitlines()))
    def pinned(rel):return json.loads(read_pinned(root,PARENT+'/'+rel,index[PARENT+'/'+rel]))
    parent_card=pinned('config/data_card.json');parent_spec=pinned('config/run_spec.json')
    for key in ('selection','selected_rows','counts','spacing','split'):
        if parent_card['scope'][key]!=scope[key]:raise ValueError('fixed population drift')
    for key in ('seed','updates','evaluate_every','microbatch','accumulation','optimizer','learning_rate',
        'weight_decay','dropout','augmentation','stage','relation_attributes','frozen_context',
        'positive_negative','loss','threshold','main_radius_m','auxiliary_radii_m','initialization',
        'background_radius_m','duplicate_radius_m'):
        if POLICY[key]!=parent_card['policy'][key]:raise ValueError('paired configuration mismatch: '+key)
    prelim=json.loads((root/PRELIM).read_text())
    if prelim['parent_seal_sha256']!=PARENT_SEAL or len(prelim['steps'])!=11:raise ValueError('missing prerequisite ceiling')
    scope.update(parent_run=dict(path=PARENT,seal_sha256=PARENT_SEAL),
        parent_environment=parent_spec['environment'],
        prerequisite=dict(path=PRELIM,sha256=sha(root/PRELIM)),
        common_implementation=common_implementation(root,parent_spec),
        parent_evidence_sha256={PARENT+'/'+rel:index[PARENT+'/'+rel] for rel in
            ('config/data_card.json','config/run_spec.json','config/schedule.json','checkpoints/step_0000.pt')})
    return scope


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'training'):errors.append('wrong operation')
        if card['scope']!=compile_scope(Path(__file__).resolve().parents[2]):errors.append('scope/source drift')
        if card['scope_sha256']!=digest(card['scope']) or card['policy']!=POLICY:errors.append('policy drift')
        a=card['approval']
        if (a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256']
            or a.get('authorized_operations')!=['training'] or a.get('authorized_gates')!=[3]
            or a.get('confirmation_reference')!=POLICY['authorization']):errors.append('exact new diagnostic authority required')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))

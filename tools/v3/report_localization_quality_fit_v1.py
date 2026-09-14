"""Non-overwriting figures/reduction of completed sealed localization-quality fit.

Only saved predictions, source evidence and original input arrays are read;
this script does not import or execute the model or any training entrypoint.
"""
from _bootstrap import PROJECT_ROOT as ROOT
from collections import Counter
import csv
import io
import json
from pathlib import Path

import numpy as np

from mtare_topo.governance_surface_selection import digest
from report_geometry_presence_a_v1 import load, sha, verify


RUN = ROOT / 'results/gate3_semantics/gate3_20260910_gse_localization_quality_fit_v1_seed0'
PARENT = ROOT / 'results/gate3_semantics/gate3_20260910_gse_candidate_center_verifier_v1_seed0'
SCORE = ROOT / 'results/gate3_semantics/gate3_20260910_gse_fixed_candidate_scoring_v1_seed0'
OUT = ROOT / 'docs/figures/gse_graph/localization_quality_20260910'


def main():
    if OUT.exists():
        raise FileExistsError('derivative evidence must not be overwritten')
    if load(RUN / 'RUN_STATE.json')['state'] != 'COMPLETED':
        raise ValueError('read only a completed experiment; no report of partial results')
    seal = verify(RUN)
    reads = {}
    indexes = {}
    for run in (PARENT, SCORE):
        indexes[run] = {p: h for h, p in
            (line.split('  ', 1) for line in (run / 'artifacts/evidence_sha256.txt').read_text().splitlines())}

    def pinned(run, relative):
        path = run / relative
        key = str(path.relative_to(ROOT))
        expected = indexes[run][key]
        if sha(path) != expected:
            raise ValueError('prior sealed input drift: ' + key)
        reads[key] = expected
        return path.read_bytes()

    final = load(RUN / 'metrics/evaluation_1000.json')
    parent = json.loads(pinned(PARENT, 'metrics/evaluation_1000.json'))
    labels = load(RUN / 'metrics/label_inventory.json')
    reference = json.loads(pinned(SCORE, 'metrics/evaluation_0000.json'))
    old_labels = json.loads(pinned(PARENT, 'metrics/validity_inventory.json'))
    followup = load(RUN / 'metrics/failure_followup.json')
    curve = load(RUN / 'metrics/curve.json')
    summary = load(RUN / 'metrics/summary.json')
    logs = [json.loads(line) for line in (RUN / 'logs/updates.jsonl').read_text().splitlines()]
    if len(final['observations']) != 16 or len(logs) != 1000 or [r['step'] for r in curve] != list(range(0, 1001, 100)):
        raise ValueError('unexpected completed population or training budget')
    OUT.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

    def save(fig, path):
        for obj in fig.findobj(Text):
            obj.set_fontproperties(FontProperties(fname=font, size=obj.get_fontsize()))
        fig.tight_layout()
        fig.savefig(path, dpi=125)
        plt.close(fig)

    def categories(observation, ref, region, stage):
        score = observation['scores'][region][stage]
        slots = score['original_query_slots']
        matched = {slots[j] for j, _ in score['pairs']}
        cov = ref['scores']['1.0' if region == 'old' else '4.0']['coverage']['query_scoreable_mask']
        result = {q: 'tp' if q in matched else 'fp' if cov[q] else 'unknown' for q in slots}
        counts = Counter(result.values())
        if (counts['tp'], counts['fp'], counts['unknown']) != (score['tp'], score['fp'], score['ignored']):
            raise ValueError('saved score/category mismatch')
        return result

    cases = []
    by_source = {}
    totals = {region: Counter() for region in ('old', 'fixed')}
    all_errors = []
    for index, (current, previous, label, old_label, ref) in enumerate(zip(
            final['observations'], parent['observations'], labels['observations'],
            old_labels['observations'], reference['observations']), 1):
        source = current['source']
        if not (source == previous['source'] == label['source'] == ref['source']):
            raise ValueError('observation order/source mismatch')
        key = digest(source)
        with np.load(io.BytesIO(pinned(PARENT, f'artifacts/input_{key}.npz'))) as data:
            xyz = data['xyz_m'].copy()
            positions = data['position_m'].copy()
            targets = data['target_positions_m'].copy()
        with np.load(RUN / f'artifacts/prediction_1000_{key}.npz') as prediction:
            if not np.array_equal(positions, prediction['position_m']) or not np.array_equal(
                    np.asarray(current['logits'], dtype=prediction['logits'].dtype), prediction['logits']):
                raise ValueError('saved final prediction differs from evaluation')
        by_source[key] = dict(index=index, current=current, previous=previous, label=label)
        details = {}
        for region in ('old', 'fixed'):
            before = categories(previous, ref, region, 'after')
            after = categories(current, ref, region, 'after')
            prior_fp = {q for q, value in before.items() if value == 'fp'}
            now_fp = {q for q, value in after.items() if value == 'fp'}
            details[region] = dict(previous_fp=sorted(prior_fp), remaining_fp=sorted(prior_fp & now_fp),
                resolved_fp=sorted(prior_fp - now_fp), new_fp=sorted(now_fp - prior_fp),
                final= current['scores'][region]['after'])
            totals[region].update({name: len(details[region][name]) for name in
                ('previous_fp', 'remaining_fp', 'resolved_fp', 'new_fp')})
            for q in sorted(now_fp):
                item = label['candidates'][q]
                all_errors.append(dict(observation=index, source=source, query_slot=q, region=region,
                    label=item['label'], localization_basis=item['localization_basis'],
                    support_type=item['support_type'], original_background_negative=item['original_background_negative'],
                    logit=current['logits'][q], position_m=positions[q].tolist(),
                    nearest_confirmed_center_m=float(np.linalg.norm(targets-positions[q], axis=1).min()) if len(targets) else None,
                    was_fp=q in prior_fp, old_training_label=old_label['candidates'][q]['new_label']))
        fig, axes = plt.subplots(2, 4, figsize=(17, 9))
        for column, (name, observation, stage) in enumerate([
            ('第17节：旧监督', previous, 'before'), ('第17节：旧监督', previous, 'after'),
            ('本次：定位质量监督', current, 'before'), ('本次：定位质量监督', current, 'after')]):
            score = observation['scores']['old'][stage]
            fixed = observation['scores']['fixed'][stage]
            identities = categories(observation, ref, 'old', stage)
            for row, (a, b) in enumerate(((0, 1), (0, 2))):
                ax = axes[row, column]
                ax.scatter(xyz[:, a], xyz[:, b], s=.2, c='#aaaaaa', alpha=.25, rasterized=True)
                ax.scatter(positions[:, a], positions[:, b], s=8, c='#cccccc')
                for q, identity in identities.items():
                    color = {'tp': '#16823b', 'fp': '#d62728', 'unknown': '#ee9b00'}[identity]
                    ax.scatter(positions[q, a], positions[q, b], marker='x', s=60, c=color)
                    ax.annotate(str(q), (positions[q, a], positions[q, b]), fontsize=7)
                if len(targets):
                    ax.scatter(targets[:, a], targets[:, b], s=65, facecolors='none', edgecolors='black', marker='D')
                ax.set(xlim=(-10.5, 10.5), ylim=(-10.5, 10.5), aspect='equal', xlabel='X / m',
                    ylabel=('Y' if b == 1 else 'Z') + ' / m',
                    title=name + (' 去重前' if stage == 'before' else ' 去重后') +
                    f"\n旧评价：正确{score['tp']} 误检{score['fp']} 漏检{score['fn']} 未知{score['ignored']}" +
                    f"\n固定区域：误检{fixed['fp']} 未知{fixed['ignored']}")
                ax.grid(alpha=.15)
        fig.suptitle(f"{index:02d} {source['task']} / 序列{source['source_sequence_id']}\n" +
            '绿叉=正确中心；红叉=旧评价误检；橙叉=旧评价未知；黑菱形=参考。数字为固定查询槽，不是图节点编号。', fontsize=11)
        image = OUT / f'case_{index:02d}.png'
        save(fig, image)
        cases.append(dict(index=index, source=source, label_counts=label['counts'], error_changes=details,
            image=str(image.relative_to(ROOT))))

    enriched = []
    group_totals = {region: {} for region in ('old', 'fixed')}
    for row in followup:
        info = by_source[digest(row['source'])]
        evidence = info['label']['candidates'][row['query_slot']]
        item = dict(row, observation=info['index'], new_label=evidence['label'],
            localization_basis=evidence['localization_basis'], support_type=evidence['support_type'],
            final_selected=row['query_slot'] in info['current']['selection']['after'])
        enriched.append(item)
        if row['was_fp']:
            group = ('already_supervised_background' if row['old_label'] == 'negative'
                else 'new_observed_localization' if evidence['support_type'] == 'candidate_observed'
                else 'new_context_localization' if evidence['support_type'] == 'confirmed_structure_context'
                else 'still_unknown')
            counter = group_totals[row['region']].setdefault(group, Counter())
            counter['previous_fp'] += 1
            counter['remaining_fp' if row['is_fp'] else 'resolved_fp'] += 1
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    steps = [row['step'] for row in curve]
    axes[0].plot([r['update'] for r in logs], [r['loss'] for r in logs])
    axes[0].set_title('实际训练损失（不是通过标准）')
    for region, text, color in [('old', '旧评价', '#0072b2'), ('fixed', '固定区域', '#d55e00')]:
        axes[1].plot(steps, [r['summary'][region]['after']['fp'] for r in curve], '-o', c=color, label=text+'误检')
        axes[1].axhline(parent['summary'][region]['after']['fp'], c=color, ls=':', label='第17节末轮'+text+'误检')
        axes[2].plot(steps, [r['summary'][region]['after']['precision'] for r in curve], '-o', c=color, label=text+'精确率')
    axes[1].plot(steps, [r['summary']['old']['after']['tp'] for r in curve], '--', c='#16823b', label='正确检出 / 共12')
    axes[1].set_title('误检下降，但仍未通过')
    axes[2].plot(steps, [r['summary']['old']['after']['recall'] for r in curve], '--', c='#16823b', label='召回率')
    axes[2].axhline(.9, c='gray', ls=':', label='预定门槛0.90')
    axes[2].set(title='最终检查点判定，不选择最好轮数', ylim=(-.05, 1.05))
    for ax in axes:
        ax.set_xlabel('实际更新数')
        ax.grid(alpha=.2)
    for ax in axes[1:]:
        ax.legend(prop=FontProperties(fname=font, size=8))
    fig.suptitle('同16观察、512冻结候选、同21185参数及step0：仅改变定位监督；不代表几何方法增益')
    save(fig, OUT / 'learning_curve.png')
    with (OUT / 'unified_results.csv').open('x') as file:
        writer = csv.DictWriter(file, fieldnames=['method', 'step', 'region', 'stage', 'tp', 'fp', 'fn',
            'ignored', 'output_count', 'precision', 'recall', 'f1'])
        writer.writeheader()
        for name, records in [('section17_final', [dict(step=1000, summary=parent['summary'])]), ('localization_quality', curve)]:
            for row in records:
                for region in ('old', 'fixed'):
                    for stage in ('before', 'after'):
                        writer.writerow(dict(method=name, step=row['step'], region=region, stage=stage,
                            **row['summary'][region][stage]))
    result = dict(run=str(RUN.relative_to(ROOT)), seal=seal, training_rerun=False, model_inference=False,
        parent_final=parent['summary'], summary=summary, cases=cases,
        totals={r: dict(c) for r, c in totals.items()},
        original_error_groups={r: {k: dict(c) for k, c in g.items()} for r, g in group_totals.items()},
        current_errors=all_errors, failure_followup=enriched,
        interpretation='Controlled same-model supervision change; not geometry/grouping superiority or generalization.')
    (OUT / 'error_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'failure_followup.json').write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + '\n')
    manifest = {str(path.relative_to(ROOT)): sha(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    (OUT / 'sha256.json').write_text(json.dumps(dict(run_seal=seal, tool_sha256=sha(Path(__file__)),
        prior_source_reads=reads, files=manifest), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(seal=seal, cases=len(cases), totals=result['totals'],
        original_error_groups=result['original_error_groups'], output=str(OUT)), ensure_ascii=False))


if __name__ == '__main__':
    main()

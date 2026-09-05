# Codex Project Operating Contract

Before any implementation, experiment, cleanup, training, or closed-loop run,
read these files in order:

1. `docs/PLAN.md` (current authoritative execution policy)
2. `docs/PROGRESS.md` (current phase and the only allowed next step)
3. `docs/MASTER_PLAN_V3.md` (retained research/governance detail)
4. `docs/SUPERVISED_STRUCTURE_LEARNING_V1.md`
5. `docs/PROJECT_STATUS.md`
6. `docs/DECISION_LOG.md`
7. `docs/REPOSITORY_STRUCTURE.md`
8. `docs/EXPERIMENT_PROTOCOL.md`
9. `results/project_status.json`

If an older document conflicts with `docs/PLAN.md`, stop and follow
`docs/PLAN.md`; record the superseded rule in `docs/DECISION_LOG.md`. Never
silently blend conflicting routes. The current implementation scope ends at
Phase 3 until the user explicitly changes it.

Before acting, report to the user:

- current Phase/Gate and its single research question;
- the exact data and sample counts to be used;
- the main method, baseline, and necessary fallback;
- the files/results that will be created or changed;
- the pass/fail evidence expected from the work.

While acting, report material problems immediately. Do not silently change the
dataset, split, teacher, metric, benchmark, method, or acceptance threshold.

If any assumption, data defect, leakage risk, insufficient sample diversity,
teacher mismatch, invalid baseline, unstable metric, or implementation blocker
is discovered, stop the affected work before adapting around it. Report:

1. the concrete evidence;
2. which conclusion or Gate it invalidates;
3. whether the issue is data, teacher, model, metric, or system related;
4. the viable options and their costs;
5. the recommended option;
6. what explicit user decision is needed.

Do not silently repair the issue, substitute data, lower an acceptance gate,
or start a different experiment.

Before any annotation pilot, AI-assisted labeling, data export, teacher
generation, self-supervised learning, or training, provide a data card containing exact
worlds, trajectories, independent sampling units, raw/effective sample counts,
spatial/temporal spacing, split logic, teacher source, and leakage audit. Formal
benchmark/test worlds are forbidden from supervised training, SSL, teacher or
threshold calibration, normalization statistics, augmentation tuning, and
checkpoint selection. The user must approve the data card before training.

Every experiment must live under the matching `results/gate*/` directory in a
descriptive run directory and preserve config, command, seed, data manifest,
environment, raw log, metrics, summary, and only scientifically relevant
previews. Model runs also preserve checkpoints. Graph/closed-loop runs also
preserve trajectory, nodes, edges, exit states, decision trace, runtime,
coverage-time curve, and failure reasons.

Before creating a material run, freeze a JSON run spec under the matching
`configs/v3/gateN/`, run `python3 tools/v3/preflight.py --spec <spec>`, and
show the user the proposed data/method/cost/evidence. Use
`python3 tools/v3/create_run.py --spec <spec>` only after preflight and the
required user approval. `create_run.py` creates evidence directories but never
executes the experiment command. Data-dependent operations must reference a
user-approved data card whose machine-readable approval is bound to the exact
operation and Gate.

At the end of each material task, update `docs/PROJECT_STATUS.md`,
`docs/DECISION_LOG.md` when a decision was made, and
`results/project_status.json`. A Gate ends only with `GATE_PASS`, `GATE_MIXED`,
or `GATE_FAIL`. Never advance automatically to the next Gate. Test data may not
be used for development or selection.

Historical artifacts outside the V3 Gate directories are reusable evidence or
baselines only; they do not imply that any V3 Gate has passed.

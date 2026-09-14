# Local Disk Cleanup

# Scope

Free local Ubuntu root filesystem space without deleting source files or
experiment results.

# Files Planned For Change

- `todo.md`

# Data Planned For Move

- `results/` to `/mnt/nas_znfy/Workspace/mtare_topo_comm_results_moved_20260806/`

# Acceptance Test

1. `df -h /` reports increased available space.
2. `/home/ubuntu/mtare_topo_comm/results` remains accessible.
3. Existing experiment result files are preserved on NAS.

# Rollback Method

Remove the `results` symlink, move the NAS directory back to
`/home/ubuntu/mtare_topo_comm/results`, and restore this `todo.md` entry from
the prior file version if needed.

# Tasks

- [x] Inspect current repository disk usage.
- [x] Inspect current Git status, noting `.git` is unavailable in this runtime.
- [ ] Move `results/` to NAS-backed storage. Stopped after user redirected to
  partition expansion; local `results/` remains unchanged.
- [ ] Recreate `results` as a symlink to the NAS copy. Not performed.
- [ ] Remove local `.pytest_cache`.
- [ ] Verify disk space and result path access. Not performed.

---

# Current Phase

Local structural map alignment validation for offline subterranean data and
M-TARE online data.

# Goal

Verify that SubT-MRS/LAMP offline data and M-TARE data can both produce the
same `LocalStructuralMap` tensor through one shared `LocalStructuralMapBuilder`,
then prove mixed-source tensors can enter one small CNN batch and complete
forward/loss/backward/optimizer step.

# Allowed Changes

- `todo.md`
- `learning/local_structural_map/**`
- focused tests for the local structural map contract
- run outputs under `results/local_map_alignment/<run_id>/`

# Forbidden Changes

- M-TARE planner behavior, frontier, viewpoint, Ghost, topology, global
  planning, communication, multi-robot allocation or target reinjection
- NAS source data writes or deletes
- long-lived `robot0`/`robot1` container deletion
- formal large-scale model training

# Files Planned For Change

- `todo.md`
- `learning/local_structural_map/config.py`
- `learning/local_structural_map/schema.py`
- `learning/local_structural_map/geometry.py`
- `learning/local_structural_map/raycast.py`
- `learning/local_structural_map/builder.py`
- `learning/local_structural_map/datasets/common.py`
- `learning/local_structural_map/datasets/lamp.py`
- `learning/local_structural_map/datasets/subt_mrs.py`
- `learning/local_structural_map/datasets/mtare_bag.py`
- `learning/local_structural_map/runtime/mtare_runtime.py`
- `learning/local_structural_map/tools/inspect_sources.py`
- `learning/local_structural_map/tools/export_samples.py`
- `learning/local_structural_map/tools/visualize_samples.py`
- `learning/local_structural_map/tools/compare_sources.py`
- `learning/local_structural_map/smoke_train.py`
- `learning/local_structural_map/tests/test_contract.py`
- `learning/local_structural_map/tests/test_raycast.py`
- `learning/local_structural_map/tests/test_coordinate_transform.py`
- `learning/local_structural_map/tests/test_replay_parity.py`

# Acceptance Test

1. Audit SubT-MRS UGV2 bags across multiple segments for LiDAR, pose/odom/path,
   tf/tf_static, extrinsics and timing; fall back to LAMP tunnel only with
   recorded evidence if SubT-MRS lacks reliable pose.
2. Confirm M-TARE bags or record a short tunnel run containing
   `/registered_scan` and `/state_estimation_at_scan`.
3. Export at least 10 valid offline subterranean samples and 10 valid M-TARE
   samples with identical shape, dtype, channel order, coordinates and missing
   value rules.
4. Generate same-code per-channel previews and source statistics.
5. Compare M-TARE bag adapter and runtime adapter replay outputs on the same
   messages, reporting per-channel max/mean absolute error and mask mismatch.
6. Run a mixed-source lightweight CNN for 20-100 steps and record batch shape,
   loss, NaN/Inf check, gradients and parameter update.

# Rollback Method

Restore `todo.md`, remove `learning/local_structural_map/`, remove only the
current `results/local_map_alignment/<run_id>/` directory, and leave all NAS
data, existing raw experiment results, containers and upstream M-TARE sources
unchanged.

# Tasks

- [x] Inspect repository state and available data/tooling
- [x] Audit SubT-MRS UGV2 topics, timing, tf and extrinsics across segments
- [x] Audit LAMP tunnel topics and ground-truth odometry
- [x] Audit or record M-TARE `/registered_scan` and `/state_estimation_at_scan`
- [x] Implement shared local structural map contract and builder
- [x] Implement dataset/runtime adapters that only emit standard frames
- [x] Export aligned samples and previews
- [x] Run contract, raycast, coordinate and replay parity tests
- [x] Run mixed-source CNN smoke train
- [x] Save run artifacts and final runtime logs

# Current Result

Completed with `READY_WITH_FIXES`: LAMP and M-TARE both generate the same
`[8, 100, 100]` float32 local structural map through one shared builder, M-TARE
bag/runtime replay parity is exact on 20 samples, and mixed-source smoke train
completed. SubT-MRS UGV2 remains blocked for direct use by the confirmed absence
of PointCloud2 plus global pose/path/tf in audited raw segments.

---

# Topological Semantic Model Feasibility Training

# Scope

Train a small causal online topological semantic model on
`results/topological_semantic_dataset_v3_traversability_fixed`, comparing a
current-only baseline against a history-aware model that uses causal history
and relative poses.

# Forbidden Changes

- topology node generation or threshold tuning
- M-TARE online planner, communication, GridWorld or allocation integration
- teacher label redesign or dataset split changes
- use of forest, campus or indoor samples for training
- use of world ID, trajectory ID, absolute pose or teacher data as model input

# Files Planned For Change

- `todo.md`
- `configs/learning/topological_semantic_model_small.yaml`
- `learning/structural_learning/tools/train_topological_semantic_model.py`

# Acceptance Test

1. Audit train/val/test sample counts, shapes, masks, relative poses, direction
   convention metadata and NaN/Inf status.
2. Save at least 24 random training previews showing current, history and
   teacher direction/distance/score summaries.
3. Overfit 24-32 fixed train samples with the history-aware model and verify
   loss drops, task losses drop, parameters update and no NaN/Inf appears.
4. Train current-only and history-aware models using only train samples, select
   checkpoints by forest validation metrics, and evaluate campus/indoor only
   after checkpoint selection.
5. Report direction, distance, exit, structural score, role retrieval,
   counterexample, history dependency and nuisance-factor metrics.
6. Save checkpoints, logs, metrics, previews and summary under
   `results/topological_semantic_model/<run_id>/`.

# Rollback Method

Remove `configs/learning/topological_semantic_model_small.yaml`,
`learning/structural_learning/tools/train_topological_semantic_model.py`, and
only the current `results/topological_semantic_model/<run_id>/` directory.

# Current Result

Completed with `TOPOLOGICAL_SEMANTIC_MODEL_FAIL` on GPU workstation run
`results/topological_semantic_model/20260806_toposem_v3_small_fix1`: the
history-aware model overfits a 32-sample subset and learns direction
reachability, but independent-exit prediction is mostly invalid, history and
relative-pose dependency are weak, and the learned role remains more correlated
with surface coverage than teacher topology.

---

# Offline M-TARE Structural Topology Node Validation

# Scope

Validate whether the frozen 128D forced bottleneck encoder plus frozen
128-128-64 projection head can drive causal offline topology node triggering
on the already recorded unseen M-TARE `campus` and `indoor` trajectories.

# Forbidden Changes

- model retraining, fine-tuning or checkpoint modification
- online M-TARE planning, GridWorld, target value, multi-robot communication or
  topology graph integration
- dataset regeneration or test world editing
- use of teacher surfaces in online node triggering

# Files Planned For Change

- `todo.md`
- `configs/learning/offline_topology_node_validation.yaml`
- `learning/structural_learning/tools/run_offline_topology_node_validation.py`

# Acceptance Test

1. Load the existing `surface_evidence_v1` M-TARE transfer samples for
   `campus` and `indoor`.
2. Load frozen encoder/projection checkpoints and confirm their SHA256 hashes.
3. Generate causal node lists for fixed-distance, online-input-geometry and
   learned-64D-representation methods.
4. Build offline teacher-only structural change references and exclude the
   fixed opening calibration segment from final metrics.
5. Report per-world node count, compression, stable-segment redundancy,
   1/2/3m change recall, precision/F1, node error, robustness and graph
   connectivity.
6. Save configs, threshold provenance, node/edge lists, metrics, previews,
   stdout and stderr under `results/offline_topology_node_validation/<run_id>/`.

# Rollback Method

Remove `configs/learning/offline_topology_node_validation.yaml`,
`learning/structural_learning/tools/run_offline_topology_node_validation.py`,
and only the current
`results/offline_topology_node_validation/<run_id>/` directory. Leave frozen
checkpoints, transfer inputs, recordings and upstream M-TARE sources unchanged.

---

# Structural Dataset v4 Multi-Environment

# Scope

Audit available real SLAM datasets and build a `surface_evidence_v1`
multi-environment structural learning dataset using the shared
`StandardFrame -> SurfaceEvidenceBuilder` path. The work is data-only.

# Forbidden Changes

- encoder/projection training, fine-tuning or checkpoint modification
- topology node experiments, M-TARE online planner integration, GridWorld
  changes, target value, communication or allocation changes
- use of teacher surfaces as model input
- use of false single-origin free/unknown raycasting or fake traversability
  labels
- use of known-unreliable SubT-MRS UGV2 as formal training data

# Files Planned For Change

- `todo.md`
- `configs/learning/structural_dataset_v4_multienv.yaml`
- `learning/structural_learning/tools/export_structural_dataset_v4_multienv.py`
- `learning/structural_learning/tools/check_structural_dataset_v4.py`

# Acceptance Test

1. Save `data_source_audit.json` covering CERBERUS, LAMP and other inspected
   NAS candidates.
2. Decode CERBERUS `/lidar/packets` into local LiDAR points, synchronize with
   GT TUM poses, apply confirmed `imu_link_to_lidar` extrinsics, and emit
   `StandardFrame`.
3. Generate at least 100 diagnostic samples per adopted source and verify
   input/teacher alignment, density/height distributions and source metrics.
4. Export `results/structural_dataset_v4_multienv/` with train/val/test/buffer
   split by environment or full trajectory, not random samples.
5. Confirm teacher source scans do not cross split boundaries.
6. Validate NumPy and, where available, PyTorch batch loading with mixed data
   sources and no NaN/Inf.

# Rollback Method

Remove `configs/learning/structural_dataset_v4_multienv.yaml`,
`learning/structural_learning/tools/export_structural_dataset_v4_multienv.py`,
`learning/structural_learning/tools/check_structural_dataset_v4.py`, and only
the current `results/structural_dataset_v4_multienv/` directory. Leave NAS
source data, existing v3/v4 diagnostics, frozen checkpoints and M-TARE upstream
sources unchanged.

---

# Structural Multi-Environment Training

# Scope

Retrain the existing forced 128D bottleneck completion model and the existing
128-128-64 projection head on `structural_dataset_v4_multienv` train only
(`tunnel + urban`), select checkpoints only by train/perlin2 validation, and
compare against the frozen single-tunnel model on perlin2, ku, campus and
indoor.

# Forbidden Changes

- model architecture changes, model search or additional model variants
- use of perlin2, ku, M-TARE campus or M-TARE indoor for training, PCA,
  normalization, thresholds or checkpoint selection
- topology node generation, node threshold tuning, M-TARE planner integration,
  communication, allocation or GridWorld changes
- dataset re-splitting or sample movement across train/val/test

# Files Planned For Change

- `todo.md`
- `configs/learning/structural_multienv_training.yaml`
- `learning/structural_learning/tools/run_structural_multienv_training.py`

# Acceptance Test

1. Run small-sample overfit on train samples and record initial/final loss.
2. Train encoder/decoder on v4 train only and select best checkpoint by
   perlin2 validation loss.
3. Freeze encoder, train projection head using train-only teacher geometry PCA.
4. Reload saved encoder/projection checkpoints and verify outputs match.
5. Evaluate old and new models with the same code on perlin2, ku, M-TARE
   campus and M-TARE indoor.
6. Save completion, representation, nuisance, M-TARE comparison, previews,
   stdout/stderr and summary under
   `results/structural_multienv_training/<run_id>/`.

# Rollback Method

Remove `configs/learning/structural_multienv_training.yaml`,
`learning/structural_learning/tools/run_structural_multienv_training.py`, and
only the current `results/structural_multienv_training/<run_id>/` directory.
Leave v4 dataset, old checkpoints and M-TARE recordings/results unchanged.

---

# Offline Topology Node Experiment With Multi-Environment Model

# Scope

Evaluate whether the frozen multi-environment 64D structural representation
improves offline topology node matching, merging, graph compactness and
robustness on frozen perlin2 calibration plus ku/campus/indoor test data.

# Forbidden Changes

- encoder or projection retraining, fine-tuning or checkpoint modification
- use of ku, campus or indoor for threshold tuning or model selection
- M-TARE online planner, GridWorld, target value, communication or allocation
  changes
- topology node threshold changes after viewing test metrics

# Files Planned For Change

- `todo.md`
- `configs/learning/offline_topology_node_experiment.yaml`
- `learning/structural_learning/tools/run_offline_topology_node_experiment.py`

# Acceptance Test

1. Verify old and new encoder/projection checkpoints, SHA256 hashes, epoch,
   reload consistency, eval mode, L2-normalized outputs and deterministic repeat
   inference.
2. Calibrate all feature matching and trigger thresholds using only perlin2
   validation, then freeze and save them to `frozen_thresholds.json`.
3. Evaluate input-geometry, old 64D, new 64D and new 128D bottleneck groups on
   perlin2, ku, campus and indoor using identical data, candidates and metrics.
4. Save node-level metrics, graph metrics, perturbation metrics, embedding
   health, figures, logs, README and summary under
   `results/offline_topology_node_experiment/20260806_multienv_v4/`.
5. Preserve stdout/stderr and do not run online M-TARE integration.

# Rollback Method

Remove `configs/learning/offline_topology_node_experiment.yaml`,
`learning/structural_learning/tools/run_offline_topology_node_experiment.py`,
and only the current
`results/offline_topology_node_experiment/20260806_multienv_v4/` directory.
Leave datasets, frozen checkpoints and M-TARE planning code unchanged.

---

# Local Topological Semantic Teacher v1

# Scope

Build a data-only teacher dataset whose labels come from complete M-TARE
simulation environment maps and describe local traversable-space connectivity,
not embedding-distance node triggering.

# Forbidden Changes

- training, fine-tuning or checkpoint modification
- embedding threshold optimization or offline node-trigger tuning
- M-TARE online planner, GridWorld, local planner, communication or allocation
  changes
- use of complete traversable maps as future model input

# Files Planned For Change

- `todo.md`
- `configs/learning/topological_semantic_teacher_v1.yaml`
- `learning/structural_learning/tools/export_topological_semantic_teacher_v1.py`

# Acceptance Test

1. Audit M-TARE complete-map sources, planner collision/terrain parameters and
   world mesh/preview pointcloud availability.
2. Build inflated traversable grids from complete environment pointclouds and
   validate trajectory centers fall in traversable space.
3. Export train/val/test by world with current and historical online surface
   inputs plus traversability-only teacher labels.
4. Save direction reachability, local connectivity, continuous structural
   scores, fixed-dimensional role teacher, previews, counterexamples and
   quality metrics under `results/topological_semantic_teacher_v1/`.
5. Compare existing frozen 64D geometry representation to surface and topology
   teacher differences without training or changing the model.

# Rollback Method

Remove `configs/learning/topological_semantic_teacher_v1.yaml`,
`learning/structural_learning/tools/export_topological_semantic_teacher_v1.py`,
and only `results/topological_semantic_teacher_v1/`. Leave M-TARE source,
recorded bags, existing checkpoints and previous results unchanged.

---

# Topological Semantic Dataset v2 Online Observation

# Scope

Build a data-only dataset that pairs real M-TARE online observations from
`/registered_scan` and `/state_estimation_at_scan` with offline complete-map
local topological semantic teachers. Do not train models or tune topology node
thresholds.

# Forbidden Changes

- training, fine-tuning or checkpoint modification
- topology node generation, threshold tuning or M-TARE planner integration
- use of complete mesh pointclouds as formal student input
- future scans or complete-map teacher fields in student history

# Files Planned For Change

- `todo.md`
- `configs/learning/topological_semantic_dataset_v2_online_observation.yaml`
- `learning/structural_learning/tools/export_topological_semantic_dataset_v2.py`

# Acceptance Test

1. Audit available M-TARE bags for `/registered_scan` and
   `/state_estimation_at_scan` exact-stamp synchronization by world.
2. Convert each usable bag through `MTAREBagAdapter -> StandardFrame ->
   SurfaceEvidenceBuilder -> surface_evidence_v1`.
3. Generate strictly causal current/history inputs with `history_valid_mask` and
   offline complete-map topological teachers at the same center pose.
4. Compare real registered-scan inputs against the old mesh range/FOV crop
   diagnostic baseline and save visibility counterexamples.
5. Save dataset stats, input/teacher contracts, split definition, sensor audit,
   traversability consistency audit and batch-read validation under
   `results/topological_semantic_dataset_v2/`.

# Rollback Method

Remove `configs/learning/topological_semantic_dataset_v2_online_observation.yaml`,
`learning/structural_learning/tools/export_topological_semantic_dataset_v2.py`, and
only `results/topological_semantic_dataset_v2/`. Leave v1 teacher data, M-TARE
source, existing bags and model checkpoints unchanged.

---

# M-TARE Online Bags for Topological Semantic Dataset v2

# Scope

Record real online M-TARE observations for tunnel, garage and forest so
`topological_semantic_dataset_v2` can be regenerated with non-empty train and
validation splits. Do not train models, redesign teacher labels, change splits or
run topology-node experiments.

# Files Planned For Change

- `todo.md`
- `scripts/record_mtare_transfer_world.sh`
- `configs/learning/topological_semantic_dataset_v2_online_observation.yaml`

# Acceptance Test

1. For tunnel, garage and forest, confirm world assets and launch files exist in
   the M-TARE runtime container.
2. Record at least two tunnel bags, two garage bags and one forest bag containing
   `/registered_scan` and `/state_estimation_at_scan`, plus optional terrain and
   trajectory audit topics.
3. Verify each bag is readable, has scan/pose topic counts, exact stamp sync,
   non-empty pointclouds, non-static pose motion and adequate duration or frame
   count.
4. Update the v2 config with the accepted bag paths without changing split
   policy.
5. Regenerate `results/topological_semantic_dataset_v2/` and verify train, val
   and test are all non-empty, batch reading passes and formal student inputs are
   sourced from registered scans only.

# Rollback Method

Restore `scripts/record_mtare_transfer_world.sh` from local backup or remove the
small tunnel/topic extension patch, restore the previous v2 config paths, and
remove only the current regenerated `results/topological_semantic_dataset_v2/`
and any failed recording directories from `results/mtare_transfer_recordings/`.

---

# Topological Semantic Supervision Audit v2

# Scope

Audit supervision learnability and causal history information for the existing
topological semantic dataset. Do not train a second full model, change test
worlds, tune topology-node thresholds, or run topology-node experiments.

# Files Planned For Change

- `todo.md`
- `learning/structural_learning/tools/audit_topological_supervision_v2.py`
- `contracts/exit_supervision_v2.json`
- `contracts/directional_teacher_v2.json`
- `contracts/canonical_role_teacher_v2.json`
- `contracts/topological_task_decomposition_v2.json`

# Acceptance Test

1. Compute per-split/world label distributions and simple baselines for all
   direction, distance and continuous-score targets.
2. Audit independent-exit density, logical consistency, circular continuity,
   checkpoint score distributions and a restricted positive-sample head fit.
3. Separate robot-frame directional supervision from rotation-canonical role
   supervision and record the proposed contracts.
4. Quantify causal history coverage gain using current-only and aligned history
   evidence without training a full second model.
5. Generate coverage-invariance positives and topology-aware hard negatives with
   fixed teacher targets and save provenance.
6. Produce `summary.json` with a gated READY/READY_WITH_LIMITATIONS/NOT_READY
   conclusion and preserve all logs and diagnostics.

# Acceptance Evidence

- direction metrics must beat constant baselines;
- exit labels must be temporally and rotationally coherent and fit a restricted
  positive subset;
- canonical role must not be defined by world heading or coverage alone;
- causal history must add measurable teacher-supporting information;
- all generated pairs must retain the original sample and teacher identifiers.

# Rollback Method

Remove only the new audit script, contract JSON files and
`results/topological_supervision_audit_v2/`. Leave the dataset, checkpoints,
teacher labels, M-TARE source and prior experiment results unchanged.

# Current Result

The GPU audit run `20260806_supervision_audit_v2_fix4` completed with
`TOPOLOGICAL_SUPERVISION_NOT_READY`. Direction labels have useful signal, but
the legacy independent 32-bin exit target is one-bin wide in every world and
the fixed-feature exit-head audit did not fit 32 positive samples (loss
0.2339 -> 0.1742, best F1 0.458). Checkpoint exit PR-AUC is 0.319 on train,
0.099 on forest and 0.225 on test; default 0.5 predicts no positives on forest.
The corrected coverage-pair audit produced 705 coverage-different/
teacher-similar pairs and 335 coverage-similar/teacher-different hard
negatives. Directional and canonical role fields are present separately but
must not be trained as one role target. History adds substantial causal union
surface evidence, while the existing model's unordered pooling cannot prove
ordered history use.

Result path:
`/home/zeng-workstation/mtare_topo_comm/results/topological_supervision_audit_v2/20260806_supervision_audit_v2_fix4`

The next permitted action is to repair/regenerate the exit-sector and role
supervision contract, then rerun this audit. Do not start a second full model
training run or topology-node experiment before the audit gate passes.

---

# Topological Supervision Contract Fix v2

# Scope

Repair direction metric semantics, convert legacy one-bin exit markers into
circular contiguous exit sectors, expose directional and canonical role fields,
make causal history timing explicit, split static/dynamic teacher masks, and
export a new v4 supervision dataset. Do not train a full second model or run
topology-node experiments.

# Files Planned For Change

- `todo.md`
- `learning/structural_learning/topological_metrics.py`
- `learning/structural_learning/topological_supervision.py`
- `learning/structural_learning/tools/export_topological_semantic_dataset_v4_supervision_fixed.py`
- `learning/structural_learning/dataset.py`
- `learning/structural_learning/tools/check_topological_semantic_dataset_v4.py`
- `tests/test_direction_metrics.py`
- `configs/learning/topological_semantic_dataset_v4_supervision_fixed.yaml`
- `contracts/exit_supervision_v2.json`
- `contracts/directional_teacher_v2.json`
- `contracts/canonical_role_teacher_v2.json`
- `contracts/topological_task_decomposition_v2.json`

# Acceptance Test

1. Direction positive-class metrics and all requested baselines pass isolated
   unit tests, including the all-positive F1 sanity check.
2. v4 samples contain circular exit sectors, fixed-size padded fields, separate
   directional/canonical role targets, static/dynamic masks, causal history
   order and time deltas.
3. Dynamic labels are valid only for adjacent same-trajectory teacher pairs.
4. Coverage-invariance and hard-negative pairs retain sample paths and teacher
   distances without using world identity as a label.
5. Train/val/test mixed batches pass shape, mask and finite-value checks; a tiny
   exit/role smoke head is run only as a learnability diagnostic.

# Rollback Method

Remove only `results/topological_semantic_dataset_v4_supervision_fixed/` and the
new v2 supervision scripts/contracts. Leave v3 data, v1/v2 audit results,
checkpoints, M-TARE source and prior experiment directories unchanged.

# Current Result

The GPU export and read/smoke checks completed. The new dataset contains
train=3024, val=356 and test=1469 samples. Direction metrics now use explicit
positive-class confusion matrices; the old forest all-positive F1 is 0.82732,
which matches the frozen model's corrected F1 0.82716 and explains the earlier
false claim of learned direction semantics. The v4 contract itself is
consistent: exit sectors have mean widths of 5.37, 8.96 and 3.16 bins for
train, val and test; canonical rotation cosine distance is approximately zero;
adjacent exit-sector IoU median is 0.731; all mixed batches are finite and
causal-history checks pass. The 64-sample smoke head fits exit sectors and
canonical role (PR-AUC 0.999999, uniform-threshold F1 0.9994, role cosine
distance 0.0073). The result is
`TOPOLOGICAL_SUPERVISION_FIXED_WITH_LIMITATIONS` because direction learnedness
has not been demonstrated above the forest constant baseline, history time
deltas are inferred from trajectory spacing, and dynamic labels are valid only
for adjacent same-trajectory samples.

Dataset path:
`/home/zeng-workstation/mtare_topo_comm/results/topological_semantic_dataset_v4_supervision_fixed`

Audit path:
`/home/zeng-workstation/mtare_topo_comm/results/topological_semantic_dataset_v4_supervision_fixed_audit_fix2`

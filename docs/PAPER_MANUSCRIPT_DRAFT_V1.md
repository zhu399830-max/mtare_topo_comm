# Planner-Consistent Structural Semantics for Persistent Topological Exploration in Underground Environments

Status: `SUPERSEDED_EXIT_ONLY_BASELINE_DRAFT_RETAINED_FOR_ABLATION_TEXT`  
Target scope: focused robotics paper; final venue depends on single-robot, multi-robot, strict-generalization and field evidence.  
Bibliography seed: `docs/references/structural_topology_novelty_20260823.bib`

> This draft describes the exit-only deterministic-graph baseline and is no longer the main-method manuscript. The active paper route is GSE-Graph; do not reuse this abstract or contribution list as the submission claim.

> This draft intentionally contains evidence-bound placeholders. A placeholder may be replaced only by a value or statement generated from a sealed run. It is not submission-ready.

## Abstract

Large-scale underground exploration requires a global representation that remains useful under perceptual aliasing, long corridors, sparse junction observations, and constrained communication. Existing hierarchical exploration systems such as M-TARE use a coarse metric global map, while purely topological underground navigation detects local tunnel directions but does not establish whether learned local exit-direction cues improve exploration when converted into causal graph state under an unchanged motion stack. We introduce a planner-consistent structural-topological layer that converts learned LiDAR exit directions, geometric count/role cues, and physical traversal evidence into persistent nodes, trace-verified edges, unresolved exit stubs, and execution-aware exit lifecycles. Only exit direction is learned; graph construction, count/role cues, node association, edge verification, and target lifecycle are deterministic. The layer replaces only M-TARE's global representation and target-selection interface; state estimation, collision handling, local planning, path following, and control remain unchanged. We evaluate the method with topology-parent-isolated supervision and paired randomized-block closed-loop experiments against original M-TARE, a pre-correction graph planner, and a complete-map diagnostic. **[AUTO:GATE6_PRIMARY_RESULT]**. In multi-robot experiments, **[AUTO:GATE7_RESULT_OR_REMOVE]**. Strict held-out evaluation shows **[AUTO:STRICT_RESULT]**. These results delimit when learned exit-direction semantics and causal graph state improve global exploration and when the additional graph machinery does not justify replacing the original global layer.

## 1. Introduction

Underground environments are dominated by repeated local geometry, extended corridors, ramps, dead ends, and sparse intersections. These properties make dense global metric maps costly to communicate and can make nearby locations difficult to distinguish from a single scan. They also make a purely reactive tunnel-direction detector insufficient for exploration: a robot must remember which exits were observed, which were physically traversed, which target was actually executed, and whether a return traversal reached a previously created node.

M-TARE addresses large-scale exploration with a high-resolution local map and a coarse global map. Its hierarchy is a strong baseline because it separates detailed nearby motion planning from computationally cheaper global exploration. Our question is narrower than replacing this full stack: can a causal structural-topological representation replace only its global representation and target layer while preserving all local execution components?

We study this question through a controlled system interface. A local LiDAR observation predicts nearby tunnel-exit directions. A deterministic online graph maintains persistent structural nodes, trace-verified traversals, and unresolved exit stubs. Global targets are selected from graph frontiers and routed only across verified edges, then handed to the unchanged M-TARE local planner. Physical execution is fed back into the graph: verified returns deterministically re-anchor the active node, while a departure that does not match the selected exit is recorded in the existing retry state before replanning.

The paper makes the following evidence-dependent contributions:

1. A traceable procedural-topology supervision pipeline for local underground exit direction, with topology-parent atomic isolation and explicit learned-versus-geometric output boundaries.
2. A causal online topometric representation with persistent structural nodes, verified edges, unresolved exit stubs, and execution-aware lifecycle updates.
3. A controlled replacement of the M-TARE global target layer, evaluated with the same sensor, world, time budget, local planner, path follower, collision handling, and control.
4. **[CONDITIONAL_ON_GATE7]** A shared-graph multi-robot allocation interface evaluated by communication, conflict, redundancy, coverage and team travel rather than by algorithmic unit tests alone.
5. A failure-preserving evaluation protocol that reports the defective graph planner, causal corrections, poor-but-valid runs, strict held-out performance, and source-sealed paper artifacts.

## 2. Related Work

### Hierarchical metric exploration

TARE and M-TARE use representation granularity to retain detailed local planning while reducing the cost of global exploration. The multi-robot system uses global allocation and communication-aware pursuit. We therefore use original M-TARE as the primary baseline and preserve its local execution stack; our method is not compared to an intentionally weakened reimplementation [cao2023representation].

### Underground topological navigation

Cano et al. learn tunnel directions from synthetic underground LiDAR and use a purely topological understanding for navigation, including simulation and real-world validation [cano2026autonomous]. This establishes that learned gallery bearings and lightweight underground topology are feasible. Our question differs in the persistent causal state and controlled exploration replacement: the graph is built from partial observations, only physical traces verify edges, unresolved exits remain first-class state, and the objective is exploration performance under the same M-TARE local stack.

GRID-FAST derives structural-semantic topometric maps from an existing 2-D grid map [fredriksson2024gridfast]. Our system instead asks whether local LiDAR semantics and causal history can support an online exploration graph before a complete global grid is available.

### Topological and semantic multi-robot exploration

Similarity-score topological memory uses shared visual graph features to estimate visitation and select global goals [lee2024multiagent]. Distributed topological graph Voronoi methods construct connectivity/frontier/coverage graphs and balance workload [ding2025balanced]. LTVMap demonstrates compact topological-volumetric communication for subterranean robot teams [petrlik2023uavs], while hierarchical terrain-aware graphs encode multimodal traversability for ground-air cooperation [patel2025hierarchical]. SAGE studies scale-independent topological logic for zero-shot multi-robot exploration [cao2025sage]. Consequently, neither graph sharing nor Hungarian assignment alone is claimed as novel. Our multi-robot hypothesis, if supported, concerns structure-exit task units, causal verified connectivity, incremental graph communication, and closed-loop execution feedback.

## 3. Problem Formulation

At time step \(t\), the robot has a local LiDAR observation \(z_t\), pose estimate \(x_t\), prior causal graph \(G_{t-1}\), and local-planner execution feedback \(u_{t-1}\). It does not have access to future observations, a complete map, teacher topology, or simulator geometry. The global layer must produce a finite map-frame target \(g_t\) or a candidate-complete state:

\[
  (G_t, g_t) = F(G_{t-1}, z_t, x_t, u_{t-1}).
\]

The local M-TARE stack executes \(g_t\) without modification. We evaluate whether replacing the original global layer with \(F\) improves coverage-time AUC, final and mean explored volume, travel, redundancy, volume per travel, and planner latency while respecting the same runtime and sensor contracts.

## 4. Method

### 4.1 Structural-semantic perception

The deployment input is the organized 16×350 AEE LiDAR scan. A frozen, exact-angle adapter maps it to the model input convention while preserving the duplicate ±π source records in provenance. The learned component predicts local exit directions. Branch count and structural role are supplied by a separately frozen geometric interface and are not presented as learned outputs. The model checkpoint is selected without reading formal held-out worlds.

### 4.2 Persistent causal topometric graph

Each graph node represents a persistent local structural event. Exit stubs represent locally observed but not yet verified outgoing possibilities. An edge is created only after a physical traversal trace links two node events. Online association may merge an event with an existing node under frozen geometric and heading rules, but it cannot infer an edge from teacher topology or complete-map proximity. The graph therefore encodes observed structural opportunity separately from verified connectivity.

### 4.3 Graph frontier and routing

Unresolved, eligible exit stubs form graph frontiers. The global layer ranks these frontiers under frozen costs and computes routes only over verified edges. The resulting map-frame waypoint is passed to M-TARE's existing local planner. No method-specific obstacle avoidance, controller, terrain processing, or state estimator is introduced.

### 4.4 Execution-aware lifecycle correction

The pre-correction planner exposed two distinct state failures. First, a verified backtrack could physically reach an old node without changing the graph's active-node identity. Second, the robot could leave through a different exit than the selected stub, or loop back to the same node, while the selected stub remained eligible and was repeatedly chosen.

The predeclared V5 candidate applies two deterministic corrections, subject to the sealed Gate 6 qualification sequence. A verified return trace re-anchors the active node when it satisfies the frozen edge, route and proximity evidence. A frontier attempt whose verified departure does not match the target stub is written into the existing retry ledger and triggers target reselection. These updates add no learned parameter or acceptance threshold. They are described here as the frozen method under evaluation, not as a completed empirical result.

### 4.5 Multi-robot extension

**[CONDITIONAL_ON_GATE7]** Robots exchange revisioned causal graph snapshots rather than dense global maps. Only trace-verified edges are merged. Eligible shared exit stubs are assigned one-to-one, with leases and execution feedback preventing simultaneous commitment to the same target. The final paper must report namespaced ROS transport, out-of-order and reconnect behavior, communication bytes, conflict safety, and 1/2/3/4-robot closed-loop results; otherwise this subsection and the corresponding contribution are removed.

## 5. Experimental Protocol

### 5.1 Data isolation and perception

Procedural topology parents are the independent sampling unit. Training, validation and strict test parents are disjoint. Formal test worlds do not contribute labels, normalization, augmentation choices, thresholds, or checkpoint selection. The final paper will report raw/effective frame counts, spatial and temporal spacing, teacher provenance and all forbidden-read audits from the approved Data Cards.

### 5.2 Causal graph replay

The graph is first qualified on sealed causal trajectories to verify that every node, edge and exit state can be reproduced from observations available at that time. Complete topology is used only for objective labels and evaluation, not online graph updates.

### 5.3 Single-robot closed loop

The development comparison uses two worlds (tunnel and garage), five environment seeds, and ten world-by-seed randomized blocks. Original M-TARE, the graph method, and the complete-map diagnostic each contribute 30 cases under a 600-s budget. Model checkpoints/repeats quantify stochastic variation but do not create additional independent worlds. Under the frozen Gate 6 protocol, the planned corrected V5 cases must exactly match the 30 pre-correction graph identities; this claim becomes a result only after the composed audit, probe, readiness run, paired run, and final comparison are sealed.

The primary metric is coverage-time AUC. Six additional metrics assess final coverage, mean coverage, travel, cumulative point redundancy, volume per travel, and planner latency. Inference uses block-level paired differences, exact sign-flip tests, 10,000-sample block bootstrap intervals and Holm correction. The complete-map method is diagnostic, not a theoretical upper bound.

### 5.4 Multi-robot, strict generalization and ablations

**[AUTO:GATE7_PROTOCOL]**  
**[AUTO:STRICT_PROTOCOL]**  
**[AUTO:ABLATION_PROTOCOL]**

Required ablations separate perception, persistent graph state, verified re-anchor, frontier execution feedback, shared graph, and allocation. Strict test worlds are opened once after development parameters are frozen.

## 6. Results

### 6.1 Structure-semantic perception and causal graph validity

**[AUTO:GATE2_TABLE_AND_TEXT]**  
**[AUTO:GATE4_TABLE_AND_TEXT]**

### 6.2 Single-robot exploration

**[AUTO:GATE6_FAMILY_SUMMARY_TABLE]**  
**[AUTO:GATE6_PAIRED_EFFECT_TABLE]**  
**[AUTO:GATE6_COVERAGE_TIME_FIGURE]**  
**[AUTO:GATE6_FIXED_TRAJECTORY_GRAPH_FIGURES]**

The final text must distinguish V5 versus original M-TARE (the performance question) from V5 versus defective V9 (the causal correction question). A valid poor outcome remains in the analysis; only system-invalid runs may be replaced under a predeclared whole-block recovery policy.

### 6.3 Mechanism and failure analysis

**[AUTO:V9_MECHANISM_AUDIT]**  
**[AUTO:V5_MECHANISM_COUNTS]**  
**[AUTO:REMAINING_FAILURES]**

### 6.4 Multi-robot and strict held-out results

**[AUTO:GATE7_RESULTS_OR_REMOVE]**  
**[AUTO:STRICT_RESULTS]**

## 7. Discussion and Limitations

The method assumes tunnel-network structure is locally observable often enough to maintain useful exit identities. Repetitive geometry, odometry drift, stacked passages, missed branches, and local-planner deviations can corrupt a graph even when the perception output is accurate. The graph is a causal partial representation, not a complete reconstruction.

The current learned component predicts exit direction; count and role remain geometric. Accordingly, the paper does not claim end-to-end structural understanding. The use of procedural supervision also requires a strict parent-level leakage audit and held-out evaluation.

If only simulation evidence is available, conclusions are limited to controlled underground exploration in the evaluated simulator. Field-robotics superiority requires real underground closed-loop trials with intervention, localization failure and communication-loss reporting.

## 8. Reproducibility and Evidence Binding

Every material experiment is created from an approved Data Card and frozen run specification. Results preserve commands, environments, seeds, schedules, raw logs, per-case evidence, metrics, failure reasons and SHA-256 seals. Paper figures and tables are generated from sealed metrics and source-verified trajectories; result values are never copied manually. The release must distinguish third-party Cano generation assets, M-TARE/AEE code, frozen model checkpoints and newly implemented graph/planner components.

## 9. Conclusion

**[AUTO:CONCLUSION_AFTER_ALL_GATES]**

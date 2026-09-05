# GSE causal geometry / risk-conflict audit plan V1

Status: `PREDECLARED_NOT_EXECUTED`  
Execution condition: only if the active bounded last-encoder-block run fails its unchanged event gate.

## Research question

The model can regress continuous width, height, slope and curvature change, but accepted change-point identities remain zero. The next audit asks one narrow question:

> Is the failure caused by insufficient causal history, by a global structural-confidence gate that suppresses rare events, or by the absence of a transferable multivariate geometry pattern?

This is a representation/metric audit. It is not a topology run and cannot authorize C09, C10 or M-TARE.

## Fixed data boundary

- C01--C06 fit: 60 worlds, 142,184 observations.
- C07--C08 selection: 20 worlds, 45,942 observations and exactly 17 corrected change-point identities.
- Corrected persistent bidirectional Teacher V1R only.
- Original three GSE checkpoints plus the sealed failed frozen and bounded-last-block outputs.
- Zero C09/C10/M-TARE reads, zero model/optimizer updates and no graph sweep.

All observations remain grouped by physical structure identity. Adjacent frames are never counted as independent evidence.

## Audits

1. **Confidence decomposition**

   Separate binary structural confidence from conditional event class. Report, for corrected change-point frames and hard corridor negatives, the structural score, conditional change-point probability, final class probability and accepted identity coverage. This tests whether the class is visible but rejected by the global safety gate.

2. **Causal-history observability**

   Recompute signed width/height/slope/curvature changes at fixed lags 2--12 m without selecting a winning lag. Report valid pair counts, transition-vs-corridor AUC, the C01--C06 corridor 99th-percentile threshold transferred unchanged to C07--C08, false-positive rate and identity coverage for every lag.

3. **Hard-negative stratification**

   Split corrected corridor rows into ordinary corridor and old-transition rows removed by the corrected Teacher. Quantify which group occupies the high-score tail. This tests whether prior seam-like labels dominate the one-percent risk budget.

4. **Multivariate capacity proof**

   Fit one predeclared low-capacity geometry-conditioned risk model on C01--C06 identities only. Inputs are signed multi-lag geometry deltas, their predicted uncertainty and the frozen conditional event probability; no pose, world, identity value, TNG or future frame is an input. C07--C08 selects only the non-vacuous acceptance threshold. Identity-balanced sampling and a leave-family-out fit diagnostic are mandatory.

## Decision rule

The geometry-conditioned route is implementable only if the selection result simultaneously provides:

- node precision at least `0.98`;
- false acceptance at most `0.01`;
- structural recall at least `0.40`;
- at least `7/17` correctly classified change-point identities;
- unchanged junction/terminal/turn identity floors;
- no family with an empty or unsafe accepted set.

If the capacity proof passes, the paper method becomes a causal, geometry-conditioned risk accumulator rather than another generic event classifier. If it fails, the current five-frame/local-event formulation is not locally observable at the required risk level; stop training larger variants and revise the input/Teacher contract before any topology or planner experiment.

## Existing diagnostic motivation (not formal evidence)

- The original backbone assigns the change-point class as framewise argmax on `206/240` corrected selection frames, but median structural confidence is only about `0.69`; the safe global threshold is approximately `0.99`.
- Frozen corrective training reduces accepted change-point identity coverage to `0/17` while continuous delta MAE improves by `39.51%`.
- Fixed 2--5 frame confidence persistence remains `0/17`.
- Increasing the geometry comparison span from 4 m to 8--10 m raises frozen-prediction AUC from about `0.68` to `0.74--0.76`, but scalar one-percent-FPR identity coverage remains zero.

These values only motivate the immutable audit. They must be regenerated, source-bound and sealed before appearing as paper claims.

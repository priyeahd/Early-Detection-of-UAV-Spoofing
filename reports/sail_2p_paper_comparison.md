# sail_2p comparison with Huang et al. (2025)

## Conclusion

Sail_2p has higher numerical classification scores on its own held-out PX4
flights than the published MSSTP-OAD scores. This is a comparison across
experiments, not evidence that Sail_2p outperforms MSSTP-OAD on a common task.
The data, attacks, labels, features, decision cadence, and recovery behavior
are not matched. The paper does not specify an 80/20 train/test protocol.

Source: the supplied 15-page PDF, *Prediction-based trajectory anomaly detection
in UAV system with GPS spoofing attack*, Tianci Huang, Huici Wu, Xiaofeng Tao,
Zhiqing Wei, Chinese Journal of Aeronautics 38(10), 103478,
DOI 10.1016/j.cja.2025.103478. Page numbers below are printed PDF pages.
Relevant tables and figures were visually checked after PDF rendering.

## Verified local experiment

The saved selection specifies a new estimator, 288 training flights, 72 test
flights, 706,080 training rows, 42 features, 200 trees, depth 4, learning rate
0.08, probability threshold 0.5, and no validation tuning. The test set has
174,726 samples: 92,168 positive and 82,558 negative, across 44 attacked and
28 normal flights. All four frozen input hashes and the model hash match.
Recomputed confusion counts and classification metrics exactly match the saved
test_metrics.json. No retraining or threshold adjustment was performed here.

## Closest shared classification metrics

The paper defines accuracy, recall and F1 with the usual binary confusion-count
formulas in Eqs. (30)-(32), p. 9. Its Fig. 12, p. 12 and abstract give the
MSSTP-OAD values below. Our numbers are pooled per-sample values at the export
cadence. The formulas match; the observations being counted do not.

| Metric | Paper MSSTP-OAD | Sail_2p XGBoost | Sail_2p XGBoost + CUSUM pulse |
| --- | ---: | ---: | ---: |
| Accuracy | 98.4% | 99.9851% | 99.9061% |
| Recall | 98.8% | 99.9859% | 99.8242% |
| F1 | 98.3% | 99.9859% | 99.9110% |
| Precision | Not directly reported | 99.9859% | 99.9978% |
| False-positive samples | Not reported | 13 | 2 |
| False-negative samples | Not reported | 13 | 162 |

XGBoost's numerical differences are +1.5851 percentage points accuracy,
+1.1859 points recall, and +1.6859 points F1. These are descriptive differences,
not matched treatment effects or statistically established improvements.
Do not infer exact paper confusion counts or precision from rounded metrics.

The paper's GA-XGBoost comparator is a separate implementation of a cited
two-step genetic-algorithm method; Sail_2p is not a reproduction of that baseline.
The paper's reported trajectory-prediction R2 and RMSE (Table 3, p. 11) do not
apply to our classifier because it does not predict future coordinates.

## Trajectory-restricted analysis without retraining

The paper uses random curved 3D trajectories. The following descriptive
subgroups remove, or separately display, our straight-flight contribution.
The dataset's scenario names do not establish identical flight dynamics or
attack difficulty. This is a post hoc breakdown of the existing test set;
the model remains the one trained on all training scenarios.

| Existing test subset | Flights | Samples | XGBoost accuracy | XGBoost recall | XGBoost F1 | CUSUM F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Curved | 22 | 96,835 | 99.9928% | 99.9922% | 99.9945% | 99.9742% |
| Random | 28 | 59,911 | 99.9750% | 99.9644% | 99.9555% | 99.7085% |
| Curved + Random | 50 | 156,746 | 99.9860% | 99.9864% | 99.9864% | 99.9188% |

On the combined subset, XGBoost has 11 FP and 11 FN. CUSUM has 2 FP and 129 FN.
Both alert during all 29 attack intervals. The high aggregate result therefore
does not depend on including straight flights. It still does not establish
performance under the paper's random-waypoint attack construction.

## Detection delay and false alarms

The paper reports distributions and means in Fig. 14, p. 13. Its proposed
method has approximately 33 s mean detection duration (read from an unlabelled
bar; not an exact tabulated value), with 24.4% reduction against its own LSTM
baseline. Recovery duration is approximately 61 s. Detection and recovery
distances are separate closed-loop results in Fig. 15.

| Timing/flight measure | Sail_2p XGBoost | Sail_2p CUSUM pulse |
| --- | ---: | ---: |
| All attacks alerted during attack | 44/44 | 44/44 |
| Attacks with a new alert during attack | 40/44 | 44/44 |
| Already alerting on sample before first active sample | 6/44 | 0/44 |
| Mean first active alert delay, all attacks | 35.5 ms | 86.2 ms |
| Median first active alert delay, all attacks | 30 ms | 82 ms |
| Maximum first active alert delay, all attacks | 124 ms | 176 ms |
| Mean first active alert delay, Curved + Random | 37.5 ms | 84.3 ms |
| Normal flights with any alarm | 0/28 | 0/28 |
| Spoofing-off false-positive samples | 13/82,558 | 2/82,558 |

The XGBoost delay is time to the first positive prediction during the active
interval, not necessarily a new alarm after onset. The six pre-alerting flights
make that distinction important. CUSUM produces a new pulse for each attack
and is the cleaner event-trigger summary here, although it has two false
pulses in spoofing-off portions of attacked flights. Zero alarms on entirely
normal flights does not mean zero false alarms overall.

These are elapsed telemetry times from the original-rate attack-enabled flag,
not CPU inference latency or demonstrated onboard response times. A 0 ms delay
means a positive prediction at the sampled onset; it does not mean zero compute
time. Preprocessing sorts merged logs, and actual online sensor delivery remains
unverified. Do not divide approximately 33 s by our milliseconds and claim a
speedup. We have no trajectory-recovery or trusted-distance result to compare.

## Protocol differences that prevent a head-to-head claim

| Aspect | Paper | Sail_2p |
| --- | --- | --- |
| Data/environment | Own SITL experiment; Ubuntu 18.04.6; 1,000 x 1,000 x 60 m space (p. 9) | Existing PX4 merged-log collection; offline replay |
| Attack implementation | Random offsets to part of preset waypoints; horizontal attack discussion (pp. 9-10) | Source GPS attack-enabled intervals; equivalence of offset mechanism/magnitude not established |
| Target | Trajectory anomaly relative to mission and motion consistency | Per-sample attack-enabled classification |
| Trajectories | Random curved 3D paths | Curved, Random, Straight scenario groups |
| Split | No reproducible split ratio, flight assignments, or seed specified | 80/20 whole flights; stratified by scenario/condition; seed 42 |
| Telemetry/decisions | 4 Hz; n=20 slots, d=5, delta=1 s, k=5; final decision every n slots = 5 s (pp. 9-10) | Approx. 20 Hz predictions; past windows 0.25/1/2 s plus cumulative features |
| Model | Stacked LSTM predictor plus two ensemble stages, each using MLP/SVM/HGB with unanimous decisions | XGBoost; optional original CUSUM alarm rule |
| Inputs | Motion-state sequences, attitude/angular/magnetic information, preset waypoints, predicted positions | Current velocity/acceleration, GPS consistency windows, pressure-altitude residual, cumulative GPS displacement errors |
| Preprocessing | Standardization/normalization described; fit partition not specified | No learned scaling; per-flight past/current feature calculations |
| Response to detection | Return toward original waypoints; detection changes subsequent flight | Passive replay; no feedback to flight controller |
| Runtime | 70.3 s for 30,000 data items (Table 4, p. 14) | No comparable end-to-end runtime benchmark saved |

The 70.3 s in Table 4 is computation time for a batch, not the approximately
33 s attack-to-detection duration in Fig. 14. These must not be mixed.
The paper does not provide enough split/preprocessing detail to verify its
leakage controls; this absence is not evidence that leakage occurred.

## What is needed for a genuinely matched comparison

1. Run both methods on exactly the same held-out flights with matching attack
   offsets, rates, magnitudes, onset definitions, and reference waypoints.
   Confirm raw mission waypoint availability before attempting EVAD replication.
2. Rebuild both input pipelines from the same causal 4 Hz telemetry stream.
   Merely retaining every fifth saved 20 Hz prediction would still use features
   formed from higher-rate history and would not reproduce the paper's protocol.
   Report native decision cadence and a separately agreed common evaluation grid.
3. Train/tune using only development flights, keeping related simulation seeds
   or flight families together. Freeze settings before opening a new final test
   set. The currently explored 360 flights cannot become historically untouched
   just by creating another new estimator or random split.
4. Report common sample metrics, per-event detection rate, mean/median/tail
   onset-to-new-alarm delay, false alarms per observed flight-hour, and per-flight
   uncertainty. Avoid treating correlated telemetry rows as independent trials.
5. Measure end-to-end inference on the same hardware, including feature
   construction and temporal buffering. Evaluate recovery and distance only
   in a common simulator with both detectors connected to the same controller.

## Suggested wording

“On a whole-flight 80/20 split of the PX4 dataset, a freshly trained XGBoost
model achieved 99.985% accuracy, 99.986% recall, and 99.986% F1. These values
are numerically above the 98.4%, 98.8%, and 98.3% reported by Huang et al.
for MSSTP-OAD. Excluding straight trajectories yielded 99.986% F1 across 50
test flights. The comparison is contextual rather than head-to-head because
the datasets, attack definitions, input features, decision cadence, and
closed-loop behavior differ.”

## Local evidence

- `results/runs/sail_2p/selection.json`: frozen configuration and provenance.
- `results/runs/sail_2p/test_metrics.json`: archived full-test metrics.
- `results/runs/sail_2p/test_alarms.csv`: recomputation of confusion counts.
- `results/runs/sail_2p/test_flight_events.csv`: event timing and false alarms.
- `../UAV-GPS-Spoofing-Dataset/sail_2p/splits.csv`: scenario and split assignments.
- `reports/sail_2p_paper_comparison_metrics.json`: verified descriptive subgroup
  metrics and mean/median/range/95th-percentile event delays from this analysis.

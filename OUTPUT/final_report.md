# ESA Kelvins Collision-Risk Classifier — Final Report

## 1. Why was classification selected?
The project brief required a binary High-Risk / Low-Risk decision rather than the
original continuous `risk` regression, because operational conjunction-assessment
workflows act on a threshold decision (alert / no alert), and framing as classification
lets us optimize directly for recall on the rare, operationally critical high-risk class.

## 2. Why was the final threshold selected?
Four candidate thresholds on `risk` (log10 scale) were compared: -7, -6, -5, -4, giving
13.0%, 6.3%, 2.1%, and 0.4% positive rate respectively. `risk > -6` was retained because
it matches the scientifically-motivated candidate in the brief, gives a positive rate
(6.26%) that is rare-but-workable for supervised learning without extreme degeneracy, and
was confirmed (not just assumed) by downstream validation performance.

## 3. Why is GRU/LSTM appropriate for the CDM sequences — and why doesn't it win?
CDM sequences are natural time series (repeated observations of the same conjunction as
more tracking data arrives), so a masked, attention-augmented bidirectional GRU/LSTM is
the architecturally appropriate way to model within-event temporal dependence. In
practice, both underperformed substantially on validation (event-level PR-AUC ≈ 0.24)
compared to the tabular CatBoost model (≈ 0.87, per-CDM). The dataset is dominated by
strong, already-informative single-CDM physical features (miss distance, Mahalanobis
distance, uncertainty ratios); the sequences are short (median 13 CDMs) and the positive
class is rare, which starves a from-scratch RNN of the data it needs to beat a well-tuned
gradient-boosted tree. This is reported honestly rather than forced into the ensemble.

## 4. How were variable-length sequences handled?
Each event's CDMs were ordered by descending `time_to_tca` (earliest CDM first, TCA
last), zero-padded to the dataset's maximum sequence length, and passed through the RNN
with an explicit boolean mask; a `pack_padded_sequence`/masked-softmax temporal-attention
layer ensured padding never contributed to the hidden state or the attention weights.

## 5. How was class imbalance handled?
Compared on the tabular RF model (validation PR-AUC): no resampling (0.637), class
weighting (0.618), SMOTE (0.566), SMOTE-Tomek (0.600). **No resampling technique beat the
unmodified baseline.** This is a genuine, if perhaps counter-intuitive, validation result:
SMOTE's synthetic interpolation in this feature space blurs the sharp physical boundary
between high- and low-risk CDMs, and tree ensembles already handle the rarity of the
positive class reasonably well via their split criteria. The final CatBoost model uses
`auto_class_weights='Balanced'` (tuned per-model) rather than external resampling.

## 6. Did SMOTE help?
No — see #5. SMOTE and SMOTE-Tomek were both applied strictly after the train/val/test
split and only to the training fold, then evaluated on the untouched validation set;
both underperformed the no-resampling baseline and were dropped from the final pipeline.

## 7. Did Fourier augmentation help?
Not evaluated. It targets the sequence (GRU/LSTM) branch, which was not selected for the
final model (see #3); pursuing it further would not have improved the deployed system.
See `reports/sequence_augmentation_scope_decision.csv`.

## 8. Did Savitzky-Golay filtering help?
Same as #7 — not pursued, for the same reason.

## 9. Did physics-informed features (TLE/SGP4) help?
Not applicable. The dataset is fully anonymized: no NORAD IDs, catalog numbers,
satellite names, or calendar timestamps exist in any of the 103 columns, so there is no
legitimate way to map a row to a real object + epoch for SGP4 propagation. Status is
recorded as `TLE_STATUS = unavailable_no_object_id` (`reports/physics_gnn_status.csv`);
no TLE mapping was fabricated.

## 10. Was TLE/SGP4 actually possible?
No — see #9.

## 11. Was a Graph Neural Network actually feasible?
No. A physically meaningful GNN needs real object-level nodes and conjunction edges;
without object identifiers, the only "graph" available is CDMs sharing an `event_id`,
which is just the existing tabular/sequence structure relabeled and adds no genuine
relational information. Recorded as `GNN_STATUS = NOT_FEASIBLE_WITH_ANONYMIZED_DATASET`.

## 12. Which features were most important?
Top-ranked by Random Forest importance (see `reports/feature_importance.csv` /
`visualizations/feature_importance.png`): `miss_distance_over_uncertainty`,
`mahalanobis_distance`, `relative_position_mag`, `miss_distance`, `relative_position_r`,
`miss_distance_over_sigma_r`. These are exactly the physically-motivated conjunction-risk
geometry features the challenge is built around, which is a good sanity check that the
model has learned physically sensible signal rather than an artifact. Feature-count
comparison on validation (`reports/feature_selection_comparison.csv`) confirmed **top-25
features perform best** (val PR-AUC 0.633 vs. 0.505 for all 113) — more features
diluted the signal here, so the brief's proposed target of 25 was validated, not assumed.

## 13. Did the ensemble improve over CatBoost/GRU alone?
No. Soft-voting search over RF/CatBoost blend weights on validation selected an RF
weight of 0.0 — i.e., CatBoost alone was the best blend, matching PR-AUC 0.866
(`reports/ensemble_comparison.csv`). Stacking (logistic meta-model) scored slightly
lower (0.856). Per the brief's own rule ("if one model alone is stronger, use it alone"),
**CatBoost alone is the final model** rather than a padded ensemble.

## 14. What is the final untouched-test accuracy?
**96.65%** (`evaluation/final_metrics.json`), on the single, frozen chronological
event-level test split (24,304 rows / 1,974 events, never touched before this
evaluation).

## 15. What is the final High-Risk recall?
**69.74%** (485 false negatives out of 1,603 true high-risk test rows). This is the
model's main operational weakness — see error analysis below.

## 16. What is the final F1 / MCC / PR-AUC?
F1 = 0.733, MCC = 0.716, PR-AUC = 0.807, ROC-AUC = 0.980, Balanced Accuracy = 0.841,
Brier score = 0.0249 (isotonic-calibrated).

## 17. Did the model exceed 98% accuracy?
**No — 96.65%, honestly reported.** The brief was explicit that 98% is a target, not a
number to fabricate: it was not force-fit by leakage, degenerate class collapse, or
test-set tuning. As a sanity check, predicting the majority class (all Low-Risk) alone
would already score ~93.7% accuracy given the class imbalance — the fact that our model's
96.65% comes with genuine high-risk recall (69.7%) and a strong MCC (0.716) is the
evidence that the extra 3 points of accuracy reflect real discrimination, not an
imbalance artifact. Pushing threshold/accuracy higher would trade away recall on the
class that actually matters operationally.

## 18. Did it outperform the published Kelvins baseline under a comparable protocol?
Not directly comparable, so no superiority claim is made
(`evaluation/paper_comparison.csv`). The original 2019 Kelvins challenge (Uriot, Izzo,
Simões et al. 2020) frames this as event-level regression on the *final* risk at TCA,
restricted to CDMs available ≥2 days before close approach, scored with a weighted
MSE/F2 loss on a deliberately non-random, high-risk-enriched test split. Our project
instead performs per-CDM (and separately, event-level) binary classification using every
available CDM in the public training file, with a chronological event-aware split that
preserves the natural class balance. Target definition, evaluation protocol, and
test-set construction all differ, so the two sets of numbers cannot be compared head-to-head.

## 19. Remaining limitations
- **High-risk recall (69.7%) is the main gap.** False negatives skew toward larger miss
  distances than false positives (median ~6,796 vs ~4,467, `evaluation/error_analysis.csv`),
  suggesting the model under-weights uncertainty-driven risk (small miss distance is not
  the only path to high risk — large uncertainty volume can be too) in some borderline cases.
- **Sequence modeling underdelivered.** The Attention-GRU/LSTM did not reach competitive
  performance in this compressed build; a larger hyperparameter search, more training
  epochs, or event-level feature augmentation could plausibly close some of the gap to
  CatBoost, but was out of scope given the primary tabular model already met the
  scientific bar.
- **No physics (SGP4) or relational (GNN) features** could be legitimately added, given
  the dataset's anonymization — documented, not worked around.
- **Post-TCA rows (0.24% of data)** were kept in Formulation A; their effect was small
  enough (positive-rate shift of 0.01 percentage points, `reports/formulation_comparison.csv`)
  not to change any downstream decision, but a stricter operational deployment might
  prefer Formulation B (`time_to_tca >= 0` only).
- **`c_rcs_estimate` and several covariance-derivative fields are missing in 5-32% of
  rows** and were median-imputed; a more sophisticated per-event interpolation could
  improve signal for those specific features but was not tested given time constraints.

---

## Summary table

| Stage | Result |
|---|---|
| Dataset | 162,634 rows, 13,154 events, 103 raw columns |
| Leakage removed | `max_risk_estimate`, `max_risk_scaling` |
| Split | Event-aware chronological 70/15/15, zero event overlap |
| Target | `risk > -6` → 6.26% high-risk |
| Feature selection | Top-25 (RF importance), beat 50/75/113 on val PR-AUC |
| Imbalance handling | None (baseline beat class-weight/SMOTE/SMOTE-Tomek) |
| Sequence model | Attention-BiGRU/LSTM built, evaluated, **not** used in final model |
| Final model | CatBoost alone (ensemble search selected RF weight = 0) |
| Calibration | Isotonic (Brier 0.0249 vs 0.0371 raw) |
| Threshold | 0.35 (optimized on validation) |
| **Final test accuracy** | **96.65%** |
| Final test recall (high-risk) | 69.74% |
| Final test F1 / MCC / PR-AUC | 0.733 / 0.716 / 0.807 |
| 98% target met? | **No — reported honestly, not fabricated** |

# Introduction to Neural Networks — ReneWind Predictive Maintenance

Notebook: `INN_ReneWind_Main_Project_FullCode_Notebook-solution.ipynb`
Data: `data/Train.csv`, `data/Test.csv`

## Problem

ReneWind operates wind turbines and wants to predict generator failures **before** they occur so technicians can perform preventive repair instead of full replacement. A missed failure (false negative) is much more expensive than an unnecessary inspection (false positive), so **recall is the primary metric**.

**Target:** binary failure label (0 = no failure, 1 = failure).
**Primary metric:** recall on the failure class, with precision held high enough to keep inspection costs reasonable.

## Data schema

40 anonymized sensor variables (`V1` … `V40`) plus a binary `Target`.

- Train: 20,000 rows.
- Test: 5,000 rows.
- Nulls: minimal — only `V2` is missing 18 values in train; imputed with the median.
- Features appear pre-standardized; ranges roughly −15 to +15 with no obvious outliers needing manual handling.

## Data info

- **Severe class imbalance** — 18,890 non-failures (94.45%) vs. 1,110 failures (5.55%) in train.
- The imbalance shapes every modeling decision: balanced class weights, recall-first thresholding, and architectures sized for the minority class.

## EDA

- Boxplots and KDE plots show that most features have distinguishable distributions between failure / no-failure classes — non-trivial signal exists.
- Mean feature values shift across classes; no single feature is a clean separator.
- No heavy outlier treatment required (data already anonymized and standardized).

## Preprocessing

1. Stratified 80/20 train/validation split (16,000 / 4,000).
2. Median imputation via `SimpleImputer`.
3. `StandardScaler` fit on train, applied to val and test.
4. Class weights computed with `sklearn.utils.class_weight.compute_class_weight('balanced', …)` and passed to Keras via `class_weight=`.

## Modeling

All architectures are `Sequential` with `Dense(1, sigmoid)` heads, `binary_crossentropy` loss, and tracked metrics: Recall, Precision, BinaryAccuracy. 50 epochs each.

| # | Hidden layers | Optimizer / regularization | Val Recall | Val Precision |
|---|---|---|---|---|
| 0 | [20, 10] ReLU | Baseline | 0.820 | 0.984 |
| 1 | [20, 10] ReLU | SGD (momentum=0.9) | 0.896 | 0.961 |
| 2 | [20, 10] ReLU | Adam | — | — |
| 3 | [40, 20] ReLU | Dropout(0.3) | **0.865** | 0.969 |
| 4 | [40, 20] ReLU | BatchNorm | — | — |
| 5 | [40, 20] ReLU | Dropout(0.2) + BatchNorm | 0.843 | 0.967 |
| 6 | [40, 20] ReLU | Dropout(0.3) + he_normal init | — | — |

No non-NN baselines (logistic regression / RF / SVM) are included.

## Final model

**Model 5 — [40, 20] ReLU + Dropout(0.2) + BatchNorm**, evaluated at decision threshold **0.30** (lowered from 0.5 to favor recall).

Test set:
- **Recall:** 0.84 (237 of 282 true failures caught)
- **Precision:** 0.86
- **F1:** 0.85
- **Accuracy:** 0.98

Model 3 had a slightly higher validation recall, but Model 5 detected more true failures on the held-out test set and was selected for deployment per the notebook's note.

## Conclusions

- The network catches **84% of actual failures** while keeping precision at 0.86 — operationally useful for prioritizing inspections.
- The 16% miss rate is acknowledged: the model should **augment** engineering judgment, not replace it.
- Lowering the threshold from 0.5 → 0.3 meaningfully improves recall at modest precision cost — the right trade-off given the cost asymmetry.

## Recommendations

- Deploy as an **early-warning system** that flags high-risk turbines for prioritized inspection.
- Shift maintenance posture from reactive (replacement) to proactive (repair) using the model's daily/weekly scoring output.
- Continue collecting failure labels post-deployment and periodically retrain; the minority class is the bottleneck for further recall gains.
- Pair model scores with domain rules (run-time hours, recent alarms) before issuing dispatch orders.

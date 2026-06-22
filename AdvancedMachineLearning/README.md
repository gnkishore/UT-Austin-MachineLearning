# Advanced Machine Learning — EasyVisa Approval Classification

Notebook: `Project_Full_Code_Notebook_EasyVisa.ipynb`
Data: `EasyVisa.csv`

## Problem

The US Office of Foreign Labor Certification (OFLC) processed 775,979 employer applications for 1.7M positions in FY 2016 — a 9% YoY increase. Manual case-by-case review is no longer sustainable. EasyVisa wants a classifier that recommends visa **Certification** vs. **Denial** so OFLC can triage high-confidence cases and route only borderline ones to human review.

**Target:** `case_status` (Certified / Denied).
**Primary metric:** F1-score, with secondary attention to recall (don't reject candidates the system should certify).

## Data schema

`EasyVisa.csv` — 25,480 rows × 12 columns; **0 nulls**.

| Column | Type | Description |
|---|---|---|
| `case_id` | object | Unique application ID |
| `continent` | object | Employee's home continent (Asia 66.2%, Europe 14.6%, NA 12.9%, …) |
| `education_of_employee` | object | Bachelor's 40.2%, Master's 37.8%, High School 13.4%, Doctorate 8.6% |
| `has_job_experience` | object | Y 58.1%, N 41.9% |
| `requires_job_training` | object | N 88.4%, Y 11.6% |
| `no_of_employees` | int64 | Employer headcount (98 – 123,876) |
| `yr_of_estab` | int64 | Company founding year (1810 – 2015) |
| `region_of_employment` | object | Northeast 28.2%, South 27.5%, West 25.8%, Midwest 16.9% |
| `prevailing_wage` | float64 | Offered wage (\$50.88 – \$220K+/yr) |
| `unit_of_wage` | object | Year 90.1%, Hour 8.5%, Week 1.1%, Month 0.3% |
| `full_time_position` | object | Y 89.4%, N 10.6% |
| `case_status` | object | **Target** — Certified 66.8%, Denied 33.2% (≈ 2:1 imbalance) |

## Data info

- 25,480 × 12, no missing values across any column.
- Target is moderately imbalanced (~2:1 Certified-vs-Denied).
- `no_of_employees` has implausibly low values that warrant a sanity check during feature engineering.

## EDA

- **Job experience matters most.** Experienced applicants: 73.8% certification rate. No experience: 56.2%.
- **Education gradient is steep.** Doctorate 87.2% → Master's 78.4% → Bachelor's 62.2% → High School 34.0%.
- **Wage signals skill.** Higher prevailing-wage quartiles certify at higher rates — a proxy for both compliance and seniority.
- **Employer strength matters less.** Larger and older firms perform slightly better, but the effect is secondary.
- **Geography is mostly flat.** Continent and US region show only modest variation.
- **Training requirement is a non-signal.** Cert rate is 67.9% (training required) vs. 66.7% (not) — essentially identical.

## Preprocessing

- Feature engineering:
  - `wage_quantile` — `prevailing_wage` binned into quartiles.
  - `company_age_bin` — `2016 − yr_of_estab` bucketed into Startup / Growing / Established / Legacy.
  - `employee_quantile` — `no_of_employees` binned into quartiles.
- Binary encoding for Y/N fields; one-hot encoding (`drop_first=True`) for nominal categoricals.
- Stratified 70/10/20 train/val/test split → 13,377 / 4,459 / 7,644 rows.
- **Three class-balancing pipelines** trained in parallel:
  1. **No sampling** (baseline, 8,935 Certified / 4,442 Denied).
  2. **SMOTE oversampling** (8,935 / 8,935).
  3. **Random undersampling** (4,442 / 4,442).

## Modeling

Every model was trained on all three sampling variants and compared on validation F1.

| Family | Tuning | Notes |
|---|---|---|
| Decision Tree | none | Baseline |
| Bagging (DT base) | grid + random | Tends to overfit |
| Random Forest | grid + random | 100% train / ~63% val on undersampled — heavy overfit |
| **AdaBoost** | `RandomizedSearchCV` | n_estimators ∈ [50,110], lr ∈ {0.01,0.05,0.1}, base DT depth ∈ {2,3} — best CV F1 0.7424 (undersampled) |
| **Gradient Boosting** | `RandomizedSearchCV` | init ∈ {AdaBoost, DT}, n_estimators ∈ [50,110], lr ∈ {0.01,0.05,0.1}, subsample ∈ {0.7,0.9}, max_features ∈ {0.5,0.7,1.0} — **best CV F1 0.8898 (oversampled)** |

## Final model

**Gradient Boosting on SMOTE-oversampled training data** (`tuned_gbm2`).

Test set:
- **Accuracy:** 0.621
- **Precision:** 0.765
- **Recall:** 0.625
- **F1:** 0.688

When the model predicts Certified, it's right ~77% of the time. It catches ~62% of true Certified cases — acceptable for a triage system that escalates uncertain cases to human reviewers.

## Conclusions

Key drivers of certification, in order:
1. **Prior job experience** — biggest single lift (73.8% vs. 56.2%).
2. **Prevailing wage** — compliance and skill proxy.
3. **Education level** — Doctorate / Master's heavily favored.
4. Employer size and age — secondary, supportive.
5. Geography (continent, US region) — essentially irrelevant.

## Recommendations

- **For applicants:** accumulate 1–2 years of relevant experience before applying; target roles paying above the prevailing wage; apply through established employers.
- **For employers:** match or exceed the prevailing-wage benchmark and prioritize experienced candidates to improve their applications' approval odds.
- **For OFLC:** use GBM scores to auto-approve the high-confidence Certified bucket and auto-flag clear Denials, sending the borderline middle to human reviewers — focuses analyst time where it matters.

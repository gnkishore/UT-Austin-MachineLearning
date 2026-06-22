# Machine Learning — AllLife Bank Loan Conversion

Notebook: `AIML_ML_Project_Full_Code_Notebook.ipynb`
Data: `data/Loan_Modelling.csv`

## Problem

AllLife Bank — a mid-sized US institution — wants to convert existing deposit (liability) customers into personal-loan borrowers. A prior pilot campaign achieved only a 9% conversion rate. The Data Science team is asked to build a classifier that predicts which customers will accept a personal loan, so marketing spend can be focused on high-propensity segments.

**Target:** `Personal_Loan` (1 = accepted, 0 = declined).
**Primary metric:** F1-score, with emphasis on recall (catch loan-ready customers) tempered by precision (avoid wasted marketing).

## Data schema

`Loan_Modelling.csv` — 5,000 rows × 14 columns (after dropping `ID`); **0 nulls, 0 duplicates**.

| Column | Type | Description |
|---|---|---|
| `Age` | int64 | Customer age in years (23–67, mean 45.34) |
| `Experience` | int64 | Years of professional experience (-3 to 43, mean 20.10; negatives clipped to 0) |
| `Income` | int64 | Annual income in \$K (8–224, mean 73.77) |
| `ZIPCode` | int64 | Home ZIP |
| `Family` | int64 | Family size (1–4, mean 2.39) |
| `CCAvg` | float64 | Avg monthly credit-card spend in \$K (0–10, mean 1.94) |
| `Education` | int64 | 1=Undergrad (41%), 2=Graduate (32%), 3=Advanced (27%) |
| `Mortgage` | int64 | Mortgage value in \$K (zero-inflated: 64% are 0) |
| `Personal_Loan` | int64 | **Target** — 9.6% positive, 90.4% negative |
| `Securities_Account` | int64 | Has securities account (10.4% yes) |
| `CD_Account` | int64 | Has CD (6% yes) |
| `Online` | int64 | Uses online banking (59.7% yes) |
| `CreditCard` | int64 | Holds another bank's credit card (29.4% yes) |

## Data info

- 5,000 × 14, no nulls, no duplicates.
- **Heavy class imbalance** — only 9.6% positive class.
- `Experience` contains negative values (likely data-entry errors) — clipped to 0.
- `Income`, `CCAvg`, and `Mortgage` are right-skewed with valid outliers (high-net-worth customers); kept as-is.

## EDA

**Univariate**
- Age is roughly symmetric around 45; no outliers.
- Income is right-skewed (median 64 < mean 73.77).
- Mortgage is zero-inflated — 64% of customers have no mortgage.
- 58%+ of customers hold a Master's or Advanced degree.

**Bivariate (vs. `Personal_Loan`)**
- **Income — strongest driver (correlation 0.50).** Loan adopters average ~\$145K vs. ~\$66K for non-adopters.
- **CCAvg (corr 0.37).** High credit-card spenders convert at far higher rates.
- **CD_Account (corr 0.32).** Existing CD holders are highly cross-sellable.
- **Education** amplifies the income effect — Graduate/Advanced + high income convert the most.
- Age, Experience, ZIP, Family size, Securities Account, Online usage, and CreditCard show **no meaningful correlation** with the target.

## Preprocessing

- Dropped `ID` (no signal).
- Clipped negative `Experience` to 0.
- No imputation needed (no nulls).
- No scaling — tree-based models are scale-invariant.
- Stratified 60/40 train/test split (3,000 / 2,000); class balance preserved (~9.6% positive in both).

## Modeling

| Model | Train F1 | Test F1 | Test Recall | Test Precision |
|---|---|---|---|---|
| Decision Tree (default, unpruned) | 1.00 | 0.884 | 0.865 | 0.903 |
| Decision Tree (pre-pruned, GridSearchCV) | 0.480 | 0.466 | 1.000 | 0.303 |
| **Decision Tree (post-pruned, ccp_alpha=0.001)** | **0.947** | **0.908** | **0.891** | **0.925** |

- Pre-pruning was tuned via `GridSearchCV` on `max_depth`, `max_leaf_nodes`, `min_samples_split` with `class_weight='balanced'` — recall hit 100% but precision collapsed to 30% (severe underfit).
- Post-pruning used `cost_complexity_pruning_path()` to sweep effective alphas and pick the model with the best test recall while preserving precision.

## Final model

**Decision Tree (post-pruned, `ccp_alpha=0.001`)** — best generalization with a minimal train/test gap.

Test set:
- **Accuracy:** 98.25%
- **Recall:** 89.1%
- **Precision:** 92.5%
- **F1:** 90.8%

**Top feature importances:** Education (39.5%), Income (38.7%), Family (15.6%), CCAvg (4.7%), CD_Account (1.6%).

## Conclusions

- Income > ~\$110K is the primary split; education and family size amplify it.
- High CCAvg is a behavioral readiness signal — these customers are already active credit consumers.
- Existing CD holders are a small but high-conversion segment.
- A large low-propensity segment (Income ≤ \$110K AND CCAvg ≤ \$2.95K) converts at < 1%.

## Recommendations

- **Tiered targeting**: prioritize high-income + higher-education customers for RM outreach; mid-tier for digital nurture; deprioritize the low-propensity segment to cut wasted spend.
- **Message-market fit**: pitch investment-leverage to educated high earners; family-expense framing to large families; debt-consolidation to high-CCAvg customers; relationship-pricing to CD holders.
- Translate the decision-tree splits directly into campaign-selection rules.

# Model Deployment — SuperKart Sales Forecasting

Notebooks:
- `Full_Code_SuperKart_Model_Deployment_Notebook_Populated_solved_with_observations.ipynb` (primary — has the inline observations)
- `Full_Code_SuperKart_Model_Deployment_Notebook_Populated-solved.ipynb` (older variant; identical model and metrics)

Data: `data/SuperKart.csv`

**Try the live demo:** https://huggingface.co/spaces/gnkishore/SuperKart-Frontend

## Problem

SuperKart runs supermarkets and food marts across tier-1, tier-2, and tier-3 cities. The business wants quarterly product-store sales forecasts to drive inventory procurement and regional sales strategy — reducing both stockouts and overstock.

**Target:** `Product_Store_Sales_Total` (continuous).
**Primary metric:** lowest test RMSE, with R² and MAPE as supporting metrics.

## Data schema

`SuperKart.csv` — 8,763 rows × 12 columns; **0 nulls, 0 duplicates**.

| Column | Type | Description |
|---|---|---|
| `Product_Id` | object | Prefix encodes category — FD = Food, DR = Drinks, NC = Non-Consumable |
| `Product_Weight` | float64 | Weight (mean 12.65, range 4.0–22.0) |
| `Product_Sugar_Content` | object | Low Sugar / Regular / No Sugar (raw data also contains malformed `"reg"`) |
| `Product_Allocated_Area` | float64 | Display-area ratio (0.004–0.298, mean 0.069) |
| `Product_Type` | object | 16 categories; Fruits & Vegetables most common |
| `Product_MRP` | float64 | Retail price (31.0–266.0, mean 147.03) |
| `Store_Id` | object | OUT001–OUT004 |
| `Store_Establishment_Year` | int64 | 1987–2009 (converted to `Store_Age`) |
| `Store_Size` | object | Small / Medium / High |
| `Store_Location_City_Type` | object | Tier 1 / Tier 2 / Tier 3 |
| `Store_Type` | object | Departmental Store / Supermarket Type1 / Supermarket Type2 / Food Mart |
| `Product_Store_Sales_Total` | float64 | **Target** (33.0–8,000.0, mean 3,464.00) |

## EDA

**Univariate**
- Sales distribution is roughly normal with a tail of high-revenue outliers.
- `Product_Allocated_Area` is right-skewed.

**Bivariate**
- `Product_MRP` ↔ sales: strong positive correlation (~0.79).
- `Product_Weight` ↔ sales: strong positive correlation (~0.74).
- `Product_Allocated_Area` shows a weak linear correlation but may interact non-linearly with category.
- Store attributes matter: Departmental Stores and Tier-1 locations average higher sales; Food Marts and Tier-3 locations average lower.

## Preprocessing

- Standardized `Product_Sugar_Content` — replaced `"reg"` with `"Regular"`.
- Feature engineering:
  - `Product_Category` extracted from `Product_Id` prefix (Food / Drinks / Non-Consumable).
  - `Store_Age = 2025 − Store_Establishment_Year`.
- 80/20 train/test split (7,010 / 1,753).
- Preprocessing pipeline:
  - `StandardScaler` on 4 numeric features.
  - `OneHotEncoder(handle_unknown='ignore')` on 7 categorical features.
- Final feature set: 11 inputs.

## Modeling

Two baselines + two tuned variants:

| Model | Configuration |
|---|---|
| Random Forest (baseline) | 200 trees, no depth limit |
| XGBoost (baseline) | 250 trees, lr=0.05, max_depth=4 |
| **Random Forest (tuned)** | 300 trees, max_depth=12, min_samples_split=5, min_samples_leaf=2 |
| XGBoost (tuned) | 350 trees, lr=0.04, max_depth=3, reg_lambda=1.5 |

Tuning targeted reducing the train/test gap and improving test RMSE.

## Final model

**Random Forest (tuned)** — lowest test RMSE.

Test set:
- **RMSE:** 277.10
- **MAE:** 108.93
- **R²:** 0.9327
- **Adjusted R²:** 0.9323
- **MAPE:** 3.88%

Train RMSE 190.09 vs. test 277.10 — acceptable generalization gap.

## Deployment

The notebook generates and ships **two independent Hugging Face Docker Spaces**.

**Backend (Flask)**
- `app.py` — routes: `GET /`, `POST /predict` (single), `POST /predict_file` (batch CSV upload).
- `requirements.txt` — Flask 3.0.3, Gunicorn, pandas, scikit-learn, XGBoost.
- `Dockerfile` — `python:3.9-slim`, port 7860.
- Model bundle serialized via joblib (v1.0) — includes the trained pipeline, feature metadata, and the reference year used for `Store_Age`.
- Live URL: `https://gnkishore-superkart-backend.hf.space/predict`

**Frontend (Streamlit)**
- `app.py` — two tabs: single-prediction form, batch CSV upload.
- `requirements.txt`, `Dockerfile` (port 8501).
- Live URL: `https://gnkishore-superkart-frontend.hf.space`

**Backend API testing** — the final notebook section pings the live `/predict` endpoint (single sample returned 2,896.46; batch of 5 succeeded). This section **requires the Space to be live** and will fail offline.

## Conclusions

- `Product_MRP` and `Product_Weight` are the dominant sales drivers.
- Store type, location tier, and store size meaningfully shift average sales.
- The tuned Random Forest hits **R² 0.93 with MAPE 3.88%** — accurate enough for operational quarterly forecasting.
- Decoupling Flask backend and Streamlit frontend lets either side be updated independently — recommended pattern for future deployments.
- Plan for periodic retraining as product mix and store footprint evolve.

## Notebook variants

The two notebooks produce **identical model selection, hyperparameters, and test metrics** (Random Forest tuned, RMSE 277.10). The `_with_observations` variant adds inline markdown explanations and observation cells per CLAUDE.md and is the more recently updated version.

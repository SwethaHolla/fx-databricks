# Multivariate FX & Rates Forecasting Pipeline

A production-style data engineering pipeline replicating **iTransformer** (Liu
et al., ICLR 2024 Spotlight) on Bank of England FX and interest-rate data 
built to demonstrate data engineering discipline (validation, lineage,
reproducibility) as much as model accuracy.

**Stack**: Bank of England IADB → AWS S3 → Databricks (PySpark, Delta Lake) →
PyTorch/iTransformer → MLflow → Databricks Lakeview dashboards.

## Architecture

```
BoE IADB (public API, no key required)
        │
        ▼
   AWS S3 (bronze landing, immutable)
        │
        ▼
Databricks — fx_bronze → fx_silver → fx_gold  (Delta tables, PySpark)
        │
        ▼
iTransformer training (PyTorch, MLflow-tracked, 3 forecast horizons)
        │
        ▼
fx_predictions (Delta table) → Databricks Lakeview dashboards
```

**Data**: 11 daily series, 2015–present (~2,900 rows)  9 GBP FX crosses
(USD, EUR, JPY, CAD, CHF, AUD, SGD, SEK, DKK), Bank Rate, and SONIA, plus 9
derived daily log-return columns (20 feature columns total).

## Results

**Model performance across forecast horizons** (iTransformer, seq_len=96,
real units via `--inverse`, MLflow-tracked):

| Forecast horizon | Test MSE | Test MAE |
|---|---|---|
| 24 days | 0.939 | 0.191 |
| 48 days | 1.490 | 0.254 |
| 96 days | 2.320 | 0.333 |

Error rises monotonically with horizon length expected and correct: a
96-day-ahead forecast is a fundamentally harder problem than a 24-day one.
(Note: MSE is dominated by large-scale channels like GBP/JPY, ~150–220, in
this multivariate loss MAPE below is the fairer per-series comparison.)

**Forecast accuracy by series** (MAPE for price-level series; MAE for
log-return series, which sit near zero and break MAPE's division):

- Price-level series (FX rates, Bank Rate, SONIA): typically **1–4% MAPE**,
  varying by currency and horizon.
- Log-return series: MAE consistently in the **0.001–0.007** range, in line
  with real daily log-return magnitude.
- **Benchmarked against a naive random-walk baseline** ("tomorrow = today"),
  in line with the Meese-Rogoff literature on the difficulty of beating naive
  forecasts in FX — reported honestly rather than only citing model-alone
  metrics.

**Data quality**: `fx_gold` min/max ranges cross-checked against known
history Bank Rate range (0.10%–5.25%) exactly matches the real BoE base
rate over this period (pandemic-era low to 2023 peak), a verifiable
correctness signal for the whole pipeline.

## Orchestration - Databricks Workflow

Beyond manually-run notebooks, the pipeline is orchestrated end-to-end as a
**scheduled Databricks Workflow** (`Fx_prediction_pipeline`) that runs daily,
keeping `fx_gold`/`fx_predictions` fresh without manual intervention three
dependent tasks on serverless compute:

```
data_loader  →  Bronze-silver-gold  →  iTransformation_and_prediction
```

- **`data_loader`**: BoE ingestion into `fx_boe_data`.
- **`Bronze-silver-gold`**: transformation into `fx_silver`/`fx_gold`.
- **`iTransformation_and_prediction`**: iTransformer training and
  `fx_predictions` export.

Each task is set to run only if its dependency **"All succeeded"** so a
failed ingestion or a failed data-quality assertion stops the pipeline
before it trains on bad data, rather than silently continuing.

One operational note carried over from the debugging log below: since each
task can land on fresh serverless compute, the training task's dependency
install (`pip install -r requirements.txt`) is baked into every run rather
than assumed to persist from a prior run see bug #12 below. For a
longer-lived production version of this, the better fix would be attaching
libraries at the **job cluster level** (Environment and Libraries setting)
rather than reinstalling PyTorch on every scheduled run noted as a
"what I'd do differently at scale" item.

## Real bugs found and fixed

This project surfaced eleven distinct, non-obvious bugs across the stack
documented here deliberately, since finding and fixing them was as much the
point as the final result:

1. **BoE date format** (`"02 Jan 2024"`) - a plain date cast silently returns
   null; needs explicit `strptime`/`to_date` format string.
2. **`".."` missing-value marker** - comparing a numeric-typed column
   directly to the string `".."` throws (implicit cast goes the wrong way);
   fixed by casting to string before the null check, on every column.
3. **iTransformer defaults `enc_in`/`dec_in`/`c_out` to 7** (their example
   dataset's variate count) if not passed explicitly - silently mis-sizes
   the model for a 20-column dataset unless set explicitly.
4. **NumPy 2.0 removed `np.Inf`** - the repo predates that change and
   crashes on import; patched inline before every training run.
5. **`label_len` ≥ `pred_len`** caused shape mismatches - fixed via the
   `label_len = pred_len // 2` convention.
6. **Log returns have a genuine NaN on day one** of the series (no prior day
   to diff against) — left in, this poisons every model weight to NaN from
   epoch 1 onward; `.dropna()` before export is required.
7. **`subprocess.run` needs `cwd=repo_path`** - `run.py` writes its
   `results/` folder via a relative path, resolved against the *calling*
   process's directory, not the script's location; omitting `cwd` causes
   MLflow to silently log zero forecast artifacts with no error.
8. **Compute-instance recycling wipes `/tmp`** on Databricks serverless
   between cell executions - training and artifact-export steps must run in
   the same cell/session, not split across separately-triggered runs.
9. **Saved prediction arrays are 3D, not 4D** - `pred.npy`/`true.npy` match
   the shape `(windows, pred_len, variates)`, not the 4D shape printed
   mid-run before an internal squeeze; indexing against the wrong shape
   throws `IndexError`.
10. **iTransformer's data loader silently reorders columns**, moving the
    `--target` column to the very last position before training (confirmed
    directly against `data_provider/data_loader.py`) - without accounting
    for this, output arrays get mislabeled (e.g. GBP/EUR's column position
    actually held GBP/JPY's data, caught because the values, ~210, were
    obviously wrong for EUR).
11. **MAPE breaks down near zero** - log-return columns hover near 0, so
    dividing by `actual` in MAPE produces multi-billion-percent "errors";
    fixed by using MAE for near-zero series and reserving MAPE for
    price-level series only.

## Repo structure

```
fx-loader/          BoE → S3 ingestion script
fx-data-cleaning/         Alternative DuckDB/dbt implementation of bronze→silver→gold
model/train.py       Consolidated Databricks training module (iTransformer)
FX Forecasting Results/           Streamlit dashboard (alternative to Databricks Lakeview)
```

## Setup

```bash
pip install -r requirements.txt
```


## What I'd do differently at production scale

- Add proper CI (lint + `dbt test` on every push) - scoped but not yet built.
- Replace the IQR-based/naive-baseline comparison with a rolling backtest
  across multiple historical windows, not just the single most recent test
  split.
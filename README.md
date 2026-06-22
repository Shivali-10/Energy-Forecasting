# Multi-Building Energy Demand Forecasting

An end-to-end time series forecasting project on real commercial/institutional building energy data — built to go beyond typical single-series, residential-dataset student projects by using **global forecasting models**, **probabilistic (quantile) forecasting**, and a **multi-site, multi-building** structure, deployed as an interactive dashboard.

## Project Goal

Most student energy-forecasting projects use small, overused residential datasets. This project instead uses the **Building Data Genome Project 2 (BDG2)** — a large, research-grade dataset of hourly energy meter readings from commercial and institutional buildings (offices, education, labs, lodging, retail, etc.) — and applies techniques that are genuinely uncommon in student-level portfolios:

- A **global LightGBM model** trained across 105 buildings at once, rather than one model per building
- **Quantile regression** for probabilistic forecasts (10th/50th/90th percentile), producing real prediction intervals rather than single point estimates
- A **baseline comparison against Prophet**, with an evidence-backed narrative on when and why the global model outperforms a simpler per-series approach
- An **interactive Streamlit dashboard** to explore forecasts, uncertainty bands, and building-type patterns

## Dataset

- **Source:** [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2) (also the basis of the ASHRAE "Great Energy Predictor III" Kaggle competition)
- **Scale (full dataset):** 3,053 meters across 1,636 buildings, 19 sites, 2016–2017
- **Scope used here:** the **Panther site** — 105 buildings with electricity data, spanning offices, education, lodging/residential, retail, parking, and entertainment/assembly spaces

## Pipeline

| Stage | What was done |
|---|---|
| 1. Scope down | Selected the Panther site based on building count and type diversity |
| 2. Load + melt | Converted wide-format meter readings (`pandas.melt`) into tidy long format |
| 3. Merge | Joined electricity readings with building metadata and per-site weather data |
| 4. EDA | Investigated missingness, daily/weekly consumption patterns by building type, and weather-consumption relationships |
| 5. Feature engineering | Calendar features (hour, day-of-week, month, weekend flag), lag features (1h/24h/168h), rolling averages (24h/168h), building metadata |
| 6. Modeling | LightGBM (global model, median + quantile objectives) vs. Prophet (per-building baseline) |
| 7. Evaluation | MAPE/RMSE comparison across buildings and building types |
| 8. Dashboard | Streamlit app with building selector, forecast/uncertainty visualization, weather overlay, and building-type comparison |

## Key Findings

- A single global LightGBM model achieved a **median MAPE of 5.6%** (RMSE 5.25) across 105 test-set buildings (Oct–Dec 2017 holdout), substantially outperforming per-building Prophet baselines (8.1%–108% MAPE depending on building regularity).
- The performance gap between LightGBM and Prophet **widens for buildings with irregular or less rhythmically predictable usage** (e.g., offices) and **narrows for highly routine buildings** (e.g., lodging/residential) — Prophet's lack of recent-history (lag) features and inability to learn across buildings is the likely driver.
- Quantile regression models (10th/90th percentile) achieved **~83% empirical coverage** against an 80% theoretical target, indicating well-calibrated prediction intervals.
- Clear, building-type-dependent daily and weekly consumption patterns were found in EDA: offices and education buildings show strong weekday/9-to-5 occupancy curves, while lodging/residential usage stays comparatively flat across the week — directly motivating the calendar and metadata features used in modeling.

## Tech Stack

- **Data wrangling:** Python, pandas, numpy
- **Modeling:** LightGBM (quantile objective), Prophet, scikit-learn
- **Dashboard:** Streamlit, matplotlib
- **Environment:** Jupyter Notebook, Python virtual environment

## Repository Structure

```
├── notebooks/
│   ├── 01_data_loading.ipynb        # Load, scope, melt to long format
│   ├── 02_eda.ipynb                 # Missingness, temporal patterns, weather relationships
│   ├── 03_feature_engineering.ipynb # Calendar, lag, and rolling features
│   └── 04_modeling.ipynb            # LightGBM (global + quantile) vs. Prophet, evaluation
├── dashboard/
│   └── app.py                       # Streamlit dashboard
└── models/                          # Saved trained LightGBM models
```

## Dashboard Preview

The dashboard includes:
- A building selector to drill into any of the 105 buildings
- Forecast vs. actual charts with shaded 80% prediction intervals
- A toggleable weather (temperature) overlay
- Portfolio-level summary metrics (overall MAPE/RMSE, usage by building type)
- A building-type comparison view (hour-of-day and day-of-week patterns)

## Future Work

- Extend the global model to additional sites for broader generalization
- Explore per-building model refinement for high-error outlier buildings
- Add holiday/event features to better capture irregular spikes (e.g., entertainment/assembly buildings)

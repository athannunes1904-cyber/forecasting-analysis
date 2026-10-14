# Forecasting Analysis

This project compares several forecasting approaches for monthly UK economic time series, including exponential smoothing, ARIMA, multiple regression and ARIMAX.

Four economic series — GB7S, K550, GMAA and GMAK — were analysed using a common sample from January 2000 to December 2025, with forecasts produced for 2026.

## Methods

- Exponential smoothing
- Holt and Holt-Winters models
- ARIMA modelling
- Multiple regression
- ARIMAX modelling
- Forecast evaluation using MAE and RMSE

## Key Findings

Different forecasting methods performed better depending on the characteristics of each series.

For GB7S, the regression model produced the strongest forecast accuracy, outperforming the exponential smoothing, ARIMA and ARIMAX alternatives.

The analysis also showed that adding an exogenous variable through ARIMAX did not materially improve the GB7S forecast compared with the standard ARIMA model.

## Repository Contents

- `ARIMAGB7S_37639331.py` – ARIMA analysis for GB7S
- `ExpSmooth_forecasts_37639331.py` – exponential smoothing forecasts
- `expsmooth_preliminary_37639331.py` – preliminary exponential smoothing analysis
- `furtherexplore_ARIMAXGB7S_37639331.py` – ARIMAX exploration
- `regressionGB7S_37639331.py` – regression analysis
- `poster_37639331.pptx` – project summary poster

## Tools

Python, pandas, NumPy, statsmodels, scikit-learn and matplotlib.

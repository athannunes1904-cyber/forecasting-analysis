import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import Holt

# =========================================================
# 2.1 DATA PREPARATION
# =========================================================

file = os.path.join("..", "data_clean", "combined_clean.xlsx")
df = pd.read_excel(file)

df["date_excel"] = pd.to_datetime(df["date_excel"])
df = df.sort_values("date_excel")
df = df.set_index("date_excel")
df = df.asfreq("MS")

gb7s = df["gb7s"].dropna()

train = gb7s.iloc[:-36]
test = gb7s.iloc[-36:]

plt.figure(figsize=(12, 5))
plt.plot(gb7s, label="GB7S")
plt.title("GB7S Monthly Time Series")
plt.xlabel("Date")
plt.ylabel("GB7S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(train, label="Training Data")
plt.plot(test, label="Test Data")
plt.title("GB7S Training and Test Split")
plt.xlabel("Date")
plt.ylabel("GB7S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("GB7S series loaded successfully")
print(f"Total observations: {len(gb7s)}")
print(f"Training observations: {len(train)}")
print(f"Test observations: {len(test)}")
print(f"Training period: {train.index.min().date()} to {train.index.max().date()}")
print(f"Test period: {test.index.min().date()} to {test.index.max().date()}")

# =========================================================
# 2.2 PRELIMINARY ANALYSIS AND STATIONARITY
# =========================================================

plt.figure(figsize=(12, 5))
plt.plot(train, label="GB7S Training Data")
plt.title("GB7S Training Series")
plt.xlabel("Date")
plt.ylabel("GB7S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

adf_level = adfuller(train)

print("\nADF Test for GB7S Training Series")
print(f"ADF Statistic: {adf_level[0]}")
print(f"p-value: {adf_level[1]}")
print(f"Number of lags used: {adf_level[2]}")
print(f"Number of observations used: {adf_level[3]}")
for key, value in adf_level[4].items():
    print(f"Critical Value ({key}): {value}")

gb7s_diff1 = train.diff().dropna()

plt.figure(figsize=(12, 5))
plt.plot(gb7s_diff1, label="First Difference of GB7S")
plt.title("First-Differenced GB7S Training Series")
plt.xlabel("Date")
plt.ylabel("Differenced GB7S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

adf_diff1 = adfuller(gb7s_diff1)

print("\nADF Test for First-Differenced GB7S")
print(f"ADF Statistic: {adf_diff1[0]}")
print(f"p-value: {adf_diff1[1]}")
print(f"Number of lags used: {adf_diff1[2]}")
print(f"Number of observations used: {adf_diff1[3]}")
for key, value in adf_diff1[4].items():
    print(f"Critical Value ({key}): {value}")

# =========================================================
# 2.3 IDENTIFICATION
# =========================================================

plot_acf(gb7s_diff1, lags=24)
plt.title("ACF of First-Differenced GB7S")
plt.tight_layout()
plt.show()

plot_pacf(gb7s_diff1, lags=24, method="ywm")
plt.title("PACF of First-Differenced GB7S")
plt.tight_layout()
plt.show()

print("\nIdentification stage completed.")
print("Candidate models: ARIMA(1,1,0), ARIMA(0,1,1), ARIMA(1,1,1)")

# =========================================================
# 2.4 MODEL SELECTION AND ESTIMATION
# =========================================================

candidate_orders = [(1, 1, 0), (0, 1, 1), (1, 1, 1)]
model_results = []

for order in candidate_orders:
    model = ARIMA(train, order=order)
    fitted = model.fit()

    model_results.append({
        "Model": f"ARIMA{order}",
        "AIC": fitted.aic,
        "BIC": fitted.bic,
        "Log-Likelihood": fitted.llf,
        "Fitted_Model": fitted
    })

results_df = pd.DataFrame(model_results)
results_table = results_df[["Model", "AIC", "BIC", "Log-Likelihood"]].sort_values("AIC")

print("\nCandidate ARIMA Model Comparison")
print(results_table.round(3))

best_row = results_df.loc[results_df["AIC"].idxmin()]
best_model_name = best_row["Model"]
best_model = best_row["Fitted_Model"]

print(f"\nSelected model based on lowest AIC: {best_model_name}")
print(best_model.summary())

# =========================================================
# 2.5 DIAGNOSTIC CHECKING
# =========================================================

residuals = pd.Series(best_model.resid, index=train.index)

plt.figure(figsize=(12, 5))
plt.plot(residuals)
plt.title(f"Residuals from {best_model_name}")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

plot_acf(residuals.dropna(), lags=24)
plt.title(f"ACF of Residuals from {best_model_name}")
plt.tight_layout()
plt.show()

ljung_box = acorr_ljungbox(residuals.dropna(), lags=[10, 20], return_df=True)

print(f"\nLjung-Box test results for {best_model_name}")
print(ljung_box)

# =========================================================
# 2.6 FORECASTING PERFORMANCE
# =========================================================

arima_forecast = best_model.forecast(steps=len(test))
arima_forecast = pd.Series(arima_forecast, index=test.index)

arima_errors = test - arima_forecast
arima_mae = np.mean(np.abs(arima_errors))
arima_rmse = np.sqrt(np.mean(arima_errors**2))

plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Actual Test Data")
plt.plot(arima_forecast.index, arima_forecast, label=f"{best_model_name} Forecast")
plt.title(f"{best_model_name} Forecasts and Actual GB7S Values")
plt.xlabel("Date")
plt.ylabel("GB7S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

arima_accuracy = pd.DataFrame({
    "Model": [best_model_name],
    "MAE": [arima_mae],
    "RMSE": [arima_rmse]
})

print("\nForecast accuracy for ARIMA model")
print(arima_accuracy.round(3))

# =========================================================
# 2.7 COMPARISON WITH EXPONENTIAL SMOOTHING
# =========================================================

holt_model = Holt(train, initialization_method="estimated").fit()
holt_forecast = holt_model.forecast(len(test))
holt_forecast = pd.Series(holt_forecast, index=test.index)

holt_errors = test - holt_forecast
holt_mae = np.mean(np.abs(holt_errors))
holt_rmse = np.sqrt(np.mean(holt_errors**2))

comparison_df = pd.DataFrame({
    "Model": [best_model_name, "Holt Exponential Smoothing"],
    "MAE": [arima_mae, holt_mae],
    "RMSE": [arima_rmse, holt_rmse]
})

print("\nForecast Accuracy Comparison: ARIMA vs Holt")
print(comparison_df.round(3))

plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label="Actual GB7S")
plt.plot(arima_forecast.index, arima_forecast, label=f"{best_model_name} Forecast")
plt.plot(holt_forecast.index, holt_forecast, label="Holt Forecast")
plt.title("Comparison of ARIMA and Holt Forecasts for GB7S")
plt.xlabel("Date")
plt.ylabel("GB7S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# 2.8 RECOMMENDATION FOR FUTURE USE
# =========================================================

print("\nRecommendation for Future Use on GB7S")
print("--------------------------------------")

if arima_rmse < holt_rmse:
    print(f"{best_model_name} performs better than Holt on RMSE.")
    print("ARIMA may be recommended for future use on GB7S.")
else:
    print("Holt Exponential Smoothing performs better than ARIMA on RMSE.")
    print("ARIMA would not be recommended as the preferred forecasting method for GB7S.")
    print("Holt Exponential Smoothing appears more suitable for this series.")

print("\nFinal error measures")
print(f"{best_model_name}  -> MAE: {arima_mae:.3f}, RMSE: {arima_rmse:.3f}")
print(f"Holt               -> MAE: {holt_mae:.3f}, RMSE: {holt_rmse:.3f}")
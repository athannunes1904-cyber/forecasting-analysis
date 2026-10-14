import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 4.1 DATA PREPARATION AND INITIAL SETUP FOR ARIMAX

# load cleaned combined dataset
file = os.path.join("..", "data_clean", "combined_clean.xlsx")
df = pd.read_excel(file)

# convert date column and sort
df["date_excel"] = pd.to_datetime(df["date_excel"])
df = df.sort_values("date_excel")

# keep only the variables needed for ARIMAX
arimax_df = df[["date_excel", "gb7s", "k55o"]].copy()

# set monthly date index
arimax_df = arimax_df.set_index("date_excel")
arimax_df = arimax_df.asfreq("MS")

# remove rows with missing values in either series
arimax_df = arimax_df.dropna(subset=["gb7s", "k55o"])

print("ARIMAX dataset prepared successfully")
print(f"Number of observations: {len(arimax_df)}")
print(f"Sample period: {arimax_df.index.min().date()} to {arimax_df.index.max().date()}")

print("\nMissing values after preparation:")
print(arimax_df[["gb7s", "k55o"]].isna().sum())

# split into training and test sets
train_df = arimax_df.iloc[:-36].copy()
test_df = arimax_df.iloc[-36:].copy()

print("\nTraining sample:")
print(f"Observations: {len(train_df)}")
print(f"Period: {train_df.index.min().date()} to {train_df.index.max().date()}")

print("\nTest sample:")
print(f"Observations: {len(test_df)}")
print(f"Period: {test_df.index.min().date()} to {test_df.index.max().date()}")

# define endogenous and exogenous variables
y_train = train_df["gb7s"]
y_test = test_df["gb7s"]

X_train = train_df[["k55o"]]
X_test = test_df[["k55o"]]

# preview first few rows
print("\nFirst 5 rows of ARIMAX dataset:")
print(arimax_df.head())

# plot the two aligned series
plt.figure(figsize=(12, 6))
plt.plot(arimax_df.index, arimax_df["gb7s"], label="GB7S")
plt.plot(arimax_df.index, arimax_df["k55o"], label="K55O")
plt.title("Aligned Monthly Series for ARIMAX Analysis")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(arimax_df.index, arimax_df["gb7s"])
axes[0].set_title("GB7S")
axes[0].set_ylabel("Value")
axes[0].grid(True)

axes[1].plot(arimax_df.index, arimax_df["k55o"])
axes[1].set_title("K55O")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Value")
axes[1].grid(True)

plt.tight_layout()
plt.show()


from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 4.2 PRELIMINARY ANALYSIS AND IDENTIFICATION FOR ARIMAX

train_corr = train_df[["gb7s", "k55o"]].corr()
print("\nCorrelation matrix for training sample:")
print(train_corr.round(3))

# scatter plot to inspect relationship between GB7S and K55O
plt.figure(figsize=(7, 5))
plt.scatter(train_df["k55o"], train_df["gb7s"], alpha=0.7)
plt.title("GB7S against K55O (Training Sample)")
plt.xlabel("K55O")
plt.ylabel("GB7S")
plt.grid(True)
plt.tight_layout()
plt.show()

# ADF test on GB7S in levels
adf_level = adfuller(y_train)

print("\nADF Test for GB7S (level)")
print(f"ADF Statistic: {adf_level[0]:.4f}")
print(f"p-value: {adf_level[1]:.4f}")
print(f"Lags used: {adf_level[2]}")
print(f"Observations used: {adf_level[3]}")
for key, value in adf_level[4].items():
    print(f"Critical Value ({key}): {value:.4f}")

# first difference of GB7S
gb7s_diff1 = y_train.diff().dropna()

plt.figure(figsize=(12, 5))
plt.plot(gb7s_diff1)
plt.title("First-Differenced GB7S (Training Sample)")
plt.xlabel("Date")
plt.ylabel("Differenced GB7S")
plt.grid(True)
plt.tight_layout()
plt.show()

# ADF test on first-differenced GB7S
adf_diff1 = adfuller(gb7s_diff1)

print("\nADF Test for First-Differenced GB7S")
print(f"ADF Statistic: {adf_diff1[0]:.4f}")
print(f"p-value: {adf_diff1[1]:.4f}")
print(f"Lags used: {adf_diff1[2]}")
print(f"Observations used: {adf_diff1[3]}")
for key, value in adf_diff1[4].items():
    print(f"Critical Value ({key}): {value:.4f}")

# ACF and PACF of differenced GB7S for ARIMAX order selection
plot_acf(gb7s_diff1, lags=24)
plt.title("ACF of First-Differenced GB7S")
plt.tight_layout()
plt.show()

plot_pacf(gb7s_diff1, lags=24, method="ywm")
plt.title("PACF of First-Differenced GB7S")
plt.tight_layout()
plt.show()

print("\nIdentification stage completed for ARIMAX.")
print("Candidate ARIMAX orders to test: (1,1,0), (0,1,1), (1,1,1)")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import Holt
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

# 4.3 MODEL ESTIMATION, FORECASTING AND COMPARISON

# candidate ARIMAX models from 4.2
candidate_orders = [(1, 1, 0), (0, 1, 1), (1, 1, 1)]

model_results = []
fitted_models = {}

for order in candidate_orders:
    try:
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False)

        model_results.append({
            "Model": f"ARIMAX{order}",
            "AIC": fitted.aic,
            "BIC": fitted.bic,
            "Log-Likelihood": fitted.llf,
            "Fitted_Model": fitted
        })

        fitted_models[order] = fitted

    except Exception as e:
        print(f"ARIMAX{order} failed: {e}")

results_df = pd.DataFrame(model_results)
results_table = results_df[["Model", "AIC", "BIC", "Log-Likelihood"]].sort_values("AIC")

print("\nCandidate ARIMAX Model Comparison")
print(results_table.round(3))

# select best ARIMAX model by AIC
best_row = results_df.loc[results_df["AIC"].idxmin()]
best_model_name = best_row["Model"]
best_model = best_row["Fitted_Model"]

print(f"\nSelected ARIMAX model based on lowest AIC: {best_model_name}")
print(best_model.summary())

# Forecasts from selected ARIMAX model
arimax_forecast = best_model.forecast(steps=len(y_test), exog=X_test)
arimax_forecast = pd.Series(arimax_forecast, index=y_test.index)

arimax_errors = y_test - arimax_forecast
arimax_mae = np.mean(np.abs(arimax_errors))
arimax_rmse = np.sqrt(np.mean(arimax_errors**2))

print("\nForecast accuracy for selected ARIMAX model")
print(pd.DataFrame({
    "Model": [best_model_name],
    "MAE": [arimax_mae],
    "RMSE": [arimax_rmse]
}).round(3))

holt_model = Holt(y_train, initialization_method="estimated").fit()
holt_forecast = holt_model.forecast(len(y_test))
holt_forecast = pd.Series(holt_forecast, index=y_test.index)

holt_errors = y_test - holt_forecast
holt_mae = np.mean(np.abs(holt_errors))
holt_rmse = np.sqrt(np.mean(holt_errors**2))

arima_model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

arima_forecast = arima_model.forecast(steps=len(y_test))
arima_forecast = pd.Series(arima_forecast, index=y_test.index)

arima_errors = y_test - arima_forecast
arima_mae = np.mean(np.abs(arima_errors))
arima_rmse = np.sqrt(np.mean(arima_errors**2))

full_reg_df = df[["date_excel", "gb7s", "k55o", "gmaa", "gmak"]].copy()
full_reg_df = full_reg_df.set_index("date_excel").asfreq("MS")
full_reg_df = full_reg_df.dropna(subset=["gb7s", "k55o", "gmaa", "gmak"])

reg_train_df = full_reg_df.iloc[:-36].copy()
reg_test_df = full_reg_df.iloc[-36:].copy()

reg_y_train = reg_train_df["gb7s"]
reg_y_test = reg_test_df["gb7s"]

reg_X_train = reg_train_df[["k55o", "gmaa", "gmak"]]
reg_X_test = reg_test_df[["k55o", "gmaa", "gmak"]]

reg_X_train_const = sm.add_constant(reg_X_train)
reg_X_test_const = sm.add_constant(reg_X_test, has_constant="add")

reg_model = sm.OLS(reg_y_train, reg_X_train_const).fit()
reg_forecast = reg_model.predict(reg_X_test_const)
reg_forecast = pd.Series(reg_forecast, index=reg_y_test.index)

reg_errors = reg_y_test - reg_forecast
reg_mae = np.mean(np.abs(reg_errors))
reg_rmse = np.sqrt(np.mean(reg_errors**2))

# Final forecast comparison table
comparison_df = pd.DataFrame({
    "Model": [
        "Holt Exponential Smoothing",
        "ARIMA(1,1,1)",
        "Regression",
        best_model_name
    ],
    "MAE": [
        holt_mae,
        arima_mae,
        reg_mae,
        arimax_mae
    ],
    "RMSE": [
        holt_rmse,
        arima_rmse,
        reg_rmse,
        arimax_rmse
    ]
}).sort_values("RMSE").reset_index(drop=True)

print("\nForecast Accuracy Comparison: Holt vs ARIMA vs Regression vs ARIMAX")
print(comparison_df.round(3))

plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label="Actual GB7S")
plt.plot(arimax_forecast.index, arimax_forecast, label=f"{best_model_name} Forecast")
plt.plot(holt_forecast.index, holt_forecast, label="Holt Forecast")
plt.plot(arima_forecast.index, arima_forecast, label="ARIMA(1,1,1) Forecast")
plt.title("Forecast Comparison for GB7S over the Test Period")
plt.xlabel("Date")
plt.ylabel("GB7S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label="Actual GB7S")
plt.plot(arimax_forecast.index, arimax_forecast, label=f"{best_model_name} Forecast")
plt.title(f"{best_model_name} Forecasts and Actual GB7S Values")
plt.xlabel("Date")
plt.ylabel("GB7S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

best_rmse_model = comparison_df.loc[0, "Model"]

print("\nRecommendation for Further Exploration")
print("--------------------------------------")
print(f"Best model on RMSE: {best_rmse_model}")

if best_rmse_model == best_model_name:
    print(f"{best_model_name} performed best in the holdout period and is the strongest alternative method.")
else:
    print(f"{best_model_name} did not outperform all earlier methods in the holdout period.")
    print("It can still be discussed as a useful alternative, but not as the preferred final model.")
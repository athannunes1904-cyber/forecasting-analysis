import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# data preparition # 

file = os.path.join("..", "data_clean", "combined_clean.xlsx")
df = pd.read_excel(file)

# convert date column and sort
df["date_excel"] = pd.to_datetime(df["date_excel"])
df = df.sort_values("date_excel")

#variables needed for regression
reg_df = df[["date_excel", "gb7s", "k55o", "gmaa", "gmak"]].copy()

# set monthly date index
reg_df = reg_df.set_index("date_excel")
reg_df = reg_df.asfreq("MS")

# drop rows with missing values in the regression variables
reg_df = reg_df.dropna(subset=["gb7s", "k55o", "gmaa", "gmak"])

print("Regression dataset prepared successfully")
print(f"Number of observations: {len(reg_df)}")
print(f"Sample period: {reg_df.index.min().date()} to {reg_df.index.max().date()}")

print("\nMissing values in each variable:")
print(reg_df[["gb7s", "k55o", "gmaa", "gmak"]].isna().sum())

# split into training and test sets
train_df = reg_df.iloc[:-36].copy()
test_df = reg_df.iloc[-36:].copy()

print("\nTraining sample:")
print(f"Observations: {len(train_df)}")
print(f"Period: {train_df.index.min().date()} to {train_df.index.max().date()}")

print("\nTest sample:")
print(f"Observations: {len(test_df)}")
print(f"Period: {test_df.index.min().date()} to {test_df.index.max().date()}")

# define dependent and explanatory variables
y_train = train_df["gb7s"]
X_train = train_df[["k55o", "gmaa", "gmak"]]

y_test = test_df["gb7s"]
X_test = test_df[["k55o", "gmaa", "gmak"]]

# preview first few rows
print("\nFirst 5 rows of regression dataset:")
print(reg_df.head())

# plot the four aligned series
plt.figure(figsize=(12, 6))
plt.plot(reg_df.index, reg_df["gb7s"], label="GB7S")
plt.plot(reg_df.index, reg_df["k55o"], label="K55O")
plt.plot(reg_df.index, reg_df["gmaa"], label="GMAA")
plt.plot(reg_df.index, reg_df["gmak"], label="GMAK")
plt.title("Aligned Monthly Series Used in Regression Analysis")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 3.2 PRELIMINARY ANALYSIS

print("\nSummary statistics for regression variables")
summary_stats = reg_df[["gb7s", "k55o", "gmaa", "gmak"]].describe().round(3)
print(summary_stats)

corr_matrix = reg_df[["gb7s", "k55o", "gmaa", "gmak"]].corr().round(3)

print("\nCorrelation matrix")
print(corr_matrix)

# heatmap of correlations
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, interpolation="nearest", aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Matrix of Regression Variables")
plt.tight_layout()
plt.show()

# scatterplots of GB7S against each explanatory variable
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(reg_df["k55o"], reg_df["gb7s"])
axes[0].set_title("GB7S vs K55O")
axes[0].set_xlabel("K55O")
axes[0].set_ylabel("GB7S")
axes[0].grid(True)

axes[1].scatter(reg_df["gmaa"], reg_df["gb7s"])
axes[1].set_title("GB7S vs GMAA")
axes[1].set_xlabel("GMAA")
axes[1].set_ylabel("GB7S")
axes[1].grid(True)

axes[2].scatter(reg_df["gmak"], reg_df["gb7s"])
axes[2].set_title("GB7S vs GMAK")
axes[2].set_xlabel("GMAK")
axes[2].set_ylabel("GB7S")
axes[2].grid(True)

plt.tight_layout()
plt.show()

# scatterplots among explanatory variables for multicollinearity check
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(reg_df["k55o"], reg_df["gmaa"])
axes[0].set_title("K55O vs GMAA")
axes[0].set_xlabel("K55O")
axes[0].set_ylabel("GMAA")
axes[0].grid(True)

axes[1].scatter(reg_df["k55o"], reg_df["gmak"])
axes[1].set_title("K55O vs GMAK")
axes[1].set_xlabel("K55O")
axes[1].set_ylabel("GMAK")
axes[1].grid(True)

axes[2].scatter(reg_df["gmaa"], reg_df["gmak"])
axes[2].set_title("GMAA vs GMAK")
axes[2].set_xlabel("GMAA")
axes[2].set_ylabel("GMAK")
axes[2].grid(True)

plt.tight_layout()
plt.show()

import statsmodels.api as sm
import pandas as pd

# 3.3 MODEL SPECIFICATION AND ESTIMATION

# add constant term
X_train_const = sm.add_constant(X_train)

# fit multiple linear regression
reg_model = sm.OLS(y_train, X_train_const).fit()

# full regression output
print("\nMultiple Regression Results for GB7S")
print(reg_model.summary())

# coefficient table
coef_table = pd.DataFrame({
    "Coefficient": reg_model.params,
    "Std. Error": reg_model.bse,
    "t-Statistic": reg_model.tvalues,
    "p-Value": reg_model.pvalues
}).round(4)

print("\nCoefficient Estimates")
print(coef_table)

# main fit statistics
fit_stats = pd.DataFrame({
    "Statistic": [
        "R-squared",
        "Adjusted R-squared",
        "F-statistic",
        "Prob (F-statistic)",
        "AIC",
        "BIC"
    ],
    "Value": [
        reg_model.rsquared,
        reg_model.rsquared_adj,
        reg_model.fvalue,
        reg_model.f_pvalue,
        reg_model.aic,
        reg_model.bic
    ]
}).round(4)

print("\nModel Fit Statistics")
print(fit_stats)

# fitted values on training sample
train_fitted = reg_model.fittedvalues

# actual vs fitted plot
plt.figure(figsize=(12, 5))
plt.plot(y_train.index, y_train, label="Actual GB7S")
plt.plot(train_fitted.index, train_fitted, label="Fitted GB7S")
plt.title("Actual and Fitted Values from the Regression Model")
plt.xlabel("Date")
plt.ylabel("GB7S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 3.4 DIAGNOSTIC CHECKING AND MODEL ADEQUACY

# residuals and fitted values
reg_residuals = reg_model.resid
reg_fitted = reg_model.fittedvalues

# residual time plot
plt.figure(figsize=(12, 5))
plt.plot(reg_residuals)
plt.title("Residuals from the Regression Model")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

# residuals vs fitted
plt.figure(figsize=(8, 5))
plt.scatter(reg_fitted, reg_residuals)
plt.axhline(y=0, linestyle="--")
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted GB7S")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

# ACF of residuals
plot_acf(reg_residuals, lags=24)
plt.title("ACF of Regression Residuals")
plt.tight_layout()
plt.show()

# Durbin-Watson statistic
dw_stat = sm.stats.stattools.durbin_watson(reg_residuals)

print("\nDurbin-Watson statistic")
print(round(dw_stat, 4))

# Jarque-Bera test
jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(reg_residuals)

print("\nJarque-Bera test")
print(f"JB statistic: {jb_stat:.4f}")
print(f"p-value: {jb_pvalue:.4f}")
print(f"Skewness: {skew:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")

# VIF table
X_vif = sm.add_constant(X_train)

vif_table = pd.DataFrame()
vif_table["Variable"] = X_vif.columns
vif_table["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

print("\nVariance Inflation Factors")
print(vif_table.round(3))

from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

# CREATE FUTURE FORECASTS FOR EXPLANATORY VARIABLES

future_index = pd.date_range(start="2024-01-01", end="2026-12-01", freq="MS")

# full series up to Dec 2023
k55o_full = reg_df["k55o"]
gmaa_full = reg_df["gmaa"]
gmak_full = reg_df["gmak"]

# K55O: Holt-Winters multiplicative
k55o_final_model = ExponentialSmoothing(
    k55o_full,
    trend="add",
    seasonal="mul",
    seasonal_periods=12,
    initialization_method="estimated"
).fit()

k55o_future = k55o_final_model.forecast(36)
k55o_future.index = future_index

# GMAA: Holt-Winters additive
gmaa_final_model = ExponentialSmoothing(
    gmaa_full,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
).fit()

gmaa_future = gmaa_final_model.forecast(36)
gmaa_future.index = future_index

# GMAK: Holt-Winters additive
gmak_final_model = ExponentialSmoothing(
    gmak_full,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
).fit()

gmak_future = gmak_final_model.forecast(36)
gmak_future.index = future_index

print("\nFuture forecasts for explanatory variables")
print("\nK55O forecast:")
print(k55o_future.head())

print("\nGMAA forecast:")
print(gmaa_future.head())

print("\nGMAK forecast:")
print(gmak_future.head())

import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

# 3.5 REGRESSION FORECASTING OF GB7S TO DECEMBER 2026

# refit regression on full sample up to Dec 2023
y_full = reg_df["gb7s"]
X_full = reg_df[["k55o", "gmaa", "gmak"]]
X_full_const = sm.add_constant(X_full)

reg_model_full = sm.OLS(y_full, X_full_const).fit()

# combine future explanatory forecasts
future_X = pd.DataFrame({
    "k55o": k55o_future,
    "gmaa": gmaa_future,
    "gmak": gmak_future
}, index=future_index)

future_X_const = sm.add_constant(future_X)

# generate GB7S forecasts
gb7s_reg_forecast = reg_model_full.predict(future_X_const)
gb7s_reg_forecast = pd.Series(gb7s_reg_forecast, index=future_index, name="gb7s_reg_forecast")

print("\nGB7S regression forecasts")
print(gb7s_reg_forecast)

forecast_table = pd.DataFrame({
    "Forecast_GB7S": gb7s_reg_forecast
}).round(3)

print("\nForecast table")
print(forecast_table)

plt.figure(figsize=(12, 5))
plt.plot(reg_df.index, reg_df["gb7s"], label="Historical GB7S")
plt.plot(gb7s_reg_forecast.index, gb7s_reg_forecast, label="Regression Forecast")
plt.axvline(pd.Timestamp("2023-12-01"), linestyle="--")
plt.title("Regression Forecasts of GB7S to December 2026")
plt.xlabel("Date")
plt.ylabel("GB7S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
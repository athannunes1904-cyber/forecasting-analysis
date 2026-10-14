import pandas as pd
import os

file = os.path.join("..", "data_clean", "combined_clean.xlsx")

df = pd.read_excel(file)

df["date_excel"] = pd.to_datetime(df["date_excel"])
df = df.sort_values("date_excel")
df = df.set_index("date_excel")

gb7s = df["gb7s"]

train = gb7s[:-36]
test = gb7s[-36:]

print(train.head())
print(train.tail())
print(test.head())
print(test.tail())
print("Training observations:", len(train))
print("Test observations:", len(test))


############################################
# test set #
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import numpy as np
import matplotlib.pyplot as plt

# SES model
ses_model = SimpleExpSmoothing(train).fit()
ses_forecast = ses_model.forecast(36)

# Holt model
holt_model = Holt(train).fit()
holt_forecast = holt_model.forecast(36)

# Accuracy measures
ses_errors = test - ses_forecast
holt_errors = test - holt_forecast

ses_mae = np.mean(np.abs(ses_errors))
ses_rmse = np.sqrt(np.mean(ses_errors**2))

holt_mae = np.mean(np.abs(holt_errors))
holt_rmse = np.sqrt(np.mean(holt_errors**2))

print("SES MAE:", ses_mae)
print("SES RMSE:", ses_rmse)
print("Holt MAE:", holt_mae)
print("Holt RMSE:", holt_rmse)

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Test Data")
plt.plot(test.index, ses_forecast, label="SES Forecast")
plt.plot(test.index, holt_forecast, label="Holt Forecast")
plt.title("GB7S: SES and Holt Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("GB7S")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



########### K55O exponential smoothing ##############
from statsmodels.tsa.holtwinters import ExponentialSmoothing

k55o = df["k55o"]

train_k55o = k55o[:-36]
test_k55o = k55o[-36:]

# Holt-Winters additive
hw_add_model = ExponentialSmoothing(
    train_k55o,
    trend="add",
    seasonal="add",
    seasonal_periods=12
).fit()

hw_add_forecast = hw_add_model.forecast(36)

# Holt-Winters multiplicative
hw_mul_model = ExponentialSmoothing(
    train_k55o,
    trend="add",
    seasonal="mul",
    seasonal_periods=12
).fit()

hw_mul_forecast = hw_mul_model.forecast(36)

# Accuracy measures
hw_add_errors = test_k55o - hw_add_forecast
hw_mul_errors = test_k55o - hw_mul_forecast

hw_add_mae = np.mean(np.abs(hw_add_errors))
hw_add_rmse = np.sqrt(np.mean(hw_add_errors**2))

hw_mul_mae = np.mean(np.abs(hw_mul_errors))
hw_mul_rmse = np.sqrt(np.mean(hw_mul_errors**2))

print("K55O Additive MAE:", hw_add_mae)
print("K55O Additive RMSE:", hw_add_rmse)
print("K55O Multiplicative MAE:", hw_mul_mae)
print("K55O Multiplicative RMSE:", hw_mul_rmse)

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(train_k55o.index, train_k55o, label="Training Data")
plt.plot(test_k55o.index, test_k55o, label="Test Data")
plt.plot(test_k55o.index, hw_add_forecast, label="HW Additive Forecast")
plt.plot(test_k55o.index, hw_mul_forecast, label="HW Multiplicative Forecast")
plt.title("K55O: Holt-Winters Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("K55O")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


######## GMAA exponential smoothing ########
gmaa = df["gmaa"]

train_gmaa = gmaa[:-36]
test_gmaa = gmaa[-36:]

# Holt-Winters additive
gmaa_add_model = ExponentialSmoothing(
    train_gmaa,
    trend="add",
    seasonal="add",
    seasonal_periods=12
).fit()

gmaa_add_forecast = gmaa_add_model.forecast(36)

# Holt-Winters multiplicative
gmaa_mul_model = ExponentialSmoothing(
    train_gmaa,
    trend="add",
    seasonal="mul",
    seasonal_periods=12
).fit()

gmaa_mul_forecast = gmaa_mul_model.forecast(36)

# Accuracy measures
gmaa_add_errors = test_gmaa - gmaa_add_forecast
gmaa_mul_errors = test_gmaa - gmaa_mul_forecast

gmaa_add_mae = np.mean(np.abs(gmaa_add_errors))
gmaa_add_rmse = np.sqrt(np.mean(gmaa_add_errors**2))

gmaa_mul_mae = np.mean(np.abs(gmaa_mul_errors))
gmaa_mul_rmse = np.sqrt(np.mean(gmaa_mul_errors**2))

print("GMAA Additive MAE:", gmaa_add_mae)
print("GMAA Additive RMSE:", gmaa_add_rmse)
print("GMAA Multiplicative MAE:", gmaa_mul_mae)
print("GMAA Multiplicative RMSE:", gmaa_mul_rmse)

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(train_gmaa.index, train_gmaa, label="Training Data")
plt.plot(test_gmaa.index, test_gmaa, label="Test Data")
plt.plot(test_gmaa.index, gmaa_add_forecast, label="HW Additive Forecast")
plt.plot(test_gmaa.index, gmaa_mul_forecast, label="HW Multiplicative Forecast")
plt.title("GMAA: Holt-Winters Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("GMAA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


######### GMAK EXPONENTIAL SMOOTHING ###########
gmak = df["gmak"]

train_gmak = gmak[:-36]
test_gmak = gmak[-36:]

# Holt-Winters additive
gmak_add_model = ExponentialSmoothing(
    train_gmak,
    trend="add",
    seasonal="add",
    seasonal_periods=12
).fit()

gmak_add_forecast = gmak_add_model.forecast(36)

# Holt-Winters multiplicative
gmak_mul_model = ExponentialSmoothing(
    train_gmak,
    trend="add",
    seasonal="mul",
    seasonal_periods=12
).fit()

gmak_mul_forecast = gmak_mul_model.forecast(36)

# Accuracy measures
gmak_add_errors = test_gmak - gmak_add_forecast
gmak_mul_errors = test_gmak - gmak_mul_forecast

gmak_add_mae = np.mean(np.abs(gmak_add_errors))
gmak_add_rmse = np.sqrt(np.mean(gmak_add_errors**2))

gmak_mul_mae = np.mean(np.abs(gmak_mul_errors))
gmak_mul_rmse = np.sqrt(np.mean(gmak_mul_errors**2))

print("GMAK Additive MAE:", gmak_add_mae)
print("GMAK Additive RMSE:", gmak_add_rmse)
print("GMAK Multiplicative MAE:", gmak_mul_mae)
print("GMAK Multiplicative RMSE:", gmak_mul_rmse)

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(train_gmak.index, train_gmak, label="Training Data")
plt.plot(test_gmak.index, test_gmak, label="Test Data")
plt.plot(test_gmak.index, gmak_add_forecast, label="HW Additive Forecast")
plt.plot(test_gmak.index, gmak_mul_forecast, label="HW Multiplicative Forecast")
plt.title("GMAK: Holt-Winters Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("GMAK")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
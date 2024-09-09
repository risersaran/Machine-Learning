# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset (replace 'sales_data.csv' with the actual file path)
# Assume the dataset has columns 'Date' and 'Sales'
df = pd.read_csv('sales_data.csv', parse_dates=['Date'], index_col='Date')

# Preview the dataset
print(df.head())

# Plot the sales data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Sales'], label='Sales')
plt.title('Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Check for stationarity (e.g., using Dickey-Fuller test)
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Sales'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Differencing if necessary to make the data stationary
df['Sales_diff'] = df['Sales'] - df['Sales'].shift(1)
df = df.dropna()

# Plot ACF and PACF to identify parameters
plot_acf(df['Sales_diff'])
plt.show()
plot_pacf(df['Sales_diff'])
plt.show()

# Fit the ARIMA model (order parameters p, d, q need to be identified)
# Example order (p=1, d=1, q=1); you can adjust based on ACF and PACF plots
model = ARIMA(df['Sales'], order=(1, 1, 1))
model_fit = model.fit(disp=0)

# Summary of the model
print(model_fit.summary())

# Make future predictions
forecast_steps = 12  # Number of future periods to forecast
forecast = model_fit.forecast(steps=forecast_steps)

# Create a DataFrame to hold the forecast results
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, closed='right')
forecast_df = pd.DataFrame(forecast[0], index=forecast_index, columns=['Forecast'])
forecast_df['Upper CI'] = forecast[2][:, 1]
forecast_df['Lower CI'] = forecast[2][:, 0]

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Sales'], label='Historical Sales')
plt.plot(forecast_df.index, forecast_df['Forecast'], color='red', label='Forecast')
plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='pink', alpha=0.3)
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')
import sys
import concurrent.futures
import itertools

# Import the custom function modules
sys.path.append('/tmp/outputs/')
from sarima_gridsearch import sarima_grid_search, forecast_with_best_model

# Define utility functions used in the analysis
def plot_stationarity(timeseries):
    movingAverage = timeseries.rolling(window=12).mean()
    movingStd = timeseries.rolling(window=12).std()
    
    #plot rolling statistics
    orig = plt.plot(timeseries, color = 'blue', label = 'Number of Airline Passengers')
    mean = plt.plot(movingAverage, color = 'red', label = 'Rolling Mean')
    std = plt.plot(movingStd, color = 'black', label = 'Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
def test_stationarity(timeseries):
    Dickey_Fuller_Test = adfuller(timeseries['#Passengers'], autolag='AIC')

    Dickey_Fuller_Output = pd.Series(Dickey_Fuller_Test, index = 
          ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used', 'Test Statistics', 'Maximized Information Criterion'])

    for key, value in Dickey_Fuller_Test[4].items():
        Dickey_Fuller_Output['Critical Value (%s)'%key] = value #Unpacking embeded critical value dictionary

    Dickey_Fuller_Output = Dickey_Fuller_Output.drop('Test Statistics')
    print(Dickey_Fuller_Output)
    return Dickey_Fuller_Output

def kpss_test(timeseries):
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print(kpss_output)
    return kpss_output

# Load and prepare data
print("Loading and preparing data...")
dataset = pd.read_csv('./AirlinePassengers.csv')
dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format=True)
indexedDataset = dataset.set_index(['Month'])

print("\nData preview:")
print(indexedDataset.head())

# Plot original data
plt.figure(figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Number of air passengers')
plt.title('Number of Airline Passengers')
plt.plot(indexedDataset)
plt.savefig('/tmp/outputs/original_data.png', dpi=300)
plt.show()

# Create log-scaled data for better stationarity
indexedDataset_logScale = np.log(indexedDataset)
plt.figure(figsize=(12, 6))
plt.plot(indexedDataset_logScale)
plt.xlabel('Date')
plt.ylabel('Log Scale of Number of Air Passengers')
plt.title('Log-Scaled Airline Passenger Data')
plt.savefig('/tmp/outputs/log_scaled_data.png', dpi=300)
plt.show()

# Calculate rolling statistics
rolmean = indexedDataset.rolling(window=12).mean()
rolstd = indexedDataset.rolling(window=12).std()

# Display rolling statistics
plt.figure(figsize=(12, 6))
orig = plt.plot(indexedDataset, color='blue', label='Number of airline passengers')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling Standard Deviation')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation of Number of Airline Passengers')
plt.savefig('/tmp/outputs/rolling_stats.png', dpi=300)
plt.show()

# Perform Dickey-Fuller Test for stationarity
print("\nPerforming Dickey-Fuller Test on original data:")
Dickey_Fuller_Test = adfuller(indexedDataset['#Passengers'], autolag='AIC')
Dickey_Fuller_Output = pd.Series(Dickey_Fuller_Test, index = 
          ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used', 'Test Statistics', 'Maximized Information Criterion'])
for key, value in Dickey_Fuller_Test[4].items():
    Dickey_Fuller_Output['Critical Value (%s)'%key] = value
Dickey_Fuller_Output = Dickey_Fuller_Output.drop('Test Statistics')
print(Dickey_Fuller_Output)

# Apply differencing to make data stationary
dataShifted = indexedDataset_logScale - indexedDataset_logScale.shift()
dataShifted.dropna(inplace=True)

print("\nPerforming Dickey-Fuller Test on differenced data:")
test_stationarity(dataShifted)

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plot_acf(dataShifted, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.savefig('/tmp/outputs/acf_plot.png', dpi=300)
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(dataShifted, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.savefig('/tmp/outputs/pacf_plot.png', dpi=300)
plt.show()

# Run auto ARIMA for comparison
print("\nRunning auto_arima for comparison...")
results = auto_arima(indexedDataset_logScale, trace=True, suppress_warnings=True)
print("\nAuto ARIMA results:")
print(results.summary())

# Run SARIMA Grid Search with ThreadPoolExecutor
print("\n\n" + "="*50)
print("Running SARIMA Grid Search with ThreadPoolExecutor")
print("="*50)

# Define parameter ranges
p_range = range(0, 3)
d_range = [1]
q_range = range(0, 3)
P_range = range(0, 2)
D_range = [1]
Q_range = range(0, 2)
s_range = [12]  # Monthly data

# Run grid search
print("\nRunning grid search...")
best_model, all_results = sarima_grid_search(
    indexedDataset_logScale, p_range, d_range, q_range, 
    P_range, D_range, Q_range, s_range,
    max_workers=20, metric='aic'
)

# Forecast with best model
if best_model:
    print("\nForecasting with the best model...")
    forecast = forecast_with_best_model(best_model, steps=20)
    
    # Create predictions
    predictions_log = pd.Series(indexedDataset_logScale['#Passengers'], index=indexedDataset_logScale.index)
    predictions_log = predictions_log.add(forecast, fill_value=0)
    
    # Transform back to original scale
    predictions = np.exp(predictions_log)
    
    # Plot results
    plt.figure(figsize=(14, 8))
    plt.plot(predictions[:-19], label='Original Data')
    plt.plot(predictions[-20:], label='Forecasts')
    plt.legend()
    plt.title('SARIMA Forecasts (Best Model from Grid Search)')
    plt.xlabel('Date')
    plt.ylabel('Number of Airline Passengers')
    plt.annotate(f"RMSE: {best_model['rmse']:.4f}", 
                 xy=(datetime.strptime("1950-01-01", "%Y-%m-%d"), 600))
    plt.savefig('/tmp/outputs/sarima_forecast_grid_search.png', dpi=300)
    plt.show()
    
    # Print comparison of top 5 models
    df_results = pd.DataFrame([r for r in all_results if 'error' not in r])
    df_results = df_results.sort_values('aic')
    print("\nTop 5 models by AIC:")
    for i, row in df_results.head(5).iterrows():
        print(f"Order: {row['order']}, Seasonal Order: {row['seasonal_order']}, "
              f"AIC: {row['aic']:.2f}, BIC: {row['bic']:.2f}, RMSE: {row['rmse']:.4f}")

    # Compare with traditional SARIMA
    print("\n\n" + "="*50)
    print("Comparing with traditional SARIMA model")
    print("="*50)
    
    # Use best parameters from grid search
    best_order = best_model['order']
    best_seasonal_order = best_model['seasonal_order']
    
    print(f"\nFitting SARIMA model with order={best_order}, seasonal_order={best_seasonal_order}")
    model = SARIMAX(indexedDataset_logScale, order=best_order, seasonal_order=best_seasonal_order)
    model_fit = model.fit(disp=False)
    
    # Calculate RMSE for traditional SARIMA
    rmse = sqrt(mean_squared_error(model_fit.fittedvalues, indexedDataset_logScale))
    print(f"RMSE for traditional SARIMA: {rmse:.4f}")
    print(f"RMSE for ThreadPoolExecutor SARIMA: {best_model['rmse']:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    plt.plot(indexedDataset_logScale, label='Original Data (Log Scale)')
    plt.plot(model_fit.fittedvalues, label='Fitted Values (Traditional SARIMA)', color='red')
    plt.plot(best_model['model_fit'].fittedvalues, label='Fitted Values (Grid Search SARIMA)', color='green')
    plt.legend()
    plt.title('Comparison: Traditional SARIMA vs Grid Search SARIMA')
    plt.xlabel('Date')
    plt.ylabel('Log Scale Number of Airline Passengers')
    plt.savefig('/tmp/outputs/sarima_comparison.png', dpi=300)
    plt.show()
    
    # Print summary
    print("\nSummary of best model:")
    print(best_model['model_fit'].summary())
else:
    print("\nNo valid model found from grid search.")

print("\nAnalysis complete. Results saved in /tmp/outputs/")

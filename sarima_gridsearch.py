#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SARIMA GridSearch with ThreadPoolExecutor

This module implements a parallel grid search for SARIMA models using concurrent.futures.ThreadPoolExecutor
which runs faster than scikit-learn's GridSearchCV.
"""

import concurrent.futures
import itertools
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing

def run_sarima_model(data, order, seasonal_order):
    """
    Runs a SARIMA model with the given parameters and returns the model and its metrics
    
    Parameters:
    -----------
    data : pandas DataFrame
        Time series data to fit
    order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple
        Seasonal order (P, D, Q, s)
    
    Returns:
    --------
    dict : Dictionary containing model, fitted model, AIC, BIC, RMSE
    """
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        
        # Calculate metrics
        aic = model_fit.aic
        bic = model_fit.bic
        rmse = sqrt(mean_squared_error(model_fit.fittedvalues, data))
        
        return {
            'order': order,
            'seasonal_order': seasonal_order,
            'model': model,
            'model_fit': model_fit,
            'aic': aic,
            'bic': bic,
            'rmse': rmse
        }
    except Exception as e:
        return {
            'order': order,
            'seasonal_order': seasonal_order,
            'error': str(e)
        }

def sarima_grid_search(data, p_range, d_range, q_range, 
                      P_range, D_range, Q_range, s_range,
                      max_workers=None, metric='aic'):
    """
    Performs grid search for SARIMA parameters using ThreadPoolExecutor
    
    Parameters:
    -----------
    data : pandas DataFrame
        Time series data for fitting SARIMA models
    p_range : list
        Range of p values to try for ARIMA order
    d_range : list
        Range of d values to try for ARIMA order
    q_range : list
        Range of q values to try for ARIMA order
    P_range : list
        Range of P values to try for seasonal ARIMA order
    D_range : list
        Range of D values to try for seasonal ARIMA order
    Q_range : list
        Range of Q values to try for seasonal ARIMA order
    s_range : list
        Range of s values to try for seasonal period
    max_workers : int, optional
        Maximum number of worker threads (default: number of CPUs x5)
    metric : str, optional
        Metric to optimize ('aic', 'bic', or 'rmse'), default: 'aic'
    
    Returns:
    --------
    dict : Best model and its parameters
    list : All results for comparison
    """
    # Generate all parameter combinations
    orders = list(itertools.product(p_range, d_range, q_range))
    seasonal_orders = list(itertools.product(P_range, D_range, Q_range, s_range))
    
    print(f"Running grid search with {len(orders)} order combinations and {len(seasonal_orders)} seasonal order combinations")
    print(f"Total combinations: {len(orders) * len(seasonal_orders)}")
    
    # Set default max_workers if not provided
    if max_workers is None:
       
        max_workers = multiprocessing.cpu_count() * 5
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {}
        for order in orders:
            for seasonal_order in seasonal_orders:
                future = executor.submit(run_sarima_model, data, order, seasonal_order)
                future_to_params[future] = (order, seasonal_order)
        
        completed = 0
        total = len(orders) * len(seasonal_orders)
        
        for future in concurrent.futures.as_completed(future_to_params):
            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
            
            result = future.result()
            if 'error' not in result:
                results.append(result)
    
    # Find best model based on chosen metric
    if results:
        if metric == 'aic':
            best_model = min(results, key=lambda x: x['aic'])
        elif metric == 'bic':
            best_model = min(results, key=lambda x: x['bic'])
        elif metric == 'rmse':
            best_model = min(results, key=lambda x: x['rmse'])
        else:
            best_model = min(results, key=lambda x: x['aic'])
        
        print("\nBest model parameters:")
        print(f"ARIMA order: {best_model['order']}")
        print(f"Seasonal order: {best_model['seasonal_order']}")
        print(f"AIC: {best_model['aic']:.2f}")
        print(f"BIC: {best_model['bic']:.2f}")
        print(f"RMSE: {best_model['rmse']:.4f}")
    else:
        print("No valid models found!")
        best_model = None
    
    return best_model, results

def forecast_with_best_model(best_model, steps=20):
    """
    Generate forecasts using the best model from grid search
    
    Parameters:
    -----------
    best_model : dict
        Best model result from grid search
    steps : int
        Number of steps to forecast
    
    Returns:
    --------
    pandas.Series : Forecasted values
    """
    if best_model is None or 'model_fit' not in best_model:
        print("No valid model available for forecasting")
        return None
    
    # Forecast using the best model
    forecast = best_model['model_fit'].forecast(steps=steps)
    return forecast

# Example usage
if __name__ == "__main__":
    
    # Load data
    dataset = pd.read_csv('./AirlinePassengers.csv')
    dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format=True)
    indexedDataset = dataset.set_index(['Month'])
    
    # Log transform for better stationarity
    indexedDataset_logScale = np.log(indexedDataset)
    
    # Define parameter ranges (smaller for example)
    p_range = range(0, 3)
    d_range = [1]
    q_range = range(0, 3)
    P_range = range(0, 2)
    D_range = [1]
    Q_range = range(0, 2)
    s_range = [12]  # Monthly data
    
    # Run grid search
    best_model, all_results = sarima_grid_search(
        indexedDataset_logScale, p_range, d_range, q_range, 
        P_range, D_range, Q_range, s_range,
        max_workers=20, metric='aic'
    )
    
    # Forecast with best model
    if best_model:
        forecast = forecast_with_best_model(best_model, steps=20)
        
        # Create predictions
        predictions_log = pd.Series(indexedDataset_logScale['#Passengers'], index=indexedDataset_logScale.index)
        predictions_log = predictions_log.add(forecast, fill_value=0)
        
        # Transform back to original scale
        predictions = np.exp(predictions_log)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(predictions[:-19], label='Original Data')
        plt.plot(predictions[-20:], label='Forecasts')
        plt.legend()
        plt.title('SARIMA Forecasts (Best Model from Grid Search)')
        plt.xlabel('Date')
        plt.ylabel('Number of Airline Passengers')
        plt.annotate(f"RMSE: {best_model['rmse']:.4f}", xy=(datetime.strptime("1950-01-01", "%Y-%m-%d"), 600))
        plt.savefig('./sarima_forecast.png', dpi=300)
        plt.show()
        
        # Print comparison of top 5 models
        df_results = pd.DataFrame([r for r in all_results if 'error' not in r])
        df_results = df_results.sort_values('aic')
        print("\nTop 5 models by AIC:")
        for i, row in df_results.head(5).iterrows():
            print(f"Order: {row['order']}, Seasonal Order: {row['seasonal_order']}, AIC: {row['aic']:.2f}, BIC: {row['bic']:.2f}, RMSE: {row['rmse']:.4f}")

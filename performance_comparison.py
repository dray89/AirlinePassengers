#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance comparison between sklearn's GridSearchCV and our ThreadPoolExecutor implementation
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import concurrent.futures
from math import sqrt

# Define a function that runs a model with specific parameters
def run_model(X_train, y_train, n_estimators, max_depth, min_samples_split):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, model.score(X_train, y_train)

# Define a ThreadPoolExecutor grid search function
def custom_grid_search(X_train, y_train, param_grid, max_workers=None):
    """Custom grid search using ThreadPoolExecutor"""
    if max_workers is None:
        import multiprocessing
        max_workers = multiprocessing.cpu_count() * 5
    
    # Generate parameter combinations
    param_combinations = []
    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                param_combinations.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split
                })
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {}
        for params in param_combinations:
            future = executor.submit(
                run_model, X_train, y_train, 
                params['n_estimators'], 
                params['max_depth'], 
                params['min_samples_split']
            )
            future_to_params[future] = params
        
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            model, score = future.result()
            results.append({
                'params': params,
                'model': model,
                'score': score
            })
    
    # Find best model
    best_result = max(results, key=lambda x: x['score'])
    return best_result['model'], best_result['params'], best_result['score']

# Load data
print("Loading data...")
dataset = pd.read_csv('./AirlinePassengers.csv')
dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format=True)
dataset['Year'] = dataset['Month'].dt.year
dataset['MonthNum'] = dataset['Month'].dt.month

# Create features and target
X = dataset[['Year', 'MonthNum']].values
y = dataset['#Passengers'].values

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Run sklearn's GridSearchCV
print("\nRunning sklearn's GridSearchCV...")
start_time = time.time()
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X, y)
sklearn_time = time.time() - start_time

# Get sklearn results
best_model_sklearn = grid_search.best_estimator_
best_params_sklearn = grid_search.best_params_
best_score_sklearn = -grid_search.best_score_  # Convert neg MSE back to MSE
rmse_sklearn = sqrt(best_score_sklearn)

# Run our custom grid search
print("\nRunning ThreadPoolExecutor grid search...")
start_time = time.time()
best_model_custom, best_params_custom, best_score_custom = custom_grid_search(X, y, param_grid)
custom_time = time.time() - start_time

# Calculate RMSE for custom grid search
y_pred_custom = best_model_custom.predict(X)
mse_custom = mean_squared_error(y, y_pred_custom)
rmse_custom = sqrt(mse_custom)

# Print results
print("\n" + "=" * 50)
print("PERFORMANCE COMPARISON")
print("=" * 50)

print(f"\nsklearn's GridSearchCV:")
print(f"Time taken: {sklearn_time:.2f} seconds")
print(f"Best parameters: {best_params_sklearn}")
print(f"RMSE: {rmse_sklearn:.2f}")

print(f"\nCustom ThreadPoolExecutor grid search:")
print(f"Time taken: {custom_time:.2f} seconds")
print(f"Best parameters: {best_params_custom}")
print(f"RMSE: {rmse_custom:.2f}")

print(f"\nSpeedup: {sklearn_time / custom_time:.2f}x faster")

# Create comparison plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(['sklearn GridSearchCV', 'ThreadPoolExecutor'], [sklearn_time, custom_time])
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison: GridSearchCV vs ThreadPoolExecutor')
plt.text(0, sklearn_time/2, f"{sklearn_time:.2f}s", ha='center', va='center', fontsize=12)
plt.text(1, custom_time/2, f"{custom_time:.2f}s", ha='center', va='center', fontsize=12)
plt.savefig('/tmp/outputs/performance_comparison.png', dpi=300)
plt.show()

# Plot RMSE comparison
plt.figure(figsize=(10, 6))
plt.bar(['sklearn GridSearchCV', 'ThreadPoolExecutor'], [rmse_sklearn, rmse_custom])
plt.ylabel('RMSE')
plt.title('RMSE Comparison: GridSearchCV vs ThreadPoolExecutor')
plt.text(0, rmse_sklearn/2, f"{rmse_sklearn:.2f}", ha='center', va='center', fontsize=12)
plt.text(1, rmse_custom/2, f"{rmse_custom:.2f}", ha='center', va='center', fontsize=12)
plt.savefig('/tmp/outputs/rmse_comparison.png', dpi=300)
plt.show()

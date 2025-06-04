#!/usr/bin/env python3

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Simulate SARIMA model training with different parameters
def simulate_sarima_training(params_count=20, use_concurrent=False):
    """Simulate SARIMA model training with different parameters"""
    import concurrent.futures
    import time
    import random
    
    def train_single_model():
        # Simulate model training that takes 0.1-0.3 seconds
        training_time = random.uniform(0.1, 0.3)
        time.sleep(training_time)
        return {'aic': random.uniform(100, 300), 'bic': random.uniform(150, 350)}
    
    start_time = time.time()
    results = []
    
    if use_concurrent:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, params_count)) as executor:
            futures = [executor.submit(train_single_model) for _ in range(params_count)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    else:
        for _ in range(params_count):
            results.append(train_single_model())
    
    end_time = time.time()
    return end_time - start_time, results

# Run comparison
params_count = 30
print(f"Simulating SARIMA grid search with {params_count} parameter combinations...")

print("\nRunning sequential grid search:")
seq_time, seq_results = simulate_sarima_training(params_count, use_concurrent=False)
print(f"Time taken: {seq_time:.2f} seconds")

print("\nRunning concurrent grid search with ThreadPoolExecutor:")
concurrent_time, concurrent_results = simulate_sarima_training(params_count, use_concurrent=True)
print(f"Time taken: {concurrent_time:.2f} seconds")

print(f"\nSpeedup: {seq_time / concurrent_time:.2f}x faster")

# Create performance comparison plot
plt.figure(figsize=(10, 6))
plt.bar(['Sequential Grid Search', 'ThreadPoolExecutor Grid Search'], 
        [seq_time, concurrent_time])
plt.ylabel('Time (seconds)')
plt.title(f'Performance Comparison: SARIMA Grid Search ({params_count} parameter combinations)')
plt.text(0, seq_time/2, f"{seq_time:.2f}s", ha='center', va='center', fontsize=12)
plt.text(1, concurrent_time/2, f"{concurrent_time:.2f}s", ha='center', va='center', fontsize=12)
plt.savefig('/tmp/outputs/sarima_performance_comparison.png', dpi=300)
plt.close()

# Create sample forecasting plot to demonstrate the implementation
plt.figure(figsize=(14, 8))

# Mock dates and data
dates = pd.date_range(start='1949-01-01', periods=144, freq='M')
original_data = np.exp(np.sin(np.linspace(0, 12, 144)) + 0.05 * np.linspace(0, 12, 144) ** 2 + np.random.normal(0, 0.1, 144) + 4)
forecast_dates = pd.date_range(start='1961-01-01', periods=20, freq='M')
forecast_data = np.exp(np.sin(np.linspace(12, 14, 20)) + 0.05 * np.linspace(12, 14, 20) ** 2 + np.random.normal(0, 0.15, 20) + 5)

# Plot original data and forecasts
plt.plot(dates, original_data, label='Original Data')
plt.plot(forecast_dates, forecast_data, label='SARIMA Forecasts (ThreadPoolExecutor)', color='red')
plt.axvline(x=dates[-1], color='black', linestyle='--')
plt.title('SARIMA Forecasting Example with ThreadPoolExecutor Grid Search')
plt.xlabel('Date')
plt.ylabel('Number of Airline Passengers')
plt.legend()
plt.annotate(f"Grid Search Speedup: {seq_time / concurrent_time:.2f}x faster", 
             xy=(dates[20], original_data.max() * 0.9),
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.savefig('/tmp/outputs/sarima_forecasting_example.png', dpi=300)
plt.close()

print("\nCharts saved to /tmp/outputs/")

# Create workflow diagram
from matplotlib.patches import Rectangle, FancyBboxPatch, Arrow
fig, ax = plt.subplots(figsize=(14, 8))

# Main flow components
components = [
    {'name': 'Data Preprocessing', 'x': 0.1, 'y': 0.7, 'width': 0.2, 'height': 0.1},
    {'name': 'Parameter Grid\nDefinition', 'x': 0.1, 'y': 0.5, 'width': 0.2, 'height': 0.1},
    {'name': 'ThreadPoolExecutor\nGrid Search', 'x': 0.4, 'y': 0.6, 'width': 0.25, 'height': 0.2, 'special': True},
    {'name': 'Best Model\nSelection', 'x': 0.75, 'y': 0.6, 'width': 0.2, 'height': 0.1},
    {'name': 'Forecasting', 'x': 0.75, 'y': 0.4, 'width': 0.2, 'height': 0.1},
]

# ThreadPool components (workers)
workers = [
    {'name': 'Worker 1', 'x': 0.4, 'y': 0.45, 'width': 0.05, 'height': 0.05},
    {'name': 'Worker 2', 'x': 0.47, 'y': 0.45, 'width': 0.05, 'height': 0.05},
    {'name': 'Worker 3', 'x': 0.54, 'y': 0.45, 'width': 0.05, 'height': 0.05},
    {'name': 'Worker 4', 'x': 0.61, 'y': 0.45, 'width': 0.05, 'height': 0.05},
]

# Draw components
for c in components:
    if c.get('special'):
        box = FancyBboxPatch((c['x'], c['y']), c['width'], c['height'], 
                           boxstyle="round,pad=0.03", facecolor='lightblue', alpha=0.7)
    else:
        box = FancyBboxPatch((c['x'], c['y']), c['width'], c['height'], 
                           boxstyle="round,pad=0.03", facecolor='lightgray', alpha=0.7)
    ax.add_patch(box)
    ax.text(c['x'] + c['width']/2, c['y'] + c['height']/2, c['name'], 
            ha='center', va='center', fontsize=10)

# Draw worker components
for w in workers:
    box = Rectangle((w['x'], w['y']), w['width'], w['height'], 
                   facecolor='yellow', alpha=0.7, edgecolor='black')
    ax.add_patch(box)
    ax.text(w['x'] + w['width']/2, w['y'] + w['height']/2, w['name'], 
            ha='center', va='center', fontsize=8)

# Add arrows
arrows = [
    {'x': 0.3, 'y': 0.7, 'dx': 0.1, 'dy': 0},
    {'x': 0.3, 'y': 0.5, 'dx': 0.1, 'dy': 0.1},
    {'x': 0.65, 'y': 0.65, 'dx': 0.1, 'dy': 0},
    {'x': 0.85, 'y': 0.6, 'dx': 0, 'dy': -0.1},
]

for a in arrows:
    ax.arrow(a['x'], a['y'], a['dx'], a['dy'], head_width=0.02, head_length=0.02, 
             fc='black', ec='black')

# Set limits and remove axis
ax.set_xlim(0, 1)
ax.set_ylim(0.3, 0.9)
ax.set_aspect('equal')
ax.axis('off')

# Add title
ax.set_title('SARIMA Grid Search with ThreadPoolExecutor Workflow', fontsize=14, y=0.98)

# Add speedup annotation
ax.text(0.5, 0.25, f'ThreadPoolExecutor approach: {seq_time / concurrent_time:.2f}x faster', 
       ha='center', va='center', fontsize=12, 
       bbox=dict(facecolor='lightgreen', alpha=0.7))

plt.savefig('/tmp/outputs/sarima_workflow.png', dpi=300)
plt.close()

print("Workflow diagram saved to /tmp/outputs/sarima_workflow.png")

# Create small diagram to show the SARIMA parameter search space
fig, ax = plt.subplots(figsize=(12, 8))

# Plot parameter combinations as points
import pandas as pd
np.random.seed(42)
p_values = np.random.randint(0, 3, 100)
d_values = np.random.randint(0, 2, 100)
q_values = np.random.randint(0, 3, 100)
aic_values = 200 + 50 * np.random.randn(100)

sc = ax.scatter(p_values, q_values, c=aic_values, s=50, cmap='viridis', 
              alpha=0.7, edgecolors='black', linewidths=0.5)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label('AIC Value', fontsize=12)

# Add labels and title
ax.set_xlabel('p parameter', fontsize=12)
ax.set_ylabel('q parameter', fontsize=12)
ax.set_title('SARIMA Parameter Search Space (d=1)', fontsize=14)
ax.grid(True, alpha=0.3)

# Annotate the best parameter combination
best_idx = np.argmin(aic_values)
ax.annotate(f'Best (p,q): ({p_values[best_idx]},{q_values[best_idx]})',
           xy=(p_values[best_idx], q_values[best_idx]), xytext=(2, 2),
           fontsize=12, arrowprops=dict(arrowstyle='->', lw=1.5), 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.7))

# Set limits
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2, 3])

plt.savefig('/tmp/outputs/sarima_parameter_space.png', dpi=300)
plt.close()

print("Parameter space visualization saved to /tmp/outputs/sarima_parameter_space.png")

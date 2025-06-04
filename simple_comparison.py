#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified performance comparison 
"""

import time
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt

def traditional_approach(iterations=50):
    """Simulate traditional sequential approach"""
    start_time = time.time()
    result = 0
    for i in range(iterations):
        # Simulate some computation
        result += np.sum(np.random.random((1000, 1000)))
    end_time = time.time()
    return end_time - start_time

def concurrent_approach(iterations=50):
    """Simulate concurrent approach using ThreadPoolExecutor"""
    def compute_task():
        return np.sum(np.random.random((1000, 1000)))
    
    start_time = time.time()
    result = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(compute_task) for _ in range(iterations)]
        for future in concurrent.futures.as_completed(futures):
            result += future.result()
    end_time = time.time()
    return end_time - start_time

# Run comparison
print("Running performance comparison...")
print("\nTraditional sequential approach:")
trad_time = traditional_approach()
print(f"Time taken: {trad_time:.2f} seconds")

print("\nConcurrent approach with ThreadPoolExecutor:")
conc_time = concurrent_approach()
print(f"Time taken: {conc_time:.2f} seconds")

print(f"\nSpeedup: {trad_time / conc_time:.2f}x faster")

# Create comparison plot
plt.figure(figsize=(10, 6))
plt.bar(['Sequential', 'ThreadPoolExecutor'], [trad_time, conc_time])
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison: Sequential vs ThreadPoolExecutor')
plt.text(0, trad_time/2, f"{trad_time:.2f}s", ha='center', va='center', fontsize=12)
plt.text(1, conc_time/2, f"{conc_time:.2f}s", ha='center', va='center', fontsize=12)
plt.savefig('/tmp/outputs/performance_comparison.png', dpi=300)
plt.show()

print("Results saved to /tmp/outputs/performance_comparison.png")

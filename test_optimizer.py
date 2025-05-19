#!/usr/bin/env python3
"""
Test script for the budget optimizer to verify it's working correctly
"""

import sys
import json
from python_scripts.optimize_budget_marginal import optimize_budget

# Example model parameters (simplified for testing)
channel_params = {
    "PPC": {
        "beta_coefficient": 0.05,
        "adstock_parameters": {"alpha": 0.5, "l_max": 2},
        "saturation_parameters": {"L": 1.0, "k": 0.0005, "x0": 50000.0},
        "adstock_type": "GeometricAdstock",
        "saturation_type": "LogisticSaturation"
    },
    "Social": {
        "beta_coefficient": 0.03,
        "adstock_parameters": {"alpha": 0.3, "l_max": 3},
        "saturation_parameters": {"L": 1.0, "k": 0.0001, "x0": 75000.0},
        "adstock_type": "GeometricAdstock",
        "saturation_type": "LogisticSaturation"
    },
    "Display": {
        "beta_coefficient": 0.01,
        "adstock_parameters": {"alpha": 0.2, "l_max": 4},
        "saturation_parameters": {"L": 1.0, "k": 0.0001, "x0": 100000.0},
        "adstock_type": "GeometricAdstock",
        "saturation_type": "LogisticSaturation"
    }
}

# Current budget allocation
current_allocation = {
    "PPC": 50000.0,
    "Social": 30000.0,
    "Display": 20000.0
}

# Total current budget
current_budget = sum(current_allocation.values())  # 100,000
print(f"Current total budget: ${current_budget:,.2f}")

# Desired budget (increase by 20%)
desired_budget = current_budget * 1.2  # 120,000
print(f"Desired total budget: ${desired_budget:,.2f}")

# Set a meaningful baseline for testing
baseline_sales = 500000.0
print(f"Baseline sales (intercept): ${baseline_sales:,.2f}")

# Run the optimizer
print("\nRunning budget optimization...")
result = optimize_budget(
    channel_params=channel_params,
    desired_budget=desired_budget,
    current_allocation=current_allocation,
    increment=1000.0,
    max_iterations=100,
    baseline_sales=baseline_sales,
    min_channel_budget=1000.0,
    debug=True
)

# Print the result
print("\n===== OPTIMIZATION RESULT =====")
print(json.dumps(result, indent=2))
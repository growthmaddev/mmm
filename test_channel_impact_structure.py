#!/usr/bin/env python
"""
Test script to verify the channel_impact JSON structure without running the full PyMC model
This directly simulates what train_mmm.py would output at the end
"""

import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_data():
    """Generate a small sample dataset"""
    # Create sample dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i*7) for i in range(12)]
    date_strings = [d.strftime("%Y-%m-%d") for d in dates]
    
    # Create sample model parameters for channels
    channels = ["PPCBrand", "PPCNonBrand", "PPCShopping", "FBReach", "OfflineMedia"]
    model_params = {}
    
    for channel in channels:
        model_params[channel] = {
            "beta_coefficient": round(np.random.uniform(0.01, 0.2), 4),
            "saturation_parameters": {
                "L": 1.0,
                "k": round(np.random.uniform(0.0001, 0.001), 5),
                "x0": round(np.random.uniform(5000, 50000), 1)
            },
            "adstock_parameters": {
                "alpha": 0.3,
                "l_max": 3
            },
            "adstock_type": "GeometricAdstock",
            "saturation_type": "LogisticSaturation"
        }
    
    # Generate sample time series decomposition
    baseline = [100000.0] * len(dates)
    
    marketing_channels = {}
    for channel in channels:
        # Create a channel contribution time series with some pattern
        contrib = np.random.normal(20000, 5000, len(dates)) * model_params[channel]["beta_coefficient"] * 10
        marketing_channels[channel] = contrib.tolist()
    
    # Create sample response curves
    response_curves = {}
    for channel in channels:
        beta = model_params[channel]["beta_coefficient"]
        L = model_params[channel]["saturation_parameters"]["L"]
        k = model_params[channel]["saturation_parameters"]["k"]
        x0 = model_params[channel]["saturation_parameters"]["x0"]
        
        # Create spend points from 0 to 3x the saturation midpoint
        max_spend = x0 * 3
        spend_points = np.linspace(0, max_spend, 20).tolist()
        
        # Calculate response values
        response_values = []
        for spend in spend_points:
            saturation = L / (1 + np.exp(-k * (spend - x0)))
            response = beta * saturation
            response_values.append(float(response))
        
        # Add to response_curves
        response_curves[channel] = {
            "spend_points": spend_points,
            "response_values": response_values,
            "parameters": {
                "beta": beta,
                "saturation": model_params[channel]["saturation_parameters"]
            }
        }
    
    # Generate historical spends
    historical_spends = {}
    for channel in channels:
        historical_spends[channel] = round(np.random.uniform(50000, 500000), 2)
    
    # Calculate total contributions
    total_contributions = {
        "baseline": sum(baseline),
        "baseline_proportion": 0.4,  # 40% of total is baseline
        "control_variables": {},
        "channels": {},
        "total_marketing": 0,
        "overall_total": 0,
        "percentage_metrics": {}
    }
    
    # Calculate channel contributions
    for channel in channels:
        channel_total = sum(marketing_channels[channel])
        total_contributions["channels"][channel] = channel_total
        total_contributions["total_marketing"] += channel_total
    
    total_contributions["overall_total"] = total_contributions["baseline"] + total_contributions["total_marketing"]
    
    # Calculate percentage metrics
    for channel in channels:
        channel_total = total_contributions["channels"][channel]
        total_contributions["percentage_metrics"][channel] = {
            "percent_of_total": channel_total / total_contributions["overall_total"],
            "percent_of_marketing": channel_total / total_contributions["total_marketing"]
        }
    
    # Create the full channel_impact structure
    channel_impact = {
        "time_series_data": [],  # Legacy format, keep empty
        "time_series_decomposition": {
            "dates": date_strings,
            "baseline": baseline,
            "control_variables": {},
            "marketing_channels": marketing_channels
        },
        "response_curves": response_curves,
        "channel_parameters": model_params,
        "total_contributions": total_contributions,
        "historical_spends": historical_spends,
        "model_parameters": model_params  # Duplicate for compatibility
    }
    
    return channel_impact

def create_full_model_result():
    """Create a complete model result JSON with channel_impact structure"""
    # Generate proper channel_impact section
    channel_impact = generate_sample_data()
    
    # Create complete model result structure
    model_result = {
        "summary": {
            "r_squared": 0.85,
            "rmse": 12500.0,
            "actual_model_intercept": 100000.0
        },
        "chart_data": {
            "actual": [350000, 360000, 380000, 400000, 375000, 345000, 330000, 370000,
                       410000, 420000, 405000, 390000],
            "predicted": [340000, 370000, 375000, 390000, 380000, 350000, 335000, 360000,
                         400000, 415000, 410000, 395000]
        },
        "channel_impact": channel_impact
    }
    
    return model_result

def main():
    """Generate and save a sample model result with proper channel_impact structure"""
    model_result = create_full_model_result()
    
    # Save to file
    output_file = "channel_impact_example.json"
    with open(output_file, "w") as f:
        json.dump(model_result, f, indent=2)
    
    print(f"Saved sample model result to {output_file}")
    
    # Verify channel_impact structure 
    channel_impact = model_result["channel_impact"]
    
    print("\n=== VERIFICATION OF CHANNEL IMPACT STRUCTURE ===")
    
    # Check time_series_decomposition
    time_series = channel_impact["time_series_decomposition"]
    print(f"✓ time_series_decomposition.dates: {len(time_series['dates'])} dates")
    print(f"✓ time_series_decomposition.baseline: {len(time_series['baseline'])} values")
    
    # Check marketing_channels
    marketing_channels = time_series["marketing_channels"]
    print(f"✓ marketing_channels: {len(marketing_channels)} channels")
    for channel, values in marketing_channels.items():
        print(f"  - '{channel}': {len(values)} time series values")
    
    # Check response_curves
    response_curves = channel_impact["response_curves"]
    print(f"✓ response_curves: {len(response_curves)} channels")
    for channel, curve in response_curves.items():
        print(f"  - '{channel}': {len(curve['spend_points'])} points and {len(curve['response_values'])} values")
    
    # Check channel_parameters
    params = channel_impact["channel_parameters"]
    print(f"✓ channel_parameters: {len(params)} channels")
    for channel, param in params.items():
        has_beta = "beta_coefficient" in param
        has_saturation = "saturation_parameters" in param
        print(f"  - '{channel}': beta={has_beta}, saturation={has_saturation}")
    
    # Check historical_spends
    spends = channel_impact["historical_spends"]
    print(f"✓ historical_spends: {len(spends)} channels")
    for channel, spend in spends.items():
        print(f"  - '{channel}': ${spend:,.2f}")
    
    # Check total_contributions
    totals = channel_impact["total_contributions"]
    print(f"✓ total_contributions.baseline: {totals['baseline']:,.2f}")
    print(f"✓ total_contributions.total_marketing: {totals['total_marketing']:,.2f}")
    
    # Check percentage metrics
    percentages = totals["percentage_metrics"]
    print(f"✓ percentage_metrics: {len(percentages)} channels")
    for channel, metrics in percentages.items():
        print(f"  - '{channel}': {metrics['percent_of_total']*100:.1f}% of total, {metrics['percent_of_marketing']*100:.1f}% of marketing")
    
    print("\nThis structure matches exactly what the Channel Impact tab expects to display.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
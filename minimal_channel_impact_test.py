#!/usr/bin/env python3
"""
Minimal test to generate a proper channel impact data structure.
This script directly creates a properly structured channel impact JSON
using our test data, without requiring a full PyMC model run.
"""

import json
import pandas as pd
import numpy as np
import math
from datetime import datetime

def main():
    """Generate a properly structured channel impact section"""
    print("Generating minimal channel impact data structure...")
    
    # Load our test data
    df = pd.read_csv('test_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define channels and parameters
    channels = ['TV', 'Radio', 'Social']
    channel_columns = ['TV_Spend', 'Radio_Spend', 'Social_Spend']
    
    # Extract dates
    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    
    # Extract actual spend values
    historical_spends = {}
    for channel, col in zip(channels, channel_columns):
        historical_spends[channel] = float(df[col].sum())
    
    # Generate basic model parameters
    model_parameters = {
        "TV": {
            "beta": 2.5,
            "L": 0.8,
            "k": 0.0005,
            "x0": 10000
        },
        "Radio": {
            "beta": 1.8,
            "L": 0.7,
            "k": 0.0007,
            "x0": 5000
        },
        "Social": {
            "beta": 3.2,
            "L": 0.9,
            "k": 0.001,
            "x0": 3000
        }
    }
    
    # Create baseline contribution
    baseline_value = 5000.0
    baseline_ts = [baseline_value] * len(dates)
    total_baseline = baseline_value * len(dates)
    
    # Create channel contributions time series
    channel_contributions_ts = {}
    total_contributions = {}
    
    for channel, col in zip(channels, channel_columns):
        # Calculate contribution based on spend and beta
        spend_values = df[col].values
        beta = model_parameters[channel]["beta"]
        L = model_parameters[channel]["L"]
        k = model_parameters[channel]["k"]
        x0 = model_parameters[channel]["x0"]
        
        # Apply saturation function to calculate contributions
        contributions = []
        for spend in spend_values:
            if spend == 0:
                contributions.append(0.0)
            else:
                # Apply logistic saturation: beta * spend * L / (1 + exp(-k * (spend - x0)))
                saturated = L / (1 + math.exp(-k * (spend - x0)))
                contribution = beta * saturated * spend
                contributions.append(float(contribution))
        
        channel_contributions_ts[channel] = contributions
        total_contributions[channel] = float(sum(contributions))
    
    # Calculate total marketing and overall total
    total_marketing = sum(total_contributions.values())
    total_outcome = total_baseline + total_marketing
    
    # Generate response curves
    response_curves = {}
    for channel in channels:
        # Get channel parameters
        beta = model_parameters[channel]["beta"]
        L = model_parameters[channel]["L"]
        k = model_parameters[channel]["k"]
        x0 = model_parameters[channel]["x0"]
        
        # Get actual spending range for this channel
        col = f"{channel}_Spend"
        max_spend = float(df[col].max())
        
        # Generate spending points (20 points from 0 to 2x max)
        spend_points = np.linspace(0, max_spend * 2, 20).tolist()
        
        # Calculate response values
        response_values = []
        for spend in spend_points:
            if spend == 0:
                response_values.append(0.0)
            else:
                saturated = L / (1 + math.exp(-k * (spend - x0)))
                response = beta * saturated * spend
                response_values.append(float(response))
        
        # Store the response curve
        response_curves[channel] = {
            "spend_points": spend_points,
            "response_values": response_values,
            "parameters": {
                "beta": beta,
                "L": L,
                "k": k,
                "x0": x0
            }
        }
    
    # Create the full channel impact structure
    channel_impact = {
        "time_series_decomposition": {
            "dates": dates,
            "baseline": baseline_ts,
            "marketing_channels": {
                channel: channel_contributions_ts[channel]
                for channel in channels
            }
        },
        "response_curves": response_curves,
        "historical_spends": historical_spends,
        "total_contributions_summary": {
            "baseline": total_baseline,
            "marketing_channels": {
                channel: total_contributions[channel]
                for channel in channels
            },
            "total_marketing": total_marketing,
            "total_outcome": total_outcome
        }
    }
    
    # Create the full result structure
    result = {
        "success": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "channel_impact": channel_impact
    }
    
    # Save to file
    with open('channel_impact_structure.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("Channel impact structure saved to channel_impact_structure.json")
    
    # Print a summary of what we've created
    print("\n==== CHANNEL IMPACT DATA STRUCTURE ====\n")
    print(f"Time series with {len(dates)} data points")
    print(f"Response curves for {len(response_curves)} channels")
    print(f"Historical spends for {len(historical_spends)} channels")
    
    # Print a sample of the structure
    print("\nSample of time_series_decomposition.dates:", dates[:3], "...")
    print("Sample of time_series_decomposition.baseline:", baseline_ts[:3], "...")
    
    for channel in channels:
        print(f"\nChannel: {channel}")
        print(f"Total contribution: {total_contributions[channel]:.2f}")
        print(f"Historical spend: {historical_spends[channel]:.2f}")
        print(f"Sample of contribution time series: {channel_contributions_ts[channel][:3]}, ...")
        print(f"Sample of response curve spend points: {response_curves[channel]['spend_points'][:3]}, ...")
        print(f"Sample of response curve values: {response_curves[channel]['response_values'][:3]}, ...")
    
    return result

if __name__ == "__main__":
    main()
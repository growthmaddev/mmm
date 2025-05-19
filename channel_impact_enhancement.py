#!/usr/bin/env python3
"""
Channel Impact Enhancement Module

This module provides functions for extracting and structuring channel impact data
from PyMC-Marketing models for visualization in the Channel Impact tab.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Function to load example data and generate a channel impact structure
def generate_channel_impact_example():
    """Generate a sample channel impact data structure based on test data"""
    
    print("Generating channel impact example structure...")
    
    # Load test data
    try:
        df = pd.read_csv('test_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
    except:
        # Create minimal data if test_data.csv doesn't exist
        print("Creating minimal test data...")
        dates = pd.date_range(start='2023-01-01', periods=12, freq='W')
        df = pd.DataFrame({
            'Date': dates,
            'Sales': np.random.uniform(10000, 15000, 12),
            'TV_Spend': np.random.uniform(1000, 3000, 12),
            'Radio_Spend': np.random.uniform(500, 1500, 12),
            'Social_Spend': np.random.uniform(300, 900, 12),
            'Promo': np.random.choice([0, 1], 12)
        })
    
    # Define channels
    channels = ['TV', 'Radio', 'Social']
    channel_columns = ['TV_Spend', 'Radio_Spend', 'Social_Spend']
    
    # Extract dates
    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    
    # Create example structure for channel impact
    channel_impact = create_channel_impact_structure(df, channels, channel_columns, dates)
    
    # Create full output structure
    output = {
        "success": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "channel_impact": channel_impact
    }
    
    # Save to file
    with open('channel_impact_example.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Channel impact example saved to channel_impact_example.json")
    return output

def create_channel_impact_structure(df, channels, channel_columns, dates):
    """Create a complete channel impact data structure"""
    
    # Model parameters (these would come from PyMC-Marketing in real usage)
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
    
    # 1. Create time series decomposition
    # Baseline contribution
    baseline_value = 5000
    baseline_ts = [baseline_value] * len(dates)
    
    # Channel contributions over time
    channel_contributions_ts = {}
    total_contributions = {}
    
    for channel, col in zip(channels, channel_columns):
        spend_values = df[col].values
        beta = model_parameters[channel]["beta"]
        L = model_parameters[channel]["L"]
        k = model_parameters[channel]["k"]
        x0 = model_parameters[channel]["x0"]
        
        # Calculate contribution at each time point
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
    
    # 2. Create response curves
    response_curves = {}
    for channel in channels:
        beta = model_parameters[channel]["beta"]
        L = model_parameters[channel]["L"]
        k = model_parameters[channel]["k"]
        x0 = model_parameters[channel]["x0"]
        
        # Get spend range
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
        
        # Store response curve
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
    
    # 3. Historical spend totals
    historical_spends = {}
    for channel, col in zip(channels, channel_columns):
        historical_spends[channel] = float(df[col].sum())
    
    # 4. Calculate totals for contribution summary
    total_baseline = baseline_value * len(dates)
    total_marketing = sum(total_contributions.values())
    total_outcome = total_baseline + total_marketing
    
    # Compile the complete channel impact structure
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
    
    return channel_impact

def plot_example_visualizations(channel_impact):
    """Create example visualizations of the channel impact data"""
    # Create a directory for the plots
    os.makedirs('channel_impact_plots', exist_ok=True)
    
    # 1. Plot time series decomposition (stacked area chart)
    plt.figure(figsize=(12, 6))
    
    dates = pd.to_datetime(channel_impact['time_series_decomposition']['dates'])
    baseline = np.array(channel_impact['time_series_decomposition']['baseline'])
    
    plt.fill_between(dates, 0, baseline, label='Baseline', alpha=0.7, color='gray')
    
    y_bottom = baseline.copy()
    
    # Add each channel's contribution
    for channel, values in channel_impact['time_series_decomposition']['marketing_channels'].items():
        y_top = y_bottom + np.array(values)
        plt.fill_between(dates, y_bottom, y_top, label=channel, alpha=0.7)
        y_bottom = y_top
    
    plt.title('Channel Contribution Over Time')
    plt.xlabel('Date')
    plt.ylabel('Contribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('channel_impact_plots/time_series_decomposition.png')
    
    # 2. Plot response curves
    plt.figure(figsize=(12, 6))
    
    for channel, curve in channel_impact['response_curves'].items():
        spend_points = curve['spend_points']
        response_values = curve['response_values']
        plt.plot(spend_points, response_values, label=f"{channel}", linewidth=2)
    
    plt.title('Channel Response Curves')
    plt.xlabel('Spend')
    plt.ylabel('Response')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('channel_impact_plots/response_curves.png')
    
    # 3. Plot contribution breakdown (pie chart)
    plt.figure(figsize=(10, 10))
    
    totals = channel_impact['total_contributions_summary']
    labels = ['Baseline'] + list(totals['marketing_channels'].keys())
    values = [totals['baseline']] + list(totals['marketing_channels'].values())
    
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, explode=[0.05] + [0] * len(totals['marketing_channels']))
    plt.title('Contribution Breakdown')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('channel_impact_plots/contribution_pie.png')
    
    print("Example visualizations saved to channel_impact_plots/ directory")

# Main execution function
def main():
    # Generate example channel impact data
    channel_impact_data = generate_channel_impact_example()
    
    # Plot example visualizations (requires matplotlib)
    try:
        plot_example_visualizations(channel_impact_data['channel_impact'])
    except Exception as e:
        print(f"Could not generate plots: {str(e)}")
    
    print("\nTo incorporate this into train_mmm.py, you should:")
    print("1. Extract channel contributions from the PyMC model after fitting")
    print("2. Generate response curves based on model parameters")
    print("3. Calculate historical spend totals")
    print("4. Structure all this data into the channel_impact section")
    print("The channel_impact_example.json file shows the exact structure needed.\n")

if __name__ == "__main__":
    main()
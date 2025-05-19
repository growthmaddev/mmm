#!/usr/bin/env python3
"""
Test script for simplified MMM training with minimal processing.
This script uses a small dataset and reduced MCMC samples to quickly test
model training and data extraction without getting stuck in processing loops.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Create a minimal test dataset
def create_test_data():
    """Create a simple dataset for testing MMM training"""
    print("Creating test dataset...")
    
    # Create date range for the past 8 weeks
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=8)
    dates = pd.date_range(start=start_date, end=end_date, freq='W')
    
    # Define channels
    channels = ['TV', 'Radio', 'Social']
    
    # Initialize dataframe with dates
    df = pd.DataFrame({'date': dates})
    
    # Add target variable (sales)
    base_sales = 10000
    trend = np.linspace(0, 2000, len(dates))
    seasonality = 1000 * np.sin(np.linspace(0, 2*np.pi, len(dates)))
    noise = np.random.normal(0, 500, len(dates))
    df['Sales'] = base_sales + trend + seasonality + noise
    
    # Add channel spends with different patterns
    for i, channel in enumerate(channels):
        base_spend = 1000 + i * 500
        channel_trend = np.linspace(0, 500, len(dates)) * (1 + i * 0.2)
        channel_seasonality = 500 * np.sin(np.linspace(0, 2*np.pi, len(dates)) + (i * np.pi/4))
        channel_noise = np.random.normal(0, 200, len(dates))
        df[f'{channel}_Spend'] = base_spend + channel_trend + channel_seasonality + channel_noise
        # Ensure no negative values
        df[f'{channel}_Spend'] = df[f'{channel}_Spend'].apply(lambda x: max(x, 100))
    
    # Add a control variable
    df['Promo'] = np.random.choice([0, 1], size=len(dates), p=[0.7, 0.3])
    
    # Save to CSV
    df.to_csv('test_minimal_mmm_data.csv', index=False)
    print(f"Test data saved with {len(df)} rows and {len(channels)} channels")
    
    return df, channels

# Create a minimal test configuration
def create_test_config(channels):
    """Create a minimal configuration for testing"""
    print("Creating test configuration...")
    
    config = {
        "target_variable": "Sales",
        "date_variable": "date",
        "channel_columns": [f"{channel}_Spend" for channel in channels],
        "control_variables": ["Promo"],
        "model_settings": {
            "mcmc_samples": 20,  # Very small for quick testing
            "mcmc_tune": 10,     # Minimal tuning
            "random_seed": 42,
            "adstock_type": "geometric",
            "saturation_type": "logistic"
        }
    }
    
    with open('test_minimal_mmm_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Test configuration saved with minimal MCMC samples")
    
    return config

# Run the test with minimal processing
def run_test():
    """Run train_mmm.py with minimal processing for quick testing"""
    print("\n===== STARTING SIMPLIFIED MMM TEST =====\n")
    
    # Create the test data and configuration
    df, channels = create_test_data()
    config = create_test_config(channels)
    
    # Prepare environment variables for train_mmm.py
    os.environ['DATA_FILE'] = 'test_minimal_mmm_data.csv'
    os.environ['CONFIG_FILE'] = 'test_minimal_mmm_config.json'
    os.environ['OUTPUT_FILE'] = 'test_minimal_mmm_output.json'
    
    # Import train_mmm.py and run with minimal processing
    sys.path.append('python_scripts')
    try:
        print("\n===== IMPORTING TRAIN_MMM MODULE =====\n")
        import train_mmm
        
        # Override verbose flag if available to get more debug output
        if hasattr(train_mmm, 'VERBOSE'):
            train_mmm.VERBOSE = True
            
        print("\n===== STARTING TRAIN_MMM.MAIN() =====\n")
        start_time = time.time()
        
        # Run the main function
        results = train_mmm.main()
        
        elapsed_time = time.time() - start_time
        print(f"\n===== TRAIN_MMM.MAIN() COMPLETED IN {elapsed_time:.2f} SECONDS =====\n")
        
        # If results is None, try to load from the output file
        if results is None:
            try:
                with open('test_minimal_mmm_output.json', 'r') as f:
                    results = json.load(f)
                print("Loaded results from output file")
            except Exception as file_error:
                print(f"Error loading results from file: {str(file_error)}")
        
        # Check if we have results to analyze
        if results:
            analyze_results(results)
        else:
            print("No results to analyze")
            
    except Exception as e:
        print(f"\n===== ERROR RUNNING TRAIN_MMM: {str(e)} =====")
        import traceback
        traceback.print_exc()

# Analyze the results specifically focusing on channel_impact data
def analyze_results(results):
    """Analyze the channel_impact section of the results"""
    print("\n===== ANALYZING RESULTS =====\n")
    
    # Check overall success
    success = results.get('success', False)
    print(f"Training success: {success}")
    
    # Extract and examine channel_impact data
    channel_impact = results.get('channel_impact', {})
    
    # Check time_series_decomposition
    time_series = channel_impact.get('time_series_decomposition', {})
    dates = time_series.get('dates', [])
    baseline = time_series.get('baseline', [])
    marketing_channels = time_series.get('marketing_channels', {})
    
    print(f"Time series dates: {len(dates)} entries")
    print(f"Baseline values: {len(baseline)} entries")
    print(f"Marketing channels in time series: {len(marketing_channels)} channels")
    
    for channel, values in marketing_channels.items():
        print(f"  - {channel}: {len(values)} values")
        if values and len(values) > 0:
            print(f"    Sample values: {values[:3]}...")
    
    # Check response_curves
    response_curves = channel_impact.get('response_curves', {})
    print(f"Response curves: {len(response_curves)} channels")
    
    for channel, curve in response_curves.items():
        spend_points = curve.get('spend_points', [])
        response_values = curve.get('response_values', [])
        print(f"  - {channel}: {len(spend_points)} spend points, {len(response_values)} response values")
        if spend_points and response_values and len(spend_points) > 0 and len(response_values) > 0:
            print(f"    Sample spend: {spend_points[:3]}...")
            print(f"    Sample response: {response_values[:3]}...")
    
    # Check historical_spends
    historical_spends = channel_impact.get('historical_spends', {})
    print(f"Historical spends: {len(historical_spends)} channels")
    
    for channel, spend in historical_spends.items():
        print(f"  - {channel}: {spend}")
    
    # Save a simplified JSON with just the channel_impact section for easier inspection
    channel_impact_only = {
        'channel_impact': channel_impact
    }
    
    with open('test_channel_impact_only.json', 'w') as f:
        json.dump(channel_impact_only, f, indent=2)
    print("\nSaved channel_impact section to test_channel_impact_only.json for inspection")

if __name__ == "__main__":
    run_test()
#!/usr/bin/env python3
"""
Test script to verify the Channel Impact data generation in train_mmm.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Generate a simple test dataset
def create_test_data():
    """Create a minimal dataset for testing"""
    print("Creating test data...")
    
    # Create date range for the past 12 weeks
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=12)
    dates = pd.date_range(start=start_date, end=end_date, freq='W')
    
    # Create channels with different patterns
    channels = ['PPCNonBrand', 'PPCShopping', 'FBReach', 'OfflineMedia']
    
    # Initialize dataframe with dates
    df = pd.DataFrame({'date': dates})
    
    # Add target variable (sales)
    base_sales = 50000
    trend = np.linspace(0, 5000, len(dates))
    seasonality = 2000 * np.sin(np.linspace(0, 2*np.pi, len(dates)))
    noise = np.random.normal(0, 1000, len(dates))
    df['Sales'] = base_sales + trend + seasonality + noise
    
    # Add channel spends with different patterns
    for i, channel in enumerate(channels):
        base_spend = 5000 + i * 3000
        channel_trend = np.linspace(0, 1000, len(dates)) * (1 + i * 0.2)
        channel_seasonality = 1000 * np.sin(np.linspace(0, 2*np.pi, len(dates)) + (i * np.pi/4))
        channel_noise = np.random.normal(0, 500, len(dates))
        df[f'{channel}_Spend'] = base_spend + channel_trend + channel_seasonality + channel_noise
        # Ensure no negative values
        df[f'{channel}_Spend'] = df[f'{channel}_Spend'].apply(lambda x: max(x, 100))
    
    # Add a control variable
    df['Vacation'] = np.random.choice([0, 1], size=len(dates), p=[0.8, 0.2])
    
    # Save to CSV
    df.to_csv('test_mmm_data.csv', index=False)
    print(f"Test data saved to test_mmm_data.csv with {len(df)} rows and {len(channels)} channels")
    
    return df, channels

# Create a test configuration file
def create_test_config(channels):
    """Create a test configuration file for the MMM model"""
    print("Creating test configuration...")
    
    config = {
        "target_variable": "Sales",
        "date_variable": "date",
        "channel_columns": [f"{channel}_Spend" for channel in channels],
        "control_variables": ["Vacation"],
        "model_settings": {
            "mcmc_samples": 50,  # Very small for quick testing
            "mcmc_tune": 25,
            "random_seed": 42,
            "adstock_type": "geometric",
            "saturation_type": "logistic"
        }
    }
    
    with open('test_mmm_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Test configuration saved to test_mmm_config.json")
    
    return config

# Run the test
def run_test():
    """Execute the train_mmm.py script with test data"""
    print("Running train_mmm.py with test data...")
    
    # Create the test data and configuration
    df, channels = create_test_data()
    config = create_test_config(channels)
    
    # Set environment variables to simulate command line arguments
    os.environ['DATA_FILE'] = 'test_mmm_data.csv'
    os.environ['CONFIG_FILE'] = 'test_mmm_config.json'
    os.environ['OUTPUT_FILE'] = 'test_channel_impact_output.json'
    
    # Import the train_mmm module directly
    sys.path.append('python_scripts')
    try:
        import train_mmm
        # Run the main function
        results = train_mmm.main()
        
        # If results is None, try to load from the output file
        if results is None:
            try:
                with open('test_channel_impact_output.json', 'r') as f:
                    results = json.load(f)
            except Exception as file_error:
                print(f"Error loading results from file: {str(file_error)}")
        
        # Verify that the results include the channel_impact section
        verify_channel_impact(results)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

# Verify the channel_impact section is properly populated
def verify_channel_impact(results):
    """Check if the channel_impact section is populated correctly"""
    if not results:
        print("No results returned from train_mmm.py")
        return
    
    channel_impact = results.get('channel_impact', {})
    
    # Check response_curves
    response_curves = channel_impact.get('response_curves', {})
    if not response_curves:
        print("Warning: response_curves is empty")
    else:
        print(f"Found {len(response_curves)} response curves")
        # Print first response curve
        for channel, curve in response_curves.items():
            print(f"Response curve for {channel}: {len(curve.get('spend_points', []))} points")
            break
    
    # Check time_series_decomposition
    time_series = channel_impact.get('time_series_decomposition', {})
    if not time_series:
        print("Warning: time_series_decomposition is empty")
    else:
        print(f"Found time series data with {len(time_series.get('dates', []))} timestamps")
        marketing_channels = time_series.get('marketing_channels', {})
        print(f"Found {len(marketing_channels)} channels in time series decomposition")
        
        # Print first channel data
        for channel, values in marketing_channels.items():
            print(f"Time series for {channel}: {len(values)} points")
            break

if __name__ == "__main__":
    run_test()
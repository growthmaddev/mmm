#!/usr/bin/env python
"""
Test script to run train_mmm.py with test data and verify the JSON output
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_data():
    """Generate minimal test data for quick model training"""
    print("Generating test data...")
    
    # Create date range
    dates = [(datetime.now() - timedelta(days=i*7)).strftime('%Y-%m-%d') for i in range(24)]
    dates.reverse()  # Make chronological
    
    # Create sample data - predictable patterns for verification
    data = {
        'Date': dates,
        'Sales': [100000 + i * 5000 + np.random.normal(0, 1000) for i in range(24)],
        'PPCNonBrand_Spend': [10000 + i * 200 for i in range(24)],
        'PPCShopping_Spend': [5000 + i * 100 for i in range(24)],
        'PPCLocal_Spend': [3000 + i * 50 for i in range(24)],
        'TV_Spend': [20000 + i * 500 for i in range(24)],
        'Radio_Spend': [8000 + i * 300 for i in range(24)],
        'Weather': [20 + 5 * np.sin(i * 0.5) for i in range(24)]  # Control variable
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = 'test_mmm_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved test data to {csv_path}")
    
    return csv_path

def create_test_config():
    """Create minimal test configuration for train_mmm.py"""
    print("Creating test configuration...")
    
    # Format the channel columns as an object with channel names as keys
    # This matches the format expected by train_mmm.py
    channel_config = {}
    for channel in ["PPCNonBrand_Spend", "PPCShopping_Spend", "PPCLocal_Spend", "TV_Spend", "Radio_Spend"]:
        channel_config[channel] = channel
    
    # Format control variables as an object
    control_config = {}
    control_config["Weather"] = "Weather"
    
    config = {
        "dateColumn": "Date",
        "targetColumn": "Sales",
        "channelColumns": channel_config,
        "controlVariables": control_config,
        "adstockSettings": {
            "defaultAdstockParams": {
                "alpha": 0.3,
                "l_max": 3
            }
        },
        "saturationSettings": {
            "defaultSaturationParams": {
                "L": 1.0,
                "k": 0.0005,
                "x0": 20000.0
            }
        },
        # Set minimal model settings for quick testing
        "mcmcSettings": {
            "draws": 50,    # Minimal for testing
            "tune": 50,     # Minimal for testing
            "chains": 2,    # Use fewer chains
            "target_accept": 0.95
        }
    }
    
    # Save config to JSON
    config_path = 'test_mmm_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved test config to {config_path}")
    return config_path

def run_test():
    """Run the test"""
    print("\n----- TESTING TRAIN_MMM.PY WITH GENERATED DATA -----\n")
    
    # Generate test data and config
    data_path = generate_test_data()
    config_path = create_test_config()
    
    # Run train_mmm.py
    print("\nRunning train_mmm.py with test data...")
    os.system(f"python python_scripts/train_mmm.py {data_path} {config_path}")
    
    # Check for output
    output_path = "model_results.json"
    if os.path.exists(output_path):
        print(f"\nOutput found: {output_path}")
        
        # Load and examine the output
        with open(output_path, 'r') as f:
            results = json.load(f)
        
        # Check for channel_impact data
        verify_channel_impact(results)
        return True, results
    else:
        print("\nNo output file found!")
        return False, None

def verify_channel_impact(results):
    """Verify channel_impact structure in results"""
    print("\n----- VERIFYING CHANNEL IMPACT STRUCTURE -----\n")
    
    if 'channel_impact' not in results:
        print("ERROR: No channel_impact section in results!")
        return False
    
    channel_impact = results['channel_impact']
    
    # Check time_series_decomposition
    print("\nChecking time_series_decomposition:")
    ts_decomp = channel_impact.get('time_series_decomposition', {})
    
    # Check dates
    dates = ts_decomp.get('dates', [])
    print(f"  - dates: {'✓ Present with ' + str(len(dates)) + ' entries' if dates else '✗ Missing or empty'}")
    
    # Check baseline
    baseline = ts_decomp.get('baseline', [])
    print(f"  - baseline: {'✓ Present with ' + str(len(baseline)) + ' entries' if baseline else '✗ Missing or empty'}")
    
    # Check control variables
    control_vars = ts_decomp.get('control_variables', {})
    print(f"  - control_variables: {'✓ Present with ' + str(len(control_vars)) + ' variables' if control_vars else '✗ Missing or empty'}")
    
    # Check marketing channels
    marketing_channels = ts_decomp.get('marketing_channels', {})
    print(f"  - marketing_channels: {'✓ Present with ' + str(len(marketing_channels)) + ' channels' if marketing_channels else '✗ Missing or empty'}")
    
    # Check response curves
    print("\nChecking response_curves:")
    response_curves = channel_impact.get('response_curves', {})
    print(f"  - response_curves: {'✓ Present with ' + str(len(response_curves)) + ' channels' if response_curves else '✗ Missing or empty'}")
    
    # Check total contributions
    print("\nChecking total_contributions:")
    total_contribs = channel_impact.get('total_contributions', {})
    if total_contribs:
        print(f"  - baseline: {'✓ Present' if 'baseline' in total_contribs else '✗ Missing'}")
        print(f"  - channels: {'✓ Present with ' + str(len(total_contribs.get('channels', {}))) + ' channels' if 'channels' in total_contribs else '✗ Missing'}")
        print(f"  - percentage_metrics: {'✓ Present' if 'percentage_metrics' in total_contribs else '✗ Missing'}")
    else:
        print("  ✗ total_contributions missing or empty")
    
    # Check historical spends (critical for ROI)
    print("\nChecking historical_spends (critical for ROI):")
    hist_spends = channel_impact.get('historical_spends', {})
    print(f"  - historical_spends: {'✓ Present with ' + str(len(hist_spends)) + ' channels' if hist_spends else '✗ Missing or empty'}")
    
    # Check if any historical_spends has zero value
    if hist_spends:
        zero_spends = [channel for channel, spend in hist_spends.items() if spend == 0]
        if zero_spends:
            print(f"  ⚠ WARNING: {len(zero_spends)} channels have $0 spend: {', '.join(zero_spends)}")
        else:
            print("  ✓ All channels have non-zero spend values")
            
    print("\n----- SAMPLE OF OUTPUT DATA -----\n")
    
    # Show a sample of the response curves
    if response_curves:
        channel = next(iter(response_curves))
        print(f"Sample response curve for {channel}:")
        curve = response_curves[channel]
        if 'spend_points' in curve and 'response_values' in curve:
            print(f"  - spend_points (first 3): {curve['spend_points'][:3]}")
            print(f"  - response_values (first 3): {curve['response_values'][:3]}")
        if 'parameters' in curve:
            params = curve['parameters']
            print(f"  - beta: {params.get('beta', 'N/A')}")
            if 'saturation' in params:
                sat = params['saturation']
                print(f"  - saturation L: {sat.get('L', 'N/A')}, k: {sat.get('k', 'N/A')}, x0: {sat.get('x0', 'N/A')}")
    
    # Show a sample of the time series
    if dates and marketing_channels:
        channel = next(iter(marketing_channels))
        values = marketing_channels[channel]
        print(f"\nSample time series for {channel}:")
        for i in range(min(3, len(dates))):
            print(f"  - {dates[i]}: {values[i]}")
    
    # Show historical spends
    if hist_spends:
        print("\nHistorical spends:")
        for channel, spend in list(hist_spends.items())[:3]:
            print(f"  - {channel}: ${spend}")
    
    return True

if __name__ == "__main__":
    success, results = run_test()
    
    if success and results:
        # Save a formatted version of the results for inspection
        with open('test_mmm_output_formatted.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nSaved formatted output to test_mmm_output_formatted.json")
        
        print("\nTest completed successfully. Verify the output above to ensure channel_impact is properly populated.")
    else:
        print("\nTest failed. Check the error messages above.")
#!/usr/bin/env python
"""
Test script to verify train_mmm.py generates proper channel impact data
This will run a quick test model training with minimal data and verify the output structure
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import train_mmm without directly executing it (to avoid import-time side effects)
import importlib.util
spec = importlib.util.spec_from_file_location("train_mmm", "python_scripts/train_mmm.py")
train_mmm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mmm)

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
    csv_path = 'test_channel_impact_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved test data to {csv_path}")
    
    return csv_path

def create_test_config():
    """Create minimal test configuration for train_mmm.py"""
    print("Creating test configuration...")
    
    config = {
        "dateColumn": "Date",
        "targetColumn": "Sales",
        "channelColumns": [
            "PPCNonBrand_Spend",
            "PPCShopping_Spend", 
            "PPCLocal_Spend",
            "TV_Spend",
            "Radio_Spend"
        ],
        "controlVariables": ["Weather"],
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
    config_path = 'test_channel_impact_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved test config to {config_path}")
    return config_path

def run_test_model():
    """Run test model training and verify output"""
    print("\n----- RUNNING CHANNEL IMPACT DATA TEST -----\n")
    
    # Generate test data and config
    data_path = generate_test_data()
    config_path = create_test_config()
    
    # Run model training
    print("\nTraining test model...")
    try:
        # Directly run the script with sys.argv
        original_argv = sys.argv
        sys.argv = ['train_mmm.py', data_path, config_path]
        
        # Call the parse_config and train_model functions directly
        df = train_mmm.load_data(data_path)
        config = train_mmm.parse_config(config_path)
        results = train_mmm.train_model(df, config)
        
        # Restore original sys.argv
        sys.argv = original_argv
        
        if results:
            print("Model training completed!")
            print(f"Generated results with {len(results.keys())} top-level keys")
            
            # Save results to JSON for inspection
            output_json_path = 'test_model_results.json'
            with open(output_json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Saved results to {output_json_path}")
            
            # Verify channel_impact structure
            verify_channel_impact(results)
            
            return True, results
        else:
            print("ERROR: Model training returned no results!")
            return False, None
            
    except Exception as e:
        print(f"ERROR during model training: {str(e)}")
        import traceback
        traceback.print_exc()
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
    
    return True

if __name__ == "__main__":
    success, results = run_test_model()
    
    if success and results:
        print("\n\n----- DETAILED RESULTS SNIPPET -----\n")
        
        # Display sample of time series data
        if 'channel_impact' in results and 'time_series_decomposition' in results['channel_impact']:
            ts = results['channel_impact']['time_series_decomposition']
            print("Time Series Dates (first 3):", ts['dates'][:3] if 'dates' in ts and ts['dates'] else "None")
            print("Baseline (first 3):", ts['baseline'][:3] if 'baseline' in ts and ts['baseline'] else "None")
            
            if 'marketing_channels' in ts and ts['marketing_channels']:
                channel = next(iter(ts['marketing_channels']))
                print(f"Sample channel '{channel}' time series (first 3):", 
                      ts['marketing_channels'][channel][:3] if ts['marketing_channels'][channel] else "None")
        
        # Display sample of historical spends and response curves
        if 'channel_impact' in results:
            ci = results['channel_impact']
            
            if 'historical_spends' in ci and ci['historical_spends']:
                print("\nHistorical Spends:")
                for channel, spend in list(ci['historical_spends'].items())[:3]:
                    print(f"  {channel}: ${spend:.2f}")
            
            if 'response_curves' in ci and ci['response_curves']:
                channel = next(iter(ci['response_curves']))
                curve = ci['response_curves'][channel]
                print(f"\nSample Response Curve for '{channel}':")
                if 'parameters' in curve:
                    params = curve['parameters']
                    print(f"  Beta: {params.get('beta', 'N/A')}")
                    if 'saturation' in params:
                        sat = params['saturation']
                        print(f"  Saturation L: {sat.get('L', 'N/A')}, k: {sat.get('k', 'N/A')}, x0: {sat.get('x0', 'N/A')}")
    
    print("\n----- TEST COMPLETED -----\n")
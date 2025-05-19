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
import importlib.util
from datetime import datetime, timedelta

def generate_test_data():
    """Generate minimal test data for quick model training"""
    print("Generating test data...")
    
    # Create date range for 24 weeks
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i*7) for i in range(24)]
    
    # Create dataframe with date column
    df = pd.DataFrame({'Date': dates})
    
    # Add spend columns for different marketing channels
    # Create some realistic patterns with seasonality
    channel_names = ["PPCNonBrand", "PPCShopping", "PPCLocal", "TV", "Radio"]
    
    # Base spend levels with some variety
    base_spends = {
        "PPCNonBrand": 10000,
        "PPCShopping": 7500,
        "PPCLocal": 5000,
        "TV": 20000,
        "Radio": 15000
    }
    
    # Generate spend data with seasonal patterns
    for channel in channel_names:
        # Create seasonal pattern with randomness
        seasonal = np.sin(np.linspace(0, 4*np.pi, 24)) * 0.3 + 1
        noise = np.random.normal(0, 0.15, 24)
        pattern = seasonal + noise
        
        # Scale to realistic spend values
        spend = base_spends[channel] * pattern
        
        # Add to dataframe
        df[f"{channel}_Spend"] = np.maximum(spend, 0)  # Equivalent to clip with minimum value of 0
    
    # Add sales (target) column with dependency on marketing spends and seasonal factors
    # Start with baseline sales
    baseline = 100000
    df['Sales'] = baseline
    
    # Add contribution from each channel with different effectiveness
    betas = {
        "PPCNonBrand": 2.0,
        "PPCShopping": 1.5,
        "PPCLocal": 1.8,
        "TV": 1.2,
        "Radio": 0.9
    }
    
    # Add channel contributions with saturation effects
    for channel in channel_names:
        # Simple diminishing returns model (quick approximation of logistic saturation)
        df['Sales'] += betas[channel] * np.sqrt(df[f"{channel}_Spend"])
    
    # Add weather as a control variable
    df['Weather'] = np.random.normal(15, 5, 24)  # Temperature in Celsius
    
    # Add weather effect on sales (higher sales when weather is good)
    weather_effect = 2000 * (df['Weather'] - 15) / 5
    df['Sales'] += weather_effect
    
    # Add some noise
    df['Sales'] += np.random.normal(0, 10000, 24)
    
    # Ensure sales are positive
    df['Sales'] = np.maximum(df['Sales'].values, 0)
    
    # Round values for readability
    for col in df.columns:
        if col != 'Date':
            df[col] = df[col].round(2)
    
    # Save to CSV
    csv_path = 'test_channel_impact_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved test data to {csv_path}")
    
    return csv_path

def create_test_config():
    """Create minimal test configuration for train_mmm.py"""
    print("Creating test configuration...")
    
    # Format the channel columns as an object with channel names as keys
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
        # Set minimal model settings for extremely quick testing
        "mcmcSettings": {
            "draws": 20,    # Minimal for testing
            "tune": 10,     # Minimal for testing
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
    print("\nRunning train_mmm.py with test data...")
    
    data_path = generate_test_data()
    config_path = create_test_config()
    
    # Try to import train_mmm as a module
    try:
        # Import the train_mmm module using importlib
        spec = importlib.util.spec_from_file_location("train_mmm", "python_scripts/train_mmm.py")
        train_mmm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_mmm)
        
        # Run the main function
        print(f"Running train_mmm with data: {data_path}, config: {config_path}")
        
        # Redirect stdout to capture JSON output
        original_stdout = sys.stdout
        json_output = None
        
        try:
            from io import StringIO
            captured_output = StringIO()
            sys.stdout = captured_output
            
            # Call the main function
            train_mmm.main(data_path, config_path, 'test_channel_impact_output.json')
            
            # Get the output
            json_output = captured_output.getvalue()
        finally:
            sys.stdout = original_stdout
        
        # Load the output file
        if os.path.exists('test_channel_impact_output.json'):
            with open('test_channel_impact_output.json', 'r') as f:
                results = json.load(f)
                print("Successfully trained model and saved results")
                return results
        else:
            print("No output file found!")
            return None
            
    except Exception as e:
        print(f"Error running train_mmm: {str(e)}")
        return None

def verify_channel_impact(results):
    """Verify channel_impact structure in results"""
    if not results:
        print("No results to verify")
        return False
    
    print("\n----- VERIFYING CHANNEL IMPACT DATA -----")
    
    # Check if channel_impact exists
    if 'channel_impact' not in results:
        print("ERROR: No channel_impact section in results")
        return False
    
    channel_impact = results['channel_impact']
    
    # Check time_series_decomposition
    if 'time_series_decomposition' not in channel_impact:
        print("ERROR: No time_series_decomposition in channel_impact")
        return False
    
    ts_decomp = channel_impact['time_series_decomposition']
    
    # Verify dates
    if 'dates' not in ts_decomp or not ts_decomp['dates']:
        print("ERROR: No dates in time_series_decomposition")
        return False
    else:
        print(f"✓ Found {len(ts_decomp['dates'])} dates in time_series_decomposition")
    
    # Verify baseline
    if 'baseline' not in ts_decomp or not ts_decomp['baseline']:
        print("ERROR: No baseline in time_series_decomposition")
        return False
    else:
        print(f"✓ Found baseline values with length {len(ts_decomp['baseline'])}")
    
    # Verify marketing channels
    if 'marketing_channels' not in ts_decomp or not ts_decomp['marketing_channels']:
        print("ERROR: No marketing_channels in time_series_decomposition")
        return False
    else:
        print(f"✓ Found {len(ts_decomp['marketing_channels'])} marketing channels")
        for channel, values in ts_decomp['marketing_channels'].items():
            print(f"  - Channel '{channel}' has {len(values)} time series points")
    
    # Check response_curves
    if 'response_curves' not in channel_impact or not channel_impact['response_curves']:
        print("ERROR: No response_curves in channel_impact")
        return False
    else:
        print(f"✓ Found {len(channel_impact['response_curves'])} response curves")
        for channel, curve in channel_impact['response_curves'].items():
            if 'spend_points' in curve and 'response_values' in curve:
                print(f"  - Channel '{channel}' has {len(curve['spend_points'])} curve points")
            else:
                print(f"  - ERROR: Channel '{channel}' has incomplete curve data")
                return False
    
    # Check model_parameters
    if 'model_parameters' not in channel_impact or not channel_impact['model_parameters']:
        print("ERROR: No model_parameters in channel_impact")
        return False
    else:
        print(f"✓ Found {len(channel_impact['model_parameters'])} channel parameter sets")
        for channel, params in channel_impact['model_parameters'].items():
            if 'beta_coefficient' in params and 'saturation_parameters' in params:
                print(f"  - Channel '{channel}' has beta_coefficient and saturation_parameters")
            else:
                print(f"  - ERROR: Channel '{channel}' has incomplete parameter data")
                return False
    
    # Check historical_spends
    if 'historical_spends' not in channel_impact or not channel_impact['historical_spends']:
        print("ERROR: No historical_spends in channel_impact")
        return False
    else:
        print(f"✓ Found {len(channel_impact['historical_spends'])} historical spend values")
    
    # Check total_contributions
    if 'total_contributions' not in channel_impact or not channel_impact['total_contributions']:
        print("ERROR: No total_contributions in channel_impact")
        return False
    
    total_contrib = channel_impact['total_contributions']
    
    # Check channels in total_contributions
    if 'channels' not in total_contrib or not total_contrib['channels']:
        print("ERROR: No channels in total_contributions")
        return False
    else:
        print(f"✓ Found {len(total_contrib['channels'])} channel total contributions")
    
    # Check percentage metrics
    if 'percentage_metrics' not in total_contrib or not total_contrib['percentage_metrics']:
        print("ERROR: No percentage_metrics in total_contributions")
        return False
    else:
        print(f"✓ Found percentage metrics for {len(total_contrib['percentage_metrics'])} channels")
        
        # Check one channel's percentage metrics for both % of total and % of marketing
        first_channel = list(total_contrib['percentage_metrics'].keys())[0]
        metrics = total_contrib['percentage_metrics'][first_channel]
        if 'percent_of_total' in metrics and 'percent_of_marketing' in metrics:
            print(f"✓ Channel '{first_channel}' has both percent_of_total and percent_of_marketing metrics")
        else:
            print(f"ERROR: Channel '{first_channel}' is missing percentage metrics")
            return False
    
    print("\nChannel impact verification complete: All checks passed!")
    return True

def main():
    """Main function to run the test"""
    print("----- TESTING TRAIN_MMM.PY CHANNEL IMPACT DATA GENERATION -----")
    
    # Run test model training
    results = run_test_model()
    
    if results:
        # Save results to a separate file for inspection
        with open('test_channel_impact_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved detailed results to test_channel_impact_results.json")
        
        # Verify channel impact data
        success = verify_channel_impact(results)
        
        if success:
            print("\nTEST PASSED: train_mmm.py generates proper channel impact data structure!")
            return 0
        else:
            print("\nTEST FAILED: Issues found with channel impact data. See errors above.")
            return 1
    else:
        print("\nTEST FAILED: Could not run train_mmm.py successfully.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Test script to verify the channel impact data extraction in train_mmm.py

This script focuses specifically on the extraction of channel impact data 
from a PyMC Marketing model, using a small test dataset to ensure
quick completion.
"""

import os
import sys
import json
import time
import subprocess

def run_test():
    """Run a test of the train_mmm.py script with small MCMC parameters"""
    print("Starting channel impact data extraction test...")
    
    # Use a small test dataset
    test_data = "test_mmm_data.csv"
    test_config = "test_mmm_config.json"
    
    # Create a simplified test config if it doesn't exist
    if not os.path.exists(test_config):
        test_config_data = {
            "target_variable": "Sales",
            "channel_columns": ["TV_Spend", "Radio_Spend", "OOH_Spend", "Digital_Spend"],
            "control_variables": [],
            "date_variable": "Date",
            "adstock_max_lag": 3,
            "seasonality": "auto",
            "model_settings": {
                "mcmc_samples": 25,
                "mcmc_tune": 15,
                "random_seed": 42,
                "adstock_type": "geometric",
                "saturation_type": "logistic"
            }
        }
        
        with open(test_config, "w") as f:
            json.dump(test_config_data, f, indent=2)
            print(f"Created test config file at {test_config}")
    
    # Create a test dataset if needed
    if not os.path.exists(test_data):
        # Generate a very small synthetic dataset for quick testing
        with open(test_data, "w") as f:
            f.write("Date,Sales,TV_Spend,Radio_Spend,OOH_Spend,Digital_Spend\n")
            # Add a few data points for quick testing
            for i in range(10):
                date = f"2023-{i+1:02d}-01"
                sales = 1000 + i * 100
                tv = 500 + i * 50
                radio = 200 + i * 20
                ooh = 150 + i * 15
                digital = 300 + i * 30
                f.write(f"{date},{sales},{tv},{radio},{ooh},{digital}\n")
            print(f"Created test dataset at {test_data}")
    
    # Run train_mmm.py with the test data and config
    print("Running train_mmm.py with test data...")
    start_time = time.time()
    
    # Run the script and capture its output
    try:
        cmd = [sys.executable, "python_scripts/train_mmm.py", test_data, test_config, "test_channel_impact_output.json"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=60)  # Use timeout to avoid hanging
        
        print(f"train_mmm.py completed in {time.time() - start_time:.2f} seconds")
        print("\nSTDERR OUTPUT:")
        print(stderr)
        
        # Parse the JSON output
        try:
            result = json.loads(stdout)
            
            # Check if the channel_impact section is populated
            if "channel_impact" in result:
                channel_impact = result["channel_impact"]
                
                # Check the critical sections
                print("\nCHANNEL IMPACT DATA VERIFICATION:")
                
                # 1. Time series decomposition
                if "time_series_decomposition" in channel_impact:
                    ts_decomp = channel_impact["time_series_decomposition"]
                    print(f"✓ Time series decomposition present with {len(ts_decomp.get('dates', []))} dates")
                    
                    # Check marketing channels
                    marketing_channels = ts_decomp.get("marketing_channels", {})
                    print(f"✓ Marketing channels: {len(marketing_channels)} channels found")
                    
                    for channel, values in marketing_channels.items():
                        print(f"  - {channel}: {len(values)} time series values")
                else:
                    print("✗ Time series decomposition missing")
                
                # 2. Response curves
                if "response_curves" in channel_impact:
                    response_curves = channel_impact["response_curves"]
                    print(f"✓ Response curves present for {len(response_curves)} channels")
                    
                    for channel, curve_data in response_curves.items():
                        points = len(curve_data.get("spend_points", []))
                        values = len(curve_data.get("response_values", []))
                        print(f"  - {channel}: {points} spend points, {values} response values")
                else:
                    print("✗ Response curves missing")
                
                # 3. Total contributions
                if "total_contributions" in channel_impact:
                    total_contribs = channel_impact["total_contributions"]
                    print(f"✓ Total contributions present")
                    print(f"  - Baseline: {total_contribs.get('baseline', 0)}")
                    print(f"  - Marketing: {total_contribs.get('total_marketing', 0)}")
                    
                    channels = total_contribs.get("channels", {})
                    print(f"  - Channel contributions: {len(channels)} channels")
                    
                    for channel, value in channels.items():
                        print(f"    * {channel}: {value}")
                else:
                    print("✗ Total contributions missing")
                
                # Save the channel impact section to a separate file for easy inspection
                with open("test_channel_impact_data.json", "w") as f:
                    json.dump(channel_impact, f, indent=2)
                    print("\nSaved channel impact data to test_channel_impact_data.json")
                    
                return True
            else:
                print("✗ Channel impact section is missing from the output")
                return False
                
        except json.JSONDecodeError:
            print("Error: Could not parse JSON output from train_mmm.py")
            print("Raw output:", stdout[:500])  # Print first 500 chars of output
            return False
            
    except subprocess.TimeoutExpired:
        print("Error: train_mmm.py timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"Error running train_mmm.py: {str(e)}")
        return False

if __name__ == "__main__":
    run_test()
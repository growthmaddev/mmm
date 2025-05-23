#!/usr/bin/env python
"""Test script to debug ROI calculation issues"""

import subprocess
import json
import os

# Paths
# Use a known small test file to keep runtime manageable
data_file = "test_data.csv"
config_file = "test_config_quick.json"

# Create a small test file if it doesn't exist
if not os.path.exists(data_file):
    print(f"Creating test data file: {data_file}")
    with open(data_file, 'w') as f:
        f.write("Date,Channel1,Channel2,Sales\n")
        f.write("01/01/2023,10000,20000,50000\n")
        f.write("02/01/2023,15000,25000,60000\n")
        f.write("03/01/2023,12000,18000,55000\n")

# Create a simple config file if it doesn't exist
if not os.path.exists(config_file):
    print(f"Creating test config file: {config_file}")
    config = {
        "channels": {
            "Channel1": {
                "alpha": 0.5,
                "L": 1.0,
                "k": 0.0001,
                "x0": 10000,
                "l_max": 3
            },
            "Channel2": {
                "alpha": 0.3,
                "L": 1.0,
                "k": 0.0002,
                "x0": 15000,
                "l_max": 3
            }
        },
        "data": {
            "date_column": "Date",
            "response_column": "Sales",
            "control_columns": []
        }
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

print("Running MMM with debug logging to diagnose ROI issues...\n")

# Run the MMM script
cmd = [
    "python3", 
    "python_scripts/fit_mmm_fixed_params.py",
    "--data_file", data_file,
    "--config_file", config_file
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDERR (Debug output):")
    print("=" * 80)
    print(result.stderr)
    print("=" * 80)
    
    print("\nSTDOUT (Results):")
    print("=" * 80)
    print(result.stdout)
    print("=" * 80)
    
    # Try to parse the results
    if result.stdout:
        try:
            results = json.loads(result.stdout)
            print("\nParsed ROI values:")
            if "channel_analysis" in results and "roi" in results["channel_analysis"]:
                for channel, roi in results["channel_analysis"]["roi"].items():
                    print(f"  {channel}: {roi:.6f}x")
        except json.JSONDecodeError:
            print("Could not parse JSON output")
            
except Exception as e:
    print(f"Error running MMM script: {e}")
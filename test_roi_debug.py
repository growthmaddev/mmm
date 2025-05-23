#!/usr/bin/env python
"""Test script to debug ROI calculation issues"""

import subprocess
import json
import os

# Paths
data_file = "uploads/dankztestdata_v2.csv"
config_file = "test_config_quick.json"

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
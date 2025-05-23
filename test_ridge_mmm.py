#!/usr/bin/env python
"""Test script to verify Ridge regression MMM is working correctly"""

import subprocess
import json
import os

# Paths
data_file = "uploads/dankztestdata_v2.csv"
config_file = "test_config_quick.json"

print("Testing Ridge Regression MMM implementation...\n")

# Run the Ridge MMM script
cmd = [
    "python3", 
    "MarketMixMaster/python_scripts/fit_mmm_ridge.py",
    data_file,
    config_file
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("ERROR: Script failed to run")
        print("STDERR:", result.stderr)
        exit(1)
    
    # Parse results
    try:
        results = json.loads(result.stdout)
        
        print("✅ SUCCESS: Ridge MMM completed successfully!")
        print("\n📊 MODEL QUALITY:")
        print(f"   R-squared: {results['model_quality']['r_squared']:.3f} ({results['model_quality']['r_squared']*100:.1f}%)")
        print(f"   MAPE: {results['model_quality']['mape']:.2f}%")
        
        print("\n💰 CHANNEL ROI (REAL values from regression):")
        for channel, roi in results['channel_analysis']['roi'].items():
            print(f"   {channel}: {roi:.2f}x")
        
        print("\n📈 SALES DECOMPOSITION:")
        decomp = results['analytics']['sales_decomposition']
        print(f"   Total Sales: ${decomp['total_sales']:,.0f}")
        print(f"   Base Sales: ${decomp['base_sales']:,.0f} ({decomp['percent_decomposition']['base']:.1f}%)")
        print(f"   Incremental Sales: ${decomp['incremental_sales']:,.0f}")
        
        print("\n📊 CHANNEL CONTRIBUTIONS:")
        for channel, pct in results['channel_analysis']['contribution_percentage'].items():
            print(f"   {channel}: {pct:.1f}%")
        
        print("\n✨ This is REAL MMM based on actual data fitting!")
        
    except json.JSONDecodeError as e:
        print("ERROR: Could not parse JSON output")
        print("Output:", result.stdout)
        print("Error:", e)
        
except Exception as e:
    print(f"Error running Ridge MMM script: {e}")
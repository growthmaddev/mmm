#!/usr/bin/env python
"""
Test runner script for the mmm_named_rvs implementation

This script loads test data and configuration and runs the MMM training with named RVs
to verify that it works correctly.
"""

import os
import sys
import json
import pandas as pd
from python_scripts.mmm_named_rvs import train_mmm_with_named_rvs

def main():
    """Run the test with named RVs implementation"""
    print("Testing MMM initialization with named RVs in PyMC model context")
    
    # Use test_config_quick.json for quick testing
    config_path = "test_config_quick.json"
    data_path = "attached_assets/dankztestdata_v2.csv"
    
    # Check if files exist
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Load config
    with open(config_path, 'r') as f:
        raw_config = json.load(f)
    
    # Convert from test_config_quick.json format to the expected format
    config = {
        "targetColumn": raw_config.get("data", {}).get("response_column", "Sales"),
        "dateColumn": raw_config.get("data", {}).get("date_column", "Date"),
        "channelColumns": raw_config.get("channels", {}),
        "controlColumns": raw_config.get("data", {}).get("control_columns", []),
        "adstockSettings": {
            "channel_specific_params": {}
        },
        "saturationSettings": {
            "channel_specific_params": {}
        },
        "mcmcParams": {
            "draws": 100,
            "tune": 50,
            "chains": 1,
            "targetAccept": 0.8
        }
    }
    
    # Map channel parameters to the expected format
    for channel, params in raw_config.get("channels", {}).items():
        config["adstockSettings"]["channel_specific_params"][channel] = {
            "adstock_alpha": params.get("alpha", 0.5),
            "adstock_l_max": params.get("l_max", 8)
        }
        config["saturationSettings"]["channel_specific_params"][channel] = {
            "saturation_L": params.get("L", 1.0),
            "saturation_k": params.get("k", 0.0001),
            "saturation_x0": params.get("x0", 50000.0)
        }
    
    # Add default values
    config["adstockSettings"]["default"] = {
        "adstock_alpha": 0.5,
        "adstock_l_max": 8
    }
    config["saturationSettings"]["default"] = {
        "saturation_L": 1.0,
        "saturation_k": 0.0001,
        "saturation_x0": 50000.0
    }
    
    # Print key configuration parameters
    print(f"Testing with configuration:")
    print(f"  Target column: {config.get('targetColumn', 'Sales')}")
    print(f"  Date column: {config.get('dateColumn', 'Date')}")
    print(f"  Channel columns: {list(config.get('channelColumns', {}).keys())}")
    print(f"  Control columns: {config.get('controlColumns', [])}")
    
    try:
        # Train the model with named RVs
        mmm = train_mmm_with_named_rvs(df, config)
        print("SUCCESS: MMM initialization with named RVs in PyMC model context worked!")
        print(f"MMM object type: {type(mmm)}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to initialize MMM with named RVs: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Test passed successfully!")
    else:
        print("Test failed!")
        sys.exit(1)
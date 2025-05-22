#!/usr/bin/env python
"""
Test runner script for the Channel-Dimensioned Global Priors MMM implementation

This script loads test data and configuration and runs the MMM training with
channel-dimensioned priors to verify that it works correctly.
"""

import os
import sys
import json
import pandas as pd
from python_scripts.mmm_fixed_dims import train_mmm_with_channel_dims

def main():
    """Run the test with channel-dimensioned priors implementation"""
    print("Testing MMM initialization with Channel-Dimensioned Global Priors")
    
    # Use test_config_quick.json for quick testing
    config_path = "test_config_quick.json"
    data_path = "attached_assets/dankztestdata_v2.csv"
    
    # Check if files exist
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return False
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return False
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Print key configuration parameters
    print(f"Testing with configuration:")
    print(f"  Target column: {config.get('targetColumn', 'Sales')}")
    print(f"  Date column: {config.get('dateColumn', 'Date')}")
    print(f"  Channel columns: {list(config.get('channelColumns', {}).keys())}")
    print(f"  Control columns: {config.get('controlColumns', [])}")
    
    try:
        # Train the model with channel-dimensioned priors
        mmm = train_mmm_with_channel_dims(df, config)
        print("SUCCESS: MMM initialization with Channel-Dimensioned Global Priors worked!")
        print(f"MMM object type: {type(mmm)}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to initialize MMM: {str(e)}")
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
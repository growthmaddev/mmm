#!/usr/bin/env python
"""
Simple test script for PyMC-Marketing MMM with direct parameters

This script attempts to create a simple MMM model with direct parameter values
following the PyMC-Marketing API documentation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pymc as pm
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from pymc_marketing.mmm import MMM


def load_json_config(config_file):
    """Load JSON configuration file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def load_data(data_file, date_column="date"):
    """Load CSV data for MMM modeling"""
    df = pd.read_csv(data_file)
    return df


def run_mmm_test(config_file, data_file, output_file=None):
    """
    Create and fit a simple MMM model using PyMC-Marketing

    Uses direct parameter values with a minimal configuration.
    """
    # 1. Load configuration and data
    print(f"Loading config from {config_file}")
    config = load_json_config(config_file)
    
    print(f"Loading data from {data_file}")
    df = load_data(data_file)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 2. Extract configuration
    target_column = config.get("targetColumn", "Sales")
    date_column = config.get("dateColumn", "Date")
    
    # Filter for channels that actually exist in the data
    all_channels = config.get("channelColumns", [])
    if not all_channels:
        all_channels = [col for col in df.columns if col.endswith("_Spend")]
    
    # Only use channels that are actually in the dataframe
    channels = [ch for ch in all_channels if ch in df.columns]
    print(f"Using channels: {channels}")
    
    # 3. Create a simple MMM model
    # First try with the minimal required parameters
    print(f"Initializing MMM model with minimal parameters...")
    
    # Make sure the Date column is properly formatted
    # Use dayfirst=True for DD/MM/YYYY format
    try:
        df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
        print(f"Successfully parsed dates with dayfirst=True")
    except ValueError:
        print(f"Trying alternative date parsing...")
        df[date_column] = pd.to_datetime(df[date_column], format='mixed')
    
    # Print date range
    print(f"Date range: {df[date_column].min()} to {df[date_column].max()}")
    
    # Add a dummy control column if none exists 
    if len(config.get("controlColumns", [])) == 0:
        df["dummy_control"] = 0
        control_columns = ["dummy_control"]
    else:
        control_columns = [col for col in config.get("controlColumns", []) if col in df.columns]
        if not control_columns:
            df["dummy_control"] = 0
            control_columns = ["dummy_control"]
    
    try:
        # Create adstock and saturation transformations
        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        
        # Create adstock and saturation objects with default parameters
        adstock = GeometricAdstock(l_max=14)  # Default l_max value
        saturation = LogisticSaturation()     # Use default parameters
        
        # Create model with required parameters
        mmm = MMM(
            date_column=date_column,
            channel_columns=channels,
            control_columns=control_columns,
            adstock=adstock,            # Required parameter
            saturation=saturation       # Required parameter
        )
        
        print(f"Successfully created MMM model")
        
        # 4. Prepare data for fitting
        X = df[[date_column] + channels + control_columns].copy()
        y = df[target_column].copy()
        
        # 5. Fit the model with minimal MCMC settings
        print(f"Fitting model with {len(X)} data points...")
        
        # Use minimal settings for testing
        mcmc_settings = {
            "draws": 100,
            "tune": 100,
            "chains": 1,
            "target_accept": 0.9,
            "return_inferencedata": True
        }
        
        # Fit the model
        trace = mmm.fit(X=X, y=y, **mcmc_settings)
        print(f"Model fitting complete!")
        
        # Extract model parameters
        print(f"Extracting model parameters...")
        
        # Create basic results dictionary
        results = {
            "success": True,
            "message": "Model successfully fitted with direct parameters",
            "model_info": {
                "channels": channels,
                "date_range": {
                    "start": df[date_column].min().strftime("%Y-%m-%d"),
                    "end": df[date_column].max().strftime("%Y-%m-%d")
                },
                "iterations": mcmc_settings["draws"],
                "target_column": target_column
            }
        }
        
        # Save results if output file is specified
        if output_file:
            print(f"Saving results to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Main function to run the test"""
    if len(sys.argv) < 3:
        print("Usage: python test_mmm_direct_params.py <config_file> <data_file> [<output_file>]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    data_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = run_mmm_test(config_file, data_file, output_file)
    
    if result["success"]:
        print("\n✅ Model fitting successful!")
        print(f"Model information:")
        for key, value in result["model_info"].items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ Model fitting failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
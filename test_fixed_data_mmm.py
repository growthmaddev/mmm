#!/usr/bin/env python
"""
Test script for PyMC-Marketing MMM with proper data preparation.

This script focuses on correct data handling to avoid type comparison issues
during model validation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pymc as pm
from typing import Dict, Any, List
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation


def load_config(config_file):
    """Load model configuration from JSON file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def prepare_data(data_file, config):
    """Load and prepare data with proper types for PyMC-Marketing"""
    # Load data
    df = pd.read_csv(data_file)
    print(f"Loaded data with shape: {df.shape}")
    
    # Get config values
    target_column = config.get("targetColumn", "Sales")
    date_column = config.get("dateColumn", "Date")
    
    # Auto-detect or use configured channels
    channel_columns = config.get("channelColumns", [])
    if not channel_columns:
        channel_columns = [col for col in df.columns if col.endswith('_Spend')]
    
    # Filter to only include channels that exist in the dataframe
    channel_columns = [col for col in channel_columns if col in df.columns]
    print(f"Using channel columns: {channel_columns}")
    
    # Convert date column to datetime with proper format
    try:
        df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
        print(f"Successfully parsed dates with dayfirst=True (DD/MM/YYYY format)")
    except:
        print(f"Trying alternative date parsing...")
        df[date_column] = pd.to_datetime(df[date_column], format='mixed')
    
    print(f"Date range: {df[date_column].min()} to {df[date_column].max()}")
    
    # Ensure all channel columns are numeric and handle NaN values
    for col in channel_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Converting {col} to numeric")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle NaN values (critical for PyMC-Marketing)
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"Warning: {col} contains {nan_count} NaN values. Replacing with zeros.")
            df[col].fillna(0, inplace=True)
        
        # Ensure no negative values (required by PyMC-Marketing)
        if (df[col] < 0).any():
            print(f"Warning: {col} contains negative values. Replacing with zeros.")
            df.loc[df[col] < 0, col] = 0
    
    # Ensure target column is numeric and handle NaN values
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        print(f"Converting {target_column} to numeric")
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    
    # Handle NaN values in target column
    nan_count = df[target_column].isna().sum()
    if nan_count > 0:
        print(f"Warning: {target_column} contains {nan_count} NaN values. Replacing with mean value.")
        # Use mean for target column rather than 0
        mean_value = df[target_column].mean(skipna=True)
        if pd.isnull(mean_value):  # If all values are NaN, use 0
            mean_value = 0
        df[target_column].fillna(mean_value, inplace=True)
    
    # Add dummy control column if needed
    control_columns = config.get("controlColumns", [])
    if not control_columns or not any(col in df.columns for col in control_columns):
        print(f"Adding dummy control column")
        df["dummy_control"] = 0.0
        control_columns = ["dummy_control"]
    else:
        control_columns = [col for col in control_columns if col in df.columns]
    
    # Ensure control columns are numeric
    for col in control_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Converting control column {col} to numeric")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Data preparation complete. Final shape: {df.shape}")
    
    # Return the prepared data and configuration
    return {
        "df": df,
        "target_column": target_column,
        "date_column": date_column,
        "channel_columns": channel_columns,
        "control_columns": control_columns
    }


def run_mmm_test(config_file, data_file, output_file=None):
    """Run MMM test with proper data preparation"""
    try:
        # 1. Load configuration
        config = load_config(config_file)
        
        # 2. Prepare data
        data_info = prepare_data(data_file, config)
        df = data_info["df"]
        target_column = data_info["target_column"]
        date_column = data_info["date_column"]
        channel_columns = data_info["channel_columns"]
        control_columns = data_info["control_columns"]
        
        # Get model settings
        model_config = config.get("model", {})
        mcmc_settings = {
            "draws": model_config.get("iterations", 100),
            "tune": model_config.get("tuning", 100),
            "chains": model_config.get("chains", 1),
            "target_accept": 0.9,
            "return_inferencedata": True
        }
        
        # 3. Create adstock and saturation objects
        print(f"Creating adstock and saturation objects...")
        
        adstock_settings = config.get("adstockSettings", {})
        l_max = adstock_settings.get("default", {}).get("l_max", 14)
        
        adstock = GeometricAdstock(l_max=l_max)
        saturation = LogisticSaturation()
        
        # 4. Create and fit MMM model
        print(f"Initializing MMM model...")
        
        with pm.Model() as model:
            mmm = MMM(
                date_column=date_column,
                channel_columns=channel_columns,
                control_columns=control_columns,
                adstock=adstock,
                saturation=saturation
            )
            
            print(f"Preparing data for fitting...")
            X = df[[date_column] + channel_columns + control_columns]
            y = df[target_column]
            
            # Diagnostic info
            print(f"X shape: {X.shape}, dtypes: {X.dtypes}")
            print(f"y shape: {y.shape}, dtype: {y.dtype}")
            
            print(f"Fitting model with {len(X)} data points...")
            trace = mmm.fit(X=X, y=y, **mcmc_settings)
            
            print(f"Model fitting complete!")
            
            # Basic results extraction
            results = {
                "success": True,
                "message": "Model fitted successfully",
                "model_info": {
                    "channels": channel_columns,
                    "data_points": len(X),
                    "mcmc_settings": mcmc_settings
                }
            }
            
            # Save results if requested
            if output_file:
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
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python test_fixed_data_mmm.py <config_file> <data_file> [<output_file>]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    data_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Run the test
    result = run_mmm_test(config_file, data_file, output_file)
    
    # Print result summary
    if result["success"]:
        print("\n✅ Model fitting successful!")
    else:
        print("\n❌ Model fitting failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
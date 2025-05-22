#!/usr/bin/env python
"""
Test script for MMM with monkey-patched dims and direct transform parameters.

This script uses a different strategy to bypass the Random Variables constraint
issues during model fitting by using fixed parameter values directly.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pymc as pm
from typing import Dict, Any, List
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation


def load_json_config(config_file):
    """Load JSON configuration file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def load_data(data_file, date_column="date"):
    """Load CSV data for MMM modeling"""
    df = pd.read_csv(data_file)
    return df


def create_and_fit_mmm_model(config_file, data_file=None, data_df=None):
    """
    Create an MMM model with fixed parameters and fit it to data
    
    This implementation bypasses Random Variables for the transformation parameters
    to avoid constraint issues during model fitting.
    """
    print(f"Loading model config from {config_file}...", file=sys.stderr)
    config = load_json_config(config_file)
    
    # Load data from CSV file or use provided DataFrame
    if data_df is not None:
        df = data_df
    elif data_file is not None:
        print(f"Loading data from {data_file}...", file=sys.stderr)
        df = load_data(data_file)
    else:
        raise ValueError("Either data_file or data_df must be provided")
    
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns", file=sys.stderr)
    print(f"Columns: {df.columns.tolist()}", file=sys.stderr)
    
    # 1. Extract config values
    target_column = config.get("targetColumn", "Sales")
    date_column = config.get("dateColumn", "Date")
    channels = config.get("channelColumns", [])
    model_config = config.get("model", {})
    control_columns = config.get("controlColumns", [])
    adstock_settings = config.get("adstockSettings", {})
    saturation_settings = config.get("saturationSettings", {})
    
    # If no channels are specified, try to auto-detect them based on column names
    if not channels:
        channels = [col for col in df.columns if col.endswith('_Spend') and col != target_column]
    
    channel_name_list = channels
    print(f"Found channels in data: {channels}", file=sys.stderr)
    print(f"Channel names: {channel_name_list}", file=sys.stderr)
    
    # 2. Extract adstock and saturation parameters
    # Get adstock alpha values for each channel
    alpha_values = []
    L_values = []
    k_values = []
    x0_values = []
    
    for channel in channel_name_list:
        # Get adstock alpha
        channel_adstock = adstock_settings.get("channel_specific_params", {}).get(
            channel, adstock_settings.get("default", {})
        )
        alpha_values.append(channel_adstock.get("adstock_alpha", 0.5))
        
        # Get saturation parameters
        channel_saturation = saturation_settings.get("channel_specific_params", {}).get(
            channel, saturation_settings.get("default", {})
        )
        L_values.append(channel_saturation.get("L", 1.0))
        k_values.append(channel_saturation.get("k", 0.0001))
        x0_values.append(channel_saturation.get("x0", 50000.0))
    
    # Convert to numpy arrays
    alpha_values = np.array(alpha_values)
    L_values = np.array(L_values)
    k_values = np.array(k_values)
    x0_values = np.array(x0_values)
    
    print(f"Prepared parameter arrays:", file=sys.stderr)
    print(f"alpha_values: {alpha_values}", file=sys.stderr)
    print(f"L_values: {L_values}", file=sys.stderr)
    print(f"k_values: {k_values}", file=sys.stderr)
    print(f"x0_values: {x0_values}", file=sys.stderr)
    
    # Get global adstock l_max
    global_l_max = adstock_settings.get("default", {}).get("l_max", 14)
    print(f"global_l_max: {global_l_max}", file=sys.stderr)
    
    # 3. Create PyMC Model Context
    with pm.Model(coords={"channel": channel_name_list}) as mmm_model_context:
        print(f"Created PyMC model context with channels: {channel_name_list}", file=sys.stderr)
        
        # Create channel-specific adstock and saturation objects directly
        adstock_objects = {}
        saturation_objects = {}
        
        for i, channel in enumerate(channel_name_list):
            # Use fixed values directly to bypass constraints issues
            # PyMC-Marketing uses 'priors' dict rather than direct parameters
            adstock_objects[channel] = GeometricAdstock(
                l_max=global_l_max,
                priors={
                    "alpha": float(alpha_values[i])
                }
            )
            
            saturation_objects[channel] = LogisticSaturation(
                priors={
                    "L": float(L_values[i]),
                    "k": float(k_values[i]),
                    "x0": float(x0_values[i])
                }
            )
            
            # Monkey-patch the object's internal parameters as needed
            for transform_obj in [adstock_objects[channel], saturation_objects[channel]]:
                for param_name, param_obj in transform_obj.__dict__.items():
                    if hasattr(param_obj, '__class__') and 'TensorVariable' in str(param_obj.__class__):
                        if not hasattr(param_obj, 'dims'):
                            print(f"DEBUG: Monkey-patching .dims for {channel} {param_name}", file=sys.stderr)
                            param_obj.dims = ("channel",)
        
        # 4. Create the MMM model
        try:
            print(f"Building MMM model with direct transform objects...", file=sys.stderr)
            
            # Ensure we have control columns
            existing_controls = [col for col in control_columns if col in df.columns]
            if not existing_controls:
                df["dummy_control"] = 0.0
                existing_controls = ["dummy_control"]
            
            print(f"Using control columns: {existing_controls}", file=sys.stderr)
            
            # Initialize MMM with channel-specific transform objects
            mmm = MMM(
                channel_columns=channels,
                date_column=date_column,
                control_columns=existing_controls,
                transforms_per_channel=True
            )
            
            # Apply channel-specific transformations
            for channel in channel_name_list:
                mmm.set_transform(
                    channel=channel,
                    adstock=adstock_objects[channel],
                    saturation=saturation_objects[channel]
                )
            
            print(f"Successfully built MMM model with direct transforms!", file=sys.stderr)
            
            # 5. Fit the model
            print(f"Preparing to fit the model...", file=sys.stderr)
            
            # Extract MCMC settings from config
            mcmc_settings = {
                "draws": model_config.get("iterations", 100),
                "tune": model_config.get("tuning", 100),
                "chains": model_config.get("chains", 1),
                "target_accept": 0.9,
                "return_inferencedata": True
            }
            
            print(f"Fitting with settings: {mcmc_settings}", file=sys.stderr)
            
            # Prepare data for fitting
            X_columns = [date_column] + channels + existing_controls
            X = df[X_columns].copy()
            y = df[target_column].copy()
            
            # Ensure date column is properly formatted
            if date_column in X.columns:
                sample_date = str(X[date_column].iloc[0])
                print(f"Sample date format: {sample_date}", file=sys.stderr)
                
                try:
                    X[date_column] = pd.to_datetime(X[date_column], dayfirst=True)
                    print(f"Successfully parsed dates with dayfirst=True", file=sys.stderr)
                except ValueError as e:
                    print(f"Error parsing dates: {str(e)}", file=sys.stderr)
                    X[date_column] = pd.to_datetime(X[date_column], format='mixed')
            
            print(f"Input data X shape: {X.shape}, columns: {X.columns.tolist()}", file=sys.stderr)
            print(f"Target y shape: {y.shape}", file=sys.stderr)
            
            # Fit the model
            print(f"Fitting model with {len(X)} data points...", file=sys.stderr)
            trace = mmm.fit(X=X, y=y, **mcmc_settings)
            print(f"Model fitting complete!", file=sys.stderr)
            
            # Get results
            channel_results = {}
            for channel in channel_name_list:
                channel_results[channel] = {
                    "adstock_alpha": alpha_values[i],
                    "saturation_L": L_values[i],
                    "saturation_k": k_values[i],
                    "saturation_x0": x0_values[i]
                }
            
            return {
                "success": True,
                "message": "Model fitted successfully with direct transforms",
                "channel_parameters": channel_results,
                "trace": trace
            }
            
        except Exception as e:
            print(f"ERROR: Failed during model fitting: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }


def main():
    """Run a test with the direct transforms approach"""
    if len(sys.argv) < 3:
        print("Usage: python test_fixed_dims_monkey_patch.py <config_file> <data_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    data_file = sys.argv[2]
    
    result = create_and_fit_mmm_model(config_file, data_file)
    
    if result["success"]:
        print("✅ Model fitting successful!")
        print(f"Channel parameters used:")
        for channel, params in result["channel_parameters"].items():
            print(f"  {channel}: {params}")
    else:
        print("❌ Model fitting failed!")
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
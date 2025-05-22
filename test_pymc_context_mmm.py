#!/usr/bin/env python
"""
Test script for MMM initialization with proper PyMC model context
This tests the approach of creating named RVs inside a model context for parameters
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pymc as pm
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

def load_test_data(file_path):
    """Load test data"""
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape: {df.shape}", file=sys.stderr)
    return df

def test_model_context_approach(df, config_path):
    """Test the PyMC model context approach for MMM initialization"""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract basic parameters
    target_column = config.get('targetColumn', 'Sales')
    date_column = config.get('dateColumn', 'Date')
    channel_columns = config.get('channelColumns', {})
    
    # Get the channel list
    channel_list = []
    if isinstance(channel_columns, dict):
        channel_list = list(channel_columns.keys())
    elif isinstance(channel_columns, list):
        channel_list = channel_columns
    else:
        raise ValueError("channel_columns must be a dict or list")
    
    # Get control columns
    control_cols = config.get('controlColumns', [])
    
    print(f"Testing model with parameters:\n"
          f"  Target: {target_column}\n"
          f"  Date: {date_column}\n"
          f"  Channels: {channel_list}\n"
          f"  Controls: {control_cols}", file=sys.stderr)
    
    # Create an explicit PyMC Model Context with channel coordinates
    with pm.Model(coords={"channel": channel_list}) as mmm_model_context:
        print(f"Created PyMC Model context: {mmm_model_context}", file=sys.stderr)
        print(f"Model coordinates: {mmm_model_context.coords}", file=sys.stderr)
        
        # Storage for channel-specific transforms
        channel_specific_transforms = {}
        
        # For each channel, create transform objects with named RVs inside this model context
        for channel in channel_list:
            print(f"Setting up transforms for channel: {channel}", file=sys.stderr)
            
            # Get adstock parameters for this channel (simplified for test)
            adstock_params = {}
            if 'adstockSettings' in config:
                if channel in config['adstockSettings']:
                    adstock_params = config['adstockSettings'][channel]
                elif 'default' in config['adstockSettings']:
                    adstock_params = config['adstockSettings']['default']
            
            # Get saturation parameters for this channel (simplified for test)
            saturation_params = {}
            if 'saturationSettings' in config:
                if channel in config['saturationSettings']:
                    saturation_params = config['saturationSettings'][channel]
                elif 'default' in config['saturationSettings']:
                    saturation_params = config['saturationSettings']['default']
            
            # Extract the float values for parameters
            alpha_float = adstock_params.get('adstock_alpha', 0.5)
            l_max_int = adstock_params.get('adstock_l_max', 8)
            L_float = saturation_params.get('saturation_L', 1.0)
            k_float = saturation_params.get('saturation_k', 0.0001)
            x0_float = saturation_params.get('saturation_x0', 50000.0)
            
            # Create a safe name for the channel to use in RV names
            safe_channel_name = channel.replace('-', '_').replace('.', '_')
            
            print(f"Defining fixed RVs for channel: {channel} in context: {pm.Model.get_context()}", file=sys.stderr)
            
            # Define Fixed Parameters as NAMED Random Variables INSIDE this Context
            alpha_rv = pm.Normal(f"fixed_alpha_{safe_channel_name}", mu=alpha_float, sigma=1e-6)
            
            L_rv = pm.Normal(f"fixed_L_{safe_channel_name}", mu=L_float, sigma=1e-6)
            k_sigma = max(abs(k_float * 0.001), 1e-7)
            x0_sigma = max(abs(x0_float * 0.001), 1e-2)
            k_rv = pm.Normal(f"fixed_k_{safe_channel_name}", mu=k_float, sigma=k_sigma)
            x0_rv = pm.Normal(f"fixed_x0_{safe_channel_name}", mu=x0_float, sigma=x0_sigma)
            
            # Pass RVs to Transform Objects
            adstock_obj = GeometricAdstock(l_max=l_max_int, priors={"alpha": alpha_rv})
            saturation_obj = LogisticSaturation(priors={"L": L_rv, "k": k_rv, "x0": x0_rv})
            
            print(f"Created adstock for {channel} with alpha={alpha_float}, l_max={l_max_int}", file=sys.stderr)
            print(f"Created saturation for {channel} with L={L_float}, k={k_float}, x0={x0_float}", file=sys.stderr)
            
            # Store these objects for this channel
            channel_specific_transforms[channel] = {
                'adstock': adstock_obj,
                'saturation': saturation_obj
            }
        
        # Get the first channel key for initialization
        first_channel_key = channel_list[0]
        
        try:
            print(f"Initializing MMM object within context: {pm.Model.get_context()}", file=sys.stderr)
            # Pass the Model Context to MMM
            mmm = MMM(
                date_column=date_column,
                channel_columns=channel_list,
                control_columns=control_cols,
                adstock=channel_specific_transforms[first_channel_key]['adstock'],
                saturation=channel_specific_transforms[first_channel_key]['saturation'],
                model=mmm_model_context  # Pass the explicit model context
            )
            print(f"MMM object initialized. Type: {type(mmm)}", file=sys.stderr)
            
            # Continue Using mmm.media_transforms for applying all channel specifics
            print(f"Attempting to assign mmm.media_transforms", file=sys.stderr)
            mmm.media_transforms = channel_specific_transforms
            print(f"Successfully assigned to mmm.media_transforms.", file=sys.stderr)
            
            # Verify the model component
            print(f"MMM model property: {mmm.model}", file=sys.stderr)
            print(f"MMM model variables count: {len(mmm.model.named_vars)}", file=sys.stderr)
            
            return "Success! MMM initialized with PyMC model context approach."
            
        except Exception as e:
            print(f"Error initializing MMM with model context: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return f"Failed: {str(e)}"

def main():
    """Main function"""
    # Use test_config_quick.json for quick testing
    config_path = "test_config_quick.json"
    data_path = "attached_assets/dankztestdata_v2.csv"
    
    # Check if files exist
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}", file=sys.stderr)
        return
    
    # Load data
    df = load_test_data(data_path)
    
    # Test the approach
    result = test_model_context_approach(df, config_path)
    print(f"Test result: {result}")

if __name__ == "__main__":
    main()
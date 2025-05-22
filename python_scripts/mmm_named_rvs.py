#!/usr/bin/env python
"""
Implementation for creating MMM with named PyMC random variables inside a model context

This approach follows the recommendation to use explicitly named PyMC Random Variables
defined within a dedicated PyMC Model context, and then pass that context to the MMM
constructor to avoid the AttributeError with TensorVariable objects.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pymc as pm
from typing import Dict, Any, List, Optional
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

def train_mmm_with_named_rvs(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> MMM:
    """
    Train a marketing mix model using named PyMC random variables for parameters
    
    Args:
        df: DataFrame containing the data
        config: Dictionary with model configuration
        
    Returns:
        Trained MMM object
    """
    # Get model parameters from config
    target_column = config.get('targetColumn', 'Sales')
    date_column = config.get('dateColumn', 'Date')
    channel_columns = config.get('channelColumns', {})
    
    # Create the channel list to pass to MMM constructor
    channel_name_list = []
    if isinstance(channel_columns, dict):
        channel_name_list = list(channel_columns.keys())
        print(f"DEBUG: Using channel names from dictionary keys: {channel_name_list}", file=sys.stderr)
    elif isinstance(channel_columns, list):
        channel_name_list = channel_columns
        print(f"DEBUG: Using channel names from list directly: {channel_name_list}", file=sys.stderr)
    else:
        print(f"DEBUG: WARNING: channel_columns is neither dict nor list: {type(channel_columns)}", file=sys.stderr)
        raise ValueError("No valid channel names available")
    
    # Get control columns from config
    control_list = []
    if 'controlColumns' in config:
        control_list = config['controlColumns']
        print(f"DEBUG: Found controlColumns in config: {control_list}", file=sys.stderr)
    elif 'control_columns' in config:
        control_list = config['control_columns']
        print(f"DEBUG: Found control_columns in config: {control_list}", file=sys.stderr)
    
    print(f"DEBUG: Using control columns for MMM: {control_list}", file=sys.stderr)
    
    # If no control columns provided, try to find one or create a dummy
    if not control_list:
        # Try to find a column that's not a channel or the target
        possible_controls = [col for col in df.columns 
                            if col not in channel_name_list
                            and col != date_column
                            and col != target_column]
        
        if possible_controls:
            control_list = [possible_controls[0]]
            print(f"DEBUG: Using fallback control column: {control_list[0]}", file=sys.stderr)
        else:
            # Create a dummy control column
            df['_dummy_control'] = 1.0
            control_list = ['_dummy_control']
            print(f"DEBUG: Created dummy control column '_dummy_control'", file=sys.stderr)
    
    # 1. Establish an Explicit PyMC Model Context with Channel Coordinates
    with pm.Model(coords={"channel": channel_name_list}) as mmm_model_context:
        print(f"DEBUG: Created PyMC Model context: {mmm_model_context}", file=sys.stderr)
        print(f"DEBUG: Model coordinates: {mmm_model_context.coords}", file=sys.stderr)
        
        # Create the channel-specific adstock and saturation transform objects
        channel_specific_transforms = {}
        
        # 2. For each channel, create named random variables inside the model context
        for channel_data_key in channel_name_list:
            print(f"DEBUG: Setting up transforms for channel: {channel_data_key}", file=sys.stderr)
            
            # Get adstock parameters for this channel
            adstock_params = {}
            if 'adstockSettings' in config:
                if ('channel_specific_params' in config['adstockSettings'] and 
                    channel_data_key in config['adstockSettings']['channel_specific_params']):
                    adstock_params = config['adstockSettings']['channel_specific_params'][channel_data_key]
                    print(f"Using channel-specific adstock params from config for {channel_data_key}", file=sys.stderr)
                # Fall back to default if no channel-specific params
                elif 'default' in config['adstockSettings']:
                    adstock_params = config['adstockSettings']['default']
                    print(f"Using default adstock params from config for {channel_data_key}", file=sys.stderr)
            
            # Get saturation parameters for this channel
            saturation_params = {}
            if 'saturationSettings' in config:
                if ('channel_specific_params' in config['saturationSettings'] and 
                    channel_data_key in config['saturationSettings']['channel_specific_params']):
                    saturation_params = config['saturationSettings']['channel_specific_params'][channel_data_key]
                    print(f"Using channel-specific saturation params from config for {channel_data_key}", file=sys.stderr)
                # Fall back to default if no channel-specific params
                elif 'default' in config['saturationSettings']:
                    saturation_params = config['saturationSettings']['default']
                    print(f"Using default saturation params from config for {channel_data_key}", file=sys.stderr)
            
            # Extract float values for parameters
            alpha_float = adstock_params.get('adstock_alpha', 0.5)
            l_max_int = adstock_params.get('adstock_l_max', 8)
            L_float = saturation_params.get('saturation_L', 1.0)
            k_float = saturation_params.get('saturation_k', 0.0001)
            x0_float = saturation_params.get('saturation_x0', 50000.0)
            
            # Create a safe name for the channel to use in RV names
            safe_channel_name_for_rv = channel_data_key.replace('-', '_').replace('.', '_')
            
            print(f"DEBUG: Defining fixed RVs for channel: {channel_data_key} in context: {pm.Model.get_context()}", file=sys.stderr)
            
            # Define Fixed Parameters as NAMED Random Variables
            alpha_rv = pm.Normal(f"fixed_alpha_{safe_channel_name_for_rv}", mu=alpha_float, sigma=1e-6)
            
            L_rv = pm.Normal(f"fixed_L_{safe_channel_name_for_rv}", mu=L_float, sigma=1e-6)
            k_sigma = max(abs(k_float * 0.001), 1e-7)  # Ensure sigma > 0
            x0_sigma = max(abs(x0_float * 0.001), 1e-2)  # Ensure sigma > 0
            k_rv = pm.Normal(f"fixed_k_{safe_channel_name_for_rv}", mu=k_float, sigma=k_sigma)
            x0_rv = pm.Normal(f"fixed_x0_{safe_channel_name_for_rv}", mu=x0_float, sigma=x0_sigma)
            
            # 3. Create Transform Objects using the named RVs
            adstock_obj = GeometricAdstock(l_max=l_max_int, priors={"alpha": alpha_rv})
            saturation_obj = LogisticSaturation(priors={"L": L_rv, "k": k_rv, "x0": x0_rv})
            
            print(f"DEBUG: Created adstock for {channel_data_key} with alpha={alpha_float}, l_max={l_max_int}", file=sys.stderr)
            print(f"DEBUG: Created saturation for {channel_data_key} with L={L_float}, k={k_float}, x0={x0_float}", file=sys.stderr)
            
            # Store these objects for this channel
            channel_specific_transforms[channel_data_key] = {
                'adstock': adstock_obj,
                'saturation': saturation_obj
            }
        
        # Get the first channel key for initialization
        first_channel_key = channel_name_list[0]
        print(f"DEBUG: Using first channel for transforms: {first_channel_key}", file=sys.stderr)
        
        try:
            print(f"DEBUG: Initializing MMM object within context: {pm.Model.get_context()}", file=sys.stderr)
            
            # 4. Initialize MMM WITHIN THE MODEL CONTEXT 
            # The model will automatically use the current PyMC context (mmm_model_context)
            # without explicitly passing it as a parameter
            mmm = MMM(
                date_column=date_column,
                channel_columns=channel_name_list,
                control_columns=control_list,
                adstock=channel_specific_transforms[first_channel_key]['adstock'],
                saturation=channel_specific_transforms[first_channel_key]['saturation']
            )
            
            # 5. Set mmm.media_transforms
            print(f"DEBUG: Setting mmm.media_transforms...", file=sys.stderr)
            mmm.media_transforms = channel_specific_transforms
            print(f"DEBUG: Successfully set mmm.media_transforms", file=sys.stderr)
            
            # Fit the model with the provided data
            print(f"DEBUG: Preparing to fit model...", file=sys.stderr)
            
            # Get MCMC parameters from config
            mcmc_params = config.get('mcmcParams', {})
            draws = mcmc_params.get('draws', 200)
            tune = mcmc_params.get('tune', 100)
            chains = mcmc_params.get('chains', 1)
            target_accept = mcmc_params.get('targetAccept', 0.8)
            
            # Fit the model
            print(f"DEBUG: Fitting model with draws={draws}, tune={tune}, chains={chains}", file=sys.stderr)
            idata = mmm.fit(
                data=df,
                target=target_column,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept
            )
            print(f"DEBUG: Model fitting complete", file=sys.stderr)
            
            return mmm
            
        except Exception as e:
            print(f"CRITICAL: Error during MMM initialization or fitting: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise e
#!/usr/bin/env python
"""
Implementation of Channel-Dimensioned Global Priors approach for MMM with PyMC

This implementation creates channel-dimensioned PyMC Random Variables with dims='channel'
to ensure they're compatible with PyMC-Marketing's validation which expects this attribute.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pymc as pm
from typing import Dict, Any, List, Optional
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

def debug_distribution_dims(dist, name="dist"):
    """
    Helper function to debug a PyMC distribution's dimensions and attributes
    
    Args:
        dist: PyMC distribution or TensorVariable
        name: Name to display in debug output
    """
    print(f"DEBUG: {name} type: {type(dist)}", file=sys.stderr)
    try:
        print(f"DEBUG: {name}.dims: {getattr(dist, 'dims', 'NO DIMS ATTRIBUTE')}", file=sys.stderr)
    except Exception as e:
        print(f"DEBUG: Error getting {name}.dims: {str(e)}", file=sys.stderr)
    
    try:
        print(f"DEBUG: {name}.name: {getattr(dist, 'name', 'NO NAME ATTRIBUTE')}", file=sys.stderr)
    except Exception as e:
        print(f"DEBUG: Error getting {name}.name: {str(e)}", file=sys.stderr)
    
    # Check if the object has PyMC dimensions
    try:
        if hasattr(dist, "eval"):
            print(f"DEBUG: {name}.eval() shape: {dist.eval().shape}", file=sys.stderr)
    except Exception as e:
        print(f"DEBUG: Error evaluating {name}: {str(e)}", file=sys.stderr)
        
    if hasattr(dist, "__dict__"):
        try:
            print(f"DEBUG: {name}.__dict__ keys: {list(dist.__dict__.keys())}", file=sys.stderr)
        except Exception as e:
            print(f"DEBUG: Error getting {name}.__dict__: {str(e)}", file=sys.stderr)

def train_mmm_with_channel_dims(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> MMM:
    """
    Train a marketing mix model using channel-dimensioned PyMC random variables
    
    This implementation uses the "Channel-Dimensioned Global Priors" approach
    where parameters are defined as single RVs with channel dimensions.
    
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
    
    # 1. Create the channel list to pass to MMM constructor
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
        
        # 2. Prepare NumPy Arrays for Fixed Parameter Values
        alpha_values = []
        l_max_values = []
        L_values = []
        k_values = []
        x0_values = []
        
        # Collect parameter values for each channel
        for channel_data_key in channel_name_list:
            print(f"DEBUG: Collecting parameters for channel: {channel_data_key}", file=sys.stderr)
            
            # Get adstock parameters for this channel
            adstock_params = {}
            if 'adstockSettings' in config:
                if ('channel_specific_params' in config['adstockSettings'] and 
                    channel_data_key in config['adstockSettings']['channel_specific_params']):
                    adstock_params = config['adstockSettings']['channel_specific_params'][channel_data_key]
                    print(f"Using channel-specific adstock params for {channel_data_key}", file=sys.stderr)
                # Fall back to default if no channel-specific params
                elif 'default' in config['adstockSettings']:
                    adstock_params = config['adstockSettings']['default']
                    print(f"Using default adstock params for {channel_data_key}", file=sys.stderr)
            
            # Get saturation parameters for this channel
            saturation_params = {}
            if 'saturationSettings' in config:
                if ('channel_specific_params' in config['saturationSettings'] and 
                    channel_data_key in config['saturationSettings']['channel_specific_params']):
                    saturation_params = config['saturationSettings']['channel_specific_params'][channel_data_key]
                    print(f"Using channel-specific saturation params for {channel_data_key}", file=sys.stderr)
                # Fall back to default if no channel-specific params
                elif 'default' in config['saturationSettings']:
                    saturation_params = config['saturationSettings']['default']
                    print(f"Using default saturation params for {channel_data_key}", file=sys.stderr)
            
            # Extract parameter values for this channel
            alpha_values.append(adstock_params.get('adstock_alpha', 0.5))
            l_max_values.append(adstock_params.get('adstock_l_max', 8))
            L_values.append(saturation_params.get('saturation_L', 1.0))
            k_values.append(saturation_params.get('saturation_k', 0.0001))
            x0_values.append(saturation_params.get('saturation_x0', 50000.0))
        
        # Convert to NumPy arrays
        alpha_values = np.array(alpha_values)
        l_max_values = np.array(l_max_values)
        L_values = np.array(L_values)
        k_values = np.array(k_values)
        x0_values = np.array(x0_values)
        
        # Use max l_max for the global adstock object
        global_l_max = int(np.max(l_max_values))
        print(f"DEBUG: Using global l_max for Adstock: {global_l_max}", file=sys.stderr)
        
        # 3. Define Channel-Dimensioned "Fixed" Priors as Named RVs
        print(f"DEBUG: Creating channel-dimensioned priors in context: {pm.Model.get_context()}", file=sys.stderr)
        
        # Create single named RVs with dims="channel"
        alpha_rv_chan = pm.Normal("fixed_alphas_per_channel", mu=alpha_values, sigma=1e-6, dims="channel")
        debug_distribution_dims(alpha_rv_chan, "alpha_rv_chan")
        
        L_rv_chan = pm.Normal("fixed_Ls_per_channel", mu=L_values, sigma=1e-6, dims="channel")
        debug_distribution_dims(L_rv_chan, "L_rv_chan")
        
        # Use small sigma values suitable for k and x0
        k_sigma_val = np.maximum(np.abs(k_values * 0.001), 1e-7)
        x0_sigma_val = np.maximum(np.abs(x0_values * 0.001), 1e-2)
        
        k_rv_chan = pm.Normal("fixed_ks_per_channel", mu=k_values, sigma=k_sigma_val, dims="channel")
        debug_distribution_dims(k_rv_chan, "k_rv_chan")
        
        x0_rv_chan = pm.Normal("fixed_x0s_per_channel", mu=x0_values, sigma=x0_sigma_val, dims="channel")
        debug_distribution_dims(x0_rv_chan, "x0_rv_chan")
        
        # 4. Create GLOBAL Adstock and Saturation Objects
        print(f"DEBUG: Creating global transform objects", file=sys.stderr)
        global_adstock_obj = GeometricAdstock(l_max=global_l_max, priors={"alpha": alpha_rv_chan})
        global_saturation_obj = LogisticSaturation(priors={"L": L_rv_chan, "k": k_rv_chan, "x0": x0_rv_chan})
        
        try:
            print(f"DEBUG: Initializing MMM object within context: {pm.Model.get_context()}", file=sys.stderr)
            
            # 5. Instantiate MMM with GLOBAL transforms (NO model parameter)
            mmm = MMM(
                date_column=date_column,
                channel_columns=channel_name_list,
                control_columns=control_list,
                adstock=global_adstock_obj,
                saturation=global_saturation_obj
                # NO model= argument as it causes validation error
                # NO mmm.media_transforms needed with global transforms
            )
            
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
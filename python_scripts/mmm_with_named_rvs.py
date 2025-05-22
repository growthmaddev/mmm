#!/usr/bin/env python
"""
Implementation for MMM using named PyMC random variables for parameters

This file contains a modified approach to create an MMM with fixed parameters
using named PyMC random variables inside a model context.
"""

import os
import sys
import pymc as pm
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

def create_mmm_with_named_rvs(
    df, 
    channel_list, 
    date_column,
    target_column,
    adstock_params=None,
    saturation_params=None,
    control_columns=None
):
    """
    Create an MMM with named PyMC random variables for parameters
    
    Args:
        df: DataFrame containing the data
        channel_list: List of channel column names
        date_column: Name of the date column
        target_column: Name of the target column
        adstock_params: Dictionary mapping channels to adstock parameters
        saturation_params: Dictionary mapping channels to saturation parameters
        control_columns: List of control column names (at least one required)
    
    Returns:
        MMM object with properly set up transforms
    """
    # Ensure default parameters
    if adstock_params is None:
        adstock_params = {}
    if saturation_params is None:
        saturation_params = {}
    if control_columns is None or len(control_columns) == 0:
        # Use the first column that isn't a channel, date, or target as a fallback control
        possible_controls = [col for col in df.columns 
                            if col not in channel_list 
                            and col != date_column 
                            and col != target_column]
        
        if len(possible_controls) > 0:
            control_columns = [possible_controls[0]]
            print(f"Using {control_columns[0]} as a fallback control column", file=sys.stderr)
        else:
            # If no suitable control found, create a dummy column of ones
            df['_dummy_control'] = 1.0
            control_columns = ['_dummy_control']
            print(f"Created dummy control column '_dummy_control'", file=sys.stderr)
    
    print(f"Using control columns: {control_columns}", file=sys.stderr)
    
    # Create a PyMC model (won't be directly passed to MMM)
    mmm_context = pm.Model()
    
    # Dictionary to store transform objects
    channel_specific_transforms = {}
    
    # Create named variables for each channel inside the PyMC model context
    with mmm_context:
        for channel in channel_list:
            print(f"Setting up transforms for channel: {channel}", file=sys.stderr)
            
            # Get adstock parameters for this channel
            channel_adstock = adstock_params.get(channel, {})
            alpha_float = channel_adstock.get('adstock_alpha', 0.5)
            l_max_int = channel_adstock.get('adstock_l_max', 8)
            
            # Get saturation parameters for this channel
            channel_saturation = saturation_params.get(channel, {})
            L_float = channel_saturation.get('saturation_L', 1.0)
            k_float = channel_saturation.get('saturation_k', 0.0001)
            x0_float = channel_saturation.get('saturation_x0', 50000.0)
            
            # Create a safe name for RVs
            safe_channel = channel.replace('-', '_').replace('.', '_')
            
            # Create named RVs with tight distributions
            alpha_rv = pm.Normal(f"alpha_{safe_channel}", mu=alpha_float, sigma=1e-6)
            L_rv = pm.Normal(f"L_{safe_channel}", mu=L_float, sigma=1e-6)
            k_rv = pm.Normal(f"k_{safe_channel}", mu=k_float, sigma=max(abs(k_float * 0.001), 1e-7))
            x0_rv = pm.Normal(f"x0_{safe_channel}", mu=x0_float, sigma=max(abs(x0_float * 0.001), 1e-2))
            
            # Create transform objects using RVs
            adstock_obj = GeometricAdstock(l_max=l_max_int, priors={"alpha": alpha_rv})
            saturation_obj = LogisticSaturation(priors={"L": L_rv, "k": k_rv, "x0": x0_rv})
            
            print(f"Created transforms for {channel} with named RVs", file=sys.stderr)
            channel_specific_transforms[channel] = {
                'adstock': adstock_obj,
                'saturation': saturation_obj
            }
    
    # Get first channel for initialization
    first_channel = channel_list[0]
    
    # Initialize MMM without passing the model context
    mmm = MMM(
        date_column=date_column,
        channel_columns=channel_list,
        control_columns=control_columns,
        adstock=channel_specific_transforms[first_channel]['adstock'],
        saturation=channel_specific_transforms[first_channel]['saturation']
    )
    
    # Set channel-specific transforms
    try:
        mmm.media_transforms = channel_specific_transforms
        print(f"Successfully set channel-specific transforms", file=sys.stderr)
    except Exception as e:
        print(f"Error setting media_transforms: {str(e)}", file=sys.stderr)
        # Try alternative approach
        try:
            mmm.set_transforms(transforms=channel_specific_transforms)
            print(f"Successfully set transforms via set_transforms method", file=sys.stderr)
        except Exception as e2:
            print(f"Error using set_transforms: {str(e2)}", file=sys.stderr)
    
    return mmm
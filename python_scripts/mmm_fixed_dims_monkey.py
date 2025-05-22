#!/usr/bin/env python
"""
Implementation of Channel-Dimensioned Global Priors approach for MMM with PyMC
with monkey-patching to fix the dims attribute error

This implementation creates channel-dimensioned PyMC Random Variables with dims='channel'
and monkey-patches the dims attribute onto the TensorVariable objects to ensure 
compatibility with PyMC-Marketing 0.13.1.
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


def load_json_config(config_file):
    """Load JSON configuration file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def create_mmm_model_with_fixed_priors(config_file):
    """
    Create an MMM model with fixed priors for channel parameters
    
    Args:
        config_file: Path to model configuration file
    
    Returns:
        MMM model object
    """
    # 1. Load Model Configuration
    print(f"Loading model config from {config_file}...", file=sys.stderr)
    config = load_json_config(config_file)
    
    # Extract relevant information
    channel_config = config["channels"]
    data_config = config["data"]
    
    # 2. Prepare Channel Parameters
    channels = list(channel_config.keys())
    channel_name_list = channels
    print(f"Channel names: {channel_name_list}", file=sys.stderr)
    
    # Create arrays for channel-specific parameters
    alpha_values = np.array([float(channel_config[ch]["alpha"]) for ch in channels])
    L_values = np.array([float(channel_config[ch]["L"]) for ch in channels])
    k_values = np.array([float(channel_config[ch]["k"]) for ch in channels])
    x0_values = np.array([float(channel_config[ch]["x0"]) for ch in channels])
    
    # Adstock l_max value (same for all channels in our simplified case)
    global_l_max = max([int(channel_config[ch]["l_max"]) for ch in channels])
    
    print(f"Prepared parameter arrays:", file=sys.stderr)
    print(f"alpha_values: {alpha_values}", file=sys.stderr)
    print(f"L_values: {L_values}", file=sys.stderr)
    print(f"k_values: {k_values}", file=sys.stderr)
    print(f"x0_values: {x0_values}", file=sys.stderr)
    print(f"global_l_max: {global_l_max}", file=sys.stderr)
    
    # Set up sigma values for the lightly regularized "fixed" priors
    k_sigma_val = np.maximum(np.abs(k_values * 0.001), 1e-7)
    x0_sigma_val = np.maximum(np.abs(x0_values * 0.001), 1e-2)
    
    # 3. Create PyMC Model Context
    with pm.Model(coords={"channel": channel_name_list}) as mmm_model_context:
        print(f"Created PyMC model context with channels: {channel_name_list}", file=sys.stderr)
        
        # Define Channel-Dimensioned "Fixed" Priors as Named RVs
        print(f"Creating channel-dimensioned priors...", file=sys.stderr)
        
        # Create alpha RV with channel dimensions
        alpha_rv_chan = pm.Normal("fixed_alphas_per_channel", mu=alpha_values, sigma=1e-6, dims="channel")
        if not hasattr(alpha_rv_chan, 'dims'):
            print(f"DEBUG: Monkey-patching .dims for alpha_rv_chan", file=sys.stderr)
            alpha_rv_chan.dims = ("channel",)  # Forcibly assign the expected dims tuple
        # Debug print
        print(f"DEBUG: AFTER PATCH alpha_rv_chan type: {type(alpha_rv_chan)}, hasattr .dims: {hasattr(alpha_rv_chan, 'dims')}, .dims value: {getattr(alpha_rv_chan, 'dims', 'NOT FOUND')}", file=sys.stderr)
        
        # Create L RV with channel dimensions
        L_rv_chan = pm.Normal("fixed_Ls_per_channel", mu=L_values, sigma=1e-6, dims="channel")
        if not hasattr(L_rv_chan, 'dims'):
            print(f"DEBUG: Monkey-patching .dims for L_rv_chan", file=sys.stderr)
            L_rv_chan.dims = ("channel",)
        # Debug print
        print(f"DEBUG: AFTER PATCH L_rv_chan type: {type(L_rv_chan)}, hasattr .dims: {hasattr(L_rv_chan, 'dims')}, .dims value: {getattr(L_rv_chan, 'dims', 'NOT FOUND')}", file=sys.stderr)
        
        # Create k RV with channel dimensions
        k_rv_chan = pm.Normal("fixed_ks_per_channel", mu=k_values, sigma=k_sigma_val, dims="channel")
        if not hasattr(k_rv_chan, 'dims'):
            print(f"DEBUG: Monkey-patching .dims for k_rv_chan", file=sys.stderr)
            k_rv_chan.dims = ("channel",)
        # Debug print
        print(f"DEBUG: AFTER PATCH k_rv_chan type: {type(k_rv_chan)}, hasattr .dims: {hasattr(k_rv_chan, 'dims')}, .dims value: {getattr(k_rv_chan, 'dims', 'NOT FOUND')}", file=sys.stderr)
        
        # Create x0 RV with channel dimensions
        x0_rv_chan = pm.Normal("fixed_x0s_per_channel", mu=x0_values, sigma=x0_sigma_val, dims="channel")
        if not hasattr(x0_rv_chan, 'dims'):
            print(f"DEBUG: Monkey-patching .dims for x0_rv_chan", file=sys.stderr)
            x0_rv_chan.dims = ("channel",)
        # Debug print
        print(f"DEBUG: AFTER PATCH x0_rv_chan type: {type(x0_rv_chan)}, hasattr .dims: {hasattr(x0_rv_chan, 'dims')}, .dims value: {getattr(x0_rv_chan, 'dims', 'NOT FOUND')}", file=sys.stderr)
        
        # 4. Create GLOBAL Adstock and Saturation Objects
        print(f"Creating global transform objects...", file=sys.stderr)
        global_adstock_obj = GeometricAdstock(l_max=global_l_max, priors={"alpha": alpha_rv_chan})
        global_saturation_obj = LogisticSaturation(priors={"L": L_rv_chan, "k": k_rv_chan, "x0": x0_rv_chan})
        
        # 5. Initialize MMM
        try:
            print(f"Building MMM model without explicit model context...", file=sys.stderr)
            
            # Add a dummy control column to satisfy validation
            control_column = "dummy_control"
            
            # Create MMM with minimal parameters based on validation errors
            mmm = MMM(
                # Only required parameters based on validation errors
                channel_columns=channels,
                adstock=global_adstock_obj,
                saturation=global_saturation_obj,
                control_columns=["dummy_control"],  # Minimum of 1 required
                date_column="date"  # This is required based on latest error
            )
            
            print(f"Successfully built MMM model!", file=sys.stderr)
            
            # Store useful properties for accessing later
            mmm._data_info = {
                "date_column": "date",
                "target_column": "y",
                "media_columns": channels
            }
            
            return mmm
            
        except Exception as e:
            print(f"ERROR: Failed to initialize MMM model: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mmm_fixed_dims_monkey.py config_file.json", file=sys.stderr)
        sys.exit(1)
        
    config_file = sys.argv[1]
    mmm = create_mmm_model_with_fixed_priors(config_file)
    print("Model created successfully!", file=sys.stderr)
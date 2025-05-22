#!/usr/bin/env python
"""
Enhanced MMM implementation with monkey-patching that includes model fitting

This module builds on our successful monkey-patching approach to fix the dims attribute
issue and adds functionality to fit the model and generate comprehensive results.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
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


def load_json_config(config_file):
    """
    Load JSON configuration file.
    
    Args:
        config_file: Path to model configuration file
        
    Returns:
        Dict containing configuration values
    """
    with open(config_file, 'r') as f:
        return json.load(f)


def load_data(data_file, date_column="date"):
    """
    Load CSV data for MMM modeling.
    
    Args:
        data_file: Path to CSV data file
        date_column: Name of date column
        
    Returns:
        DataFrame with data for modeling
    """
    print(f"Loading data from {data_file}...", file=sys.stderr)
    df = pd.read_csv(data_file)
    
    # Ensure date column is properly formatted
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
    
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns", file=sys.stderr)
    print(f"Columns: {df.columns.tolist()}", file=sys.stderr)
    return df


def create_and_fit_mmm_model(config_file, data_file=None, data_df=None):
    """
    Create an MMM model with fixed priors, fit it to data, and return results
    
    Args:
        config_file: Path to model configuration file
        data_file: Path to CSV data file (optional if data_df is provided)
        data_df: DataFrame with data (optional if data_file is provided)
    
    Returns:
        Dict containing model results
    """
    # 1. Load Model Configuration
    print(f"Loading model config from {config_file}...", file=sys.stderr)
    config = load_json_config(config_file)
    
    # Extract relevant information
    channel_config = config["channels"]
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    
    # 2. Load Data
    if data_df is not None:
        df = data_df
    elif data_file is not None:
        df = load_data(data_file)
    else:
        raise ValueError("Either data_file or data_df must be provided")
    
    # Get data configuration
    date_column = data_config.get("date_column", "date")
    target_column = data_config.get("response_column", "y")
    
    # 3. Prepare Channel Parameters
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
    
    # 4. Create PyMC Model Context
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
        
        # 5. Create GLOBAL Adstock and Saturation Objects
        print(f"Creating global transform objects...", file=sys.stderr)
        global_adstock_obj = GeometricAdstock(l_max=global_l_max, priors={"alpha": alpha_rv_chan})
        global_saturation_obj = LogisticSaturation(priors={"L": L_rv_chan, "k": k_rv_chan, "x0": x0_rv_chan})
        
        # 6. Initialize MMM
        try:
            print(f"Building MMM model without explicit model context...", file=sys.stderr)
            
            # Ensure we have a dummy control column for the MMM validation
            if "dummy_control" not in df.columns:
                df["dummy_control"] = 0.0
            
            # Create MMM with minimal parameters based on validation errors
            mmm = MMM(
                # Only required parameters based on validation errors
                channel_columns=channels,
                adstock=global_adstock_obj,
                saturation=global_saturation_obj,
                control_columns=["dummy_control"],
                date_column=date_column
            )
            
            print(f"Successfully built MMM model!", file=sys.stderr)
            
            # Store useful properties for debugging
            mmm._debug_info = {
                "date_column": date_column,
                "target_column": target_column,
                "media_columns": channels
            }
            
            # 7. Fit the model
            print(f"Preparing to fit the model...", file=sys.stderr)
            
            # Extract MCMC settings from config
            mcmc_settings = {
                "draws": model_config.get("iterations", 500),
                "tune": model_config.get("tuning", 200),
                "chains": model_config.get("chains", 2),
                "target_accept": 0.9,
                "return_inferencedata": True
            }
            
            print(f"Fitting with settings: {mcmc_settings}", file=sys.stderr)
            
            # Prepare data for fitting
            X = df[channels + ["dummy_control"]].copy()
            y = df[target_column].copy()
            
            # Fit the model
            print(f"Fitting model with {len(X)} data points...", file=sys.stderr)
            trace = mmm.fit(X=X, y=y, **mcmc_settings)
            print(f"Model fitting complete!", file=sys.stderr)
            
            # 8. Generate Results
            print(f"Generating model results and predictions...", file=sys.stderr)
            
            # Get model predictions (posterior predictive)
            post_pred = mmm.predict(X=X)
            
            # Generate contributions
            contributions = mmm.get_total_effects(X)
            
            # Calculate ROI
            channel_spend = X[channels].sum().to_dict()
            channel_contrib = {}
            channel_roi = {}
            
            for channel in channels:
                # Get contribution for this channel
                channel_contrib[channel] = float(contributions[channel].sum())
                
                # Calculate ROI for this channel
                spend = float(channel_spend[channel])
                contrib = channel_contrib[channel]
                
                if spend > 0:
                    channel_roi[channel] = contrib / spend
                else:
                    channel_roi[channel] = 0.0
            
            # Create final results dictionary
            results = {
                "model_info": {
                    "channels": channels,
                    "data_points": len(X),
                    "target_column": target_column,
                    "date_column": date_column
                },
                "parameters": {
                    "alpha": {ch: float(alpha_values[i]) for i, ch in enumerate(channels)},
                    "L": {ch: float(L_values[i]) for i, ch in enumerate(channels)},
                    "k": {ch: float(k_values[i]) for i, ch in enumerate(channels)},
                    "x0": {ch: float(x0_values[i]) for i, ch in enumerate(channels)}
                },
                "contributions": channel_contrib,
                "spend": channel_spend,
                "roi": channel_roi,
                "fit_summary": {
                    "prediction_mean": float(post_pred.mean()),
                    "target_mean": float(y.mean()),
                    "mape": float(np.mean(np.abs((y - post_pred) / y)) * 100)
                }
            }
            
            print(f"Results generation complete!", file=sys.stderr)
            
            return {
                "success": True,
                "mmm": mmm,
                "trace": trace,
                "results": results
            }
            
        except Exception as e:
            print(f"ERROR: Failed during model fitting: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "error": str(e)
            }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fit_mmm_with_monkey_patch.py config_file.json data_file.csv", file=sys.stderr)
        sys.exit(1)
        
    config_file = sys.argv[1]
    data_file = sys.argv[2]
    
    results = create_and_fit_mmm_model(config_file, data_file)
    
    if results["success"]:
        # Print the final results as JSON
        print(json.dumps(results["results"], indent=2))
        print("Model fitting complete and successful!", file=sys.stderr)
    else:
        print(f"Model fitting failed: {results['error']}", file=sys.stderr)
        sys.exit(1)
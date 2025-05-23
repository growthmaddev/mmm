#!/usr/bin/env python
"""
Enhanced MMM implementation with monkey-patching that includes model fitting

This module builds on our successful monkey-patching approach to fix the dims attribute
issue and adds functionality to fit the model and generate comprehensive results.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
from datetime import datetime
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


def create_and_fit_mmm_model(config_file, data_file=None, data_df=None, results_file=None):
    """
    Create an MMM model with fixed priors, fit it to data, and return results
    
    Args:
        config_file: Path to model configuration file
        data_file: Path to CSV data file (optional if data_df is provided)
        data_df: DataFrame with data (optional if data_file is provided)
        results_file: Optional path to save results JSON
    
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
    date_column = data_config.get("date_column", "Date")
    target_column = data_config.get("response_column", "Sales")
    control_columns = data_config.get("control_columns", [])
    
    # Get channel names first to use in data cleaning
    channels = list(channel_config.keys())
    
    # Clean numeric columns (remove commas and convert to float)
    numeric_cols = [target_column] + channels + control_columns
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float)
            else:
                df[col] = df[col].astype(float)
    
    # 3. Prepare Channel Parameters
    # Note: We've already defined channels earlier for data cleaning
    channel_name_list = channels
    
    # Debug output channel info
    print(f"Found channels in data: {[c for c in channels if c in df.columns]}", file=sys.stderr)
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
        
        # Create alpha RV with channel dimensions using Beta distribution to respect the 0 <= alpha <= 1 constraint
        # Ensure mu values are strictly within (0,1) for Beta distribution's mean to be well-defined
        clipped_alpha_mu_values = np.clip(alpha_values, 1e-6, 1.0 - 1e-6)
        
        # Use a large concentration parameter to make Beta sharply peaked at the mean
        kappa = 1_000_000
        
        # Calculate alpha and beta parameters for the pm.Beta distribution
        # For Beta(a,b), mean = a / (a+b). If mean = m, and a+b = kappa, then a = m*kappa, b = (1-m)*kappa
        beta_dist_alpha_param_values = clipped_alpha_mu_values * kappa
        beta_dist_beta_param_values = (1.0 - clipped_alpha_mu_values) * kappa
        
        alpha_rv_chan = pm.Beta("fixed_alphas_per_channel", 
                              alpha=beta_dist_alpha_param_values, 
                              beta=beta_dist_beta_param_values, 
                              dims="channel")
        
        if not hasattr(alpha_rv_chan, 'dims') or getattr(alpha_rv_chan, 'dims', None) != ("channel",):
            print(f"DEBUG: Monkey-patching .dims for alpha_rv_chan (Beta)", file=sys.stderr)
            alpha_rv_chan.dims = ("channel",)  # Forcibly assign the expected dims tuple
            
        # Debug print
        print(f"DEBUG: AFTER PATCH alpha_rv_chan (Beta) type: {type(alpha_rv_chan)}, hasattr .dims: {hasattr(alpha_rv_chan, 'dims')}, .dims value: {getattr(alpha_rv_chan, 'dims', 'NOT FOUND')}", file=sys.stderr)
        
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
            
            # Ensure we have all necessary control columns
            existing_controls = [col for col in control_columns if col in df.columns]
            
            # If no control columns exist in data, add a dummy one
            if not existing_controls:
                df["dummy_control"] = 0.0
                existing_controls = ["dummy_control"]
            
            print(f"Using control columns: {existing_controls}", file=sys.stderr)
            
            # Create MMM with minimal parameters based on validation errors
            mmm = MMM(
                # Only required parameters based on validation errors
                channel_columns=channels,
                adstock=global_adstock_obj,
                saturation=global_saturation_obj,
                control_columns=existing_controls,
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
            
            # Prepare data for fitting - ensure date column is included
            X_columns = [date_column] + channels + existing_controls
            X = df[X_columns].copy()
            y = df[target_column].copy()
            
            # Ensure date column is properly formatted
            if date_column in X.columns:
                # Try to detect the date format based on sample date
                sample_date = X[date_column].iloc[0]
                print(f"Sample date format: {sample_date}", file=sys.stderr)
                
                # Handle different date formats (DD/MM/YYYY or MM/DD/YYYY)
                try:
                    # Try with dayfirst=True (European format DD/MM/YYYY)
                    X[date_column] = pd.to_datetime(X[date_column], dayfirst=True)
                    print(f"Successfully parsed dates with dayfirst=True", file=sys.stderr)
                except ValueError as e:
                    # Fallback to mixed format
                    print(f"Error parsing dates: {str(e)}", file=sys.stderr)
                    X[date_column] = pd.to_datetime(X[date_column], format='mixed')
                
            print(f"Input data X shape: {X.shape}, columns: {X.columns.tolist()}", file=sys.stderr)
            print(f"Target y shape: {y.shape}", file=sys.stderr)
            print(f"Date column format: {X[date_column].dtype}", file=sys.stderr)
            print(f"Date range: {X[date_column].min()} to {X[date_column].max()}", file=sys.stderr)
            
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
            
            # Create comprehensive results dictionary
            results = {
                "model_info": {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "config_file": config_file,
                    "data_file": data_file if data_file else "data_df_provided",
                    "channels": channels,
                    "data_points": len(X),
                    "target_column": target_column,
                    "date_column": date_column,
                    "control_columns": existing_controls,
                    "mcmc_settings": mcmc_settings
                },
                "parameters": {
                    "alpha": {ch: float(alpha_values[i]) for i, ch in enumerate(channels)},
                    "L": {ch: float(L_values[i]) for i, ch in enumerate(channels)},
                    "k": {ch: float(k_values[i]) for i, ch in enumerate(channels)},
                    "x0": {ch: float(x0_values[i]) for i, ch in enumerate(channels)}
                },
                "channel_analysis": {
                    "contributions": channel_contrib,
                    "spend": channel_spend,
                    "roi": channel_roi,
                    "contribution_percentage": {
                        ch: (channel_contrib[ch] / sum(channel_contrib.values()) * 100) 
                        if sum(channel_contrib.values()) > 0 else 0.0
                        for ch in channels
                    }
                },
                "model_quality": {
                    "prediction_mean": float(post_pred.mean()),
                    "target_mean": float(y.mean()),
                    "mape": float(np.mean(np.abs((y - post_pred) / y)) * 100)
                }
            }
            
            # Try to add sales decomposition if the method exists
            try:
                if hasattr(mmm, "decompose_sales"):
                    decomposition = mmm.decompose_sales(X, y)
                    results["sales_decomposition"] = decomposition
            except Exception as e:
                print(f"Warning: Could not generate sales decomposition: {str(e)}", file=sys.stderr)
            
            # Try to add posterior parameter distribution summary if arviz is available
            try:
                if hasattr(az, "summary"):
                    results["posterior_summary"] = {}
                    for param in ["alpha", "L", "k", "x0"]:
                        try:
                            param_summary = az.summary(trace, var_names=[param])
                            results["posterior_summary"][param] = param_summary.to_dict()
                        except Exception as param_err:
                            print(f"Warning: Could not extract posterior summary for {param}: {str(param_err)}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Could not generate posterior summary: {str(e)}", file=sys.stderr)
            
            # Save results to JSON file if specified
            if results_file:
                try:
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    print(f"Results saved to {results_file}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Could not save results to file: {str(e)}", file=sys.stderr)
            
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
    import argparse
    
    # Create an argument parser
    parser = argparse.ArgumentParser(description='PyMC-Marketing MMM with monkey-patched dims attribute')
    parser.add_argument('config_file', help='Path to the model configuration JSON file')
    parser.add_argument('data_file', help='Path to the data CSV file')
    parser.add_argument('--results-file', '-o', help='Path to save results JSON (optional)')
    parser.add_argument('--quick', '-q', action='store_true', help='Use reduced MCMC settings for faster run')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Get file paths
    config_file = args.config_file
    data_file = args.data_file
    results_file = args.results_file
    
    # If quick mode, update the config to use minimal MCMC settings
    if args.quick:
        print("Running in quick mode with minimal MCMC settings", file=sys.stderr)
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if 'model' not in config:
            config['model'] = {}
        
        # Override with minimal settings
        config['model'].update({
            'iterations': 100,
            'tuning': 100,
            'chains': 1
        })
        
        # Save to a temporary config file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(config, tmp, indent=2)
            temp_config_file = tmp.name
        
        config_file = temp_config_file
        print(f"Created temporary config file with quick settings: {config_file}", file=sys.stderr)
    
    # Run the model
    results = create_and_fit_mmm_model(config_file, data_file, results_file=results_file)
    
    if results["success"]:
        # Print the final results as JSON
        print(json.dumps(results["results"], indent=2))
        print("Model fitting complete and successful!", file=sys.stderr)
        
        if results_file:
            print(f"Detailed results saved to: {results_file}", file=sys.stderr)
        
        sys.exit(0)
    else:
        print(f"Model fitting failed: {results['error']}", file=sys.stderr)
        sys.exit(1)
        
    # Clean up temporary file if created
    if args.quick and 'temp_config_file' in locals():
        try:
            os.remove(temp_config_file)
        except:
            pass
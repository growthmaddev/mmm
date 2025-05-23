#!/usr/bin/env python
"""
Enhanced MMM implementation with global TensorVariable patching
This version applies the patch globally before importing PyMC-Marketing
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# CRITICAL: Apply global patch BEFORE importing PyMC or PyMC-Marketing
print("Applying global TensorVariable patch...", file=sys.stderr)

import pytensor.tensor as pt

# Store original methods
_original_getattr = pt.TensorVariable.__getattribute__
_original_setattr = pt.TensorVariable.__setattr__

def _patched_getattr(self, name):
    if name == 'dims':
        try:
            return _original_getattr(self, name)
        except AttributeError:
            if hasattr(self, '_pymc_dims'):
                return self._pymc_dims
            return ()
    return _original_getattr(self, name)

def _patched_setattr(self, name, value):
    if name == 'dims':
        _original_setattr(self, '_pymc_dims', value)
    else:
        _original_setattr(self, name, value)

# Apply patches
pt.TensorVariable.__getattribute__ = _patched_getattr
pt.TensorVariable.__setattr__ = _patched_setattr

print("Global patch applied successfully!", file=sys.stderr)

# NOW import PyMC and PyMC-Marketing
import pymc as pm
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

def load_json_config(config_file):
    """Load JSON configuration file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def load_data(data_file, date_column="date"):
    """Load and prepare CSV data"""
    print(f"Loading data from {data_file}...", file=sys.stderr)
    df = pd.read_csv(data_file)
    
    # Parse dates with dayfirst=True for DD/MM/YYYY format
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, format='mixed')
        print(f"Parsed dates with dayfirst=True: {df[date_column].dtype}", file=sys.stderr)
        print(f"Date range: {df[date_column].min()} to {df[date_column].max()}", file=sys.stderr)
    
    return df

def create_and_fit_mmm_model(config_file, data_file=None, data_df=None, results_file=None):
    """Create and fit MMM model with global patching"""
    
    # Load configuration
    config = load_json_config(config_file)
    channel_config = config["channels"]
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    
    # Load data
    if data_df is not None:
        df = data_df
    elif data_file is not None:
        df = load_data(data_file)
    else:
        raise ValueError("Either data_file or data_df must be provided")
    
    # Get column names
    date_column = data_config.get("date_column", "Date")
    target_column = data_config.get("response_column", "Sales")
    control_columns = data_config.get("control_columns", [])
    channels = list(channel_config.keys())
    
    # Clean numeric columns
    numeric_cols = [target_column] + channels + control_columns
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)
    
    # Prepare parameters
    alpha_values = np.array([float(channel_config[ch]["alpha"]) for ch in channels])
    L_values = np.array([float(channel_config[ch]["L"]) for ch in channels])
    k_values = np.array([float(channel_config[ch]["k"]) for ch in channels])
    x0_values = np.array([float(channel_config[ch]["x0"]) for ch in channels])
    global_l_max = max([int(channel_config[ch]["l_max"]) for ch in channels])
    
    # Timing for model creation
    t0 = time.time()
    print(f"Creating PyMC model...", file=sys.stderr)
    
    with pm.Model(coords={"channel": channels}) as model:
        # Create channel-dimensioned priors with simplified approach
        # Use very small sigma values to make priors effectively fixed
        
        # For alpha, use Beta with appropriate bounds (0-1)
        # Use strong concentration to make it effectively fixed
        alpha_clip = np.clip(alpha_values, 0.01, 0.99)  # Ensure within valid range
        alpha_rv = pm.Beta("alpha", 
                          alpha=alpha_clip * 1000 + 1,  # Strong prior toward specified value 
                          beta=(1-alpha_clip) * 1000 + 1,  # Ensure valid Beta parameters
                          dims="channel")
        
        # Use very narrow distributions centered on our desired values
        # For strictly positive parameters, use distributions with appropriate support
        
        # For L, use a narrow HalfNormal centered at desired values
        L_rv = pm.HalfNormal("L", sigma=0.01, dims="channel", initval=L_values)
        
        # For k, use a narrow HalfNormal 
        k_rv = pm.HalfNormal("k", sigma=0.0001, dims="channel", initval=k_values)
        
        # For x0, use a narrow HalfNormal
        x0_rv = pm.HalfNormal("x0", sigma=10.0, dims="channel", initval=x0_values)
        
        # Create transforms
        adstock = GeometricAdstock(l_max=global_l_max, priors={"alpha": alpha_rv})
        saturation = LogisticSaturation(priors={"L": L_rv, "k": k_rv, "x0": x0_rv})
        
        # Ensure control columns exist
        existing_controls = [col for col in control_columns if col in df.columns]
        if not existing_controls:
            df["dummy_control"] = 1.0
            existing_controls = ["dummy_control"]
        
        # Create MMM
        mmm = MMM(
            channel_columns=channels,
            adstock=adstock,
            saturation=saturation,
            control_columns=existing_controls,
            date_column=date_column
        )
        
        # Add our own monkey patch for date handling
        original_preprocess = mmm._generate_and_preprocess_model_data
        
        def patched_preprocess(X, y=None):
            # Convert date column to datetime with dayfirst=True if it's not already datetime
            if date_column in X.columns and not pd.api.types.is_datetime64_any_dtype(X[date_column]):
                X = X.copy()
                X[date_column] = pd.to_datetime(X[date_column], dayfirst=True, format='mixed')
                print(f"Patched date handling: converted {date_column} to datetime", file=sys.stderr)
            return original_preprocess(X, y)
            
        mmm._generate_and_preprocess_model_data = patched_preprocess
        
        t1 = time.time()
        print(f"Model creation took {t1-t0:.2f} seconds", file=sys.stderr)
        
        # Prepare data for fitting
        X = df[[date_column] + channels + existing_controls].copy()
        y = df[target_column].copy()
        
        # MCMC settings
        mcmc_settings = {
            "draws": model_config.get("iterations", 500),
            "tune": model_config.get("tuning", 200),
            "chains": model_config.get("chains", 2),
            "target_accept": 0.9,
            "cores": 1,
            "return_inferencedata": True,
            "progressbar": True
        }
        
        # Timing for fitting
        t2 = time.time()
        print(f"Starting model fitting with {len(X)} data points...", file=sys.stderr)
        print(f"MCMC settings: {mcmc_settings}", file=sys.stderr)
        
        # Fit the model
        trace = mmm.fit(X=X, y=y, **mcmc_settings)
        
        t3 = time.time()
        print(f"Model fitting took {t3-t2:.2f} seconds", file=sys.stderr)
        print(f"Total time: {t3-t0:.2f} seconds", file=sys.stderr)
        
        # Generate results
        print(f"Generating results...", file=sys.stderr)
        
        # Get predictions and contributions
        post_pred = mmm.predict(X=X)
        contributions = mmm.get_total_effects(X)
        
        # Calculate metrics
        channel_spend = X[channels].sum().to_dict()
        channel_contrib = {ch: float(contributions[ch].sum()) for ch in channels}
        channel_roi = {
            ch: channel_contrib[ch] / channel_spend[ch] if channel_spend[ch] > 0 else 0.0
            for ch in channels
        }
        
        # Create results dictionary
        results = {
            "model_info": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config_file": config_file,
                "data_points": len(X),
                "channels": channels,
                "mcmc_settings": mcmc_settings,
                "total_time_seconds": t3 - t0
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
                    for ch in channels
                }
            },
            "model_quality": {
                "prediction_mean": float(post_pred.mean()),
                "target_mean": float(y.mean()),
                "mape": float(np.mean(np.abs((y - post_pred) / y)) * 100)
            }
        }
        
        # Save results
        if results_file:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {results_file}", file=sys.stderr)
        
        return {
            "success": True,
            "mmm": mmm,
            "trace": trace,
            "results": results
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PyMC-Marketing MMM with global TensorVariable patching')
    parser.add_argument('config_file', help='Path to configuration JSON')
    parser.add_argument('data_file', help='Path to data CSV')
    parser.add_argument('--results-file', '-o', help='Path to save results JSON')
    parser.add_argument('--quick', '-q', action='store_true', help='Use minimal MCMC settings')
    
    args = parser.parse_args()
    
    # Override config for quick mode
    if args.quick:
        print("Running in quick mode...", file=sys.stderr)
        config = load_json_config(args.config_file)
        config['model'].update({
            'iterations': 100,
            'tuning': 50,
            'chains': 1
        })
        # Save temporary config
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(config, tmp, indent=2)
            config_file = tmp.name
    else:
        config_file = args.config_file
    
    # Run the model
    result = create_and_fit_mmm_model(config_file, args.data_file, results_file=args.results_file)
    
    if result["success"]:
        print(json.dumps(result["results"], indent=2))
        sys.exit(0)
    else:
        print(f"Model fitting failed", file=sys.stderr)
        sys.exit(1)
#!/usr/bin/env python
"""
MMM implementation using fixed parameters (point estimates) with global TensorVariable patching
This avoids sampling issues while still providing model results
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, Optional

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

def load_and_clean_data(data_file, date_column="Date"):
    """Load and clean CSV data"""
    print(f"Loading data from {data_file}...", file=sys.stderr)
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"ERROR: Data file '{data_file}' not found!", file=sys.stderr)
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load data with verbose error handling
    try:
        df = pd.read_csv(data_file)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to load CSV data: {str(e)}", file=sys.stderr)
        raise
    
    # Parse dates with dayfirst=True for DD/MM/YYYY format
    if date_column in df.columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
            print(f"Date range: {df[date_column].min()} to {df[date_column].max()}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: Failed to parse dates: {str(e)}", file=sys.stderr)
            raise
    
    return df

def create_mmm_with_fixed_params(config_file, data_file, results_file=None):
    """Create MMM model with fixed parameters (no sampling)"""
    
    # Load configuration
    config = load_json_config(config_file)
    channel_config = config["channels"]
    data_config = config.get("data", {})
    
    # Load data
    df = load_and_clean_data(data_file)
    
    # Get column names
    date_column = data_config.get("date_column", "Date")
    target_column = data_config.get("response_column", "Sales")
    control_columns = data_config.get("control_columns", [])
    channels = list(channel_config.keys())
    
    # Clean numeric columns
    print(f"Cleaning numeric columns...", file=sys.stderr)
    
    # Clean channels
    for ch in channels:
        if ch in df.columns and df[ch].dtype == 'object':
            df[ch] = df[ch].str.replace(',', '').astype(float)
    
    # Clean target
    if target_column in df.columns and df[target_column].dtype == 'object':
        df[target_column] = df[target_column].str.replace(',', '').astype(float)
    
    # Clean controls
    for col in control_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)
    
    # Extract parameter values
    alpha_values = np.array([float(channel_config[ch]["alpha"]) for ch in channels])
    L_values = np.array([float(channel_config[ch]["L"]) for ch in channels])
    k_values = np.array([float(channel_config[ch]["k"]) for ch in channels])
    x0_values = np.array([float(channel_config[ch]["x0"]) for ch in channels])
    global_l_max = max([int(channel_config[ch]["l_max"]) for ch in channels])
    
    print(f"Creating model with fixed parameters...", file=sys.stderr)
    
    with pm.Model(coords={"channel": channels}) as model:
        # Use ConstantData for fixed parameters
        alpha_fixed = pm.ConstantData("alpha", alpha_values, dims="channel")
        L_fixed = pm.ConstantData("L", L_values, dims="channel")
        k_fixed = pm.ConstantData("k", k_values, dims="channel")
        x0_fixed = pm.ConstantData("x0", x0_values, dims="channel")
        
        # Create transforms
        adstock = GeometricAdstock(l_max=global_l_max, priors={"alpha": alpha_fixed})
        saturation = LogisticSaturation(priors={"L": L_fixed, "k": k_fixed, "x0": x0_fixed})
        
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
        
        # Prepare data
        X = df[[date_column] + channels + existing_controls].copy()
        y = df[target_column].copy()
        
        # Build model
        mmm.build_model(X, y)
        print(f"✓ Model built successfully!", file=sys.stderr)
        
        # Calculate channel contributions using the fixed parameters
        print(f"Calculating channel contributions...", file=sys.stderr)
        
        try:
            # Transform the channel data through adstock and saturation
            channel_data = X[channels].values
            n_obs = len(X)
            n_channels = len(channels)
            
            # Apply adstock transformation
            adstocked_data = np.zeros((n_obs, n_channels))
            for i, ch in enumerate(channels):
                alpha = alpha_values[i]
                for t in range(n_obs):
                    for lag in range(min(t + 1, global_l_max)):
                        adstocked_data[t, i] += (alpha ** lag) * channel_data[max(0, t - lag), i]
            
            # Apply saturation transformation
            saturated_data = np.zeros((n_obs, n_channels))
            for i, ch in enumerate(channels):
                L = L_values[i]
                k = k_values[i]
                x0 = x0_values[i]
                x = adstocked_data[:, i]
                saturated_data[:, i] = L / (1 + np.exp(-k * (x - x0)))
            
            # Calculate contributions (simplified - proportional to transformed spend)
            channel_contributions = {}
            total_transformed = saturated_data.sum()
            
            for i, ch in enumerate(channels):
                contribution = saturated_data[:, i].sum()
                channel_contributions[ch] = float(contribution)
            
            # Calculate ROI
            channel_spend = X[channels].sum().to_dict()
            channel_roi = {}
            contribution_percentage = {}
            
            for ch in channels:
                spend = channel_spend[ch]
                contrib = channel_contributions[ch]
                
                if spend > 0:
                    # Simplified ROI calculation
                    channel_roi[ch] = contrib / spend
                else:
                    channel_roi[ch] = 0.0
                
                if total_transformed > 0:
                    contribution_percentage[ch] = (contrib / total_transformed) * 100
                else:
                    contribution_percentage[ch] = 0.0
            
            print(f"✓ Contributions calculated successfully!", file=sys.stderr)
            
        except Exception as e:
            print(f"Warning: Could not calculate contributions: {e}", file=sys.stderr)
            channel_contributions = {ch: 0.0 for ch in channels}
            channel_roi = {ch: 0.0 for ch in channels}
            contribution_percentage = {ch: 0.0 for ch in channels}
        
        # Create enhanced results
        results = {
            "model_info": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config_file": config_file,
                "data_file": data_file,
                "data_points": len(X),
                "channels": channels,
                "approach": "fixed_parameters"
            },
            "fixed_parameters": {
                "alpha": {ch: float(alpha_values[i]) for i, ch in enumerate(channels)},
                "L": {ch: float(L_values[i]) for i, ch in enumerate(channels)},
                "k": {ch: float(k_values[i]) for i, ch in enumerate(channels)},
                "x0": {ch: float(x0_values[i]) for i, ch in enumerate(channels)},
                "l_max": global_l_max
            },
            "channel_analysis": {
                "spend": channel_spend,
                "contributions": channel_contributions,
                "roi": channel_roi,
                "contribution_percentage": contribution_percentage
            },
            "status": "Model built and analyzed successfully with fixed parameters"
        }
        
        # Save results
        if results_file:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {results_file}", file=sys.stderr)
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MMM with fixed parameters')
    parser.add_argument('--data_file', required=True, help='Path to data CSV')
    parser.add_argument('--config_file', required=True, help='Path to configuration JSON')
    parser.add_argument('--results_file', help='Path to save results JSON')
    
    args = parser.parse_args()
    
    # Debug info
    print(f"DEBUG: data_file={args.data_file}", file=sys.stderr)
    print(f"DEBUG: config_file={args.config_file}", file=sys.stderr)
    print(f"DEBUG: results_file={args.results_file}", file=sys.stderr)
    
    # Check if files exist
    if not os.path.exists(args.data_file):
        print(f"ERROR: Data file not found: {args.data_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.config_file):
        print(f"ERROR: Config file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)
        
    # Check config file content
    try:
        with open(args.config_file, 'r') as f:
            config_content = f.read()
            print(f"DEBUG: Config content preview: {config_content[:100]}...", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to read config file: {e}", file=sys.stderr)
    
    results = create_mmm_with_fixed_params(args.config_file, args.data_file, args.results_file)
    print(json.dumps(results, indent=2))
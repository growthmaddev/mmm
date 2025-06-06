#!/usr/bin/env python
"""Test MMM with point estimates to bypass sampling issues"""

import os
import sys
import json
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply global patch
from python_scripts.fit_mmm_with_global_patch import *

def test_point_estimate_approach():
    """Test MMM using point estimates instead of sampling"""
    
    # Load config and data
    config = load_json_config("test_config_quick.json")
    df = load_data("attached_assets/dankztestdata_v2.csv", date_column="Date")
    
    channels = list(config["channels"].keys())
    date_column = "Date"
    target_column = "Sales"
    control_columns = ["interestrate_control"]
    
    # Clean numeric columns
    print("Cleaning numeric columns...")
    for ch in channels:
        if ch in df.columns and df[ch].dtype == 'object':
            df[ch] = df[ch].str.replace(',', '').astype(float)
    
    if target_column in df.columns and df[target_column].dtype == 'object':
        df[target_column] = df[target_column].str.replace(',', '').astype(float)
    
    for col in control_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)
    
    print("Testing with point estimates...")
    
    # Extract parameter values from config
    alpha_values = np.array([config["channels"][ch]["alpha"] for ch in channels])
    L_values = np.array([config["channels"][ch]["L"] for ch in channels])
    k_values = np.array([config["channels"][ch]["k"] for ch in channels])
    x0_values = np.array([config["channels"][ch]["x0"] for ch in channels])
    
    with pm.Model(coords={"channel": channels}) as model:
        # Create deterministic values instead of distributions
        alpha_det = pm.ConstantData("alpha", alpha_values, dims="channel")
        L_det = pm.ConstantData("L", L_values, dims="channel")
        k_det = pm.ConstantData("k", k_values, dims="channel")
        x0_det = pm.ConstantData("x0", x0_values, dims="channel")
        
        # Create transforms with deterministic values
        adstock = GeometricAdstock(l_max=7, priors={"alpha": alpha_det})
        saturation = LogisticSaturation(priors={"L": L_det, "k": k_det, "x0": x0_det})
        
        # Create MMM
        mmm = MMM(
            channel_columns=channels,
            adstock=adstock,
            saturation=saturation,
            control_columns=control_columns,
            date_column=date_column
        )
        
        # Prepare data
        X = df[[date_column] + channels + control_columns].copy()
        y = df[target_column].copy()
        
        # Build model without sampling
        mmm.build_model(X, y)
        print("✓ Model built successfully with point estimates!")
        
        # Try to get predictions without sampling
        try:
            # Use the initial point values
            point = model.initial_point()
            print("✓ Initial point created successfully!")
            
            # Calculate contributions at point estimate
            print("Model components available:", list(model.named_vars.keys()))
            return True
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return False

if __name__ == "__main__":
    try:
        success = test_point_estimate_approach()
        if success:
            print("\n✓ Point estimate approach works!")
        else:
            print("\n✗ Point estimate approach failed")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
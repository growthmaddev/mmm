#!/usr/bin/env python
"""
Minimal test script to diagnose PyMC-Marketing compatibility issues
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pymc as pm
import traceback
from typing import Dict, Any
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

def monkey_patch_dims(tensor_var, dims_value=("channel",)):
    """Add .dims attribute to TensorVariable objects"""
    tensor_var.dims = dims_value
    return tensor_var

def test_minimal_mmm(data_path, config_path):
    """Run minimal test of MMM with monkey-patching"""
    print("Starting minimal MMM test...")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Load and process config
    with open(config_path, 'r') as f:
        raw_config = json.load(f)
    
    # Extract key parameters
    channels = list(raw_config.get("channels", {}).keys())
    date_column = raw_config.get("data", {}).get("date_column", "Date")
    target_column = raw_config.get("data", {}).get("response_column", "Sales")
    control_columns = raw_config.get("data", {}).get("control_columns", [])
    
    print(f"Channels: {channels}")
    print(f"Date column: {date_column}")
    print(f"Target column: {target_column}")
    print(f"Control columns: {control_columns}")
    
    # Step 1: Initialize model and test monkey-patching
    print("\n=== Step 1: Model Initialization ===")
    try:
        with pm.Model(coords={"channel": channels}) as model:
            # Create channel-dimensioned arrays for parameters
            alpha_values = np.array([raw_config["channels"][ch].get("alpha", 0.5) for ch in channels])
            l_max_values = np.array([raw_config["channels"][ch].get("l_max", 8) for ch in channels])
            L_values = np.array([raw_config["channels"][ch].get("L", 1.0) for ch in channels])
            k_values = np.array([raw_config["channels"][ch].get("k", 0.0001) for ch in channels])
            x0_values = np.array([raw_config["channels"][ch].get("x0", 50000.0) for ch in channels])
            
            print(f"Parameter values:")
            print(f"  alpha: {alpha_values}")
            print(f"  L: {L_values}")
            print(f"  k: {k_values}")
            print(f"  x0: {x0_values}")
            
            # Create RVs with proper distributions and dimensions
            alpha_rv = pm.Beta("alpha_rv", alpha=alpha_values*10, beta=(1-alpha_values)*10, dims="channel")
            L_rv = pm.Normal("L_rv", mu=L_values, sigma=1e-6, dims="channel")
            k_rv = pm.Normal("k_rv", mu=k_values, sigma=np.maximum(np.abs(k_values * 0.001), 1e-7), dims="channel")
            x0_rv = pm.Normal("x0_rv", mu=x0_values, sigma=np.maximum(np.abs(x0_values * 0.001), 1), dims="channel")
            
            # Apply monkey-patch to ensure .dims attribute exists
            print("Applying monkey-patch to add .dims attribute...")
            monkey_patch_dims(alpha_rv)
            monkey_patch_dims(L_rv)
            monkey_patch_dims(k_rv)
            monkey_patch_dims(x0_rv)
            
            # Verify dims attributes
            print(f"After patch - alpha_rv.dims: {getattr(alpha_rv, 'dims', 'Missing')}")
            print(f"After patch - L_rv.dims: {getattr(L_rv, 'dims', 'Missing')}")
            
            # Create global transform objects
            print("Creating transform objects...")
            global_l_max = int(np.max(l_max_values))
            print(f"Using global l_max: {global_l_max}")
            
            adstock = GeometricAdstock(l_max=global_l_max, priors={"alpha": alpha_rv})
            saturation = LogisticSaturation(priors={"L": L_rv, "k": k_rv, "x0": x0_rv})
            
            # Create MMM object
            print("Creating MMM object...")
            mmm = MMM(
                date_column=date_column,
                channel_columns=channels,
                control_columns=control_columns if control_columns else None,
                adstock=adstock,
                saturation=saturation
            )
            
            print("MMM object created successfully!")
            
            # Step 2: Test initial point and logp calculation
            print("\n=== Step 2: Testing logp calculation ===")
            try:
                # Get initial point
                print("Getting initial point...")
                point = model.initial_point()
                print(f"Initial point keys: {list(point.keys())}")
                
                # Calculate logp
                print("Calculating logp...")
                logp = model.logp(point)
                print(f"logp result: {logp}")
                
                # Step 3: Test minimal model fitting
                print("\n=== Step 3: Testing minimal model fitting ===")
                try:
                    # Prepare data
                    print("Preparing data for fit...")
                    if not pd.api.types.is_datetime64_dtype(df[date_column]):
                        print(f"Converting {date_column} to datetime...")
                        df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
                    
                    # Try to fit the model with minimal samples
                    print("Fitting model with minimal samples...")
                    idata = mmm.fit(
                        data=df,
                        target=target_column,
                        draws=5,
                        tune=5,
                        chains=1,
                        return_inferencedata=True
                    )
                    
                    print("Model fitting successful!")
                    
                except Exception as e:
                    print(f"Error during model fitting: {str(e)}")
                    traceback.print_exc()
            
            except Exception as e:
                print(f"Error during logp calculation: {str(e)}")
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        traceback.print_exc()
    
    print("\nTest completed.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} DATA_PATH CONFIG_PATH")
        sys.exit(1)
    
    data_path = sys.argv[1]
    config_path = sys.argv[2]
    
    test_minimal_mmm(data_path, config_path)
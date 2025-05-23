#!/usr/bin/env python
"""
Quick diagnostic test for the monkey-patching implementation

This script tests the critical parts of the MMM implementation:
1. Model initialization with monkey-patched dims
2. Early model fitting stages to catch parameter validation errors
3. logp() execution

It uses minimal MCMC settings and adds extra debug output.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pymc as pm
import warnings
import traceback
from typing import Dict, Any, List, Optional

# Import the monkey-patching implementation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from python_scripts.fit_mmm_with_monkey_patch import (
    create_mmm_with_channel_dimensions, 
    train_mmm_with_monkey_patch
)

def run_quick_diagnosis(
    config_path: str,
    data_path: str,
    debug: bool = True
) -> None:
    """
    Run a quick diagnosis of the MMM implementation with monkey-patching
    
    Args:
        config_path: Path to configuration file
        data_path: Path to data file
        debug: Whether to print debug information
    """
    # Load configuration
    print(f"Loading model config from {config_path}...")
    with open(config_path, 'r') as f:
        raw_config = json.load(f)
    
    # Convert config format if needed
    if "channels" in raw_config:
        # Convert from test_config_quick.json format
        config = {
            "targetColumn": raw_config.get("data", {}).get("response_column", "Sales"),
            "dateColumn": raw_config.get("data", {}).get("date_column", "Date"),
            "channelColumns": raw_config.get("channels", {}),
            "controlColumns": raw_config.get("data", {}).get("control_columns", []),
            "adstockSettings": {
                "channel_specific_params": {}
            },
            "saturationSettings": {
                "channel_specific_params": {}
            },
            "mcmcParams": {
                "draws": 10,  # Very minimal for quick testing
                "tune": 10,
                "chains": 1,
                "targetAccept": 0.9
            }
        }
        
        # Map channel parameters to the expected format
        for channel, params in raw_config.get("channels", {}).items():
            config["adstockSettings"]["channel_specific_params"][channel] = {
                "adstock_alpha": params.get("alpha", 0.5),
                "adstock_l_max": params.get("l_max", 8)
            }
            config["saturationSettings"]["channel_specific_params"][channel] = {
                "saturation_L": params.get("L", 1.0),
                "saturation_k": params.get("k", 0.0001),
                "saturation_x0": params.get("x0", 50000.0)
            }
        
        # Add default values
        config["adstockSettings"]["default"] = {
            "adstock_alpha": 0.5,
            "adstock_l_max": 8
        }
        config["saturationSettings"]["default"] = {
            "saturation_L": 1.0,
            "saturation_k": 0.0001,
            "saturation_x0": 50000.0
        }
    else:
        config = raw_config
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Get key configuration values
    target_column = config.get('targetColumn', 'Sales')
    date_column = config.get('dateColumn', 'Date')
    channel_columns = config.get('channelColumns', {})
    
    if isinstance(channel_columns, dict):
        channel_name_list = list(channel_columns.keys())
    else:
        channel_name_list = channel_columns
    
    print(f"Channel names: {channel_name_list}")
    
    try:
        # Step 1: Test model initialization only
        print("===== Step 1: Testing Model Initialization =====")
        
        with pm.Model(coords={"channel": channel_name_list}) as mmm_model_context:
            try:
                # Call the function to create the MMM model
                mmm, alpha_rv_chan, L_rv_chan, k_rv_chan, x0_rv_chan = create_mmm_with_channel_dimensions(
                    df, config, mmm_model_context, debug=True
                )
                print("SUCCESS: Successfully initialized MMM model with monkey-patched dims!")
                
                # Step 2: Test model.logp() execution
                print("\n===== Step 2: Testing model.logp() execution =====")
                try:
                    # Convert data to the format expected by the logp function
                    X = df.copy()
                    if not pd.api.types.is_datetime64_dtype(X[date_column]):
                        print(f"Converting {date_column} to datetime...")
                        X[date_column] = pd.to_datetime(X[date_column], dayfirst=True)
                    
                    # Extract target variable
                    y = X[target_column].copy()
                    
                    # Prepare model inputs
                    print("Preparing model inputs for logp test...")
                    # Get the current point (values for all RVs in the model)
                    point = mmm_model_context.initial_point()
                    
                    # Debug point values for the random variables
                    if debug:
                        print(f"Initial point keys: {list(point.keys())}")
                        # Show values for our RVs if they exist in the point
                        for rv_name in ["fixed_alphas_ch", "fixed_Ls_ch", "fixed_ks_ch", "fixed_x0s_ch"]:
                            if rv_name in point:
                                print(f"Initial {rv_name}: {point[rv_name]}")
                    
                    # Try to compute logp
                    print("Computing model.logp()...")
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        try:
                            logp_value = mmm_model_context.logp(point)
                            print(f"SUCCESS: model.logp() returned: {logp_value}")
                        except Exception as e:
                            print(f"ERROR during model.logp(): {str(e)}")
                            traceback.print_exc()
                            
                            # Try to locate exactly which part of logp is failing
                            print("\nInvestigating logp failure...")
                            for rv in mmm_model_context.basic_RVs:
                                try:
                                    rv_logp = rv.logp(point)
                                    print(f"RV {rv.name} logp: {rv_logp}")
                                except Exception as e:
                                    print(f"ERROR calculating logp for {rv.name}: {str(e)}")
                        
                        # Print any warnings that occurred
                        for warning in w:
                            print(f"WARNING during logp: {warning.message}")
                
                except Exception as e:
                    print(f"ERROR during logp test: {str(e)}")
                    traceback.print_exc()
                
                # Step 3: Test minimal model fitting
                print("\n===== Step 3: Testing minimal model fitting (10 samples) =====")
                try:
                    # Try to run a minimal fit
                    print("Starting minimal model fitting...")
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        try:
                            idata = mmm.fit(
                                data=X, 
                                target=target_column,
                                draws=10,  # Very minimal for quick testing
                                tune=10,
                                chains=1,
                                return_inferencedata=True,
                                progressbar=True
                            )
                            print("SUCCESS: Model fitting completed!")
                        except Exception as e:
                            print(f"ERROR during model fitting: {str(e)}")
                            traceback.print_exc()
                        
                        # Print any warnings that occurred
                        for warning in w:
                            print(f"WARNING during fitting: {warning.message}")
                
                except Exception as e:
                    print(f"ERROR during model fitting test: {str(e)}")
                    traceback.print_exc()
            
            except Exception as e:
                print(f"ERROR during model initialization: {str(e)}")
                traceback.print_exc()
    
    except Exception as e:
        print(f"ERROR during test execution: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick diagnostic test for MMM implementation")
    parser.add_argument("config_path", help="Path to configuration file")
    parser.add_argument("data_path", help="Path to data file")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    run_quick_diagnosis(args.config_path, args.data_path, args.debug)
#!/usr/bin/env python3
"""
Test script to debug PyMC-Marketing data extraction for Channel Impact tab

This script runs a minimal MMM model and focuses on detailed logging during 
the post-processing phase to identify where train_mmm.py might be getting stuck.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Set environment variables for the test
os.environ['DATA_FILE'] = 'test_data.csv'
os.environ['CONFIG_FILE'] = 'test_mmm_config.json' 
os.environ['OUTPUT_FILE'] = 'test_channel_impact_output.json'

# Create a minimal test configuration if it doesn't exist
def create_test_config():
    """Create a minimal configuration for testing if needed"""
    if not os.path.exists('test_mmm_config.json'):
        config = {
            "target_variable": "Sales",
            "date_variable": "Date",
            "channel_columns": ["TV_Spend", "Radio_Spend", "Social_Spend"],
            "control_variables": ["Promo"],
            "model_settings": {
                "mcmc_samples": 10,  # Very small for quick testing
                "mcmc_tune": 5,      # Minimal tuning
                "chains": 1,         # Single chain for speed
                "cores": 1,          # Single core for compatibility
                "random_seed": 42
            }
        }
        
        with open('test_mmm_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("Created test configuration with minimal MCMC settings")

# Patch train_mmm.py with detailed logging
def patch_train_mmm():
    """Add detailed logging to train_mmm.py by monkey patching"""
    
    # Import the original train_mmm module
    sys.path.insert(0, 'python_scripts')
    import train_mmm
    
    # Add timestamp logging function
    def log_timestamp(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)
    
    # Save original train_model function
    original_train_model = train_mmm.train_model
    
    # Create patched version with detailed logging
    def patched_train_model(df, config):
        """Patched version of train_model with detailed logging"""
        log_timestamp("==== STARTING PATCHED TRAIN_MODEL FUNCTION ====")
        log_timestamp(f"DataFrame shape: {df.shape}")
        log_timestamp(f"Config: {config}")
        
        # Apply minimal MCMC settings for quick testing
        log_timestamp("Applying minimal MCMC settings for debugging")
        if hasattr(train_mmm, 'mmm'):
            log_timestamp("Reducing MCMC parameters to minimum for testing")
            mcmc_draws = 10
            mcmc_tune = 5
        
        # Override mmm.fit to add timing logs
        from pymc_marketing.mmm import MMM
        original_fit = MMM.fit
        
        def patched_fit(self, *args, **kwargs):
            log_timestamp("==== STARTING MMM.FIT() CALL ====")
            log_timestamp(f"MMM.fit() parameters: draws={kwargs.get('draws', 'default')}, tune={kwargs.get('tune', 'default')}")
            # Override with minimal settings for quick testing
            kwargs['draws'] = mcmc_draws if 'mcmc_draws' in locals() else 10
            kwargs['tune'] = mcmc_tune if 'mcmc_tune' in locals() else 5
            kwargs['chains'] = 1
            kwargs['cores'] = 1
            log_timestamp(f"Overridden MMM.fit() parameters: draws={kwargs['draws']}, tune={kwargs['tune']}")
            
            fit_start_time = time.time()
            result = original_fit(self, *args, **kwargs)
            fit_duration = time.time() - fit_start_time
            log_timestamp(f"==== MMM.FIT() COMPLETED in {fit_duration:.2f} seconds ====")
            return result
        
        # Apply the monkey patch
        MMM.fit = patched_fit
        
        # Add logging to post-processing stages
        try:
            # Run the original function
            log_timestamp("Calling original train_model function")
            result = original_train_model(df, config)
            
            # Log key output structures
            log_timestamp("==== POST-PROCESSING COMPLETED SUCCESSFULLY ====")
            if result and 'channel_impact' in result:
                channel_impact = result['channel_impact']
                
                # Check time_series_decomposition
                if 'time_series_decomposition' in channel_impact:
                    ts_decomp = channel_impact['time_series_decomposition']
                    log_timestamp(f"time_series_decomposition: dates={len(ts_decomp.get('dates', []))}, baseline={len(ts_decomp.get('baseline', []))}")
                    log_timestamp(f"Marketing channels in decomposition: {list(ts_decomp.get('marketing_channels', {}).keys())}")
                else:
                    log_timestamp("WARNING: time_series_decomposition section missing")
                
                # Check response_curves
                if 'response_curves' in channel_impact:
                    log_timestamp(f"response_curves: {len(channel_impact['response_curves'])} channels")
                    for channel, curve in channel_impact['response_curves'].items():
                        log_timestamp(f"  - {channel}: {len(curve.get('spend_points', []))} points")
                else:
                    log_timestamp("WARNING: response_curves section missing")
                    
                # Check historical_spends
                if 'historical_spends' in channel_impact:
                    log_timestamp(f"historical_spends: {len(channel_impact['historical_spends'])} channels")
                else:
                    log_timestamp("WARNING: historical_spends section missing")
            else:
                log_timestamp("WARNING: channel_impact section missing in results")
                
            return result
            
        except Exception as e:
            log_timestamp(f"==== ERROR IN PATCHED TRAIN_MODEL: {str(e)} ====")
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
    
    # Apply the patch
    train_mmm.train_model = patched_train_model
    
    # Add detailed logging points for specific extraction functions
    
    # Patch extract_time_series_decomposition if it exists
    if hasattr(train_mmm, 'extract_time_series_decomposition'):
        original_extract_ts = train_mmm.extract_time_series_decomposition
        
        def patched_extract_ts(*args, **kwargs):
            log_timestamp("==== STARTING EXTRACT_TIME_SERIES_DECOMPOSITION ====")
            try:
                result = original_extract_ts(*args, **kwargs)
                log_timestamp("==== EXTRACT_TIME_SERIES_DECOMPOSITION COMPLETED ====")
                return result
            except Exception as e:
                log_timestamp(f"==== ERROR IN EXTRACT_TIME_SERIES_DECOMPOSITION: {str(e)} ====")
                raise
        
        train_mmm.extract_time_series_decomposition = patched_extract_ts
    
    # Patch extract_response_curves if it exists
    if hasattr(train_mmm, 'extract_response_curves'):
        original_extract_curves = train_mmm.extract_response_curves
        
        def patched_extract_curves(*args, **kwargs):
            log_timestamp("==== STARTING EXTRACT_RESPONSE_CURVES ====")
            try:
                result = original_extract_curves(*args, **kwargs)
                log_timestamp("==== EXTRACT_RESPONSE_CURVES COMPLETED ====")
                return result
            except Exception as e:
                log_timestamp(f"==== ERROR IN EXTRACT_RESPONSE_CURVES: {str(e)} ====")
                raise
        
        train_mmm.extract_response_curves = patched_extract_curves
    
    return train_mmm

# Main function
def main():
    """Run the test with detailed logging"""
    print("\n===== STARTING CHANNEL IMPACT DEBUG TEST =====\n")
    
    # Create test configuration if needed
    create_test_config()
    
    # Patch train_mmm.py with detailed logging
    train_mmm = patch_train_mmm()
    
    # Run the main function from train_mmm.py
    try:
        start_time = time.time()
        results = train_mmm.main()
        elapsed_time = time.time() - start_time
        print(f"\n===== TEST COMPLETED IN {elapsed_time:.2f} SECONDS =====\n")
        
        # Check for essential sections in the output
        check_output_file('test_channel_impact_output.json')
    except Exception as e:
        print(f"\n===== TEST FAILED: {str(e)} =====\n")
        import traceback
        traceback.print_exc()

# Check the output file structure
def check_output_file(filename):
    """Check if the output file contains the required sections"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            
            print("\n===== OUTPUT FILE STRUCTURE =====\n")
            
            if 'channel_impact' in data:
                channel_impact = data['channel_impact']
                print("channel_impact section: PRESENT")
                
                # Check key subsections
                time_series = channel_impact.get('time_series_decomposition', {})
                response_curves = channel_impact.get('response_curves', {})
                historical_spends = channel_impact.get('historical_spends', {})
                
                print(f"time_series_decomposition: {'PRESENT' if time_series else 'MISSING'}")
                if time_series:
                    print(f"  - dates: {len(time_series.get('dates', []))} entries")
                    print(f"  - baseline: {len(time_series.get('baseline', []))} entries")
                    marketing_channels = time_series.get('marketing_channels', {})
                    print(f"  - marketing_channels: {len(marketing_channels)} channels")
                    for channel, values in marketing_channels.items():
                        print(f"    - {channel}: {len(values)} values")
                
                print(f"response_curves: {'PRESENT' if response_curves else 'MISSING'}")
                if response_curves:
                    for channel, curve in response_curves.items():
                        spend_points = curve.get('spend_points', [])
                        response_values = curve.get('response_values', [])
                        print(f"  - {channel}: {len(spend_points)} points, {len(response_values)} values")
                
                print(f"historical_spends: {'PRESENT' if historical_spends else 'MISSING'}")
                if historical_spends:
                    for channel, spend in historical_spends.items():
                        print(f"  - {channel}: {spend}")
                
                # Save a simplified version for inspection
                simplified = {'channel_impact': channel_impact}
                with open('test_channel_impact_simplified.json', 'w') as f:
                    json.dump(simplified, f, indent=2)
                print("\nSaved simplified channel_impact to test_channel_impact_simplified.json")
            else:
                print("WARNING: channel_impact section MISSING in output")
        else:
            print(f"WARNING: Output file {filename} not found")
    except Exception as e:
        print(f"Error checking output file: {str(e)}")

if __name__ == "__main__":
    main()
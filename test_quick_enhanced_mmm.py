#!/usr/bin/env python
"""
Quick test script for the enhanced MMM model training
This runs with minimal settings for fast evaluation of new features
"""

import sys
import json
import pandas as pd
import os
from python_scripts.train_mmm import load_data, transform_target, run_diagnostics, train_model

# Test configuration with enhanced features
test_config = {
    'date_column': 'Date',
    'target_column': 'Sales',
    'channel_columns': {
        'PPCBrand_Spend': 'PPC Brand',
        'PPCNonBrand_Spend': 'PPC Non-Brand',
        'PPCShopping_Spend': 'PPC Shopping',
        'FBReach_Spend': 'Facebook Reach',
        'OfflineMedia_Spend': 'Offline Media'
    },
    'adstock_settings': {
        'PPCBrand_Spend': 0.3,
        'PPCNonBrand_Spend': 0.5,
        'PPCShopping_Spend': 0.4,
        'FBReach_Spend': 0.7,
        'OfflineMedia_Spend': 0.8
    },
    'saturation_settings': {
        'PPCBrand_Spend': 0.8,
        'PPCNonBrand_Spend': 0.7,
        'PPCShopping_Spend': 0.7,
        'FBReach_Spend': 0.6,
        'OfflineMedia_Spend': 0.5
    },
    'control_variables': {},
    'transform_target_method': 'auto',  # Test auto-selection
    'scale_predictors_method': 'standardize',  # Test predictor scaling
    'auto_transform': True,
    'mcmc_settings': {
        'draws': 100,  # Reduced for quick testing
        'tune': 50,
        'chains': 2,
        'target_accept': 0.9
    }
}

def main():
    # Load test data
    data_path = "test_data.csv"
    df = load_data(data_path)
    
    # Run diagnostics
    print("Running data diagnostics...", file=sys.stderr)
    diagnostics_report = run_diagnostics(df, test_config)
    
    # Train model with enhanced features
    print("Training model with enhanced features...", file=sys.stderr)
    results = train_model(df, test_config)
    
    # Output results
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
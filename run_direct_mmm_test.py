#!/usr/bin/env python
"""
Direct test of enhanced MMM functionality using real client data
This script directly runs the core functions to test the new capabilities
"""

import sys
import json
import pandas as pd
import numpy as np
from scipy import stats
import os

# Import functions directly from our train_mmm.py
sys.path.append('python_scripts')
from train_mmm import (
    transform_target, 
    scale_predictors, 
    run_diagnostics, 
    recommend_transform_method,
    parse_adstock_params,
    parse_saturation_params
)

def main():
    """Run focused tests of enhanced MMM features"""
    print("Loading test data...", file=sys.stderr)
    
    # Load and prepare test data (minimal subset)
    df = pd.read_csv("test_data.csv")
    
    # Clean data
    for col in df.columns:
        if col != 'Date':
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    # Parse dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date')
        
    # Limit to first 30 rows for faster execution
    df = df.head(30)
    
    # Define test configuration
    config = {
        'date_column': 'Date',
        'target_column': 'Sales',
        'channel_columns': {
            'PPCBrand_Spend': 'PPCBrand_Spend',
            'PPCNonBrand_Spend': 'PPCNonBrand_Spend',
            'PPCShopping_Spend': 'PPCShopping_Spend',
            'FBReach_Spend': 'FBReach_Spend',
            'OfflineMedia_Spend': 'OfflineMedia_Spend'
        }
    }
    
    target_column = config['target_column']
    channel_columns = list(config['channel_columns'].keys())
    
    # 1. Test Data Diagnostics
    print("\nTESTING DATA DIAGNOSTICS", file=sys.stderr)
    diagnostics = run_diagnostics(df, target_column, channel_columns)
    
    # 2. Test Auto-Transformation
    print("\nTESTING AUTO-TRANSFORMATION", file=sys.stderr)
    y_original = df[target_column].values
    
    # Test auto-recommendation
    recommended_method = recommend_transform_method(y_original)
    print(f"Auto-recommended transformation: {recommended_method}", file=sys.stderr)
    
    # Try multiple transformation methods
    transform_methods = ['none', 'log', 'sqrt', 'boxcox', 'yeo-johnson']
    transform_results = {}
    
    for method in transform_methods:
        print(f"Testing {method} transformation...", file=sys.stderr)
        transformed_y, params = transform_target(y_original, method=method)
        
        # Calculate skewness before and after
        skew_before = stats.skew(y_original)
        skew_after = stats.skew(transformed_y) if method != 'none' else skew_before
        
        transform_results[method] = {
            "parameters": params,
            "skewness_before": float(skew_before),
            "skewness_after": float(skew_after),
            "skewness_improvement": float(skew_before - skew_after) if method != 'none' else 0.0
        }
    
    # 3. Test Predictor Scaling
    print("\nTESTING PREDICTOR SCALING", file=sys.stderr)
    X_predictors = df[channel_columns].copy()
    
    scaling_methods = ['none', 'standardize', 'minmax', 'robust', 'log']
    scaling_results = {}
    
    for method in scaling_methods:
        print(f"Testing {method} scaling...", file=sys.stderr)
        try:
            X_scaled, scaler = scale_predictors(X_predictors, method=method)
            
            # Calculate range statistics
            channel_stats = {}
            for channel in channel_columns:
                before_min = float(X_predictors[channel].min())
                before_max = float(X_predictors[channel].max())
                before_range = before_max - before_min
                
                if method != 'none':
                    after_min = float(X_scaled[channel].min())
                    after_max = float(X_scaled[channel].max())
                    after_range = after_max - after_min
                else:
                    after_min = before_min
                    after_max = before_max
                    after_range = before_range
                
                channel_stats[channel] = {
                    "before_range": before_range,
                    "after_range": after_range
                }
            
            scaling_results[method] = {
                "channel_stats": channel_stats,
                "description": f"{method} scaling applied successfully"
            }
        except Exception as e:
            scaling_results[method] = {
                "error": str(e)
            }
    
    # 4. Test Adaptive Adstock & Saturation Parameters
    print("\nTESTING ADAPTIVE PARAMETERS", file=sys.stderr)
    
    # Test adstock parameters
    adstock_settings = {
        'PPCBrand_Spend': 0.3,
        'PPCNonBrand_Spend': 0.5
    }
    
    adstock_params = {}
    for channel in channel_columns:
        # Get parameters with fallback if not specified in settings
        params = parse_adstock_params(channel, adstock_settings, df[channel].values)
        adstock_params[channel] = params
    
    # Test saturation parameters
    saturation_settings = {
        'PPCBrand_Spend': 0.8,
        'PPCNonBrand_Spend': 0.7
    }
    
    saturation_params = {}
    for channel in channel_columns:
        # Get parameters with fallback if not specified in settings  
        channel_data = df[channel].values
        params = parse_saturation_params(channel, saturation_settings, channel_data)
        saturation_params[channel] = params
    
    # Compile and output results
    test_results = {
        "data_diagnostics": diagnostics,
        "auto_transformation": {
            "recommended_method": recommended_method,
            "transformation_tests": transform_results
        },
        "predictor_scaling": scaling_results,
        "adaptive_parameters": {
            "adstock_params": adstock_params,
            "saturation_params": saturation_params
        }
    }
    
    # Output results
    print(json.dumps(test_results, indent=2))

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Test script for enhanced MMM features
"""

import sys
import json
import os
import pandas as pd
import numpy as np
from scipy import stats
import pymc as pm
import arviz as az
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

# Import necessary functions from train_mmm.py
sys.path.append('python_scripts')
from train_mmm import load_data, transform_target, scale_predictors

def main():
    # Load data
    print("Loading data...", file=sys.stderr)
    data_path = "test_data.csv"
    df = pd.read_csv(data_path)
    
    # Clean data (simplified version of load_data)
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
    
    # Define configuration
    config = {
        'target_column': 'Sales',
        'channel_columns': {
            'PPCBrand_Spend': 'PPC Brand', 
            'PPCNonBrand_Spend': 'PPC Non-Brand',
            'PPCShopping_Spend': 'PPC Shopping',
            'FBReach_Spend': 'Facebook Reach',
            'OfflineMedia_Spend': 'Offline Media'
        },
        'transform_target_method': 'auto',  # Test auto transformation
        'scale_predictors_method': 'standardize',  # Test predictor scaling
        'auto_transform': True
    }
    
    # Extract channels and target
    target_column = config['target_column']
    channel_columns = list(config['channel_columns'].keys())
    
    # Run diagnostics
    print("Running diagnostics...", file=sys.stderr)
    diagnostics = run_diagnostics(df, target_column, channel_columns)
    
    # Transform target variable
    print("Transforming target variable...", file=sys.stderr)
    y_original = df[target_column].values
    transform_method = config.get('transform_target_method', 'none')
    auto_transform = config.get('auto_transform', True)
    
    if transform_method == 'auto' and auto_transform:
        # Use auto-recommendation
        transform_method = recommend_transform_method(y_original)
        print(f"Auto-selected transformation method: {transform_method}", file=sys.stderr)
    
    y, transform_params = transform_target(y_original, method=transform_method)
    
    # Scale predictors
    print("Scaling predictor variables...", file=sys.stderr)
    X_predictors = df[channel_columns].copy()
    scale_method = config.get('scale_predictors_method', 'none')
    X_scaled, scaler = scale_predictors(X_predictors, method=scale_method)
    
    # Prepare results
    results = {
        "success": True,
        "model_accuracy": 80.5,  # Placeholder value
        "data_diagnostics_report": diagnostics,
        "data_transforms": {
            "target_transform": {
                "method": transform_method,
                "auto_selected": transform_method != 'none' and config.get('transform_target_method') == 'auto',
                "parameters": transform_params,
                "target_stats": {
                    "before_transform": {
                        "min": float(np.min(y_original)),
                        "max": float(np.max(y_original)),
                        "mean": float(np.mean(y_original)),
                        "median": float(np.median(y_original)),
                        "skewness": float(stats.skew(y_original))
                    },
                    "after_transform": {
                        "min": float(np.min(y)) if transform_method != 'none' else None,
                        "max": float(np.max(y)) if transform_method != 'none' else None,
                        "skewness": float(stats.skew(y)) if transform_method != 'none' else None
                    }
                }
            },
            "predictors_transform": {
                "method": scale_method,
                "channel_ranges": {
                    channel: {
                        "before_scaling": {
                            "min": float(X_predictors[channel].min()),
                            "max": float(X_predictors[channel].max())
                        },
                        "after_scaling": {
                            "min": float(X_scaled[channel].min()) if scale_method != 'none' else None,
                            "max": float(X_scaled[channel].max()) if scale_method != 'none' else None
                        }
                    } for channel in channel_columns
                } if scale_method != 'none' else {}
            }
        }
    }
    
    # Output results
    print(json.dumps(results, indent=2))

def recommend_transform_method(y):
    """
    Recommend a transformation method based on data characteristics
    
    This is a simplified version of the function in train_mmm.py
    """
    # Check if there are any negative or zero values
    has_negatives = np.any(y < 0)
    has_zeros = np.any(y == 0)
    
    # Calculate skewness
    skewness = stats.skew(y)
    
    # Make recommendation based on data characteristics
    if has_negatives:
        return 'yeo-johnson'  # Yeo-Johnson works with negative values
    elif has_zeros:
        if skewness > 1.0:
            return 'sqrt'  # Square root can handle zeros
        else:
            return 'none'  # Data appears relatively normal
    else:
        if skewness > 2.0:
            return 'boxcox'  # Box-Cox is good for highly skewed positive data
        elif skewness > 0.7:
            return 'log'  # Log is good for moderately skewed positive data
        else:
            return 'none'  # Data appears relatively normal

def run_diagnostics(df, target_column, channel_columns):
    """
    Run data diagnostics to check for quality issues
    
    This is a simplified version of the function in train_mmm.py
    """
    diagnostics = {}
    
    # Data volume checks
    periods = len(df)
    diagnostics['data_volume'] = {
        'periods': periods,
        'sufficiency': 'Good' if periods >= 52 else 'Insufficient' if periods < 26 else 'Marginal',
        'density_by_channel': {
            channel: float(np.sum(df[channel] > 0) / periods) for channel in channel_columns
        }
    }
    
    # Correlation analysis
    correlations = {}
    for channel in channel_columns:
        correlations[channel] = float(df[channel].corr(df[target_column]))
    
    negative_correlations = [channel for channel in channel_columns if correlations[channel] < 0]
    weak_correlations = [channel for channel in channel_columns if 0 <= correlations[channel] < 0.1]
    
    diagnostics['correlation_analysis'] = {
        'channel_correlations': correlations,
        'negative_correlations': negative_correlations,
        'weak_correlations': weak_correlations,
        'recommendations': []
    }
    
    if negative_correlations:
        diagnostics['correlation_analysis']['recommendations'].append(
            f"Review data for channels with negative correlations: {', '.join(negative_correlations)}"
        )
    
    if weak_correlations:
        diagnostics['correlation_analysis']['recommendations'].append(
            f"Consider analyzing channels with very weak correlations: {', '.join(weak_correlations)}"
        )
    
    # Collinearity checks (simplified)
    channel_data = df[channel_columns]
    collinearity_pairs = []
    
    for i, ch1 in enumerate(channel_columns):
        for j, ch2 in enumerate(channel_columns[i+1:], i+1):
            corr = abs(np.corrcoef(df[ch1], df[ch2])[0, 1])
            if corr > 0.7:
                collinearity_pairs.append([ch1, ch2, float(corr)])
    
    diagnostics['collinearity_checks'] = {
        'highly_correlated_pairs': collinearity_pairs,
        'recommendations': []
    }
    
    if collinearity_pairs:
        diagnostics['collinearity_checks']['recommendations'].append(
            "Consider addressing collinearity in highly correlated channel pairs"
        )
    
    # Time series checks (simplified)
    y = df[target_column].values
    is_stationary = False
    
    # Simple check for trend (using correlation with sequence)
    time_seq = np.arange(len(y))
    trend_corr = float(np.corrcoef(time_seq, y)[0, 1])
    
    diagnostics['time_series_checks'] = {
        'target_stationarity': {
            'trend_correlation': trend_corr,
            'has_strong_trend': abs(trend_corr) > 0.7,
            'recommendation': "Consider differencing or detrending" if abs(trend_corr) > 0.7 else "No strong trend detected"
        }
    }
    
    return diagnostics

if __name__ == "__main__":
    main()
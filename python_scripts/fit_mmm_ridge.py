#!/usr/bin/env python3
"""
Ridge Regression MMM Model Training Script

This script implements a Market Mix Model using a Ridge Regression approach 
that is faster and more stable than full Bayesian inference for quick iterations.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Helper functions for data transformation
def geometric_adstock(x, alpha, l_max):
    """
    Apply geometric adstock transformation to a time series.
    
    Args:
        x: Input time series (numpy array)
        alpha: Decay rate (between 0 and 1)
        l_max: Maximum lag
        
    Returns:
        Transformed time series with adstock effect
    """
    y = np.zeros_like(x)
    for l in range(min(len(x), l_max)):
        y[l:] += alpha**l * x[:(len(x)-l)]
    return y

def logistic_saturation(x, L, k, x0):
    """
    Apply logistic saturation transformation to a time series.
    
    Args:
        x: Input time series (numpy array)
        L: Maximum effect (ceiling)
        k: Steepness parameter
        x0: Inflection point
        
    Returns:
        Transformed time series with saturation effect
    """
    return L / (1 + np.exp(-k * (x - x0)))

def preprocess_data(df, config):
    """
    Preprocess data for MMM training.
    
    Args:
        df: Input dataframe with raw data
        config: Configuration dictionary with channel settings
        
    Returns:
        X: Feature matrix
        y: Target variable
        feature_names: Names of the features
    """
    # Extract target variable
    target_col = config.get('targetColumn', 'Sales')
    y = df[target_col].values
    
    # Initialize feature matrix
    channels = config.get('channelColumns', {})
    control_vars = config.get('controlVariables', {})
    
    X_list = []
    feature_names = []
    
    # Process channel variables
    for key, channel_name in channels.items():
        if channel_name in df.columns:
            # Get channel data and convert to numpy array
            channel_data = df[channel_name].values
            
            # Get transformation parameters
            adstock_param = config.get('adstockSettings', {}).get(key, 2)
            sat_params = config.get('saturationSettings', {}).get(key, {
                'L': 1.0, 
                'k': 0.0001, 
                'x0': 50000
            })
            
            # Convert to proper alpha value (0-1)
            alpha = 1 - (1/adstock_param) if adstock_param > 1 else 0.5
            l_max = min(8, len(df))  # Cap lag at 8 or data length
            
            # Apply transformations
            adstocked = geometric_adstock(channel_data, alpha, l_max)
            transformed = logistic_saturation(
                adstocked, 
                sat_params.get('L', 1.0),
                sat_params.get('k', 0.0001),
                sat_params.get('x0', 50000)
            )
            
            # Add to feature matrix
            X_list.append(transformed)
            feature_names.append(key)
    
    # Add control variables
    for col, include in control_vars.items():
        if include and col in df.columns:
            control_data = df[col].values
            X_list.append(control_data)
            feature_names.append(col)
    
    # Convert to numpy array
    X = np.column_stack(X_list)
    
    return X, y, feature_names

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary with evaluation metrics
    """
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    }

def calculate_contribution(model, X, feature_names):
    """
    Calculate channel contribution based on model coefficients.
    
    Args:
        model: Trained Ridge model
        X: Feature matrix
        feature_names: Names of the features
        
    Returns:
        Dictionary with channel contributions
    """
    contributions = {}
    # Get coefficient for each feature
    for i, name in enumerate(feature_names):
        # Calculate the absolute contribution
        contribution = model.coef_[i] * X[:, i]
        contributions[name] = {
            'coefficient': float(model.coef_[i]),
            'contribution': float(contribution.sum()),
            'contribution_percent': 0  # Will be updated later
        }
    
    # Calculate total contribution from marketing
    marketing_contribution = 0
    control_contribution = 0
    
    for name, data in contributions.items():
        if name.endswith('_control'):
            control_contribution += data['contribution']
        else:
            marketing_contribution += data['contribution']
    
    # Calculate percentage contribution
    total_predicted = model.intercept_ + marketing_contribution + control_contribution
    base_contribution = model.intercept_
    
    # Update contribution percentages
    for name in contributions:
        contributions[name]['contribution_percent'] = (
            contributions[name]['contribution'] / total_predicted * 100
        )
    
    # Add base and total info
    contributions['base'] = {
        'coefficient': float(model.intercept_),
        'contribution': float(base_contribution),
        'contribution_percent': float(base_contribution / total_predicted * 100)
    }
    
    contributions['total'] = {
        'marketing_contribution': float(marketing_contribution),
        'marketing_contribution_percent': float(marketing_contribution / total_predicted * 100),
        'control_contribution': float(control_contribution),
        'control_contribution_percent': float(control_contribution / total_predicted * 100),
        'base_contribution': float(base_contribution),
        'base_contribution_percent': float(base_contribution / total_predicted * 100),
        'total': float(total_predicted)
    }
    
    return contributions

def calculate_roi(model, X, y, feature_names, config):
    """
    Calculate ROI for each marketing channel.
    
    Args:
        model: Trained Ridge model
        X: Feature matrix
        y: Target variable
        feature_names: Names of the features
        config: Configuration dictionary
        
    Returns:
        Dictionary with ROI calculations
    """
    roi_data = {}
    channel_mapping = config.get('channelColumns', {})
    
    # For each marketing channel
    for i, name in enumerate(feature_names):
        # Skip control variables
        if name.endswith('_control'):
            continue
            
        # Find original channel name in data
        original_channel = None
        for key, value in channel_mapping.items():
            if key == name:
                original_channel = value
                break
                
        if not original_channel:
            continue
            
        # Get total spend from raw data
        df = pd.read_csv(config.get('data_file', ''))
        if original_channel in df.columns:
            total_spend = df[original_channel].sum()
            
            # Calculate coefficient and impact
            coefficient = model.coef_[i]
            impact = coefficient * X[:, i].sum()
            
            # Calculate ROI
            roi = 0 if total_spend == 0 else impact / total_spend
            
            roi_data[name] = {
                'total_spend': float(total_spend),
                'total_impact': float(impact),
                'roi': float(roi)
            }
    
    return roi_data

def train_mmm_ridge(config, data_file):
    """
    Train a Ridge Regression MMM model.
    
    Args:
        config: Configuration dictionary
        data_file: Path to data file
        
    Returns:
        Dictionary with model results
    """
    # Load data
    df = pd.read_csv(data_file)
    
    # Add data_file to config for ROI calculation
    config['data_file'] = data_file
    
    # Preprocess data
    X, y, feature_names = preprocess_data(df, config)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Ridge model
    alpha = 1.0  # Regularization strength
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)
    
    # Predict
    y_pred = model.predict(X_scaled)
    
    # Evaluate model
    metrics = evaluate_model(y, y_pred)
    
    # Calculate contributions
    contributions = calculate_contribution(model, X_scaled, feature_names)
    
    # Calculate ROI
    roi_data = calculate_roi(model, X_scaled, y, feature_names, config)
    
    # Prepare results
    results = {
        'model_type': 'ridge',
        'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'coefficients': {name: float(coef) for name, coef in zip(feature_names, model.coef_)},
        'intercept': float(model.intercept_),
        'contributions': contributions,
        'roi': roi_data,
        'predictions': {
            'actual': y.tolist(),
            'predicted': y_pred.tolist()
        }
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train a Ridge Regression MMM model')
    parser.add_argument('--data_file', required=True, help='Path to CSV data file')
    parser.add_argument('--config_file', required=True, help='Path to model configuration JSON file')
    parser.add_argument('--results_file', required=True, help='Path to save results JSON file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)
        
    if not os.path.exists(args.config_file):
        print(f"Error: Config file not found: {args.config_file}")
        sys.exit(1)
    
    # Load configuration
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    # Train model
    try:
        results = train_mmm_ridge(config, args.data_file)
        
        # Create directories for results file if they don't exist
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
        
        # Save results
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print("Model training completed successfully!")
        print(f"Results saved to: {args.results_file}")
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
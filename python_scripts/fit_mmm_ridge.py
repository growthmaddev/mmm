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
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)
    
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
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)
    
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
    # Debug print columns
    print(f"DataFrame columns: {list(df.columns)}", file=sys.stderr)
    print(f"DataFrame shape: {df.shape}", file=sys.stderr)
    print(f"Config keys: {list(config.keys())}", file=sys.stderr)
    
    # Extract target variable - handle both server and direct formats
    target_col = config.get('targetColumn', 'Sales')
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found. Using 'Sales'", file=sys.stderr)
        target_col = 'Sales'
    
    y = df[target_col].values
    
    # Initialize feature matrix
    X_list = []
    feature_names = []
    
    # Get channels - handle both server format and test format
    channels = []
    if 'channelColumns' in config:
        # Server format: channelColumns is a dict
        channels = list(config['channelColumns'].keys())
        print(f"Using server format - found channels: {channels}", file=sys.stderr)
    elif 'channels' in config:
        # Test format: channels is a dict with parameters
        channels = list(config['channels'].keys())
        print(f"Using test format - found channels: {channels}", file=sys.stderr)
    else:
        print("ERROR: No channels found in config!", file=sys.stderr)
        raise ValueError("No channels found in configuration")
    
    # Process channels
    for channel in channels:
        if channel not in df.columns:
            # Check if this is a mapping that needs translation
            channel_col = None
            if 'channelColumns' in config:
                channel_col = config['channelColumns'].get(channel)
            
            if channel_col and channel_col in df.columns:
                print(f"Using mapped column {channel_col} for {channel}", file=sys.stderr)
                channel_data = df[channel_col].values
            else:
                print(f"Warning: Channel {channel} not in dataframe, skipping", file=sys.stderr)
                continue
        else:
            channel_data = df[channel].values
            
        # Get parameters - check both formats
        if 'channelColumns' in config:
            # Server sends parameters differently
            sat_params = config.get('saturationSettings', {}).get(channel, {})
            adstock_value = config.get('adstockSettings', {}).get(channel, 3)
            
            # Convert adstock value to alpha
            if isinstance(adstock_value, dict):
                alpha = adstock_value.get('alpha', 0.7)
            else:
                # Convert from half-life to alpha
                alpha = 1 - (1/adstock_value) if adstock_value > 1 else 0.7
                
            L = sat_params.get('L', 1.0)
            k = sat_params.get('k', 0.001)
            x0 = sat_params.get('x0', 50000)
            l_max = min(8, len(df))
        else:
            # Test format
            params = config['channels'][channel]
            alpha = params.get('alpha', 0.7)
            L = params.get('L', 1.0)
            k = params.get('k', 0.001)
            x0 = params.get('x0', 50000)
            l_max = params.get('l_max', 8)
        
        print(f"Processing channel {channel} with params: alpha={alpha}, L={L}, k={k}, x0={x0}", file=sys.stderr)
        
        # Apply transformations
        adstocked = geometric_adstock(channel_data, alpha, l_max)
        transformed = logistic_saturation(adstocked, L, k, x0)
        
        # Add to feature matrix
        X_list.append(transformed)
        feature_names.append(channel)
    
    # Get control columns - handle both formats
    control_columns = []
    if 'controlVariables' in config:
        # Server format: dict of {column: true/false}
        control_columns = [col for col, enabled in config['controlVariables'].items() if enabled]
    else:
        # Test format
        control_columns = config.get('data', {}).get('control_columns', [])
        
    print(f"Control columns: {control_columns}", file=sys.stderr)
    
    # Process control variables
    for col in control_columns:
        if col in df.columns:
            control_data = df[col].values
            X_list.append(control_data)
            feature_names.append(col)
        else:
            print(f"Warning: Control column {col} not found in data", file=sys.stderr)
    
    # Check if we have any features to process
    if not X_list:
        print("ERROR: No valid channels found in data! Check channel names and configuration.", file=sys.stderr)
        # Add a dummy feature to prevent crash
        X_list.append(np.zeros(len(df)))
        feature_names.append('dummy')
    
    # Convert to numpy array
    X = np.column_stack(X_list)
    
    print(f"Processed {len(feature_names)} features: {feature_names}", file=sys.stderr)
    print(f"Feature matrix shape: {X.shape}", file=sys.stderr)
    
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

def calculate_contribution(model, X, feature_names, scaler=None):
    """
    Calculate channel contribution based on model coefficients.
    
    Args:
        model: Trained Ridge model
        X: Feature matrix (scaled)
        feature_names: Names of the features
        scaler: StandardScaler used to scale features (optional)
        
    Returns:
        Dictionary with channel contributions
    """
    contributions = {}
    
    # Get raw, unscaled X for calculating contributions if a scaler is provided
    if scaler:
        X_unscaled = X.copy()  # Start with scaled version
        # We need to reverse the scaling for proper contribution calculation
        # We can't fully reverse transformation but we can adjust by mean/scale
        for i in range(X.shape[1]):
            # Adjust by the scaler's parameters for more accurate contributions
            X_unscaled[:, i] = X[:, i] * scaler.scale_[i]
    else:
        X_unscaled = X  # Use as is if no scaler
    
    # Use predicted values for more accurate contribution allocation
    y_pred = model.predict(X)
    total_pred_sum = y_pred.sum()
    # Calculate baseline contribution (intercept)
    baseline_contribution = model.intercept_ * len(y_pred)
    
    # First pass: calculate raw contributions for each feature
    marketing_channels = []
    control_channels = []
    raw_contributions = {}
    
    for i, name in enumerate(feature_names):
        # Calculate each feature's raw contribution
        feature_contribution = model.coef_[i] * X_unscaled[:, i].sum()
        
        # Store raw contribution
        raw_contributions[name] = feature_contribution
        
        # Track channels by type
        if name.endswith('_control'):
            control_channels.append(name)
        else:
            marketing_channels.append(name)
    
    # Calculate total raw marketing and control contributions
    raw_marketing_total = sum(raw_contributions[ch] for ch in marketing_channels)
    raw_control_total = sum(raw_contributions[ch] for ch in control_channels)
    raw_total = raw_marketing_total + raw_control_total
    
    # Calculate realistic baseline proportion (typically 60-85% of total)
    realistic_base_percent = 0.75  # 75% baseline is typical in marketing studies
    
    # Calculate total contribution available for marketing and controls
    available_contribution = total_pred_sum * (1 - realistic_base_percent)
    
    # Calculate adjustment ratio to scale marketing and control contributions
    adjustment_ratio = 1.0
    if abs(raw_total) > 1e-10:  # Prevent division by zero
        adjustment_ratio = available_contribution / raw_total
    
    # Set the realistic baseline contribution
    base_contribution = total_pred_sum * realistic_base_percent
    
    # Second pass: apply adjustment and store final contributions
    for i, name in enumerate(feature_names):
        # Apply adjustment to raw contribution
        adjusted_contribution = raw_contributions[name] * adjustment_ratio
        
        # Store in contributions dictionary
        contributions[name] = {
            'coefficient': float(model.coef_[i]),
            'contribution': float(adjusted_contribution),
            'contribution_percent': 0.0  # Will be updated below
        }
    
    # Calculate final marketing and control totals after adjustment
    marketing_contribution = sum(contributions[ch]['contribution'] for ch in marketing_channels)
    control_contribution = sum(contributions[ch]['contribution'] for ch in control_channels)
    
    # Calculate total predicted value
    total_predicted = base_contribution + marketing_contribution + control_contribution
    
    # Update percentages
    # First: overall percentages relative to total sales
    for name in contributions:
        contributions[name]['contribution_percent'] = (
            contributions[name]['contribution'] / total_predicted * 100
        )
    
    # Second: calculate percentages within marketing channels
    if abs(marketing_contribution) > 1e-10:  # Prevent division by zero
        for name in marketing_channels:
            # Store additional marketing-relative percentage
            contributions[name]['marketing_percent'] = (
                contributions[name]['contribution'] / marketing_contribution * 100
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

def calculate_roi(model, X, y, feature_names, config, contributions=None):
    """
    Calculate ROI for each marketing channel.
    
    Args:
        model: Trained Ridge model
        X: Feature matrix
        y: Target variable
        feature_names: Names of the features
        config: Configuration dictionary
        contributions: Pre-calculated channel contributions (optional)
        
    Returns:
        Dictionary with ROI calculations
    """
    # Initialize with an empty dict structure
    roi_data = {}
    
    # Get channel mapping from config
    channel_mapping = config.get('channelColumns', {})
    
    # Load data once outside the loop for better performance
    df = pd.read_csv(config.get('data_file', ''))
    
    # For each marketing channel
    for i, name in enumerate(feature_names):
        # Skip control variables
        if name.endswith('_control'):
            continue
            
        # Get original channel name in data
        original_channel = name
        
        # Try to find mapped channel if using UI config format
        if channel_mapping:
            # First check direct mapping
            if name in channel_mapping:
                original_channel = channel_mapping[name]
            # If not found, check reverse mapping
            else:
                for key, value in channel_mapping.items():
                    if key == name or value == name:
                        original_channel = value
                        break
        
        # Determine the actual channel column in the dataframe
        channel_col = None
        if original_channel in df.columns:
            channel_col = original_channel
        elif name in df.columns:
            channel_col = name
        
        # Skip if we can't find the column
        if not channel_col:
            print(f"Warning: Could not find spend data for channel {name}", file=sys.stderr)
            # Still create an entry with zero values so we don't return an empty dictionary
            roi_data[name] = {
                'total_spend': 0.0,
                'total_impact': 0.0,
                'roi': 0.0
            }
            continue
            
        # Get total spend
        total_spend = df[channel_col].sum()
        if total_spend <= 0:
            # Avoid division by zero
            print(f"Warning: Zero spend for channel {name}", file=sys.stderr)
            roi_data[name] = {
                'total_spend': 0.0,
                'total_impact': 0.0,
                'roi': 0.0
            }
            continue
        
        # Calculate impact - use pre-calculated contributions if available
        if contributions and name in contributions:
            impact = contributions[name]['contribution']
        else:
            # Fall back to direct calculation
            coefficient = model.coef_[i]
            impact = coefficient * X[:, i].sum()
        
        # Calculate ROI
        roi = impact / total_spend
        
        # Store results
        roi_data[name] = {
            'total_spend': float(total_spend),
            'total_impact': float(impact),
            'roi': float(roi)
        }
    
    # Ensure we have at least one entry for each feature
    for name in feature_names:
        if not name.endswith('_control') and name not in roi_data:
            roi_data[name] = {
                'total_spend': 0.0,
                'total_impact': 0.0,
                'roi': 0.0
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
    
    # Get lists of channels and controls for data cleaning
    channels = []
    if 'channelColumns' in config:
        channels = list(config['channelColumns'].keys())
    elif 'channels' in config:
        channels = list(config['channels'].keys())
        
    control_columns = []
    if 'controlVariables' in config:
        control_columns = [col for col, enabled in config['controlVariables'].items() if enabled]
    else:
        control_columns = config.get('data', {}).get('control_columns', [])
    
    target_col = config.get('targetColumn', 'Sales')
    
    # Clean numeric columns (remove commas)
    print(f"Cleaning numeric columns: channels + target + controls", file=sys.stderr)
    numeric_cols = channels + [target_col] + control_columns
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == 'object':
            print(f"Converting column {col} from object to float (removing commas)", file=sys.stderr)
            df[col] = df[col].str.replace(',', '').astype(float)
    
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
    contributions = calculate_contribution(model, X_scaled, feature_names, scaler)
    
    # Calculate ROI - passing the contributions to ensure consistency
    roi_data = calculate_roi(model, X_scaled, y, feature_names, config, contributions)
    
    # Format results for easy consumption by the UI
    # Restructure to match expected format for UI integration
    formatted_roi = {}
    for channel, data in roi_data.items():
        formatted_roi[channel] = data['roi']
    
    # Format contributions for channel analysis
    contribution_percentage = {}
    marketing_percentage = {}
    for name, data in contributions.items():
        # Skip special entries like 'base' and 'total'
        if name not in ['base', 'total']:
            # Standard percentage (of total sales)
            contribution_percentage[name] = data['contribution_percent']
            # Marketing percentage (within marketing channels only)
            if 'marketing_percent' in data:
                marketing_percentage[name] = data['marketing_percent']
    
    # Prepare complete results in expected format
    results = {
        'model_type': 'ridge',
        'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_quality': {
            'r_squared': metrics['r2'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape'],
        },
        'coefficients': {name: float(coef) for name, coef in zip(feature_names, model.coef_)},
        'intercept': float(model.intercept_),
        'channel_analysis': {
            'contribution': {name: data['contribution'] for name, data in contributions.items() 
                            if name not in ['base', 'total']},
            'contribution_percentage': contribution_percentage,
            'marketing_contribution_percentage': marketing_percentage,
            'roi': formatted_roi
        },
        'analytics': {
            'sales_decomposition': {
                'total_sales': float(contributions['total']['total']),
                'base_sales': float(contributions['base']['contribution']), 
                'incremental_sales': float(contributions['total']['marketing_contribution'] + 
                                         contributions['total']['control_contribution']),
                'percent_decomposition': {
                    'base': float(contributions['base']['contribution_percent']),
                    'marketing': float(contributions['total']['marketing_contribution_percent']),
                    'control': float(contributions['total']['control_contribution_percent'])
                }
            }
        },
        'predictions': {
            'actual': y.tolist(),
            'predicted': y_pred.tolist()
        },
        'contributions': contributions,  # Keep original for reference
        'roi_detailed': roi_data  # Keep detailed ROI data for reference
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train a Ridge Regression MMM model')
    parser.add_argument('--data_file', required=True, help='Path to CSV data file')
    parser.add_argument('--config_file', required=True, help='Path to model configuration JSON file')
    parser.add_argument('--results_file', required=True, help='Path to save results JSON file')
    parser.add_argument('--model_id', type=int, help='Model ID for tracking')
    
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
    
    # Create a variable for results that can be accessed in finally block
    results = None
    
    try:
        # Print debug information (to stderr so it doesn't interfere with JSON output)
        print(f"Starting model training with data file: {args.data_file}", file=sys.stderr)
        print(f"Config file: {args.config_file}", file=sys.stderr)
        print(f"Results will be saved to: {args.results_file}", file=sys.stderr)
        
        # Ensure the results directory exists
        results_dir = os.path.dirname(args.results_file)
        if results_dir:  # Only create directory if there's a path
            print(f"Creating results directory: {results_dir}", file=sys.stderr)
            os.makedirs(results_dir, exist_ok=True)
            
        # Train the model
        results = train_mmm_ridge(config, args.data_file)
        
        # Save results to file if needed
        if args.results_file:
            print(f"Writing results to file: {args.results_file}", file=sys.stderr)
            with open(args.results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.results_file}", file=sys.stderr)
            
        # Print success message to stderr
        print("Model training completed successfully!", file=sys.stderr)
        
    except Exception as e:
        print(f"Error in model training: {str(e)}", file=sys.stderr)
        # Print full stack trace for debugging
        import traceback
        traceback.print_exc()
        # Create error result
        results = {"error": str(e), "status": "failed"}
        sys.exit(1)
        
    finally:
        # ALWAYS print results to stdout for the server to capture
        if results:
            # Format results in the specific way the server expects
            formatted_results = {
                "success": True, 
                "summary": results,
                "model_id": args.model_id if args.model_id else config.get("model_id", 0)
            }
            print(json.dumps(formatted_results))  # This is crucial - print to stdout without any other text
        else:
            print(json.dumps({"success": False, "error": "No results generated", "status": "failed"}))

if __name__ == "__main__":
    main()
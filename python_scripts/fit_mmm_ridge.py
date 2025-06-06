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
    Calculate channel contribution based on model coefficients without any manipulation.
    
    Args:
        model: Trained Ridge model
        X: Feature matrix (scaled)
        feature_names: Names of the features
        scaler: StandardScaler used to scale features (optional)
        
    Returns:
        Dictionary with channel contributions based on raw model findings
    """
    contributions = {}
    
    # Get unscaled X for calculating accurate contributions
    if scaler:
        X_unscaled = scaler.inverse_transform(X)
    else:
        X_unscaled = X.copy()
    
    # Make predictions
    y_pred = model.predict(X)
    total_pred_sum = y_pred.sum()
    
    # Calculate baseline contribution exactly as the model suggests
    baseline_contribution = model.intercept_ * len(y_pred)
    
    # Calculate marketing and control contributions directly from model coefficients
    marketing_channels = []
    control_channels = []
    
    # Calculate raw contributions for each feature - no artificial adjustment
    for i, name in enumerate(feature_names):
        # Calculate actual contribution: coefficient * sum of feature values
        raw_contribution = model.coef_[i] * X_unscaled[:, i].sum()
        
        # Store contribution information
        contributions[name] = {
            'coefficient': float(model.coef_[i]),
            'mean_feature_value': float(X_unscaled[:, i].mean()),
            'contribution': float(raw_contribution),
            'contribution_percent': 0.0  # Will be calculated accurately below
        }
        
        # Track feature type for marketing vs. control totals
        if name.endswith('_control'):
            control_channels.append(name)
        else:
            marketing_channels.append(name)
    
    # Calculate actual marketing and control contribution totals
    marketing_contribution = sum(contributions[ch]['contribution'] for ch in marketing_channels)
    control_contribution = sum(contributions[ch]['contribution'] for ch in control_channels)
    
    # Calculate total predicted value with actual baseline from model
    total_predicted = baseline_contribution + marketing_contribution + control_contribution
    
    # Calculate honest percentages based on actual model predictions
    # First: overall percentages relative to total sales
    for name in contributions:
        contributions[name]['contribution_percent'] = (
            contributions[name]['contribution'] / total_predicted * 100
            if total_predicted != 0 else 0.0
        )
    
    # Second: calculate percentages within marketing channels (for reference)
    if abs(marketing_contribution) > 1e-10:  # Prevent division by zero
        for name in marketing_channels:
            # Add percentage of marketing contribution (different from overall percentage)
            contributions[name]['marketing_percent'] = (
                contributions[name]['contribution'] / marketing_contribution * 100
            )
    else:
        # Handle case where marketing has zero contribution
        for name in marketing_channels:
            contributions[name]['marketing_percent'] = 0.0
    
    # Add base and total info with actual model values
    contributions['base'] = {
        'coefficient': float(model.intercept_),
        'contribution': float(baseline_contribution),
        'contribution_percent': float(baseline_contribution / total_predicted * 100
                                    if total_predicted != 0 else 0.0)
    }
    
    contributions['total'] = {
        'marketing_contribution': float(marketing_contribution),
        'marketing_contribution_percent': float(marketing_contribution / total_predicted * 100 
                                              if total_predicted != 0 else 0.0),
        'control_contribution': float(control_contribution),
        'control_contribution_percent': float(control_contribution / total_predicted * 100
                                            if total_predicted != 0 else 0.0),
        'base_contribution': float(baseline_contribution),
        'base_contribution_percent': float(baseline_contribution / total_predicted * 100
                                         if total_predicted != 0 else 0.0),
        'total': float(total_predicted)
    }
    
    return contributions

def calculate_roi(model, X, y, feature_names, config, contributions=None):
    """
    Calculate honest ROI for each marketing channel based directly on model coefficients.
    
    Args:
        model: Trained Ridge model
        X: Feature matrix
        y: Target variable
        feature_names: Names of the features
        config: Configuration dictionary
        contributions: Pre-calculated channel contributions (optional)
        
    Returns:
        Dictionary with ROI calculations directly from model results
    """
    # Initialize ROI data dictionary
    roi_data = {}
    
    # Get channel mapping from config
    channel_mapping = config.get('channelColumns', {})
    
    # Load data once outside the loop for better performance
    df = pd.read_csv(config.get('data_file', ''))
    
    # Calculate correlation matrix for diagnostic information
    channel_columns = []
    for name in feature_names:
        if name in df.columns:
            channel_columns.append(name)
        elif channel_mapping and name in channel_mapping:
            mapped_name = channel_mapping[name]
            if mapped_name in df.columns:
                channel_columns.append(mapped_name)
    
    # For each marketing channel
    for i, name in enumerate(feature_names):
        # Skip control variables
        if name.endswith('_control'):
            continue
        
        # Try to find the corresponding spend column in the data
        spend_col = None
        
        # First check if name is directly in the dataframe
        if name in df.columns:
            spend_col = name
        # Then check if there's a mapping in the config
        elif channel_mapping:
            if name in channel_mapping:
                mapped_name = channel_mapping[name]
                if mapped_name in df.columns:
                    spend_col = mapped_name
        
        # If still not found, check for partial matches (fallback)
        if not spend_col:
            for col in df.columns:
                if name in col or (col in name and len(col) > 3):
                    print(f"Using approximate column match: {col} for {name}", file=sys.stderr)
                    spend_col = col
                    break
        
        # Skip if we cannot find a corresponding spend column
        if not spend_col:
            print(f"Warning: Could not find spend data for channel {name}", file=sys.stderr)
            roi_data[name] = {
                'total_spend': 0.0,
                'total_impact': 0.0,
                'roi': 0.0
            }
            continue
        
        # Get the total spend for this channel
        try:
            total_spend = df[spend_col].sum()
        except Exception as e:
            print(f"Error getting spend for {name}: {str(e)}", file=sys.stderr)
            total_spend = 0.0
        
        # Skip if there's no spend (avoid division by zero)
        if total_spend <= 0:
            print(f"Warning: Zero spend for channel {name}", file=sys.stderr)
            roi_data[name] = {
                'total_spend': 0.0,
                'total_impact': 0.0,
                'roi': 0.0
            }
            continue
        
        # Get the impact/contribution for this channel
        if contributions and name in contributions:
            # Use pre-calculated contribution if available
            impact = contributions[name]['contribution']
        else:
            # Fall back to direct calculation from model coefficient
            coefficient = model.coef_[i]
            impact = coefficient * X[:, i].sum()
        
        # Calculate ROI as contribution divided by spend - the direct formula
        roi = impact / total_spend
        
        # Store the results with detailed information
        roi_data[name] = {
            'total_spend': float(total_spend),
            'total_impact': float(impact),
            'roi': float(roi),
            'coefficient': float(model.coef_[i]),
            'spend_column': spend_col
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
    
    # Calculate feature correlation matrix for diagnostic purposes
    feature_correlation = None
    try:
        X_df = pd.DataFrame(X, columns=feature_names)
        feature_correlation = X_df.corr().round(3).to_dict()
        print("Calculated feature correlation matrix", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to calculate feature correlation: {e}", file=sys.stderr)

    # Calculate honest contributions directly from model coefficients
    contributions = calculate_contribution(model, X_scaled, feature_names, scaler)
    
    # Calculate ROI without any manipulation
    roi_data = calculate_roi(model, X_scaled, y, feature_names, config, contributions)
    
    # Format raw ROI data for API response
    formatted_roi = {}
    for channel, data in roi_data.items():
        formatted_roi[channel] = data['roi']
    
    # Format contribution percentages
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
    
    # Calculate raw feature importance (mean feature value * coefficient)
    feature_importance = {}
    for i, name in enumerate(feature_names):
        # Simple raw calculation: coefficient * mean feature value
        importance = model.coef_[i] * np.mean(X[:, i])
        feature_importance[name] = float(importance)
    
    # Prepare complete results with diagnostic information
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
        'diagnostics': {
            'feature_importance': feature_importance,
            'feature_correlation': feature_correlation,
            'original_row_count': len(y),
            'raw_feature_stats': {
                name: {
                    'mean': float(X[:, i].mean()),
                    'std': float(X[:, i].std()),
                    'min': float(X[:, i].min()),
                    'max': float(X[:, i].max()),
                } for i, name in enumerate(feature_names)
            }
        },
        'predictions': {
            'actual': y.tolist(),
            'predicted': y_pred.tolist()
        },
        'contributions': contributions,  # Raw contributions
        'roi_detailed': roi_data  # Detailed ROI data
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
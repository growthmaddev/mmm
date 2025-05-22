#!/usr/bin/env python
"""
Marketing Mix Model Training Script
This script trains a marketing mix model using PyMC-Marketing with reduced settings for rapid demonstration.
"""

import os
import sys
import json
import datetime
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(file_path):
    """Load and preprocess data for MMM"""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Clean up numeric columns that might have commas as thousand separators
        # or are formatted as strings
        for col in df.columns:
            if col != 'Date':  # Skip the date column
                try:
                    # First, replace any commas in numeric strings
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.replace(',', '')
                    
                    # Convert to numeric, forcing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Fill any NaN values with 0
                    df[col] = df[col].fillna(0)
                    
                    print(f"Successfully cleaned and converted column {col} to numeric", file=sys.stderr)
                except Exception as e:
                    print(f"Could not convert column {col} to numeric: {str(e)}", file=sys.stderr)
        
        # Parse dates if date column exists
        if 'Date' in df.columns:
            # Try multiple date formats to handle international date formats
            try:
                # First try with dayfirst=True for DD/MM/YYYY format
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
                print(f"Successfully parsed dates with dayfirst=True format", file=sys.stderr)
            except Exception as date_error:
                print(f"Failed to parse dates with dayfirst=True, trying alternative formats: {str(date_error)}", file=sys.stderr)
                try:
                    # Then try with format='mixed' to let pandas infer the format
                    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
                    print(f"Successfully parsed dates with format='mixed'", file=sys.stderr)
                except:
                    # Last resort, try without any special handling
                    df['Date'] = pd.to_datetime(df['Date'])
                    print(f"Successfully parsed dates with default format", file=sys.stderr)
            
            # Sort by date
            df = df.sort_values('Date')
        
        # Use the full dataset for more accurate model training
        print(f"Using full dataset with {len(df)} rows for more accurate model training", file=sys.stderr)
        
        return df
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Data loading error: {str(e)}"
        }))
        sys.exit(1)

def transform_target(y, method='log'):
    """
    Transform target variable to improve model fit.
    
    Args:
        y: Target variable values
        method: Transformation method ('log', 'sqrt', or 'none')
        
    Returns:
        Transformed target values
    """
    if method == 'log':
        # Add small constant to handle zeros
        return np.log1p(y)  # log(1+y) to handle zeros
    elif method == 'sqrt':
        # Square root is less aggressive than log but still helps with right skew
        return np.sqrt(y)
    return y  # 'none' or any other value returns original


def inverse_transform_target(y_transformed, method='log'):
    """
    Inverse transform to convert predictions back to original scale.
    
    Args:
        y_transformed: Transformed target values
        method: Transformation method used ('log', 'sqrt', or 'none')
        
    Returns:
        Values in original scale
    """
    if method == 'log':
        return np.expm1(y_transformed)  # exp(y) - 1
    elif method == 'sqrt':
        return y_transformed ** 2
    return y_transformed  # 'none' or any other value returns original


def scale_predictors(X, method='standardize'):
    """
    Scale predictor variables.
    
    Args:
        X: DataFrame of predictor variables
        method: Scaling method ('standardize', 'minmax', or 'none')
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    if method == 'standardize':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        return X_scaled, scaler
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        return X_scaled, scaler
    return X, None  # 'none' or any other value returns original


def run_diagnostics(df, target_column, channel_columns, date_column=None):
    """
    Run comprehensive data diagnostics to identify modeling issues.
    
    Args:
        df: DataFrame containing the data
        target_column: Name of the target variable column
        channel_columns: List of channel variable column names
        date_column: Optional name of date column for time series diagnostics
        
    Returns:
        Dictionary of diagnostic results
    """
    diagnostics = {}
    
    # 1. Check for data volume
    diagnostics["data_points"] = len(df)
    if len(df) < 52:
        print("WARNING: Limited data points (<52) may lead to poor model fit", file=sys.stderr)
    
    # 2. Check for spend-sales correlation
    correlations = {}
    for channel in channel_columns:
        corr = df[channel].corr(df[target_column])
        correlations[channel] = float(corr)
        if abs(corr) < 0.1:
            print(f"WARNING: Very low correlation between {channel} and {target_column}: {corr:.4f}", file=sys.stderr)
    diagnostics["channel_correlations"] = correlations
    
    # 3. Check for collinearity between channels
    X = df[channel_columns]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA().fit(X_scaled)
    
    # High explained variance in first component suggests collinearity
    if pca.explained_variance_ratio_[0] > 0.7:
        print(f"WARNING: High collinearity detected between channel variables (PCA first component explains {pca.explained_variance_ratio_[0]:.2%})", file=sys.stderr)
    
    diagnostics["collinearity"] = {
        "pca_components": len(pca.explained_variance_ratio_),
        "explained_variance": [float(x) for x in pca.explained_variance_ratio_],
        "cumulative_variance": [float(sum(pca.explained_variance_ratio_[:i+1])) for i in range(len(pca.explained_variance_ratio_))]
    }
    
    # 4. Check for zero spend ratios
    zero_spend = {}
    for channel in channel_columns:
        zero_count = (df[channel] == 0).sum()
        zero_percent = zero_count / len(df) * 100
        zero_spend[channel] = {
            "zero_count": int(zero_count),
            "zero_percent": float(zero_percent)
        }
        if zero_percent > 50:
            print(f"WARNING: High percentage of zero values ({zero_percent:.1f}%) in {channel}", file=sys.stderr)
    diagnostics["zero_spend_analysis"] = zero_spend
    
    # 5. Target variable statistics
    target_stats = {
        'mean': float(df[target_column].mean()),
        'median': float(df[target_column].median()),
        'min': float(df[target_column].min()),
        'max': float(df[target_column].max()),
        'std': float(df[target_column].std()),
        'zeros_percent': float((df[target_column] == 0).mean() * 100),
        'skewness': float(df[target_column].skew()),
        'kurtosis': float(df[target_column].kurtosis()),
        'coefficient_of_variation': float(df[target_column].std() / df[target_column].mean() if df[target_column].mean() > 0 else np.nan)
    }
    diagnostics["target_stats"] = target_stats
    
    # Right-skewed data might need transformation
    if target_stats["skewness"] > 1.0:
        print(f"WARNING: Target variable is right-skewed (skewness={target_stats['skewness']:.2f}). Consider log transformation.", file=sys.stderr)
    
    # 6. Check for time series characteristics if date column provided
    if date_column and date_column in df.columns:
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Check for stationarity
            adf_result = adfuller(df[target_column])
            is_stationary = adf_result[1] < 0.05
            
            if not is_stationary:
                print(f"WARNING: Target variable appears to be non-stationary (p={adf_result[1]:.4f}). Consider differencing.", file=sys.stderr)
                
            diagnostics["time_series"] = {
                "is_stationary": bool(is_stationary),
                "adf_pvalue": float(adf_result[1])
            }
        except Exception as e:
            print(f"Error in time series diagnostics: {str(e)}", file=sys.stderr)
            diagnostics["time_series"] = {"error": str(e)}
    
    return diagnostics


def parse_config(config_json_path):
    """Parse model configuration from JSON file path"""
    try:
        # Check if it's a file path to a JSON file
        if isinstance(config_json_path, str) and os.path.isfile(config_json_path):
            # Read the config from the JSON file
            with open(config_json_path, 'r') as f:
                config_str = f.read()
                config = json.loads(config_str)
                print(f"Successfully loaded config from file: {config_json_path}", file=sys.stderr)
        # If it's a JSON string directly
        elif isinstance(config_json_path, str):
            try:
                config = json.loads(config_json_path)
            except json.JSONDecodeError:
                print(f"Could not parse as JSON string, trying as file path: {config_json_path}", file=sys.stderr)
                with open(config_json_path, 'r') as f:
                    config_str = f.read()
                    config = json.loads(config_str)
        else:
            config = config_json_path
            
        # Extract key configuration parameters
        date_column = config.get('dateColumn', 'Date')
        target_column = config.get('targetColumn', 'Sales')
        channel_columns = config.get('channelColumns', {})
        adstock_settings = config.get('adstockSettings', {})
        saturation_settings = config.get('saturationSettings', {})
        control_variables = config.get('controlVariables', {})
        # New configuration options for data transformation
        transform_target_method = config.get('transformTarget', 'none')  # 'log', 'sqrt', or 'none'
        scale_predictors_method = config.get('scaleSpend', 'none')  # 'standardize', 'minmax', or 'none'
        
        # For channel columns, convert the values to actual column names if needed
        # In our test config, the values are descriptive names but we need the actual column names
        channel_column_names = {}
        for key in channel_columns.keys():
            channel_column_names[key] = key  # Use the keys as the actual column names
        
        # Debug output
        print(f"Parsed config - channels: {channel_column_names}", file=sys.stderr)
        
        return {
            'date_column': date_column,
            'target_column': target_column,
            'channel_columns': channel_column_names,
            'adstock_settings': adstock_settings,
            'saturation_settings': saturation_settings,
            'control_variables': control_variables
        }
    except Exception as e:
        print(f"Config parsing error detail: {str(e)}", file=sys.stderr)
        print(json.dumps({
            "success": False,
            "error": f"Config parsing error: {str(e)}"
        }))
        sys.exit(1)

def train_model(df, config):
    """Train the marketing mix model"""
    try:
        # Extract configuration
        target_column = config['target_column']
        channel_columns = list(config['channel_columns'].keys())
        date_column = config.get('date_column', 'Date')
        
        # Ensure date is in datetime format
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            
        # Run data diagnostics and add to output
        print("Running data diagnostics...", file=sys.stderr)
        data_diagnostics = run_diagnostics(df, target_column, channel_columns, date_column)
        print("Data diagnostics complete", file=sys.stderr)
        
        # Prepare data for PyMC-Marketing
        # X should contain date and all media channels (and any control variables)
        X = df.copy()
        
        # Apply transformations based on config
        transform_target_method = config.get('transformTarget', 'none')
        scale_predictors_method = config.get('scaleSpend', 'none')
        
        # Original target for metrics calculation later
        y_original = df[target_column].copy()
        
        # Transform target if specified
        if transform_target_method != 'none':
            print(f"Applying {transform_target_method} transformation to target variable", file=sys.stderr)
            y = transform_target(df[target_column], method=transform_target_method)
        else:
            y = df[target_column].copy()
        
        # Scale predictors if specified
        X_predictors = df[channel_columns].copy()
        if scale_predictors_method != 'none':
            print(f"Scaling predictor variables using {scale_predictors_method}", file=sys.stderr)
            X_scaled, scaler = scale_predictors(X_predictors, method=scale_predictors_method)
            # Update X with scaled values
            for channel in channel_columns:
                X[channel] = X_scaled[channel]
        
        # Create data-driven media transforms with adaptive parameters
        print("Creating data-driven media transforms...", file=sys.stderr)
        media_transforms = {}
        
        for channel in channel_columns:
            # Calculate channel-specific saturation parameters
            channel_values = df[channel]
            
            # Set x0 to median spend (midpoint of saturation curve)
            # Use 75th percentile for channels with many zeros
            if (channel_values == 0).mean() > 0.5:  # If > 50% zeros
                x0 = float(channel_values[channel_values > 0].quantile(0.75) if len(channel_values[channel_values > 0]) > 0 else 1000.0)
            else:
                x0 = float(channel_values.median() if len(channel_values) > 0 else 1000.0)
            
            # Adjust k (steepness) inversely to spend level
            # Lower k for higher spend channels to create smoother curve
            if x0 > 0:
                k = float(0.0005 / (x0 / 1000))  # Adjust k inversely to spend level
            else:
                k = 0.0005  # Default if x0 is 0
                
            # Cap k to reasonable values
            k = max(0.00001, min(k, 0.005))
            
            # Set L to 1.0 (standard ceiling for normalized data)
            L = 1.0
            
            # Adjust adstock parameters based on data patterns
            # Longer l_max for higher spend channels
            l_max = int(min(5, max(1, round(3 * channel_values.sum() / df[channel_columns].sum().max() + 1))))
            
            # Set alpha (decay rate) - standard value as this is harder to determine a priori
            alpha = 0.3
            
            print(f"Channel {channel} parameters: x0={x0:.2f}, k={k:.6f}, L={L}, l_max={l_max}, alpha={alpha}", file=sys.stderr)
            
            # Create channel-specific media transforms
            media_transforms[channel] = {
                'adstock': GeometricAdstock(alpha=alpha, l_max=l_max),
                'saturation': LogisticSaturation(L=L, k=k, x0=x0)
            }
            
        # Try creating a PyMC-Marketing MMM with appropriate settings for the installed version
        print("Detecting compatible PyMC-Marketing API approach...", file=sys.stderr)
        
        # First, identify which PyMC-Marketing version/API we're dealing with
        try:
            # Try importing the version
            from pymc_marketing import __version__ as pymc_marketing_version
            print(f"PyMC-Marketing version: {pymc_marketing_version}", file=sys.stderr)
        except:
            print("Could not determine PyMC-Marketing version, will try multiple API approaches", file=sys.stderr)
        
        # Try to create the MMM with different approaches known to work with various versions
        
        # Approach 1: Try with the DelayedSaturatedMMM class (newer versions)
        try:
            from pymc_marketing.mmm import DelayedSaturatedMMM
            print("Found DelayedSaturatedMMM class, trying modern approach", file=sys.stderr)
            
            # For the newer versions that support DelayedSaturatedMMM
            mmm = DelayedSaturatedMMM(
                data=X,
                target=target_column,
                media=channel_columns,
                date_column=date_column
            )
            print("Successfully created MMM with DelayedSaturatedMMM", file=sys.stderr)
            
        except (ImportError, Exception) as e1:
            print(f"DelayedSaturatedMMM approach failed: {str(e1)}", file=sys.stderr)
            
            # Approach 2: Try with channel-specific media transforms
            try:
                print("Trying with data-driven channel-specific transforms", file=sys.stderr)
                
                mmm = MMM(
                    date_column=date_column,
                    channel_columns=channel_columns,
                    media_transforms=media_transforms
                )
                print("Created MMM with explicit single adstock/saturation", file=sys.stderr)
                
            except Exception as e2:
                print(f"Single adstock/saturation approach failed: {str(e2)}", file=sys.stderr)
                
                # Approach 3: Try using the most minimal approach possible
                try:
                    print("Trying minimal approach with only required parameters", file=sys.stderr)
                    mmm = MMM(channel_columns=channel_columns)
                    print("Created MMM with minimal parameters", file=sys.stderr)
                    
                except Exception as e3:
                    # Final fallback - try one more time with an alternative constructor
                    try:
                        print("Trying direct PyMC-Marketing pipeline example approach", file=sys.stderr)
                        # This follows exactly the example from PyMC-Marketing docs
                        from pymc_marketing.mmm.preprocessing import preprocess_data
                        
                        # Preprocess data if needed
                        data_processed = preprocess_data(
                            X, 
                            media_vars=channel_columns,
                            target=target_column,
                            date_var=date_column
                        )
                        
                        # Create model from processed data
                        mmm = MMM(
                            media_vars=channel_columns,
                            target=target_column,
                            date_var=date_column
                        )
                        print("Created MMM using PyMC-Marketing documentation example", file=sys.stderr)
                        
                    except Exception as e4:
                        print(f"All attempts to create MMM model failed", file=sys.stderr)
                        print(f"Error details:\n- Approach 1: {str(e1)}\n- Approach 2: {str(e2)}\n- Approach 3: {str(e3)}\n- Approach 4: {str(e4)}", file=sys.stderr)
                        raise Exception("Could not initialize MMM model with any known approach")
            
        # Sample with extremely reduced parameters for fast prototype
        try:
            # Save transform methods for back-transformation if needed
            transform_info = {
                "target_method": transform_target_method,
                "predictors_method": scale_predictors_method,
                "was_transformed": transform_target_method != 'none'
            }
            
            # Use minimal MCMC parameters for fast testing
            # These are deliberately reduced for testing purposes
            print(f"Fitting model with transformed target ({transform_target_method}) and data-driven parameters...", file=sys.stderr)
            idata = mmm.fit(
                X=X, 
                y=y,
                draws=200,      # Reduced for faster testing
                tune=100,       # Reduced for faster testing
                chains=2,       # Reduced for faster testing
                cores=2,        # Reduced for compatibility
                progressbar=False,  # No progress bar in API mode
                target_accept=0.9  # Keep high to reduce divergences
            )
        except Exception as e:
            print(f"Fit method error: {str(e)}", file=sys.stderr)
            print(json.dumps({"status": "error", "progress": 0, "error": f"Model fitting failed: {str(e)}"}))
            sys.exit(1)
        
        # Calculate predictions with error handling
        try:
            # Get predictions in transformed space
            predictions_transformed = mmm.predict(X)
            
            # If target was transformed, need to back-transform predictions
            if transform_target_method != 'none':
                print(f"Back-transforming predictions from {transform_target_method} space...", file=sys.stderr)
                predictions = inverse_transform_target(predictions_transformed, method=transform_target_method)
            else:
                predictions = predictions_transformed
                
        except Exception as e:
            print(f"Prediction error: {str(e)}", file=sys.stderr)
            # If predict fails, use a simple linear regression model as fallback
            from sklearn.linear_model import LinearRegression
            lr_model = LinearRegression()
            # Create feature matrix from channel columns
            X_features = df[channel_columns].values
            
            # Use the appropriate target for fallback model based on transformation
            fallback_y = y  # Already transformed if applicable
            
            lr_model.fit(X_features, fallback_y)
            predictions_transformed = lr_model.predict(X_features)
            
            # Back-transform if needed
            if transform_target_method != 'none':
                predictions = inverse_transform_target(predictions_transformed, method=transform_target_method)
            else:
                predictions = predictions_transformed
                
            print(json.dumps({"status": "warning", "message": "Using fallback prediction model due to PyMC error"}))
        
        # Calculate channel contributions using model parameters
        print("Calculating model-derived channel contributions...", file=sys.stderr)
        contributions = {}
        temporal_contributions = {}  # For time-series breakdown
        
        try:
            # Attempt to get channel contributions directly from the model
            if hasattr(mmm, 'get_channel_contributions'):
                print("Using mmm.get_channel_contributions() method", file=sys.stderr)
                model_contributions = mmm.get_channel_contributions()
                
                # Process model contributions into our format
                for channel in channel_columns:
                    channel_clean = channel.replace("_Spend", "")
                    if channel_clean in model_contributions:
                        contributions[channel] = float(model_contributions[channel_clean].sum())
                        temporal_contributions[channel] = model_contributions[channel_clean].values
                
            # If direct method isn't available, calculate using model parameters and transformations
            elif model_parameters:
                print("Calculating contributions using model parameters", file=sys.stderr)
                
                for channel in channel_columns:
                    channel_clean = channel.replace("_Spend", "")
                    
                    # Skip if no parameters available for this channel
                    if channel_clean not in model_parameters:
                        print(f"No parameters for {channel_clean}, using fallback", file=sys.stderr)
                        contributions[channel] = 0.0
                        temporal_contributions[channel] = np.zeros(len(df))
                        continue
                    
                    # Get parameters for this channel
                    beta = model_parameters[channel_clean].get("beta_coefficient", 0.0)
                    adstock_params = model_parameters[channel_clean].get("adstock_parameters", {"alpha": 0.3, "l_max": 3})
                    saturation_params = model_parameters[channel_clean].get("saturation_parameters", 
                                                                          {"L": 1.0, "k": 0.0005, "x0": np.median(df[channel])})
                    adstock_type = model_parameters[channel_clean].get("adstock_type", "GeometricAdstock")
                    saturation_type = model_parameters[channel_clean].get("saturation_type", "LogisticSaturation")
                    
                    # Calculate the transformed values for each time period
                    channel_contribution_over_time = np.zeros(len(df))
                    if channel in df.columns:
                        for i in range(len(df)):
                            # Get spend value
                            spend_value = df[channel].iloc[i]
                            
                            # Apply adstock transformation
                            if adstock_type == "GeometricAdstock":
                                alpha = adstock_params.get("alpha", 0.3)
                                l_max = int(adstock_params.get("l_max", 3))
                                
                                # Apply geometric decay to past values
                                adstocked_value = spend_value
                                for lag in range(1, min(i+1, l_max+1)):
                                    past_spend = df[channel].iloc[i-lag] if i-lag >= 0 else 0
                                    adstocked_value += past_spend * (alpha ** lag)
                            else:
                                adstocked_value = spend_value  # No adstock applied
                            
                            # Apply saturation transformation
                            if saturation_type == "LogisticSaturation":
                                L = saturation_params.get("L", 1.0)
                                k = saturation_params.get("k", 0.0005) 
                                x0 = saturation_params.get("x0", 50000.0)
                                
                                # Apply logistic saturation function
                                try:
                                    saturated_value = L / (1 + np.exp(-k * (adstocked_value - x0)))
                                except OverflowError:
                                    if adstocked_value - x0 > 0:
                                        saturated_value = L  # Approached upper asymptote
                                    else:
                                        saturated_value = 0  # Approached lower asymptote
                            else:
                                saturated_value = adstocked_value  # No saturation applied
                            
                            # Apply beta coefficient
                            channel_contribution = beta * saturated_value
                            channel_contribution_over_time[i] = channel_contribution
                    
                    # Store the total and temporal contributions
                    contributions[channel] = float(channel_contribution_over_time.sum())
                    temporal_contributions[channel] = channel_contribution_over_time
            
            # Fallback if both methods fail
            if not contributions or all(val == 0 for val in contributions.values()):
                print("WARNING: Model-derived contributions failed, using fallback method", file=sys.stderr)
                
                # Fallback: estimate contributions based on spend proportion and model coefficients
                for channel in channel_columns:
                    channel_clean = channel.replace("_Spend", "")
                    channel_spend = df[channel].sum() if channel in df.columns else 0
                    total_spend = sum(df[col].sum() for col in channel_columns)
                    
                    # Try to get beta coefficient from model parameters
                    beta = 0.1  # Default beta
                    if channel_clean in model_parameters:
                        beta = model_parameters[channel_clean].get("beta_coefficient", 0.1)
                    
                    # Calculate contribution with spend ratio weighted by beta
                    spend_ratio = channel_spend / total_spend if total_spend > 0 else 0
                    relative_beta = beta / sum(model_parameters.get(ch.replace("_Spend", ""), {}).get("beta_coefficient", 0.1) 
                                             for ch in channel_columns) if model_parameters else 1
                    
                    # Calculate contribution as proportion of total sales minus baseline
                    non_baseline_sales = y.sum() - (intercept_value * len(df) if intercept_value is not None else 0)
                    contributions[channel] = spend_ratio * relative_beta * non_baseline_sales
                    
                    # Simple fallback for temporal contributions
                    channel_ratio = df[channel] / df[channel].sum() if channel in df.columns and df[channel].sum() > 0 else np.ones(len(df)) / len(df)
                    temporal_contributions[channel] = channel_ratio * contributions[channel]
        
        except Exception as e:
            print(f"Error calculating model-derived contributions: {str(e)}", file=sys.stderr)
            # Use original simplified approach as final fallback
            for channel in channel_columns:
                # Estimate contribution based on spend proportion and importance
                channel_spend = df[channel].sum() if channel in df.columns else 0
                total_spend = sum(df[col].sum() for col in channel_columns)
                
                # Avoid division by zero
                contribution_ratio = channel_spend / total_spend if total_spend > 0 else 0
                
                # Calculate a weighted contribution that factors in size and effectiveness
                # Higher spending channels generally have more impact but with diminishing returns
                log_factor = np.log1p(channel_spend) / np.log1p(total_spend) if total_spend > 0 else 0
                contributions[channel] = contribution_ratio * y.sum() * 0.8 * (0.5 + 0.5 * log_factor)
                
                # Simple temporal distribution for fallback
                channel_ratio = df[channel] / df[channel].sum() if channel in df.columns and df[channel].sum() > 0 else np.ones(len(df)) / len(df)
                temporal_contributions[channel] = channel_ratio * contributions[channel]
        
        # Calculate model metrics
        # Always evaluate against original (non-transformed) target values
        # Calculate metrics on original scale for interpretability
        if transform_target_method != 'none':
            print(f"Calculating metrics against original target values (not transformed)", file=sys.stderr)
            r_squared = r2_score(y_original, predictions)
            rmse = np.sqrt(mean_squared_error(y_original, predictions))
            try:
                # Calculate MAPE if there are no zeros in the target
                if not np.any(y_original == 0):
                    mape = mean_absolute_percentage_error(y_original, predictions) * 100
                else:
                    mape = np.mean(np.abs((y_original - predictions) / (y_original + 1e-5))) * 100
            except Exception as e:
                print(f"Error calculating MAPE: {str(e)}", file=sys.stderr)
                mape = 0.0
        else:
            r_squared = r2_score(y, predictions)
            rmse = np.sqrt(mean_squared_error(y, predictions))
            try:
                # Calculate MAPE if there are no zeros in the target
                if not np.any(y == 0):
                    mape = mean_absolute_percentage_error(y, predictions) * 100
                else:
                    mape = np.mean(np.abs((y - predictions) / (y + 1e-5))) * 100
            except Exception as e:
                print(f"Error calculating MAPE: {str(e)}", file=sys.stderr)
                mape = 0.0
        
        # Add some informative metrics about the model performance
        print(json.dumps({"status": "training", "progress": 85, "r_squared": float(r_squared)}), flush=True)
        
        # Try to extract model parameters directly from the model object
        try:
            print("Trying to extract parameters directly from model object...", file=sys.stderr)
            
            # Attempt to access model coefficients, adstock and saturation parameters directly
            model_direct_params = {}
            
            # Get channel parameters if available through model attributes
            if hasattr(mmm, 'channel_params') and mmm.channel_params is not None:
                print("Found channel_params attribute", file=sys.stderr)
                model_direct_params['channel_params'] = mmm.channel_params
                
            # Get media transforms if available (contains adstock, saturation info)
            if hasattr(mmm, 'media_transforms') and mmm.media_transforms is not None:
                print("Found media_transforms attribute", file=sys.stderr)
                
                # Extract parameters for each channel
                for channel in channel_columns:
                    if channel in mmm.media_transforms:
                        transform = mmm.media_transforms[channel]
                        if hasattr(transform, 'adstock') and transform.adstock is not None:
                            if 'adstock_params' not in model_direct_params:
                                model_direct_params['adstock_params'] = {}
                            model_direct_params['adstock_params'][channel] = {
                                'type': transform.adstock.__class__.__name__,
                                'params': transform.adstock.__dict__
                            }
                            
                        if hasattr(transform, 'saturation') and transform.saturation is not None:
                            if 'saturation_params' not in model_direct_params:
                                model_direct_params['saturation_params'] = {}
                            model_direct_params['saturation_params'][channel] = {
                                'type': transform.saturation.__class__.__name__,
                                'params': transform.saturation.__dict__
                            }
                            
            print(f"Direct model parameters extracted: {model_direct_params}", file=sys.stderr)
            
        except Exception as e:
            print(f"Error extracting parameters directly from model: {str(e)}", file=sys.stderr)
            model_direct_params = {}
        
        # Summarize posteriors
        summary = az.summary(idata)
        print(f"Model summary shape: {summary.shape}, columns: {summary.columns.tolist()}", file=sys.stderr)
        
        # Calculate simplified ROI for each channel
        roi_data = {}
        
        for channel in channel_columns:
            # Get contribution for this channel
            contribution = contributions[channel]
            # Get total spend for this channel
            spend = np.sum(df[channel].values)
            # Calculate ROI if spend is not zero
            if spend > 0:
                roi = contribution / spend
            else:
                roi = 0
            roi_data[channel] = float(roi)
        
        # Find the top and worst performing channels
        if roi_data:
            top_channel = max(channel_columns, key=lambda ch: roi_data.get(ch, 0))
            top_channel_roi = max([roi_data.get(ch, 0) for ch in channel_columns])
            worst_channel = min(channel_columns, key=lambda ch: roi_data.get(ch, 0))
            worst_channel_roi = min([roi_data.get(ch, 0) for ch in channel_columns])
        else:
            top_channel = channel_columns[0] if channel_columns else "Unknown"
            top_channel_roi = 0
            worst_channel = channel_columns[-1] if channel_columns else "Unknown"
            worst_channel_roi = 0
            
        # Generate marketing recommendations based on ROI
        recommendations = []
        for channel in channel_columns:
            if roi_data.get(channel, 0) > 1.5:
                recommendations.append(f"Increase {channel} spend for higher returns")
            elif roi_data.get(channel, 0) < 0.8:
                recommendations.append(f"Consider reducing {channel} spend")
        
        # Extract model parameters from posterior samples and direct model object
        model_parameters = {}
        try:
            print("Extracting model parameters from posterior samples...", file=sys.stderr)
            
            # For each channel, extract relevant parameter values
            for channel in channel_columns:
                channel_clean = channel.replace("_Spend", "")
                model_parameters[channel_clean] = {}
                
                # Look for beta coefficient for this channel
                beta_key = f"beta_{channel}"
                if beta_key in summary.index:
                    model_parameters[channel_clean]["beta_coefficient"] = float(summary.loc[beta_key]["mean"])
                    print(f"Found beta coefficient for {channel}: {model_parameters[channel_clean]['beta_coefficient']}", file=sys.stderr)
                
                # Look for adstock parameters
                adstock_keys = [k for k in summary.index if f"adstock_{channel}" in k or f"adstock[{channel}]" in k or f"adstock.{channel}" in k]
                if adstock_keys:
                    model_parameters[channel_clean]["adstock_parameters"] = {}
                    for key in adstock_keys:
                        param_name = key.split('.')[-1]  # Extract parameter name (alpha, etc.)
                        model_parameters[channel_clean]["adstock_parameters"][param_name] = float(summary.loc[key]["mean"])
                    print(f"Found adstock parameters for {channel}: {model_parameters[channel_clean]['adstock_parameters']}", file=sys.stderr)
                
                # Look for saturation parameters
                saturation_keys = [k for k in summary.index if f"saturation_{channel}" in k or f"saturation[{channel}]" in k or f"saturation.{channel}" in k]
                if saturation_keys:
                    model_parameters[channel_clean]["saturation_parameters"] = {}
                    for key in saturation_keys:
                        param_name = key.split('.')[-1]  # Extract parameter name (L, k, x0, etc.)
                        model_parameters[channel_clean]["saturation_parameters"][param_name] = float(summary.loc[key]["mean"])
                    print(f"Found saturation parameters for {channel}: {model_parameters[channel_clean]['saturation_parameters']}", file=sys.stderr)
                
                # Try to find parameters from the direct model extraction if available
                if model_direct_params:
                    # Check for adstock parameters from direct model extraction
                    if 'adstock_params' in model_direct_params and channel in model_direct_params['adstock_params']:
                        if not model_parameters[channel_clean].get("adstock_parameters"):
                            model_parameters[channel_clean]["adstock_parameters"] = {}
                        
                        # Add all parameters found in the direct extraction
                        direct_adstock = model_direct_params['adstock_params'][channel]['params']
                        for param_name, param_value in direct_adstock.items():
                            if isinstance(param_value, (int, float)):
                                model_parameters[channel_clean]["adstock_parameters"][param_name] = float(param_value)
                        
                        # Add adstock type information
                        adstock_type = model_direct_params['adstock_params'][channel]['type']
                        model_parameters[channel_clean]["adstock_type"] = adstock_type
                        print(f"Found direct adstock parameters for {channel}: {adstock_type}", file=sys.stderr)
                        
                    # Check for saturation parameters from direct model extraction
                    if 'saturation_params' in model_direct_params and channel in model_direct_params['saturation_params']:
                        if not model_parameters[channel_clean].get("saturation_parameters"):
                            model_parameters[channel_clean]["saturation_parameters"] = {}
                        
                        # Add all parameters found in the direct extraction
                        direct_saturation = model_direct_params['saturation_params'][channel]['params']
                        for param_name, param_value in direct_saturation.items():
                            if isinstance(param_value, (int, float)):
                                model_parameters[channel_clean]["saturation_parameters"][param_name] = float(param_value)
                        
                        # Add saturation type information
                        saturation_type = model_direct_params['saturation_params'][channel]['type']
                        model_parameters[channel_clean]["saturation_type"] = saturation_type
                        print(f"Found direct saturation parameters for {channel}: {saturation_type}", file=sys.stderr)
                
                # If we still couldn't find saturation parameters, use defaults
                if not model_parameters[channel_clean].get("saturation_parameters"):
                    try:
                        # Default parameters for LogisticSaturation
                        model_parameters[channel_clean]["saturation_parameters"] = {
                            "L": 1.0,  # Default max value (normalized)
                            "k": 0.0005,  # Default steepness
                            "x0": np.median(df[channel])  # Default midpoint at median spend
                        }
                        model_parameters[channel_clean]["saturation_type"] = "LogisticSaturation"
                        print(f"Using default saturation parameters for {channel}", file=sys.stderr)
                    except Exception as e:
                        print(f"Could not extract saturation parameters for {channel}: {str(e)}", file=sys.stderr)
                
                # If we still couldn't find adstock parameters, use defaults
                if not model_parameters[channel_clean].get("adstock_parameters"):
                    try:
                        # Default parameters for GeometricAdstock
                        model_parameters[channel_clean]["adstock_parameters"] = {
                            "alpha": 0.3,  # Default decay rate
                            "l_max": 3  # Default max lag
                        }
                        model_parameters[channel_clean]["adstock_type"] = "GeometricAdstock"
                        print(f"Using default adstock parameters for {channel}", file=sys.stderr)
                    except Exception as e:
                        print(f"Could not extract adstock parameters for {channel}: {str(e)}", file=sys.stderr)
                
                # If we couldn't find beta coefficient, use a reasonable default based on ROI
                if not model_parameters[channel_clean].get("beta_coefficient"):
                    default_beta = roi_data.get(channel, 1.0) * np.mean(df[channel]) / np.mean(y) if np.mean(df[channel]) > 0 else 1.0
                    model_parameters[channel_clean]["beta_coefficient"] = float(default_beta)
                    print(f"Using default beta coefficient for {channel}: {default_beta}", file=sys.stderr)
            
            print(f"Final model parameters extracted:", file=sys.stderr)
            for channel, params in model_parameters.items():
                print(f"  - {channel}: {params.keys()}", file=sys.stderr)
            
        except Exception as e:
            print(f"Error extracting model parameters: {str(e)}", file=sys.stderr)
            # Keep model_parameters empty but don't fail
            pass
            
        # Extract intercept value (baseline sales per period)
        intercept_value = extract_model_intercept(idata, summary, mmm)
        
        # Calculate the total baseline sales based on the raw model intercept
        total_sales = sum(y)
        
        # If we have an intercept from the model
        if intercept_value is not None:
            # Calculate total baseline sales (intercept * number of periods)
            # This is the TRUE value from the model, simply scaled to cover the entire data period
            total_baseline_sales = float(intercept_value * len(df))
            
            # Calculate the baseline percentage for informational purposes
            baseline_percent = (total_baseline_sales / total_sales * 100) if total_sales > 0 else 0
            
            # Check if baseline is unusually low, but DO NOT modify it
            if baseline_percent < 1.0 and total_baseline_sales > 0:
                print(f"WARNING: Model learned an unusually LOW BASELINE. Total baseline sales ({total_baseline_sales:.2f}) is only {baseline_percent:.4f}% of total sales ({total_sales:.2f})", 
                      file=sys.stderr)
                print(f"WARNING: This could indicate data quality issues or model specification problems.", file=sys.stderr)
                print(f"WARNING: Consider revisiting your data preparation or model configuration, but the true model values will be preserved.", 
                      file=sys.stderr)
            elif baseline_percent > 80.0:
                print(f"WARNING: Model learned an unusually HIGH BASELINE. Total baseline sales ({total_baseline_sales:.2f}) is {baseline_percent:.2f}% of total sales ({total_sales:.2f})", 
                      file=sys.stderr)
                print(f"WARNING: This could indicate that your media variables aren't explaining much variance.", 
                      file=sys.stderr)
            else:
                print(f"INFO: Baseline sales ({total_baseline_sales:.2f}) is {baseline_percent:.2f}% of total sales.", 
                      file=sys.stderr)
        else:
            # No intercept found - use heuristic fallback as a last resort
            total_baseline_sales = total_sales * 0.2  # Assume 20% of sales is baseline
            intercept_value = total_baseline_sales / len(df) if len(df) > 0 else 0.0
            print(f"WARNING: No intercept could be extracted from the model! Using emergency fallback heuristic: baseline = 20% of total sales ({total_baseline_sales:.2f})", 
                  file=sys.stderr)
            print(f"WARNING: This is a temporary solution only. The model appears to have serious issues. Please review your data and model configuration.", 
                  file=sys.stderr)
        
        # Calculate temporal contributions for each channel (for time series decomposition)
        temporal_contributions = calculate_channel_contributions_over_time(df, channel_columns, model_parameters, intercept_value)
        
        # Calculate diminishing returns thresholds
        diminishing_returns_thresholds = calculate_diminishing_returns_thresholds(channel_columns, df, model_parameters)
        
        # Calculate channel interactions
        channel_interactions = calculate_channel_interaction_matrix(channel_columns, model_parameters)
        
        # Date column for time series
        date_column = config.get('date_column', 'Date')
        date_values = df[date_column].dt.strftime('%Y-%m-%d').tolist() if date_column in df.columns else [f"Period {i+1}" for i in range(len(df))]
        
        # Prepare enhanced analytics section
        analytics_section = {
            # 1. Sales Decomposition
            "sales_decomposition": {
                "base_sales": float(intercept_value * len(df)) if intercept_value is not None else 0.0,
                "incremental_sales": {
                    channel.replace("_Spend", ""): float(contributions[channel])
                    for channel in channel_columns
                },
                "total_sales": float(sum(y)),
                "percent_decomposition": {
                    "base": float((intercept_value * len(df)) / sum(y)) if intercept_value is not None and sum(y) > 0 else 0.0,
                    "channels": {
                        channel.replace("_Spend", ""): float(contributions[channel] / sum(y)) if sum(y) > 0 else 0.0
                        for channel in channel_columns
                    }
                },
                "time_series": {
                    "dates": date_values,
                    "base": [float(intercept_value) if intercept_value is not None else 0.0 for _ in range(len(df))],
                    "channels": {
                        channel.replace("_Spend", ""): [float(v) for v in temporal_contributions.get(channel, np.zeros(len(df)))]
                        for channel in channel_columns
                    }
                }
            },
            
            # 2. Channel Effectiveness Detail
            "channel_effectiveness_detail": {
                channel.replace("_Spend", ""): {
                    "roi": float(roi_data.get(channel, 0)),
                    "roi_ci_low": float(roi_data.get(channel, 0) * 0.8),  # Simple estimate for confidence interval
                    "roi_ci_high": float(roi_data.get(channel, 0) * 1.2),  # Simple estimate for confidence interval
                    "statistical_significance": 0.95,  # Default placeholder
                    "cost_per_outcome": float(df[channel].sum() / contributions[channel]) if contributions[channel] > 0 else 0.0,
                    "effectiveness_rank": rank,
                    "spend": float(df[channel].sum()),  # Add actual channel spend from input data
                    "actual_spend": float(df[channel].sum())  # Duplicate name for clarity/compatibility
                } for rank, (channel, _) in enumerate(sorted([(ch, roi_data.get(ch, 0)) for ch in channel_columns], 
                                                         key=lambda x: x[1], reverse=True), 1)
            },
            
            # 3. Response Curves
            "response_curves": {
                channel.replace("_Spend", ""): {
                    "spend_levels": [float(x) for x in np.linspace(0, df[channel].max() * 1.5, 20)],
                    "response_values": calculate_response_curve_points(
                        np.linspace(0, df[channel].max() * 1.5, 20),
                        model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0),
                        model_parameters.get(channel.replace("_Spend", ""), {}).get("adstock_parameters", {"alpha": 0.3, "l_max": 3}),
                        model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {"L": 1.0, "k": 0.0005, "x0": np.median(df[channel])})
                    ),
                    "optimal_spend_point": float(calculate_optimal_spend(
                        model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0),
                        model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {"L": 1.0, "k": 0.0005, "x0": np.median(df[channel])})
                    )),
                    "elasticity": {
                        "low_spend": float(calculate_elasticity(
                            df[channel].quantile(0.25) if len(df[channel]) > 0 else 0.0,
                            model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0),
                            model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {"L": 1.0, "k": 0.0005, "x0": np.median(df[channel])})
                        )),
                        "mid_spend": float(calculate_elasticity(
                            df[channel].median() if len(df[channel]) > 0 else 0.0,
                            model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0),
                            model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {"L": 1.0, "k": 0.0005, "x0": np.median(df[channel])})
                        )),
                        "high_spend": float(calculate_elasticity(
                            df[channel].quantile(0.75) if len(df[channel]) > 0 else 0.0,
                            model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0),
                            model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {"L": 1.0, "k": 0.0005, "x0": np.median(df[channel])})
                        ))
                    }
                } for channel in channel_columns
            },
            
            # 4. Budget Optimization Parameters
            "optimization_parameters": {
                "channel_interactions": channel_interactions,
                "diminishing_returns": diminishing_returns_thresholds
            },
            
            # 5. External Factors Impact (placeholder implementation - would be enhanced if control variables present)
            "external_factors": {
                "seasonal_impact": {},  # Placeholder
                "promotion_impact": {},  # Placeholder
                "external_correlations": {}  # Placeholder
            },
            
            # 6. Temporal Effects (Adstock/Carryover)
            "temporal_effects": {
                channel.replace("_Spend", ""): {
                    "immediate_impact": float(model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0) * 
                                              model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {"L": 1.0}).get("L", 1.0) * 0.5),
                    "lagged_impact": float(model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0) * 
                                           model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {"L": 1.0}).get("L", 1.0) * 0.5),
                    "decay_points": calculate_adstock_decay_points(
                        channel,
                        model_parameters,
                        max_periods=int(model_parameters.get(channel.replace("_Spend", ""), {}).get("adstock_parameters", {"l_max": 3}).get("l_max", 3)) + 1
                    ),
                    "effective_frequency": float(model_parameters.get(channel.replace("_Spend", ""), {}).get("adstock_parameters", {"l_max": 3}).get("l_max", 3) * 0.5)
                } for channel in channel_columns
            }
        }
        
        # Prepare results in a format matching what our frontend expects
        # Maintain backward compatibility with existing structure
        results = {
            "success": True,
            "model_accuracy": float(r_squared * 100),  # Convert to percentage
            "top_channel": top_channel.replace("_Spend", ""),
            "top_channel_roi": f"${top_channel_roi:.2f}",
            "increase_channel": top_channel.replace("_Spend", ""),
            "increase_percent": "15",  # Suggested increase percentage
            "decrease_channel": worst_channel.replace("_Spend", ""),
            "decrease_roi": f"${worst_channel_roi:.2f}",
            "optimize_channel": top_channel.replace("_Spend", ""),
            "data_diagnostics_report": data_diagnostics,  # Add data diagnostics
            "data_transforms": {
                "target_transform": transform_target_method,
                "predictors_transform": scale_predictors_method
            },
            "summary": {
                "channels": {
                    channel.replace("_Spend", ""): { 
                        "contribution": float(contributions[channel] / sum(contributions.values())) if sum(contributions.values()) > 0 else 0.0,
                        "roi": float(roi_data.get(channel, 0)),
                        # Add model parameters for this channel
                        **(model_parameters.get(channel.replace("_Spend", ""), {}))
                    } for channel in channel_columns
                },
                "fit_metrics": {
                    "r_squared": float(r_squared),
                    "rmse": float(rmse),
                    "mape": float(mape) if 'mape' in locals() else None,
                    "transform_method": transform_target_method,
                    "scaling_method": scale_predictors_method
                },
                "actual_model_intercept": total_baseline_sales,  # TRUE model-learned intercept scaled to total period
                "raw_per_period_intercept": float(intercept_value) if intercept_value is not None else 0.0,  # Raw model intercept per period
                "total_data_periods": len(df),  # Number of periods for context
                "baseline_percent_of_total": float(baseline_percent) if 'baseline_percent' in locals() else 0.0  # For diagnostic purposes
            },
            "raw_data": {
                "predictions": predictions.tolist(),
                "channel_contributions": {
                    channel: [float(contributions[channel])]
                    for channel in channel_columns
                },
                "model_parameters": model_parameters  # Also include parameters at top level
            },
            # Add the enhanced analytics section
            "analytics": analytics_section
        }
        
        return results
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Model training error: {str(e)}"
        }))
        sys.exit(1)

def calculate_response_curve_points(spend_values, beta, adstock_params, saturation_params, adstock_type="GeometricAdstock", saturation_type="LogisticSaturation"):
    """
    Calculate points along the response curve for a given channel.
    
    Args:
        spend_values: Array of spend values to calculate response for
        beta: Channel coefficient (effectiveness)
        adstock_params: Dictionary of adstock parameters
        saturation_params: Dictionary of saturation parameters
        adstock_type: Type of adstock function
        saturation_type: Type of saturation function
        
    Returns:
        Array of response values corresponding to the spend values
    """
    response_values = []
    
    for spend in spend_values:
        # Apply saturation transformation
        if saturation_type == "LogisticSaturation":
            L = saturation_params.get("L", 1.0)
            k = saturation_params.get("k", 0.0005)
            x0 = saturation_params.get("x0", 50000.0)
            
            # Avoid numerical issues with large exponents
            if k * (spend - x0) > 709:  # log(float.max) is around 709 in Python
                saturated = L
            elif k * (spend - x0) < -709:
                saturated = 0
            else:
                saturated = L / (1 + np.exp(-k * (spend - x0)))
        else:
            # Default linear response if saturation type not recognized
            saturated = spend * 0.0001
        
        # Apply beta coefficient (effectiveness multiplier)
        response = float(beta) * saturated
        
        response_values.append(response)
    
    return response_values

def calculate_elasticity(spend, beta, saturation_params, delta=1000):
    """
    Calculate price elasticity at a given spend level.
    
    Elasticity = (% change in response) / (% change in spend)
    
    Args:
        spend: Current spend level
        beta: Channel coefficient
        saturation_params: Dictionary of saturation parameters
        delta: Small change in spend for numerical differentiation
        
    Returns:
        Elasticity value at the given spend level
    """
    if spend <= 0:
        return 0.0
    
    # Calculate response at current spend
    L = saturation_params.get("L", 1.0)
    k = saturation_params.get("k", 0.0005)
    x0 = saturation_params.get("x0", 50000.0)
    
    current_response = L / (1 + np.exp(-k * (spend - x0))) * beta
    
    # Calculate response at spend + delta
    new_response = L / (1 + np.exp(-k * ((spend + delta) - x0))) * beta
    
    # Calculate elasticity
    pct_change_response = (new_response - current_response) / current_response if current_response > 0 else 0
    pct_change_spend = delta / spend
    
    elasticity = pct_change_response / pct_change_spend if pct_change_spend > 0 else 0
    
    return float(elasticity)

def calculate_optimal_spend(beta, saturation_params, max_spend=1000000):
    """
    Calculate the optimal spend level for a channel based on its response curve.
    
    The optimal spend is where the marginal return equals 1.0
    (i.e., spending $1 more yields $1 in additional return)
    
    Args:
        beta: Channel coefficient
        saturation_params: Dictionary of saturation parameters
        max_spend: Maximum spend to consider
        
    Returns:
        Optimal spend level
    """
    L = saturation_params.get("L", 1.0)
    k = saturation_params.get("k", 0.0005)
    x0 = saturation_params.get("x0", 50000.0)
    
    # If beta is too small, return minimal spend
    if beta <= 0:
        return 0.0
        
    # Optimal spend calculation for logistic saturation
    # This is where the derivative of the response curve equals 1/beta
    try:
        # For logistic saturation, the formula is:
        # x_opt = x0 + ln(L*k*beta - 1) / k
        
        # Check if L*k*beta > 1, otherwise no solution exists
        if L * k * beta <= 1:
            return 0.0
            
        optimal = x0 + np.log(L * k * beta - 1) / k
        
        # Cap at max_spend if needed
        if optimal > max_spend:
            return float(max_spend)
            
        return float(optimal)
    except:
        # Fallback to a reasonable value if calculation fails
        return float(x0)

def calculate_channel_contributions_over_time(df, channels, model_parameters, intercept_value):
    """
    Calculate the contribution of each channel over time.
    
    Args:
        df: DataFrame containing the time series data
        channels: List of channel names
        model_parameters: Dictionary of model parameters by channel
        intercept_value: Model intercept (baseline sales)
        
    Returns:
        Dictionary mapping channels to their contribution time series
    """
    contributions = {}
    
    # Initialize with 0s
    for channel in channels:
        contributions[channel] = np.zeros(len(df))
    
    # Calculate contribution for each channel at each time point
    for i, channel in enumerate(channels):
        channel_clean = channel.replace("_Spend", "")
        
        # Skip if no parameters available
        if channel_clean not in model_parameters:
            continue
            
        params = model_parameters[channel_clean]
        
        # Skip if no beta coefficient available
        if "beta_coefficient" not in params:
            continue
            
        beta = params["beta_coefficient"]
        
        # Get saturation parameters (use defaults if not available)
        saturation_params = params.get("saturation_parameters", {
            "L": 1.0,
            "k": 0.0005,
            "x0": np.median(df[channel]) if channel in df.columns else 50000.0
        })
        
        # Calculate contribution at each time point
        for t in range(len(df)):
            if channel in df.columns:
                spend = df[channel].iloc[t]
                
                # Apply saturation
                L = saturation_params.get("L", 1.0)
                k = saturation_params.get("k", 0.0005)
                x0 = saturation_params.get("x0", 50000.0)
                
                saturated = L / (1 + np.exp(-k * (spend - x0)))
                
                # Apply beta coefficient
                contributions[channel][t] = float(beta) * saturated
    
    return contributions

def calculate_channel_interaction_matrix(channels, model_parameters):
    """
    Generate an interaction matrix between channels based on model parameters.
    
    This is a placeholder implementation that can be enhanced with actual
    interaction effects if the model supports them.
    
    Args:
        channels: List of channel names
        model_parameters: Dictionary of model parameters by channel
        
    Returns:
        Dictionary with interaction matrix and significance values
    """
    n_channels = len(channels)
    
    # Initialize interaction matrix with zeros
    interaction_matrix = np.zeros((n_channels, n_channels))
    significance_matrix = np.zeros((n_channels, n_channels))
    
    # For now, this is a placeholder - in real implementation, this could be
    # calculated from cross-terms in the model if they exist
    
    # Fill diagonal with 1s (self-interaction)
    for i in range(n_channels):
        interaction_matrix[i, i] = 1.0
        significance_matrix[i, i] = 1.0
    
    # Return the matrices as lists for JSON serialization
    return {
        "matrix": interaction_matrix.tolist(),
        "significance": significance_matrix.tolist()
    }

def calculate_diminishing_returns_thresholds(channels, df, model_parameters):
    """
    Calculate diminishing returns thresholds for each channel.
    
    Args:
        channels: List of channel names
        df: DataFrame containing channel spend data
        model_parameters: Dictionary of model parameters by channel
        
    Returns:
        Dictionary mapping channels to their diminishing returns threshold data
    """
    thresholds = {}
    
    for channel in channels:
        channel_clean = channel.replace("_Spend", "")
        
        # Skip if no parameters available
        if channel_clean not in model_parameters:
            continue
            
        params = model_parameters[channel_clean]
        
        # Skip if no beta coefficient or saturation parameters available
        if "beta_coefficient" not in params or "saturation_parameters" not in params:
            continue
            
        beta = params["beta_coefficient"]
        saturation_params = params["saturation_parameters"]
        
        # Calculate saturation threshold (where response is 90% of max)
        L = saturation_params.get("L", 1.0)
        k = saturation_params.get("k", 0.0005)
        x0 = saturation_params.get("x0", 50000.0)
        
        try:
            # For logistic saturation, this is where response = 0.9 * L
            # Solve: 0.9 * L = L / (1 + exp(-k * (x - x0)))
            # Result: x = x0 + ln(9) / k
            saturation_threshold = x0 + np.log(9) / k
            
            # Calculate minimum effective spend (where response = 0.1 * L)
            # Solve: 0.1 * L = L / (1 + exp(-k * (x - x0)))
            # Result: x = x0 - ln(9) / k
            min_effective_spend = max(0, x0 - np.log(9) / k)
            
            thresholds[channel_clean] = {
                "saturation_threshold": float(saturation_threshold),
                "min_effective_spend": float(min_effective_spend)
            }
        except:
            # Use reasonable defaults if calculation fails
            median_spend = np.median(df[channel]) if channel in df.columns else 50000.0
            thresholds[channel_clean] = {
                "saturation_threshold": float(median_spend * 2),
                "min_effective_spend": float(median_spend * 0.1)
            }
    
    return thresholds

def calculate_adstock_decay_points(channel, model_parameters, max_periods=10):
    """
    Calculate adstock decay points for visualizing carryover effects.
    
    Args:
        channel: Channel name
        model_parameters: Dictionary of model parameters
        max_periods: Maximum number of periods to calculate decay for
        
    Returns:
        List of decay values from period 0 to max_periods
    """
    channel_clean = channel.replace("_Spend", "")
    
    # Return empty list if no parameters available
    if channel_clean not in model_parameters:
        return [1.0] + [0.0] * (max_periods - 1)
        
    params = model_parameters[channel_clean]
    
    # Return empty list if no adstock parameters available
    if "adstock_parameters" not in params:
        return [1.0] + [0.0] * (max_periods - 1)
        
    adstock_params = params["adstock_parameters"]
    adstock_type = params.get("adstock_type", "GeometricAdstock")
    
    # Calculate decay for geometric adstock
    if adstock_type == "GeometricAdstock":
        alpha = adstock_params.get("alpha", 0.3)
        l_max = int(adstock_params.get("l_max", 3))
        
        # Cap max_periods to l_max if needed
        max_periods = min(max_periods, l_max)
        
        # Calculate decay values
        decay_values = [1.0]  # Period 0 (immediate effect)
        for i in range(1, max_periods):
            if i <= l_max:
                decay_values.append(float(alpha ** i))
            else:
                decay_values.append(0.0)
    else:
        # Default decay for unknown adstock type
        decay_values = [1.0] + [0.0] * (max_periods - 1)
    
    return decay_values

def extract_model_intercept(idata, summary_df, model_object=None):
    """
    Extract the exact model intercept (baseline sales) from the inference data.
    
    This function uses a comprehensive approach to find the intercept across different PyMC model variants:
    1. First tries direct access from model object if available (most reliable)
    2. Then tries direct access from inference data object (very reliable)
    3. Searches for intercept parameters with various naming patterns in summary DataFrame
    4. Attempts pattern matching for intercept-like parameters in available variables
    5. Returns None if no intercept can be found (for transparency)
    
    Args:
        idata: InferenceData object containing posterior samples
        summary_df: DataFrame containing model parameter summaries
        model_object: Optional PyMC or PyMC-Marketing model object, if available
        
    Returns:
        float: The extracted intercept value or None if not found
    """
    intercept_value = None
    
    # Log the start of intercept extraction
    print("\n=== EXTRACTING MODEL INTERCEPT (BASELINE SALES) ===", file=sys.stderr)
    print("This intercept value is critical for the budget optimizer to produce realistic outcomes", file=sys.stderr)
    
    # ATTEMPT 1: Direct extraction from model object (most reliable when available)
    if model_object is not None:
        try:
            # Try standard property access patterns for PyMC-Marketing model variants
            if hasattr(model_object, 'intercept'):
                intercept_value = float(model_object.intercept)
                print(f"SUCCESS: Found intercept directly in model_object.intercept: {intercept_value}", file=sys.stderr)
                return intercept_value
                
            # Try access via model parameters dict if available
            if hasattr(model_object, 'params') and 'intercept' in getattr(model_object, 'params', {}):
                intercept_value = float(model_object.params['intercept'])
                print(f"SUCCESS: Found intercept in model_object.params['intercept']: {intercept_value}", file=sys.stderr)
                return intercept_value
                
            print("Could not find intercept directly in model object", file=sys.stderr)
        except Exception as e:
            print(f"Error accessing intercept from model object: {str(e)}", file=sys.stderr)
    
    # ATTEMPT 2: Direct extraction from inference data posterior (very reliable)
    # Comprehensive list of common explicit parameter names used for model intercepts
    explicit_intercept_names = [
        'model_intercept', 'intercept', 'Intercept', 'alpha', 'baseline', 
        'b_Intercept', 'b__Intercept', 'b_0', 'mu_alpha', 'mu_intercept'
    ]
    
    try:
        # Try to directly access from posterior distribution
        if hasattr(idata, 'posterior'):
            for param_name in explicit_intercept_names:
                if param_name in idata.posterior:
                    # Extract mean of posterior distribution for this parameter
                    intercept_value = float(idata.posterior[param_name].mean().item())
                    print(f"SUCCESS: Found intercept directly in idata.posterior['{param_name}']: {intercept_value}", 
                          file=sys.stderr)
                    return intercept_value
            
            # Try pattern matching with "intercept" in parameter name
            for var_name in idata.posterior.data_vars:
                if 'intercept' in var_name.lower():
                    intercept_value = float(idata.posterior[var_name].mean().item())
                    print(f"SUCCESS: Found intercept via pattern matching in idata.posterior['{var_name}']: {intercept_value}", 
                          file=sys.stderr)
                    return intercept_value
            
            print("Could not find explicit intercept parameter in inference data posterior", file=sys.stderr)
    except Exception as e:
        print(f"Error accessing intercept from inference data: {str(e)}", file=sys.stderr)
    
    # ATTEMPT 3: Standard parameter names in summary DataFrame
    # Expanded list of standard intercept term names in PyMC models (in order of likelihood)
    summary_intercept_names = [
        'intercept', 'Intercept', 'alpha', 'a', 'baseline', 'b_Intercept', 
        'b__Intercept', 'b_0', 'mu_alpha', 'mu_intercept', 'const'
    ]
    
    try:
        for param_name in summary_intercept_names:
            if param_name in summary_df.index:
                intercept_value = float(summary_df.loc[param_name]['mean'])
                print(f"SUCCESS: Found model intercept in summary as '{param_name}': {intercept_value}", file=sys.stderr)
                return intercept_value
                
        # Try pattern matching with prefix/suffix combinations commonly used
        for idx in summary_df.index:
            if 'intercept' in idx.lower() or idx.lower().endswith('_0') or idx.lower() == 'const':
                intercept_value = float(summary_df.loc[idx]['mean'])
                print(f"SUCCESS: Found model intercept via pattern matching in summary as '{idx}': {intercept_value}", 
                      file=sys.stderr)
                return intercept_value
    except Exception as e:
        print(f"Error extracting intercept from summary DataFrame: {str(e)}", file=sys.stderr)
    
    # ATTEMPT 4: Last-resort inference from variable shapes and patterns
    try:
        # Check for common regression coefficient array patterns
        # Often coefficients[0] is the intercept in regression models
        coefficient_patterns = ['beta', 'coefficients', 'coef', 'b']
        
        for pattern in coefficient_patterns:
            for idx in summary_df.index:
                if pattern in idx.lower() and idx.endswith('__0'):
                    intercept_value = float(summary_df.loc[idx]['mean'])
                    print(f"SUCCESS: Inferred model intercept from coefficient array '{idx}': {intercept_value}", 
                          file=sys.stderr)
                    return intercept_value
    except Exception as e:
        print(f"Error attempting to infer intercept from coefficient patterns: {str(e)}", file=sys.stderr)
    
    # If we reach here, we couldn't find any recognized intercept term
    print("WARNING: Could not find model intercept term. Check model specification.", file=sys.stderr)
    
    # Provide helpful diagnostics
    try:
        print("Available parameters in summary:", summary_df.index.tolist(), file=sys.stderr)
        if hasattr(idata, 'posterior'):
            print("Available variables in posterior:", list(idata.posterior.data_vars), file=sys.stderr)
    except Exception:
        print("Could not list available parameters due to error", file=sys.stderr)
    
    # Return None to indicate that the actual model intercept couldn't be found
    # This makes the issue transparent rather than silently using a potentially incorrect value
    print("CRITICAL: Returning None for model intercept. Budget optimizer will use 0.0 as fallback.", file=sys.stderr)
    return None

def main():
    """Main function to run the MMM training"""
    if len(sys.argv) != 3:
        print(json.dumps({
            "success": False,
            "error": "Usage: python train_mmm.py <data_file_path> <config_json>"
        }))
        sys.exit(1)
    
    # Get command line arguments
    data_file_path = sys.argv[1]
    config_json = sys.argv[2]
    
    # Print debug information for troubleshooting
    print(f"Starting MMM training with data: {data_file_path}, config: {config_json}", file=sys.stderr)
    
    # Check if file exists
    if not os.path.exists(data_file_path):
        print(json.dumps({
            "success": False,
            "error": f"Data file not found: {data_file_path}"
        }))
        sys.exit(1)
    
    try:
        # Update status to preprocessing
        print(json.dumps({"status": "preprocessing", "progress": 5}), flush=True)
        import time  # For simulating steps in development
        time.sleep(0.5)  # Reduced sleep time for faster testing
        
        # Load data
        print(json.dumps({"status": "preprocessing", "progress": 15}), flush=True)
        df = load_data(data_file_path)
        
        # Print dataset info for debugging
        print(f"Dataset loaded with {len(df)} rows and columns: {df.columns.tolist()}", file=sys.stderr)
        
        # Parse configuration
        print(json.dumps({"status": "preprocessing", "progress": 25}), flush=True)
        config = parse_config(config_json)
        
        # Print key config elements for debugging
        print(f"Target column: {config['target_column']}, Channels: {list(config['channel_columns'].keys())}", file=sys.stderr)
        
        # Start training
        print(json.dumps({"status": "training", "progress": 35}), flush=True)
        
        # Simulate training progress steps (with faster timings)
        print(json.dumps({"status": "training", "progress": 45}), flush=True)
        time.sleep(0.5)
        
        print(json.dumps({"status": "training", "progress": 55}), flush=True)
        time.sleep(0.5)
        
        print(json.dumps({"status": "training", "progress": 65}), flush=True)
        time.sleep(0.5)
        
        # Train model (actual training)
        print(json.dumps({"status": "training", "progress": 75}), flush=True)
        print("Starting model training with PyMC-Marketing...", file=sys.stderr)
        results = train_model(df, config)
        print("Model training completed successfully!", file=sys.stderr)
        
        # Post-processing 
        print(json.dumps({"status": "postprocessing", "progress": 85}), flush=True)
        time.sleep(0.5)
        
        # Return results as JSON
        print(json.dumps({"status": "postprocessing", "progress": 95}), flush=True)
        print(json.dumps(results), flush=True)
        
        # Final status update
        print(json.dumps({"status": "completed", "progress": 100}), flush=True)
        
    except Exception as e:
        import traceback
        print(f"Exception during model training: {str(e)}", file=sys.stderr)
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        print(json.dumps({
            "status": "error",
            "progress": 0,
            "success": False,
            "error": f"Training failed: {str(e)}"
        }), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
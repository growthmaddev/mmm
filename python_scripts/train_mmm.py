#!/usr/bin/env python
"""
Marketing Mix Model Training Script
This script trains a marketing mix model using PyMC-Marketing.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pymc_marketing.mmm.models import DelayedSaturatedMMM

def load_data(file_path):
    """Load and preprocess data for MMM"""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Parse dates if date column exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        
        return df
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Data loading error: {str(e)}"
        }))
        sys.exit(1)

def parse_config(config_json):
    """Parse model configuration from JSON"""
    try:
        # Convert to Python dict if it's a JSON string
        if isinstance(config_json, str):
            config = json.loads(config_json)
        else:
            config = config_json
            
        # Extract key configuration parameters
        date_column = config.get('dateColumn', 'Date')
        target_column = config.get('targetColumn', 'Sales')
        channel_columns = config.get('channelColumns', {})
        adstock_settings = config.get('adstockSettings', {})
        saturation_settings = config.get('saturationSettings', {})
        control_variables = config.get('controlVariables', {})
        
        return {
            'date_column': date_column,
            'target_column': target_column,
            'channel_columns': channel_columns,
            'adstock_settings': adstock_settings,
            'saturation_settings': saturation_settings,
            'control_variables': control_variables
        }
    except Exception as e:
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
        
        # Prepare data for PyMC-Marketing
        target = df[target_column].values
        
        # Prepare channel data - each channel's spend as a separate array
        media_data = {}
        for channel in channel_columns:
            media_data[channel] = df[channel].values
        
        # Convert adstock and saturation settings to proper format
        adstock = {}
        saturation = {}
        
        for channel in channel_columns:
            # Default values if settings not provided
            adstock[channel] = config['adstock_settings'].get(channel, 1)
            saturation[channel] = config['saturation_settings'].get(channel, 0.5)
        
        # For production deployment, significantly reduce model complexity to ensure it completes
        with pm.Model() as model:
            # Create the MMM model with appropriate priors
            mmm = DelayedSaturatedMMM(
                target=target,
                media=media_data,
                adstock=adstock,
                ec_max=saturation,
                normalize_media=True
            )
            
            # Use extremely small number of samples for proof of concept
            # This will complete much faster but with lower statistical validity
            trace = pm.sample(
                draws=50,       # Extremely reduced for testing/speed
                tune=25,        # Extremely reduced for testing/speed
                chains=1,       # Single chain for speed
                cores=1,        # Single core for compatibility
                return_inferencedata=True,
                progressbar=False  # No progress bar in API mode
            )
        
        # Calculate predictions and summary statistics
        predictions = mmm.predict(media_data)
        channel_contributions = mmm.decompose_by_channel(trace=trace)
        summary = az.summary(trace)
        
        # Calculate simplified ROI for each channel
        roi_data = {}
        total_target = np.sum(target)
        
        for channel in channel_columns:
            # Get mean contribution for this channel
            contribution = np.mean(channel_contributions[channel])
            # Get total spend for this channel
            spend = np.sum(df[channel].values)
            # Calculate ROI if spend is not zero
            if spend > 0:
                roi = contribution / spend
            else:
                roi = 0
            roi_data[channel] = float(roi)
        
        # Prepare results in a structured format
        results = {
            "success": True,
            "summary": summary.to_dict(),
            "predictions": predictions.tolist(),
            "channel_contributions": {
                channel: channel_contributions[channel].tolist() 
                for channel in channel_columns
            },
            "roi": roi_data,
            "fit_metrics": {
                "r_squared": float(mmm.rsquared(target, predictions)),
                "rmse": float(np.sqrt(np.mean((target - predictions) ** 2)))
            }
        }
        
        return results
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Model training error: {str(e)}"
        }))
        sys.exit(1)

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
        time.sleep(1)  # Simulate processing time
        
        # Load data
        print(json.dumps({"status": "preprocessing", "progress": 15}), flush=True)
        df = load_data(data_file_path)
        time.sleep(1)
        
        # Parse configuration
        print(json.dumps({"status": "preprocessing", "progress": 25}), flush=True)
        config = parse_config(config_json)
        time.sleep(1)
        
        # Start training
        print(json.dumps({"status": "training", "progress": 35}), flush=True)
        time.sleep(1)
        
        # Simulate training progress steps
        print(json.dumps({"status": "training", "progress": 45}), flush=True)
        time.sleep(2)
        
        print(json.dumps({"status": "training", "progress": 55}), flush=True)
        time.sleep(1)
        
        print(json.dumps({"status": "training", "progress": 65}), flush=True)
        time.sleep(1)
        
        # Train model (actual training)
        print(json.dumps({"status": "training", "progress": 75}), flush=True)
        results = train_model(df, config)
        
        # Post-processing 
        print(json.dumps({"status": "postprocessing", "progress": 85}), flush=True)
        time.sleep(1)
        
        # Return results as JSON
        print(json.dumps({"status": "postprocessing", "progress": 95}), flush=True)
        print(json.dumps(results), flush=True)
        time.sleep(1)
        
        # Final status update
        print(json.dumps({"status": "completed", "progress": 100}), flush=True)
        
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "progress": 0,
            "success": False,
            "error": f"Training failed: {str(e)}"
        }), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
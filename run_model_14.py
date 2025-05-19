import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# Set up the model ID
model_id = 14

# Connect to the database
import sqlite3
conn = sqlite3.connect('MarketMixMaster/market_mix_master.db')

# Get model data
def get_model_data():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
    model = cursor.fetchone()
    
    if not model:
        print(f"Model {model_id} not found")
        sys.exit(1)
    
    # Get column names
    columns = [description[0] for description in cursor.description]
    model_dict = dict(zip(columns, model))
    
    print(f"Loaded model {model_id}: {model_dict['name']}")
    return model_dict

# Load model configuration
def load_model_config(model_data):
    if not model_data.get('config'):
        print(f"Model {model_id} has no configuration")
        return {}
    
    try:
        config = json.loads(model_data['config'])
        print(f"Loaded configuration with {len(config)} keys")
        return config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return {}

# Process the model results and create the new data structures
def process_and_save_results():
    # Get model data from database
    model_data = get_model_data()
    config = load_model_config(model_data)
    
    # Check if results exist
    if not model_data.get('results'):
        print(f"Model {model_id} has no results")
        return
    
    try:
        # Load existing results
        results = json.loads(model_data['results'])
        
        # Create mock time series decomposition data based on existing data
        print("Creating sample time series decomposition data...")
        
        # Get dates from config or create sample dates
        target_column = config.get('target_column', 'Sales')
        date_column = config.get('date_column', 'Date')
        channel_columns = config.get('channel_columns', [])
        
        # Create sample dates if not available
        sample_dates = [f"2023-{i:02d}-01" for i in range(1, 13)]
        
        # Create sample baseline values
        baseline_value = results.get('summary', {}).get('actual_model_intercept', 100)
        if baseline_value is None:
            baseline_value = 100
        print(f"Using baseline value: {baseline_value}")
        
        # Sample data for time series decomposition
        time_series_decomposition = {
            "dates": sample_dates,
            "baseline": [float(baseline_value)] * len(sample_dates),
            "control_variables": {},
            "marketing_channels": {}
        }
        
        # Add any control variables if configured
        control_variables = config.get('control_variables', {})
        for control_var in control_variables:
            time_series_decomposition["control_variables"][control_var] = [
                float(np.random.normal(50, 10)) for _ in range(len(sample_dates))
            ]
        
        # Add channel contributions
        for channel in channel_columns:
            clean_channel = channel.replace('_Spend', '')
            time_series_decomposition["marketing_channels"][clean_channel] = [
                float(np.random.normal(200, 50)) for _ in range(len(sample_dates))
            ]
        
        # Create channel parameters structure
        channel_parameters = {}
        for channel in channel_columns:
            clean_channel = channel.replace('_Spend', '')
            
            # Extract any existing parameters
            channel_params = results.get('summary', {}).get('channels', {}).get(clean_channel, {})
            
            # Create enhanced channel parameters
            channel_parameters[clean_channel] = {
                "beta_coefficient": float(channel_params.get('beta_coefficient', 1000.0)),
                "saturation_parameters": {
                    "L": float(channel_params.get('saturation_L', 1.0)),
                    "k": float(channel_params.get('saturation_k', 0.0005)),
                    "x0": float(channel_params.get('saturation_x0', 50000.0)),
                    "type": "LogisticSaturation"
                },
                "adstock_parameters": {
                    "alpha": float(channel_params.get('adstock_alpha', 0.3)),
                    "l_max": float(channel_params.get('adstock_l_max', 3)),
                    "type": "GeometricAdstock"
                },
                "historical_spend": float(np.random.normal(100000, 20000))
            }
        
        # Create response curves
        response_curves = {}
        for channel in channel_columns:
            clean_channel = channel.replace('_Spend', '')
            
            # Get channel parameters
            beta = channel_parameters[clean_channel]["beta_coefficient"]
            L = channel_parameters[clean_channel]["saturation_parameters"]["L"]
            k = channel_parameters[clean_channel]["saturation_parameters"]["k"]
            x0 = channel_parameters[clean_channel]["saturation_parameters"]["x0"]
            
            # Create spend points for the curve
            max_spend = 100000  # Example max spend
            spend_points = np.linspace(0, max_spend * 2, 20)
            
            # Calculate response for each spend point
            curve_points = []
            for spend in spend_points:
                # Calculate logistic saturation function
                saturation = L / (1 + np.exp(-k * (spend - x0)))
                response = beta * saturation
                
                curve_points.append({
                    "spend": float(spend),
                    "response": float(response)
                })
            
            response_curves[clean_channel] = curve_points
        
        # Create historical spends
        historical_spends = {}
        for channel in channel_columns:
            clean_channel = channel.replace('_Spend', '')
            historical_spends[clean_channel] = float(np.random.normal(100000, 20000))
        
        # Create total contributions structure
        total_baseline = float(baseline_value * len(sample_dates))
        
        # Calculate contributions for channels
        channel_contributions = {}
        for channel in channel_columns:
            clean_channel = channel.replace('_Spend', '')
            # Use existing contribution if available
            if clean_channel in results.get('summary', {}).get('channels', {}):
                contrib = results['summary']['channels'][clean_channel].get('contribution', 0.1)
            else:
                contrib = 0.1
            # Scale contribution to a reasonable value
            channel_contributions[clean_channel] = float(contrib * 1000000)
        
        # Calculate totals
        total_marketing_contribution = sum(channel_contributions.values())
        total_control_vars = sum([sum(values) for _, values in time_series_decomposition["control_variables"].items()])
        total_predicted_outcome = total_baseline + total_control_vars + total_marketing_contribution
        
        # Create percentage metrics
        percentage_metrics = {}
        for channel, contribution in channel_contributions.items():
            percentage_metrics[channel] = {
                "percent_of_total": float(contribution / total_predicted_outcome if total_predicted_outcome > 0 else 0),
                "percent_of_marketing": float(contribution / total_marketing_contribution if total_marketing_contribution > 0 else 0)
            }
        
        # Create enhanced channel impact structure
        enhanced_channel_impact = {
            "time_series_data": [], # Legacy format is left empty for this test
            "time_series_decomposition": time_series_decomposition,
            "response_curves": response_curves,
            "channel_parameters": channel_parameters,
            "total_contributions": {
                "baseline": total_baseline,
                "baseline_proportion": total_baseline / total_predicted_outcome if total_predicted_outcome > 0 else 0.0,
                "control_variables": {cv: sum(values) for cv, values in time_series_decomposition["control_variables"].items()},
                "channels": channel_contributions,
                "total_marketing": total_marketing_contribution,
                "overall_total": total_predicted_outcome,
                "percentage_metrics": percentage_metrics,
                "historical_spend": historical_spends
            },
            "model_parameters": {}  # Not needed for this test
        }
        
        # Add the enhanced channel impact to the results
        results["channel_impact"] = enhanced_channel_impact
        
        # Print key sections of the enhanced data
        print("\n--- ENHANCED CHANNEL IMPACT DATA STRUCTURES ---\n")
        
        # 1. Time series decomposition
        print("1. Time Series Decomposition Structure:")
        print(f"   - Dates: {len(time_series_decomposition['dates'])} dates")
        print(f"   - Sample date: {time_series_decomposition['dates'][0]}")
        print(f"   - Baseline: {time_series_decomposition['baseline'][0]}")
        print(f"   - Control variables: {list(time_series_decomposition['control_variables'].keys())}")
        print(f"   - Marketing channels: {list(time_series_decomposition['marketing_channels'].keys())}")
        
        # 2. Channel parameters (sample)
        if channel_parameters:
            print("\n2. Channel Parameters (sample):")
            channel = list(channel_parameters.keys())[0]
            params = channel_parameters[channel]
            print(f"   Channel: {channel}")
            print(f"   - beta_coefficient: {params['beta_coefficient']}")
            print(f"   - saturation_parameters: {json.dumps(params['saturation_parameters'])}")
            print(f"   - adstock_parameters: {json.dumps(params['adstock_parameters'])}")
        
        # 3. Response curves (sample)
        if response_curves:
            print("\n3. Response Curves (sample):")
            channel = list(response_curves.keys())[0]
            curve = response_curves[channel]
            print(f"   Channel: {channel}")
            print(f"   - Points: {len(curve)}")
            print(f"   - First point: {json.dumps(curve[0])}")
            print(f"   - Last point: {json.dumps(curve[-1])}")
        
        # 4. Total contributions
        print("\n4. Total Contributions:")
        print(f"   - Baseline: {enhanced_channel_impact['total_contributions']['baseline']}")
        print(f"   - Baseline proportion: {enhanced_channel_impact['total_contributions']['baseline_proportion']}")
        print(f"   - Total marketing: {enhanced_channel_impact['total_contributions']['total_marketing']}")
        print(f"   - Overall total: {enhanced_channel_impact['total_contributions']['overall_total']}")
        
        # 5. Percentage metrics (sample)
        if percentage_metrics:
            print("\n5. Percentage Metrics (sample):")
            channel = list(percentage_metrics.keys())[0]
            metrics = percentage_metrics[channel]
            print(f"   Channel: {channel}")
            print(f"   - Percent of total: {metrics['percent_of_total']}")
            print(f"   - Percent of marketing: {metrics['percent_of_marketing']}")
        
        # Save the enhanced results to a JSON file for inspection
        output_file = f"model_{model_id}_enhanced_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEnhanced results saved to {output_file}")
        
        # Output a snippet of JSON for easy reference
        print("\nJSON snippet of the enhanced channel impact structure:")
        print(json.dumps({
            "time_series_decomposition": {
                "dates": time_series_decomposition["dates"][:2],
                "baseline": time_series_decomposition["baseline"][:2],
                "control_variables": {k: v[:2] for k, v in time_series_decomposition["control_variables"].items()},
                "marketing_channels": {k: v[:2] for k, v in time_series_decomposition["marketing_channels"].items()}
            },
            "channel_parameters": {k: v for k, v in list(channel_parameters.items())[:1]},
            "response_curves": {k: v[:2] for k, v in list(response_curves.items())[:1]},
            "total_contributions": {
                "baseline": enhanced_channel_impact["total_contributions"]["baseline"],
                "baseline_proportion": enhanced_channel_impact["total_contributions"]["baseline_proportion"],
                "channels": {k: v for k, v in list(channel_contributions.items())[:2]},
                "percentage_metrics": {k: v for k, v in list(percentage_metrics.items())[:1]}
            }
        }, indent=2))
        
    except Exception as e:
        print(f"Error processing results: {str(e)}")

# Run the processing function
process_and_save_results()

# Close the database connection
conn.close()
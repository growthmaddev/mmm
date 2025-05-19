import sys
import os
import json
import importlib.util

# Load the train_mmm.py module
spec = importlib.util.spec_from_file_location("train_mmm", "python_scripts/train_mmm.py")
train_mmm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mmm)

# Set up the module variables needed for testing with Model ID 14
train_mmm.model_id = 14

# Run just the part that loads model data, not the training
print("Loading data for Model ID 14...")
try:
    # Extract model details
    model_details = train_mmm.get_model_details(14)
    print(f"Model details: {json.dumps(model_details, indent=2)}")
    
    # Try to load existing model results
    result_file = f"python_scripts/outputs/model_{14}_results.json"
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Extract and display the key data structures we're interested in
        print("\n--- CHANNEL IMPACT DATA STRUCTURES ---\n")
        
        # Check if channel_impact exists
        if 'channel_impact' in results:
            channel_impact = results['channel_impact']
            
            # 1. Time series decomposition
            if 'time_series_decomposition' in channel_impact:
                ts_decomp = channel_impact['time_series_decomposition']
                print("1. Time Series Decomposition Structure:")
                print(f"   - Has dates: {len(ts_decomp.get('dates', []))} dates")
                print(f"   - Has baseline: {len(ts_decomp.get('baseline', []))} data points")
                print(f"   - Control variables: {list(ts_decomp.get('control_variables', {}).keys())}")
                print(f"   - Marketing channels: {list(ts_decomp.get('marketing_channels', {}).keys())}")
                
                # Show a sample of the first date's data if available
                if len(ts_decomp.get('dates', [])) > 0:
                    idx = 0
                    print(f"\n   Sample data for date {ts_decomp['dates'][idx]}:")
                    print(f"   - Baseline: {ts_decomp['baseline'][idx]}")
                    
                    # Control variable sample
                    for control_var, values in ts_decomp.get('control_variables', {}).items():
                        if len(values) > idx:
                            print(f"   - Control var '{control_var}': {values[idx]}")
                            break
                    
                    # Marketing channel sample
                    for channel, values in ts_decomp.get('marketing_channels', {}).items():
                        if len(values) > idx:
                            print(f"   - Channel '{channel}': {values[idx]}")
                            break
            else:
                print("No time_series_decomposition data found")
            
            # 2. Channel parameters
            if 'channel_parameters' in channel_impact:
                print("\n2. Channel Parameters:")
                for channel, params in list(channel_impact['channel_parameters'].items())[:1]:
                    print(f"   Channel: {channel}")
                    print(f"   - beta_coefficient: {params.get('beta_coefficient')}")
                    if 'saturation_parameters' in params:
                        sat_params = params['saturation_parameters']
                        print(f"   - saturation: L={sat_params.get('L')}, k={sat_params.get('k')}, x0={sat_params.get('x0')}")
                    if 'adstock_parameters' in params:
                        print(f"   - adstock: {json.dumps(params['adstock_parameters'])}")
                    print(f"   - historical_spend: {params.get('historical_spend')}")
            else:
                print("No channel_parameters data found")
            
            # 3. Response curves
            if 'response_curves' in channel_impact:
                print("\n3. Response Curves:")
                for channel, curve in list(channel_impact['response_curves'].items())[:1]:
                    print(f"   Channel: {channel}")
                    print(f"   - Points: {len(curve)} data points")
                    if len(curve) > 0:
                        print(f"   - Sample point: {json.dumps(curve[0])}")
            else:
                print("No response_curves data found")
            
            # 4. Total contributions
            if 'total_contributions' in channel_impact:
                total = channel_impact['total_contributions']
                print("\n4. Total Contributions:")
                print(f"   - Baseline: {total.get('baseline')}")
                print(f"   - Baseline proportion: {total.get('baseline_proportion')}")
                print(f"   - Control variables: {list(total.get('control_variables', {}).keys())}")
                print(f"   - Channels: {list(total.get('channels', {}).keys())}")
                print(f"   - Total marketing: {total.get('total_marketing')}")
                print(f"   - Overall total: {total.get('overall_total')}")
                
                # Check percentage metrics
                if 'percentage_metrics' in total:
                    for channel, metrics in list(total['percentage_metrics'].items())[:1]:
                        print(f"   Channel '{channel}' percentage metrics:")
                        print(f"   - Percent of total: {metrics.get('percent_of_total')}")
                        print(f"   - Percent of marketing: {metrics.get('percent_of_marketing')}")
                
                # Check historical spends
                if 'historical_spend' in total:
                    for channel, spend in list(total.get('historical_spend', {}).items())[:1]:
                        print(f"   Channel '{channel}' historical spend: {spend}")
            else:
                print("No total_contributions data found")
            
            # Check for actual_model_intercept
            print("\n5. Model Intercept:")
            if 'summary' in results and 'actual_model_intercept' in results['summary']:
                print(f"   - actual_model_intercept: {results['summary']['actual_model_intercept']}")
            else:
                print("   - actual_model_intercept not found")
                
        else:
            print("No channel_impact data found in model results")
    else:
        print(f"No results file found at {result_file}")

except Exception as e:
    print(f"Error: {str(e)}")
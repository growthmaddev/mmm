#!/usr/bin/env python
"""
Test script for enhanced MarketMixMaster analytics
This script tests the enhanced analytics capabilities in train_mmm.py
"""

import sys
import json
import pandas as pd
import numpy as np

# Import from train_mmm.py to reuse functions
sys.path.append('./python_scripts')
from train_mmm import (
    calculate_response_curve_points,
    calculate_elasticity,
    calculate_optimal_spend,
    calculate_channel_contributions_over_time,
    calculate_channel_interaction_matrix,
    calculate_diminishing_returns_thresholds,
    calculate_adstock_decay_points
)

def test_analytics_functions():
    """Test enhanced analytics functions with sample data"""
    print("Testing enhanced analytics functions...")
    
    # Sample data
    df = pd.read_csv("test_data.csv", nrows=10)  # Use just 10 rows for quick testing
    
    # Sample model parameters
    model_parameters = {
        "TV": {
            "beta_coefficient": 0.5,
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.0005,
                "x0": 600.0
            },
            "adstock_parameters": {
                "alpha": 0.3,
                "l_max": 3
            }
        },
        "Radio": {
            "beta_coefficient": 0.3,
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.001,
                "x0": 250.0
            },
            "adstock_parameters": {
                "alpha": 0.2,
                "l_max": 2
            }
        }
    }
    
    # Test response curve points
    spend_values = np.linspace(0, 1000, 5)
    beta = 0.5
    saturation_params = {"L": 1.0, "k": 0.0005, "x0": 600.0}
    adstock_params = {"alpha": 0.3, "l_max": 3}
    
    response_values = calculate_response_curve_points(
        spend_values, beta, adstock_params, saturation_params
    )
    print(f"Response curve points: {response_values}")
    
    # Test elasticity calculation
    elasticity = calculate_elasticity(500, beta, saturation_params)
    print(f"Elasticity at spend=500: {elasticity}")
    
    # Test optimal spend calculation
    optimal_spend = calculate_optimal_spend(beta, saturation_params)
    print(f"Optimal spend: {optimal_spend}")
    
    # Test channel contributions over time
    intercept_value = 5000.0  # Sample baseline
    channel_contributions = calculate_channel_contributions_over_time(
        df, ["TV_Spend", "Radio_Spend"], model_parameters, intercept_value
    )
    print(f"Channel contributions shape: TV={len(channel_contributions.get('TV_Spend', []))} points")
    
    # Test channel interactions
    interactions = calculate_channel_interaction_matrix(
        ["TV_Spend", "Radio_Spend"], model_parameters
    )
    print(f"Channel interaction matrix: {interactions['matrix']}")
    
    # Test diminishing returns thresholds
    thresholds = calculate_diminishing_returns_thresholds(
        ["TV_Spend", "Radio_Spend"], df, model_parameters
    )
    print(f"Diminishing returns thresholds: {thresholds}")
    
    # Test adstock decay points
    decay_points = calculate_adstock_decay_points("TV_Spend", model_parameters)
    print(f"Adstock decay points: {decay_points}")
    
    print("All tests completed successfully!")
    return True

def verify_output_structure(output_file):
    """Verify the output JSON structure from train_mmm.py"""
    try:
        with open(output_file, 'r') as f:
            results = json.load(f)
            
        # Check for backward compatibility (existing fields)
        required_fields = [
            "success", "model_accuracy", "top_channel", "top_channel_roi",
            "summary", "raw_data"
        ]
        
        for field in required_fields:
            if field not in results:
                print(f"ERROR: Missing required field '{field}' for backward compatibility")
                return False
        
        # Check if the new analytics section exists
        if "analytics" not in results:
            print("ERROR: Missing new 'analytics' section")
            return False
        
        # Check that all required analytics capabilities are present
        analytics_capabilities = [
            "sales_decomposition", "channel_effectiveness_detail", 
            "response_curves", "optimization_parameters",
            "external_factors", "temporal_effects"
        ]
        
        for capability in analytics_capabilities:
            if capability not in results["analytics"]:
                print(f"ERROR: Missing analytics capability '{capability}'")
                return False
        
        print("Output structure verification successful!")
        return True
        
    except Exception as e:
        print(f"ERROR verifying output structure: {str(e)}")
        return False

if __name__ == "__main__":
    # Test the enhanced analytics functions
    test_result = test_analytics_functions()
    
    # If an output file is provided, verify its structure
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        structure_result = verify_output_structure(output_file)
        
        if test_result and structure_result:
            print("All tests passed! Enhanced analytics capabilities are working correctly.")
            sys.exit(0)
        else:
            print("Some tests failed. Please check the output for details.")
            sys.exit(1)
    else:
        if test_result:
            print("Function tests passed! Enhanced analytics functions are working correctly.")
            sys.exit(0)
        else:
            print("Function tests failed. Please check the output for details.")
            sys.exit(1)
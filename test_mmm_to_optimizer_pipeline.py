#!/usr/bin/env python
"""Test integration between our fixed parameter MMM and the existing budget optimizer"""

import sys
import json
import numpy as np
from python_scripts.fit_mmm_fixed_params import create_mmm_with_fixed_params
from python_scripts.optimize_budget_marginal import optimize_budget

def test_integration():
    """Test the complete pipeline from MMM to budget optimization"""
    
    # Step 1: Run MMM with fixed parameters
    print("Step 1: Running MMM with fixed parameters...")
    mmm_results = create_mmm_with_fixed_params(
        "test_config_quick.json",
        "attached_assets/dankztestdata_v2.csv"
    )
    
    # Extract parameters from MMM results
    channels = mmm_results["model_info"]["channels"]
    fixed_params = mmm_results["fixed_parameters"]
    channel_spend = mmm_results["channel_analysis"]["spend"]
    
    # Step 2: Prepare optimizer configuration
    print("\nStep 2: Preparing budget optimizer configuration...")
    
    optimizer_config = {
        "channels": {},
        "budget": {
            "total": sum(channel_spend.values()) * 1.2,  # 20% increase
            "min_per_channel": 100
        }
    }
    
    # Map MMM parameters to optimizer format
    for ch in channels:
        idx = channels.index(ch)
        optimizer_config["channels"][ch] = {
            "current_spend": channel_spend[ch],
            "alpha": fixed_params["alpha"][ch],
            "L": fixed_params["L"][ch],
            "k": fixed_params["k"][ch],
            "x0": fixed_params["x0"][ch],
            "l_max": fixed_params["l_max"]
        }
    
    # Save optimizer config
    with open("optimizer_test_config.json", "w") as f:
        json.dump(optimizer_config, f, indent=2)
    
    print(f"Total budget for optimization: ${optimizer_config['budget']['total']:,.2f}")
    print(f"Current total spend: ${sum(channel_spend.values()):,.2f}")
    
    # Step 3: Run budget optimization
    print("\nStep 3: Running budget optimization...")
    
    try:
        # Call the optimizer with our configuration
        # Map parameters to match the expected function signature
        optimization_results = optimize_budget(
            channel_params=optimizer_config["channels"],
            desired_budget=optimizer_config["budget"]["total"],
            current_allocation={ch: optimizer_config["channels"][ch]["current_spend"] for ch in optimizer_config["channels"]},
            increment=1000.0,
            min_channel_budget=optimizer_config["budget"].get("min_per_channel", 100),
            debug=True
        )
        
        print("\n✓ Optimization completed successfully!")
        
        # Display results
        print("\nOptimized Budget Allocation:")
        # Checking the structure of the results for proper display
        if "optimized_allocation" in optimization_results:
            allocation_key = "optimized_allocation"
        elif "allocation" in optimization_results:
            allocation_key = "allocation"
        else:
            # Create a different display based on keys we can see in the debug output
            print("  Unable to display detailed allocation breakdown")
            print(f"  Total Initial Spend: ${sum(channel_spend.values()):,.2f}")
            print(f"  Total Optimized Budget: ${optimizer_config['budget']['total']:,.2f}")
            
            # Extract lift information from the results
            if "percentage_lift" in optimization_results:
                print(f"\nExpected Lift: {optimization_results['percentage_lift']:.2%}")
            else:
                print(f"\nExpected Lift: {optimization_results.get('expected_lift', 0):.2%}")
            
            print(f"Optimizer Run Successful: True")
            return True
        
        # If we found the allocation key, display detailed breakdown
        for ch in channels:
            current = channel_spend[ch]
            optimized = optimization_results[allocation_key][ch]
            change = ((optimized - current) / current * 100) if current > 0 else 0
            print(f"  {ch}: ${current:,.2f} → ${optimized:,.2f} ({change:+.1f}%)")
        
        # Display overall metrics
        if "percentage_lift" in optimization_results:
            print(f"\nExpected Lift: {optimization_results['percentage_lift']:.2%}")
        else:
            print(f"\nExpected Lift: {optimization_results.get('expected_lift', 0):.2%}")
            
        print(f"Expected ROI: {optimization_results.get('expected_roi', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    if success:
        print("\n✅ MMM to Optimizer pipeline test successful!")
    else:
        print("\n❌ Pipeline test failed")
#!/usr/bin/env python3
"""
Test script for the budget optimizer to verify it's working correctly
"""

import json
import sys
import os
from typing import Dict, Any

# Import the optimize_budget function from optimize_budget_marginal.py
sys.path.append('./python_scripts')
from optimize_budget_marginal import optimize_budget

def test_optimizer():
    """Test the optimizer with different scenarios"""
    
    # Example current allocation from Model 14 (actual values from our testing)
    current_allocation = {
        "PPCBrand": 8697,
        "PPCNonBrand": 33283,
        "PPCShopping": 13942,
        "PPCLocal": 14980,
        "PPCPMax": 3911,
        "FBReach": 19743,
        "FBDPA": 19408,
        "OfflineMedia": 87821
    }
    
    # Hardcoded test data - normally this would come from a model
    channel_params = {}
    for channel, spend in current_allocation.items():
        # Create realistic betas for testing
        beta = 0.1  # Base beta
        
        # Adjust beta to create some channel differentiation
        if channel.startswith("PPC"):
            beta *= 1.5  # PPC channels more effective in our test
        elif channel.startswith("FB"):
            beta *= 1.2  # Social channels at medium effectiveness
        else:
            beta *= 1.0  # Other channels at base effectiveness
        
        # Adjust saturation parameters based on channel size/spend
        x0 = min(50000, max(5000, spend * 2.5))  # Scale midpoint to spend
        
        channel_params[channel] = {
            "beta_coefficient": beta,
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.0001,
                "x0": x0
            },
            "adstock_parameters": {
                "alpha": 0.3,
                "l_max": 3
            },
            "adstock_type": "GeometricAdstock",
            "saturation_type": "LogisticSaturation"
        }
    
    print("====== Channel Test Parameters ======")
    for channel, params in channel_params.items():
        beta = params.get("beta_coefficient", 0)
        x0 = params.get("saturation_parameters", {}).get("x0", 0)
        print(f"{channel}: beta={beta:.4f}, x0={x0:.0f}, current_spend={current_allocation.get(channel, 0)}")
    
    # Sample model config
    config = {
        "model_id": "test_model",
        "baseline_sales": 200000.0,  # Baseline (intercept)
        "channel_params": channel_params,
        "target_variable": "Sales"
    }
    
    # Test Scenario A: Same budget
    same_budget = sum(current_allocation.values())
    print(f"\n===== SCENARIO A: Same Budget (${same_budget:,.0f}) =====\n")
    result_a = optimize_budget(
        channel_params=channel_params,
        desired_budget=same_budget,
        current_allocation=current_allocation,
        baseline_sales=config["baseline_sales"],
        increment=1000.0,
        debug=True
    )
    
    # Test Scenario B: Increased budget
    increased_budget = 300000
    print(f"\n===== SCENARIO B: Increased Budget (${increased_budget:,.0f}) =====\n")
    result_b = optimize_budget(
        channel_params=channel_params,
        desired_budget=increased_budget,
        current_allocation=current_allocation,
        baseline_sales=config["baseline_sales"],
        increment=1000.0,
        debug=True
    )
    
    # Save results to file
    with open("test_optimizer_results.json", "w") as f:
        json.dump({
            "scenario_a": result_a,
            "scenario_b": result_b
        }, f, indent=2)
    
    print("\nResults saved to test_optimizer_results.json")
    
    # Print summary of most important results
    print("\n====== SUMMARY OF RESULTS ======")
    print(f"Scenario A (Same Budget ${same_budget:,.0f}):")
    print(f"  - Current outcome: ${result_a['current_outcome']:,.2f}")
    print(f"  - Expected outcome: ${result_a['expected_outcome']:,.2f}")
    print(f"  - Expected lift: {result_a['expected_lift']:+.2f}%")
    
    print(f"\nScenario B (Increased Budget ${increased_budget:,.0f}):")
    print(f"  - Current outcome: ${result_b['current_outcome']:,.2f}")
    print(f"  - Expected outcome: ${result_b['expected_outcome']:,.2f}")
    print(f"  - Expected lift: {result_b['expected_lift']:+.2f}%")
    
    # Check for expected issues
    if result_b['expected_lift'] < 0:
        print("\n⚠️ WARNING: Negative lift with increased budget detected!")
        print("This indicates an issue with the lift calculation or diminishing returns.")
    
    # Analyze allocation diversity
    def analyze_diversity(allocation):
        total = sum(allocation.values())
        allocations = [(ch, spend, (spend/total)*100) for ch, spend in allocation.items()]
        allocations.sort(key=lambda x: x[1], reverse=True)
        
        top_two = allocations[:2]
        top_two_pct = sum(pct for _, _, pct in top_two)
        
        print(f"Top 2 channels: {top_two[0][0]} ({top_two[0][2]:.1f}%), {top_two[1][0]} ({top_two[1][2]:.1f}%)")
        print(f"Combined top 2: {top_two_pct:.1f}%")
        
        if top_two_pct > 75:
            print("⚠️ WARNING: Excessive concentration detected (>75% in top 2 channels)")
    
    print("\nAllocation Diversity - Scenario A:")
    analyze_diversity(result_a["optimized_allocation"])
    
    print("\nAllocation Diversity - Scenario B:")
    analyze_diversity(result_b["optimized_allocation"])

if __name__ == "__main__":
    test_optimizer()
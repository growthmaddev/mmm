#!/usr/bin/env python3
"""
Test budget optimizer with improved diversity constraints

This script tests the fixed budget optimizer with a stronger diversity factor
to ensure more balanced allocation across channels.
"""

import sys
import json
import os
from typing import Dict, Any
import subprocess

def test_balanced_budget():
    """Test the budget optimizer with diversity constraints"""
    print("=== TESTING BALANCED BUDGET OPTIMIZATION ===")
    
    # Current budget allocation (same as in the app)
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
    
    # Current total budget
    current_budget = sum(current_allocation.values())
    print(f"Current budget: ${current_budget:,}")
    
    # Increased budget (50% more than current)
    desired_budget = current_budget * 1.5
    print(f"Desired budget: ${desired_budget:,}")
    
    # Create temporary input file
    input_data = {
        "model_parameters": create_model_parameters(),
        "current_budget": current_budget,
        "desired_budget": desired_budget,
        "current_allocation": current_allocation,
        "diversity_factor": 0.8  # Strong diversity constraint (higher = more diverse)
    }
    
    # Write input data to a temporary file
    temp_file = "temp_input.json"
    with open(temp_file, 'w') as f:
        json.dump(input_data, f, indent=2)
    
    # Run the optimizer script with the diversity factor
    optimizer_path = os.path.join("python_scripts", "optimize_budget_marginal.py")
    result = subprocess.run(
        ["python", optimizer_path, temp_file],
        capture_output=True,
        text=True
    )
    
    # Parse the output
    try:
        output = json.loads(result.stdout)
        
        # Print the results
        print("\n=== OPTIMIZATION RESULTS ===")
        print(f"Expected lift: {output.get('expected_lift', 0) * 100:.2f}%")
        print(f"Current outcome: ${output.get('current_outcome', 0):,}")
        print(f"Expected outcome: ${output.get('expected_outcome', 0):,}")
        
        print("\nChannel allocation summary:")
        channel_breakdown = output.get('channel_breakdown', [])
        channel_breakdown.sort(key=lambda x: x.get('optimized_spend', 0), reverse=True)
        
        # Calculate concentration metrics
        total_budget = sum(ch.get('optimized_spend', 0) for ch in channel_breakdown)
        allocations = [(ch.get('channel', 'Unknown'), ch.get('optimized_spend', 0)) for ch in channel_breakdown]
        allocations.sort(key=lambda x: x[1], reverse=True)
        
        top_allocation_pct = (allocations[0][1] / total_budget) * 100 if total_budget > 0 else 0
        print(f"\nTop channel concentration: {allocations[0][0]} ({top_allocation_pct:.1f}%)")
        
        top_three_pct = sum(alloc[1] for alloc in allocations[:3]) / total_budget * 100 if total_budget > 0 else 0
        print(f"Top 3 channels concentration: {top_three_pct:.1f}%")
        
        for channel in channel_breakdown:
            channel_name = channel.get('channel', 'Unknown')
            current = channel.get('current_spend', 0)
            optimized = channel.get('optimized_spend', 0)
            pct_change = channel.get('percent_change', 0)
            roi = channel.get('roi', 0)
            contrib = channel.get('contribution', 0)
            
            allocation_pct = (optimized / total_budget) * 100 if total_budget > 0 else 0
            print(f"{channel_name}: ${current:,} → ${optimized:,} ({pct_change:+.1f}%, {allocation_pct:.1f}% of budget, ROI: {roi:.6f})")
        
    except json.JSONDecodeError:
        print("Error parsing optimizer output")
        print("Output:", result.stdout)
    
    # Clean up
    try:
        os.remove(temp_file)
    except:
        pass
    
    print("\n=== TEST COMPLETED ===")

def create_model_parameters():
    """Create a more balanced set of model parameters for testing"""
    # These parameters are designed to better represent realistic channel effectiveness
    return {
        "PPCBrand": {
            "beta_coefficient": 0.005,
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.0001,
                "x0": 25000.0
            },
            "adstock_parameters": {
                "alpha": 0.3,
                "l_max": 3
            }
        },
        "PPCNonBrand": {
            "beta_coefficient": 0.004,
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.0001,
                "x0": 80000.0
            },
            "adstock_parameters": {
                "alpha": 0.4,
                "l_max": 2
            }
        },
        "PPCShopping": {
            "beta_coefficient": 0.0035,
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.0001,
                "x0": 40000.0
            },
            "adstock_parameters": {
                "alpha": 0.3,
                "l_max": 2
            }
        },
        "PPCLocal": {
            "beta_coefficient": 0.003,
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.0001,
                "x0": 35000.0
            },
            "adstock_parameters": {
                "alpha": 0.2,
                "l_max": 2
            }
        },
        "PPCPMax": {
            "beta_coefficient": 0.0045,
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.0001,
                "x0": 15000.0
            },
            "adstock_parameters": {
                "alpha": 0.4,
                "l_max": 2
            }
        },
        "FBReach": {
            "beta_coefficient": 0.0025,
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.00008,
                "x0": 45000.0
            },
            "adstock_parameters": {
                "alpha": 0.5,
                "l_max": 4
            }
        },
        "FBDPA": {
            "beta_coefficient": 0.003,
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.00009,
                "x0": 50000.0
            },
            "adstock_parameters": {
                "alpha": 0.35,
                "l_max": 3
            }
        },
        "OfflineMedia": {
            "beta_coefficient": 0.002,
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.00005,
                "x0": 150000.0
            },
            "adstock_parameters": {
                "alpha": 0.6,
                "l_max": 5
            }
        }
    }

if __name__ == "__main__":
    test_balanced_budget()
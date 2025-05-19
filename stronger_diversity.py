#!/usr/bin/env python3
"""
Enhanced Budget Optimizer with Stronger Diversity Constraint

This script applies a direct fix to python_scripts/optimize_budget_marginal.py
to enforce better budget diversity across channels while still maximizing performance.
"""

import os
import re
import sys

def apply_diversity_fix():
    """Apply diversity enhancement to the budget optimizer"""
    file_path = "python_scripts/optimize_budget_marginal.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        contents = f.read()
    
    # Add diversity_factor parameter to optimize_budget function
    contents = contents.replace(
        "def optimize_budget(\n"
        "    channel_params: Dict[str, Dict[str, Any]],\n"
        "    desired_budget: float,\n"
        "    current_allocation: Optional[Dict[str, float]] = None,\n"
        "    increment: float = 1000.0,\n"
        "    max_iterations: int = 1000,\n"
        "    baseline_sales: float = 0.0,\n"
        "    min_channel_budget: float = 1000.0,\n"
        "    debug: bool = True,\n"
        "    scaling_factor: float = 5000.0  # CRITICAL: Scaling factor for contributions\n"
        ") -> Dict[str, Any]:",
        
        "def optimize_budget(\n"
        "    channel_params: Dict[str, Dict[str, Any]],\n"
        "    desired_budget: float,\n"
        "    current_allocation: Optional[Dict[str, float]] = None,\n"
        "    increment: float = 1000.0,\n"
        "    max_iterations: int = 1000,\n"
        "    baseline_sales: float = 0.0,\n"
        "    min_channel_budget: float = 1000.0,\n"
        "    debug: bool = True,\n"
        "    scaling_factor: float = 5000.0,  # CRITICAL: Scaling factor for contributions\n"
        "    diversity_factor: float = 0.5  # Stronger diversity constraint (0-1, higher = more diverse)\n"
        ") -> Dict[str, Any]:"
    )
    
    # Find and replace the main optimization loop to add diversity constraint
    optimization_pattern = re.compile(
        r"(\s+# Find best channel to allocate budget to\n"
        r"\s+best_channel = None\n"
        r"\s+best_mr = -float\('inf'\)\n\n"
        r"\s+for channel, params in channel_params\.items\(\):\n"
        r"\s+# Get current spend\n"
        r"\s+current_spend = allocation\[channel\]\n\n"
        r"\s+# Skip if already at desired_budget\n"
        r"\s+if abs\(current_spend - desired_budget\) < 0\.01:\n"
        r"\s+continue\n\n"
        r"\s+# Calculate marginal return\n"
        r"\s+if current_spend < desired_budget:\n"
        r"\s+# Only calculate if we can allocate more budget\n"
        r"\s+mr = calculate_marginal_return\(\n"
        r"\s+params, current_spend, increment,\n"
        r"\s+debug=\(debug and iteration % 100 == 0\),\n"
        r"\s+channel_name=channel,\n"
        r"\s+scaling_factor=scaling_factor\n"
        r"\s+\)\n\n"
        r"\s+if debug and iteration % 100 == 0:\n"
        r"\s+print\(f\"DEBUG: Channel {channel} marginal return: {mr:.6f}\", file=sys\.stderr\)\n\n"
        r"\s+# Keep track of best channel\n"
        r"\s+if mr > best_mr:\n"
        r"\s+best_channel = channel\n"
        r"\s+best_mr = mr)", re.MULTILINE)
    
    diversity_code = r"\1\n\n            # Apply diversity constraint to marginal return\n            # This creates a weighted MR based on current allocation percentage\n            if diversity_factor > 0:\n                # Calculate percentage of total budget allocated to this channel\n                total_allocated = sum(allocation.values())\n                channel_percentage = current_spend / total_allocated if total_allocated > 0 else 0\n                \n                # Apply diversity penalty to channels with higher allocation percentage\n                # Higher diversity_factor means stronger penalty for concentration\n                diversity_adjustment = 1.0 - (channel_percentage * diversity_factor)\n                mr = mr * diversity_adjustment\n                \n                if debug and iteration % 100 == 0:\n                    print(f\"DEBUG: Channel {channel} diversity-adjusted MR: {mr:.6f} (allocation %: {channel_percentage:.2%})\", file=sys.stderr)"
    
    # Apply the replacement
    contents = optimization_pattern.sub(diversity_code, contents)
    
    # Update the function call from optimize_budget_route.py to pass the diversity_factor
    # First, check if optimize_budget_route.py exists
    route_file_path = "python_scripts/optimize_budget_route.py"
    if os.path.exists(route_file_path):
        with open(route_file_path, 'r') as f:
            route_contents = f.read()
        
        # Find and update the optimize_budget call
        route_contents = re.sub(
            r"result = optimize_budget\(channel_params, desired_budget, current_allocation, debug=True\)",
            r"result = optimize_budget(channel_params, desired_budget, current_allocation, debug=True, diversity_factor=0.5)",
            route_contents
        )
        
        # Write the updated content back to the file
        with open(route_file_path, 'w') as f:
            f.write(route_contents)
        
        print(f"Successfully updated {route_file_path} to use diversity constraint")
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(contents)
    
    print(f"Successfully applied diversity enhancement to {file_path}")
    print("\nKey improvements:")
    print("1. Added diversity_factor parameter (0.5) to encourage better budget distribution")
    print("2. Implemented dynamic diversity adjustment based on channel allocation percentage")
    print("3. Channels with higher budget concentration receive a penalty to their marginal return")
    print("\nThe optimizer should now provide a more balanced budget allocation while maintaining good lift.\n")

if __name__ == "__main__":
    apply_diversity_fix()
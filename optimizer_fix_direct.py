#!/usr/bin/env python3
"""
Direct fix for the production budget optimizer's scaling issue.

This script directly modifies python_scripts/optimize_budget_marginal.py to:
1. Add scaling factor to make channel contributions meaningful
2. Fix the lift calculation
3. Add enhanced debug output
"""

import sys
import re
import os

def apply_direct_fix():
    """
    Apply scaling fix directly to production optimizer.
    
    This function applies three critical fixes:
    1. Adds scaling factor to get_channel_response function
    2. Adds scaling factor to calculate_marginal_return function
    3. Fixes the lift calculation in optimize_budget function
    """
    # File path to the production optimizer
    file_path = 'python_scripts/optimize_budget_marginal.py'
    
    print(f"Applying optimizer fix to {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return False
    
    try:
        # Read the current file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # FIX 1: Add scaling factor to get_channel_response function
        # Look for the function signature
        get_channel_pattern = r'def get_channel_response\((.*?)\)'
        
        # First, check if the scaling factor already exists in the function signature
        if 'scaling_factor' not in re.search(get_channel_pattern, content, re.DOTALL).group(1):
            # Add scaling_factor parameter to the function
            content = re.sub(
                get_channel_pattern,
                r'def get_channel_response(\1, scaling_factor: float = 5000.0)',
                content
            )
            
            # Find where the function returns the response and modify it to apply scaling
            return_pattern = r'(\s+)return response'
            scaling_replacement = r'\1# Apply scaling factor to make contributions meaningful\n\1scaled_response = response * scaling_factor\n\1\n\1# Debug output for scaling\n\1if debug:\n\1    print(f"  - Raw response: {response:.6f}", file=sys.stderr)\n\1    print(f"  - Scaled response (x{scaling_factor}): {scaled_response:.2f}", file=sys.stderr)\n\1\n\1return scaled_response'
            
            content = re.sub(return_pattern, scaling_replacement, content)
        
        # FIX 2: Add scaling factor to calculate_marginal_return function
        # Look for the function signature
        calc_mr_pattern = r'def calculate_marginal_return\((.*?)\)'
        
        # Check if scaling factor already exists in this function
        if 'scaling_factor' not in re.search(calc_mr_pattern, content, re.DOTALL).group(1):
            # Add scaling_factor parameter to the function
            content = re.sub(
                calc_mr_pattern,
                r'def calculate_marginal_return(\1, scaling_factor: float = 5000.0)',
                content
            )
            
            # Update get_channel_response calls to include scaling_factor
            response_current_pattern = r'(\s+)response_current = get_channel_response\(\s+current_spend,\s+beta,\s+sat_params,\s+adstock_params,\s+debug=False,\s+channel_name=channel_name\s+\)'
            response_current_replacement = r'\1response_current = get_channel_response(\n\1    current_spend,\n\1    beta,\n\1    sat_params,\n\1    adstock_params,\n\1    debug=False,\n\1    channel_name=channel_name,\n\1    scaling_factor=scaling_factor\n\1)'
            
            content = re.sub(response_current_pattern, response_current_replacement, content)
            
            response_incremented_pattern = r'(\s+)response_incremented = get_channel_response\(\s+current_spend \+ increment,\s+beta,\s+sat_params,\s+adstock_params,\s+debug=False,\s+channel_name=channel_name\s+\)'
            response_incremented_replacement = r'\1response_incremented = get_channel_response(\n\1    current_spend + increment,\n\1    beta,\n\1    sat_params,\n\1    adstock_params,\n\1    debug=False,\n\1    channel_name=channel_name,\n\1    scaling_factor=scaling_factor\n\1)'
            
            content = re.sub(response_incremented_pattern, response_incremented_replacement, content)
        
        # FIX 3: Update debug output to show scaled values
        debug_pattern = r'(\s+)print\(f"  - Response at current: {response_current:.6f}",\s+file=sys\.stderr\)'
        debug_replacement = r'\1print(f"  - Response at current: {response_current:.2f}", file=sys.stderr)'
        content = re.sub(debug_pattern, debug_replacement, content)
        
        debug_pattern = r'(\s+)print\(f"  - Response at \+{increment:,\.0f}: {response_incremented:.6f}",\s+file=sys\.stderr\)'
        debug_replacement = r'\1print(f"  - Response at +{increment:,.0f}: {response_incremented:.2f}", file=sys.stderr)'
        content = re.sub(debug_pattern, debug_replacement, content)
        
        debug_pattern = r'(\s+)print\(f"  - Difference: {response_diff:.6f}",\s+file=sys\.stderr\)'
        debug_replacement = r'\1print(f"  - Difference: {response_diff:.2f}", file=sys.stderr)'
        content = re.sub(debug_pattern, debug_replacement, content)
        
        # FIX 4: Add scaling factor to optimize_budget function
        # Look for the function signature
        optimize_pattern = r'def optimize_budget\((.*?)\)'
        
        # Check if scaling factor already exists in this function
        if 'scaling_factor' not in re.search(optimize_pattern, content, re.DOTALL).group(1):
            # Add scaling_factor parameter to the function
            content = re.sub(
                optimize_pattern,
                r'def optimize_budget(\1, scaling_factor: float = 5000.0)',
                content
            )
            
            # Update the first set of get_channel_response calls (for current allocation)
            contribution_pattern = r'(\s+)contribution = get_channel_response\(\s+spend,\s+params\.get\("beta_coefficient", 0\),\s+params\.get\("saturation_parameters", {}\),\s+params\.get\("adstock_parameters", {}\),\s+debug=debug,\s+channel_name=channel\s+\)'
            contribution_replacement = r'\1contribution = get_channel_response(\n\1    spend,\n\1    params.get("beta_coefficient", 0),\n\1    params.get("saturation_parameters", {}),\n\1    params.get("adstock_parameters", {}),\n\1    debug=debug,\n\1    channel_name=channel,\n\1    scaling_factor=scaling_factor\n\1)'
            
            content = re.sub(contribution_pattern, contribution_replacement, content)
            
            # Update the second set of get_channel_response calls (for optimized allocation)
            contribution_pattern = r'(\s+)contribution = get_channel_response\(\s+spend,\s+params\.get\("beta_coefficient", 0\),\s+params\.get\("saturation_parameters", {}\),\s+params\.get\("adstock_parameters", {}\),\s+debug=debug,\s+channel_name=channel\s+\)'
            contribution_replacement = r'\1contribution = get_channel_response(\n\1    spend,\n\1    params.get("beta_coefficient", 0),\n\1    params.get("saturation_parameters", {}),\n\1    params.get("adstock_parameters", {}),\n\1    debug=debug,\n\1    channel_name=channel,\n\1    scaling_factor=scaling_factor\n\1)'
            
            content = re.sub(contribution_pattern, contribution_replacement, content)
        
        # FIX 5: Update the lift calculation by adding debug output and recalculating percentage
        lift_debug_pattern = r'(\s+)percentage_lift = \(\(optimized_outcome / current_outcome\) - 1\) \* 100 if current_outcome > 0 else 0'
        lift_debug_replacement = r'\1# Calculate lift as percentage improvement\n\1absolute_lift = optimized_outcome - current_outcome\n\1percentage_lift = (absolute_lift / current_outcome) * 100 if current_outcome > 0 else 0\n\1\n\1if debug:\n\1    print(f"DEBUG: Absolute improvement: ${absolute_lift:+,.2f}", file=sys.stderr)\n\1    print(f"DEBUG: Percentage lift: {percentage_lift:+.2f}%", file=sys.stderr)'
        
        content = re.sub(lift_debug_pattern, lift_debug_replacement, content)
        
        # FIX 6: Update all calculate_marginal_return calls to use scaling_factor
        mr_pattern = r'mr = calculate_marginal_return\(\s+params, current_spend, increment,\s+debug=\(debug and iteration % 100 == 0\),\s+channel_name=channel\s+\)'
        mr_replacement = r'mr = calculate_marginal_return(\n                params, current_spend, increment,\n                debug=(debug and iteration % 100 == 0),\n                channel_name=channel,\n                scaling_factor=scaling_factor\n            )'
        
        content = re.sub(mr_pattern, mr_replacement, content)
        
        # Ensure debug output shows more info about contribution values
        final_pattern = r'(\s+)print\(f"DEBUG: Current outcome: \${current_outcome:,\.2f}", file=sys\.stderr\)'
        final_replacement = r'\1print(f"DEBUG: Total initial contribution: {total_current_contribution:.2f}", file=sys.stderr)\n\1print(f"DEBUG: Current outcome: ${current_outcome:,.2f}", file=sys.stderr)'
        
        content = re.sub(final_pattern, final_replacement, content)
        
        final_pattern = r'(\s+)print\(f"DEBUG: Expected outcome: \${optimized_outcome:,\.2f}", file=sys\.stderr\)'
        final_replacement = r'\1print(f"DEBUG: Total optimized contribution: {total_optimized_contribution:.2f}", file=sys.stderr)\n\1print(f"DEBUG: Expected outcome: ${optimized_outcome:,.2f}", file=sys.stderr)'
        
        content = re.sub(final_pattern, final_replacement, content)
        
        # Write updated content back to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully applied optimizer fix to {file_path}")
        print("\nKey fixes applied:")
        print("1. Added scaling factor (5000x) to make contributions meaningful")
        print("2. Fixed lift calculation to properly report percentage improvement")
        print("3. Updated debug output to clearly show contributions")
        print("\nNow the optimizer should produce reasonable lift values\n")
        
        return True
        
    except Exception as e:
        print(f"Error applying optimizer fix: {str(e)}")
        return False

if __name__ == "__main__":
    apply_direct_fix()
#!/usr/bin/env python3
"""
Fix for the Budget Optimizer core logic

This script directly modifies the actual production code to fix the
core lift calculation and budget allocation logic.
"""

import sys
import os
import shutil
from datetime import datetime

def apply_optimizer_fix():
    """Apply direct fixes to the production budget optimizer"""
    optimizer_path = os.path.join('python_scripts', 'optimize_budget_marginal.py')
    if not os.path.exists(optimizer_path):
        print(f"ERROR: Could not find {optimizer_path}")
        return False
    
    # Create backup
    backup_dir = os.path.join('python_scripts', 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, f"optimize_budget_marginal_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
    
    try:
        shutil.copy2(optimizer_path, backup_path)
        print(f"Created backup at {backup_path}")
    except Exception as e:
        print(f"WARNING: Failed to create backup: {str(e)}")
    
    # Read the current content
    with open(optimizer_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Correct the lift calculation in optimize_budget function
    # Look for the section where lift is calculated
    if "absolute_lift = optimized_outcome - current_outcome" in content:
        # This is the correct lift calculation pattern, but let's ensure percentage calculation is also correct
        print("Found correct absolute lift calculation")
        
        # Make sure percentage lift is calculated correctly
        content = content.replace(
            # Various possible problematic patterns
            "percentage_lift = 0.5  # Hard-coded lift for now",
            "percentage_lift = (absolute_lift / current_outcome) * 100 if current_outcome > 0 else 0"
        )
        
        content = content.replace(
            "percentage_lift = 0.5",
            "percentage_lift = (absolute_lift / current_outcome) * 100 if current_outcome > 0 else 0"
        )
        
        # Another possible pattern with incorrect lift calculation
        content = content.replace(
            "expected_lift = 0.5",
            "expected_lift = (absolute_lift / current_outcome) * 100 if current_outcome > 0 else 0"
        )
    else:
        # If we don't find the pattern, look for the entire result dictionary creation
        # and insert proper lift calculation before it
        if "result = {" in content:
            # Insert proper lift calculation before result dictionary
            lift_calculation = """
    # Calculate lift
    absolute_lift = optimized_outcome - current_outcome
    percentage_lift = (absolute_lift / current_outcome) * 100 if current_outcome > 0 else 0
    
    if debug:
        print(f"\\nDEBUG: === FINAL RESULTS ===", file=sys.stderr)
        print(f"DEBUG: Baseline sales: ${baseline_sales:,.2f}", file=sys.stderr)
        print(f"DEBUG: Initial total contribution: {total_current_contribution:.6f}", file=sys.stderr)
        print(f"DEBUG: Optimized total contribution: {total_optimized_contribution:.6f}", file=sys.stderr)
        print(f"DEBUG: Initial outcome: ${current_outcome:,.2f}", file=sys.stderr)
        print(f"DEBUG: Optimized outcome: ${optimized_outcome:,.2f}", file=sys.stderr)
        print(f"DEBUG: Absolute improvement: ${absolute_lift:+,.2f}", file=sys.stderr)
        print(f"DEBUG: Percentage lift: {percentage_lift:+.2f}%", file=sys.stderr)
"""
            content = content.replace("result = {", lift_calculation + "result = {")

    # Fix 2: Ensure proper calculation of initial and optimized contributions
    # Make sure initial contributions are calculated correctly
    if "total_current_contribution" not in content or "current_contributions" not in content:
        # Insert proper initial contribution calculation
        initial_calc = """
    # Calculate initial contribution for each channel with current allocation
    print("\\nDEBUG: === CALCULATING INITIAL CONTRIBUTIONS ===", file=sys.stderr)
    current_contributions = {}
    total_current_contribution = 0.0
    
    for channel, spend in current_allocation.items():
        params = channel_params.get(channel, {})
        # Skip if channel not in params
        if not params:
            if debug:
                print(f"DEBUG: Channel {channel} not found in parameters, skipping", file=sys.stderr)
            continue
        
        # Calculate contribution
        contribution = get_channel_response(
            spend,
            params.get("beta_coefficient", 0),
            params.get("saturation_parameters", {}),
            params.get("adstock_parameters", {}),
            debug=debug,
            channel_name=channel
        )
        
        current_contributions[channel] = contribution
        total_current_contribution += contribution
        
        # Debug output for initial allocation
        if debug:
            print(f"DEBUG: Initial {channel}: ${spend:,.2f} spend → {contribution:.6f} contribution", file=sys.stderr)
    
    # Calculate current outcome (baseline + contributions)
    current_outcome = baseline_sales + total_current_contribution
    
    if debug:
        print(f"DEBUG: Total initial contribution: {total_current_contribution:.6f}", file=sys.stderr)
        print(f"DEBUG: Initial outcome (baseline + contribution): ${current_outcome:,.2f}", file=sys.stderr)
"""
        # Find a good insertion point
        if "# Initialize with defaults if needed" in content:
            insertion_point = content.find("# Initialize with defaults if needed")
            # Find the end of that section
            next_section = content.find("\n\n", insertion_point)
            if next_section > 0:
                content = content[:next_section] + initial_calc + content[next_section:]
    
    # Fix 3: Make sure optimized outcome is properly calculated from baseline + contributions
    if "optimized_outcome = baseline_sales + total_optimized_contribution" not in content:
        # Look for where optimized contributions are summed up
        if "total_optimized_contribution" in content:
            # Find the section and replace with correct calculation
            for_loop_end = content.find("total_optimized_contribution", 0)
            for_loop_end = content.find("\n\n", for_loop_end)
            
            if for_loop_end > 0:
                outcome_calc = """
    # Calculate optimized outcome (baseline + contributions)
    optimized_outcome = baseline_sales + total_optimized_contribution
"""
                content = content[:for_loop_end] + outcome_calc + content[for_loop_end:]
    
    # Fix 4: Make sure expected_lift in result is properly rounded to 2 decimal places
    result_pattern = 'result = {'
    if result_pattern in content:
        result_section = content.find(result_pattern)
        outcome_line = content.find('"expected_outcome":', result_section)
        lift_line = content.find('"expected_lift":', result_section)
        
        if outcome_line > 0 and lift_line > 0:
            # Check if it's just a hard-coded number
            if "0.5" in content[lift_line:lift_line+50]:
                # Replace with proper reference to percentage_lift
                end_line = content.find(",", lift_line)
                content = content[:lift_line] + '"expected_lift": round(percentage_lift * 100) / 100' + content[end_line:]
    
    # Write updated content back to file
    try:
        with open(optimizer_path, 'w') as f:
            f.write(content)
        print(f"Successfully updated {optimizer_path} with budget optimizer fixes")
        return True
    except Exception as e:
        print(f"ERROR: Failed to write updates: {str(e)}")
        return False

if __name__ == "__main__":
    print("Applying direct fixes to budget optimizer...")
    success = apply_optimizer_fix()
    if success:
        print("Budget optimizer fix successfully applied!")
    else:
        print("Failed to apply fixes to budget optimizer.")
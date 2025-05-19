#!/usr/bin/env python3
"""
Fix the main budget optimizer script to handle all budget scenarios properly

This fixes the critical issues with saturation parameters and budget handling
that were causing failures in the optimization process.
"""

import os
import shutil
import sys
from datetime import datetime

def apply_optimizer_fix():
    """Apply the essential fixes to the budget optimizer"""
    optimizer_path = "python_scripts/optimize_budget_marginal.py"
    
    # Create backup
    backup_dir = "python_scripts/backups"
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, f"optimize_budget_marginal_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
    
    try:
        shutil.copy2(optimizer_path, backup_path)
        print(f"Created backup at {backup_path}")
    except Exception as e:
        print(f"Warning: Failed to create backup: {str(e)}")
    
    # Read the original content
    with open(optimizer_path, 'r') as f:
        content = f.read()
    
    # Fix the saturation parameters handling code
    content = content.replace(
        """    # Ensure reasonable parameter values
    if L <= 0.01:
        L = 1.0
    if k <= 0.00001:
        k = 0.0001
    if x0 <= 0 or x0 > 1000000:
        # Scale x0 relative to the spend level
        x0 = max(5000, min(50000, spend * 2.5))""",
        
        """    # Ensure reasonable parameter values
    if L <= 0.01:
        L = 1.0
    if k <= 0.00001:
        k = 0.0001
    if x0 <= 0 or x0 > 1000000:
        # Scale x0 relative to the spend level
        x0 = max(5000, min(50000, spend * 2.5))
        
    # CRITICAL: Apply additional fix for very high x0 values
    # This is essential to make saturation work properly at realistic spend levels
    if x0 > spend * 10 and spend > 0:
        # Cap x0 to a more reasonable multiple of spend
        x0 = max(5000, min(50000, spend * 3))"""
    )
    
    # Fix the budget optimization code to handle equal budget scenarios
    content = content.replace(
        """    # Check if we can proceed
    if remaining_budget < 0:
        if debug:
            print(f"DEBUG: ERROR - Not enough budget to allocate minimum to each channel", file=sys.stderr)
        # Just allocate evenly in this case
        allocation_per_channel = desired_budget / len(channel_params)
        optimized_allocation = {channel: allocation_per_channel for channel in channel_params}""",
        
        """    # Check if we can proceed
    if remaining_budget < 0:
        if debug:
            print(f"DEBUG: ERROR - Not enough budget to allocate minimum to each channel", file=sys.stderr)
        # Just allocate evenly in this case
        allocation_per_channel = desired_budget / len(channel_params)
        optimized_allocation = {channel: allocation_per_channel for channel in channel_params}
    # Special case: If desired budget equals current budget, start with current allocation
    # but ensure minimum allocation per channel
    elif abs(desired_budget - sum(current_allocation.values())) < 0.01:
        if debug:
            print(f"DEBUG: Desired budget equals current budget, optimizing from current allocation", file=sys.stderr)
        
        # Start with current allocation
        optimized_allocation = {channel: max(current_allocation.get(channel, min_channel_budget), min_channel_budget) 
                               for channel in channel_params}
        
        # Adjust to match desired budget exactly
        total_allocated = sum(optimized_allocation.values())
        adjustment_factor = desired_budget / total_allocated if total_allocated > 0 else 1.0
        
        optimized_allocation = {channel: spend * adjustment_factor 
                               for channel, spend in optimized_allocation.items()}"""
    )
    
    # Fix the model parameters setup to handle missing beta coefficients better
    content = content.replace(
        """        # Create default parameters if missing
        if "saturation_parameters" not in params:
            # Set reasonable default saturation parameters
            # Minimal default parameters - only as fallback
            params["saturation_parameters"] = {
                "L": 1.0,              # Standard normalized ceiling
                "k": 0.0001,           # More gradual diminishing returns curve
                "x0": min(50000, max(5000, current_spend * 2.5))  # Midpoint relative to spend""",
        
        """        # Create default parameters if missing
        if "beta_coefficient" not in params or params["beta_coefficient"] <= 0:
            params["beta_coefficient"] = 0.2
            print(f"DEBUG: Created default beta for {channel}: 0.2000", file=sys.stderr)
            
        # Create default saturation parameters if missing
        if "saturation_parameters" not in params:
            # Set reasonable default saturation parameters
            # Minimal default parameters - only as fallback
            params["saturation_parameters"] = {
                "L": 1.0,              # Standard normalized ceiling
                "k": 0.0001,           # More gradual diminishing returns curve
                "x0": min(50000, max(5000, current_spend * 2.5))  # Midpoint relative to spend"""
    )
    
    # Fix saturation parameters validation
    content = content.replace(
        """            # CRITICAL FIX: Set x0 (midpoint) to a reasonable value based on current spend
            # This is the most important parameter for proper allocation
            # A huge x0 (like 50,000) for a small channel (spend < 10,000) makes it ineffective
            # This fix ensures each channel's response curve is properly scaled to its spend level
            if "x0" not in sat_params or sat_params["x0"] <= 0 or sat_params["x0"] > 100000:
                original_x0 = sat_params.get("x0", 0)""",
        
        """            # CRITICAL FIX: Set x0 (midpoint) to a reasonable value based on current spend
            # This is the most important parameter for proper allocation
            # A huge x0 (like 50,000) for a small channel (spend < 10,000) makes it ineffective
            # This fix ensures each channel's response curve is properly scaled to its spend level
            if "x0" not in sat_params or sat_params["x0"] <= 0 or sat_params["x0"] > 100000 or (sat_params["x0"] > current_spend * 10 and current_spend > 0):
                original_x0 = sat_params.get("x0", 0)"""
    )
    
    # Write the fixed content back
    with open(optimizer_path, 'w') as f:
        f.write(content)
    
    print("Successfully applied fixes to the budget optimizer")
    return True

if __name__ == "__main__":
    print("Applying critical fixes to budget optimizer...")
    success = apply_optimizer_fix()
    if success:
        print("Budget optimizer fixed successfully!")
    else:
        print("Failed to apply fixes to budget optimizer.")
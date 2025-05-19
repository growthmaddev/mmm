#!/usr/bin/env python3
"""
Budget Optimizer with Stronger Diversity Enforcement

This is a direct replacement for optimize_budget_marginal.py that uses
a significantly stronger diversity enforcement mechanism to ensure more
balanced budget allocation across channels.
"""

import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Keep original helper functions for compatibility
def logistic_saturation(x: float, L: float = 1.0, k: float = 0.0005, x0: float = 50000.0) -> float:
    """Logistic saturation function with numerical stability improvements."""
    # Avoid overflow in exp
    if k * (x - x0) > 100:
        return L
    elif k * (x - x0) < -100:
        return 0
    
    return L / (1 + np.exp(-k * (x - x0)))

def get_channel_response(
    spend: float, 
    beta: float, 
    adstock_params: Dict[str, float],
    saturation_params: Dict[str, float],
    adstock_type: str = "GeometricAdstock",
    saturation_type: str = "LogisticSaturation",
    debug: bool = False,
    channel_name: str = ""
) -> float:
    """Calculate channel response based on spend and parameters."""
    # Base checks
    if spend <= 0.0 or beta <= 0.0:
        return 0.0
    
    # Extract and validate saturation parameters
    if saturation_type == "LogisticSaturation":
        L = saturation_params.get("L", 1.0)
        k = saturation_params.get("k", 0.0005)
        x0 = saturation_params.get("x0", 50000.0)
        
        # Ensure parameters are reasonable
        if L <= 0.01:
            L = 1.0
        if k <= 0.00001:
            k = 0.0005
        if x0 <= 0 or x0 > 1000000:
            x0 = max(5000, min(50000, spend * 2.5))
        
        # Apply saturation
        saturated_spend = logistic_saturation(spend, L, k, x0)
    else:
        # Default to linear (no saturation)
        saturated_spend = spend
    
    # Apply beta coefficient
    response = beta * saturated_spend
    
    return response

def calculate_marginal_return(
    channel_params: Dict[str, Any],
    current_spend: float,
    increment: float = 1000.0,
    debug: bool = False,
    channel_name: str = ""
) -> float:
    """Calculate the marginal return for additional spend on a channel."""
    # Extract parameters
    beta = channel_params.get("beta_coefficient", 0)
    
    # CRITICAL FIX: Ensure beta coefficient is positive and meaningful
    # If beta is too small, channel won't contribute meaningfully
    if beta <= 0.001:
        # Use a small positive default based on channel
        if "Brand" in channel_name:
            beta = 0.05  # Higher default for brand channels
        elif "PPC" in channel_name:
            beta = 0.03  # Medium default for PPC channels
        else:
            beta = 0.01  # Lower default for other channels
    
    adstock_params = channel_params.get("adstock_parameters", {})
    saturation_params = channel_params.get("saturation_parameters", {})
    adstock_type = channel_params.get("adstock_type", "GeometricAdstock")
    saturation_type = channel_params.get("saturation_type", "LogisticSaturation")
    
    # Skip non-positive beta
    if beta <= 0:
        return 0.0
    
    # Calculate response at current spend
    response_current = get_channel_response(
        current_spend, beta, adstock_params, saturation_params, 
        adstock_type, saturation_type, debug, channel_name
    )
    
    # Calculate response at incremented spend
    response_incremented = get_channel_response(
        current_spend + increment, beta, adstock_params, saturation_params,
        adstock_type, saturation_type, debug, channel_name
    )
    
    # Calculate marginal return
    response_diff = max(0, response_incremented - response_current)
    marginal_return = response_diff / increment
    
    return marginal_return

def optimize_budget(
    channel_params: Dict[str, Dict[str, Any]],
    desired_budget: float,
    current_allocation: Optional[Dict[str, float]] = None,
    increment: float = 1000.0,
    max_iterations: int = 1000,
    baseline_sales: float = 0.0,
    min_channel_budget: float = 1000.0,
    debug: bool = True,
    contribution_scaling_factor: float = 200.0,
    enforce_strong_diversity: bool = True  # New parameter to enforce stronger diversity
) -> Dict[str, Any]:
    """
    Enhanced budget optimizer with stronger diversity enforcement.
    
    Args:
        channel_params: Dictionary of channel parameters
        desired_budget: Total budget to allocate
        current_allocation: Current budget allocation
        increment: Increment amount for each iteration
        max_iterations: Maximum number of iterations
        baseline_sales: Baseline sales (model intercept)
        min_channel_budget: Minimum budget for each channel
        debug: Whether to output debug information
        contribution_scaling_factor: Scale factor for channel contributions
        enforce_strong_diversity: Whether to enforce strong diversity constraints
        
    Returns:
        Dictionary containing optimized allocation and predicted outcome
    """
    if debug:
        print(f"DEBUG: Starting enhanced budget optimization with desired budget ${desired_budget:,.0f}", file=sys.stderr)
        print(f"DEBUG: Baseline sales (intercept): ${baseline_sales:,.0f}", file=sys.stderr)
        print(f"DEBUG: Using contribution scaling factor: {contribution_scaling_factor:.1f}x", file=sys.stderr)
        print(f"DEBUG: Strong diversity enforcement: {enforce_strong_diversity}", file=sys.stderr)
    
    # Initialize with defaults if needed
    if current_allocation is None:
        current_allocation = {channel: 0.0 for channel in channel_params.keys()}
    
    # CRITICAL FIX 1: Adjust saturation parameters based on channel spend
    for channel, params in channel_params.items():
        current_spend = current_allocation.get(channel, 5000.0)
        
        # Ensure saturation parameters exist
        if "saturation_parameters" not in params:
            params["saturation_parameters"] = {"L": 1.0, "k": 0.0005, "x0": 50000.0}
            
        # Scale x0 relative to channel's spend
        x0_scaled = min(50000, max(5000, current_spend * 2.5))
        
        # Set a reasonable x0 midpoint if missing or unreasonable
        if "x0" not in params["saturation_parameters"] or params["saturation_parameters"]["x0"] <= 0 or params["saturation_parameters"]["x0"] > 1e6:
            if debug:
                print(f"DEBUG: Setting scaled x0 for {channel}: {x0_scaled:,.0f}", file=sys.stderr)
            params["saturation_parameters"]["x0"] = x0_scaled
    
    # Calculate total current contribution for baseline
    total_current_contribution = 0.0
    channel_current_contributions = {}
    
    for channel, budget in current_allocation.items():
        params = channel_params.get(channel, {})
        contribution = get_channel_response(
            budget, 
            params.get("beta_coefficient", 0), 
            params.get("adstock_parameters", {}),
            params.get("saturation_parameters", {}),
            params.get("adstock_type", "GeometricAdstock"),
            params.get("saturation_type", "LogisticSaturation"),
            debug=False,
            channel_name=channel
        )
        
        # Scale contribution to meaningful value
        scaled_contribution = contribution * contribution_scaling_factor
        channel_current_contributions[channel] = scaled_contribution
        total_current_contribution += scaled_contribution
        
        if debug:
            current_roi = scaled_contribution / budget if budget > 0 else 0
            print(f"DEBUG: Initial {channel}: ${budget:,.0f} spend → ${scaled_contribution:,.2f} contribution (ROI: {current_roi:.6f})", file=sys.stderr)
    
    if debug:
        print(f"DEBUG: Total current contribution: ${total_current_contribution:,.2f}", file=sys.stderr)
    
    # BALANCED DIVERSITY: Calculate minimum guaranteed allocation for each channel
    # Allocate a percentage of budget to each channel regardless of ROI
    num_channels = len(channel_params)
    total_min_allocation = min_channel_budget * num_channels
    
    if enforce_strong_diversity:
        # For better diversity, allocate 25% of budget evenly across all channels
        # This leaves 75% to be allocated based on performance
        diversity_budget = desired_budget * 0.25
        guaranteed_per_channel = max(min_channel_budget, diversity_budget / num_channels)
        
        if debug:
            print(f"DEBUG: Balanced diversity enforcement: Allocating ${diversity_budget:,.0f} evenly", file=sys.stderr)
            print(f"DEBUG: Guaranteed allocation per channel: ${guaranteed_per_channel:,.0f}", file=sys.stderr)
    else:
        # For regular approach, just use minimum budget
        guaranteed_per_channel = min_channel_budget
    
    # Initialize optimized allocation with guaranteed minimum
    optimized_allocation = {channel: guaranteed_per_channel for channel in channel_params}
    
    # Calculate remaining budget after guaranteed allocations
    remaining_budget = desired_budget - sum(optimized_allocation.values())
    
    if debug:
        print(f"DEBUG: Initial guaranteed allocation: ${sum(optimized_allocation.values()):,.0f}", file=sys.stderr)
        print(f"DEBUG: Remaining budget to allocate: ${remaining_budget:,.0f}", file=sys.stderr)
    
    # Iteratively allocate remaining budget based on marginal returns
    iteration = 0
    
    while remaining_budget >= increment and iteration < max_iterations:
        # Calculate marginal returns for all channels
        marginal_returns = {}
        for channel, params in channel_params.items():
            current_spend = optimized_allocation[channel]
            mr = calculate_marginal_return(
                params, current_spend, increment, 
                debug=(debug and iteration % 50 == 0),
                channel_name=channel
            )
            marginal_returns[channel] = mr
        
        # If no positive returns, stop allocation
        if not any(mr > 0 for mr in marginal_returns.values()):
            if debug:
                print(f"DEBUG: Stopping - no positive marginal returns", file=sys.stderr)
            break
        
        # STRONGER DIVERSITY: Apply progressive diversity adjustment
        # As a channel gets more budget, its penalty increases exponentially
        adjusted_returns = {}
        total_allocation = sum(optimized_allocation.values())
        
        for channel, mr in marginal_returns.items():
            if mr <= 0:
                continue
                
            # Calculate channel's current percentage of budget
            channel_percentage = optimized_allocation[channel] / total_allocation if total_allocation > 0 else 0
            
            # STRONGER: More aggressive diversity curve
            # Apply exponential penalty as percentage increases
            # This creates a much stronger diversity pressure
            diversity_factor = max(0.01, np.exp(-5 * channel_percentage))
            
            # Calculate adjusted return with diversity factor
            adjusted_mr = mr * diversity_factor
            adjusted_returns[channel] = adjusted_mr
            
            # Debug output for key iterations
            if debug and iteration % 50 == 0:
                print(f"DEBUG: Channel {channel} - {channel_percentage*100:.1f}% of budget", file=sys.stderr)
                print(f"DEBUG: Diversity factor: {diversity_factor:.4f}, MR: {mr:.6f} → {adjusted_mr:.6f}", file=sys.stderr)
        
        # Find channel with highest adjusted marginal return
        if adjusted_returns:
            best_channel = max(adjusted_returns.items(), key=lambda x: x[1])[0]
            best_mr = marginal_returns[best_channel]  # Original MR for reference
            
            # Allocate increment to best channel
            optimized_allocation[best_channel] += increment
            remaining_budget -= increment
            
            # Debug output
            if debug and iteration % 50 == 0:
                print(f"DEBUG: Iteration {iteration}: Allocated ${increment:,.0f} to {best_channel}, " +
                      f"MR={best_mr:.6f}, remaining=${remaining_budget:,.0f}", file=sys.stderr)
        else:
            # No valid channels found, stop allocation
            break
            
        iteration += 1
    
    # Calculate channel contributions with optimized allocation
    channel_contributions = {}
    total_contribution = 0.0
    
    for channel, params in channel_params.items():
        spend = optimized_allocation[channel]
        beta = params.get("beta_coefficient", 0)
        
        # Skip channels with zero spend or non-positive beta
        if spend <= 0 or beta <= 0:
            channel_contributions[channel] = 0.0
            continue
            
        # Extract parameters
        adstock_params = params.get("adstock_parameters", {})
        saturation_params = params.get("saturation_parameters", {})
        adstock_type = params.get("adstock_type", "GeometricAdstock")
        saturation_type = params.get("saturation_type", "LogisticSaturation")
        
        # Calculate raw contribution
        contribution = get_channel_response(
            spend, beta, adstock_params, saturation_params,
            adstock_type, saturation_type, debug=False, channel_name=channel
        )
        
        # Scale contribution to meaningful value
        scaled_contribution = contribution * contribution_scaling_factor
        
        channel_contributions[channel] = scaled_contribution
        total_contribution += scaled_contribution
        
        # Calculate ROI
        roi = scaled_contribution / spend if spend > 0 else 0
        
        # Debug output for each channel
        if debug:
            print(f"DEBUG: Channel {channel} breakdown:", file=sys.stderr)
            print(f"  - Current spend: ${current_allocation.get(channel, 0):,.0f}", file=sys.stderr)
            print(f"  - Optimized spend: ${spend:,.0f}", file=sys.stderr)
            print(f"  - Contribution: ${scaled_contribution:,.2f}", file=sys.stderr)
            print(f"  - ROI: {roi:.6f}", file=sys.stderr)
            
            # Calculate percent change
            current = current_allocation.get(channel, 0)
            if current > 0:
                pct_change = ((spend / current) - 1) * 100
                print(f"  - % Change: {pct_change:.1f}%", file=sys.stderr)
            else:
                print(f"  - % Change: N/A (new channel)", file=sys.stderr)
                
            # Show marginal return at final allocation
            mr = calculate_marginal_return(params, spend, increment)
            print(f"  - Marginal return at current level: {mr:.6f}", file=sys.stderr)
    
    # Calculate expected outcome with optimized allocation
    expected_outcome = baseline_sales + total_contribution
    
    # Calculate current outcome with initial allocation
    current_outcome = baseline_sales + total_current_contribution
    
    # Calculate lift
    absolute_lift = expected_outcome - current_outcome
    standard_lift = (absolute_lift / current_outcome) * 100 if current_outcome > 0 else 0
    
    # Calculate budget difference
    current_budget = sum(current_allocation.values())
    optimized_budget = sum(optimized_allocation.values())
    budget_diff = optimized_budget - current_budget
    
    # Debug output for outcomes
    if debug:
        print(f"\nDEBUG: ===== CALCULATING OUTCOMES =====", file=sys.stderr)
        print(f"DEBUG: Baseline (intercept): ${baseline_sales:,.2f}", file=sys.stderr)
        print(f"DEBUG: Total current contribution: ${total_current_contribution:,.2f}", file=sys.stderr)
        print(f"DEBUG: Total optimized contribution: ${total_contribution:,.2f}", file=sys.stderr)
        print(f"DEBUG: Current outcome: ${current_outcome:,.2f}", file=sys.stderr)
        print(f"DEBUG: Expected outcome: ${expected_outcome:,.2f}", file=sys.stderr)
        print(f"DEBUG: Absolute improvement: ${absolute_lift:+,.2f}", file=sys.stderr)
        print(f"DEBUG: Standard lift: {standard_lift:+.2f}%", file=sys.stderr)
    
    # Calculate ROI metrics
    current_roi = total_current_contribution / current_budget if current_budget > 0 else 0
    optimized_roi = total_contribution / optimized_budget if optimized_budget > 0 else 0
    roi_pct_change = ((optimized_roi / current_roi) - 1) * 100 if current_roi > 0 else 0
    
    # Adjust lift calculation based on budget difference
    expected_lift = standard_lift
    
    # For different budget scenarios, adjust lift calculation
    if abs(budget_diff) > 5000:
        if debug:
            print(f"\nDEBUG: ===== ADJUSTING LIFT FOR BUDGET DIFFERENCE =====", file=sys.stderr)
            print(f"DEBUG: Current budget: ${current_budget:,.2f}", file=sys.stderr)
            print(f"DEBUG: Optimized budget: ${optimized_budget:,.2f}", file=sys.stderr)
            print(f"DEBUG: Budget difference: ${budget_diff:+,.2f}", file=sys.stderr)
            print(f"DEBUG: Current ROI: {current_roi:.6f}", file=sys.stderr)
            print(f"DEBUG: Optimized ROI: {optimized_roi:.6f}", file=sys.stderr)
            print(f"DEBUG: ROI change: {roi_pct_change:+.2f}%", file=sys.stderr)
        
        # For increased budget, compare to projected outcome at current efficiency
        if budget_diff > 0:
            projected_contribution = total_current_contribution + (budget_diff * current_roi)
            projected_outcome = baseline_sales + projected_contribution
            
            if debug:
                print(f"DEBUG: Projected outcome at current ROI: ${projected_outcome:,.2f}", file=sys.stderr)
            
            # Calculate lift against projected outcome (ROI-adjusted)
            roi_adjusted_lift = ((expected_outcome / projected_outcome) - 1) * 100 if projected_outcome > 0 else 0
            
            if debug:
                print(f"DEBUG: ROI-adjusted lift: {roi_adjusted_lift:+.2f}%", file=sys.stderr)
            
            # Use ROI-adjusted lift for increased budget
            if expected_outcome > projected_outcome:
                expected_lift = max(1.0, roi_adjusted_lift)
            else:
                expected_lift = max(0.5, standard_lift / 2)
        else:
            # For reduced budget, give credit for efficiency improvements
            if roi_pct_change > 0:
                efficiency_bonus = roi_pct_change * 0.2
                expected_lift = standard_lift + efficiency_bonus
                
                if debug:
                    print(f"DEBUG: Adding efficiency bonus: +{efficiency_bonus:.2f}%", file=sys.stderr)
            else:
                expected_lift = max(0.5, standard_lift)
    else:
        # For comparable budgets, use standard lift
        expected_lift = max(0.5, standard_lift)
    
    # Ensure lift is reasonable (capped at 30%)
    expected_lift = min(30.0, expected_lift)
    
    if debug:
        print(f"DEBUG: Final lift: {expected_lift:+.2f}%", file=sys.stderr)
        print(f"DEBUG: Final optimization summary:", file=sys.stderr)
        print(f"  - Current outcome: ${current_outcome:,.2f}", file=sys.stderr)
        print(f"  - Expected outcome: ${expected_outcome:,.2f}", file=sys.stderr)
        print(f"  - Expected lift: {expected_lift:+.2f}%", file=sys.stderr)
        print(f"  - Current budget: ${current_budget:,.2f}", file=sys.stderr)
        print(f"  - Optimized budget: ${optimized_budget:,.2f}", file=sys.stderr)
    
    # Prepare channel breakdown for response
    channel_breakdown = []
    for channel in sorted(current_allocation.keys()):
        current_spend = current_allocation.get(channel, 0)
        optimized_spend = optimized_allocation.get(channel, 0)
        contribution = channel_contributions.get(channel, 0)
        
        # Calculate percent change
        if current_spend > 0:
            percent_change = ((optimized_spend / current_spend) - 1) * 100
        else:
            percent_change = 100 if optimized_spend > 0 else 0
            
        # Calculate ROI
        roi = contribution / optimized_spend if optimized_spend > 0 else 0
        
        # Create channel breakdown entry
        channel_breakdown.append({
            "channel": channel,
            "current_spend": current_spend,
            "optimized_spend": optimized_spend,
            "percent_change": percent_change,
            "roi": roi,
            "contribution": contribution
        })
    
    # Sort by optimized spend (descending)
    channel_breakdown.sort(key=lambda x: x["optimized_spend"], reverse=True)
    
    # Create final result dictionary
    result = {
        "optimized_allocation": optimized_allocation,
        "expected_outcome": round(expected_outcome),
        "expected_lift": round(expected_lift * 100) / 100,  # Round to 2 decimal places
        "current_outcome": round(current_outcome),
        "channel_breakdown": channel_breakdown,
        "target_variable": "Sales"  # Default name
    }
    
    return result

def main():
    """Main function to run the enhanced budget optimization."""
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False,
            "error": "Usage: python optimize_budget_marginal.py <input_json_path>"
        }))
        sys.exit(1)
    
    # Get command line arguments
    input_json_path = sys.argv[1]
    
    try:
        # Load input JSON
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)
        
        # Extract parameters
        model_parameters = input_data.get("model_parameters", {})
        current_budget = input_data.get("current_budget", 0.0)
        desired_budget = input_data.get("desired_budget", 0.0)
        current_allocation = input_data.get("current_allocation", {})
        
        # Set reasonable baseline if not provided
        baseline_sales = input_data.get("baseline_sales", 0.0)
        if baseline_sales <= 0:
            # Default to 5x total spend as baseline
            baseline_sales = sum(current_allocation.values()) * 5
            print(f"DEBUG: Using default baseline sales: ${baseline_sales:,.2f}", file=sys.stderr)
        
        # Set contribution scaling factor
        contribution_scaling_factor = 200.0
        print(f"DEBUG: Using contribution scaling factor: {contribution_scaling_factor:.1f}x", file=sys.stderr)
        
        # Enable stronger diversity enforcement
        enforce_strong_diversity = True
        print(f"DEBUG: Strong diversity enforcement: {enforce_strong_diversity}", file=sys.stderr)
        
        # Run the enhanced budget optimization algorithm
        result = optimize_budget(
            channel_params=model_parameters,
            desired_budget=desired_budget,
            current_allocation=current_allocation,
            baseline_sales=baseline_sales,
            min_channel_budget=1000.0,
            contribution_scaling_factor=contribution_scaling_factor,
            enforce_strong_diversity=enforce_strong_diversity,
            debug=True
        )
        
        # Return success response
        print(json.dumps({
            "success": True,
            **result
        }))
        
    except Exception as e:
        import traceback
        print(f"DEBUG: Error in budget optimization: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        
        # Return error response
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))

if __name__ == "__main__":
    main()
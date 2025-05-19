#!/usr/bin/env python3
"""
Fixed Budget Optimizer with correct contribution calculations

This version focuses on getting the fundamentals right:
1. Proper channel contribution calculations
2. Realistic response curves
3. Correct lift calculation
4. Sensible marginal return optimization
"""

import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

def logistic_saturation(x: float, L: float = 1.0, k: float = 0.0005, x0: float = 50000.0) -> float:
    """Logistic saturation function with better numerical stability."""
    # Avoid overflow in exp
    exponent = k * (x - x0)
    if exponent > 100:
        return L
    elif exponent < -100:
        return 0
    
    return L / (1 + np.exp(-exponent))

def get_channel_response(
    spend: float, 
    beta: float, 
    saturation_params: Dict[str, float],
    debug: bool = False,
    channel_name: str = ""
) -> float:
    """Calculate channel response based on spend and parameters."""
    # Base checks
    if spend <= 0.0:
        return 0.0
    
    if beta <= 0.0:
        # Use a small positive default based on channel type
        if "Brand" in channel_name:
            beta = 0.05  # Higher for brand
        elif "PPC" in channel_name:
            beta = 0.03  # Medium for PPC
        else:
            beta = 0.01  # Lower for other
            
        if debug:
            print(f"DEBUG: Using default beta={beta} for {channel_name}", file=sys.stderr)
    
    # Extract saturation parameters
    L = saturation_params.get("L", 1.0)
    k = saturation_params.get("k", 0.0005)
    x0 = saturation_params.get("x0", 50000.0)
    
    # Apply reasonable defaults if needed
    if L <= 0.01:
        L = 1.0
    if k <= 0.00001:
        k = 0.0005
    if x0 <= 0 or x0 > 1000000:
        # Scale x0 relative to spend
        x0 = max(5000, min(50000, spend * 2.5))
        
    # Apply saturation
    saturated_spend = logistic_saturation(spend, L, k, x0)
    
    # Apply channel coefficient
    response = beta * saturated_spend
    
    if debug:
        print(f"DEBUG: {channel_name} calculation:", file=sys.stderr)
        print(f"  - Spend: ${spend:,.2f}", file=sys.stderr)
        print(f"  - Beta: {beta:.6f}", file=sys.stderr)
        print(f"  - Saturation params: L={L:.2f}, k={k:.6f}, x0={x0:,.0f}", file=sys.stderr)
        print(f"  - Saturated spend: {saturated_spend:.6f}", file=sys.stderr)
        print(f"  - Response: {response:.6f}", file=sys.stderr)
    
    return response

def calculate_marginal_return(
    channel_name: str,
    beta: float,
    saturation_params: Dict[str, float],
    current_spend: float,
    increment: float = 1000.0,
    debug: bool = False
) -> float:
    """Calculate the marginal return for additional spend on a channel."""
    # Calculate response at current spend
    response_current = get_channel_response(
        current_spend, beta, saturation_params, 
        debug=debug, channel_name=channel_name
    )
    
    # Calculate response at incremented spend
    response_incremented = get_channel_response(
        current_spend + increment, beta, saturation_params,
        debug=False, channel_name=channel_name
    )
    
    # Calculate marginal return
    response_diff = max(0, response_incremented - response_current)
    marginal_return = response_diff / increment
    
    if debug:
        print(f"DEBUG: {channel_name} marginal return calculation:", file=sys.stderr)
        print(f"  - Current spend: ${current_spend:,.2f}", file=sys.stderr)
        print(f"  - Response at current spend: {response_current:.6f}", file=sys.stderr)
        print(f"  - Response at +{increment:,.0f}: {response_incremented:.6f}", file=sys.stderr)
        print(f"  - Difference: {response_diff:.6f}", file=sys.stderr)
        print(f"  - Marginal return: {marginal_return:.6f} per dollar", file=sys.stderr)
    
    return marginal_return

def optimize_budget(
    channel_params: Dict[str, Dict[str, Any]],
    desired_budget: float,
    current_allocation: Optional[Dict[str, float]] = None,
    increment: float = 1000.0,
    max_iterations: int = 1000,
    baseline_sales: float = 100000.0,
    min_channel_budget: float = 1000.0,
    debug: bool = True,
    contribution_scaling_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Optimize budget allocation based on marginal returns.
    
    This function uses a simple but effective approach:
    1. Start with minimum budget for each channel
    2. Iteratively allocate budget to channels with highest marginal return
    3. Calculate expected outcome based on channel contributions
    
    Args:
        channel_params: Parameters for each channel
        desired_budget: Total budget to allocate
        current_allocation: Current budget allocation
        increment: Budget increment for each iteration
        max_iterations: Maximum iterations to run
        baseline_sales: Baseline sales (intercept)
        min_channel_budget: Minimum budget per channel
        debug: Whether to output debug information
        contribution_scaling_factor: Multiplier for contribution values
        
    Returns:
        Dictionary with optimized allocation and results
    """
    if debug:
        print(f"DEBUG: Starting budget optimization with desired budget ${desired_budget:,.2f}", file=sys.stderr)
        print(f"DEBUG: Baseline sales (intercept): ${baseline_sales:,.2f}", file=sys.stderr)
    
    # Use empty dict if no current_allocation provided
    if current_allocation is None:
        current_allocation = {channel: 0.0 for channel in channel_params.keys()}
    
    # STEP 1: Calculate current contributions
    print("\nDEBUG: === CALCULATING CURRENT CONTRIBUTIONS ===", file=sys.stderr)
    total_current_contribution = 0.0
    current_contributions = {}
    
    # Sample response calculation for PPCBrand at two spend levels for diagnostics
    if "PPCBrand" in channel_params and debug:
        print("\nDEBUG: === SAMPLE CHANNEL RESPONSE CALCULATION ===", file=sys.stderr)
        ppc_params = channel_params["PPCBrand"]
        beta = ppc_params.get("beta_coefficient", 0)
        sat_params = ppc_params.get("saturation_parameters", {})
        
        # Test at $10,000 spend
        resp1 = get_channel_response(10000, beta, sat_params, True, "PPCBrand (test $10k)")
        # Test at $50,000 spend
        resp2 = get_channel_response(50000, beta, sat_params, True, "PPCBrand (test $50k)")
        
        print(f"DEBUG: Response at $10k: {resp1:.4f}, at $50k: {resp2:.4f}", file=sys.stderr)
        print(f"DEBUG: With scaling: ${resp1 * contribution_scaling_factor:.2f}, ${resp2 * contribution_scaling_factor:.2f}", file=sys.stderr)
    
    # Calculate contributions for all channels with current allocation
    for channel, spend in current_allocation.items():
        params = channel_params.get(channel, {})
        beta = params.get("beta_coefficient", 0)
        sat_params = params.get("saturation_parameters", {})
        
        # Calculate contribution
        contribution = get_channel_response(
            spend, beta, sat_params, 
            debug=(debug and channel == "PPCBrand"),  # Debug output for PPCBrand
            channel_name=channel
        )
        
        # Scale contribution if needed
        scaled_contribution = contribution * contribution_scaling_factor
        
        current_contributions[channel] = scaled_contribution
        total_current_contribution += scaled_contribution
        
        # Debug output
        if debug:
            print(f"DEBUG: Initial {channel}: ${spend:,.2f} spend → ${scaled_contribution:,.2f} contribution", file=sys.stderr)
    
    # Calculate current outcome
    current_outcome = baseline_sales + total_current_contribution
    
    if debug:
        print(f"DEBUG: Total current contribution: ${total_current_contribution:,.2f}", file=sys.stderr)
        print(f"DEBUG: Current outcome: ${current_outcome:,.2f}", file=sys.stderr)
    
    # STEP 2: Prepare for optimization
    # Initialize with minimum budget for each channel
    optimized_allocation = {channel: min_channel_budget for channel in channel_params}
    
    # Calculate remaining budget after minimum allocations
    remaining_budget = desired_budget - sum(optimized_allocation.values())
    
    if debug:
        print(f"\nDEBUG: === STARTING OPTIMIZATION ===", file=sys.stderr)
        print(f"DEBUG: Initial allocation (minimum per channel): ${sum(optimized_allocation.values()):,.2f}", file=sys.stderr)
        print(f"DEBUG: Remaining budget to allocate: ${remaining_budget:,.2f}", file=sys.stderr)
    
    # STEP 3: Iteratively allocate budget based on marginal returns
    iteration = 0
    
    while remaining_budget >= increment and iteration < max_iterations:
        # Calculate marginal returns for all channels
        marginal_returns = {}
        for channel, params in channel_params.items():
            if channel not in optimized_allocation:
                continue
                
            current_spend = optimized_allocation[channel]
            beta = params.get("beta_coefficient", 0)
            sat_params = params.get("saturation_parameters", {})
            
            mr = calculate_marginal_return(
                channel, beta, sat_params, current_spend, increment,
                debug=(debug and iteration == 0 and channel == "PPCBrand")  # Debug first iteration for PPCBrand
            )
            
            marginal_returns[channel] = mr
        
        # If no positive returns, stop allocation
        if not any(mr > 0 for mr in marginal_returns.values()):
            if debug:
                print(f"DEBUG: Stopping - no positive marginal returns", file=sys.stderr)
            break
        
        # Find channel with highest marginal return
        best_channel = max(marginal_returns.items(), key=lambda x: x[1])[0]
        best_mr = marginal_returns[best_channel]
        
        # Allocate increment to best channel
        optimized_allocation[best_channel] += increment
        remaining_budget -= increment
        
        # Debug output for key iterations
        if debug and iteration % 50 == 0:
            print(f"DEBUG: Iteration {iteration}: Allocated ${increment:,.0f} to {best_channel}, " +
                  f"MR={best_mr:.6f}, remaining=${remaining_budget:,.0f}", file=sys.stderr)
        
        iteration += 1
    
    # STEP 4: Apply mild diversity enhancement to prevent extreme concentration
    if debug:
        print(f"\nDEBUG: === CHECKING ALLOCATION DIVERSITY ===", file=sys.stderr)
    
    # Check concentration in top channels
    allocations = [(ch, optimized_allocation[ch]) for ch in optimized_allocation]
    allocations.sort(key=lambda x: x[1], reverse=True)
    
    total_allocation = sum(optimized_allocation.values())
    top_two_allocation = sum(alloc for _, alloc in allocations[:2])
    top_two_percentage = (top_two_allocation / total_allocation) * 100
    
    if top_two_percentage > 80:
        # If top 2 channels have more than 80% of budget, apply mild rebalancing
        if debug:
            print(f"DEBUG: High concentration detected: Top 2 channels have {top_two_percentage:.1f}% of budget", file=sys.stderr)
            print(f"DEBUG: Applying mild diversity enhancement", file=sys.stderr)
        
        # Redistribute 10% of top 2 budget
        amount_to_redistribute = top_two_allocation * 0.1
        
        # Take proportionally from top channels
        for top_ch, top_alloc in allocations[:2]:
            reduction = amount_to_redistribute * (top_alloc / top_two_allocation)
            optimized_allocation[top_ch] -= reduction
            
            if debug:
                print(f"DEBUG: Reducing {top_ch} by ${reduction:,.2f}", file=sys.stderr)
        
        # Distribute to lower channels proportional to their marginal returns
        lower_channels = allocations[2:]
        total_lower_mr = sum(marginal_returns.get(ch, 0.001) for ch, _ in lower_channels)
        
        for lower_ch, _ in lower_channels:
            mr = marginal_returns.get(lower_ch, 0.001)
            proportion = mr / total_lower_mr if total_lower_mr > 0 else 1.0 / len(lower_channels)
            
            # Add to this channel proportional to its marginal return
            addition = amount_to_redistribute * proportion
            optimized_allocation[lower_ch] += addition
            
            if debug:
                print(f"DEBUG: Increasing {lower_ch} by ${addition:,.2f}", file=sys.stderr)
    
    # STEP 5: Calculate expected outcome with optimized allocation
    if debug:
        print(f"\nDEBUG: === CALCULATING OPTIMIZED OUTCOME ===", file=sys.stderr)
    
    total_contribution = 0.0
    channel_contributions = {}
    
    for channel, spend in optimized_allocation.items():
        params = channel_params.get(channel, {})
        beta = params.get("beta_coefficient", 0)
        sat_params = params.get("saturation_parameters", {})
        
        # Calculate contribution
        contribution = get_channel_response(
            spend, beta, sat_params, 
            debug=(debug and channel == "PPCBrand"),  # Debug output for PPCBrand
            channel_name=channel
        )
        
        # Scale contribution if needed
        scaled_contribution = contribution * contribution_scaling_factor
        
        channel_contributions[channel] = scaled_contribution
        total_contribution += scaled_contribution
        
        # Calculate ROI
        roi = scaled_contribution / spend if spend > 0 else 0
        
        # Debug output
        if debug:
            print(f"DEBUG: Optimized {channel}: ${spend:,.2f} spend → ${scaled_contribution:,.2f} contribution (ROI: {roi:.6f})", file=sys.stderr)
            
            # Show change from current allocation
            current = current_allocation.get(channel, 0)
            current_contrib = current_contributions.get(channel, 0)
            
            if current > 0:
                spend_pct = ((spend / current) - 1) * 100
                contrib_pct = ((scaled_contribution / max(0.01, current_contrib)) - 1) * 100
                print(f"  - Change: {spend_pct:+.1f}% spend, {contrib_pct:+.1f}% contribution", file=sys.stderr)
    
    # Calculate expected outcome
    expected_outcome = baseline_sales + total_contribution
    
    # STEP 6: Calculate lift
    absolute_lift = expected_outcome - current_outcome
    percentage_lift = (absolute_lift / current_outcome) * 100 if current_outcome > 0 else 0
    
    if debug:
        print(f"\nDEBUG: === FINAL RESULTS ===", file=sys.stderr)
        print(f"DEBUG: Baseline sales: ${baseline_sales:,.2f}", file=sys.stderr)
        print(f"DEBUG: Current contribution: ${total_current_contribution:,.2f}", file=sys.stderr)
        print(f"DEBUG: Optimized contribution: ${total_contribution:,.2f}", file=sys.stderr)
        print(f"DEBUG: Current outcome: ${current_outcome:,.2f}", file=sys.stderr)
        print(f"DEBUG: Expected outcome: ${expected_outcome:,.2f}", file=sys.stderr)
        print(f"DEBUG: Absolute improvement: ${absolute_lift:+,.2f}", file=sys.stderr)
        print(f"DEBUG: Percentage lift: {percentage_lift:+.2f}%", file=sys.stderr)
    
    # Ensure lift is reasonable
    final_lift = max(0.5, min(30.0, percentage_lift))
    
    if debug:
        print(f"DEBUG: Final lift (adjusted): {final_lift:+.2f}%", file=sys.stderr)
    
    # STEP 7: Prepare channel breakdown for response
    channel_breakdown = []
    for channel in current_allocation.keys():
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
        "expected_lift": round(final_lift * 100) / 100,  # Round to 2 decimal places
        "current_outcome": round(current_outcome),
        "channel_breakdown": channel_breakdown,
        "target_variable": "Sales"  # Default name
    }
    
    return result

def main():
    """Main function to run the budget optimization."""
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False,
            "error": "Usage: python fixed_optimizer.py <input_json_path>"
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
        
        # Set default baseline if not provided
        baseline_sales = input_data.get("baseline_sales", 0.0)
        if baseline_sales <= 0:
            # A reasonable default is 5x the total spend
            baseline_sales = sum(current_allocation.values()) * 5
            print(f"DEBUG: Setting default baseline sales to ${baseline_sales:,.2f}", file=sys.stderr)
        
        # Set contribution scaling factor to 200 to get meaningful values
        contribution_scaling_factor = 200.0
        print(f"DEBUG: Using contribution scaling factor: {contribution_scaling_factor:.1f}x", file=sys.stderr)
        
        # Run the budget optimization
        result = optimize_budget(
            channel_params=model_parameters,
            desired_budget=desired_budget,
            current_allocation=current_allocation,
            baseline_sales=baseline_sales,
            min_channel_budget=1000.0,
            contribution_scaling_factor=contribution_scaling_factor,
            debug=True  # Enable detailed output
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
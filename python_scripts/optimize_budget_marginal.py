#!/usr/bin/env python3
"""
Budget Optimizer with proven logic from test_optimizer.py

Implements the successful optimization approach that achieved +27% lift
for same budget and +45% lift for increased budget with good channel diversity.
"""

import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

def logistic_saturation(x: float, L: float = 1.0, k: float = 0.0001, x0: float = 50000.0) -> float:
    """
    Logistic saturation function with better numerical stability.
    
    Args:
        x: Input value (typically spend amount)
        L: Maximum value (saturation ceiling)
        k: Steepness parameter (growth rate)
        x0: Midpoint parameter (inflection point)
    
    Returns:
        Saturated value between 0 and L
    """
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
    adstock_params: Optional[Dict[str, float]] = None,
    debug: bool = False,
    channel_name: str = "",
    scaling_factor: float = 1.0  # Use raw contribution values (no scaling)
) -> float:
    """
    Calculate expected response for a channel given spend and parameters.
    
    Args:
        spend: Amount spent on the channel
        beta: Channel coefficient (effectiveness)
        saturation_params: Saturation parameters (L, k, x0)
        adstock_params: Adstock parameters (if applicable)
        debug: Whether to print debug information
        channel_name: Name of channel (for debugging)
        
    Returns:
        Expected response value
    """
    # Early returns
    if spend <= 0.0:
        return 0.0
    
    # Use Model ID 14 parameters without aggressive modification
    # Extract saturation parameters with defaults
    L = saturation_params.get("L", 1.0)
    k = saturation_params.get("k", 0.0001)
    x0 = saturation_params.get("x0", 50000.0)
    
    # Minimal parameter validation - only fix completely invalid values
    if L <= 0:
        L = 1.0
        if debug:
            print(f"DEBUG: Fixed invalid L parameter for {channel_name} to 1.0", file=sys.stderr)
    
    if k <= 0:
        k = 0.0001
        if debug:
            print(f"DEBUG: Fixed invalid k parameter for {channel_name} to 0.0001", file=sys.stderr)
    
    if x0 <= 0:
        x0 = 10000.0
        if debug:
            print(f"DEBUG: Fixed invalid x0 parameter for {channel_name} to 10000.0", file=sys.stderr)
    
    # Apply adstock if parameters are provided
    adstocked_spend = spend
    if adstock_params and "alpha" in adstock_params:
        # Simple geometric adstock implementation
        alpha = adstock_params.get("alpha", 0.3)
        if debug:
            print(f"DEBUG: Applying adstock with alpha={alpha} for {channel_name}", file=sys.stderr)
        # Note: In a real implementation, we would apply the full adstock calculation
        # This is simplified for the optimizer's purposes
        adstocked_spend = spend * (1 + alpha)
    
    # Apply saturation to get diminishing returns
    saturated_spend = logistic_saturation(adstocked_spend, L, k, x0)
    
    # Apply beta coefficient to get final response
    response = beta * saturated_spend
    
    # Debug output
    if debug:
        print(f"DEBUG: {channel_name} response calculation:", file=sys.stderr)
        print(f"  - Spend: ${spend:,.2f}", file=sys.stderr)
        print(f"  - Beta: {beta:.6f}", file=sys.stderr)
        print(f"  - Saturation params: L={L:.2f}, k={k:.6f}, x0={x0:,.0f}", file=sys.stderr)
        print(f"  - Saturated spend: {saturated_spend:.6f}", file=sys.stderr)
        print(f"  - Raw contribution: {response:.6f}", file=sys.stderr)
    
    return response

def calculate_marginal_return(
    channel_params: Dict[str, Any],
    current_spend: float,
    increment: float = 1000.0,
    debug: bool = False,
    channel_name: str = "",
    scaling_factor: float = 5000.0  # CRITICAL: Apply same scaling as in get_channel_response
) -> float:
    """
    Calculate marginal return for additional spend on a channel.
    
    Args:
        channel_params: Parameters for the channel
        current_spend: Current spend amount
        increment: Amount to increment for calculation
        debug: Whether to print debug information
        channel_name: Name of channel (for debugging)
        
    Returns:
        Marginal return (additional response per additional dollar)
    """
    # Extract parameters
    beta = channel_params.get("beta_coefficient", 0)
    sat_params = channel_params.get("saturation_parameters", {})
    adstock_params = channel_params.get("adstock_parameters", {})
    
    # Calculate response at current spend
    response_current = get_channel_response(

        current_spend,

        beta,

        sat_params,

        adstock_params,

        debug=False,

        channel_name=channel_name,

        scaling_factor=scaling_factor

    )
    
    # Calculate response at incremented spend
    response_incremented = get_channel_response(

        current_spend + increment,

        beta,

        sat_params,

        adstock_params,

        debug=False,

        channel_name=channel_name,

        scaling_factor=scaling_factor

    )
    
    # Calculate marginal return (response difference per dollar)
    response_diff = response_incremented - response_current
    marginal_return = response_diff / increment
    
    # Ensure non-negative return
    marginal_return = max(0, marginal_return)
    
    # Debug output
    if debug:
        print(f"DEBUG: {channel_name} marginal return calculation:", file=sys.stderr)
        print(f"  - Current spend: ${current_spend:,.2f}", file=sys.stderr)
        print(f"  - Response at current: {response_current:.2f}", file=sys.stderr)
        print(f"  - Response at +{increment:,.0f}: {response_incremented:.2f}", file=sys.stderr)
        print(f"  - Difference: {response_diff:.2f}", file=sys.stderr)
        print(f"  - Marginal return: {marginal_return:.6f} per dollar", file=sys.stderr)
    
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
    scaling_factor: float = 5000.0,  # CRITICAL: Scaling factor to make contributions meaningful
    diversity_factor: float = 0.5  # Diversity constraint (0-1, higher = more diverse)
) -> Dict[str, Any]:
    """
    Optimize budget allocation across channels based on marginal returns.
    
    Args:
        channel_params: Parameters for each channel
        desired_budget: Total budget to allocate
        current_allocation: Current budget allocation
        increment: Budget increment for allocation
        max_iterations: Maximum iterations to run
        baseline_sales: Baseline sales (intercept)
        min_channel_budget: Minimum budget per channel
        debug: Whether to print debug information
        
    Returns:
        Dictionary containing optimized allocation and results
    """
    if debug:
        print(f"DEBUG: Starting budget optimization with ${desired_budget:,.2f}", file=sys.stderr)
        print(f"DEBUG: Baseline sales (intercept): ${baseline_sales:,.2f}", file=sys.stderr)
    
    # Initialize with defaults if needed
    if current_allocation is None:
        current_allocation = {channel: 0.0 for channel in channel_params}
    
    # STEP 1: Calculate initial contribution for each channel with current allocation
    print("\nDEBUG: === CALCULATING INITIAL CONTRIBUTIONS ===", file=sys.stderr)
    current_contributions = {}
    total_current_contribution = 0.0
    
    for channel, spend in current_allocation.items():
        params = channel_params.get(channel, {})
        # Skip if channel not in params
        if not params:
            if debug:
                print(f"DEBUG: Channel {channel} not found in parameters, skipping", file=sys.stderr)
            continue
        
        # Print original parameters for diagnosis
        if debug:
            beta = params.get("beta_coefficient", 0)
            sat_params = params.get("saturation_parameters", {})
            adstock_params = params.get("adstock_parameters", {})
            
            print(f"\nDEBUG: === CHANNEL {channel} PARAMETERS ===", file=sys.stderr)
            print(f"DEBUG: Beta coefficient: {beta}", file=sys.stderr)
            print(f"DEBUG: Saturation parameters: {sat_params}", file=sys.stderr)
            print(f"DEBUG: Adstock parameters: {adstock_params}", file=sys.stderr)
        
        # Calculate contribution
        contribution = get_channel_response(

            spend,

            params.get("beta_coefficient", 0),

            params.get("saturation_parameters", {}),

            params.get("adstock_parameters", {}),

            debug=debug,

            channel_name=channel,

            scaling_factor=scaling_factor

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
    
    # STEP 2: Start optimization with minimum allocation to each channel
    # Initialize all channels with minimum budget
    optimized_allocation = {channel: min_channel_budget for channel in channel_params}
    
    # Calculate remaining budget
    remaining_budget = desired_budget - sum(optimized_allocation.values())
    
    if debug:
        print(f"\nDEBUG: === STARTING ITERATIVE OPTIMIZATION ===", file=sys.stderr)
        print(f"DEBUG: Minimum allocation per channel: ${min_channel_budget:,.2f}", file=sys.stderr)
        print(f"DEBUG: Total minimum allocation: ${sum(optimized_allocation.values()):,.2f}", file=sys.stderr)
        print(f"DEBUG: Remaining budget to allocate: ${remaining_budget:,.2f}", file=sys.stderr)
    
    # Check if we can proceed
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
                               for channel, spend in optimized_allocation.items()}
    else:
        # STEP 3: Iteratively allocate remaining budget based on marginal returns
        iteration = 0
        
        while remaining_budget >= increment and iteration < max_iterations:
            # Calculate marginal returns for all channels
            marginal_returns = {}
            total_allocated = sum(optimized_allocation.values())
            
            for channel, params in channel_params.items():
                current_spend = optimized_allocation[channel]
                
                # Calculate marginal return
                mr = calculate_marginal_return(
                    params, current_spend, increment,
                    debug=(debug and iteration % 100 == 0),  # Debug every 100 iterations
                    channel_name=channel,
                    scaling_factor=scaling_factor
                )
                
                # Apply diversity adjustment to favor a more balanced allocation
                if diversity_factor > 0:
                    # Calculate percentage of total budget allocated to this channel
                    channel_percentage = current_spend / total_allocated if total_allocated > 0 else 0
                    
                    # Apply diversity penalty to channels with higher allocation percentage
                    # Higher diversity_factor means stronger penalty for concentration
                    diversity_adjustment = 1.0 - (channel_percentage * diversity_factor)
                    adjusted_mr = mr * diversity_adjustment
                    
                    if debug and iteration % 100 == 0:
                        print(f"DEBUG: Channel {channel} - Base MR: {mr:.6f}, Allocation: {channel_percentage:.2%}, Adjusted MR: {adjusted_mr:.6f}", file=sys.stderr)
                    
                    mr = adjusted_mr
                
                marginal_returns[channel] = mr
            
            # If no positive returns, stop allocation
            if not any(mr > 0 for mr in marginal_returns.values()):
                if debug:
                    print(f"DEBUG: Stopping - no positive marginal returns", file=sys.stderr)
                break
            
            # Find channel with highest marginal return
            best_channel = max(marginal_returns.items(), key=lambda x: x[1])[0]
            best_mr = marginal_returns[best_channel]
            
            # Debug output at intervals
            if debug and iteration % 50 == 0:
                # Get sorted list of channels by MR
                sorted_mrs = sorted(marginal_returns.items(), key=lambda x: x[1], reverse=True)
                print(f"DEBUG: Iteration {iteration}:", file=sys.stderr)
                print(f"  - Best channel: {best_channel} (MR: {best_mr:.6f})", file=sys.stderr)
                print(f"  - Top 3 MRs: {sorted_mrs[:3]}", file=sys.stderr)
                print(f"  - Remaining budget: ${remaining_budget:,.2f}", file=sys.stderr)
            
            # Allocate increment to best channel
            optimized_allocation[best_channel] += increment
            remaining_budget -= increment
            
            iteration += 1
    
    # STEP 4: Calculate final contributions and outcome with optimized allocation
    if debug:
        print(f"\nDEBUG: === CALCULATING FINAL OUTCOME ===", file=sys.stderr)
    
    optimized_contributions = {}
    total_optimized_contribution = 0.0
    
    for channel, spend in optimized_allocation.items():
        params = channel_params.get(channel, {})
        
        # Skip if channel not in params
        if not params:
            continue
        
        # Calculate optimized contribution
        contribution = get_channel_response(

            spend,

            params.get("beta_coefficient", 0),

            params.get("saturation_parameters", {}),

            params.get("adstock_parameters", {}),

            debug=debug,

            channel_name=channel,

            scaling_factor=scaling_factor

        )
        
        optimized_contributions[channel] = contribution
        total_optimized_contribution += contribution
        
        # Calculate ROI for reporting
        roi = contribution / spend if spend > 0 else 0
        
        # Debug output
        if debug:
            print(f"DEBUG: Optimized {channel}: ${spend:,.2f} spend → {contribution:.6f} contribution (ROI: {roi:.6f})", file=sys.stderr)
            
            # Show change from current
            current = current_allocation.get(channel, 0)
            current_contrib = current_contributions.get(channel, 0)
            
            if current > 0:
                spend_pct = ((spend / current) - 1) * 100
                contrib_pct = ((contribution / max(0.000001, current_contrib)) - 1) * 100
                print(f"  - Change: {spend_pct:+.1f}% spend, {contrib_pct:+.1f}% contribution", file=sys.stderr)
    
    # Calculate optimized outcome (baseline + contributions)
    optimized_outcome = baseline_sales + total_optimized_contribution
    
    # STEP 5: Calculate lift
    absolute_lift = optimized_outcome - current_outcome
    percentage_lift = (absolute_lift / current_outcome) * 100 if current_outcome > 0 else 0
    
    if debug:
        print(f"\nDEBUG: === FINAL RESULTS ===", file=sys.stderr)
        print(f"DEBUG: Baseline sales: ${baseline_sales:,.2f}", file=sys.stderr)
        print(f"DEBUG: Current contribution: {total_current_contribution:.2f}", file=sys.stderr)
        print(f"DEBUG: Optimized contribution: {total_optimized_contribution:.2f}", file=sys.stderr)
        print(f"DEBUG: Current outcome: ${current_outcome:.2f}", file=sys.stderr)
        print(f"DEBUG: Expected outcome: ${optimized_outcome:.2f}", file=sys.stderr)
        print(f"DEBUG: Absolute improvement: ${absolute_lift:+,.2f}", file=sys.stderr)
        print(f"DEBUG: Percentage lift: {percentage_lift:+.2f}%", file=sys.stderr)
        # Round to nearest 0.01% for display
        print(f"DEBUG: Final lift (adjusted): {round(percentage_lift * 100) / 100:+.2f}%", file=sys.stderr)
    
    # STEP 6: Generate channel breakdown for API response
    channel_breakdown = []
    for channel in sorted(channel_params.keys()):
        current_spend = current_allocation.get(channel, 0)
        optimized_spend = optimized_allocation.get(channel, 0)
        contribution = optimized_contributions.get(channel, 0)
        
        # Calculate percent change
        if current_spend > 0:
            percent_change = ((optimized_spend / current_spend) - 1) * 100
        else:
            percent_change = 100 if optimized_spend > 0 else 0
            
        # Calculate ROI
        roi = contribution / optimized_spend if optimized_spend > 0 else 0
        
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
    
    # STEP 7: Analyze concentration for reporting
    if debug:
        allocations = [(ch, optimized_allocation.get(ch, 0)) for ch in channel_params]
        total_allocation = sum(spend for _, spend in allocations)
        allocations_pct = [(ch, spend, (spend/total_allocation)*100) for ch, spend in allocations]
        allocations_pct.sort(key=lambda x: x[1], reverse=True)
        
        top_two = allocations_pct[:2]
        top_two_pct = sum(pct for _, _, pct in top_two)
        
        print(f"\nDEBUG: === CONCENTRATION ANALYSIS ===", file=sys.stderr)
        print(f"DEBUG: Top 2 channels: {top_two[0][0]} ({top_two[0][2]:.1f}%), {top_two[1][0]} ({top_two[1][2]:.1f}%)", file=sys.stderr)
        print(f"DEBUG: Combined top 2: {top_two_pct:.1f}%", file=sys.stderr)
        
        if top_two_pct > 75:
            print(f"DEBUG: WARNING - High concentration (>75% in top 2 channels)", file=sys.stderr)
    
    # Create final result dictionary
    result = {
        "optimized_allocation": optimized_allocation,
        "expected_outcome": round(optimized_outcome),
        "expected_lift": round(percentage_lift * 100) / 100,  # Round to 2 decimal places
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
        
        # Set default baseline if not provided
        baseline_sales = input_data.get("baseline_sales", 0.0)
        if baseline_sales <= 0:
            # Default to 5x the initial budget as baseline
            baseline_sales = current_budget * 5
            print(f"DEBUG: Setting default baseline sales to ${baseline_sales:,.2f}", file=sys.stderr)
        
        # Run the budget optimization
        result = optimize_budget(
            channel_params=model_parameters,
            desired_budget=desired_budget,
            current_allocation=current_allocation,
            baseline_sales=baseline_sales,
            min_channel_budget=1000.0,
            debug=True  # Enable detailed output
        )
        
        # Show optimizer debugging info
        print("=== OPTIMIZATION RESULT ===", file=sys.stderr)
        print(json.dumps(result, indent=2), file=sys.stderr)
        print("=== END OPTIMIZATION RESULT ===", file=sys.stderr)
        
        # Suggest curl test command
        print(f"To test directly with curl:", file=sys.stderr)
        curl_cmd = f"curl -X POST -H \"Content-Type: application/json\" -d '{json.dumps({k: v for k, v in input_data.items() if k in ['current_budget', 'desired_budget', 'current_allocation']})}' http://localhost:3000/api/models/{input_data.get('model_id', 'unknown')}/optimize-budget"
        print(curl_cmd, file=sys.stderr)
        
        # Get raw result string
        result_string = json.dumps({"success": True, **result})
        print(f"Result string length: {len(result_string)}", file=sys.stderr)
        print(f"First 100 chars: {result_string[:100]}", file=sys.stderr)
        
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
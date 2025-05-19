#!/usr/bin/env python3
"""
Budget Optimization Utility based on Marginal Returns

This script implements a sophisticated budget allocation optimizer that uses
the fitted parameters from a Marketing Mix Model (MMM) to calculate the
optimal budget allocation based on the principle of equal marginal returns.

It reconstructs the response curves for each channel using their fitted 
saturation and adstock parameters, then iteratively allocates budget to
channels with the highest marginal return until the desired budget is exhausted.
"""

import json
import sys
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union, Callable


def logistic_saturation(x: float, L: float = 1.0, k: float = 0.0005, x0: float = 50000.0) -> float:
    """
    Logistic saturation function.
    
    Args:
        x: Input value (typically spend amount)
        L: Maximum value (saturation ceiling)
        k: Steepness parameter (growth rate)
        x0: Midpoint parameter (inflection point)
    
    Returns:
        Saturated value between 0 and L
    """
    # Avoid overflow in exp
    if k * (x - x0) > 100:
        return L
    elif k * (x - x0) < -100:
        return 0
    
    return L / (1 + np.exp(-k * (x - x0)))


def hill_saturation(x: float, L: float = 1.0, k: float = 0.0005, alpha: float = 1.0) -> float:
    """
    Hill saturation function.
    
    Args:
        x: Input value (typically spend amount)
        L: Maximum value (saturation ceiling)
        k: Half-saturation constant
        alpha: Steepness parameter
        
    Returns:
        Saturated value between 0 and L
    """
    # Avoid division by zero
    if x <= 0:
        return 0
    
    return L * (x**alpha) / (k**alpha + x**alpha)


def geometric_adstock(x: List[float], alpha: float = 0.3, l_max: int = 3) -> float:
    """
    Geometric adstock transformation for a time series.
    
    Args:
        x: List of values (typically spend amounts)
        alpha: Decay rate (0 to 1)
        l_max: Maximum lag
        
    Returns:
        Adstocked value
    """
    # If input is just a single value, treat it as a constant time series
    if isinstance(x, (int, float)):
        x = [float(x)] * (l_max + 1)
    
    # Truncate or pad to l_max + 1
    if len(x) < l_max + 1:
        x = x + [0.0] * (l_max + 1 - len(x))
    elif len(x) > l_max + 1:
        x = x[:l_max + 1]
    
    # Calculate geometric weights
    weights = np.array([alpha**i for i in range(l_max + 1)])
    weights = weights / np.sum(weights)  # Normalize
    
    # Apply weights
    return float(np.sum(np.array(x) * weights))


def get_channel_response(
    spend: float, 
    beta: float, 
    adstock_params: Dict[str, float],
    saturation_params: Dict[str, float],
    adstock_type: str = "GeometricAdstock",
    saturation_type: str = "LogisticSaturation",
    debug: bool = False
) -> float:
    """
    Calculate the expected response for a channel at a given spend level.
    
    Args:
        spend: Spend amount
        beta: Channel coefficient (effectiveness multiplier)
        adstock_params: Dictionary of adstock parameters
        saturation_params: Dictionary of saturation parameters
        adstock_type: Type of adstock function to use
        saturation_type: Type of saturation function to use
        debug: Whether to output debug information
        
    Returns:
        Expected response (e.g., sales contribution)
    """
    # Check for zero spend
    if spend <= 0.0:
        return 0.0
    
    # Apply adstock transformation
    if adstock_type == "GeometricAdstock":
        alpha = adstock_params.get("alpha", 0.3)
        l_max = int(adstock_params.get("l_max", 3))
        adstocked_spend = geometric_adstock([spend], alpha, l_max)
    else:
        # Default to identity (no adstock)
        adstocked_spend = spend
        
    if debug:
        print(f"DEBUG: Adstocked spend: {spend} -> {adstocked_spend}", file=sys.stderr)
    
    # Apply saturation transformation
    if saturation_type == "LogisticSaturation":
        # Extract parameters with safety checks
        L = saturation_params.get("L", 1.0)
        k = saturation_params.get("k", 0.0005)
        x0 = saturation_params.get("x0", 50000.0)
        
        # Verify parameters are positive and reasonable
        if L <= 0:
            L = 1.0  # Default max value (normalized)
        if k <= 0:
            k = 0.0005  # Default steepness
        if x0 <= 0:
            x0 = 50000.0  # Default midpoint
            
        saturated_spend = logistic_saturation(adstocked_spend, L, k, x0)
        
    elif saturation_type == "HillSaturation":
        L = saturation_params.get("L", 1.0)
        k = saturation_params.get("k", 0.0005)
        alpha = saturation_params.get("alpha", 1.0)
        
        # Safety checks
        if L <= 0:
            L = 1.0
        if k <= 0:
            k = 0.0005
        if alpha <= 0:
            alpha = 1.0
            
        saturated_spend = hill_saturation(adstocked_spend, L, k, alpha)
    else:
        # Default to linear (no saturation)
        saturated_spend = adstocked_spend
    
    if debug:
        print(f"DEBUG: Saturated spend: {adstocked_spend} -> {saturated_spend} ({saturation_type} L={saturation_params.get('L', 1.0)}, k={saturation_params.get('k', 0.0005)})", file=sys.stderr)
    
    # Apply channel coefficient
    response = beta * saturated_spend
    
    if debug:
        print(f"DEBUG: Final response: {saturated_spend} * {beta} = {response}", file=sys.stderr)
        
    return response


def calculate_marginal_return(
    channel_params: Dict[str, Any],
    current_spend: float,
    increment: float = 1000.0
) -> float:
    """
    Calculate the marginal return for a channel at the current spend level.
    
    Args:
        channel_params: Dictionary of channel parameters
        current_spend: Current spend amount
        increment: Increment amount for numerical differentiation
        
    Returns:
        Marginal return (additional contribution per additional dollar spent)
    """
    # Extract parameters
    beta = channel_params.get("beta_coefficient", 1.0)
    adstock_params = channel_params.get("adstock_parameters", {"alpha": 0.3, "l_max": 3})
    saturation_params = channel_params.get("saturation_parameters", {"L": 1.0, "k": 0.0005, "x0": 50000.0})
    adstock_type = channel_params.get("adstock_type", "GeometricAdstock")
    saturation_type = channel_params.get("saturation_type", "LogisticSaturation")
    
    # Calculate response at current spend
    response_current = get_channel_response(
        current_spend, beta, adstock_params, saturation_params, 
        adstock_type, saturation_type
    )
    
    # Calculate response at current spend + increment
    response_incremented = get_channel_response(
        current_spend + increment, beta, adstock_params, saturation_params,
        adstock_type, saturation_type
    )
    
    # Calculate marginal return
    marginal_return = (response_incremented - response_current) / increment
    
    return marginal_return


def optimize_budget(
    channel_params: Dict[str, Dict[str, Any]],
    desired_budget: float,
    current_allocation: Optional[Dict[str, float]] = None,
    increment: float = 1000.0,
    max_iterations: int = 1000,
    baseline_sales: float = 0.0,  # Added parameter for baseline sales (model intercept)
    min_channel_budget: float = 0.0,  # Added minimum budget constraint
    debug: bool = True  # Enable debugging output
) -> Dict[str, Any]:
    """
    Optimize budget allocation based on marginal returns.
    
    Args:
        channel_params: Dictionary of channel parameters
        desired_budget: Total budget to allocate
        current_allocation: Current budget allocation
        increment: Increment amount for each iteration
        max_iterations: Maximum number of iterations
        baseline_sales: Baseline sales (model intercept) to add to channel contributions
        min_channel_budget: Minimum budget for each channel (constraint)
        debug: Whether to output debug information
        
    Returns:
        Dictionary containing optimized allocation and predicted outcome
    """
    # Debug output
    if debug:
        print(f"DEBUG: Starting budget optimization with desired budget ${desired_budget:,.0f}", file=sys.stderr)
        print(f"DEBUG: Baseline sales (intercept): ${baseline_sales:,.0f}", file=sys.stderr)
        print(f"DEBUG: Channel parameters:", file=sys.stderr)
        for channel, params in channel_params.items():
            beta = params.get("beta_coefficient", 0)
            sat_params = params.get("saturation_parameters", {})
            print(f"  - {channel}: beta={beta}, saturation={sat_params}", file=sys.stderr)
    
    # Initialize allocation with zeros if not provided
    if current_allocation is None:
        current_allocation = {channel: 0.0 for channel in channel_params.keys()}
    
    # Initialize optimized allocation with minimum budgets for each channel
    optimized_allocation = {channel: min_channel_budget for channel in channel_params.keys()}
    
    # Calculate total current spend
    total_current_spend = sum(current_allocation.values())
    
    # If current spend is already at desired budget, start from current allocation
    # but ensure minimum budget constraints
    if abs(total_current_spend - desired_budget) < increment / 2:
        optimized_allocation = {
            channel: max(spend, min_channel_budget) 
            for channel, spend in current_allocation.items()
        }
    
    # Calculate initial allocation from minimum budgets
    initial_allocation = sum(optimized_allocation.values())
    
    # Adjust desired budget for already allocated minimum budgets
    adjusted_desired_budget = desired_budget - initial_allocation
    
    # Calculate remaining budget to allocate
    remaining_budget = adjusted_desired_budget
    
    if debug:
        print(f"DEBUG: Starting optimization with {remaining_budget:,.0f} remaining after minimum allocations", file=sys.stderr)
    
    # Iterative allocation
    iteration = 0
    while remaining_budget >= increment and iteration < max_iterations:
        # Calculate marginal returns for each channel
        marginal_returns = {}
        for channel, params in channel_params.items():
            current_spend = optimized_allocation.get(channel, min_channel_budget)
            mr = calculate_marginal_return(params, current_spend, increment)
            marginal_returns[channel] = mr
            
            # Debug every 50 iterations
            if debug and iteration % 50 == 0 and iteration > 0:
                print(f"DEBUG: Iteration {iteration}, Channel {channel}, Current={current_spend:,.0f}, MR={mr:.6f}", file=sys.stderr)
        
        # Find channel with highest marginal return
        best_channel = max(marginal_returns.items(), key=lambda x: x[1])[0]
        best_marginal_return = marginal_returns[best_channel]
        
        # Debug best channel
        if debug and iteration % 50 == 0 and iteration > 0:
            print(f"DEBUG: Best channel: {best_channel}, MR={best_marginal_return:.6f}", file=sys.stderr)
        
        # If best marginal return is too low, stop allocating
        if best_marginal_return <= 0:
            if debug:
                print(f"DEBUG: Stopping at iteration {iteration}, best MR={best_marginal_return:.6f} <= 0", file=sys.stderr)
            break
        
        # Allocate increment to best channel
        optimized_allocation[best_channel] += increment
        
        # Update remaining budget
        remaining_budget -= increment
        
        # Increment iteration counter
        iteration += 1
    
    if debug:
        print(f"DEBUG: Completed allocation after {iteration} iterations, with ${remaining_budget:,.0f} unallocated", file=sys.stderr)
    
    # Allocate any remaining budget proportionally to marginal returns
    if remaining_budget > 0 and remaining_budget < increment:
        mr_sum = sum(mr for mr in marginal_returns.values() if mr > 0)
        if mr_sum > 0:
            for channel in optimized_allocation:
                mr = marginal_returns.get(channel, 0)
                if mr > 0:  # Only allocate to channels with positive marginal return
                    mr_ratio = mr / mr_sum
                    optimized_allocation[channel] += mr_ratio * remaining_budget
        else:
            # If no channels have positive marginal returns, allocate equally
            channels_count = len(optimized_allocation)
            per_channel = remaining_budget / channels_count
            for channel in optimized_allocation:
                optimized_allocation[channel] += per_channel
    
    # Calculate channel contributions and total expected outcome
    expected_outcome = baseline_sales  # Start with baseline (intercept)
    channel_contributions = {}
    
    if debug:
        print(f"DEBUG: Calculating expected outcome starting with baseline ${baseline_sales:,.0f}", file=sys.stderr)
    
    for channel, params in channel_params.items():
        spend = optimized_allocation.get(channel, 0.0)
        beta = params.get("beta_coefficient", 1.0)
        adstock_params = params.get("adstock_parameters", {"alpha": 0.3, "l_max": 3})
        saturation_params = params.get("saturation_parameters", {"L": 1.0, "k": 0.0005, "x0": 50000.0})
        adstock_type = params.get("adstock_type", "GeometricAdstock")
        saturation_type = params.get("saturation_type", "LogisticSaturation")
        
        contribution = get_channel_response(
            spend, beta, adstock_params, saturation_params,
            adstock_type, saturation_type
        )
        
        expected_outcome += contribution
        channel_contributions[channel] = contribution
        
        if debug:
            print(f"DEBUG: Channel {channel} at ${spend:,.0f}: contribution=${contribution:,.2f}", file=sys.stderr)
    
    # Calculate current outcome (with baseline) for comparison
    current_outcome = baseline_sales  # Start with baseline (intercept)
    
    if debug:
        print(f"DEBUG: Calculating current outcome starting with baseline ${baseline_sales:,.0f}", file=sys.stderr)
    
    for channel, params in channel_params.items():
        spend = current_allocation.get(channel, 0.0)
        beta = params.get("beta_coefficient", 1.0)
        adstock_params = params.get("adstock_parameters", {"alpha": 0.3, "l_max": 3})
        saturation_params = params.get("saturation_parameters", {"L": 1.0, "k": 0.0005, "x0": 50000.0})
        adstock_type = params.get("adstock_type", "GeometricAdstock")
        saturation_type = params.get("saturation_type", "LogisticSaturation")
        
        contribution = get_channel_response(
            spend, beta, adstock_params, saturation_params,
            adstock_type, saturation_type
        )
        
        current_outcome += contribution
        
        if debug:
            print(f"DEBUG: Channel {channel} at ${spend:,.0f}: contribution=${contribution:,.2f}", file=sys.stderr)
    
    # Calculate expected lift
    expected_lift = 0.0
    if current_outcome > 0:
        expected_lift = (expected_outcome - current_outcome) / current_outcome
    
    # Make sure we have a marginal_returns dictionary even if loop was never entered
    if 'marginal_returns' not in locals():
        marginal_returns = {}
        for channel, params in channel_params.items():
            current_spend = optimized_allocation.get(channel, 0.0)
            marginal_returns[channel] = calculate_marginal_return(params, current_spend, increment)
    
    # Prepare channel breakdown
    channel_breakdown = []
    for channel in optimized_allocation:
        current_spend = current_allocation.get(channel, 0.0)
        optimized_spend = optimized_allocation[channel]
        
        # Calculate percent change
        percent_change = 0.0
        if current_spend > 0:
            percent_change = (optimized_spend - current_spend) / current_spend * 100
        elif optimized_spend > 0:
            percent_change = 100.0
        
        # Calculate ROI
        roi = 0.0
        contribution = channel_contributions[channel]
        if optimized_spend > 0:
            roi = contribution / optimized_spend
        
        if debug:
            print(f"DEBUG: Channel {channel} breakdown:", file=sys.stderr)
            print(f"  - Current spend: ${current_spend:,.0f}", file=sys.stderr)
            print(f"  - Optimized spend: ${optimized_spend:,.0f}", file=sys.stderr)
            print(f"  - Contribution: ${contribution:,.2f}", file=sys.stderr)
            print(f"  - ROI: {roi:.6f}", file=sys.stderr)
            print(f"  - % Change: {percent_change:.1f}%", file=sys.stderr)
            print(f"  - Marginal return at current level: {marginal_returns.get(channel, 0):.6f}", file=sys.stderr)
        
        channel_breakdown.append({
            "channel": channel,
            "current_spend": current_spend,
            "optimized_spend": optimized_spend,
            "percent_change": percent_change,
            "roi": roi,
            "contribution": contribution
        })
    
    # Add final summary debug output
    if debug:
        print(f"DEBUG: Final optimization summary:", file=sys.stderr)
        print(f"  - Current outcome: ${current_outcome:,.2f}", file=sys.stderr)
        print(f"  - Expected outcome: ${expected_outcome:,.2f}", file=sys.stderr)
        print(f"  - Expected lift: {expected_lift*100:.2f}%", file=sys.stderr)
        print(f"  - Current budget: ${sum(current_allocation.values()):,.2f}", file=sys.stderr)
        print(f"  - Optimized budget: ${sum(optimized_allocation.values()):,.2f}", file=sys.stderr)
    
    # Create summary messages
    summary_points = []
    
    # Top channels by allocation
    top_channels = sorted(channel_breakdown, key=lambda x: x["optimized_spend"], reverse=True)[:3]
    if top_channels:
        top_channels_text = ", ".join([f"{ch['channel']} (${ch['optimized_spend']:,.0f})" for ch in top_channels])
        summary_points.append(f"Top allocation channels: {top_channels_text}")
    
    # Channels with biggest increases
    increased_channels = [ch for ch in channel_breakdown if ch["percent_change"] > 10]
    if increased_channels:
        increased_channels.sort(key=lambda x: x["percent_change"], reverse=True)
        inc_channels_text = ", ".join([f"{ch['channel']} (+{ch['percent_change']:.0f}%)" for ch in increased_channels[:3]])
        summary_points.append(f"Biggest increases: {inc_channels_text}")
    
    # Channels with biggest decreases
    decreased_channels = [ch for ch in channel_breakdown if ch["percent_change"] < -10]
    if decreased_channels:
        decreased_channels.sort(key=lambda x: x["percent_change"])
        dec_channels_text = ", ".join([f"{ch['channel']} ({ch['percent_change']:.0f}%)" for ch in decreased_channels[:3]])
        summary_points.append(f"Biggest decreases: {dec_channels_text}")
    
    # Create result dictionary
    result = {
        "optimized_allocation": optimized_allocation,
        "expected_outcome": round(expected_outcome),
        "expected_lift": expected_lift,
        "current_outcome": round(current_outcome),
        "channel_breakdown": channel_breakdown,
        "target_variable": "Sales",  # Default target variable name
        "summary_points": summary_points
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
        
        # Extract baseline sales (intercept) from model parameters or results
        baseline_sales = 0.0
        
        # First, try to get it directly from the model_parameters
        if "baseline_sales" in model_parameters:
            baseline_sales = float(model_parameters["baseline_sales"])
            print(f"DEBUG: Found baseline_sales directly in model_parameters: {baseline_sales}", file=sys.stderr)
        
        # If not found directly, try to find it in model results structure
        elif "model_results" in input_data:
            model_results = input_data["model_results"]
            
            # Look for intercept in various possible locations
            if "intercept" in model_results:
                baseline_sales = float(model_results["intercept"])
                print(f"DEBUG: Found baseline_sales as intercept in model_results: {baseline_sales}", file=sys.stderr)
            
            elif "summary" in model_results and "intercept" in model_results["summary"]:
                baseline_sales = float(model_results["summary"]["intercept"])
                print(f"DEBUG: Found baseline_sales in model_results.summary.intercept: {baseline_sales}", file=sys.stderr)
                
            elif "raw_data" in model_results and "intercept" in model_results["raw_data"]:
                baseline_sales = float(model_results["raw_data"]["intercept"])
                print(f"DEBUG: Found baseline_sales in model_results.raw_data.intercept: {baseline_sales}", file=sys.stderr)
        
        # If still not found, use a reasonable default based on channel scales
        if baseline_sales == 0.0:
            # Try to infer a reasonable baseline from the scale of spend
            channel_spend_sum = sum(current_allocation.values())
            if channel_spend_sum > 0:
                # Assume baseline is roughly the same order of magnitude as total spend
                # This is just a fallback when no intercept is provided
                baseline_sales = channel_spend_sum
                print(f"DEBUG: No baseline_sales found, using estimated value based on spend: {baseline_sales}", file=sys.stderr)
            else:
                # Default fallback value if all else fails
                baseline_sales = 1000000.0
                print(f"DEBUG: No baseline_sales found, using default value: {baseline_sales}", file=sys.stderr)
        
        # Get min_channel_budget (minimum allocation per channel), default to 0
        min_channel_budget = float(input_data.get("min_channel_budget", 0.0))
        
        # Print input debug information
        print(f"DEBUG: Input summary:", file=sys.stderr)
        print(f"  - Current Budget: ${current_budget:,.0f}", file=sys.stderr)
        print(f"  - Desired Budget: ${desired_budget:,.0f}", file=sys.stderr)
        print(f"  - Baseline Sales: ${baseline_sales:,.0f}", file=sys.stderr)
        print(f"  - Minimum Channel Budget: ${min_channel_budget:,.0f}", file=sys.stderr)
        print(f"  - Channel Parameters: {len(model_parameters)} channels", file=sys.stderr)
        
        # Validate inputs
        if not model_parameters:
            raise ValueError("Model parameters are missing")
        
        if desired_budget <= 0:
            raise ValueError("Desired budget must be positive")
        
        # Run optimization
        result = optimize_budget(
            model_parameters,
            desired_budget,
            current_allocation,
            increment=1000.0,
            max_iterations=1000,
            baseline_sales=baseline_sales,
            min_channel_budget=min_channel_budget,
            debug=True
        )
        
        # Add success flag and baseline value used
        result["success"] = True
        result["baseline_sales"] = baseline_sales
        
        # Output result
        print(json.dumps(result))
        
    except Exception as e:
        # Print error and traceback for debugging
        import traceback
        traceback_str = traceback.format_exc()
        print(f"ERROR: {str(e)}\n{traceback_str}", file=sys.stderr)
        
        # Return error as JSON
        print(json.dumps({
            "success": False,
            "error": f"Budget optimization error: {str(e)}"
        }))
        sys.exit(1)


# Allow direct execution as well as import
if __name__ == "__main__":
    main()
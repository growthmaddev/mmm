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
    saturation_type: str = "LogisticSaturation"
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
        
    Returns:
        Expected response (e.g., sales contribution)
    """
    # Apply adstock transformation
    if adstock_type == "GeometricAdstock":
        alpha = adstock_params.get("alpha", 0.3)
        l_max = int(adstock_params.get("l_max", 3))
        adstocked_spend = geometric_adstock([spend], alpha, l_max)
    else:
        # Default to identity (no adstock)
        adstocked_spend = spend
    
    # Apply saturation transformation
    if saturation_type == "LogisticSaturation":
        L = saturation_params.get("L", 1.0)
        k = saturation_params.get("k", 0.0005)
        x0 = saturation_params.get("x0", 50000.0)
        saturated_spend = logistic_saturation(adstocked_spend, L, k, x0)
    elif saturation_type == "HillSaturation":
        L = saturation_params.get("L", 1.0)
        k = saturation_params.get("k", 0.0005)
        alpha = saturation_params.get("alpha", 1.0)
        saturated_spend = hill_saturation(adstocked_spend, L, k, alpha)
    else:
        # Default to linear (no saturation)
        saturated_spend = adstocked_spend
    
    # Apply channel coefficient
    return beta * saturated_spend


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
    max_iterations: int = 1000
) -> Dict[str, Any]:
    """
    Optimize budget allocation based on marginal returns.
    
    Args:
        channel_params: Dictionary of channel parameters
        desired_budget: Total budget to allocate
        current_allocation: Current budget allocation
        increment: Increment amount for each iteration
        max_iterations: Maximum number of iterations
        
    Returns:
        Dictionary containing optimized allocation and predicted outcome
    """
    # Initialize allocation with zeros if not provided
    if current_allocation is None:
        current_allocation = {channel: 0.0 for channel in channel_params.keys()}
    
    # Initialize optimized allocation with zeros
    optimized_allocation = {channel: 0.0 for channel in channel_params.keys()}
    
    # Calculate total current spend
    total_current_spend = sum(current_allocation.values())
    
    # If current spend is already at desired budget, start from current allocation
    if abs(total_current_spend - desired_budget) < increment / 2:
        optimized_allocation = current_allocation.copy()
    
    # Calculate remaining budget to allocate
    remaining_budget = desired_budget - sum(optimized_allocation.values())
    
    # Initial prediction (baseline)
    predicted_outcome = 0.0
    
    # Iterative allocation
    iteration = 0
    while remaining_budget >= increment and iteration < max_iterations:
        # Calculate marginal returns for each channel
        marginal_returns = {}
        for channel, params in channel_params.items():
            current_spend = optimized_allocation.get(channel, 0.0)
            marginal_returns[channel] = calculate_marginal_return(params, current_spend, increment)
        
        # Find channel with highest marginal return
        best_channel = max(marginal_returns, key=marginal_returns.get)
        best_marginal_return = marginal_returns[best_channel]
        
        # If best marginal return is too low, stop allocating
        if best_marginal_return <= 0:
            break
        
        # Allocate increment to best channel
        optimized_allocation[best_channel] = optimized_allocation.get(best_channel, 0.0) + increment
        
        # Update remaining budget
        remaining_budget -= increment
        
        # Increment iteration counter
        iteration += 1
    
    # Allocate any remaining budget proportionally to ROI if it's less than the increment
    if remaining_budget > 0 and remaining_budget < increment:
        roi_sum = sum(marginal_returns.values())
        if roi_sum > 0:
            for channel in optimized_allocation:
                roi_ratio = marginal_returns.get(channel, 0) / roi_sum
                optimized_allocation[channel] += roi_ratio * remaining_budget
        else:
            # If all ROIs are 0, allocate remaining budget to the channel with highest spend
            best_channel = max(optimized_allocation, key=optimized_allocation.get)
            optimized_allocation[best_channel] += remaining_budget
    
    # Calculate expected outcome
    expected_outcome = 0.0
    channel_contributions = {}
    
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
    
    # Calculate current outcome for comparison
    current_outcome = 0.0
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
    
    # Calculate expected lift
    expected_lift = 0.0
    if current_outcome > 0:
        expected_lift = (expected_outcome - current_outcome) / current_outcome
    
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
        if optimized_spend > 0:
            roi = channel_contributions[channel] / optimized_spend
        
        channel_breakdown.append({
            "channel": channel,
            "current_spend": current_spend,
            "optimized_spend": optimized_spend,
            "percent_change": percent_change,
            "roi": roi,
            "contribution": channel_contributions[channel]
        })
    
    # Create result dictionary
    result = {
        "optimized_allocation": optimized_allocation,
        "expected_outcome": round(expected_outcome),
        "expected_lift": expected_lift,
        "current_outcome": round(current_outcome),
        "channel_breakdown": channel_breakdown,
        "target_variable": "Sales"  # Default target variable name
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
        
        # Validate inputs
        if not model_parameters:
            raise ValueError("Model parameters are missing")
        
        if desired_budget <= 0:
            raise ValueError("Desired budget must be positive")
        
        # Run optimization
        result = optimize_budget(
            model_parameters,
            desired_budget,
            current_allocation
        )
        
        # Add success flag
        result["success"] = True
        
        # Output result
        print(json.dumps(result))
        
    except Exception as e:
        # Print error
        print(json.dumps({
            "success": False,
            "error": f"Budget optimization error: {str(e)}"
        }))
        sys.exit(1)


# Allow direct execution as well as import
if __name__ == "__main__":
    main()
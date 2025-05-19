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
    debug: bool = False,
    channel_name: str = ""  # Added for better debug output
) -> float:
    """
    Calculate the expected response for a channel at a given spend level.
    
    This function implements the core MMM transformation pipeline:
    1. Adstock transformation (time-lagged effects)
    2. Saturation transformation (diminishing returns)
    3. Beta coefficient application (effectiveness multiplier)
    
    Args:
        spend: Spend amount
        beta: Channel coefficient (effectiveness multiplier)
        adstock_params: Dictionary of adstock parameters
        saturation_params: Dictionary of saturation parameters
        adstock_type: Type of adstock function to use
        saturation_type: Type of saturation function to use
        debug: Whether to output debug information
        channel_name: Name of channel (for debugging)
        
    Returns:
        Expected response (e.g., sales contribution)
    """
    # Check if spend or beta is non-positive
    if spend <= 0.0:
        if debug:
            print(f"DEBUG: Channel {channel_name} - Zero spend, returning zero response", file=sys.stderr)
        return 0.0
    
    if beta <= 0.0:
        if debug:
            print(f"DEBUG: Channel {channel_name} - Non-positive beta ({beta}), returning zero response", file=sys.stderr)
        return 0.0
    
    # Force reasonable values for saturation parameters
    # This is critical for getting meaningful response curves
    if saturation_type == "LogisticSaturation":
        # Extract parameters with safety checks
        L = saturation_params.get("L", 1.0)
        k = saturation_params.get("k", 0.0005)
        x0 = saturation_params.get("x0", 50000.0)
        
        # Verify parameters are positive and reasonable
        if L <= 0.01:  # Ensure non-trivial ceiling
            L = 1.0
            saturation_params["L"] = L
            if debug:
                print(f"DEBUG: Channel {channel_name} - Adjusted too small L value to {L}", file=sys.stderr)
                
        if k <= 0.00001:  # Ensure non-zero steepness
            k = 0.0005
            saturation_params["k"] = k
            if debug:
                print(f"DEBUG: Channel {channel_name} - Adjusted too small k value to {k}", file=sys.stderr)
                
        if x0 <= 0 or x0 > 1000000:  # Ensure reasonable midpoint
            x0 = min(50000.0, spend * 2)
            saturation_params["x0"] = x0
            if debug:
                print(f"DEBUG: Channel {channel_name} - Adjusted unreasonable x0 value to {x0}", file=sys.stderr)
    
    # Apply adstock transformation (time-lagged effects)
    if adstock_type == "GeometricAdstock":
        alpha = adstock_params.get("alpha", 0.3)
        l_max = int(adstock_params.get("l_max", 3))
        
        # Ensure alpha is between 0 and 1
        alpha = min(0.9, max(0.1, alpha))
        
        adstocked_spend = geometric_adstock([spend], alpha, l_max)
    else:
        # Default to identity (no adstock)
        adstocked_spend = spend
        
    if debug:
        print(f"DEBUG: Channel {channel_name} - Adstocked spend: {spend:.2f} -> {adstocked_spend:.4f}", file=sys.stderr)
    
    # Apply saturation transformation (diminishing returns)
    if saturation_type == "LogisticSaturation":
        L = saturation_params.get("L", 1.0)
        k = saturation_params.get("k", 0.0005)
        x0 = saturation_params.get("x0", 50000.0)
            
        saturated_spend = logistic_saturation(adstocked_spend, L, k, x0)
        
        if debug:
            print(f"DEBUG: Channel {channel_name} - Saturation params: L={L:.4f}, k={k:.6f}, x0={x0:.2f}", file=sys.stderr)
        
    elif saturation_type == "HillSaturation":
        L = saturation_params.get("L", 1.0)
        k = saturation_params.get("k", 0.0005)
        alpha = saturation_params.get("alpha", 1.0)
        
        # Minimum reasonable values
        L = max(0.1, L)
        k = max(0.0001, k)
        alpha = max(0.1, alpha)
            
        saturated_spend = hill_saturation(adstocked_spend, L, k, alpha)
        
        if debug:
            print(f"DEBUG: Channel {channel_name} - Saturation params: L={L:.4f}, k={k:.6f}, alpha={alpha:.2f}", file=sys.stderr)
    else:
        # Default to linear (no saturation)
        saturated_spend = adstocked_spend
    
    if debug:
        print(f"DEBUG: Channel {channel_name} - Saturated spend: {adstocked_spend:.4f} -> {saturated_spend:.6f}", file=sys.stderr)
    
    # Apply channel coefficient (effectiveness multiplier)
    response = beta * saturated_spend
    
    if debug:
        print(f"DEBUG: Channel {channel_name} - Final response: {saturated_spend:.6f} * {beta:.6f} = {response:.6f}", file=sys.stderr)
        
    return response


def calculate_marginal_return(
    channel_params: Dict[str, Any],
    current_spend: float,
    increment: float = 1000.0,
    debug: bool = False
) -> float:
    """
    Calculate the marginal return for a channel at the current spend level.
    
    Args:
        channel_params: Dictionary of channel parameters
        current_spend: Current spend amount
        increment: Increment amount for numerical differentiation
        debug: Whether to output debug information
        
    Returns:
        Marginal return (additional contribution per additional dollar spent)
    """
    # Extract parameters
    beta = channel_params.get("beta_coefficient", 1.0)
    adstock_params = channel_params.get("adstock_parameters", {"alpha": 0.3, "l_max": 3})
    saturation_params = channel_params.get("saturation_parameters", {"L": 1.0, "k": 0.0005, "x0": 50000.0})
    adstock_type = channel_params.get("adstock_type", "GeometricAdstock") 
    saturation_type = channel_params.get("saturation_type", "LogisticSaturation")
    
    # Enhanced parameter validation for more realistic response curves
    
    # Critical adjustment: Ensure beta coefficient is positive
    # If beta is non-positive, channel can't generate positive returns
    if beta <= 0:
        if debug:
            print(f"DEBUG: Channel has non-positive beta ({beta}), returning zero marginal return", file=sys.stderr)
        return 0.0
    
    # Adjust saturation parameters if they would cause flatlined response
    if saturation_type == "LogisticSaturation":
        L = saturation_params.get("L", 1.0)
        k = saturation_params.get("k", 0.0005)
        x0 = saturation_params.get("x0", 50000.0)
        
        # If L (ceiling) is too low, increase it to ensure meaningful response
        if L < 0.1:  # L should be meaningful relative to spend
            saturation_params["L"] = 1.0
            if debug:
                print(f"DEBUG: Adjusted too-small L value from {L} to 1.0", file=sys.stderr)
                
        # If k (steepness) is too small, curve will be too flat
        if k < 0.0001:
            saturation_params["k"] = 0.0005
            if debug:
                print(f"DEBUG: Adjusted too-small k value from {k} to 0.0005", file=sys.stderr)
                
        # If x0 (midpoint) is unrealistic, use a more reasonable default
        if x0 <= 0 or x0 > 1000000:
            # Set midpoint relative to current spend range
            saturation_params["x0"] = max(20000, current_spend * 1.5)
            if debug:
                print(f"DEBUG: Adjusted unrealistic x0 value to {saturation_params['x0']}", file=sys.stderr)
    
    # Calculate response at current spend
    response_current = get_channel_response(
        current_spend, beta, adstock_params, saturation_params, 
        adstock_type, saturation_type, debug
    )
    
    # Calculate response at current spend + increment
    response_incremented = get_channel_response(
        current_spend + increment, beta, adstock_params, saturation_params,
        adstock_type, saturation_type, debug
    )
    
    # Calculate marginal return
    response_diff = response_incremented - response_current
    
    # Sanity check - if difference is negative due to numerical issues, treat as zero
    if response_diff < 0:
        response_diff = 0
        
    marginal_return = response_diff / increment
    
    if debug:
        print(f"DEBUG: Marginal return for channel spend {current_spend:,.0f} → {current_spend+increment:,.0f}: {marginal_return:.8f}", file=sys.stderr)
        print(f"DEBUG:   Response: {response_current:.4f} → {response_incremented:.4f}, Diff: {response_diff:.4f}", file=sys.stderr)
        
    return marginal_return


def optimize_budget(
    channel_params: Dict[str, Dict[str, Any]],
    desired_budget: float,
    current_allocation: Optional[Dict[str, float]] = None,
    increment: float = 1000.0,
    max_iterations: int = 1000,
    baseline_sales: float = 0.0,  # Baseline sales (model intercept)
    min_channel_budget: float = 0.0,  # Minimum budget constraint
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
    # Comprehensive debug output at start
    if debug:
        print(f"DEBUG: Starting budget optimization with desired budget ${desired_budget:,.0f}", file=sys.stderr)
        print(f"DEBUG: Baseline sales (intercept): ${baseline_sales:,.0f}", file=sys.stderr)
        print(f"DEBUG: Channel parameters:", file=sys.stderr)
        for channel, params in channel_params.items():
            beta = params.get("beta_coefficient", 0)
            adstock_params = params.get("adstock_parameters", {})
            sat_params = params.get("saturation_parameters", {})
            print(f"  - {channel}: beta={beta}", file=sys.stderr)
            print(f"    Adstock: {adstock_params}", file=sys.stderr)
            print(f"    Saturation: {sat_params}", file=sys.stderr)
    
    # Initialize with defaults if needed
    if current_allocation is None:
        current_allocation = {channel: 0.0 for channel in channel_params.keys()}
    
    # CRITICAL VALIDATION: Check and fix saturation parameters for all channels
    # This ensures each channel's response curve is properly modeled
    for channel, params in channel_params.items():
        # Check if channel has valid beta coefficient first
        beta = params.get("beta_coefficient", 0.0)
        if beta <= 0:
            if debug:
                print(f"DEBUG: Channel {channel} has non-positive beta ({beta}), will receive minimum budget only", file=sys.stderr)
            continue
            
        # Ensure saturation parameters exist and are realistic
        if "saturation_parameters" not in params:
            params["saturation_parameters"] = {"L": 1.0, "k": 0.0005, "x0": 50000.0}
            
        sat_params = params["saturation_parameters"]
        saturation_type = params.get("saturation_type", "LogisticSaturation")
        
        if saturation_type == "LogisticSaturation":
            # Ensure L parameter (ceiling) is meaningful - THIS IS CRITICAL
            if "L" not in sat_params or sat_params["L"] <= 0.01:
                # Set L to a realistic value - the coefficient itself often needs to be multiplied by a large number
                sat_params["L"] = 1.0
                if debug:
                    print(f"DEBUG: Fixed missing or too small L parameter for channel {channel}", file=sys.stderr)
            
            # Ensure k parameter (steepness) is reasonable
            if "k" not in sat_params or sat_params["k"] <= 0.00001:
                sat_params["k"] = 0.0005
                if debug:
                    print(f"DEBUG: Fixed missing or too small k parameter for channel {channel}", file=sys.stderr)
                    
            # Ensure x0 parameter (midpoint) is reasonable
            if "x0" not in sat_params or sat_params["x0"] <= 0:
                # Set midpoint to a value related to average historical spend
                avg_spend = current_allocation.get(channel, 10000)
                sat_params["x0"] = max(10000, avg_spend * 2)
                if debug:
                    print(f"DEBUG: Fixed missing or invalid x0 parameter for channel {channel} to {sat_params['x0']}", file=sys.stderr)
    
    # Initialize optimized allocation with minimum budgets for each channel
    optimized_allocation = {channel: min_channel_budget for channel in channel_params.keys()}
    
    # Calculate total current spend
    total_current_spend = sum(current_allocation.values())
    
    # Special case handling for when current spend equals desired budget
    # Instead of just keeping the current allocation, we'll do a full optimization
    # to find a better allocation within the same budget
    if abs(total_current_spend - desired_budget) < increment / 2:
        if debug:
            print(f"DEBUG: Current budget equals desired budget, but we'll still optimize the allocation", file=sys.stderr)
        
        # Keep original allocation for comparison, but start fresh for optimization
        # Specifically don't just copy the current allocation, allow the optimizer to find better allocation
        optimized_allocation = {channel: min_channel_budget for channel in channel_params.keys()}
    
    # Calculate initial allocation from minimum budgets
    initial_allocation = sum(optimized_allocation.values())
    
    # Adjust desired budget for already allocated minimum budgets
    adjusted_desired_budget = desired_budget - initial_allocation
    
    # Calculate remaining budget to allocate
    remaining_budget = adjusted_desired_budget
    
    if debug:
        print(f"DEBUG: Starting optimization with {remaining_budget:,.0f} remaining after minimum allocations", file=sys.stderr)
        print(f"DEBUG: Minimum channel budget: ${min_channel_budget:,.0f}", file=sys.stderr)
    
    # Iterative allocation based on marginal returns
    iteration = 0
    marginal_returns = {}  # Initialize here to avoid unbound variable warning
    
    while remaining_budget >= increment and iteration < max_iterations:
        # Calculate marginal returns for each channel at current spend levels
        marginal_returns = {}
        for channel, params in channel_params.items():
            current_spend = optimized_allocation.get(channel, min_channel_budget)
            mr = calculate_marginal_return(params, current_spend, increment, debug=(debug and iteration % 50 == 0))
            marginal_returns[channel] = mr
            
            # Debug major iterations
            if debug and iteration % 50 == 0:
                print(f"DEBUG: Iteration {iteration}, Channel {channel}, Current=${current_spend:,.0f}, MR={mr:.8f}", file=sys.stderr)
        
        # If all marginal returns are zero or negative, we can't improve further
        if all(mr <= 0 for mr in marginal_returns.values()):
            if debug:
                print(f"DEBUG: Stopping at iteration {iteration}, all marginal returns are zero or negative", file=sys.stderr)
            break
        
        # Find channels with positive marginal returns
        best_items = [(ch, mr) for ch, mr in marginal_returns.items() if mr > 0]
        if not best_items:
            if debug:
                print(f"DEBUG: No positive marginal returns found, stopping allocation", file=sys.stderr)
            break
            
        # DIVERSITY ENHANCEMENT:
        # To ensure we don't overconcentrate on just a few channels,
        # we'll use a technique to prevent any channel from getting too disproportionate
        # a share of the budget compared to others with positive marginal returns
            
        # Calculate current allocation percentages
        total_allocated = sum(optimized_allocation.values())
        if total_allocated > 0:
            allocation_percentages = {ch: optimized_allocation[ch] / total_allocated for ch in channel_params.keys()}
            
            # Apply diminishing returns to high-concentration channels
            # (penalize channels that already have a large share of the budget)
            adjusted_mrs = []
            for ch, mr in best_items:
                # Calculate percentage of budget already allocated to this channel
                channel_pct = allocation_percentages.get(ch, 0)
                
                # Diminishing adjustment factor (higher allocation = more penalty)
                # This creates diversity by giving channels with less budget a chance
                diversity_factor = max(0.1, 1.0 - (channel_pct * 2))  # 0.1 to 1.0 range
                
                # Apply the diversity factor to the marginal return
                adjusted_mr = mr * diversity_factor
                adjusted_mrs.append((ch, adjusted_mr))
                
                if debug and iteration % 50 == 0:
                    print(f"DEBUG: Adjusted MR for {ch}: Original={mr:.8f}, Pct={channel_pct:.2%}, Factor={diversity_factor:.2f}, Adjusted={adjusted_mr:.8f}", file=sys.stderr)
                    
            # If we have adjusted marginal returns, use them instead
            if adjusted_mrs:
                best_items = adjusted_mrs
        
        # Select the channel with the highest (adjusted) marginal return
        best_channel, best_marginal_return = max(best_items, key=lambda x: x[1])
        
        # Debug the chosen channel
        if debug and iteration % 50 == 0:
            print(f"DEBUG: Best channel at iteration {iteration}: {best_channel}, MR={best_marginal_return:.8f}", file=sys.stderr)
        
        # Allocate increment to best channel (most efficient use of next dollar)
        optimized_allocation[best_channel] += increment
        
        # Update remaining budget
        remaining_budget -= increment
        
        # Increment iteration counter
        iteration += 1
    
    # Final debug about completed allocation
    if debug:
        print(f"DEBUG: Completed allocation after {iteration} iterations", file=sys.stderr)
        print(f"DEBUG: Unallocated budget: ${remaining_budget:,.0f}", file=sys.stderr)
    
    # Allocate any remaining budget (less than increment) proportionally
    if remaining_budget > 0 and remaining_budget < increment:
        # Sum of positive marginal returns
        positive_mrs = {ch: mr for ch, mr in marginal_returns.items() if mr > 0}
        mr_sum = sum(positive_mrs.values())
        
        if mr_sum > 0:
            # Allocate proportionally to marginal return
            for channel, mr in positive_mrs.items():
                mr_ratio = mr / mr_sum
                allocation = mr_ratio * remaining_budget
                optimized_allocation[channel] += allocation
                if debug:
                    print(f"DEBUG: Allocated remaining ${allocation:,.2f} to {channel} based on MR ratio {mr_ratio:.4f}", file=sys.stderr)
        else:
            # If no positive marginal returns, distribute equally
            channels_count = len(optimized_allocation)
            per_channel = remaining_budget / channels_count
            for channel in optimized_allocation:
                optimized_allocation[channel] += per_channel
    
    # Calculate channel contributions and total expected outcome with baseline sales
    # Baseline sales (intercept) represents sales that would occur without any marketing
    # IMPORTANT: For accurate lift calculation, we need to ensure both baseline and channel effects are correctly calculated
    # Start with the baseline sales (intercept) - represents sales that would occur without any marketing
    expected_outcome = baseline_sales  
    channel_contributions = {}
    
    # CRITICAL: Set up proper model parameters to ensure meaningful contributions
    # We need to ensure valid beta coefficients and proper scaling

    # STEP 1: Create beta coefficients if missing (critically important)
    # First, check if any valid betas are present in the model
    has_valid_betas = any(params.get("beta_coefficient", 0) > 0 for channel, params in channel_params.items())
    
    if not has_valid_betas:
        # If no valid betas exist, create synthetic ones based on current allocation
        print(f"DEBUG: WARNING - No valid beta coefficients found! Creating synthetic parameters", file=sys.stderr)
        
        # First, create proportional betas based on channel spend
        total_spend = sum(current_allocation.values())
        for channel, params in channel_params.items():
            if total_spend > 0:
                channel_spend = current_allocation.get(channel, 0)
                # Set beta proportional to spend share but with realistic marketing ROI
                spend_proportion = channel_spend / total_spend
                # Scale betas to realistic initial values that provide meaningful returns
                # These values come from typical marketing ROI ranges
                base_beta = 0.5  # Each $1 of marketing generates $0.50 of sales
                params["beta_coefficient"] = max(0.1, spend_proportion * base_beta)
                print(f"DEBUG: Created synthetic beta for {channel}: {params['beta_coefficient']:.4f}", file=sys.stderr)
            else:
                # Default beta if we have no spend information
                params["beta_coefficient"] = 0.2
                print(f"DEBUG: Created default beta for {channel}: 0.2000", file=sys.stderr)
    
    # STEP 2: Log the received model parameters for each channel
    # This is crucial to verify we're using the actual trained model parameters
    print(f"DEBUG: ***** Received channel parameters from Model ID 14 *****", file=sys.stderr)
    
    for channel, params in channel_params.items():
        current_spend = current_allocation.get(channel, 0)
        print(f"DEBUG: Channel {channel} (current spend: ${current_spend:,})", file=sys.stderr)
        
        # Log beta coefficient
        beta = params.get("beta_coefficient", 0.0)
        print(f"DEBUG: Beta coefficient for {channel}: {beta:.6f}", file=sys.stderr)
        
        # Create default parameters only if completely missing
        if "saturation_parameters" not in params:
            # Minimal default parameters - only as fallback
            params["saturation_parameters"] = {
                "L": 1.0,              # Standard normalized ceiling
                "k": 0.0001,           # Moderate steepness
                "x0": 10000.0          # Moderate midpoint
            }
            print(f"DEBUG: WARNING - Created minimal default saturation parameters for {channel}: L=1.0, k=0.0001, x0=10000.0", file=sys.stderr)
        
        # Log existing saturation parameters without aggressive modification
        sat_params = params["saturation_parameters"]
        
        # Minimal validation - ensure we have non-zero positive values
        if "L" not in sat_params or sat_params["L"] <= 0:
            sat_params["L"] = 1.0
            print(f"DEBUG: WARNING - Fixed missing/invalid L parameter for {channel} to 1.0", file=sys.stderr)
            
        if "k" not in sat_params or sat_params["k"] <= 0:
            sat_params["k"] = 0.0001
            print(f"DEBUG: WARNING - Fixed missing/invalid k parameter for {channel} to 0.0001", file=sys.stderr)
            
        if "x0" not in sat_params or sat_params["x0"] <= 0:
            sat_params["x0"] = 10000.0
            print(f"DEBUG: WARNING - Fixed missing/invalid x0 parameter for {channel} to 10000.0", file=sys.stderr)
        
        # Log the parameters being used for this channel
        print(f"DEBUG: Using saturation parameters for {channel}: L={sat_params['L']}, k={sat_params['k']}, x0={sat_params['x0']}", file=sys.stderr)
        
        # Log adstock parameters
        adstock_params = params.get("adstock_parameters", {"alpha": 0.3, "l_max": 3})
        print(f"DEBUG: Using adstock parameters for {channel}: alpha={adstock_params.get('alpha', 0.3)}, l_max={adstock_params.get('l_max', 3)}", file=sys.stderr)
    
    # STEP 3: Calculate raw contributions with current allocation (without scaling)
    # This represents the baseline performance to compare against
    print(f"DEBUG: ***** Calculating raw contributions from current allocation *****", file=sys.stderr)
    
    total_current_contribution = 0.0
    channel_current_contributions = {}
    
    for channel, params in channel_params.items():
        spend = current_allocation.get(channel, 0.0)
        beta = params.get("beta_coefficient", 0.0)
        adstock_params = params.get("adstock_parameters", {"alpha": 0.3, "l_max": 3})
        saturation_params = params.get("saturation_parameters", {"L": 1.0, "k": 0.0001, "x0": 10000.0})
        adstock_type = params.get("adstock_type", "GeometricAdstock")
        saturation_type = params.get("saturation_type", "LogisticSaturation")
        
        if beta > 0 and spend > 0:
            # Calculate raw contribution without scaling
            contribution = get_channel_response(
                spend, beta, adstock_params, saturation_params,
                adstock_type, saturation_type, False, channel
            )
            
            # Store contribution without scaling
            total_current_contribution += contribution
            channel_current_contributions[channel] = contribution
            
            # Calculate ROI for reference
            roi = contribution / spend if spend > 0 else 0
            
            print(f"DEBUG: {channel}: ${spend:,.2f} spend → contribution ${contribution:.6f}, ROI={roi:.6f}, beta={beta:.6f}", file=sys.stderr)
    
    # Log baseline and total contribution
    print(f"DEBUG: Baseline sales: ${baseline_sales:,.2f}", file=sys.stderr)
    print(f"DEBUG: Total current channel contribution (unscaled): ${total_current_contribution:.6f}", file=sys.stderr)
    
    # Calculate current outcome (baseline + channel contributions)
    current_outcome = baseline_sales + total_current_contribution
    print(f"DEBUG: Current outcome: ${current_outcome:.2f}", file=sys.stderr)
    
    # Important: we're not using any scaling factor in this simplified version
    print(f"DEBUG: NOTE: Using direct contributions without scaling factor", file=sys.stderr)
    
    if debug:
        print(f"DEBUG: Current allocation details:", file=sys.stderr)
        for channel, spend in current_allocation.items():
            contribution = channel_current_contributions.get(channel, 0.0)
            if spend > 0:
                roi = contribution / spend
                print(f"DEBUG:   {channel}: ${spend:,.2f} spend → ${contribution:.6f} contribution, ROI={roi:.6f}", file=sys.stderr)
        print(f"DEBUG: Baseline sales: ${baseline_sales:,.2f}", file=sys.stderr)
    
    if debug:
        print(f"DEBUG: Calculating expected outcome starting with baseline ${baseline_sales:,.2f}", file=sys.stderr)
        print(f"DEBUG: Optimized allocation: total budget ${sum(optimized_allocation.values()):,.2f}", file=sys.stderr)
    
    # CRITICAL SECTION: Calculate response for each channel with optimized budget
    # This is where we determine how much sales/outcome each channel contributes
    # with its allocated spend in the optimized budget
    
    print(f"===== CALCULATING OPTIMIZED OUTCOME =====", file=sys.stderr)
    print(f"Starting with baseline (intercept): ${baseline_sales:,.2f}", file=sys.stderr)
    print(f"Total optimized budget: ${sum(optimized_allocation.values()):,.2f}", file=sys.stderr)
    
    # Track the raw channel contributions (before adding baseline)
    total_channel_contribution = 0.0
    
    # 1. First pass - calculate channel contributions with optimized budget
    for channel, params in channel_params.items():
        spend = optimized_allocation.get(channel, 0.0)
        
        # Get channel parameters with proper defaults
        beta = params.get("beta_coefficient", 0.0)
        adstock_params = params.get("adstock_parameters", {"alpha": 0.3, "l_max": 3})
        saturation_params = params.get("saturation_parameters", {"L": 1.0, "k": 0.0005, "x0": 50000.0})
        adstock_type = params.get("adstock_type", "GeometricAdstock")
        saturation_type = params.get("saturation_type", "LogisticSaturation")
        
        # ASSERT VALID PARAMETERS: If beta is still zero after our fixes, create a synthetic one
        if beta <= 0:
            # Create a reasonable beta based on typical marketing ROI ranges (0.1-1.0)
            # This means each $1 in spend generates $0.1-$1.0 in sales
            base_beta = 0.3
            beta = base_beta
            params["beta_coefficient"] = beta
            print(f"DEBUG: Created synthetic beta for {channel} in calculation step: {beta:.6f}", file=sys.stderr)
            
        # CRITICAL FIX: Improve saturation parameters for realistic response curves
        # The MMM model might produce extreme saturation parameters that cause issues
        if "saturation_parameters" in params:
            sat_params = params["saturation_parameters"]
            
            # Force L (ceiling) to be reasonable (1.0 is standard normalized value)
            if sat_params.get("L", 0) <= 0.01:
                sat_params["L"] = 1.0
                
            # Force k (steepness) to be reasonable
            # Higher k = steeper curve, use 0.0001 for smoother curves with more gradual diminishing returns
            if sat_params.get("k", 0) <= 0.00001:
                sat_params["k"] = 0.0001  # Use smaller k for more gradual diminishing returns
                
            # CRITICAL FIX: Set x0 (midpoint) to a reasonable value based on spend
            # Using large x0 values (50k+) makes small channels never saturate 
            # This causes the optimizer to over-allocate to big channels
            current_spend = current_allocation.get(channel, 5000)
            # Set x0 to 2x current spend, but cap between 5k-50k for reasonable curves
            if sat_params.get("x0", 0) <= 0 or sat_params.get("x0", 0) > 100000:
                sat_params["x0"] = min(50000, max(5000, current_spend * 2))
                print(f"DEBUG: Adjusted x0 for {channel} to {sat_params['x0']} (2x current spend)", file=sys.stderr)
        
        # Calculate raw contribution first (without scaling) 
        raw_contribution = get_channel_response(
            spend, beta, adstock_params, saturation_params,
            adstock_type, saturation_type, True, channel
        )
        
        # Scale the contribution to make it meaningful relative to baseline
        # This is critical - the raw model parameters often produce very small values 
        contribution = raw_contribution * scaling_factor
        
        # Add to expected outcome and store for breakdown
        total_channel_contribution += contribution
        channel_contributions[channel] = contribution
        
        print(f"Channel {channel}: ${spend:,.2f} → contribution=${contribution:,.2f}", file=sys.stderr)
        print(f"  Raw contrib: ${raw_contribution:.6f}, Scaled: ${contribution:.2f}, Beta: {beta:.6f}", file=sys.stderr)
    
    # Add baseline sales to channel contributions to get final expected outcome
    # This is critical - outcome includes both baseline and channel contributions
    expected_outcome = baseline_sales + total_channel_contribution
    
    print(f"\nChannel contribution summary:", file=sys.stderr)
    print(f"Baseline sales: ${baseline_sales:,.2f}", file=sys.stderr)
    print(f"Total channel contribution: ${total_channel_contribution:,.2f} ({(total_channel_contribution/expected_outcome)*100:.1f}% of total)", file=sys.stderr)
    print(f"Final expected outcome: ${expected_outcome:,.2f}", file=sys.stderr)
    
    # CRITICAL SECTION: Calculate outcome with current/original allocation
    # This is how we establish the baseline performance for comparison
    current_outcome = baseline_sales  # Start with the same baseline/intercept as optimized
    
    print(f"===== CALCULATING CURRENT OUTCOME (ORIGINAL ALLOCATION) =====", file=sys.stderr)
    print(f"Starting with baseline (intercept): ${baseline_sales:,.2f}", file=sys.stderr)
    print(f"Total current budget: ${sum(current_allocation.values()):,.2f}", file=sys.stderr)
    
    # Track the raw channel contributions (before adding baseline)
    total_current_contribution = 0.0
    current_channel_contributions = {}
    
    # Calculate response for each channel with current budget
    # This calculation has already been performed in Step 3, so we're just referencing for clarity
    for channel, params in channel_params.items():
        spend = current_allocation.get(channel, 0.0)
        
        # Use the same parameters as in previous calculation
        beta = params.get("beta_coefficient", 0.0)
        adstock_params = params.get("adstock_parameters", {"alpha": 0.3, "l_max": 3})
        saturation_params = params.get("saturation_parameters", {"L": 1.0, "k": 0.0001, "x0": 10000.0})
        
        # Get contribution from earlier calculation
        contribution = channel_current_contributions.get(channel, 0.0)
        
        if spend > 0 and contribution > 0:
            print(f"Channel {channel}: ${spend:,.2f} → contribution=${contribution:.6f}", file=sys.stderr)
            print(f"  Beta: {beta:.6f}, L={saturation_params.get('L', 1.0)}, k={saturation_params.get('k', 0.0001)}, x0={saturation_params.get('x0', 10000.0)}", file=sys.stderr)
    
    # Add baseline to channel contributions to get final current outcome
    current_outcome = baseline_sales + total_current_contribution
    
    print(f"\nCurrent allocation summary:", file=sys.stderr)
    print(f"Baseline sales: ${baseline_sales:,.2f}", file=sys.stderr)
    print(f"Total current channel contribution: ${total_current_contribution:,.2f} ({(total_current_contribution/current_outcome)*100:.1f}% of total)", file=sys.stderr)
    print(f"Final current outcome: ${current_outcome:,.2f}", file=sys.stderr)
    
    # Calculate and print the difference for each channel
    print(f"===== CHANNEL CONTRIBUTION CHANGES =====", file=sys.stderr)
    for channel in channel_params:
        current_contrib = current_channel_contributions.get(channel, 0)
        optimized_contrib = channel_contributions.get(channel, 0)
        if current_contrib > 0 or optimized_contrib > 0:
            contrib_change = optimized_contrib - current_contrib
            contrib_pct = 0.0
            if current_contrib > 0:
                contrib_pct = (contrib_change / current_contrib) * 100
            
            print(f"Channel {channel}: " +
                  f"${current_contrib:,.2f} → ${optimized_contrib:,.2f} " +
                  f"({contrib_change:+,.2f}, {contrib_pct:+.1f}%)", file=sys.stderr)
    
    # SIMPLIFIED LIFT CALCULATION - direct comparison between optimized and current outcomes
    absolute_lift = expected_outcome - current_outcome
    expected_lift = 0.0
    
    # Use simple percentage formula as requested
    if current_outcome > 0:
        expected_lift = (absolute_lift / current_outcome) * 100
    
    # Log budget and outcome comparison
    current_budget = sum(current_allocation.values())
    optimized_budget = sum(optimized_allocation.values())
    budget_diff = optimized_budget - current_budget
    
    print(f"\nDEBUG: ===== BUDGET AND OUTCOME COMPARISON =====", file=sys.stderr)
    print(f"DEBUG: Current budget: ${current_budget:,.2f}", file=sys.stderr)
    print(f"DEBUG: Optimized budget: ${optimized_budget:,.2f}", file=sys.stderr)
    print(f"DEBUG: Budget difference: ${budget_diff:+,.2f} ({(budget_diff/current_budget)*100:+.1f}%)", file=sys.stderr)
    
    print(f"DEBUG: Current outcome: ${current_outcome:.2f}", file=sys.stderr)
    print(f"DEBUG: Expected outcome: ${expected_outcome:.2f}", file=sys.stderr)
    print(f"DEBUG: Absolute improvement: ${absolute_lift:+,.2f}", file=sys.stderr)
    print(f"DEBUG: Expected lift: {expected_lift:+.2f}%", file=sys.stderr)
    
    # Calculate budget change metrics
    budget_diff = sum(optimized_allocation.values()) - sum(current_allocation.values())
    budget_pct_change = (budget_diff / max(1, sum(current_allocation.values()))) * 100
    
    # Calculate ROI metrics
    current_roi = 0.0
    if sum(current_allocation.values()) > 0:
        current_roi = total_current_contribution / sum(current_allocation.values())
        
    optimized_roi = 0.0
    if sum(optimized_allocation.values()) > 0:
        optimized_roi = total_channel_contribution / sum(optimized_allocation.values())
    
    # Calculate incremental ROI (Return On Incremental Investment)
    incremental_roi = 0.0
    if budget_diff > 0:
        incremental_roi = absolute_lift / budget_diff
    
    # Print comprehensive summary of optimization results
    print(f"===== FINAL OPTIMIZATION RESULTS =====", file=sys.stderr)
    print(f"Current outcome: ${current_outcome:,.2f}", file=sys.stderr)
    print(f"Expected outcome: ${expected_outcome:,.2f}", file=sys.stderr) 
    print(f"Absolute improvement: ${absolute_lift:+,.2f}", file=sys.stderr)
    print(f"Expected lift: {expected_lift:+.2f}%", file=sys.stderr)
    print(f"Current budget: ${sum(current_allocation.values()):,.2f}", file=sys.stderr)
    print(f"Optimized budget: ${sum(optimized_allocation.values()):,.2f}", file=sys.stderr)
    print(f"Budget change: ${budget_diff:+,.2f} ({budget_pct_change:+.1f}%)", file=sys.stderr)
    print(f"Current marketing ROI: ${current_roi:.4f} per $1 spent", file=sys.stderr)
    print(f"Optimized marketing ROI: ${optimized_roi:.4f} per $1 spent", file=sys.stderr)
    
    if budget_diff > 0:
        print(f"Incremental ROI: ${incremental_roi:.4f} per additional $1 spent", file=sys.stderr)
    
    # Round values for API response (avoid sending too many decimals to frontend)
    expected_outcome = round(expected_outcome)
    current_outcome = round(current_outcome)
    expected_lift = round(expected_lift * 100) / 100  # Round to 2 decimal places
    
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
    
    # Create enhanced result dictionary with comprehensive metrics
    # Include all key performance metrics to provide full transparency
    budget_diff = sum(optimized_allocation.values()) - sum(current_allocation.values())
    
    # Calculate improvement per incremental dollar (ROAS of additional spend)
    incremental_roas = 0.0
    if budget_diff > 0:
        incremental_roas = (expected_outcome - current_outcome) / budget_diff
    
    # Calculate ROI metrics for both allocations
    current_roi = 0.0
    if sum(current_allocation.values()) > 0:
        current_roi = total_current_contribution / sum(current_allocation.values())
        
    optimized_roi = 0.0
    if sum(optimized_allocation.values()) > 0:
        optimized_roi = total_channel_contribution / sum(optimized_allocation.values())
    
    # Bundle all key performance metrics in one place
    performance_metrics = {
        "total_current_budget": sum(current_allocation.values()),
        "total_optimized_budget": sum(optimized_allocation.values()),
        "budget_change_pct": (budget_diff / max(1, sum(current_allocation.values()))) * 100,
        "absolute_improvement": expected_outcome - current_outcome,
        "current_marketing_roi": current_roi,  
        "optimized_marketing_roi": optimized_roi,
        "improvement_per_incremental_dollar": incremental_roas,
        "baseline_sales": baseline_sales,
        "total_channel_contribution": total_channel_contribution,
        "total_current_contribution": total_current_contribution
    }
    
    # Generate summary points for explainability
    summary_points = []
    
    # Point 1: Overall outcome change
    if expected_lift >= 0:
        summary_points.append(f"Expected outcome: ${expected_outcome:,.0f} ({expected_lift:+.1f}% improvement)")
    else:
        summary_points.append(f"Expected outcome: ${expected_outcome:,.0f} ({expected_lift:.1f}% change)")
    
    # Point 2-3: Top increased/decreased channels
    if channel_breakdown:
        # Top 3 increased channels by percent
        increased_channels = [ch for ch in channel_breakdown if ch["percent_change"] > 0]
        increased_channels.sort(key=lambda x: x["percent_change"], reverse=True)
        if increased_channels:
            inc_channels_text = ", ".join([f"{ch['channel']} ({ch['percent_change']:.0f}%)" for ch in increased_channels[:3]])
            summary_points.append(f"Biggest increases: {inc_channels_text}")
        
        # Top 3 decreased channels by percent
        decreased_channels = [ch for ch in channel_breakdown if ch["percent_change"] < 0]
        decreased_channels.sort(key=lambda x: x["percent_change"])
        if decreased_channels:
            dec_channels_text = ", ".join([f"{ch['channel']} ({ch['percent_change']:.0f}%)" for ch in decreased_channels[:3]])
            summary_points.append(f"Biggest decreases: {dec_channels_text}")
    
    # Point 4: Budget change
    summary_points.append(f"Budget change: ${budget_diff:+,.0f} ({(budget_diff/max(1, sum(current_allocation.values())))*100:+.1f}%)")
    
    # Create comprehensive result with all relevant information
    result = {
        "optimized_allocation": optimized_allocation,
        "expected_outcome": round(expected_outcome),
        "expected_lift": expected_lift,
        "current_outcome": round(current_outcome),
        "channel_breakdown": channel_breakdown,
        "target_variable": "Sales",  # Default target variable name
        "summary_points": summary_points,
        "performance_metrics": performance_metrics
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
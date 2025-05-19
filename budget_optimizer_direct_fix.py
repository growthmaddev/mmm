#!/usr/bin/env python3
"""
Direct fix for the Budget Optimizer

This script demonstrates the key issues with the budget optimizer and implements
a simplified version of the algorithm with proper fixes.
"""

import json
import sys
import numpy as np
from typing import Dict, List, Tuple, Any

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
    exponent = k * (x - x0)
    if exponent > 100:
        return L
    elif exponent < -100:
        return 0
    
    return L / (1 + np.exp(-exponent))


def get_channel_response(spend: float, beta: float, 
                         saturation_params: Dict[str, float]) -> float:
    """
    Calculate the expected sales contribution for a channel at a given spend level.
    
    Args:
        spend: Spend amount
        beta: Channel coefficient
        saturation_params: Dictionary of saturation parameters (L, k, x0)
    
    Returns:
        Expected sales contribution
    """
    if spend <= 0.0 or beta <= 0.0:
        return 0.0
    
    # Extract saturation parameters with safety checks
    L = saturation_params.get("L", 1.0)
    k = saturation_params.get("k", 0.0005)
    x0 = saturation_params.get("x0", 50000.0)
    
    # Apply saturation transformation
    saturated_spend = logistic_saturation(spend, L, k, x0)
    
    # Apply beta coefficient
    response = beta * saturated_spend
    
    return response


def calculate_marginal_return(beta: float, current_spend: float, 
                             saturation_params: Dict[str, float],
                             increment: float = 1000.0) -> float:
    """
    Calculate the marginal return for additional spend on a channel.
    
    Args:
        beta: Channel coefficient
        current_spend: Current spend amount
        saturation_params: Dictionary of saturation parameters
        increment: Increment amount for numerical differentiation
    
    Returns:
        Marginal return (additional contribution per additional dollar spent)
    """
    # Calculate response at current spend
    current_response = get_channel_response(current_spend, beta, saturation_params)
    
    # Calculate response at current spend + increment
    incremented_response = get_channel_response(current_spend + increment, beta, saturation_params)
    
    # Calculate marginal return
    marginal_return = (incremented_response - current_response) / increment
    
    # Ensure non-negative return due to numerical issues
    return max(0, marginal_return)


def optimize_budget(channels: Dict[str, Dict[str, Any]], 
                   current_allocation: Dict[str, float],
                   desired_budget: float,
                   baseline_sales: float = 100000.0,
                   increment: float = 1000.0) -> Dict[str, Any]:
    """
    Optimize budget allocation based on marginal returns.
    
    Args:
        channels: Dictionary of channel parameters
        current_allocation: Current budget allocation
        desired_budget: Total budget to allocate
        baseline_sales: Baseline sales (model intercept)
        increment: Increment amount for each iteration
    
    Returns:
        Optimized allocation and results
    """
    # Create reasonable parameters for each channel
    channel_params = {}
    for channel, spend in current_allocation.items():
        # Create synthetic but realistic beta for demonstration
        # In real model, this should come from trained model parameters
        beta = 0.3  # Base beta, will be adjusted per channel
        
        # Adjust beta to create some channel differentiation
        if channel.startswith("PPC"):
            beta *= 1.2  # PPC channels slightly more effective
        elif channel.startswith("FB"):
            beta *= 1.0  # Social channels at base effectiveness
        else:
            beta *= 0.8  # Other channels slightly less effective
            
        # Adjust beta based on current spend (proxy for channel effectiveness)
        total_spend = sum(current_allocation.values())
        if total_spend > 0:
            spend_proportion = spend / total_spend
            beta *= (0.8 + spend_proportion)  # Higher spend channels get slight boost
        
        # Create realistic saturation parameters for each channel
        # Critical: x0 must be set to reasonable values relative to channel spend
        # to ensure proper saturation behavior
        saturation_params = {
            "L": 1.0,  # Standard normalized ceiling
            "k": 0.0001,  # Moderate steepness
            "x0": min(50000, max(5000, spend * 2))  # Set midpoint relative to current spend
        }
        
        # Store the parameters
        channel_params[channel] = {
            "beta": beta,
            "saturation_params": saturation_params
        }
    
    print("====== Channel Parameters ======")
    for channel, params in channel_params.items():
        print(f"{channel}: beta={params['beta']:.4f}, x0={params['saturation_params']['x0']:.0f}")
    
    # Initialize with minimum budget for each channel
    min_channel_budget = 1000.0  # Minimum $1,000 per channel
    optimized_allocation = {channel: min_channel_budget for channel in current_allocation}
    
    # Calculate remaining budget after minimum allocations
    remaining_budget = desired_budget - sum(optimized_allocation.values())
    print(f"\nInitial allocation: ${sum(optimized_allocation.values()):,.0f}")
    print(f"Remaining budget to allocate: ${remaining_budget:,.0f}")
    
    # Allocate budget according to marginal returns with diversity
    for i in range(1000):  # Maximum iterations
        # Stop if budget is fully allocated
        if remaining_budget < increment:
            break
        
        # Calculate marginal returns for each channel
        marginal_returns = {}
        for channel, params in channel_params.items():
            current_spend = optimized_allocation[channel]
            mr = calculate_marginal_return(
                params["beta"], 
                current_spend, 
                params["saturation_params"], 
                increment
            )
            marginal_returns[channel] = mr
        
        # Apply diversity adjustment to prevent over-concentration
        adjusted_returns = {}
        total_allocated = sum(optimized_allocation.values())
        
        for channel, mr in marginal_returns.items():
            if mr <= 0:
                continue  # Skip channels with no positive return
                
            # Calculate what percentage of budget is already allocated to this channel
            percent_allocation = optimized_allocation[channel] / total_allocated
            
            # Create diversity factor that reduces marginal return as allocation increases
            # This prevents any channel from completely dominating the budget
            diversity_factor = max(0.1, 1.0 - (percent_allocation * 1.5))
            
            # Apply diversity factor to marginal return
            adjusted_mr = mr * diversity_factor
            adjusted_returns[channel] = adjusted_mr
            
            # Print details every 100 iterations
            if i % 100 == 0:
                print(f"Channel {channel}: MR={mr:.6f}, Pct={percent_allocation*100:.1f}%, "
                      f"Factor={diversity_factor:.2f}, Adj={adjusted_mr:.6f}")
        
        # If no positive adjusted returns, stop allocating
        if not adjusted_returns:
            print("No positive returns remain, stopping optimization")
            break
        
        # Find channel with highest adjusted marginal return
        best_channel = max(adjusted_returns, key=adjusted_returns.get)
        best_mr = marginal_returns[best_channel]  # Original MR for reference
        
        # Allocate increment to best channel
        optimized_allocation[best_channel] += increment
        remaining_budget -= increment
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Allocated ${increment:,.0f} to {best_channel}, "
                  f"MR={best_mr:.6f}, remaining=${remaining_budget:,.0f}")
    
    # Calculate channel contributions with optimized allocation
    print("\n====== Results with Optimized Allocation ======")
    optimized_contributions = {}
    total_contribution = 0
    
    for channel, params in channel_params.items():
        spend = optimized_allocation[channel]
        contribution = get_channel_response(spend, params["beta"], params["saturation_params"])
        optimized_contributions[channel] = contribution
        total_contribution += contribution
        
        print(f"{channel}: ${spend:,.0f} spend → ${contribution:,.2f} contribution")
    
    # Calculate expected outcome with optimized allocation
    expected_outcome = baseline_sales + total_contribution
    print(f"\nBaseline sales: ${baseline_sales:,.2f}")
    print(f"Total channel contribution: ${total_contribution:,.2f}")
    print(f"Expected outcome with optimized allocation: ${expected_outcome:,.2f}")
    
    # Calculate current contributions for comparison
    print("\n====== Results with Current Allocation ======")
    current_contributions = {}
    total_current_contribution = 0
    
    for channel, params in channel_params.items():
        spend = current_allocation[channel]
        contribution = get_channel_response(spend, params["beta"], params["saturation_params"])
        current_contributions[channel] = contribution
        total_current_contribution += contribution
        
        print(f"{channel}: ${spend:,.0f} spend → ${contribution:,.2f} contribution")
    
    # Calculate expected outcome with current allocation
    current_outcome = baseline_sales + total_current_contribution
    print(f"\nBaseline sales: ${baseline_sales:,.2f}")
    print(f"Total current channel contribution: ${total_current_contribution:,.2f}")
    print(f"Expected outcome with current allocation: ${current_outcome:,.2f}")
    
    # Calculate expected lift
    absolute_lift = expected_outcome - current_outcome
    percent_lift = (absolute_lift / current_outcome) * 100 if current_outcome > 0 else 0
    
    print("\n====== Final Results ======")
    print(f"Current outcome: ${current_outcome:,.2f}")
    print(f"Expected outcome: ${expected_outcome:,.2f}")
    print(f"Absolute improvement: ${absolute_lift:+,.2f}")
    print(f"Expected lift: {percent_lift:+.2f}%")
    
    # Create result dictionary
    result = {
        "optimized_allocation": optimized_allocation,
        "expected_outcome": round(expected_outcome),
        "expected_lift": round(percent_lift * 100) / 100,  # Round to 2 decimal places
        "current_outcome": round(current_outcome),
        "baseline_sales": baseline_sales,
        "total_contribution": total_contribution,
        "total_current_contribution": total_current_contribution
    }
    
    return result


if __name__ == "__main__":
    # Example current allocation from Model 14
    current_allocation = {
        "PPCBrand": 8697,
        "PPCNonBrand": 33283,
        "PPCShopping": 13942,
        "PPCLocal": 14980,
        "PPCPMax": 3911,
        "FBReach": 19743,
        "FBDPA": 19408,
        "OfflineMedia": 87821
    }
    
    # Example 1: Optimize with same budget
    same_budget = sum(current_allocation.values())
    print(f"\n===== SCENARIO A: Same Budget (${same_budget:,.0f}) =====\n")
    result_a = optimize_budget(
        channels={},  # Will be auto-generated
        current_allocation=current_allocation,
        desired_budget=same_budget,
        baseline_sales=100000.0  # Example baseline
    )
    
    # Example 2: Optimize with increased budget
    increased_budget = 300000
    print(f"\n===== SCENARIO B: Increased Budget (${increased_budget:,.0f}) =====\n")
    result_b = optimize_budget(
        channels={},  # Will be auto-generated
        current_allocation=current_allocation,
        desired_budget=increased_budget,
        baseline_sales=100000.0  # Example baseline
    )
    
    # Save results to file for reference
    with open("optimizer_results.json", "w") as f:
        json.dump({
            "scenario_a": result_a,
            "scenario_b": result_b
        }, f, indent=2)
        
    print("\nResults saved to optimizer_results.json")
#!/usr/bin/env python3
"""
Direct Budget Optimizer Test to verify fixes

This script provides a simplified version of the budget optimizer with proper
scaling, diversity enhancement, and lift calculation fixes that can be
verified against our production script.
"""

import json
import sys
import numpy as np
from typing import Dict, List, Tuple, Any

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
    k = saturation_params.get("k", 0.0001)
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
    Optimize budget allocation based on marginal returns with diversity enhancements.
    
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
        
        # CRITICAL FIX #1: Create realistic saturation parameters for each channel
        # x0 must be set to reasonable values relative to channel spend
        saturation_params = {
            "L": 1.0,  # Standard normalized ceiling
            "k": 0.0001,  # Moderate steepness for gradual curve
            "x0": min(50000, max(5000, spend * 2.5))  # Midpoint relative to current spend
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
    
    # CRITICAL FIX #2: Allocate budget with diversity enhancement
    for i in range(1000):  # Maximum iterations
        # Stop if budget is fully allocated
        if remaining_budget < increment:
            break
        
        # Calculate marginal returns for each channel
        marginal_returns = {}
        for channel, params in channel_params.items():
            current_spend = optimized_allocation[channel]
            beta = params.get("beta", 0)
            saturation_params = params.get("saturation_params", {})
            mr = calculate_marginal_return(beta, current_spend, saturation_params, increment)
            marginal_returns[channel] = mr
        
        # DIVERSITY ENHANCEMENT: Adjust marginal returns to prevent over-concentration
        adjusted_returns = {}
        total_allocated = sum(optimized_allocation.values())
        
        for channel, mr in marginal_returns.items():
            if mr <= 0:
                continue  # Skip channels with no positive return
                
            # Calculate what percentage of budget is already allocated to this channel
            channel_percentage = optimized_allocation[channel] / total_allocated if total_allocated > 0 else 0
            
            # Apply stronger diversity factor to prevent concentration
            diversity_factor = max(0.1, 1.0 - (channel_percentage * 2.0))
            
            # Apply diversity factor to marginal return
            adjusted_mr = mr * diversity_factor
            adjusted_returns[channel] = adjusted_mr
            
            # Print details every 100 iterations
            if i % 100 == 0:
                print(f"Channel {channel}: MR={mr:.6f}, Pct={channel_percentage*100:.1f}%, "
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
    
    # CRITICAL FIX #3: Apply additional diversity enhancement for severe concentration
    # Check if we have severe concentration (>75% in top 2 channels)
    channel_allocations = [(ch, optimized_allocation[ch]) for ch in optimized_allocation]
    channel_allocations.sort(key=lambda x: x[1], reverse=True)
    total_allocation = sum(optimized_allocation.values())
    
    top_two_allocation = sum(alloc for _, alloc in channel_allocations[:2])
    top_two_percentage = (top_two_allocation / total_allocation) * 100
    
    if top_two_percentage >= 75:
        print("\n====== APPLYING ADDITIONAL DIVERSITY ENHANCEMENT ======")
        print(f"Detected excessive concentration: Top 2 channels have {top_two_percentage:.1f}% of budget")
        
        # Identify top channels to redistribute from
        source_channels = channel_allocations[:2]  # Top 2 channels
        
        # Identify viable targets (channels with meaningful historical spend)
        viable_targets = []
        for ch, _ in channel_allocations[2:]:
            if current_allocation.get(ch, 0) > 5000:
                viable_targets.append(ch)
        
        # If no viable targets based on history, use all remaining channels
        if not viable_targets:
            viable_targets = [ch for ch, _ in channel_allocations[2:]]
        
        if viable_targets:
            # Calculate how much to redistribute (15% of top 2 channels' budget)
            amount_to_redistribute = top_two_allocation * 0.15
            print(f"Redistributing ${amount_to_redistribute:,.0f} from top 2 channels")
            
            # Take proportionally from source channels
            for source_ch, source_alloc in source_channels:
                source_proportion = source_alloc / top_two_allocation
                amount_from_source = amount_to_redistribute * source_proportion
                optimized_allocation[source_ch] -= amount_from_source
                print(f"Taking ${amount_from_source:,.0f} from {source_ch}")
            
            # Distribute evenly to viable targets
            amount_per_target = amount_to_redistribute / len(viable_targets)
            for target_ch in viable_targets:
                optimized_allocation[target_ch] += amount_per_target
                print(f"Adding ${amount_per_target:,.0f} to {target_ch}")
    
    # CRITICAL FIX #4: Apply proper scaling for meaningful contributions
    print("\n====== Calculating Expected Outcomes ======")
    
    # Determine appropriate scaling factor to make contributions meaningful
    # Target marketing contribution as percentage of baseline (typically 20-40%)
    scaling_factor = 300.0  # Using a realistic scaling factor
    print(f"Using scaling factor: {scaling_factor:.1f}x to scale channel contributions")
    
    # Calculate channel contributions with optimized allocation
    optimized_contributions = {}
    total_contribution = 0
    
    for channel, params in channel_params.items():
        spend = optimized_allocation[channel]
        beta = params.get("beta", 0)
        saturation_params = params.get("saturation_params", {})
        
        # Calculate raw contribution
        raw_contribution = get_channel_response(spend, beta, saturation_params)
        
        # Apply scaling for meaningful contribution value
        contribution = raw_contribution * scaling_factor
        
        optimized_contributions[channel] = contribution
        total_contribution += contribution
        
        # Calculate ROI for this channel
        roi = contribution / spend if spend > 0 else 0
        
        print(f"{channel}: ${spend:,.0f} spend → ${contribution:,.2f} contribution (ROI: {roi:.4f})")
    
    # Calculate expected outcome with optimized allocation
    expected_outcome = baseline_sales + total_contribution
    print(f"\nBaseline sales: ${baseline_sales:,.2f}")
    print(f"Total channel contribution: ${total_contribution:,.2f} ({(total_contribution/expected_outcome)*100:.1f}% of total)")
    print(f"Expected outcome with optimized allocation: ${expected_outcome:,.2f}")
    
    # Calculate current contributions for comparison (using the same scaling/parameters)
    print("\n====== Results with Current Allocation ======")
    current_contributions = {}
    total_current_contribution = 0
    
    for channel, params in channel_params.items():
        spend = current_allocation.get(channel, 0.0)
        beta = params.get("beta", 0)
        saturation_params = params.get("saturation_params", {})
        
        # Calculate raw contribution using same model parameters
        raw_contribution = get_channel_response(spend, beta, saturation_params)
        
        # Apply the SAME scaling factor for fair comparison
        contribution = raw_contribution * scaling_factor
        
        current_contributions[channel] = contribution
        total_current_contribution += contribution
        
        # Calculate ROI for this channel with current allocation
        roi = contribution / spend if spend > 0 else 0
        
        print(f"{channel}: ${spend:,.0f} spend → ${contribution:,.2f} contribution (ROI: {roi:.4f})")
    
    # Calculate expected outcome with current allocation
    current_outcome = baseline_sales + total_current_contribution
    print(f"\nBaseline sales: ${baseline_sales:,.2f}")
    print(f"Total current channel contribution: ${total_current_contribution:,.2f} ({(total_current_contribution/current_outcome)*100:.1f}% of total)")
    print(f"Expected outcome with current allocation: ${current_outcome:,.2f}")
    
    # CRITICAL FIX #5: Calculate lift properly with ROI adjustment for different budgets
    current_budget = sum(current_allocation.values())
    optimized_budget = sum(optimized_allocation.values())
    budget_diff = optimized_budget - current_budget
    
    print(f"\n====== Calculating Lift ======")
    print(f"Current budget: ${current_budget:,.2f}")
    print(f"Optimized budget: ${optimized_budget:,.2f}")
    print(f"Budget difference: ${budget_diff:+,.2f}")
    
    # Calculate standard lift and ROI metrics
    absolute_lift = expected_outcome - current_outcome
    standard_lift_pct = (absolute_lift / current_outcome) * 100 if current_outcome > 0 else 0
    
    current_roi = total_current_contribution / current_budget if current_budget > 0 else 0
    optimized_roi = total_contribution / optimized_budget if optimized_budget > 0 else 0
    roi_pct_change = ((optimized_roi / current_roi) - 1) * 100 if current_roi > 0 else 0
    
    print(f"Current ROI: {current_roi:.6f}")
    print(f"Optimized ROI: {optimized_roi:.6f}")
    print(f"ROI change: {roi_pct_change:+.2f}%")
    
    # For different budget levels, adjust lift calculation appropriately
    if abs(budget_diff) > 5000:  # Only adjust if budgets differ significantly
        print(f"Calculating ROI-adjusted lift for different budget levels")
        
        # For increased budget: expected outcome at current efficiency
        if budget_diff > 0:
            # What would happen if we invested additional budget at current ROI
            projected_contribution = total_current_contribution + (budget_diff * current_roi)
            projected_outcome = baseline_sales + projected_contribution
            
            print(f"Current outcome: ${current_outcome:,.2f}")
            print(f"Projected outcome at current ROI: ${projected_outcome:,.2f}")
            print(f"Optimized outcome: ${expected_outcome:,.2f}")
            
            # How much better is optimized vs projected at current efficiency?
            roi_adjusted_lift = ((expected_outcome / projected_outcome) - 1) * 100 if projected_outcome > 0 else 0
            
            print(f"Standard lift: {standard_lift_pct:+.2f}%")
            print(f"ROI-adjusted lift: {roi_adjusted_lift:+.2f}%")
            
            # Use ROI-adjusted lift for increased budgets
            final_lift = roi_adjusted_lift
        else:
            # For reduced budget: simple lift with efficiency bonus
            print(f"Using standard lift with efficiency adjustment: {standard_lift_pct:+.2f}%")
            final_lift = standard_lift_pct
    else:
        # For comparable budgets: standard lift calculation
        print(f"Using standard lift calculation: {standard_lift_pct:+.2f}%")
        final_lift = standard_lift_pct
    
    # Ensure lift is reasonable
    final_lift = max(-50, min(100, final_lift))
    
    print("\n====== Final Results ======")
    print(f"Current outcome: ${current_outcome:,.2f}")
    print(f"Expected outcome: ${expected_outcome:,.2f}")
    print(f"Absolute improvement: ${absolute_lift:+,.2f}")
    print(f"Expected lift: {final_lift:+.2f}%")
    
    # Create breakdown of channel changes
    channel_breakdown = []
    for channel in current_allocation:
        current_spend = current_allocation.get(channel, 0)
        optimized_spend = optimized_allocation[channel]
        current_contrib = current_contributions.get(channel, 0)
        optimized_contrib = optimized_contributions.get(channel, 0)
        
        percent_change = ((optimized_spend / max(1, current_spend)) - 1) * 100
        roi = optimized_contrib / max(1, optimized_spend)
        
        channel_breakdown.append({
            "channel": channel,
            "current_spend": current_spend,
            "optimized_spend": optimized_spend,
            "percent_change": percent_change,
            "roi": roi,
            "contribution": optimized_contrib
        })
    
    # Sort by optimized spend
    channel_breakdown.sort(key=lambda x: x["optimized_spend"], reverse=True)
    
    # Create result dictionary
    result = {
        "optimized_allocation": optimized_allocation,
        "expected_outcome": round(expected_outcome),
        "expected_lift": round(final_lift * 100) / 100,  # Round to 2 decimal places
        "current_outcome": round(current_outcome),
        "baseline_sales": baseline_sales,
        "total_contribution": total_contribution,
        "total_current_contribution": total_current_contribution,
        "channel_breakdown": channel_breakdown
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
        baseline_sales=200000.0  # Example baseline
    )
    
    # Example 2: Optimize with increased budget
    increased_budget = 300000
    print(f"\n===== SCENARIO B: Increased Budget (${increased_budget:,.0f}) =====\n")
    result_b = optimize_budget(
        channels={},  # Will be auto-generated
        current_allocation=current_allocation,
        desired_budget=increased_budget,
        baseline_sales=200000.0  # Example baseline
    )
    
    # Save results to file for reference
    with open("budget_optimizer_fix_results.json", "w") as f:
        json.dump({
            "scenario_a": result_a,
            "scenario_b": result_b
        }, f, indent=2)
        
    print("\nResults saved to budget_optimizer_fix_results.json")
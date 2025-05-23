#!/usr/bin/env python
"""
MMM Optimizer Service

This script provides a standalone service that connects our fixed parameter MMM solution
with the budget optimizer. It loads fixed parameters from a model, transforms them into
the right format for the optimizer, and returns the optimized budget allocation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import time

# Apply the same global TensorVariable patch as in fit_mmm_fixed_params.py
print("Applying global TensorVariable patch...", file=sys.stderr)

import pytensor.tensor as pt

# Store original methods
_original_getattr = pt.TensorVariable.__getattribute__
_original_setattr = pt.TensorVariable.__setattr__

def _patched_getattr(self, name):
    if name == 'dims':
        try:
            return _original_getattr(self, name)
        except AttributeError:
            if hasattr(self, '_pymc_dims'):
                return self._pymc_dims
            return ()
    return _original_getattr(self, name)

def _patched_setattr(self, name, value):
    if name == 'dims':
        _original_setattr(self, '_pymc_dims', value)
    else:
        _original_setattr(self, name, value)

# Apply patches
pt.TensorVariable.__getattribute__ = _patched_getattr
pt.TensorVariable.__setattr__ = _patched_setattr

print("Global patch applied successfully!", file=sys.stderr)

# Function to implement logistic saturation
def logistic_saturation(x: float, L: float = 1.0, k: float = 0.0005, x0: float = 50000.0) -> float:
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
    # Handle potential overflow in exp(-k * (x - x0))
    if k * (x - x0) > 100:
        return L  # Saturated at max value
    elif k * (x - x0) < -100:
        return 0  # Effectively zero
    else:
        return L / (1 + np.exp(-k * (x - x0)))

# Function to calculate channel response (fixed version)
def get_channel_response(
    spend: float,
    beta: float,
    saturation_params: Dict[str, float],
    adstock_params: Optional[Dict[str, float]] = None,
    adstock_type: str = "GeometricAdstock",
    saturation_type: str = "LogisticSaturation",
    scaling_factor: float = 5000.0  # Apply scaling to make contributions meaningful
) -> float:
    """
    Calculate the expected response for a channel at a given spend level.
    
    Args:
        spend: Spend amount
        beta: Channel coefficient (effectiveness multiplier)
        saturation_params: Dictionary of saturation parameters
        adstock_params: Dictionary of adstock parameters (optional, for future use)
        adstock_type: Type of adstock function to use
        saturation_type: Type of saturation function to use
        scaling_factor: Multiplier to scale contributions to meaningful level
        
    Returns:
        Expected response (e.g., sales contribution)
    """
    # For now, we're only supporting LogisticSaturation
    if saturation_type != "LogisticSaturation":
        print(f"Warning: Unsupported saturation type {saturation_type}, using LogisticSaturation", file=sys.stderr)
    
    # Extract saturation parameters (with defaults if not provided)
    L = saturation_params.get("L", 1.0)
    k = saturation_params.get("k", 0.0005)
    x0 = saturation_params.get("x0", 50000.0)
    
    # Apply saturation transformation
    saturated_spend = logistic_saturation(spend, L, k, x0)
    
    # Apply beta coefficient
    response = beta * saturated_spend
    
    # Apply scaling factor
    response *= scaling_factor
    
    return response

# Function to calculate the marginal return for a given channel
def calculate_marginal_return(
    beta: float,
    current_spend: float,
    saturation_params: Dict[str, float],
    adstock_params: Optional[Dict[str, float]] = None,
    increment: float = 1000.0,
    scaling_factor: float = 5000.0  # Apply same scaling as in get_channel_response
) -> float:
    """
    Calculate the marginal return for additional spend on a channel.
    
    Args:
        beta: Channel coefficient
        current_spend: Current spend amount
        saturation_params: Dictionary of saturation parameters
        adstock_params: Dictionary of adstock parameters (optional)
        increment: Increment amount for numerical differentiation
        scaling_factor: Multiplier to scale contributions to meaningful level
        
    Returns:
        Marginal return (additional contribution per additional dollar spent)
    """
    # Calculate response at current spend
    current_response = get_channel_response(
        current_spend,
        beta,
        saturation_params,
        adstock_params,
        scaling_factor=scaling_factor
    )
    
    # Calculate response at current spend + increment
    incremented_response = get_channel_response(
        current_spend + increment,
        beta,
        saturation_params,
        adstock_params,
        scaling_factor=scaling_factor
    )
    
    # Calculate marginal return
    marginal_return = (incremented_response - current_response) / increment
    
    return marginal_return

# Main budget optimization function
def optimize_budget(
    channel_params: Dict[str, Dict[str, Any]],
    current_allocation: Dict[str, float],
    desired_budget: float,
    baseline_sales: float = 100000.0,
    increment: float = 1000.0,
    min_channel_budget: float = 1000.0,
    max_iterations: int = 1000,
    scaling_factor: float = 5000.0
) -> Dict[str, Any]:
    """
    Optimize budget allocation based on channel parameters.
    
    Args:
        channel_params: Dictionary of channel parameters
        current_allocation: Dictionary of current budget allocation
        desired_budget: Total budget to allocate
        baseline_sales: Baseline sales (intercept)
        increment: Increment amount for each iteration
        min_channel_budget: Minimum budget for each channel
        max_iterations: Maximum number of iterations
        scaling_factor: Multiplier for channel contributions
        
    Returns:
        Dictionary with optimized allocation and results
    """
    print(f"Starting budget optimization with {len(channel_params)} channels", file=sys.stderr)
    print(f"Current allocation: {current_allocation}", file=sys.stderr)
    print(f"Desired budget: {desired_budget}", file=sys.stderr)
    
    # Initialize optimized allocation with minimum budgets
    optimized_allocation = {channel: min_channel_budget for channel in channel_params}
    
    # Calculate remaining budget
    allocated_budget = sum(optimized_allocation.values())
    remaining_budget = desired_budget - allocated_budget
    
    # Iteratively allocate budget based on marginal returns
    iterations = 0
    while remaining_budget >= increment and iterations < max_iterations:
        iterations += 1
        
        # Calculate marginal returns for each channel
        marginal_returns = {}
        for channel, params in channel_params.items():
            # Extract parameters
            beta = params.get("beta_coefficient", 1.0)
            
            # Get saturation parameters
            saturation_params = params.get("saturation_parameters", {
                "L": 1.0,
                "k": 0.0005,
                "x0": 50000.0
            })
            
            # Get adstock parameters
            adstock_params = params.get("adstock_parameters", {
                "alpha": 0.6,
                "l_max": 8
            })
            
            # Calculate marginal return
            marginal_returns[channel] = calculate_marginal_return(
                beta=beta,
                current_spend=optimized_allocation[channel],
                saturation_params=saturation_params,
                adstock_params=adstock_params,
                increment=increment,
                scaling_factor=scaling_factor
            )
        
        # Find channel with highest marginal return
        best_channel = max(marginal_returns, key=marginal_returns.get)
        
        # Allocate increment to best channel
        optimized_allocation[best_channel] += increment
        remaining_budget -= increment
        
        # Every 100 iterations, log progress
        if iterations % 100 == 0:
            print(f"Iteration {iterations}: {remaining_budget:.2f} remaining", file=sys.stderr)
    
    print(f"Budget optimization completed in {iterations} iterations", file=sys.stderr)
    
    # Calculate expected outcome for optimized allocation
    optimized_outcome = calculate_expected_outcome(
        channel_params, optimized_allocation, baseline_sales, scaling_factor
    )
    
    # Calculate expected outcome for current allocation
    current_outcome = calculate_expected_outcome(
        channel_params, current_allocation, baseline_sales, scaling_factor
    )
    
    # Calculate channel breakdown
    channel_breakdown = []
    for channel in channel_params:
        current = current_allocation.get(channel, 0)
        optimized = optimized_allocation.get(channel, 0)
        
        # Calculate percent change
        if current > 0:
            percent_change = ((optimized - current) / current) * 100
        else:
            percent_change = 100 if optimized > 0 else 0
        
        # Calculate ROI and contribution
        roi = calculate_channel_roi(
            channel_params[channel], 
            optimized, 
            scaling_factor
        )
        
        contribution = get_channel_response(
            optimized,
            channel_params[channel].get("beta_coefficient", 1.0),
            channel_params[channel].get("saturation_parameters", {"L": 1.0, "k": 0.0005, "x0": 50000.0}),
            channel_params[channel].get("adstock_parameters", {"alpha": 0.6, "l_max": 8}),
            scaling_factor=scaling_factor
        )
        
        channel_breakdown.append({
            "channel": channel,
            "current_spend": current,
            "optimized_spend": optimized,
            "percent_change": percent_change,
            "roi": roi,
            "contribution": contribution
        })
    
    # Calculate expected lift
    if current_outcome > 0:
        expected_lift = ((optimized_outcome - current_outcome) / current_outcome) * 100
    else:
        expected_lift = 100 if optimized_outcome > 0 else 0
    
    return {
        "optimized_allocation": optimized_allocation,
        "current_allocation": current_allocation,
        "expected_outcome": optimized_outcome,
        "current_outcome": current_outcome,
        "expected_lift": expected_lift,
        "channel_breakdown": channel_breakdown,
        "target_variable": "Sales",
        "baseline_sales": baseline_sales,
        "iterations": iterations
    }

# Helper function to calculate expected outcome
def calculate_expected_outcome(
    channel_params: Dict[str, Dict[str, Any]],
    allocation: Dict[str, float],
    baseline_sales: float,
    scaling_factor: float
) -> float:
    """
    Calculate expected outcome for a given allocation.
    
    Args:
        channel_params: Dictionary of channel parameters
        allocation: Dictionary of budget allocation
        baseline_sales: Baseline sales (intercept)
        scaling_factor: Multiplier for channel contributions
        
    Returns:
        Expected outcome
    """
    outcome = baseline_sales
    
    for channel, params in channel_params.items():
        spend = allocation.get(channel, 0)
        if spend > 0:
            # Extract parameters
            beta = params.get("beta_coefficient", 1.0)
            
            # Get saturation parameters
            saturation_params = params.get("saturation_parameters", {
                "L": 1.0,
                "k": 0.0005,
                "x0": 50000.0
            })
            
            # Get adstock parameters
            adstock_params = params.get("adstock_parameters", {
                "alpha": 0.6,
                "l_max": 8
            })
            
            # Calculate contribution
            contribution = get_channel_response(
                spend, beta, saturation_params, adstock_params, scaling_factor=scaling_factor
            )
            
            outcome += contribution
    
    return outcome

# Helper function to calculate channel ROI
def calculate_channel_roi(
    channel_params: Dict[str, Any],
    spend: float,
    scaling_factor: float
) -> float:
    """
    Calculate ROI for a channel.
    
    Args:
        channel_params: Channel parameters
        spend: Spend amount
        scaling_factor: Multiplier for channel contributions
        
    Returns:
        ROI (return on investment)
    """
    if spend <= 0:
        return 0
    
    # Extract parameters
    beta = channel_params.get("beta_coefficient", 1.0)
    
    # Get saturation parameters
    saturation_params = channel_params.get("saturation_parameters", {
        "L": 1.0,
        "k": 0.0005,
        "x0": 50000.0
    })
    
    # Get adstock parameters
    adstock_params = channel_params.get("adstock_parameters", {
        "alpha": 0.6,
        "l_max": 8
    })
    
    # Calculate contribution
    contribution = get_channel_response(
        spend, beta, saturation_params, adstock_params, scaling_factor=scaling_factor
    )
    
    # Calculate ROI
    roi = contribution / spend
    
    return roi

# Main function to run the service
def main(config_file: str) -> Dict[str, Any]:
    """
    Main function to run the MMM optimizer service.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Dictionary with optimization results
    """
    print(f"Loading configuration from {config_file}", file=sys.stderr)
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    model_id = config.get("model_id")
    project_id = config.get("project_id")
    channel_params = config.get("channel_params", {})
    data_file = config.get("data_file")
    current_budget = config.get("current_budget", 0)
    desired_budget = config.get("desired_budget", 0)
    current_allocation = config.get("current_allocation", {})
    baseline_sales = config.get("baseline_sales", 100000.0)
    
    print(f"Model ID: {model_id}", file=sys.stderr)
    print(f"Project ID: {project_id}", file=sys.stderr)
    print(f"Data file: {data_file}", file=sys.stderr)
    print(f"Current budget: {current_budget}", file=sys.stderr)
    print(f"Desired budget: {desired_budget}", file=sys.stderr)
    print(f"Baseline sales: {baseline_sales}", file=sys.stderr)
    
    # Transform channel parameters if needed
    # For this example, we'll take alpha, L, k, and x0 from the channel_params
    # and convert them to the format needed by our optimization function
    transformed_params = {}
    
    # Handle different parameter formats
    for k, v in channel_params.items():
        # Check what kind of parameters we have
        if k in ['alpha', 'L', 'k', 'x0']:
            # We have flattened parameters by name
            # Convert to channel-specific format
            for channel in current_allocation:
                if channel not in transformed_params:
                    transformed_params[channel] = {
                        "beta_coefficient": 1.0,
                        "saturation_parameters": {},
                        "adstock_parameters": {}
                    }
                
                if k == 'alpha':
                    transformed_params[channel]["adstock_parameters"]["alpha"] = v.get(channel, 0.6)
                elif k in ['L', 'k', 'x0']:
                    transformed_params[channel]["saturation_parameters"][k] = v.get(channel, 
                        1.0 if k == 'L' else 0.0005 if k == 'k' else 50000.0)
                
        else:
            # Assuming this is a channel name
            channel = k
            if channel not in transformed_params:
                transformed_params[channel] = {
                    "beta_coefficient": 1.0,
                    "saturation_parameters": {
                        "L": 1.0,
                        "k": 0.0005,
                        "x0": 50000.0
                    },
                    "adstock_parameters": {
                        "alpha": 0.6,
                        "l_max": 8
                    }
                }
            
            # Extract parameters from the channel object
            params = v
            
            if isinstance(params, dict):
                # Beta coefficient
                if "beta" in params:
                    transformed_params[channel]["beta_coefficient"] = params["beta"]
                
                # Saturation parameters
                if "L" in params:
                    transformed_params[channel]["saturation_parameters"]["L"] = params["L"]
                if "k" in params:
                    transformed_params[channel]["saturation_parameters"]["k"] = params["k"]
                if "x0" in params:
                    transformed_params[channel]["saturation_parameters"]["x0"] = params["x0"]
                
                # Adstock parameters
                if "alpha" in params:
                    transformed_params[channel]["adstock_parameters"]["alpha"] = params["alpha"]
                if "l_max" in params:
                    transformed_params[channel]["adstock_parameters"]["l_max"] = params["l_max"]
    
    # If we still don't have parameters for all channels, create defaults
    for channel in current_allocation:
        if channel not in transformed_params:
            # Create default parameters
            transformed_params[channel] = {
                "beta_coefficient": 1.0,
                "saturation_parameters": {
                    "L": 1.0,
                    "k": 0.0005,
                    "x0": 50000.0
                },
                "adstock_parameters": {
                    "alpha": 0.6,
                    "l_max": 8
                }
            }
    
    print(f"Transformed parameters for {len(transformed_params)} channels", file=sys.stderr)
    
    # Run budget optimization
    try:
        start_time = time.time()
        result = optimize_budget(
            channel_params=transformed_params,
            current_allocation=current_allocation,
            desired_budget=desired_budget,
            baseline_sales=baseline_sales
        )
        end_time = time.time()
        
        # Add some metadata
        result["optimization_time"] = end_time - start_time
        result["model_id"] = model_id
        result["project_id"] = project_id
        result["success"] = True
        
        print(f"Optimization completed in {end_time - start_time:.2f} seconds", file=sys.stderr)
        return result
        
    except Exception as e:
        print(f"Error optimizing budget: {str(e)}", file=sys.stderr)
        return {
            "success": False,
            "error": str(e),
            "model_id": model_id,
            "project_id": project_id
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mmm_optimizer_service.py <config_file>", file=sys.stderr)
        sys.exit(1)
    
    config_file = sys.argv[1]
    result = main(config_file)
    
    # Print result as JSON for the Node.js server to parse
    print(json.dumps(result, indent=2))
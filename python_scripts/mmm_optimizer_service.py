#!/usr/bin/env python
"""
Production service that combines MMM analysis with budget optimization
Designed to be called from the Express.js backend
"""

import sys
import json
import argparse
from datetime import datetime

# Import our modules
from fit_mmm_fixed_params import create_mmm_with_fixed_params
from optimize_budget_marginal import optimize_budget

def run_mmm_and_optimize(data_file, config_file, optimization_params):
    """
    Run complete MMM analysis and budget optimization pipeline
    
    Args:
        data_file: Path to CSV data file
        config_file: Path to MMM configuration JSON
        optimization_params: Dict with budget optimization parameters
    
    Returns:
        Combined results dictionary
    """
    
    try:
        # Step 1: Run MMM Analysis
        print("Running MMM analysis...", file=sys.stderr)
        mmm_results = create_mmm_with_fixed_params(config_file, data_file)
        
        if not mmm_results or "error" in mmm_results:
            return {
                "success": False,
                "error": "MMM analysis failed",
                "details": mmm_results
            }
        
        # Extract data from MMM results
        channels = mmm_results["model_info"]["channels"]
        fixed_params = mmm_results["fixed_parameters"]
        channel_analysis = mmm_results["channel_analysis"]
        current_spend = channel_analysis["spend"]
        
        # Step 2: Prepare optimizer configuration
        print("Preparing budget optimization...", file=sys.stderr)
        
        # Get optimization parameters
        budget_multiplier = optimization_params.get("budget_multiplier", 1.0)
        min_per_channel = optimization_params.get("min_per_channel", 100)
        diversity_penalty = optimization_params.get("diversity_penalty", 0.1)
        
        total_budget = sum(current_spend.values()) * budget_multiplier
        
        # Prepare channel parameters for optimizer
        channel_params = {}
        for ch in channels:
            # Map MMM parameters to optimizer-expected format
            channel_params[ch] = {
                "beta_coefficient": fixed_params.get("alpha", {}).get(ch, 0.2),
                "saturation_parameters": {
                    "L": fixed_params.get("L", {}).get(ch, 1.0),
                    "k": fixed_params.get("k", {}).get(ch, 0.0001),
                    "x0": fixed_params.get("x0", {}).get(ch, 50000.0)
                }
            }
        
        # Create current allocation dictionary
        current_allocation = {ch: current_spend[ch] for ch in channels}
        
        # Step 3: Run optimization
        print("Running budget optimization...", file=sys.stderr)
        opt_results = optimize_budget(
            channel_params=channel_params,
            desired_budget=total_budget,
            current_allocation=current_allocation,
            min_channel_budget=min_per_channel,
            debug=True,
            enable_dynamic_diversity=(diversity_penalty > 0)
        )
        
        # Step 4: Combine results
        optimized_allocation = {}
        allocation_changes = {}
        
        # Extract optimized allocation from results
        if "optimized_allocation" in opt_results:
            optimized_spend = opt_results["optimized_allocation"]
        else:
            # Fallback if the key name is different
            # Search for a dictionary in the results that contains all channels
            for key, value in opt_results.items():
                if isinstance(value, dict) and all(ch in value for ch in channels):
                    optimized_spend = value
                    break
            else:
                # If not found, use a copy of the current allocation
                optimized_spend = dict(current_allocation)
        
        for ch in channels:
            current = current_spend[ch]
            optimized = optimized_spend.get(ch, current)
            optimized_allocation[ch] = optimized
            
            if current > 0:
                allocation_changes[ch] = {
                    "current": current,
                    "optimized": optimized,
                    "change_amount": optimized - current,
                    "change_percent": ((optimized - current) / current) * 100
                }
            else:
                allocation_changes[ch] = {
                    "current": current,
                    "optimized": optimized,
                    "change_amount": optimized,
                    "change_percent": 0
                }
        
        # Calculate summary metrics
        total_current = sum(current_spend.values())
        total_optimized = sum(optimized_allocation.values())
        
        # Extract lift from results
        if "percentage_lift" in opt_results:
            expected_lift = opt_results["percentage_lift"]
        elif "expected_lift" in opt_results:
            expected_lift = opt_results["expected_lift"]
        else:
            # Calculate estimated lift based on optimized vs current outcome
            if "current_outcome" in opt_results and "expected_outcome" in opt_results:
                current_outcome = opt_results["current_outcome"]
                expected_outcome = opt_results["expected_outcome"]
                if current_outcome > 0:
                    expected_lift = (expected_outcome - current_outcome) / current_outcome
                else:
                    expected_lift = 0
            else:
                expected_lift = 0
        
        combined_results = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "mmm_analysis": {
                "channel_roi": channel_analysis["roi"],
                "channel_contributions": channel_analysis["contribution_percentage"],
                "model_parameters": fixed_params
            },
            "optimization_results": {
                "current_allocation": current_spend,
                "optimized_allocation": optimized_allocation,
                "allocation_changes": allocation_changes,
                "total_budget": {
                    "current": total_current,
                    "optimized": total_optimized,
                    "change": total_optimized - total_current
                },
                "expected_lift": expected_lift,
                "optimization_params": optimization_params
            },
            "recommendations": generate_recommendations(allocation_changes)
        }
        
        return combined_results
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def generate_recommendations(allocation_changes):
    """Generate human-readable recommendations based on optimization results"""
    
    recommendations = []
    
    # Sort by absolute change amount
    sorted_channels = sorted(
        allocation_changes.items(),
        key=lambda x: abs(x[1]["change_amount"]),
        reverse=True
    )
    
    for channel, changes in sorted_channels[:3]:  # Top 3 changes
        change_pct = changes["change_percent"]
        change_amt = changes["change_amount"]
        
        if change_pct > 50:
            recommendations.append(
                f"Significantly increase {channel} budget by ${change_amt:,.0f} "
                f"({change_pct:+.1f}%) - this channel shows high potential returns"
            )
        elif change_pct > 10:
            recommendations.append(
                f"Increase {channel} budget by ${change_amt:,.0f} "
                f"({change_pct:+.1f}%) to improve overall ROI"
            )
        elif change_pct < -30:
            recommendations.append(
                f"Reduce {channel} budget by ${abs(change_amt):,.0f} "
                f"({change_pct:.1f}%) - this channel appears oversaturated"
            )
        elif change_pct < -10:
            recommendations.append(
                f"Consider reducing {channel} budget by ${abs(change_amt):,.0f} "
                f"({change_pct:.1f}%) to reallocate to higher-performing channels"
            )
    
    return recommendations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMM Analysis and Budget Optimization Service')
    parser.add_argument('data_file', help='Path to marketing data CSV')
    parser.add_argument('config_file', help='Path to MMM configuration JSON')
    parser.add_argument('--budget-multiplier', type=float, default=1.0,
                        help='Budget multiplier (1.0 = current budget, 1.2 = 20% increase)')
    parser.add_argument('--min-per-channel', type=float, default=100,
                        help='Minimum budget per channel')
    parser.add_argument('--diversity-penalty', type=float, default=0.1,
                        help='Diversity penalty (0-1, higher = more balanced allocation)')
    parser.add_argument('--output', '-o', help='Output file for results JSON')
    
    args = parser.parse_args()
    
    # Prepare optimization parameters
    opt_params = {
        "budget_multiplier": args.budget_multiplier,
        "min_per_channel": args.min_per_channel,
        "diversity_penalty": args.diversity_penalty
    }
    
    # Run the pipeline
    results = run_mmm_and_optimize(args.data_file, args.config_file, opt_params)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if results.get("success", False) else 1)
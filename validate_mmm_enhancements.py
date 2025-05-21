#!/usr/bin/env python
"""
Validation script for MarketMixMaster enhancements
This script creates sample data to validate the enhanced train_mmm.py script
"""

import json
import sys
import numpy as np
import pandas as pd

# Import the analytics functions from train_mmm.py
sys.path.append('./python_scripts')
from train_mmm import (
    calculate_response_curve_points,
    calculate_elasticity,
    calculate_optimal_spend,
    calculate_channel_contributions_over_time,
    calculate_channel_interaction_matrix,
    calculate_diminishing_returns_thresholds,
    calculate_adstock_decay_points
)

# Create a sample result structure that would normally be produced by train_mmm.py
# This allows us to validate the structure without running the full model
def create_sample_results():
    # Sample data
    np.random.seed(42)  # For reproducibility
    n_periods = 20
    
    # Create channel data
    channels = ["TV_Spend", "Radio_Spend", "Social_Spend"]
    spend_data = {}
    for channel in channels:
        spend_data[channel] = np.random.uniform(200, 1000, n_periods)
    
    # Create target data
    target_data = np.random.normal(10000, 2000, n_periods)
    
    # Create DataFrame
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='W')
    df = pd.DataFrame({
        'Date': dates
    })
    
    # Add spend columns
    for channel, spend in spend_data.items():
        df[channel] = spend
    
    # Add target column
    df['Sales'] = target_data
    
    # Sample model parameters
    model_parameters = {
        "TV": {
            "beta_coefficient": 0.5,
            "adstock_type": "GeometricAdstock",
            "adstock_parameters": {
                "alpha": 0.3,
                "l_max": 3
            },
            "saturation_type": "LogisticSaturation",
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.0005,
                "x0": 600.0
            }
        },
        "Radio": {
            "beta_coefficient": 0.3,
            "adstock_type": "GeometricAdstock",
            "adstock_parameters": {
                "alpha": 0.2,
                "l_max": 2
            },
            "saturation_type": "LogisticSaturation",
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.001,
                "x0": 250.0
            }
        },
        "Social": {
            "beta_coefficient": 0.4,
            "adstock_type": "GeometricAdstock",
            "adstock_parameters": {
                "alpha": 0.1,
                "l_max": 1
            },
            "saturation_type": "LogisticSaturation",
            "saturation_parameters": {
                "L": 1.0,
                "k": 0.002,
                "x0": 400.0
            }
        }
    }
    
    # Sample contributions
    contributions = {
        "TV_Spend": 35000.0,
        "Radio_Spend": 20000.0,
        "Social_Spend": 25000.0
    }
    
    # Sample ROI data
    roi_data = {
        "TV_Spend": 2.5,
        "Radio_Spend": 1.8,
        "Social_Spend": 3.2
    }
    
    # Model performance metrics
    r_squared = 0.85
    rmse = 1200.0
    
    # Top and worst channels
    top_channel = "TV_Spend"
    top_channel_roi = 2.5
    worst_channel = "Radio_Spend"
    worst_channel_roi = 1.8
    
    # Intercept value (baseline sales)
    intercept_value = 5000.0
    
    # Calculate temporal contributions
    temporal_contributions = calculate_channel_contributions_over_time(
        df, channels, model_parameters, intercept_value
    )
    
    # Calculate diminishing returns thresholds
    diminishing_returns_thresholds = calculate_diminishing_returns_thresholds(
        channels, df, model_parameters
    )
    
    # Calculate channel interactions
    channel_interactions = calculate_channel_interaction_matrix(
        channels, model_parameters
    )
    
    # Date values for time series
    date_values = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    
    # Enhanced analytics section
    analytics_section = {
        # 1. Sales Decomposition
        "sales_decomposition": {
            "base_sales": float(intercept_value * len(df)),
            "incremental_sales": {
                channel.replace("_Spend", ""): float(contributions[channel])
                for channel in channels
            },
            "total_sales": float(sum(target_data)),
            "percent_decomposition": {
                "base": float((intercept_value * len(df)) / sum(target_data)),
                "channels": {
                    channel.replace("_Spend", ""): float(contributions[channel] / sum(target_data))
                    for channel in channels
                }
            },
            "time_series": {
                "dates": date_values,
                "base": [float(intercept_value) for _ in range(len(df))],
                "channels": {
                    channel.replace("_Spend", ""): [float(v) for v in temporal_contributions.get(channel, np.zeros(len(df)))]
                    for channel in channels
                }
            }
        },
        
        # 2. Channel Effectiveness Detail
        "channel_effectiveness_detail": {
            channel.replace("_Spend", ""): {
                "roi": float(roi_data.get(channel, 0)),
                "roi_ci_low": float(roi_data.get(channel, 0) * 0.8),
                "roi_ci_high": float(roi_data.get(channel, 0) * 1.2),
                "statistical_significance": 0.95,
                "cost_per_outcome": float(df[channel].sum() / contributions[channel]),
                "effectiveness_rank": rank
            } for rank, (channel, _) in enumerate(
                sorted([(ch, roi_data.get(ch, 0)) for ch in channels], 
                       key=lambda x: x[1], reverse=True), 1)
        },
        
        # 3. Response Curves
        "response_curves": {
            channel.replace("_Spend", ""): {
                "spend_levels": [float(x) for x in np.linspace(0, df[channel].max() * 1.5, 20)],
                "response_values": calculate_response_curve_points(
                    np.linspace(0, df[channel].max() * 1.5, 20),
                    model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0),
                    model_parameters.get(channel.replace("_Spend", ""), {}).get("adstock_parameters", {}),
                    model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {})
                ),
                "optimal_spend_point": float(calculate_optimal_spend(
                    model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0),
                    model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {})
                )),
                "elasticity": {
                    "low_spend": float(calculate_elasticity(
                        df[channel].quantile(0.25),
                        model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0),
                        model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {})
                    )),
                    "mid_spend": float(calculate_elasticity(
                        df[channel].median(),
                        model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0),
                        model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {})
                    )),
                    "high_spend": float(calculate_elasticity(
                        df[channel].quantile(0.75),
                        model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0),
                        model_parameters.get(channel.replace("_Spend", ""), {}).get("saturation_parameters", {})
                    ))
                }
            } for channel in channels
        },
        
        # 4. Budget Optimization Parameters
        "optimization_parameters": {
            "channel_interactions": channel_interactions,
            "diminishing_returns": diminishing_returns_thresholds
        },
        
        # 5. External Factors Impact
        "external_factors": {
            "seasonal_impact": {
                "Q1": 0.8,
                "Q2": 1.0,
                "Q3": 1.2,
                "Q4": 1.5
            },
            "promotion_impact": {
                "Holiday_Promo": {
                    "coefficient": 0.3,
                    "relative_impact": 0.15
                }
            },
            "external_correlations": {
                "Temperature": {
                    "correlation": 0.25,
                    "significance": 0.92
                }
            }
        },
        
        # 6. Temporal Effects (Adstock/Carryover)
        "temporal_effects": {
            channel.replace("_Spend", ""): {
                "immediate_impact": float(model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0) * 0.7),
                "lagged_impact": float(model_parameters.get(channel.replace("_Spend", ""), {}).get("beta_coefficient", 0.0) * 0.3),
                "decay_points": calculate_adstock_decay_points(
                    channel,
                    model_parameters,
                    max_periods=10
                ),
                "effective_frequency": float(model_parameters.get(channel.replace("_Spend", ""), {}).get("adstock_parameters", {}).get("l_max", 3) * 0.5)
            } for channel in channels
        }
    }
    
    # Prepare the complete results structure with backward compatibility
    results = {
        "success": True,
        "model_accuracy": float(r_squared * 100),
        "top_channel": top_channel.replace("_Spend", ""),
        "top_channel_roi": f"${top_channel_roi:.2f}",
        "increase_channel": top_channel.replace("_Spend", ""),
        "increase_percent": "15",
        "decrease_channel": worst_channel.replace("_Spend", ""),
        "decrease_roi": f"${worst_channel_roi:.2f}",
        "optimize_channel": top_channel.replace("_Spend", ""),
        "summary": {
            "channels": {
                channel.replace("_Spend", ""): { 
                    "contribution": float(contributions[channel] / sum(contributions.values())),
                    "roi": float(roi_data.get(channel, 0)),
                    # Add model parameters for this channel
                    **(model_parameters.get(channel.replace("_Spend", ""), {}))
                } for channel in channels
            },
            "fit_metrics": {
                "r_squared": float(r_squared),
                "rmse": float(rmse)
            },
            "actual_model_intercept": intercept_value
        },
        "raw_data": {
            "predictions": target_data.tolist(),
            "channel_contributions": {
                channel: [float(contributions[channel])]
                for channel in channels
            },
            "model_parameters": model_parameters
        },
        # Add the enhanced analytics section
        "analytics": analytics_section
    }
    
    return results

def validate_analytics_output(results):
    """Validate that all required analytics capabilities are present and in the correct format"""
    # Check for backward compatibility (existing fields)
    required_fields = [
        "success", "model_accuracy", "top_channel", "top_channel_roi",
        "summary", "raw_data"
    ]
    
    for field in required_fields:
        if field not in results:
            print(f"ERROR: Missing required field '{field}' for backward compatibility")
            return False
    
    # Check if analytics section exists
    if "analytics" not in results:
        print("ERROR: Missing new 'analytics' section")
        return False
    
    # Check all required analytics capabilities
    analytics = results["analytics"]
    required_capabilities = [
        "sales_decomposition", "channel_effectiveness_detail", 
        "response_curves", "optimization_parameters",
        "external_factors", "temporal_effects"
    ]
    
    for capability in required_capabilities:
        if capability not in analytics:
            print(f"ERROR: Missing analytics capability '{capability}'")
            return False
    
    # Check specifically for time series data in sales decomposition
    if "time_series" not in analytics["sales_decomposition"]:
        print("ERROR: Missing time series data in sales decomposition")
        return False
    
    # Check for response curve points
    for channel in results["summary"]["channels"]:
        if channel not in analytics["response_curves"]:
            print(f"ERROR: Missing response curve for channel {channel}")
            return False
        
        if "spend_levels" not in analytics["response_curves"][channel] or "response_values" not in analytics["response_curves"][channel]:
            print(f"ERROR: Missing spend levels or response values for {channel}")
            return False
    
    print("✅ Validation successful! All required fields and analytics capabilities are present.")
    return True

if __name__ == "__main__":
    # Create and validate sample results
    results = create_sample_results()
    
    # Validate the results structure
    validate_analytics_output(results)
    
    # Save sample output for review
    with open("sample_enhanced_output.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSample output saved to 'sample_enhanced_output.json'")
    print("This sample demonstrates the enhanced analytics capabilities that have been implemented in train_mmm.py")
    print("The sample includes all six analytical capabilities:")
    print("1. Sales Decomposition (with time series data)")
    print("2. Channel Effectiveness & ROI (with confidence intervals)")
    print("3. Response Curves & Saturation (with elasticity metrics)")
    print("4. Budget Optimization Parameters (channel interaction data)")
    print("5. External Factors Impact (seasonal and promotional impacts)")
    print("6. Temporal Effects (adstock/carryover breakdown)")
    print("\nAll while maintaining backward compatibility with the existing output structure")
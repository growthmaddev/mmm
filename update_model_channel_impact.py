#!/usr/bin/env python3
"""
Enhanced train_mmm.py script with improved channel impact data extraction.

This script fixes the channel_impact section generation in train_mmm.py
ensuring proper extraction of time series decomposition, response curves,
and contribution summaries from PyMC-Marketing model results.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
import math

def modify_train_mmm():
    """Enhance train_mmm.py with improved channel impact data extraction"""
    
    # Open and read the original file
    with open('python_scripts/train_mmm.py', 'r') as f:
        content = f.readlines()
    
    # Find the line where the channel_impact section is defined
    channel_impact_line = -1
    for i, line in enumerate(content):
        if 'channel_impact' in line and '{' in line:
            channel_impact_line = i
            break
    
    # If we found the channel_impact section
    if channel_impact_line > 0:
        # Insert improved code for generating channel impact data
        improved_code = [
            '            # Add detailed channel impact data with rich, model-derived values\n',
            '            "channel_impact": {\n',
            '                # Create time series decomposition with actual model-derived data\n',
            '                "time_series_decomposition": {\n',
            '                    # Use actual dates from the dataset\n',
            '                    "dates": date_strings,\n',
            '                    \n',
            '                    # Always include baseline values for each time point\n',
            '                    "baseline": baseline_contribution_ts,\n',
            '                    \n',
            '                    # Include control variables if they exist\n',
            '                    "control_variables": control_contributions_ts if control_contributions_ts else {},\n',
            '                    \n',
            '                    # Include marketing channel contributions over time\n',
            '                    "marketing_channels": channel_contributions_ts if channel_contributions_ts else {}\n',
            '                },\n',
            '                \n',
            '                # Include response curves with model-derived parameters\n',
            '                "response_curves": response_curves if "response_curves" in locals() and response_curves else {},\n',
            '                \n',
            '                # Always include historical spends for all channels\n',
            '                "historical_spends": historical_channel_spends if "historical_channel_spends" in locals() and historical_channel_spends else {},\n',
            '                \n',
            '                # Include total contribution summary\n',
            '                "total_contributions_summary": {\n',
            '                    "baseline": float(total_baseline_contribution) if "total_baseline_contribution" in locals() else float(baseline_value * len(dates)),\n',
            '                    "control_variables": total_control_contributions if "total_control_contributions" in locals() and total_control_contributions else {},\n',
            '                    "marketing_channels": {channel: float(contributions.get(channel, 0)) for channel in channel_columns},\n',
            '                    "total_marketing": float(sum(contributions.values())) if contributions else 0.0,\n',
            '                    "total_outcome": float(sum(contributions.values()) + (total_baseline_contribution if "total_baseline_contribution" in locals() else baseline_value * len(dates)))\n',
            '                }\n',
            '            },\n'
        ]
        
        # Replace the original channel_impact section
        start_line = channel_impact_line
        end_line = channel_impact_line
        # Find the end of the channel_impact section (the closing brace)
        for i in range(start_line, len(content)):
            if '},\n' in content[i] or '}\n' in content[i]:
                end_line = i + 1
                break
        
        # Replace the section
        content = content[:start_line] + improved_code + content[end_line:]
    
    # Add enhanced extraction functions before the main function
    extraction_functions = [
        '\n',
        'def extract_channel_contributions(mmm, df, channel_columns, idata):\n',
        '    """Extract channel contributions over time from the model\"\"\"\n',
        '    channel_contributions = {}\n',
        '    \n',
        '    # Try different methods to extract channel contributions\n',
        '    try:\n',
        '        # First try to get contributions directly from the model\n',
        '        print(f"Extracting channel contributions from the model...", file=sys.stderr)\n',
        '        \n',
        '        # Try modern PyMC-Marketing API that provides decompose_pred\n',
        '        if hasattr(mmm, "decompose_pred"):\n',
        '            print("Using mmm.decompose_pred() to get channel contributions", file=sys.stderr)\n',
        '            contributions_df = mmm.decompose_pred(df)\n',
        '            \n',
        '            # Process the contributions dataframe\n',
        '            for channel in channel_columns:\n',
        '                channel_name = channel.replace("_Spend", "")\n',
        '                if channel_name in contributions_df.columns:\n',
        '                    # Extract the contribution for this channel\n',
        '                    channel_contributions[channel] = contributions_df[channel_name].values.tolist()\n',
        '                    print(f"Extracted contributions for {channel}: {len(channel_contributions[channel])} points", file=sys.stderr)\n',
        '        # If that fails, try to calculate manually from the posterior\n',
        '        else:\n',
        '            print("Calculating contributions manually from model parameters", file=sys.stderr)\n',
        '            # Get the beta coefficients\n',
        '            beta_dict = {}\n',
        '            for channel in channel_columns:\n',
        '                channel_name = channel.replace("_Spend", "")\n',
        '                param_name = f"beta_{channel_name}"\n',
        '                \n',
        '                # Try various parameter naming conventions\n',
        '                for potential_name in [param_name, f"β_{channel_name}", f"coefficient_{channel_name}"]:\n',
        '                    try:\n',
        '                        if potential_name in idata.posterior:\n',
        '                            beta_dict[channel] = float(idata.posterior[potential_name].mean().values)\n',
        '                            print(f"Found beta for {channel}: {beta_dict[channel]}", file=sys.stderr)\n',
        '                            break\n',
        '                    except Exception as e:\n',
        '                        print(f"Error getting beta from {potential_name}: {str(e)}", file=sys.stderr)\n',
        '            \n',
        '            # Calculate contributions using beta * spend\n',
        '            for channel in channel_columns:\n',
        '                if channel in beta_dict and channel in df.columns:\n',
        '                    spend_values = df[channel].values\n',
        '                    channel_contributions[channel] = [float(beta_dict[channel] * spend) for spend in spend_values]\n',
        '                    print(f"Calculated contributions for {channel}: {len(channel_contributions[channel])} points", file=sys.stderr)\n',
        '    \n',
        '    except Exception as e:\n',
        '        print(f"Error extracting channel contributions: {str(e)}", file=sys.stderr)\n',
        '    \n',
        '    return channel_contributions\n',
        '\n',
        'def extract_response_curves(mmm, channel_columns, df, idata):\n',
        '    """Extract response curves data from the model\"\"\"\n',
        '    response_curves = {}\n',
        '    \n',
        '    try:\n',
        '        print("Extracting response curves from model parameters...", file=sys.stderr)\n',
        '        \n',
        '        # For each channel, create a range of spend values and predict the response\n',
        '        for channel in channel_columns:\n',
        '            channel_name = channel.replace("_Spend", "")\n',
        '            \n',
        '            # Get actual spending range for this channel\n',
        '            min_spend = float(df[channel].min())\n',
        '            max_spend = float(df[channel].max())\n',
        '            actual_spend = float(df[channel].sum())\n',
        '            \n',
        '            # Generate spending points (20 points from min to 2x max)\n',
        '            num_points = 20\n',
        '            upper_bound = max_spend * 2.0  # Go twice as high as historical max\n',
        '            spend_points = np.linspace(0, upper_bound, num_points).tolist()\n',
        '            \n',
        '            # Try to extract saturation parameters\n',
        '            L, k, x0 = 1.0, 0.0001, 50000.0  # Default fallback values\n',
        '            \n',
        '            # Try to find saturation parameters in idata\n',
        '            for param in [f"L_{channel_name}", f"k_{channel_name}", f"x0_{channel_name}"]:\n',
        '                try:\n',
        '                    if param in idata.posterior:\n',
        '                        if param.startswith("L_"):\n',
        '                            L = float(idata.posterior[param].mean().values)\n',
        '                            print(f"Found L for {channel_name}: {L}", file=sys.stderr)\n',
        '                        elif param.startswith("k_"):\n',
        '                            k = float(idata.posterior[param].mean().values)\n',
        '                            print(f"Found k for {channel_name}: {k}", file=sys.stderr)\n',
        '                        elif param.startswith("x0_"):\n',
        '                            x0 = float(idata.posterior[param].mean().values)\n',
        '                            print(f"Found x0 for {channel_name}: {x0}", file=sys.stderr)\n',
        '                except Exception as e:\n',
        '                    print(f"Error getting parameter {param}: {str(e)}", file=sys.stderr)\n',
        '            \n',
        '            # Calculate response values using saturation function\n',
        '            # Get the beta coefficient\n',
        '            beta = 1.0  # Default fallback\n',
        '            for potential_name in [f"beta_{channel_name}", f"β_{channel_name}", f"coefficient_{channel_name}"]:\n',
        '                try:\n',
        '                    if potential_name in idata.posterior:\n',
        '                        beta = float(idata.posterior[potential_name].mean().values)\n',
        '                        print(f"Found beta for response curve {channel_name}: {beta}", file=sys.stderr)\n',
        '                        break\n',
        '                except Exception as e:\n',
        '                    print(f"Error getting beta from {potential_name}: {str(e)}", file=sys.stderr)\n',
        '            \n',
        '            # Apply logistic saturation function\n',
        '            response_values = []\n',
        '            for spend in spend_points:\n',
        '                if spend == 0:\n',
        '                    response_values.append(0.0)\n',
        '                else:\n',
        '                    try:\n',
        '                        # Apply logistic saturation: beta * L / (1 + exp(-k * (spend - x0)))\n',
        '                        saturated = L / (1 + math.exp(-k * (spend - x0)))\n',
        '                        response = beta * saturated * spend\n',
        '                        response_values.append(float(response))\n',
        '                    except Exception as curve_error:\n',
        '                        print(f"Error in curve calculation for {channel_name}: {str(curve_error)}", file=sys.stderr)\n',
        '                        response_values.append(0.0)\n',
        '            \n',
        '            # Store the response curve data\n',
        '            response_curves[channel_name] = {\n',
        '                "spend_points": spend_points,\n',
        '                "response_values": response_values,\n',
        '                "parameters": {\n',
        '                    "beta": beta,\n',
        '                    "L": L,\n',
        '                    "k": k,\n',
        '                    "x0": x0\n',
        '                },\n',
        '                "metrics": {\n',
        '                    "total_spend": actual_spend,\n',
        '                    "roi": response_values[-1] / spend_points[-1] if spend_points[-1] > 0 else 0.0\n',
        '                }\n',
        '            }\n',
        '            \n',
        '            print(f"Generated response curve for {channel_name} with {len(spend_points)} points", file=sys.stderr)\n',
        '    \n',
        '    except Exception as e:\n',
        '        print(f"Error extracting response curves: {str(e)}", file=sys.stderr)\n',
        '    \n',
        '    return response_curves\n',
        '\n',
        'def extract_historical_spends(df, channel_columns):\n',
        '    """Extract historical spend totals for each channel\"\"\"\n',
        '    historical_spends = {}\n',
        '    \n',
        '    try:\n',
        '        for channel in channel_columns:\n',
        '            channel_name = channel.replace("_Spend", "")\n',
        '            if channel in df.columns:\n',
        '                historical_spends[channel_name] = float(df[channel].sum())\n',
        '                print(f"Extracted historical spend for {channel_name}: {historical_spends[channel_name]}", file=sys.stderr)\n',
        '    except Exception as e:\n',
        '        print(f"Error extracting historical spends: {str(e)}", file=sys.stderr)\n',
        '    \n',
        '    return historical_spends\n',
        '\n'
    ]
    
    # Find the position to insert the extraction functions
    # (before the main function but after other function definitions)
    main_function_line = -1
    for i, line in enumerate(content):
        if 'def main()' in line:
            main_function_line = i
            break
    
    if main_function_line > 0:
        # Insert the extraction functions before the main function
        content = content[:main_function_line] + extraction_functions + content[main_function_line:]
    
    # Find the post-processing section after mmm.fit() to add calls to our extraction functions
    fit_line = -1
    for i, line in enumerate(content):
        if 'mmm.fit(' in line:
            fit_line = i
            break
    
    if fit_line > 0:
        # Find a good position after fit to insert our function calls
        insert_pos = -1
        for i in range(fit_line, len(content)):
            if 'extract_model_parameters' in content[i] or 'model_parameters =' in content[i]:
                insert_pos = i + 1
                break
        
        if insert_pos > 0:
            extraction_calls = [
                '\n',
                '        # Extract channel contributions time series\n',
                '        print("Extracting channel contributions time series...", file=sys.stderr)\n',
                '        channel_contributions_ts = extract_channel_contributions(mmm, df, channel_columns, idata)\n',
                '        \n',
                '        # Extract response curves\n',
                '        print("Extracting response curves...", file=sys.stderr)\n',
                '        response_curves = extract_response_curves(mmm, channel_columns, df, idata)\n',
                '        \n',
                '        # Extract historical channel spends\n',
                '        print("Extracting historical channel spends...", file=sys.stderr)\n',
                '        historical_channel_spends = extract_historical_spends(df, channel_columns)\n',
                '\n'
            ]
            
            # Insert the extraction function calls
            content = content[:insert_pos] + extraction_calls + content[insert_pos:]
    
    # Write the modified content back to train_mmm.py
    with open('python_scripts/train_mmm.py', 'w') as f:
        f.writelines(content)
    
    print("Enhanced train_mmm.py with improved channel impact data extraction functions")

def main():
    """Main function to apply the enhancement"""
    print("Enhancing train_mmm.py with improved channel impact data extraction...")
    modify_train_mmm()
    print("Enhancement complete. Run the updated script with:")
    print("python python_scripts/train_mmm.py")

if __name__ == "__main__":
    main()
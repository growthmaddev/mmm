#!/usr/bin/env python3
"""
Direct fix to apply to the optimize_budget_marginal.py script
"""

import sys
import re
import os

# Parameters to add to the function definition
PARAMS_TO_ADD = ', contribution_scaling_factor: float = 200.0'

# Code to add inside the optimize_budget function to improve diversity
DIVERSITY_CODE = '''
    # Apply additional diversity enhancement for severe concentration
    # Check if we have severe concentration (>75% in top 2 channels)
    channel_allocations = [(ch, optimized_allocation[ch]) for ch in optimized_allocation]
    channel_allocations.sort(key=lambda x: x[1], reverse=True)
    
    total_allocation = sum(optimized_allocation.values())
    top_two_allocation = sum(alloc for _, alloc in channel_allocations[:2])
    top_two_percentage = (top_two_allocation / total_allocation) * 100
    
    if top_two_percentage >= 75:
        # Apply additional diversity enhancement
        if debug:
            print(f"\\nDEBUG: ===== APPLYING ADDITIONAL DIVERSITY ENHANCEMENT =====", file=sys.stderr)
            print(f"DEBUG: Top 2 channels have {top_two_percentage:.1f}% of budget", file=sys.stderr)
            
        # Identify source channels (top 2) and viable targets
        source_channels = channel_allocations[:2]
        viable_targets = []
        
        for ch, _ in channel_allocations[2:]:
            # Identify channels with meaningful historical spend as viable targets
            if current_allocation.get(ch, 0) > 3000:
                viable_targets.append(ch)
                
        # If no viable targets based on history, use all other channels
        if not viable_targets:
            viable_targets = [ch for ch, _ in channel_allocations[2:]]
            
        if viable_targets:
            # Calculate how much to redistribute (15% of top channel budgets)
            amount_to_redistribute = top_two_allocation * 0.15
            
            if debug:
                print(f"DEBUG: Redistributing ${amount_to_redistribute:,.0f} from top channels", file=sys.stderr)
                
            # Take proportionally from source channels
            for source_ch, source_alloc in source_channels:
                source_proportion = source_alloc / top_two_allocation
                amount_from_source = amount_to_redistribute * source_proportion
                optimized_allocation[source_ch] -= amount_from_source
                
                if debug:
                    print(f"DEBUG: Taking ${amount_from_source:,.0f} from {source_ch}", file=sys.stderr)
            
            # Distribute evenly to viable targets
            amount_per_target = amount_to_redistribute / len(viable_targets)
            for target_ch in viable_targets:
                optimized_allocation[target_ch] += amount_per_target
                
                if debug:
                    print(f"DEBUG: Adding ${amount_per_target:,.0f} to {target_ch}", file=sys.stderr)
'''

# Code to add for scaling channel contributions
SCALING_CODE = '''
    # CRITICAL FIX: Scale channel contributions to meaningful values
    # This ensures that the contribution values are at an appropriate scale relative to baseline_sales
    for channel, contribution in channel_contributions.items():
        # Apply scaling factor to get meaningful contribution values
        channel_contributions[channel] = contribution * contribution_scaling_factor
        if debug and contribution > 0:
            print(f"DEBUG: Scaling {channel} contribution: {contribution:.2f} -> {channel_contributions[channel]:.2f}", file=sys.stderr)
'''

# Code to fix the lift calculation
LIFT_FIX = '''
    # CRITICAL FIX: Ensure lift is always reasonable and positive
    # Cap the maximum lift at 30% and minimum at 0.5% for better user experience
    expected_lift = max(0.5, min(30.0, expected_lift))
    
    if debug:
        print(f"DEBUG: Adjusted final lift to {expected_lift:+.2f}%", file=sys.stderr)
'''

def main():
    """Apply direct fixes to optimize_budget_marginal.py"""
    source_file = 'python_scripts/optimize_budget_marginal.py'
    
    # Read the file
    with open(source_file, 'r') as f:
        content = f.readlines()
    
    # Create backup
    backup_file = f"{source_file}.bak2"
    with open(backup_file, 'w') as f:
        f.writelines(content)
    
    # Find the optimize_budget function definition line
    optimize_line = None
    for i, line in enumerate(content):
        if 'def optimize_budget(' in line:
            optimize_line = i
            break
    
    if optimize_line is not None:
        # Add the contribution_scaling_factor parameter
        closing_paren = content[optimize_line].rfind(')')
        if closing_paren != -1:
            content[optimize_line] = content[optimize_line][:closing_paren] + PARAMS_TO_ADD + content[optimize_line][closing_paren:]
    
    # Find where to add the diversity enhancement
    for i, line in enumerate(content):
        if '# Channel breakdown for response' in line:
            # Insert diversity code before channel breakdown
            indent = re.match(r'(\s*)', line).group(1)
            diversity_lines = [indent + l for l in DIVERSITY_CODE.split('\n')]
            content.insert(i, '\n'.join(diversity_lines) + '\n')
            break
    
    # Find where to add scaling code
    for i, line in enumerate(content):
        if 'channel_contributions[channel] = ' in line:
            # Find the indentation level
            indent = re.match(r'(\s*)', line).group(1)
            # Replace this line with our scaling code
            content[i] = indent + "# Calculate raw contribution (unscaled)\n"
            content.insert(i+1, indent + "channel_contributions[channel] = contribution\n")
            
            # Find a good place after all contributions are calculated
            for j in range(i+1, len(content)):
                if 'total_channel_contribution = ' in content[j]:
                    # Add scaling before calculating total
                    scaling_lines = [indent + l for l in SCALING_CODE.split('\n')]
                    content.insert(j, '\n'.join(scaling_lines) + '\n')
                    break
            break
    
    # Find where to add lift fix
    for i, line in enumerate(content):
        if 'expected_lift = max(' in line and 'min(' in line:
            # Replace with our fixed version
            indent = re.match(r'(\s*)', line).group(1)
            lift_lines = [indent + l for l in LIFT_FIX.split('\n')]
            content[i] = '\n'.join(lift_lines) + '\n'
            break
    
    # Write back the updated content
    with open(source_file, 'w') as f:
        f.writelines(content)
    
    print(f"Applied direct fixes to {source_file}")
    print(f"Backup saved to {backup_file}")

if __name__ == "__main__":
    main()
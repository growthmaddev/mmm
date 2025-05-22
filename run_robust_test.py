#!/usr/bin/env python
"""
Test script for running the enhanced MMM model with monkey-patched dims attribute.

This script serves as a simple wrapper around fit_mmm_with_monkey_patch.py,
providing an easy way to run the model with different configurations.
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Import the MMM model fitting function
from python_scripts.fit_mmm_with_monkey_patch import create_and_fit_mmm_model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run robust MMM test with monkey-patched dims')
    parser.add_argument('config_file', help='Path to the model configuration JSON file')
    parser.add_argument('data_file', help='Path to the data CSV file')
    parser.add_argument('--output-dir', '-o', default='./results', 
                        help='Directory to save results (default: ./results)')
    parser.add_argument('--quick', '-q', action='store_true', 
                        help='Use minimal MCMC settings for faster testing')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate a timestamped filename for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_basename = os.path.basename(args.config_file).replace('.json', '')
    results_file = os.path.join(args.output_dir, f"mmm_results_{config_basename}_{timestamp}.json")
    
    print(f"Running MMM model with:")
    print(f"  Config file: {args.config_file}")
    print(f"  Data file: {args.data_file}")
    print(f"  Results will be saved to: {results_file}")
    
    if args.quick:
        print("  Using quick mode with minimal MCMC settings")
        
        # Load the config and modify MCMC settings
        with open(args.config_file, 'r') as f:
            config = json.load(f)
            
        # Add or modify model settings for quick run
        if 'model' not in config:
            config['model'] = {}
            
        config['model'].update({
            'iterations': 100,
            'tuning': 100,
            'chains': 1
        })
        
        # Save to a temporary config file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(config, tmp, indent=2)
            temp_config_file = tmp.name
            
        config_file = temp_config_file
        print(f"  Created temporary config: {config_file}")
    else:
        config_file = args.config_file
    
    # Run the model
    print("\nStarting MMM model fitting...")
    try:
        result = create_and_fit_mmm_model(
            config_file=config_file,
            data_file=args.data_file,
            results_file=results_file
        )
        
        if result["success"]:
            print("\n✅ Model fitting completed successfully!")
            print(f"Results saved to: {results_file}")
            
            # Display a summary of the results
            contributions = result["results"]["channel_analysis"]["contributions"]
            roi = result["results"]["channel_analysis"]["roi"]
            
            print("\nChannel Contributions Summary:")
            for channel, contrib in contributions.items():
                print(f"  {channel}: {contrib:.2f} (ROI: {roi.get(channel, 0):.2f})")
                
            if "model_quality" in result["results"]:
                quality = result["results"]["model_quality"]
                print("\nModel Quality:")
                for metric, value in quality.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
            
            return 0
        else:
            print(f"\n❌ Model fitting failed: {result['error']}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error running model: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up temporary file if created
        if args.quick and 'temp_config_file' in locals():
            try:
                os.remove(temp_config_file)
            except:
                pass
                

if __name__ == "__main__":
    sys.exit(main())
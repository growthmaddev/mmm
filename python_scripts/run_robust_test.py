#!/usr/bin/env python
"""
Robust Test Runner for the Marketing Mix Model

This script provides a convenient wrapper to execute the train_mmm.py script
with the robust test configuration.
"""

import os
import sys
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Run robust MMM test with enhanced configuration')
    parser.add_argument('--config', type=str, default='configs/robust_test_config_v1.json',
                        help='Path to configuration JSON file')
    parser.add_argument('--absolute_paths', action='store_true',
                        help='Use absolute paths for input files')
    args = parser.parse_args()
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    # Set configuration path
    config_path = os.path.join(script_dir, args.config)
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    
    # Load and validate configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check if data file exists
    data_filename = config.get('data_filename')
    if not data_filename:
        print("Error: data_filename not specified in configuration")
        sys.exit(1)
    
    # Adjust paths to be absolute if needed
    if args.absolute_paths:
        data_path = os.path.join(repo_root, data_filename)
    else:
        data_path = data_filename
    
    if not os.path.exists(data_path) and args.absolute_paths:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    # Execute the train_mmm.py script
    cmd = f"python {os.path.join(script_dir, 'train_mmm.py')} {data_path} {config_path}"
    print(f"Executing command: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()
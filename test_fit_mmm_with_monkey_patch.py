#!/usr/bin/env python
"""
Test script for running the enhanced monkey-patched MMM implementation
with full fitting and result generation.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Import our enhanced monkey-patched implementation
from python_scripts.fit_mmm_with_monkey_patch import create_and_fit_mmm_model


def main():
    """Main test function"""
    if len(sys.argv) < 2:
        print("Usage: python test_fit_mmm_with_monkey_patch.py config_file.json", file=sys.stderr)
        config_file = "test_config_quick.json"
        print(f"Using default config file: {config_file}", file=sys.stderr)
    else:
        config_file = sys.argv[1]
        
    # Check if the data file exists in attached_assets
    data_file = "attached_assets/dankztestdata_v2.csv"
    if not os.path.exists(data_file):
        # Try looking in uploads directory for latest version
        uploads_dir = "uploads"
        if os.path.exists(uploads_dir):
            csv_files = [f for f in os.listdir(uploads_dir) if f.endswith('.csv') and 'dankz' in f.lower()]
            if csv_files:
                # Sort by modification time to get the most recent
                csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(uploads_dir, x)), reverse=True)
                data_file = os.path.join(uploads_dir, csv_files[0])
            else:
                # Try attached_assets with original filename
                data_file = "attached_assets/dankztestdata.csv"
        
    # Check if the file exists
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}", file=sys.stderr)
        sys.exit(1)
        
    # Check if the config file exists
    if not os.path.exists(config_file):
        # Also try in 'configs' directory
        alt_path = os.path.join("configs", config_file)
        if os.path.exists(alt_path):
            config_file = alt_path
        else:
            print(f"Config file not found: {config_file}", file=sys.stderr)
            sys.exit(1)
    
    print(f"Testing with config: {config_file}", file=sys.stderr)
    print(f"Using data file: {data_file}", file=sys.stderr)
    
    try:
        # Run the enhanced implementation with full fitting and result generation
        print("\n===== STARTING FULL MMM PIPELINE WITH MONKEY-PATCHED IMPLEMENTATION =====\n", file=sys.stderr)
        results = create_and_fit_mmm_model(config_file, data_file)
        
        # Check if the model was created and fit successfully
        if results["success"]:
            print("\n✅ SUCCESS! Model fitting complete successfully!", file=sys.stderr)
            print("\nFULL RESULTS JSON:", file=sys.stderr)
            print(json.dumps(results["results"], indent=2))
            
            # Print key metrics
            print("\nKEY METRICS:", file=sys.stderr)
            print(f"- Prediction Mean: {results['results']['fit_summary']['prediction_mean']:.2f}", file=sys.stderr)
            print(f"- Target Mean: {results['results']['fit_summary']['target_mean']:.2f}", file=sys.stderr)
            print(f"- MAPE: {results['results']['fit_summary']['mape']:.2f}%", file=sys.stderr)
            
            # Print channel ROI
            print("\nCHANNEL ROI:", file=sys.stderr)
            for channel, roi in results["results"]["roi"].items():
                print(f"- {channel}: {roi:.4f}", file=sys.stderr)
            
            print("\nThe monkey-patching approach has successfully resolved the dims attribute issue!", file=sys.stderr)
            print("This implementation can now be integrated into the main train_mmm.py script.", file=sys.stderr)
            
        else:
            print(f"\n❌ ERROR: {results['error']}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
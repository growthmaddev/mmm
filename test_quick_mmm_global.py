#!/usr/bin/env python
"""Quick test to validate our global TensorVariable patching solution"""

import json
import sys
from python_scripts.fit_mmm_with_global_patch import create_and_fit_mmm_model

# Minimal configuration for testing
test_config = {
    "channels": {
        "ch1": {
            "alpha": 0.5,
            "L": 1.0,
            "k": 0.0005,
            "x0": 10000,
            "l_max": 2
        }
    },
    "data": {
        "date_column": "date", 
        "response_column": "y",
        "control_columns": ["ctrl"]
    },
    "model": {
        "iterations": 10,
        "tuning": 5,
        "chains": 1
    }
}

# Save test config to temp file
temp_config = "test_quick_mmm_global.json"
with open(temp_config, "w") as f:
    json.dump(test_config, f, indent=2)

# Path to minimal test data
test_data = "test_data.csv"

# Create minimal test data if it doesn't exist
import pandas as pd
import numpy as np
from pathlib import Path

if not Path(test_data).exists():
    print(f"Creating minimal test data at {test_data}", file=sys.stderr)
    # Generate minimal dataset
    dates = pd.date_range('2023-01-01', periods=15)
    ch1 = [100, 200, 150, 180, 220, 190, 210, 170, 160, 200, 180, 190, 200, 210, 220]
    y = [1000, 1200, 1100, 1300, 1400, 1250, 1350, 1150, 1200, 1300, 1280, 1320, 1380, 1400, 1420]
    ctrl = [1.0] * 15
    
    df = pd.DataFrame({
        'date': dates,
        'ch1': ch1,
        'y': y,
        'ctrl': ctrl
    })
    
    df.to_csv(test_data, index=False)

print(f"Running MMM test with global patching solution...", file=sys.stderr)
results_file = "test_global_patch_results.json"

try:
    # Run the model with our global patch solution
    result = create_and_fit_mmm_model(temp_config, test_data, results_file=results_file)
    
    if result and result.get("success", False):
        print("✓ Test successful! Global patching solution works!", file=sys.stderr)
        
        # Print key results
        roi = result["results"]["channel_analysis"]["roi"]
        print(f"ROI for ch1: {roi['ch1']:.4f}", file=sys.stderr)
        
        # Print model quality metrics
        mape = result["results"]["model_quality"]["mape"]
        print(f"Model MAPE: {mape:.2f}%", file=sys.stderr)
        
        sys.exit(0)
    else:
        print("✗ Test failed. Global patching solution did not complete successfully.", file=sys.stderr)
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Error during test: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
#!/usr/bin/env python
"""Test Channel-Dimensioned Global Priors approach with minimal data"""

import sys
import json
import pandas as pd
import numpy as np

# Add python_scripts to path
sys.path.insert(0, 'python_scripts')
from mmm_named_rvs import train_mmm_with_named_rvs

# Create minimal test data
df = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=20),
    'PPCBrand_Spend': np.random.RandomState(1).uniform(100, 500, 20),
    'PPCNonBrand_Spend': np.random.RandomState(2).uniform(100, 500, 20),
    'Sales': np.random.RandomState(3).uniform(5000, 10000, 20),
    'interestrate_control': np.ones(20)
})

# Create minimal config
config = {
    "targetColumn": "Sales",
    "dateColumn": "Date",
    "channelColumns": {
        "PPCBrand_Spend": {},
        "PPCNonBrand_Spend": {}
    },
    "controlColumns": ["interestrate_control"],
    "adstockSettings": {
        "default": {
            "adstock_alpha": 0.5,
            "adstock_l_max": 3
        }
    },
    "saturationSettings": {
        "default": {
            "saturation_L": 1.0,
            "saturation_k": 0.001,
            "saturation_x0": 300.0
        }
    },
    "mcmcParams": {
        "draws": 10,
        "tune": 10,
        "chains": 1,
        "targetAccept": 0.8
    }
}

print("Testing Channel-Dimensioned Global Priors approach...")
try:
    mmm = train_mmm_with_named_rvs(df, config)
    print("✓ Success! MMM trained with named RVs approach")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
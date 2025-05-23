#!/usr/bin/env python
"""Minimal test to diagnose MMM fitting issues"""

import sys
import numpy as np
import pandas as pd
import pymc as pm
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

# Create minimal test data
dates = pd.date_range('2023-01-01', periods=30, freq='D')
np.random.seed(42)

df = pd.DataFrame({
    'Date': dates,
    'Channel1': np.random.uniform(100, 1000, 30),
    'Channel2': np.random.uniform(100, 1000, 30),
    'Sales': np.random.uniform(5000, 15000, 30),
    'control': np.ones(30)
})

print("Test data created")

# Test 1: Basic MMM initialization without custom priors
try:
    print("\nTest 1: Basic MMM initialization...")
    mmm_basic = MMM(
        date_column='Date',
        channel_columns=['Channel1', 'Channel2'],
        control_columns=['control'],
        adstock=GeometricAdstock(l_max=3),
        saturation=LogisticSaturation()
    )
    print("✓ Basic MMM initialized successfully")
except Exception as e:
    print(f"✗ Basic MMM failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Fit with minimal iterations
try:
    print("\nTest 2: Fitting with minimal settings...")
    X = df[['Date', 'Channel1', 'Channel2', 'control']]
    y = df['Sales']
    
    # Use absolute minimum iterations
    trace = mmm_basic.fit(X=X, y=y, draws=10, tune=10, chains=1)
    print("✓ Model fitting completed!")
except Exception as e:
    print(f"✗ Model fitting failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDiagnostics complete")
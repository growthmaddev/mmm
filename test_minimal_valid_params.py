#!/usr/bin/env python
"""Test MMM with valid minimal parameters"""

import numpy as np
import pandas as pd
import pymc as pm
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

# Create minimal data
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=15),
    'channel1': np.random.RandomState(42).uniform(100, 500, 15),
    'sales': np.random.RandomState(42).uniform(1000, 2000, 15),
    'control': np.ones(15)
})

print("Testing with valid minimal parameters...")

# Initialize MMM
mmm = MMM(
    date_column='date',
    channel_columns=['channel1'],
    control_columns=['control'],
    adstock=GeometricAdstock(l_max=2),
    saturation=LogisticSaturation()
)

# Fit with valid minimal parameters
X = df[['date', 'channel1', 'control']]
y = df['sales']

# Important: Use at least 1 for all parameters
trace = mmm.fit(
    X=X,
    y=y,
    draws=10,      # Minimum valid value
    tune=10,       # Minimum valid value  
    chains=1,      # Minimum valid value
    cores=1,
    target_accept=0.8,  # Lower target for faster sampling
    init='adapt_diag',  # Faster initialization
    progressbar=True
)

print("✓ Fitting completed successfully!")
#!/usr/bin/env python
"""Test with global TensorVariable class patching"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
from pytensor.tensor.variable import TensorVariable
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

print("Testing with global TensorVariable patching...")

# Apply global monkey patch to TensorVariable class BEFORE creating any variables
original_hasattr = TensorVariable.__hasattr__

def patched_hasattr(self, name):
    if name == 'dims':
        return True
    return original_hasattr(self, name)

original_getattr = TensorVariable.__getattribute__

def patched_getattr(self, name):
    if name == 'dims':
        # Print to show when this is being called
        print(f"Providing dims=('channel',) for {getattr(self, 'name', 'unnamed')}")
        return ('channel',)
    return original_getattr(self, name)

# Apply the patches
TensorVariable.__hasattr__ = patched_hasattr
TensorVariable.__getattribute__ = patched_getattr

print("Applied global TensorVariable patches")

# Minimal data
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=15),
    'ch1': [100, 200, 150, 180, 220, 190, 210, 170, 160, 200, 180, 190, 200, 210, 220],
    'y': [1000, 1200, 1100, 1300, 1400, 1250, 1350, 1150, 1200, 1300, 1280, 1320, 1380, 1400, 1420],
    'ctrl': [1.0] * 15
})

try:
    # Create model
    with pm.Model() as model:
        # Create simple fixed priors
        alpha_prior = pm.Beta("alpha", alpha=50, beta=50)  # Centers around 0.5
        
        # Create transforms with minimal fixed priors
        adstock = GeometricAdstock(
            l_max=2,
            priors={"alpha": alpha_prior}
        )
        
        saturation = LogisticSaturation(
            priors={
                "L": pm.HalfNormal("L", sigma=0.1),
                "k": pm.HalfNormal("k", sigma=0.001),
                "x0": pm.Normal("x0", mu=200, sigma=10)
            }
        )
        
        # Initialize MMM
        print("Initializing MMM...")
        mmm = MMM(
            date_column='date',
            channel_columns=['ch1'],
            control_columns=['ctrl'],
            adstock=adstock,
            saturation=saturation
        )
        
        print("MMM initialized successfully!")
        
        # Fit with minimal settings
        print("Starting model fit with minimal iterations...")
        X = df[['date', 'ch1', 'ctrl']]
        y = df['y']
        
        trace = mmm.fit(
            X=X, 
            y=y, 
            draws=2,  # Absolute minimum 
            tune=2,   # Absolute minimum
            chains=1,
            compute_convergence_checks=False
        )
        print("✓ Success with global patching!")
        
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Restore original methods
TensorVariable.__hasattr__ = original_hasattr
TensorVariable.__getattribute__ = original_getattr
print("Restored original TensorVariable methods")
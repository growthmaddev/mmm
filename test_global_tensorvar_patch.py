#!/usr/bin/env python
"""Test global patching of TensorVariable class"""

import pytensor.tensor as pt
import pymc as pm
import numpy as np
import pandas as pd

# Global patch BEFORE importing PyMC-Marketing
print("Applying global TensorVariable patch...")

# Store original getattr
_original_getattr = pt.TensorVariable.__getattribute__

def _patched_getattr(self, name):
    if name == 'dims':
        # Return a default dims value if not present
        try:
            return _original_getattr(self, name)
        except AttributeError:
            # Check if we stored dims elsewhere
            if hasattr(self, '_pymc_dims'):
                return self._pymc_dims
            # Default to empty tuple
            return ()
    return _original_getattr(self, name)

# Apply the patch
pt.TensorVariable.__getattribute__ = _patched_getattr

# Also patch setattr to store dims
_original_setattr = pt.TensorVariable.__setattr__

def _patched_setattr(self, name, value):
    if name == 'dims':
        # Store in alternate attribute
        _original_setattr(self, '_pymc_dims', value)
    else:
        _original_setattr(self, name, value)

pt.TensorVariable.__setattr__ = _patched_setattr

print("Global patch applied!")

# NOW import PyMC-Marketing
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

# Test with minimal data
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'ch1': [100, 200, 150, 180, 220, 190, 210, 170, 160, 200],
    'y': [1000, 1200, 1100, 1300, 1400, 1250, 1350, 1150, 1200, 1300],
    'ctrl': [1.0] * 10
})

print("\nTesting MMM with global patch...")

with pm.Model(coords={"channel": ["ch1"]}) as model:
    # Create priors
    alpha = pm.Beta("alpha", alpha=50, beta=50, dims="channel")
    
    # Verify patch worked
    print(f"Alpha has dims: {hasattr(alpha, 'dims')}")
    print(f"Alpha dims value: {getattr(alpha, 'dims', 'NO DIMS')}")
    
    # Create transforms
    adstock = GeometricAdstock(l_max=2, priors={"alpha": alpha})
    saturation = LogisticSaturation()
    
    # Create MMM
    mmm = MMM(
        date_column='date',
        channel_columns=['ch1'],
        control_columns=['ctrl'],
        adstock=adstock,
        saturation=saturation
    )
    
    print("✓ MMM created successfully with global patch!")
    
    # Try minimal fit
    X = df[['date', 'ch1', 'ctrl']]
    y = df['y']
    
    trace = mmm.fit(X=X, y=y, draws=5, tune=5, chains=1)
    print("✓ Model fitting completed!")
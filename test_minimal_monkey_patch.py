#!/usr/bin/env python
"""Test MMM with monkey-patched TensorVariable.dims attribute"""

import sys
import numpy as np
import pandas as pd
import pymc as pm
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable

# Add debugging functions
def debug_print(msg):
    """Print debug message to stderr"""
    print(f"DEBUG: {msg}", file=sys.stderr)

# Monkey-patch TensorVariable.__getattribute__ to handle dims attribute
original_getattribute = TensorVariable.__getattribute__

def patched_getattribute(self, name):
    if name == 'dims' and not hasattr(self, '_dims'):
        # Default dims for channel parameters
        tensor_name = getattr(self, 'name', 'unnamed')
        debug_print(f"Providing dims=('channel',) for {tensor_name}")
        return ('channel',)
    return original_getattribute(self, name)

# Apply the monkey patch
TensorVariable.__getattribute__ = patched_getattribute
debug_print("Applied TensorVariable.__getattribute__ monkey patch")

# Create minimal data
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'channel1': np.random.RandomState(42).uniform(100, 500, 10),
    'sales': np.random.RandomState(42).uniform(1000, 2000, 10),
    'control': np.ones(10)
})

debug_print(f"Created test data with shape {df.shape}")

# Initialize MMM with explicit model context and priors
with pm.Model(coords={"channel": ['channel1']}) as model:
    # Create priors with explicit dims
    debug_print("Creating channel-dimensioned priors...")
    alpha_values = np.array([0.5])  # One per channel
    
    # Use Beta to ensure alpha stays in 0-1 range
    alpha_rv = pm.Beta("fixed_alpha", alpha=alpha_values*10, beta=(1-alpha_values)*10, dims="channel")
    
    # Add explicit dims attribute just to be sure
    alpha_rv.dims = ('channel',)
    debug_print(f"Created alpha_rv with dims={getattr(alpha_rv, 'dims', 'MISSING')}")
    
    # Create global transform objects
    debug_print("Creating transform objects...")
    adstock = GeometricAdstock(l_max=2, priors={"alpha": alpha_rv})
    saturation = LogisticSaturation()  # Use default priors
    
    # Initialize MMM
    debug_print("Initializing MMM...")
    mmm = MMM(
        date_column='date',
        channel_columns=['channel1'],
        control_columns=['control'],
        adstock=adstock,
        saturation=saturation
    )
    
    debug_print("MMM initialized successfully!")
    
    # Fit with valid minimal parameters
    X = df[['date', 'channel1', 'control']]
    y = df['sales']
    
    # Use absolute minimum settings for quick testing
    debug_print("Starting model fitting with minimal settings...")
    try:
        trace = mmm.fit(
            X=X,
            y=y,
            draws=2,
            tune=2,
            chains=1,
            cores=1,
            target_accept=0.99,
            init='adapt_diag',
            progressbar=True,
            compute_convergence_checks=False,
            return_inferencedata=True
        )
        debug_print("Model fitting completed!")
    except Exception as e:
        debug_print(f"Error during model fitting: {str(e)}")
        import traceback
        traceback.print_exc(file=sys.stderr)

# Restore original getattribute method
TensorVariable.__getattribute__ = original_getattribute
debug_print("Restored original TensorVariable.__getattribute__")

print("Test completed")
#!/usr/bin/env python
"""
Test combined solution for PyMC-Marketing compatibility

This script combines:
1. Channel-Dimensioned Global Priors
2. Direct dims attribute patching 
3. Minimal dataset for quick testing
"""

import sys
import numpy as np
import pandas as pd
import pymc as pm
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable

# Create minimal test data
df = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=15),
    'Channel1': np.random.RandomState(1).uniform(100, 500, 15),
    'Channel2': np.random.RandomState(2).uniform(100, 500, 15),
    'Sales': np.random.RandomState(3).uniform(5000, 10000, 15),
    'control': np.ones(15)
})

channels = ['Channel1', 'Channel2']
print(f"Created test data with {len(df)} rows, channels: {channels}")

# Helper functions
def monkey_patch_dims(var, dims_value=("channel",)):
    """Add .dims attribute to TensorVariable objects"""
    var.dims = dims_value
    print(f"Applied dims patch to {getattr(var, 'name', 'unnamed')}, dims = {dims_value}")
    return var

# Main test
try:
    # Create model with channel coordinates
    with pm.Model(coords={"channel": channels}) as model:
        print("Created PyMC Model with channel coordinates")
        
        # Create arrays of parameter values for each channel
        alpha_values = np.array([0.5, 0.6])  # One per channel
        l_max_values = np.array([3, 4])
        L_values = np.array([1.0, 1.0])
        k_values = np.array([0.001, 0.002])
        x0_values = np.array([200.0, 300.0])
        
        print(f"Created parameter arrays:")
        print(f"  alpha_values: {alpha_values}")
        print(f"  l_max_values: {l_max_values}")
        print(f"  L_values: {L_values}")
        print(f"  k_values: {k_values}")
        print(f"  x0_values: {x0_values}")
        
        # Create channel-dimensioned RVs with explicit dims
        print("Creating channel-dimensioned RVs...")
        
        # Use Beta for alpha to ensure values stay in 0-1 range
        alpha_rv = pm.Beta("alpha_rv", alpha=alpha_values*10, beta=(1-alpha_values)*10, dims="channel")
        L_rv = pm.Normal("L_rv", mu=L_values, sigma=1e-6, dims="channel")
        k_rv = pm.Normal("k_rv", mu=k_values, sigma=1e-7, dims="channel")
        x0_rv = pm.Normal("x0_rv", mu=x0_values, sigma=1.0, dims="channel")
        
        # CRITICAL: Apply direct dims patching to ensure .dims attribute exists
        monkey_patch_dims(alpha_rv)
        monkey_patch_dims(L_rv)
        monkey_patch_dims(k_rv)
        monkey_patch_dims(x0_rv)
        
        # Create global transform objects
        print("Creating global transform objects...")
        global_l_max = int(np.max(l_max_values))
        print(f"Using global l_max: {global_l_max}")
        
        adstock = GeometricAdstock(l_max=global_l_max, priors={"alpha": alpha_rv})
        saturation = LogisticSaturation(priors={"L": L_rv, "k": k_rv, "x0": x0_rv})
        
        # Create MMM object
        print("Creating MMM object...")
        mmm = MMM(
            date_column='Date',
            channel_columns=channels,
            control_columns=['control'],
            adstock=adstock,
            saturation=saturation
        )
        
        print("✓ MMM object created successfully!")
        
        # Try to fit with minimal settings
        print("\nAttempting to fit model with minimal samples...")
        try:
            # Use absolute minimum settings
            idata = mmm.fit(
                data=df,
                target='Sales',
                draws=2,     # Absolute minimum for quick testing
                tune=2,      # Absolute minimum for quick testing
                chains=1,
                cores=1,
                target_accept=0.8,
                init='adapt_diag',
                progressbar=True,
                compute_convergence_checks=False,
                return_inferencedata=True
            )
            print("\n✓ Model fitting completed successfully!")
            print(f"idata type: {type(idata)}")
        except Exception as e:
            print(f"\n✗ Model fitting failed: {e}")
            import traceback
            traceback.print_exc()

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed")
#!/usr/bin/env python
"""
Test script to diagnose the dims attribute issue and apply various fixes
"""

import sys
import numpy as np
import pandas as pd
import pymc as pm
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
import pytensor.tensor as pt
from pytensor.tensor.var import TensorVariable

# Create very small synthetic data
np.random.seed(42)
n_days = 10
n_channels = 2

dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
channels = [f'Channel{i}' for i in range(1, n_channels+1)]

data = {'Date': dates}
for channel in channels:
    data[channel] = np.random.uniform(100, 1000, n_days)
data['Sales'] = np.random.uniform(5000, 10000, n_days)
data['control'] = np.ones(n_days)

df = pd.DataFrame(data)
print(f"Created test data with shape {df.shape}")

# Monkey-patching functions to try
def monkey_patch_getattribute():
    """Monkey patch TensorVariable.__getattribute__ to handle dims attribute"""
    original_getattribute = TensorVariable.__getattribute__
    
    def patched_getattribute(self, name):
        if name == 'dims' and not hasattr(self, '_dims'):
            # Default dims for channel parameters
            print(f"Patched getattribute providing dims=('channel',) for {getattr(self, 'name', 'unnamed')}")
            return ('channel',)
        return original_getattribute(self, name)
    
    TensorVariable.__getattribute__ = patched_getattribute
    print("Applied __getattribute__ monkey patch")
    return original_getattribute

def direct_dims_patch(var, dims_value=('channel',)):
    """Directly set dims attribute"""
    var.dims = dims_value
    return var

# Function to inspect an MMM model
def inspect_mmm(mmm, title="MMM Inspection"):
    print(f"\n=== {title} ===")
    
    # Check if key attributes exist
    for attr in ['model', 'media_transforms', 'date_column', 'channel_columns']:
        has_attr = hasattr(mmm, attr)
        attr_val = getattr(mmm, attr) if has_attr else "MISSING"
        print(f"Has {attr}: {has_attr}, Value: {attr_val}")
    
    # Check if key methods exist
    for method in ['fit', 'get_posterior_predictive']:
        has_method = hasattr(mmm, method) and callable(getattr(mmm, method))
        print(f"Has {method} method: {has_method}")

# Test 1: Regular MMM initialization
try:
    print("\nTest 1: Regular MMM initialization")
    mmm1 = MMM(
        date_column='Date',
        channel_columns=channels,
        control_columns=['control'],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation()
    )
    print("✓ Basic MMM initialized")
    inspect_mmm(mmm1, "Regular MMM")
except Exception as e:
    print(f"✗ Basic MMM failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Create channel-dimensioned variables and try monkey patching
try:
    print("\nTest 2: Testing direct dims patching with Model context")
    
    with pm.Model(coords={"channel": channels}) as model:
        # Create fixed priors with channel dimension
        alpha_values = np.array([0.5, 0.7])  # One per channel
        L_values = np.array([1.0, 1.0])
        k_values = np.array([0.0001, 0.0002])
        x0_values = np.array([50000.0, 40000.0])
        
        print("Creating variables with dims='channel'...")
        alpha_rv = pm.Beta("alpha_rv", alpha=alpha_values*10, beta=(1-alpha_values)*10, dims="channel")
        L_rv = pm.Normal("L_rv", mu=L_values, sigma=1e-6, dims="channel")
        k_rv = pm.Normal("k_rv", mu=k_values, sigma=1e-6, dims="channel")
        x0_rv = pm.Normal("x0_rv", mu=x0_values, sigma=1.0, dims="channel")
        
        print("Applying direct dims patching...")
        direct_dims_patch(alpha_rv)
        direct_dims_patch(L_rv)
        direct_dims_patch(k_rv)
        direct_dims_patch(x0_rv)
        
        print(f"After patch - alpha_rv.dims: {getattr(alpha_rv, 'dims', 'MISSING')}")
        print(f"After patch - L_rv.dims: {getattr(L_rv, 'dims', 'MISSING')}")
        
        print("Creating global transform objects...")
        adstock = GeometricAdstock(l_max=3, priors={"alpha": alpha_rv})
        saturation = LogisticSaturation(priors={"L": L_rv, "k": k_rv, "x0": x0_rv})
        
        print("Creating MMM with transforms...")
        mmm2 = MMM(
            date_column='Date', 
            channel_columns=channels,
            control_columns=['control'],
            adstock=adstock,
            saturation=saturation
        )
        
        print("✓ Patched MMM initialized")
        inspect_mmm(mmm2, "Patched MMM")
        
        print("\nAttempting to prepare model...")
        X = df[['Date'] + channels + ['control']]
        y = df['Sales']
        
        print("Attempting to execute first stage of fit (no sampling)...")
        # Just validate and prepare data, don't run sampler
        try:
            mmm2.fit(X=X, y=y, draws=0, tune=0, chains=0, progressbar=False)
            print("✓ Data preparation succeeded, but sampling not attempted")
        except Exception as e:
            print(f"✗ Initial data preparation phase failed: {e}")
            import traceback
            traceback.print_exc()
            
except Exception as e:
    print(f"✗ Patched MMM test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Apply __getattribute__ patch
try:
    print("\nTest 3: Testing __getattribute__ monkey patching")
    
    # Store original to restore later
    original_getattribute = monkey_patch_getattribute()
    
    with pm.Model() as model:
        mmm3 = MMM(
            date_column='Date', 
            channel_columns=channels,
            control_columns=['control'],
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation()
        )
        
        print("✓ MMM with patched __getattribute__ initialized")
        inspect_mmm(mmm3, "Patched __getattribute__ MMM")
    
    # Restore original to avoid affecting other tests
    TensorVariable.__getattribute__ = original_getattribute
    print("Restored original __getattribute__")
    
except Exception as e:
    print(f"✗ __getattribute__ patch test failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Make sure we restore even if there's an error
    try:
        TensorVariable.__getattribute__ = original_getattribute
        print("Restored original __getattribute__ after error")
    except:
        pass

print("\nAll tests complete")
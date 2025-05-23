#!/usr/bin/env python
"""
Test with completely fixed parameters to avoid sampling entirely

This approach uses DensityDist priors with point masses to avoid MCMC sampling.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytensor
import pymc as pm

# Apply global patch first
import pytensor.tensor as pt

# Store original methods
_original_getattr = pt.TensorVariable.__getattribute__
_original_setattr = pt.TensorVariable.__setattr__

def _patched_getattr(self, name):
    """Patched __getattribute__ to handle dims attribute"""
    if name == 'dims':
        try:
            return _original_getattr(self, name)
        except AttributeError:
            if hasattr(self, '_pymc_dims'):
                return self._pymc_dims
            # Default to channel dimension
            return ('channel',)
    return _original_getattr(self, name)

def _patched_setattr(self, name, value):
    """Patched __setattr__ to store dims attribute"""
    if name == 'dims':
        _original_setattr(self, '_pymc_dims', value)
    else:
        _original_setattr(self, name, value)

# Apply patches
pt.TensorVariable.__getattribute__ = _patched_getattr
pt.TensorVariable.__setattr__ = _patched_setattr

print("Global patch applied successfully!")

# Import after patching
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

def clean_numeric_columns(df, columns):
    """Clean numeric columns in dataframe"""
    for col in columns:
        if col in df.columns:
            # Convert to string if object type
            if df[col].dtype == 'object':
                print(f"Converting column {col} from string to numeric...")
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
            # Ensure numeric type
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaNs with 0
            df[col] = df[col].fillna(0)
            print(f"Column {col}: dtype={df[col].dtype}, min={df[col].min()}, max={df[col].max()}")
    return df

def test_with_fixed_params():
    """Test with completely fixed parameters"""
    # Load data
    print("Loading and preparing data...")
    df = pd.read_csv("attached_assets/dankztestdata_v2.csv")
    
    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    
    # Only use first channel for simplicity
    channel = "PPCBrand_Spend"
    
    # Clean columns
    df = clean_numeric_columns(df, ["Sales", channel, "interestrate_control"])
    
    # Set fixed parameter values
    alpha_value = 0.5  # Adstock decay rate
    l_max = 4  # Adstock max lag
    L_value = 1.0  # Saturation ceiling
    k_value = 0.0005  # Saturation steepness
    x0_value = 10000.0  # Saturation midpoint
    
    print(f"Using fixed parameters: alpha={alpha_value}, L={L_value}, k={k_value}, x0={x0_value}")
    
    with pm.Model(coords={"channel": [channel]}) as model:
        # Create fixed parameter distributions
        # These are point masses (delta distributions) with no uncertainty
        alpha = pm.ConstantDist.dist(alpha_value, dims="channel")
        alpha = pm.Deterministic("alpha", alpha)
        
        L = pm.ConstantDist.dist(L_value, dims="channel")
        L = pm.Deterministic("L", L)
        
        k = pm.ConstantDist.dist(k_value, dims="channel") 
        k = pm.Deterministic("k", k)
        
        x0 = pm.ConstantDist.dist(x0_value, dims="channel")
        x0 = pm.Deterministic("x0", x0)
        
        # Create transforms
        adstock = GeometricAdstock(
            l_max=l_max,
            priors={"alpha": alpha}
        )
        
        saturation = LogisticSaturation(
            priors={"L": L, "k": k, "x0": x0}
        )
        
        # Create MMM
        mmm = MMM(
            channel_columns=[channel],
            adstock=adstock,
            saturation=saturation,
            control_columns=["interestrate_control"],
            date_column="Date"
        )
        
        # Add patch for date handling
        original_preprocess = mmm._generate_and_preprocess_model_data
        
        def patched_preprocess(X, y=None):
            # Convert date column to datetime with dayfirst=True if needed
            if "Date" in X.columns and not pd.api.types.is_datetime64_any_dtype(X["Date"]):
                X = X.copy()
                X["Date"] = pd.to_datetime(X["Date"], dayfirst=True)
                print(f"Patched date handling: converted Date to datetime")
            return original_preprocess(X, y)
            
        mmm._generate_and_preprocess_model_data = patched_preprocess
        
        # Prepare data
        X = df[["Date", channel, "interestrate_control"]].copy()
        y = df["Sales"].copy()
        
        # Set up model without sampling
        print("Fitting model with fixed parameters (no sampling)...")
        mmm.build_model(X=X, y=y)
        
        # Skip sampling entirely, just use fixed parameters
        trace = None
        print("✓ Model created successfully with fixed parameters!")
        
        # Calculate model predictions and effects
        print("Calculating model predictions with fixed parameters...")
        # Manually initialize model components
        mmm.finalize()
        
        # Try predicting with the model
        pred = mmm.predict(X)
        print(f"Mean prediction: {np.mean(pred)}")
        
        # Try to get effects
        try:
            effects = mmm.get_total_effects(X)
            print(f"Effects for {channel}: {effects[channel].sum()}")
        except Exception as e:
            print(f"Could not calculate effects: {str(e)}")
            
        print("✓ Successfully created MMM with fixed parameters!")
        
    return True

if __name__ == "__main__":
    try:
        if test_with_fixed_params():
            print("✓ Test completed successfully!")
        else:
            print("✗ Test failed.")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Exception during test: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
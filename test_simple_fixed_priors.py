#!/usr/bin/env python
"""
Test with simplified fixed prior values to avoid sampling issues

This script skips sampling entirely and uses fixed values for the parameters.
"""

import os
import sys
import numpy as np
import pandas as pd
import pymc as pm

# Apply global patch first - add dims attribute to TensorVariable
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
            return None
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

# Import PyMC Marketing after patching
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

def clean_data(df, channels, target_column, control_columns):
    """Clean numeric columns in dataframe"""
    # Clean channels
    for ch in channels:
        if ch in df.columns:
            if df[ch].dtype == 'object':
                print(f"Converting {ch} from string to float")
                df[ch] = df[ch].str.replace(',', '').astype(float)
            else:
                df[ch] = df[ch].astype(float)
    
    # Clean target column
    if df[target_column].dtype == 'object':
        print(f"Converting {target_column} from string to float")
        df[target_column] = df[target_column].str.replace(',', '').astype(float)
    
    # Clean control columns
    for col in control_columns:
        if col in df.columns and df[col].dtype == 'object':
            print(f"Converting {col} from string to float")
            df[col] = df[col].str.replace(',', '').astype(float)
    
    print(f"Data types after cleaning: {df[channels + [target_column]].dtypes}")
    return df

def test_with_fixed_priors():
    """Test with fixed priors to avoid sampling"""
    # Select just one channel for simplicity
    channels = ["PPCBrand_Spend"]
    target_column = "Sales"
    control_columns = ["interestrate_control"]
    date_column = "Date"
    
    # Load and clean data
    print("Loading data...")
    df = pd.read_csv("attached_assets/dankztestdata_v2.csv")
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
    print(f"Date range: {df[date_column].min()} to {df[date_column].max()}")
    
    # Clean data
    df = clean_data(df, channels, target_column, control_columns)
    
    print("Creating model with fixed priors...")
    
    # Set fixed values for the hyperparameters
    alpha_value = 0.7  # Adstock decay parameter
    L_value = 1.0      # Saturation ceiling
    k_value = 0.0005   # Saturation steepness
    x0_value = 1000.0  # Saturation midpoint
    
    with pm.Model(coords={"channel": channels}) as model:
        # Create deterministic variables with fixed values
        alpha = pm.Deterministic("alpha", pt.ones((len(channels),)) * alpha_value)
        alpha.dims = "channel"  # Set dims directly using our patch
        
        L = pm.Deterministic("L", pt.ones((len(channels),)) * L_value)
        L.dims = "channel"
        
        k = pm.Deterministic("k", pt.ones((len(channels),)) * k_value)
        k.dims = "channel"
        
        x0 = pm.Deterministic("x0", pt.ones((len(channels),)) * x0_value)
        x0.dims = "channel"
        
        # Create transforms with fixed parameters
        adstock = GeometricAdstock(
            l_max=4,
            priors={"alpha": alpha}
        )
        
        saturation = LogisticSaturation(
            priors={"L": L, "k": k, "x0": x0}
        )
        
        print("Setting up MMM...")
        # Create MMM
        mmm = MMM(
            channel_columns=channels,
            adstock=adstock,
            saturation=saturation,
            control_columns=control_columns,
            date_column=date_column
        )
        
        # Prepare data
        X = df[[date_column] + channels + control_columns].copy()
        y = df[target_column].copy()
        
        # Build model without sampling
        print("Building model (no sampling)...")
        try:
            mmm.build_model(X=X, y=y)
            print("✓ Model built successfully!")
            return mmm, model, X
        except Exception as e:
            print(f"✗ Error building model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None

if __name__ == "__main__":
    try:
        mmm, model, X = test_with_fixed_priors()
        if mmm is not None:
            print("Attempting to predict with model...")
            try:
                # Try to initialize without sampling
                pred = mmm.predict(X)
                print(f"✓ Model prediction successful! Mean prediction: {np.mean(pred)}")
            except Exception as e:
                print(f"✗ Error predicting: {str(e)}")
                import traceback
                traceback.print_exc()
        print("✓ Test completed!")
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
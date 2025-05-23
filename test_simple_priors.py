#!/usr/bin/env python
"""Test with simplified priors to isolate constraint issues"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply global patch first
from python_scripts.fit_mmm_with_global_patch import *

# Override the create_and_fit_mmm_model function with simpler priors
def test_simple_priors():
    config = load_json_config("test_config_quick.json")
    df = load_data("attached_assets/dankztestdata_v2.csv", date_column="Date")
    
    channels = list(config["channels"].keys())
    date_column = "Date"
    target_column = "Sales"
    control_columns = ["interestrate_control"]
    
    # Clean numeric columns - remove commas and convert to float
    print("Cleaning numeric columns...")
    for ch in channels:
        if ch in df.columns:
            if df[ch].dtype == 'object':
                df[ch] = df[ch].str.replace(',', '').astype(float)
            else:
                df[ch] = df[ch].astype(float)

    # Also clean the target column
    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].str.replace(',', '').astype(float)

    # Clean control columns
    for col in control_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)

    print(f"Data types after cleaning: {df[channels + [target_column]].dtypes}")
    
    print("Testing with simplified priors...")
    
    with pm.Model(coords={"channel": channels}) as model:
        # Use simple Uniform priors for alpha (avoiding Beta constraints)
        alpha_rv = pm.Uniform("alpha", lower=0.4, upper=0.9, dims="channel")
        L_rv = pm.HalfNormal("L", sigma=2.0, dims="channel")
        k_rv = pm.HalfNormal("k", sigma=0.01, dims="channel")
        x0_rv = pm.HalfNormal("x0", sigma=50000, dims="channel")
        
        # Create transforms
        adstock = GeometricAdstock(l_max=7, priors={"alpha": alpha_rv})
        saturation = LogisticSaturation(priors={"L": L_rv, "k": k_rv, "x0": x0_rv})
        
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
        
        # Use NormalWalker sampler which is often more tolerant of constraint issues
        import pytensor
        
        # Additional debug info
        print("Current PyTensor config:")
        print(f"- Mode: {pytensor.config.mode}")
        print(f"- Computing backend: {pytensor.config.device}")
        
        # Set more lenient parameters
        pytensor.config.compute_test_value = 'ignore'
        
        # Try with specialized sampler
        trace = mmm.fit(
            X=X, 
            y=y, 
            draws=10,  # Very minimal sampling just to test
            tune=10,   
            chains=1,
            step=pm.NUTS(target_accept=0.95),  # Higher target acceptance
            compute_convergence_checks=False,
            return_inferencedata=True,
            progressbar=True,
            discard_tuned_samples=True
        )
        print("✓ Success with simplified priors!")
        return True

if __name__ == "__main__":
    try:
        test_simple_priors()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
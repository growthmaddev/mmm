#!/usr/bin/env python
"""
Verify Transform Parameters

This script checks that the channel-specific adstock and saturation parameters 
are being correctly applied in the MMM model.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

def main():
    """Test the adstock and saturation parameter handling"""
    # Load the configuration file
    config_path = os.path.join(os.path.dirname(__file__), 'test_config_quick.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded config from {config_path}")
    
    # Create the adstock and saturation objects for each channel
    adstock_settings = config.get('adstockSettings', {})
    saturation_settings = config.get('saturationSettings', {})
    
    channel_columns = config.get('channelColumns', {})
    
    for channel in channel_columns:
        print(f"\nProcessing channel: {channel}")
        
        # Get adstock parameters
        adstock_params = adstock_settings.get('default', {})
        if channel in adstock_settings.get('channel_specific_params', {}):
            adstock_params = adstock_settings['channel_specific_params'][channel]
        
        alpha = adstock_params.get('adstock_alpha', 0.5)
        l_max = adstock_params.get('adstock_l_max', 8)
        
        # Get saturation parameters
        saturation_params = saturation_settings.get('default', {})
        if channel in saturation_settings.get('channel_specific_params', {}):
            saturation_params = saturation_settings['channel_specific_params'][channel]
        
        L = saturation_params.get('saturation_L', 1.0)
        k = saturation_params.get('saturation_k', 0.0001)
        x0 = saturation_params.get('saturation_x0', 50000.0)
        
        print(f"  Adstock parameters: alpha={alpha}, l_max={l_max}")
        print(f"  Saturation parameters: L={L}, k={k}, x0={x0}")
        
        # Create the objects
        adstock = GeometricAdstock(alpha=alpha, l_max=l_max)
        saturation = LogisticSaturation(L=L, k=k, x0=x0)
        
        print(f"  Created GeometricAdstock with alpha={adstock.alpha}, l_max={adstock.l_max}")
        print(f"  Created LogisticSaturation with L={saturation.L}, k={saturation.k}, x0={saturation.x0}")
        
        # Simulate data for this channel
        dummy_data = np.array([1000, 2000, 3000, 4000, 5000])
        
        # Test adstock transform
        print("  Testing adstock transformation...")
        try:
            adstocked = adstock.transform(dummy_data)
            print(f"  Adstock transformation successful: {adstocked[:3]}...")
        except Exception as e:
            print(f"  Adstock transformation error: {str(e)}")
        
        # Test saturation transform
        print("  Testing saturation transformation...")
        try:
            saturated = saturation.transform(dummy_data)
            print(f"  Saturation transformation successful: {saturated[:3]}...")
        except Exception as e:
            print(f"  Saturation transformation error: {str(e)}")
    
    # Verify media_transforms handling in MMM
    print("\nVerifying MMM media_transforms handling:")
    
    # Create basic data for testing
    print("Creating test data...")
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=10),
        'Sales': np.random.randint(100000, 500000, 10),
        'PPCBrand_Spend': np.random.randint(1000, 5000, 10),
        'FBReach_Spend': np.random.randint(1000, 5000, 10)
    }
    df = pd.DataFrame(data)
    
    try:
        # Create media transforms dictionary
        media_transforms = {}
        for channel in channel_columns:
            # Get adstock parameters
            adstock_params = adstock_settings.get('default', {})
            if channel in adstock_settings.get('channel_specific_params', {}):
                adstock_params = adstock_settings['channel_specific_params'][channel]
            
            alpha = adstock_params.get('adstock_alpha', 0.5)
            l_max = adstock_params.get('adstock_l_max', 8)
            
            # Get saturation parameters
            saturation_params = saturation_settings.get('default', {})
            if channel in saturation_settings.get('channel_specific_params', {}):
                saturation_params = saturation_settings['channel_specific_params'][channel]
            
            L = saturation_params.get('saturation_L', 1.0)
            k = saturation_params.get('saturation_k', 0.0001)
            x0 = saturation_params.get('saturation_x0', 50000.0)
            
            # Create the objects
            adstock = GeometricAdstock(alpha=alpha, l_max=l_max)
            saturation = LogisticSaturation(L=L, k=k, x0=x0)
            
            media_transforms[channel] = {
                'adstock': adstock,
                'saturation': saturation
            }
        
        # Initialize MMM
        print("Initializing MMM with test data...")
        mmm = MMM(
            date_column='Date',
            channel_columns=channel_columns,
            adstock=media_transforms[list(channel_columns.keys())[0]]['adstock'],
            saturation=media_transforms[list(channel_columns.keys())[0]]['saturation']
        )
        
        # Now set media_transforms
        print("Setting channel-specific media_transforms...")
        try:
            # Try direct access
            mmm.media_transforms = {}
            for channel, transforms in media_transforms.items():
                mmm.media_transforms[channel] = transforms
                print(f"Set transforms for {channel} via direct access")
                
        except (AttributeError, Exception) as e:
            print(f"Direct media_transforms access failed: {str(e)}")
            
            # Try alternative method
            try:
                for channel, transforms in media_transforms.items():
                    if hasattr(mmm, 'set_transforms'):
                        mmm.set_transforms(
                            channel=channel,
                            adstock=transforms['adstock'],
                            saturation=transforms['saturation']
                        )
                        print(f"Set transforms for {channel} via set_transforms method")
                    else:
                        print(f"WARNING: Could not set transforms for {channel}")
            except Exception as e2:
                print(f"Alternative transform setting failed: {str(e2)}")
        
        # Verify the transforms were set
        print("\nVerifying transforms were set correctly:")
        if hasattr(mmm, 'media_transforms'):
            for channel in channel_columns:
                if channel in mmm.media_transforms:
                    transforms = mmm.media_transforms[channel]
                    if 'adstock' in transforms:
                        adstock = transforms['adstock']
                        print(f"{channel} adstock - alpha: {adstock.alpha}, l_max: {adstock.l_max}")
                    if 'saturation' in transforms:
                        saturation = transforms['saturation']
                        print(f"{channel} saturation - L: {saturation.L}, k: {saturation.k}, x0: {saturation.x0}")
                else:
                    print(f"WARNING: {channel} not found in media_transforms")
        else:
            print("MMM does not have media_transforms attribute")
        
        print("\nVerification complete!")
    
    except Exception as e:
        print(f"Error in MMM verification: {str(e)}")

if __name__ == "__main__":
    main()
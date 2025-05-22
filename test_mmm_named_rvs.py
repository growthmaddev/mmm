#!/usr/bin/env python
"""
Test script for the new named RVs approach for MMM
"""

import os
import sys
import json
import pandas as pd
from python_scripts.mmm_with_named_rvs import create_mmm_with_named_rvs

def main():
    """Test the create_mmm_with_named_rvs function"""
    # Load data and config
    data_path = "attached_assets/dankztestdata_v2.csv"
    config_path = "test_config_quick.json"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    channel_list = list(config['channelColumns'].keys())
    date_column = config.get('dateColumn', 'Date')
    target_column = config.get('targetColumn', 'Sales')
    
    # Prepare parameter dictionaries
    adstock_params = {}
    saturation_params = {}
    
    # Extract parameters from config
    if 'adstockSettings' in config and 'default' in config['adstockSettings']:
        default_adstock = config['adstockSettings']['default']
        for channel in channel_list:
            adstock_params[channel] = default_adstock
    
    if 'saturationSettings' in config and 'default' in config['saturationSettings']:
        default_saturation = config['saturationSettings']['default']
        for channel in channel_list:
            saturation_params[channel] = default_saturation
    
    print(f"Testing create_mmm_with_named_rvs with:")
    print(f"  Channels: {channel_list}")
    print(f"  Date column: {date_column}")
    print(f"  Target column: {target_column}")
    
    # Create MMM with named RVs
    try:
        mmm = create_mmm_with_named_rvs(
            df=df,
            channel_list=channel_list,
            date_column=date_column,
            target_column=target_column,
            adstock_params=adstock_params,
            saturation_params=saturation_params
        )
        
        print(f"Successfully created MMM with named RVs")
        print(f"MMM object type: {type(mmm)}")
        
        # Verify the transforms
        if hasattr(mmm, 'media_transforms'):
            media_keys = list(mmm.media_transforms.keys())
            print(f"Media transforms keys: {media_keys}")
            print("Success! Named RVs approach works.")
            return True
        else:
            print("MMM created but media_transforms not accessible")
            return False
    
    except Exception as e:
        print(f"Error creating MMM with named RVs: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
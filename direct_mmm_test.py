#!/usr/bin/env python
"""
Direct test of the Ridge regression MMM implementation
"""

import subprocess
import json
import os
import sys

def test_ridge_mmm():
    """Run a direct test of the Ridge regression MMM"""
    
    print("🔍 Testing Ridge Regression MMM Implementation")
    print("------------------------------------------------")
    
    # Test with the sample data
    data_file = "uploads/dankztestdata_v2.csv"
    config_file = "test_config_quick.json"
    
    # Make sure files exist
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file {data_file} not found")
        return False
        
    if not os.path.exists(config_file):
        print(f"❌ Error: Config file {config_file} not found")
        return False
    
    print(f"✅ Using data file: {data_file}")
    print(f"✅ Using config file: {config_file}")
    
    # Run the ridge regression
    print("\n🔬 Running Ridge Regression Model")
    print("------------------------------------------------")
    
    script_path = "MarketMixMaster/python_scripts/fit_mmm_ridge.py"
    
    try:
        result = subprocess.run(
            [sys.executable, script_path, data_file, config_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        mmm_results = json.loads(result.stdout)
        
        # Display key results
        print("\n📊 Ridge Regression Results")
        print("------------------------------------------------")
        print(f"✅ Model R-squared: {mmm_results['model_quality']['r_squared']:.3f}")
        
        # Show channel ROIs
        print("\n📈 Channel ROIs")
        print("------------------------------------------------")
        for channel, roi in mmm_results['channel_analysis']['roi'].items():
            print(f"  {channel}: {roi:.2f}x")
        
        # Show sales decomposition
        print("\n🔄 Sales Decomposition")
        print("------------------------------------------------")
        base_percent = mmm_results['analytics']['sales_decomposition']['percent_decomposition']['base']
        channels_percent = sum(mmm_results['channel_analysis']['contribution_percentage'].values())
        print(f"  Base Sales: {base_percent:.1f}%")
        print(f"  Marketing Contribution: {channels_percent:.1f}%")
        
        # Verify config with saturation parameters
        print("\n⚙️ Configuration Parameters")
        print("------------------------------------------------")
        if 'config' in mmm_results and 'channels' in mmm_results['config']:
            first_channel = list(mmm_results['config']['channels'].keys())[0]
            if all(k in mmm_results['config']['channels'][first_channel] for k in ['L', 'k', 'x0']):
                print(f"✅ Saturation parameters present for channel: {first_channel}")
                print(f"  L: {mmm_results['config']['channels'][first_channel]['L']}")
                print(f"  k: {mmm_results['config']['channels'][first_channel]['k']}")
                print(f"  x0: {mmm_results['config']['channels'][first_channel]['x0']}")
            else:
                print("❌ Missing saturation parameters in config")
        else:
            print("❌ Config structure missing")
            
        # Verify channel characteristics for budget optimization
        print("\n🏷️ Channel Characteristics")
        print("------------------------------------------------")
        if 'channel_characteristics' in mmm_results:
            print("✅ Channel type metadata present:")
            for channel, metadata in mmm_results['channel_characteristics'].items():
                ch_type = metadata.get('type', 'unknown')
                high_roi = metadata.get('typically_high_roi', False)
                print(f"  {channel}: {ch_type} (High ROI: {'Yes' if high_roi else 'No'})")
        else:
            print("❌ Channel characteristics not found (for budget optimization)")
            
        return mmm_results
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Ridge regression script: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except json.JSONDecodeError:
        print(f"❌ Error parsing output from Ridge regression script")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    results = test_ridge_mmm()
    
    if results:
        print("\n✅ Ridge Regression MMM Test Successful")
    else:
        print("\n❌ Ridge Regression MMM Test Failed")
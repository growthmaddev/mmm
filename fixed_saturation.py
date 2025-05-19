#!/usr/bin/env python3
"""
Fixed Saturation Function utility for MMM models
"""

import numpy as np
import sys
import math
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

def logistic_saturation(x: float, L: float = 1.0, k: float = 0.0005, x0: float = 50000.0) -> float:
    """
    Logistic saturation function with better numerical stability.
    
    Args:
        x: Input value (typically spend amount)
        L: Maximum value (saturation ceiling)
        k: Steepness parameter (growth rate)
        x0: Midpoint parameter (inflection point)
    
    Returns:
        Saturated value between 0 and L
    """
    # Avoid overflow in exp
    exponent = k * (x - x0)
    if exponent > 100:
        return L
    elif exponent < -100:
        return 0
    
    return L / (1 + np.exp(-exponent))

# Test different scenarios
if __name__ == "__main__":
    # Test different spend levels with realistic parameters
    spend_levels = [1000, 5000, 10000, 50000, 100000, 500000]
    
    # Test with default parameters: L=1.0, k=0.0005, x0=50000.0
    print("===== Testing with default parameters =====")
    for spend in spend_levels:
        result = logistic_saturation(spend)
        print(f"Spend: ${spend:,}, Saturation: {result:.6f}")
    
    # Test with different L values (ceiling)
    print("\n===== Testing with different L values =====")
    for L in [0.5, 1.0, 2.0]:
        print(f"\nL = {L}")
        for spend in spend_levels:
            result = logistic_saturation(spend, L=L)
            print(f"Spend: ${spend:,}, Saturation: {result:.6f}")
    
    # Test with different k values (steepness)
    print("\n===== Testing with different k values =====")
    for k in [0.0001, 0.0005, 0.001]:
        print(f"\nk = {k}")
        for spend in spend_levels:
            result = logistic_saturation(spend, k=k)
            print(f"Spend: ${spend:,}, Saturation: {result:.6f}")
    
    # Test with different x0 values (midpoint)
    print("\n===== Testing with different x0 values =====")
    for x0 in [10000, 50000, 100000]:
        print(f"\nx0 = {x0}")
        for spend in spend_levels:
            result = logistic_saturation(spend, x0=x0)
            print(f"Spend: ${spend:,}, Saturation: {result:.6f}")
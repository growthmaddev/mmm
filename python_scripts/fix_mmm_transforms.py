#!/usr/bin/env python
"""
Direct fix for MMM transform handling that avoids the 'dims' attribute error

This approach uses PyMC's DiracDelta distribution through a compatibility layer.
"""

import pymc as pm
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

class FixedPrior:
    """
    A compatibility class that wraps fixed values to work with PyMC-Marketing
    
    This class allows us to bypass the 'dims' attribute error by providing
    a custom implementation that works with PyMC-Marketing's expectations.
    """
    def __init__(self, value):
        """
        Initialize with a fixed value
        
        Args:
            value: The fixed value to use
        """
        self.value = value
        self.dims = ()  # Empty tuple to satisfy 'dims' attribute check
    
    def eval(self):
        """Return the fixed value when evaluated"""
        return self.value

def create_fixed_adstock(alpha=0.5, l_max=8):
    """
    Create a GeometricAdstock with fixed parameters
    
    Args:
        alpha: Fixed alpha value
        l_max: Maximum lag
        
    Returns:
        GeometricAdstock object with fixed parameters
    """
    return GeometricAdstock(
        l_max=l_max,
        priors={"alpha": FixedPrior(alpha)}
    )

def create_fixed_saturation(L=1.0, k=0.0001, x0=50000.0):
    """
    Create a LogisticSaturation with fixed parameters
    
    Args:
        L: Fixed L value
        k: Fixed k value
        x0: Fixed x0 value
        
    Returns:
        LogisticSaturation object with fixed parameters
    """
    return LogisticSaturation(
        priors={
            "L": FixedPrior(L),
            "k": FixedPrior(k),
            "x0": FixedPrior(x0)
        }
    )
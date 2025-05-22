#!/usr/bin/env python
"""
Monkey-patch for TensorVariable to add the missing 'dims' attribute

This module provides a monkey-patch that adds a 'dims' attribute to PyTensor TensorVariable 
objects, ensuring they can be used with PyMC-Marketing 0.13.1 which expects this attribute.
"""

import pytensor
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable

# Store original __getattribute__ method
original_getattribute = TensorVariable.__getattribute__

def patched_getattribute(self, name):
    """
    Patched __getattribute__ that adds a 'dims' attribute to TensorVariable
    
    If 'dims' is requested and does not exist, it returns ("channel",) which is
    what PyMC-Marketing 0.13.1 expects in its default_model_config method.
    """
    # If 'dims' is requested
    if name == 'dims':
        try:
            # First try to get it normally
            return original_getattribute(self, name)
        except AttributeError:
            # If it doesn't exist, return what PyMC-Marketing expects
            return ("channel",)
    
    # For all other attributes, use the original method
    return original_getattribute(self, name)

# Apply the monkey-patch
TensorVariable.__getattribute__ = patched_getattribute

print("Monkey-patched TensorVariable.__getattribute__ to provide 'dims' attribute")
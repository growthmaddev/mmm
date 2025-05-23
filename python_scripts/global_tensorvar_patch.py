#!/usr/bin/env python
"""
Global patch for TensorVariable to add missing dims attribute

This module provides a function to apply a global patch to PyTensor's TensorVariable class
to add the missing 'dims' attribute required by PyMC-Marketing. This should be imported
and called before any PyMC or PyMC-Marketing operations.
"""

import pytensor.tensor as pt
import sys

_ORIGINAL_GETATTR = None
_ORIGINAL_SETATTR = None
_PATCH_APPLIED = False

def apply_global_tensorvar_patch(verbose=True):
    """
    Apply global patch to TensorVariable class to fix dims attribute issues
    
    This function adds proper handling of the 'dims' attribute to TensorVariable instances
    by patching __getattribute__ and __setattr__ methods. This addresses compatibility issues
    between PyMC and PyMC-Marketing.
    
    Args:
        verbose: Whether to print status messages
        
    Returns:
        True if patch was applied, False if already applied
    """
    global _ORIGINAL_GETATTR, _ORIGINAL_SETATTR, _PATCH_APPLIED
    
    if _PATCH_APPLIED:
        if verbose:
            print("TensorVariable patch already applied", file=sys.stderr)
        return False
    
    if verbose:
        print("Applying global TensorVariable patch...", file=sys.stderr)
    
    # Store original methods
    _ORIGINAL_GETATTR = pt.TensorVariable.__getattribute__
    _ORIGINAL_SETATTR = pt.TensorVariable.__setattr__
    
    # Define patched methods
    def _patched_getattr(self, name):
        """Patched __getattribute__ to handle dims attribute"""
        if name == 'dims':
            # Return a default dims value if not present
            try:
                return _ORIGINAL_GETATTR(self, name)
            except AttributeError:
                # Check if we stored dims elsewhere
                if hasattr(self, '_pymc_dims'):
                    return self._pymc_dims
                # Default to channel dimension
                return ('channel',)  # This is the dimension expected by PyMC-Marketing
        return _ORIGINAL_GETATTR(self, name)
    
    def _patched_setattr(self, name, value):
        """Patched __setattr__ to store dims attribute"""
        if name == 'dims':
            # Store dims in alternate attribute since TensorVariable normally doesn't have this
            _ORIGINAL_SETATTR(self, '_pymc_dims', value)
        else:
            _ORIGINAL_SETATTR(self, name, value)
    
    # Apply the patches
    pt.TensorVariable.__getattribute__ = _patched_getattr
    pt.TensorVariable.__setattr__ = _patched_setattr
    
    _PATCH_APPLIED = True
    
    if verbose:
        print("TensorVariable patch successfully applied", file=sys.stderr)
    
    return True

def remove_global_tensorvar_patch(verbose=True):
    """
    Remove the global TensorVariable patch
    
    This function restores the original methods if the patch was previously applied.
    
    Args:
        verbose: Whether to print status messages
        
    Returns:
        True if patch was removed, False if not currently applied
    """
    global _ORIGINAL_GETATTR, _ORIGINAL_SETATTR, _PATCH_APPLIED
    
    if not _PATCH_APPLIED:
        if verbose:
            print("TensorVariable patch not currently applied", file=sys.stderr)
        return False
    
    if verbose:
        print("Removing TensorVariable patch...", file=sys.stderr)
    
    # Restore original methods
    pt.TensorVariable.__getattribute__ = _ORIGINAL_GETATTR
    pt.TensorVariable.__setattr__ = _ORIGINAL_SETATTR
    
    _PATCH_APPLIED = False
    
    if verbose:
        print("Original TensorVariable methods restored", file=sys.stderr)
    
    return True

# For use as context manager
class TensorVarPatchContext:
    """
    Context manager for temporarily applying the TensorVariable patch
    
    Example:
        with TensorVarPatchContext():
            # Do PyMC-Marketing operations
            mmm = MMM(...)
            mmm.fit(...)
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def __enter__(self):
        apply_global_tensorvar_patch(verbose=self.verbose)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        remove_global_tensorvar_patch(verbose=self.verbose)
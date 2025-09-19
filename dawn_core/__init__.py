"""
DAWN Core Module
================

Legacy compatibility module for dawn_core path structure.
This module provides backward compatibility for systems that expect
the old dawn_core module structure.

This is a compatibility layer - the main DAWN system uses the new
dawn.core.foundation structure, but this module ensures that legacy
references continue to work.
"""

# Version info
__version__ = "1.0.0"
__author__ = "DAWN Consciousness System"
__description__ = "Legacy compatibility layer for DAWN core functionality"

# Import key components for easy access
from . import state

# Expose commonly used functions at module level
from .state import (
    get_state, 
    set_state, 
    reset_state,
    is_transcendent,
    is_meta_aware, 
    is_coherent,
    get_state_summary,
    update_unity_delta,
    update_awareness_delta,
    evolve_consciousness,
    is_ready_for_level,
    set_session_info,
    label_from_metrics,
    clamp,
    label_for
)

__all__ = [
    'state',
    'get_state',
    'set_state', 
    'reset_state',
    'is_transcendent',
    'is_meta_aware',
    'is_coherent', 
    'get_state_summary',
    'update_unity_delta',
    'update_awareness_delta',
    'evolve_consciousness',
    'is_ready_for_level',
    'set_session_info',
    'label_from_metrics',
    'clamp',
    'label_for'
]

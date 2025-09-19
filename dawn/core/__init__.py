"""
DAWN Core Foundation
===================

Core framework components providing foundational systems for:
- Module lifecycle management
- Communication protocols
- Configuration management
- Event systems
- Registry services

This package provides the stable foundation that all other DAWN modules build upon.
"""

from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

# Lazy imports for foundation components
def __getattr__(name: str):
    """Lazy import system for core modules"""
    if name == 'foundation':
        from . import foundation
        return foundation
    elif name == 'communication':
        from . import communication
        return communication
    elif name == 'configuration':
        from . import configuration
        return configuration
    else:
        raise AttributeError(f"module 'dawn.core' has no attribute '{name}'")

__all__ = ['foundation', 'communication', 'configuration']

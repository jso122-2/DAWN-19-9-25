"""
Module Registry
==============

Registry for managing DAWN modules and capabilities.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ModuleRegistry:
    """Registry for module management"""
    
    def __init__(self):
        self.modules: Dict[str, Any] = {}
    
    def register(self, module_name: str, module: Any) -> bool:
        """Register a module"""
        self.modules[module_name] = module
        logger.info(f"Registered module: {module_name}")
        return True
    
    def get(self, module_name: str) -> Any:
        """Get a registered module"""
        return self.modules.get(module_name)

class CapabilityRegistry:
    """Registry for capability management"""
    
    def __init__(self):
        self.capabilities: Dict[str, Any] = {}
    
    def register_capability(self, name: str, capability: Any) -> bool:
        """Register a capability"""
        self.capabilities[name] = capability
        return True

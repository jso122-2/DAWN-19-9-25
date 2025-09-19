"""
DAWN - Deep AI With Neural architectures
========================================

Expansive Modular Consciousness Architecture

This package provides a comprehensive framework for artificial consciousness,
featuring dynamic module discovery, plugin architecture, and unlimited 
expandability across multiple consciousness domains.

Modules:
    core: Foundation systems and communication
    consciousness: Consciousness engines and models
    processing: Processing engines and pipelines
    memory: Memory systems and storage
    subsystems: Specialized domain systems
    interfaces: User interaction systems
    capabilities: Dynamic capability system
    extensions: Plugin and extension system
    tools: Development and analysis tools
    research: Research and experimentation

Usage:
    >>> import dawn
    >>> # Auto-discover available capabilities
    >>> capabilities = dawn.discover_capabilities()
    >>> # Load specific consciousness engine
    >>> engine = dawn.consciousness.engines.core.primary_engine()
    >>> # Initialize processing pipeline
    >>> processor = dawn.processing.engines.tick.synchronous.engine()
"""

import sys
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Version information
__version__ = "2.0.0"
__author__ = "DAWN Research Team"
__description__ = "Expansive Modular Consciousness Architecture"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DAWNModuleDiscovery:
    """Dynamic module discovery and loading system"""
    
    def __init__(self):
        self.registered_modules = {}
        self.capabilities = {}
        self.loaded_engines = {}
        
    def discover_capabilities(self, namespace: Optional[str] = None) -> Dict[str, List[str]]:
        """Discover all available capabilities in the DAWN system"""
        capabilities = {}
        
        # Define namespaces to scan
        namespaces = [namespace] if namespace else [
            'consciousness', 'processing', 'memory', 'subsystems',
            'interfaces', 'capabilities', 'extensions', 'tools', 'research'
        ]
        
        for ns in namespaces:
            try:
                namespace_path = Path(__file__).parent / ns
                if namespace_path.exists():
                    capabilities[ns] = self._scan_namespace(namespace_path)
            except Exception as e:
                logger.warning(f"Failed to scan namespace {ns}: {e}")
                
        return capabilities
    
    def _scan_namespace(self, path: Path) -> List[str]:
        """Scan a namespace directory for available modules"""
        modules = []
        for item in path.rglob("*.py"):
            if item.name != "__init__.py":
                # Convert path to module name
                relative_path = item.relative_to(Path(__file__).parent)
                module_name = str(relative_path.with_suffix("")).replace("/", ".")
                modules.append(module_name)
        return modules
    
    def load_module(self, module_path: str, **kwargs):
        """Dynamically load a module"""
        if module_path in self.loaded_engines:
            return self.loaded_engines[module_path]
            
        try:
            module = importlib.import_module(f"dawn.{module_path}")
            if hasattr(module, "create_engine"):
                engine = module.create_engine(**kwargs)
                self.loaded_engines[module_path] = engine
                return engine
            return module
        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {e}")
            return None
    
    def register_plugin(self, plugin_name: str, plugin_class, namespace: str = "custom"):
        """Register a plugin for dynamic loading"""
        if namespace not in self.registered_modules:
            self.registered_modules[namespace] = {}
        self.registered_modules[namespace][plugin_name] = plugin_class
        logger.info(f"Registered plugin {plugin_name} in namespace {namespace}")

# Global discovery instance
_discovery = DAWNModuleDiscovery()

# Export key functions at package level
discover_capabilities = _discovery.discover_capabilities
load_module = _discovery.load_module
register_plugin = _discovery.register_plugin

# Auto-discovery on import
try:
    logger.info("DAWN: Initializing modular consciousness architecture...")
    available_capabilities = discover_capabilities()
    logger.info(f"DAWN: Discovered {sum(len(modules) for modules in available_capabilities.values())} modules across {len(available_capabilities)} namespaces")
except Exception as e:
    logger.warning(f"DAWN: Failed to auto-discover capabilities: {e}")

# Lazy imports for main namespaces
def __getattr__(name: str):
    """Lazy import system for namespaces"""
    if name in ['consciousness', 'processing', 'memory', 'subsystems', 
                'interfaces', 'capabilities', 'extensions', 'tools', 'research']:
        try:
            return importlib.import_module(f"dawn.{name}")
        except ImportError as e:
            logger.warning(f"Failed to import namespace {name}: {e}")
            return None
    raise AttributeError(f"module 'dawn' has no attribute '{name}'")

# Export all namespaces
__all__ = [
    # Core functions
    'discover_capabilities', 'load_module', 'register_plugin',
    # Namespaces (lazy loaded)
    'core', 'consciousness', 'processing', 'memory', 'subsystems',
    'interfaces', 'capabilities', 'extensions', 'tools', 'research',
    # Metadata
    '__version__', '__author__', '__description__'
]

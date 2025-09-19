#!/usr/bin/env python3
"""
🔌 Plugin System - DAWN Extension Framework
═══════════════════════════════════════════════════════════

Implements a dynamic plugin loading and management system for DAWN.
Enables community modules, experimental features, and custom extensions.
"""

import logging
import importlib
import inspect
import os
import sys
from typing import Dict, List, Any, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class PluginStatus(Enum):
    """Plugin status states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"

class PluginType(Enum):
    """Plugin type categories"""
    CAPABILITY = "capability"
    EXTENSION = "extension"
    INTERFACE = "interface"
    PROCESSOR = "processor"
    ANALYZER = "analyzer"
    CONNECTOR = "connector"
    EXPERIMENTAL = "experimental"

@dataclass
class PluginMetadata:
    """Plugin metadata and configuration"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    api_version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    entry_point: str = "main"
    min_dawn_version: str = "1.0.0"
    permissions: List[str] = field(default_factory=list)
    load_priority: int = 100
    auto_start: bool = False

@dataclass
class PluginInstance:
    """Runtime plugin instance"""
    metadata: PluginMetadata
    module: Any
    instance: Any
    status: PluginStatus
    config: Dict[str, Any] = field(default_factory=dict)
    load_time: Optional[datetime] = None
    error_message: Optional[str] = None
    performance_stats: Dict[str, Any] = field(default_factory=dict)

class PluginInterface:
    """Base interface that all plugins must implement"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.is_active = False
    
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        return True
    
    def start(self) -> bool:
        """Start the plugin. Return True if successful."""
        self.is_active = True
        return True
    
    def stop(self) -> bool:
        """Stop the plugin. Return True if successful."""
        self.is_active = False
        return True
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure the plugin with new settings."""
        self.config.update(config)
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current plugin status and metrics."""
        return {
            'name': self.name,
            'active': self.is_active,
            'config': self.config
        }
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this plugin provides."""
        return []

class PluginManager:
    """
    Manages the loading, configuration, and lifecycle of DAWN plugins.
    Provides dynamic extension capabilities for the DAWN system.
    """
    
    def __init__(self, plugin_directories: List[str] = None):
        self.plugin_directories = plugin_directories or [
            "dawn/extensions/official/modules",
            "dawn/extensions/community/modules", 
            "dawn/extensions/experimental/modules",
            "dawn/extensions/custom/modules"
        ]
        
        self.plugins: Dict[str, PluginInstance] = {}
        self.plugin_registry: Dict[str, PluginMetadata] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.load_order: List[str] = []
        
        self.security_policy = {
            'allow_experimental': True,
            'require_signatures': False,
            'allowed_permissions': ['read_data', 'write_logs', 'network_access'],
            'sandbox_mode': False
        }
        
        logger.info("🔌 Plugin Manager initialized")
        self._discover_plugins()
    
    def discover_plugins(self) -> Dict[str, PluginMetadata]:
        """Discover all available plugins in configured directories"""
        discovered = {}
        
        for directory in self.plugin_directories:
            if not os.path.exists(directory):
                logger.warning(f"🔌 Plugin directory not found: {directory}")
                continue
            
            logger.info(f"🔌 Scanning plugin directory: {directory}")
            
            for item in os.listdir(directory):
                plugin_path = os.path.join(directory, item)
                
                if os.path.isdir(plugin_path):
                    metadata = self._load_plugin_metadata(plugin_path)
                    if metadata:
                        discovered[metadata.name] = metadata
                        logger.info(f"🔌 Discovered plugin: {metadata.name} v{metadata.version}")
        
        self.plugin_registry.update(discovered)
        return discovered
    
    def load_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """
        Load a specific plugin by name.
        
        Args:
            plugin_name: Name of the plugin to load
            config: Optional configuration for the plugin
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if plugin_name in self.plugins:
            logger.warning(f"🔌 Plugin {plugin_name} already loaded")
            return True
        
        if plugin_name not in self.plugin_registry:
            logger.error(f"🔌 Plugin {plugin_name} not found in registry")
            return False
        
        metadata = self.plugin_registry[plugin_name]
        
        try:
            # Check dependencies
            if not self._check_dependencies(metadata):
                logger.error(f"🔌 Plugin {plugin_name} has unmet dependencies")
                return False
            
            # Security check
            if not self._security_check(metadata):
                logger.error(f"🔌 Plugin {plugin_name} failed security check")
                return False
            
            # Load the plugin module
            plugin_module = self._load_plugin_module(metadata)
            if not plugin_module:
                return False
            
            # Create plugin instance
            plugin_class = self._get_plugin_class(plugin_module, metadata)
            if not plugin_class:
                return False
            
            plugin_instance = plugin_class(config or {})
            
            # Initialize plugin
            if not plugin_instance.initialize():
                logger.error(f"🔌 Plugin {plugin_name} initialization failed")
                return False
            
            # Create plugin wrapper
            plugin_wrapper = PluginInstance(
                metadata=metadata,
                module=plugin_module,
                instance=plugin_instance,
                status=PluginStatus.LOADED,
                config=config or {},
                load_time=datetime.now()
            )
            
            self.plugins[plugin_name] = plugin_wrapper
            logger.info(f"🔌 Plugin {plugin_name} loaded successfully")
            
            # Auto-start if configured
            if metadata.auto_start:
                self.start_plugin(plugin_name)
            
            return True
            
        except Exception as e:
            logger.error(f"🔌 Failed to load plugin {plugin_name}: {e}")
            
            # Create error instance for tracking
            error_instance = PluginInstance(
                metadata=metadata,
                module=None,
                instance=None,
                status=PluginStatus.ERROR,
                error_message=str(e)
            )
            self.plugins[plugin_name] = error_instance
            
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin"""
        if plugin_name not in self.plugins:
            logger.warning(f"🔌 Plugin {plugin_name} not loaded")
            return True
        
        plugin = self.plugins[plugin_name]
        
        try:
            # Stop plugin if active
            if plugin.status == PluginStatus.ACTIVE and plugin.instance:
                plugin.instance.stop()
            
            # Remove from plugins
            del self.plugins[plugin_name]
            
            logger.info(f"🔌 Plugin {plugin_name} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"🔌 Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def start_plugin(self, plugin_name: str) -> bool:
        """Start a loaded plugin"""
        if plugin_name not in self.plugins:
            logger.error(f"🔌 Plugin {plugin_name} not loaded")
            return False
        
        plugin = self.plugins[plugin_name]
        
        if plugin.status != PluginStatus.LOADED:
            logger.error(f"🔌 Plugin {plugin_name} not in loaded state: {plugin.status}")
            return False
        
        try:
            if plugin.instance.start():
                plugin.status = PluginStatus.ACTIVE
                logger.info(f"🔌 Plugin {plugin_name} started successfully")
                return True
            else:
                logger.error(f"🔌 Plugin {plugin_name} start method returned False")
                return False
                
        except Exception as e:
            logger.error(f"🔌 Failed to start plugin {plugin_name}: {e}")
            plugin.status = PluginStatus.ERROR
            plugin.error_message = str(e)
            return False
    
    def stop_plugin(self, plugin_name: str) -> bool:
        """Stop an active plugin"""
        if plugin_name not in self.plugins:
            logger.error(f"🔌 Plugin {plugin_name} not loaded")
            return False
        
        plugin = self.plugins[plugin_name]
        
        try:
            if plugin.instance and plugin.instance.stop():
                plugin.status = PluginStatus.LOADED
                logger.info(f"🔌 Plugin {plugin_name} stopped successfully")
                return True
            else:
                logger.error(f"🔌 Plugin {plugin_name} stop method returned False")
                return False
                
        except Exception as e:
            logger.error(f"🔌 Failed to stop plugin {plugin_name}: {e}")
            return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin (unload and load again)"""
        config = None
        if plugin_name in self.plugins:
            config = self.plugins[plugin_name].config
        
        if not self.unload_plugin(plugin_name):
            return False
        
        return self.load_plugin(plugin_name, config)
    
    def get_plugin_status(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a plugin"""
        if plugin_name not in self.plugins:
            return None
        
        plugin = self.plugins[plugin_name]
        
        status = {
            'name': plugin_name,
            'status': plugin.status.value,
            'metadata': plugin.metadata.__dict__,
            'config': plugin.config,
            'load_time': plugin.load_time.isoformat() if plugin.load_time else None,
            'error_message': plugin.error_message
        }
        
        # Add instance status if available
        if plugin.instance:
            try:
                instance_status = plugin.instance.get_status()
                status['instance_status'] = instance_status
            except Exception as e:
                status['instance_status'] = {'error': str(e)}
        
        return status
    
    def list_plugins(self, status_filter: Optional[PluginStatus] = None) -> List[str]:
        """List all plugins, optionally filtered by status"""
        if status_filter:
            return [name for name, plugin in self.plugins.items() 
                   if plugin.status == status_filter]
        else:
            return list(self.plugins.keys())
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get all capabilities provided by active plugins"""
        capabilities = {}
        
        for name, plugin in self.plugins.items():
            if plugin.status == PluginStatus.ACTIVE and plugin.instance:
                try:
                    plugin_capabilities = plugin.instance.get_capabilities()
                    capabilities[name] = plugin_capabilities
                except Exception as e:
                    logger.warning(f"🔌 Failed to get capabilities from {name}: {e}")
                    capabilities[name] = []
        
        return capabilities
    
    def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Configure a plugin with new settings"""
        if plugin_name not in self.plugins:
            logger.error(f"🔌 Plugin {plugin_name} not loaded")
            return False
        
        plugin = self.plugins[plugin_name]
        
        try:
            if plugin.instance and plugin.instance.configure(config):
                plugin.config.update(config)
                logger.info(f"🔌 Plugin {plugin_name} configured successfully")
                return True
            else:
                logger.error(f"🔌 Plugin {plugin_name} configuration failed")
                return False
                
        except Exception as e:
            logger.error(f"🔌 Failed to configure plugin {plugin_name}: {e}")
            return False
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get overall plugin manager status"""
        status_counts = {}
        for plugin in self.plugins.values():
            status = plugin.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_plugins': len(self.plugins),
            'registry_size': len(self.plugin_registry),
            'plugin_directories': self.plugin_directories,
            'status_counts': status_counts,
            'active_capabilities': len(self.get_capabilities()),
            'security_policy': self.security_policy
        }
    
    # Helper methods
    def _discover_plugins(self) -> None:
        """Internal method to discover plugins on initialization"""
        self.discover_plugins()
    
    def _load_plugin_metadata(self, plugin_path: str) -> Optional[PluginMetadata]:
        """Load plugin metadata from plugin directory"""
        metadata_files = ['plugin.json', 'plugin.yaml', 'plugin.yml', 'manifest.json']
        
        for metadata_file in metadata_files:
            metadata_path = os.path.join(plugin_path, metadata_file)
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        if metadata_file.endswith('.json'):
                            data = json.load(f)
                        else:
                            data = yaml.safe_load(f)
                    
                    # Convert to PluginMetadata
                    plugin_type = PluginType(data.get('plugin_type', 'experimental'))
                    
                    metadata = PluginMetadata(
                        name=data['name'],
                        version=data['version'],
                        description=data.get('description', ''),
                        author=data.get('author', 'Unknown'),
                        plugin_type=plugin_type,
                        dependencies=data.get('dependencies', []),
                        api_version=data.get('api_version', '1.0'),
                        tags=data.get('tags', []),
                        config_schema=data.get('config_schema', {}),
                        entry_point=data.get('entry_point', 'main'),
                        min_dawn_version=data.get('min_dawn_version', '1.0.0'),
                        permissions=data.get('permissions', []),
                        load_priority=data.get('load_priority', 100),
                        auto_start=data.get('auto_start', False)
                    )
                    
                    return metadata
                    
                except Exception as e:
                    logger.error(f"🔌 Failed to load metadata from {metadata_path}: {e}")
        
        return None
    
    def _check_dependencies(self, metadata: PluginMetadata) -> bool:
        """Check if plugin dependencies are satisfied"""
        for dependency in metadata.dependencies:
            if dependency not in self.plugins or self.plugins[dependency].status != PluginStatus.LOADED:
                logger.warning(f"🔌 Dependency {dependency} not satisfied for {metadata.name}")
                return False
        return True
    
    def _security_check(self, metadata: PluginMetadata) -> bool:
        """Perform security checks on plugin"""
        # Check permissions
        for permission in metadata.permissions:
            if permission not in self.security_policy['allowed_permissions']:
                logger.warning(f"🔌 Plugin {metadata.name} requests unauthorized permission: {permission}")
                return False
        
        # Check experimental plugins
        if (metadata.plugin_type == PluginType.EXPERIMENTAL and 
            not self.security_policy['allow_experimental']):
            logger.warning(f"🔌 Experimental plugins not allowed: {metadata.name}")
            return False
        
        return True
    
    def _load_plugin_module(self, metadata: PluginMetadata) -> Any:
        """Load the plugin module"""
        try:
            # Construct module path
            plugin_dir = None
            for directory in self.plugin_directories:
                potential_path = os.path.join(directory, metadata.name)
                if os.path.exists(potential_path):
                    plugin_dir = potential_path
                    break
            
            if not plugin_dir:
                logger.error(f"🔌 Plugin directory not found for {metadata.name}")
                return None
            
            # Add plugin directory to Python path
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            # Import the module
            module_name = metadata.entry_point
            if not module_name.endswith('.py'):
                module_name += '.py'
            
            module_path = os.path.join(plugin_dir, module_name)
            if not os.path.exists(module_path):
                logger.error(f"🔌 Entry point not found: {module_path}")
                return None
            
            # Dynamic import
            spec = importlib.util.spec_from_file_location(metadata.name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            logger.error(f"🔌 Failed to load module for {metadata.name}: {e}")
            return None
    
    def _get_plugin_class(self, module: Any, metadata: PluginMetadata) -> Optional[Type]:
        """Get the main plugin class from the module"""
        # Look for classes that inherit from PluginInterface
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, PluginInterface) and obj != PluginInterface:
                return obj
        
        # Fallback: look for class with same name as plugin
        if hasattr(module, metadata.name):
            return getattr(module, metadata.name)
        
        # Fallback: look for main class
        if hasattr(module, 'Plugin'):
            return getattr(module, 'Plugin')
        
        logger.error(f"🔌 No suitable plugin class found in {metadata.name}")
        return None


# Global plugin manager instance
_plugin_manager = None

def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


# Example usage and testing
if __name__ == "__main__":
    print("🔌 Testing DAWN Plugin System")
    print("=" * 50)
    
    # Create plugin manager
    manager = PluginManager([])  # Empty directories for testing
    
    # Test discovery
    discovered = manager.discover_plugins()
    print(f"Discovered plugins: {list(discovered.keys())}")
    
    # Test manager status
    status = manager.get_manager_status()
    print(f"Manager status: {status}")
    
    print("\n🔌 Plugin System ready for use!")

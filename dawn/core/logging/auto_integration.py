#!/usr/bin/env python3
"""
🔗 Automatic Universal JSON Logging Integration
===============================================

This module automatically integrates universal JSON logging into ALL existing
DAWN modules, engines, and systems without requiring code changes. It uses
Python's import hooks, metaclasses, and monkey patching to ensure that
absolutely every process and module writes its state to disk in JSON format.

Key Features:
- Automatic discovery and hooking of all DAWN classes
- Zero-code-change integration for existing modules
- Metaclass injection for automatic state logging
- Method interception for tick-based logging
- Import hook for automatic registration
- Performance monitoring integration
"""

import sys
import importlib
import importlib.util
import inspect
import threading
import time
from typing import Dict, List, Any, Optional, Type, Callable, Set
from pathlib import Path
import logging

# Import our universal logger
try:
    from .universal_json_logger import (
        get_universal_logger, log_object_state, register_for_logging,
        auto_log_state, BatchLogging, LoggingConfig, LogFormat, StateScope
    )
except ImportError:
    # Handle relative import issues
    import sys
    sys.path.append(str(Path(__file__).parent))
    from universal_json_logger import (
        get_universal_logger, log_object_state, register_for_logging,
        auto_log_state, BatchLogging, LoggingConfig, LogFormat, StateScope
    )

logger = logging.getLogger(__name__)

class UniversalLoggingMeta(type):
    """Metaclass that automatically adds universal logging to all classes"""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Create the class normally
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        
        # Skip built-in classes and some problematic ones
        if (name.startswith('_') or 
            any(base.__module__ in ['builtins', 'abc'] for base in bases) or
            'logging' in cls.__module__ or
            'unittest' in cls.__module__):
            return cls
        
        # Add automatic logging to DAWN classes
        if (hasattr(cls, '__module__') and 
            cls.__module__ and 
            ('dawn' in cls.__module__ or 
             any(pattern in name.lower() for pattern in ['engine', 'manager', 'system', 'controller', 'orchestrator']))):
            
            # Hook __init__ for automatic registration
            original_init = cls.__init__
            
            def logged_init(self, *args, **kwargs):
                try:
                    original_init(self, *args, **kwargs)
                    # Register with universal logger
                    register_for_logging(self, f"{cls.__module__}.{cls.__name__}")
                except Exception as e:
                    logger.warning(f"Failed to register {cls.__name__} for logging: {e}")
                    original_init(self, *args, **kwargs)
            
            cls.__init__ = logged_init
            
            # Hook common methods for state logging
            methods_to_hook = ['tick', 'process', 'update', 'execute', 'run', 'step']
            
            for method_name in methods_to_hook:
                if hasattr(cls, method_name) and callable(getattr(cls, method_name)):
                    original_method = getattr(cls, method_name)
                    
                    def create_logged_method(orig_method, method_name):
                        def logged_method(self, *args, **kwargs):
                            result = orig_method(*args, **kwargs)
                            try:
                                # Log state after method execution
                                log_object_state(self, custom_metadata={
                                    'method_called': method_name,
                                    'call_time': time.time()
                                })
                            except Exception as e:
                                logger.debug(f"Failed to log state after {method_name}: {e}")
                            return result
                        return logged_method
                    
                    setattr(cls, method_name, create_logged_method(original_method, method_name))
        
        return cls

class DAWNImportHook:
    """Import hook that automatically applies universal logging to DAWN modules"""
    
    def __init__(self):
        self.hooked_modules: Set[str] = set()
        # Handle different __builtins__ formats
        if isinstance(__builtins__, dict):
            self.original_import = __builtins__['__import__']
        else:
            self.original_import = __builtins__.__import__
        
    def install(self):
        """Install the import hook"""
        if isinstance(__builtins__, dict):
            __builtins__['__import__'] = self.logged_import
        else:
            __builtins__.__import__ = self.logged_import
        logger.info("🔗 DAWN import hook installed for universal logging")
    
    def uninstall(self):
        """Uninstall the import hook"""
        if isinstance(__builtins__, dict):
            __builtins__['__import__'] = self.original_import
        else:
            __builtins__.__import__ = self.original_import
        logger.info("🔗 DAWN import hook uninstalled")
    
    def logged_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Hooked import function that adds logging to DAWN modules"""
        # Call original import
        module = self.original_import(name, globals, locals, fromlist, level)
        
        # Hook DAWN modules
        if (name.startswith('dawn.') or 
            (hasattr(module, '__name__') and module.__name__.startswith('dawn.'))):
            
            if name not in self.hooked_modules:
                self._hook_module(module)
                self.hooked_modules.add(name)
        
        return module
    
    def _hook_module(self, module):
        """Hook all classes in a module for universal logging"""
        try:
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if (inspect.isclass(attr) and 
                    hasattr(attr, '__module__') and 
                    attr.__module__ == module.__name__):
                    
                    # Apply universal logging metaclass
                    self._apply_logging_to_class(attr)
                    
        except Exception as e:
            logger.debug(f"Failed to hook module {module.__name__}: {e}")
    
    def _apply_logging_to_class(self, cls):
        """Apply universal logging to a class"""
        try:
            # Skip if already hooked
            if hasattr(cls, '_universal_logging_hooked'):
                return
            
            # Hook __init__ for registration
            if hasattr(cls, '__init__'):
                original_init = cls.__init__
                
                def logged_init(self, *args, **kwargs):
                    original_init(self, *args, **kwargs)
                    try:
                        register_for_logging(self, f"{cls.__module__}.{cls.__name__}")
                    except Exception as e:
                        logger.debug(f"Failed to register {cls.__name__}: {e}")
                
                cls.__init__ = logged_init
            
            # Mark as hooked
            cls._universal_logging_hooked = True
            
        except Exception as e:
            logger.debug(f"Failed to apply logging to class {cls.__name__}: {e}")

class AutoIntegrationManager:
    """Manages automatic integration of universal logging across all DAWN systems"""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or LoggingConfig(
            format=LogFormat.JSONL,
            scope=StateScope.FULL,
            enable_auto_discovery=True,
            flush_interval_seconds=1.0
        )
        
        self.universal_logger = get_universal_logger(self.config)
        self.import_hook = DAWNImportHook()
        
        # Integration statistics
        self.integration_stats = {
            'modules_hooked': 0,
            'classes_hooked': 0,
            'objects_registered': 0,
            'start_time': time.time()
        }
        
        # Background integration
        self._running = True
        self._integration_thread = threading.Thread(target=self._continuous_integration, daemon=True)
        
        logger.info("🔗 AutoIntegrationManager initialized")
    
    def start_integration(self):
        """Start automatic integration"""
        logger.info("🚀 Starting universal JSON logging integration...")
        
        # Install import hook
        self.import_hook.install()
        
        # Hook existing modules
        self._hook_existing_modules()
        
        # Start continuous integration
        self._integration_thread.start()
        
        logger.info("✅ Universal JSON logging integration active")
    
    def stop_integration(self):
        """Stop automatic integration"""
        logger.info("🛑 Stopping universal JSON logging integration...")
        
        self._running = False
        
        # Uninstall import hook
        self.import_hook.uninstall()
        
        # Wait for integration thread
        if self._integration_thread.is_alive():
            self._integration_thread.join(timeout=2.0)
        
        # Shutdown universal logger
        self.universal_logger.shutdown()
        
        logger.info("✅ Universal JSON logging integration stopped")
    
    def _hook_existing_modules(self):
        """Hook all existing DAWN modules"""
        dawn_modules = []
        
        # Find all loaded DAWN modules
        for module_name, module in sys.modules.items():
            if (module_name.startswith('dawn.') and 
                module is not None and 
                hasattr(module, '__file__')):
                dawn_modules.append((module_name, module))
        
        # Hook each module
        for module_name, module in dawn_modules:
            try:
                self.import_hook._hook_module(module)
                self.integration_stats['modules_hooked'] += 1
            except Exception as e:
                logger.debug(f"Failed to hook existing module {module_name}: {e}")
        
        logger.info(f"🔗 Hooked {len(dawn_modules)} existing DAWN modules")
    
    def _continuous_integration(self):
        """Continuously discover and integrate new objects"""
        while self._running:
            try:
                # Discover new objects
                discovered = self.universal_logger.object_tracker.discover_dawn_objects()
                self.integration_stats['objects_registered'] += len(discovered)
                
                # Log all tracked objects periodically
                logged_count = self.universal_logger.log_all_tracked_objects()
                
                if logged_count > 0:
                    logger.debug(f"🔍 Logged state for {logged_count} objects")
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in continuous integration: {e}")
                time.sleep(5.0)
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        runtime = time.time() - self.integration_stats['start_time']
        logger_stats = self.universal_logger.get_stats()
        
        return {
            **self.integration_stats,
            'runtime_seconds': runtime,
            'logger_stats': logger_stats,
            'total_objects_tracked': logger_stats['objects_tracked'],
            'total_states_logged': logger_stats['states_logged']
        }
    
    def force_log_all_dawn_objects(self, tick_number: Optional[int] = None) -> int:
        """Force logging of all discovered DAWN objects"""
        return self.universal_logger.log_all_tracked_objects(tick_number)

# Global integration manager
_integration_manager: Optional[AutoIntegrationManager] = None
_manager_lock = threading.Lock()

def get_integration_manager(config: Optional[LoggingConfig] = None) -> AutoIntegrationManager:
    """Get the global integration manager"""
    global _integration_manager
    
    with _manager_lock:
        if _integration_manager is None:
            _integration_manager = AutoIntegrationManager(config)
        return _integration_manager

def start_universal_logging(config: Optional[LoggingConfig] = None):
    """Start universal JSON logging for all DAWN systems"""
    manager = get_integration_manager(config)
    manager.start_integration()
    return manager

def stop_universal_logging():
    """Stop universal JSON logging"""
    global _integration_manager
    if _integration_manager:
        _integration_manager.stop_integration()
        _integration_manager = None

# Convenience functions for manual integration
def integrate_object(obj: Any, name: Optional[str] = None) -> str:
    """Manually integrate an object with universal logging"""
    return register_for_logging(obj, name)

def integrate_class(cls: Type, methods_to_hook: Optional[List[str]] = None):
    """Manually integrate a class with universal logging"""
    methods_to_hook = methods_to_hook or ['tick', 'process', 'update', 'execute']
    
    # Apply auto-logging decorator
    return auto_log_state(name=cls.__name__, log_on_methods=methods_to_hook)(cls)

def log_dawn_system_state(tick_number: Optional[int] = None) -> int:
    """Log state of all DAWN system objects"""
    manager = get_integration_manager()
    return manager.force_log_all_dawn_objects(tick_number)

# Context manager for batch DAWN system logging
class DAWNSystemBatchLogging(BatchLogging):
    """Specialized batch logging for DAWN systems"""
    
    def __init__(self, tick_number: Optional[int] = None, include_system_metrics: bool = True):
        super().__init__(tick_number)
        self.include_system_metrics = include_system_metrics
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log all DAWN objects in the system
        manager = get_integration_manager()
        logged_count = manager.force_log_all_dawn_objects(self.tick_number)
        
        if self.include_system_metrics:
            stats = manager.get_integration_stats()
            log_object_state(stats, name="integration_stats", 
                           custom_metadata={'logged_objects': logged_count})

# Automatic initialization when imported
def _auto_initialize():
    """Automatically initialize universal logging when this module is imported"""
    try:
        # Only auto-start if we're in a DAWN context
        if any('dawn' in module_name for module_name in sys.modules.keys()):
            logger.info("🔗 Auto-initializing universal JSON logging for DAWN")
            
            # Use a lightweight config for auto-initialization
            config = LoggingConfig(
                format=LogFormat.JSONL,
                scope=StateScope.CHANGES_ONLY,  # Less intensive
                flush_interval_seconds=2.0,
                enable_auto_discovery=True,
                max_file_size_mb=50
            )
            
            start_universal_logging(config)
    except Exception as e:
        logger.warning(f"Failed to auto-initialize universal logging: {e}")

# Auto-initialize when imported (can be disabled by setting environment variable)
import os
# DISABLED: Auto-initialization causes verbose output in CLI
# To enable, set DAWN_ENABLE_AUTO_LOGGING=1 environment variable
if os.getenv('DAWN_ENABLE_AUTO_LOGGING') == '1' and not hasattr(_auto_initialize, '_already_initialized'):
    _auto_initialize._already_initialized = True
    _auto_initialize()

if __name__ == "__main__":
    # Test the integration system
    logging.basicConfig(level=logging.INFO)
    
    # Create test DAWN-like classes
    class TestEngine:
        def __init__(self, name):
            self.name = name
            self.status = "initialized"
            self.tick_count = 0
        
        def tick(self):
            self.tick_count += 1
            self.status = f"ticking_{self.tick_count}"
        
        def process(self, data):
            return f"processed_{data}"
    
    class TestManager:
        def __init__(self):
            self.engines = []
            self.active = True
        
        def add_engine(self, engine):
            self.engines.append(engine)
        
        def update(self):
            for engine in self.engines:
                engine.tick()
    
    # Start universal logging
    manager = start_universal_logging(LoggingConfig(
        format=LogFormat.JSONL,
        flush_interval_seconds=0.5
    ))
    
    # Create test objects
    engine1 = TestEngine("consciousness_engine")
    engine2 = TestEngine("processing_engine")
    test_manager = TestManager()
    
    test_manager.add_engine(engine1)
    test_manager.add_engine(engine2)
    
    # Simulate activity
    for i in range(5):
        test_manager.update()
        time.sleep(0.2)
    
    # Get stats
    stats = manager.get_integration_stats()
    print(f"Integration Stats: {stats}")
    
    # Stop logging
    stop_universal_logging()

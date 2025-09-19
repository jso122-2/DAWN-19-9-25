#!/usr/bin/env python3
"""
DAWN Global Signal Configuration System
======================================

Provides standardized signal handling across all DAWN components.
Ensures consistent Ctrl-C behavior and graceful shutdown patterns.

Features:
- Global signal handler registration
- Configurable shutdown timeouts
- Graceful cleanup patterns
- Component-specific shutdown callbacks
- Thread-safe signal handling
- Logging integration for shutdown events

Usage:
    from dawn.core.signal_config import setup_global_signals, register_shutdown_callback
    
    # Set up global signal handling
    setup_global_signals(timeout=10.0)
    
    # Register component cleanup
    register_shutdown_callback("my_component", my_cleanup_function)
    
    # Your main loop
    while not is_shutdown_requested():
        do_work()
"""

import signal
import sys
import time
import threading
import logging
from typing import Dict, Callable, Optional, Any, List
from contextlib import contextmanager
from datetime import datetime
import traceback

# Global state for signal handling
_shutdown_requested = threading.Event()
_shutdown_callbacks: Dict[str, Callable[[], None]] = {}
_signal_handlers_installed = False
_shutdown_timeout = 30.0
_shutdown_in_progress = threading.Lock()
_logger = None

class SignalConfig:
    """Configuration class for global signal handling."""
    
    def __init__(self):
        self.timeout = 30.0
        self.log_shutdowns = True
        self.graceful_exit_codes = {
            signal.SIGINT: 130,   # Ctrl-C
            signal.SIGTERM: 143,  # Termination request
        }
        self.emergency_timeout = 5.0  # Hard exit after this many seconds
        self.verbose_shutdown = True
        self.cleanup_on_exit = True

# Global configuration instance
config = SignalConfig()

def get_logger():
    """Get or create logger for signal handling."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger('dawn.core.signal_config')
        if not _logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger

def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_requested.is_set()

def request_shutdown():
    """Request graceful shutdown."""
    _shutdown_requested.set()

def register_shutdown_callback(name: str, callback: Callable[[], None]):
    """
    Register a callback to be called during shutdown.
    
    Args:
        name: Unique name for this callback
        callback: Function to call during shutdown (should be quick)
    """
    global _shutdown_callbacks
    _shutdown_callbacks[name] = callback
    get_logger().debug(f"Registered shutdown callback: {name}")

def unregister_shutdown_callback(name: str):
    """Remove a shutdown callback."""
    global _shutdown_callbacks
    if name in _shutdown_callbacks:
        del _shutdown_callbacks[name]
        get_logger().debug(f"Unregistered shutdown callback: {name}")

def _signal_handler(signum, frame):
    """Global signal handler for graceful shutdown."""
    logger = get_logger()
    
    # Check if we're already shutting down
    if not _shutdown_in_progress.acquire(blocking=False):
        logger.warning("Shutdown already in progress, ignoring additional signals")
        return
    
    try:
        signal_name = signal.Signals(signum).name
        if config.verbose_shutdown:
            print(f"\n🛑 Received {signal_name} - Initiating graceful shutdown...")
        
        logger.info(f"Signal {signal_name} ({signum}) received, starting shutdown")
        
        # Set shutdown flag
        request_shutdown()
        
        # Start shutdown process in a separate thread to avoid blocking signal handler
        shutdown_thread = threading.Thread(
            target=_perform_shutdown,
            args=(signum,),
            name="SignalShutdownThread",
            daemon=True
        )
        shutdown_thread.start()
        
        # Set up emergency exit timer
        def emergency_exit():
            time.sleep(config.emergency_timeout)
            if shutdown_thread.is_alive():
                print(f"\n💥 Emergency exit after {config.emergency_timeout}s timeout!")
                logger.error("Emergency exit due to shutdown timeout")
                os._exit(config.graceful_exit_codes.get(signum, 1))
        
        emergency_thread = threading.Thread(
            target=emergency_exit,
            name="EmergencyExitThread",
            daemon=True
        )
        emergency_thread.start()
        
    finally:
        _shutdown_in_progress.release()

def _perform_shutdown(signum: int):
    """Perform the actual shutdown process."""
    logger = get_logger()
    start_time = time.time()
    
    try:
        if config.verbose_shutdown:
            print("🔄 Running shutdown callbacks...")
        
        # Execute shutdown callbacks
        failed_callbacks = []
        for name, callback in _shutdown_callbacks.items():
            try:
                if config.verbose_shutdown:
                    print(f"   • {name}...")
                
                callback()
                logger.debug(f"Shutdown callback '{name}' completed successfully")
                
            except Exception as e:
                failed_callbacks.append((name, str(e)))
                logger.error(f"Shutdown callback '{name}' failed: {e}")
                if config.verbose_shutdown:
                    print(f"   ❌ {name} failed: {e}")
        
        # Report results
        elapsed = time.time() - start_time
        if config.verbose_shutdown:
            if failed_callbacks:
                print(f"⚠️  Shutdown completed in {elapsed:.2f}s with {len(failed_callbacks)} failures")
                for name, error in failed_callbacks:
                    print(f"   ❌ {name}: {error}")
            else:
                print(f"✅ Graceful shutdown completed in {elapsed:.2f}s")
        
        logger.info(f"Shutdown completed in {elapsed:.2f}s")
        
        # Exit with appropriate code
        exit_code = config.graceful_exit_codes.get(signum, 0)
        if config.verbose_shutdown:
            print(f"👋 Exiting with code {exit_code}")
        
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        if config.verbose_shutdown:
            print(f"❌ Shutdown error: {e}")
        sys.exit(1)

def setup_global_signals(timeout: float = 30.0, 
                        verbose: bool = True,
                        emergency_timeout: float = 5.0,
                        log_level: str = "INFO"):
    """
    Set up global signal handling for consistent Ctrl-C behavior.
    
    Args:
        timeout: Maximum time to wait for graceful shutdown
        verbose: Whether to print shutdown progress messages
        emergency_timeout: Time before emergency exit
        log_level: Logging level for signal events
    """
    global _signal_handlers_installed, config
    
    if _signal_handlers_installed:
        get_logger().warning("Global signal handlers already installed")
        return
    
    # Update configuration
    config.timeout = timeout
    config.verbose_shutdown = verbose
    config.emergency_timeout = emergency_timeout
    
    # Set up logging
    logger = get_logger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    try:
        # Install signal handlers
        signal.signal(signal.SIGINT, _signal_handler)   # Ctrl-C
        signal.signal(signal.SIGTERM, _signal_handler)  # Termination
        
        # On Unix systems, also handle SIGHUP
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, _signal_handler)
        
        _signal_handlers_installed = True
        
        logger.info(f"Global signal handlers installed (timeout: {timeout}s)")
        if verbose:
            print("🛡️  Global signal handling enabled - Ctrl-C will trigger graceful shutdown")
        
    except Exception as e:
        logger.error(f"Failed to install signal handlers: {e}")
        raise

def configure_signals(**kwargs):
    """
    Configure signal handling options.
    
    Available options:
        timeout: Shutdown timeout in seconds
        verbose_shutdown: Print progress messages
        emergency_timeout: Emergency exit timeout
        log_shutdowns: Log shutdown events
    """
    global config
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            get_logger().debug(f"Signal config updated: {key} = {value}")
        else:
            get_logger().warning(f"Unknown signal config option: {key}")

@contextmanager
def signal_protection(name: str, cleanup_func: Optional[Callable[[], None]] = None):
    """
    Context manager that provides signal protection for a code block.
    
    Args:
        name: Name for this protected section
        cleanup_func: Optional cleanup function to call on shutdown
    
    Usage:
        with signal_protection("my_operation", cleanup_database):
            do_important_work()
    """
    if cleanup_func:
        register_shutdown_callback(name, cleanup_func)
    
    try:
        yield
    finally:
        if cleanup_func:
            unregister_shutdown_callback(name)

class GracefulShutdownMixin:
    """
    Mixin class that provides graceful shutdown capabilities to any class.
    
    Usage:
        class MyService(GracefulShutdownMixin):
            def __init__(self):
                super().__init__()
                self.setup_graceful_shutdown("my_service")
            
            def _cleanup(self):
                # Your cleanup code here
                pass
    """
    
    def setup_graceful_shutdown(self, name: str):
        """Set up graceful shutdown for this instance."""
        self._shutdown_name = name
        register_shutdown_callback(name, self._cleanup)
    
    def _cleanup(self):
        """Override this method to provide cleanup logic."""
        pass
    
    def __del__(self):
        """Ensure cleanup callback is removed."""
        if hasattr(self, '_shutdown_name'):
            unregister_shutdown_callback(self._shutdown_name)

def wait_for_shutdown(check_interval: float = 0.1):
    """
    Wait for shutdown signal in a main loop.
    
    Args:
        check_interval: How often to check for shutdown (seconds)
    
    Usage:
        setup_global_signals()
        start_my_services()
        wait_for_shutdown()  # Blocks until Ctrl-C
    """
    try:
        while not is_shutdown_requested():
            time.sleep(check_interval)
    except KeyboardInterrupt:
        # In case the signal handler didn't catch it
        request_shutdown()

def get_shutdown_status() -> Dict[str, Any]:
    """
    Get current shutdown status information.
    
    Returns:
        Dictionary with shutdown status details
    """
    return {
        'shutdown_requested': is_shutdown_requested(),
        'handlers_installed': _signal_handlers_installed,
        'registered_callbacks': list(_shutdown_callbacks.keys()),
        'config': {
            'timeout': config.timeout,
            'verbose_shutdown': config.verbose_shutdown,
            'emergency_timeout': config.emergency_timeout,
            'log_shutdowns': config.log_shutdowns,
        }
    }

# Convenience function for quick setup
def enable_graceful_shutdown(**kwargs):
    """
    Quick setup function for graceful shutdown.
    
    This is a convenience function that sets up signal handling with common defaults.
    
    Args:
        **kwargs: Configuration options passed to setup_global_signals
    """
    defaults = {
        'timeout': 30.0,
        'verbose': True,
        'emergency_timeout': 5.0,
        'log_level': 'INFO'
    }
    defaults.update(kwargs)
    
    setup_global_signals(**defaults)

# Auto-setup for imports (can be disabled by setting environment variable)
import os
if os.getenv('DAWN_AUTO_SIGNAL_SETUP', '1').lower() in ('1', 'true', 'yes'):
    try:
        enable_graceful_shutdown(verbose=False)
    except Exception:
        # Silently fail if we can't set up signals
        pass

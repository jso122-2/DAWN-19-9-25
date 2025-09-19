"""
Base Engine Interface
====================

Foundational interface that all DAWN engines must implement.
Provides standardized lifecycle, state management, and communication protocols.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import logging
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class EngineState(Enum):
    """Standard engine states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class EngineMetrics:
    """Standard metrics for all engines"""
    start_time: Optional[datetime] = None
    tick_count: int = 0
    error_count: int = 0
    last_tick_duration: float = 0.0
    average_tick_duration: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

class BaseEngine(ABC):
    """
    Base class for all DAWN engines.
    
    Provides standardized:
    - State management
    - Lifecycle methods
    - Event handling
    - Metrics collection
    - Error handling
    """
    
    def __init__(self, engine_id: str, config: Optional[Dict[str, Any]] = None):
        self.engine_id = engine_id
        self.config = config or {}
        self.state = EngineState.UNINITIALIZED
        self.metrics = EngineMetrics()
        self.event_handlers = {}
        self.lock = threading.RLock()
        self.error_callbacks = []
        self.state_change_callbacks = []
        
        logger.info(f"Created engine {engine_id}")
    
    # Abstract methods that must be implemented
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the engine. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the engine. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the engine. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def tick(self) -> Dict[str, Any]:
        """Execute one tick cycle. Must be implemented by subclasses."""
        pass
    
    # State management
    def get_state(self) -> EngineState:
        """Get current engine state"""
        return self.state
    
    def _set_state(self, new_state: EngineState):
        """Internal method to change state and notify callbacks"""
        with self.lock:
            old_state = self.state
            self.state = new_state
            logger.debug(f"Engine {self.engine_id} state: {old_state} -> {new_state}")
            
            # Notify state change callbacks
            for callback in self.state_change_callbacks:
                try:
                    callback(self.engine_id, old_state, new_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")
    
    # Event handling
    def on(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def emit(self, event_type: str, data: Any = None):
        """Emit an event"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")
    
    # Callback registration
    def on_error(self, callback: Callable[[str, Exception], None]):
        """Register error callback"""
        self.error_callbacks.append(callback)
    
    def on_state_change(self, callback: Callable[[str, EngineState, EngineState], None]):
        """Register state change callback"""
        self.state_change_callbacks.append(callback)
    
    # Error handling
    def _handle_error(self, error: Exception, context: str = ""):
        """Internal error handling"""
        self.metrics.error_count += 1
        self._set_state(EngineState.ERROR)
        
        logger.error(f"Engine {self.engine_id} error in {context}: {error}")
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(context, error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    # Metrics
    def get_metrics(self) -> EngineMetrics:
        """Get current engine metrics"""
        return self.metrics
    
    def update_metrics(self, **kwargs):
        """Update custom metrics"""
        self.metrics.custom_metrics.update(kwargs)
    
    # Configuration
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration"""
        self.config.update(updates)
    
    # Lifecycle helpers
    async def safe_initialize(self) -> bool:
        """Safe initialization with error handling"""
        if self.state != EngineState.UNINITIALIZED:
            logger.warning(f"Engine {self.engine_id} already initialized")
            return True
            
        try:
            self._set_state(EngineState.INITIALIZING)
            result = await self.initialize()
            if result:
                self._set_state(EngineState.READY)
                logger.info(f"Engine {self.engine_id} initialized successfully")
            else:
                self._set_state(EngineState.ERROR)
                logger.error(f"Engine {self.engine_id} initialization failed")
            return result
        except Exception as e:
            self._handle_error(e, "initialization")
            return False
    
    async def safe_start(self) -> bool:
        """Safe start with error handling"""
        if self.state not in [EngineState.READY, EngineState.STOPPED]:
            logger.warning(f"Engine {self.engine_id} not ready to start (state: {self.state})")
            return False
            
        try:
            result = await self.start()
            if result:
                self._set_state(EngineState.RUNNING)
                self.metrics.start_time = datetime.now()
                logger.info(f"Engine {self.engine_id} started successfully")
            return result
        except Exception as e:
            self._handle_error(e, "start")
            return False
    
    async def safe_stop(self) -> bool:
        """Safe stop with error handling"""
        if self.state not in [EngineState.RUNNING, EngineState.PAUSED]:
            logger.warning(f"Engine {self.engine_id} not running (state: {self.state})")
            return True
            
        try:
            self._set_state(EngineState.STOPPING)
            result = await self.stop()
            if result:
                self._set_state(EngineState.STOPPED)
                logger.info(f"Engine {self.engine_id} stopped successfully")
            return result
        except Exception as e:
            self._handle_error(e, "stop")
            return False
    
    async def safe_tick(self) -> Optional[Dict[str, Any]]:
        """Safe tick execution with metrics and error handling"""
        if self.state != EngineState.RUNNING:
            return None
            
        start_time = time.time()
        try:
            result = await self.tick()
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics.tick_count += 1
            self.metrics.last_tick_duration = duration
            
            # Update average duration
            if self.metrics.average_tick_duration == 0:
                self.metrics.average_tick_duration = duration
            else:
                self.metrics.average_tick_duration = (
                    self.metrics.average_tick_duration * 0.9 + duration * 0.1
                )
            
            return result
        except Exception as e:
            self._handle_error(e, "tick")
            return None
    
    # Utility methods
    def is_running(self) -> bool:
        """Check if engine is running"""
        return self.state == EngineState.RUNNING
    
    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.state in [EngineState.READY, EngineState.RUNNING, EngineState.PAUSED]
    
    def has_error(self) -> bool:
        """Check if engine is in error state"""
        return self.state == EngineState.ERROR
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.engine_id}, state={self.state.value})>"

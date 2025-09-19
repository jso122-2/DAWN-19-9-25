#!/usr/bin/env python3
"""
DAWN Core Telemetry System
==========================

Unified telemetry logging and monitoring system for the entire DAWN consciousness system.
Provides structured, high-performance telemetry collection across all subsystems.
"""

from .logger import DAWNTelemetryLogger, TelemetryLevel, TelemetryEvent
from .collector import TelemetryCollector
from .config import TelemetryConfig
from .exporters import JSONExporter, CSVExporter, PrometheusExporter

__version__ = "1.0.0"

# Global telemetry logger instance
_global_logger = None

def get_telemetry_logger() -> DAWNTelemetryLogger:
    """Get the global telemetry logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = DAWNTelemetryLogger()
    return _global_logger

def log_event(subsystem: str, component: str, event_type: str, 
              level: TelemetryLevel = TelemetryLevel.INFO, 
              data: dict = None, metadata: dict = None, tick_id: int = None) -> None:
    """
    Log a telemetry event to the global logger.
    
    Args:
        subsystem: Name of the DAWN subsystem (e.g., 'pulse_system', 'memory')
        component: Specific component within the subsystem
        event_type: Type of event (e.g., 'zone_transition', 'memory_allocation')
        level: Telemetry level
        data: Event-specific data
        metadata: Additional metadata
        tick_id: Current DAWN tick ID if available
    """
    logger = get_telemetry_logger()
    logger.log_event(subsystem, component, event_type, level, data, metadata, tick_id)

def log_performance(subsystem: str, component: str, operation: str, 
                   duration_ms: float, success: bool = True, 
                   metadata: dict = None, tick_id: int = None) -> None:
    """
    Log a performance telemetry event.
    
    Args:
        subsystem: Name of the DAWN subsystem
        component: Specific component within the subsystem  
        operation: Name of the operation being measured
        duration_ms: Duration in milliseconds
        success: Whether the operation succeeded
        metadata: Additional metadata
        tick_id: Current DAWN tick ID if available
    """
    data = {
        'operation': operation,
        'duration_ms': duration_ms,
        'success': success
    }
    log_event(subsystem, component, 'performance_metric', TelemetryLevel.INFO, data, metadata, tick_id)

def log_error(subsystem: str, component: str, error: Exception, 
              context: dict = None, tick_id: int = None) -> None:
    """
    Log an error telemetry event.
    
    Args:
        subsystem: Name of the DAWN subsystem
        component: Specific component within the subsystem
        error: The exception that occurred
        context: Additional context about the error
        tick_id: Current DAWN tick ID if available
    """
    data = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context or {}
    }
    log_event(subsystem, component, 'error', TelemetryLevel.ERROR, data, {}, tick_id)

def log_state_change(subsystem: str, component: str, from_state: str, to_state: str,
                    trigger: str = None, data: dict = None, tick_id: int = None) -> None:
    """
    Log a state change telemetry event.
    
    Args:
        subsystem: Name of the DAWN subsystem
        component: Specific component within the subsystem
        from_state: Previous state
        to_state: New state
        trigger: What triggered the state change
        data: Additional state data
        tick_id: Current DAWN tick ID if available
    """
    event_data = {
        'from_state': from_state,
        'to_state': to_state,
        'trigger': trigger
    }
    if data:
        event_data.update(data)
    
    log_event(subsystem, component, 'state_change', TelemetryLevel.INFO, event_data, {}, tick_id)

# Export key classes and functions
__all__ = [
    'DAWNTelemetryLogger',
    'TelemetryLevel', 
    'TelemetryEvent',
    'TelemetryCollector',
    'TelemetryConfig',
    'JSONExporter',
    'CSVExporter', 
    'PrometheusExporter',
    'get_telemetry_logger',
    'log_event',
    'log_performance',
    'log_error',
    'log_state_change'
]

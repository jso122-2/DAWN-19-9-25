#!/usr/bin/env python3
"""
DAWN Unified Telemetry System
=============================

Complete telemetry system integration for DAWN consciousness system.
Provides unified telemetry collection, aggregation, export, and monitoring
across all DAWN subsystems.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import uuid

from .logger import DAWNTelemetryLogger, TelemetryLevel
from .collector import TelemetryCollector
from .config import TelemetryConfig, load_telemetry_config
from .exporters import create_exporter, TelemetryExporter

logger = logging.getLogger(__name__)

class DAWNTelemetrySystem:
    """
    Unified telemetry system for DAWN consciousness system.
    
    Integrates:
    - Telemetry logging with structured events
    - Real-time data collection and aggregation
    - Multiple export formats (JSON, CSV, Prometheus, InfluxDB)
    - Configurable filtering and performance monitoring
    - Alert generation and health monitoring
    """
    
    def __init__(self, config: Optional[TelemetryConfig] = None, config_file: str = None, profile: str = None):
        """
        Initialize DAWN telemetry system.
        
        Args:
            config: Telemetry configuration instance
            config_file: Path to configuration file
            profile: Configuration profile name
        """
        # Load configuration
        if config:
            self.config = config
        else:
            self.config = load_telemetry_config(config_file=config_file, profile=profile)
        
        # Validate configuration
        config_issues = self.config.validate()
        if config_issues:
            logger.warning(f"Telemetry configuration issues: {config_issues}")
        
        # System identification
        self.system_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.running = False
        
        # Core components
        self.telemetry_logger = None
        self.telemetry_collector = None
        self.exporters = []
        
        # Integration state
        self.integrated_subsystems = set()
        self.alert_handlers = []
        self.performance_monitors = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"🔍 DAWN Telemetry System initialized: {self.system_id[:8]}")
        logger.info(f"   Profile: {getattr(self.config, 'profile', 'custom')}")
        logger.info(f"   Output formats: {self.config.output.enabled_formats}")
        logger.info(f"   Buffer size: {self.config.buffer.max_size}")
    
    def _initialize_components(self) -> None:
        """Initialize telemetry system components."""
        try:
            # Initialize telemetry logger
            logger_config = {
                'enabled': self.config.enabled,
                'buffer_size': self.config.buffer.max_size,
                'min_level': self.config.filtering.min_level,
                'auto_flush_interval': self.config.buffer.auto_flush_interval,
                'auto_flush_enabled': self.config.buffer.auto_flush_enabled,
                'collect_system_metrics': self.config.system_metrics.enabled,
                'system_metrics_interval': self.config.system_metrics.collection_interval,
                'subsystem_filters': self.config.filtering.subsystem_filters,
                'component_filters': self.config.filtering.component_filters
            }
            
            self.telemetry_logger = DAWNTelemetryLogger(logger_config)
            
            # Initialize telemetry collector
            self.telemetry_collector = TelemetryCollector(self.config, self.telemetry_logger)
            
            # Initialize exporters
            self._initialize_exporters()
            
            logger.info("✅ Telemetry system components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize telemetry components: {e}")
            raise
    
    def _initialize_exporters(self) -> None:
        """Initialize telemetry exporters based on configuration."""
        output_dir = Path(self.config.output.output_directory)
        
        for format_type in self.config.output.enabled_formats:
            try:
                exporter_config = {
                    'max_file_size_mb': self.config.output.file_rotation_size_mb,
                    'max_files': self.config.output.max_files,
                    'compress': self.config.output.compress_rotated
                }
                
                # Create format-specific output path
                if format_type == 'json':
                    output_path = output_dir / 'json'
                elif format_type == 'csv':
                    output_path = output_dir / 'csv'
                elif format_type == 'prometheus':
                    output_path = output_dir / 'prometheus'
                    exporter_config['export_interval'] = 60
                elif format_type == 'influxdb':
                    output_path = output_dir / 'influxdb'
                    exporter_config['batch_size'] = 1000
                else:
                    logger.warning(f"Unknown export format: {format_type}")
                    continue
                
                exporter = create_exporter(format_type, str(output_path), exporter_config)
                self.exporters.append(exporter)
                
                # Register exporter with telemetry logger
                self.telemetry_logger.add_output_handler(exporter)
                
                logger.info(f"✅ {format_type.upper()} exporter initialized: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {format_type} exporter: {e}")
    
    def start(self) -> None:
        """Start the telemetry system."""
        if self.running:
            logger.warning("Telemetry system already running")
            return
        
        try:
            self.running = True
            
            # Start telemetry logger
            if self.telemetry_logger:
                self.telemetry_logger.start()
            
            # Start telemetry collector
            if self.telemetry_collector:
                self.telemetry_collector.start()
            
            # Log system startup
            self.log_event(
                'telemetry_system', 'core', 'system_started',
                TelemetryLevel.INFO,
                {
                    'system_id': self.system_id,
                    'config_profile': getattr(self.config, 'profile', 'custom'),
                    'enabled_formats': self.config.output.enabled_formats,
                    'buffer_size': self.config.buffer.max_size
                }
            )
            
            logger.info("🚀 DAWN Telemetry System started")
            
        except Exception as e:
            logger.error(f"Failed to start telemetry system: {e}")
            self.running = False
            raise
    
    def stop(self) -> None:
        """Stop the telemetry system."""
        if not self.running:
            return
        
        try:
            # Log system shutdown
            self.log_event(
                'telemetry_system', 'core', 'system_stopping',
                TelemetryLevel.INFO,
                {
                    'uptime_seconds': time.time() - self.start_time,
                    'events_logged': self.get_system_metrics().get('events_logged', 0)
                }
            )
            
            self.running = False
            
            # Stop collector
            if self.telemetry_collector:
                self.telemetry_collector.stop()
            
            # Stop logger
            if self.telemetry_logger:
                self.telemetry_logger.stop()
            
            # Close exporters
            for exporter in self.exporters:
                try:
                    exporter.close()
                except Exception as e:
                    logger.error(f"Error closing exporter: {e}")
            
            logger.info("🛑 DAWN Telemetry System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping telemetry system: {e}")
    
    def log_event(self, subsystem: str, component: str, event_type: str,
                  level: TelemetryLevel = TelemetryLevel.INFO,
                  data: Optional[Dict[str, Any]] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  tick_id: Optional[int] = None) -> None:
        """
        Log a telemetry event.
        
        Args:
            subsystem: DAWN subsystem name
            component: Component within subsystem
            event_type: Type of event
            level: Telemetry level
            data: Event-specific data
            metadata: Additional metadata
            tick_id: Current DAWN tick ID
        """
        if self.telemetry_logger:
            self.telemetry_logger.log_event(
                subsystem, component, event_type, level, data, metadata, tick_id
            )
    
    def log_performance(self, subsystem: str, component: str, operation: str,
                       duration_ms: float, success: bool = True,
                       metadata: Optional[Dict[str, Any]] = None,
                       tick_id: Optional[int] = None) -> None:
        """Log a performance telemetry event."""
        data = {
            'operation': operation,
            'duration_ms': duration_ms,
            'success': success
        }
        self.log_event(subsystem, component, 'performance_metric', TelemetryLevel.INFO, data, metadata, tick_id)
    
    def log_error(self, subsystem: str, component: str, error: Exception,
                  context: Optional[Dict[str, Any]] = None,
                  tick_id: Optional[int] = None) -> None:
        """Log an error telemetry event."""
        data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        self.log_event(subsystem, component, 'error', TelemetryLevel.ERROR, data, {}, tick_id)
    
    def log_state_change(self, subsystem: str, component: str, from_state: str, to_state: str,
                        trigger: str = None, data: Optional[Dict[str, Any]] = None,
                        tick_id: Optional[int] = None) -> None:
        """Log a state change telemetry event."""
        event_data = {
            'from_state': from_state,
            'to_state': to_state,
            'trigger': trigger
        }
        if data:
            event_data.update(data)
        
        self.log_event(subsystem, component, 'state_change', TelemetryLevel.INFO, event_data, {}, tick_id)
    
    def create_performance_context(self, subsystem: str, component: str, operation: str, tick_id: int = None):
        """Create a context manager for performance telemetry."""
        class PerformanceContext:
            def __init__(self, telemetry_system):
                self.telemetry_system = telemetry_system
                self.start_time = None
                self.success = True
                self.metadata = {}
            
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration_ms = (time.perf_counter() - self.start_time) * 1000
                self.success = exc_type is None
                
                self.telemetry_system.log_performance(
                    subsystem, component, operation, duration_ms, self.success, self.metadata, tick_id
                )
                
                if not self.success:
                    self.telemetry_system.log_error(subsystem, component, exc_val, {}, tick_id)
            
            def add_metadata(self, key: str, value: Any):
                self.metadata[key] = value
        
        return PerformanceContext(self)
    
    def integrate_subsystem(self, subsystem_name: str, 
                          components: List[str] = None,
                          custom_filters: Dict[str, Any] = None) -> None:
        """
        Integrate a DAWN subsystem with telemetry.
        
        Args:
            subsystem_name: Name of the subsystem
            components: List of components in the subsystem
            custom_filters: Custom filtering rules for this subsystem
        """
        with self.lock:
            self.integrated_subsystems.add(subsystem_name)
            
            # Apply custom filters if provided
            if custom_filters and self.telemetry_logger:
                for component in (components or []):
                    self.telemetry_logger.configure_filtering(
                        subsystem=subsystem_name,
                        component=component,
                        **custom_filters
                    )
            
            # Log integration
            self.log_event(
                'telemetry_system', 'integration', 'subsystem_integrated',
                TelemetryLevel.INFO,
                {
                    'subsystem': subsystem_name,
                    'components': components or [],
                    'custom_filters': bool(custom_filters)
                }
            )
            
            logger.info(f"✅ Integrated subsystem: {subsystem_name}")
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Add an alert handler for telemetry alerts."""
        if self.telemetry_collector:
            self.telemetry_collector.add_alert_handler(handler)
        
        self.alert_handlers.append(handler)
        logger.info(f"Added telemetry alert handler: {handler}")
    
    def configure_filtering(self, subsystem: str = None, component: str = None,
                          enabled: bool = True, min_level: str = None) -> None:
        """Configure telemetry filtering."""
        if self.telemetry_logger:
            self.telemetry_logger.configure_filtering(subsystem, component, enabled, min_level)
            
            self.log_event(
                'telemetry_system', 'config', 'filtering_updated',
                TelemetryLevel.INFO,
                {
                    'subsystem': subsystem,
                    'component': component,
                    'enabled': enabled,
                    'min_level': min_level
                }
            )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive telemetry system metrics."""
        metrics = {
            'system_id': self.system_id,
            'uptime_seconds': time.time() - self.start_time,
            'running': self.running,
            'integrated_subsystems': list(self.integrated_subsystems)
        }
        
        # Add logger metrics
        if self.telemetry_logger:
            logger_metrics = self.telemetry_logger.get_metrics()
            metrics.update({f'logger_{k}': v for k, v in logger_metrics.items()})
        
        # Add collector metrics
        if self.telemetry_collector:
            collector_metrics = self.telemetry_collector.get_collector_metrics()
            metrics.update({f'collector_{k}': v for k, v in collector_metrics.items()})
        
        # Add exporter info
        metrics['exporters'] = [type(exp).__name__ for exp in self.exporters]
        
        return metrics
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get system health summary."""
        health = {
            'overall_status': 'healthy' if self.running else 'stopped',
            'components': {
                'logger': 'healthy' if self.telemetry_logger and self.telemetry_logger.running else 'stopped',
                'collector': 'healthy' if self.telemetry_collector and self.telemetry_collector.running else 'stopped',
                'exporters': len(self.exporters)
            }
        }
        
        # Add subsystem health from collector
        if self.telemetry_collector:
            subsystem_health = self.telemetry_collector.get_subsystem_health()
            health['subsystems'] = subsystem_health
            
            # Calculate overall health score
            if subsystem_health:
                health_scores = [info.get('health_score', 1.0) for info in subsystem_health.values()]
                health['overall_health_score'] = sum(health_scores) / len(health_scores)
            else:
                health['overall_health_score'] = 1.0
        
        return health
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if self.telemetry_collector:
            return self.telemetry_collector.get_performance_summary()
        return {}
    
    def export_telemetry_data(self, hours: int = 1) -> Dict[str, Any]:
        """Export telemetry data for analysis."""
        if self.telemetry_collector:
            return self.telemetry_collector.export_aggregation_data(hours)
        return {}
    
    def get_recent_events(self, count: int = 100, subsystem: str = None) -> List[Dict[str, Any]]:
        """Get recent telemetry events."""
        if not self.telemetry_logger:
            return []
        
        events = self.telemetry_logger.get_recent_events(count * 2)  # Get more to filter
        
        if subsystem:
            events = [e for e in events if e.get('subsystem') == subsystem]
        
        return events[:count]
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

# Global telemetry system instance
_global_telemetry_system = None

def get_telemetry_system() -> Optional[DAWNTelemetrySystem]:
    """Get the global telemetry system instance."""
    return _global_telemetry_system

def initialize_telemetry_system(config: Optional[TelemetryConfig] = None, 
                               config_file: str = None, 
                               profile: str = None) -> DAWNTelemetrySystem:
    """
    Initialize the global telemetry system.
    
    Args:
        config: Telemetry configuration instance
        config_file: Path to configuration file
        profile: Configuration profile name
        
    Returns:
        DAWNTelemetrySystem instance
    """
    global _global_telemetry_system
    
    if _global_telemetry_system is not None:
        logger.warning("Telemetry system already initialized")
        return _global_telemetry_system
    
    _global_telemetry_system = DAWNTelemetrySystem(config, config_file, profile)
    return _global_telemetry_system

def shutdown_telemetry_system() -> None:
    """Shutdown the global telemetry system."""
    global _global_telemetry_system
    
    if _global_telemetry_system:
        _global_telemetry_system.stop()
        _global_telemetry_system = None

# Convenience functions that use the global system
def log_event(subsystem: str, component: str, event_type: str,
              level: TelemetryLevel = TelemetryLevel.INFO,
              data: Optional[Dict[str, Any]] = None,
              metadata: Optional[Dict[str, Any]] = None,
              tick_id: Optional[int] = None) -> None:
    """Log event using global telemetry system."""
    system = get_telemetry_system()
    if system:
        system.log_event(subsystem, component, event_type, level, data, metadata, tick_id)

def log_performance(subsystem: str, component: str, operation: str,
                   duration_ms: float, success: bool = True,
                   metadata: Optional[Dict[str, Any]] = None,
                   tick_id: Optional[int] = None) -> None:
    """Log performance event using global telemetry system."""
    system = get_telemetry_system()
    if system:
        system.log_performance(subsystem, component, operation, duration_ms, success, metadata, tick_id)

def log_error(subsystem: str, component: str, error: Exception,
              context: Optional[Dict[str, Any]] = None,
              tick_id: Optional[int] = None) -> None:
    """Log error event using global telemetry system."""
    system = get_telemetry_system()
    if system:
        system.log_error(subsystem, component, error, context, tick_id)

def create_performance_context(subsystem: str, component: str, operation: str, tick_id: int = None):
    """Create performance context using global telemetry system."""
    system = get_telemetry_system()
    if system:
        return system.create_performance_context(subsystem, component, operation, tick_id)
    
    # Return a no-op context if no telemetry system
    class NoOpContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def add_metadata(self, key, value): pass
    
    return NoOpContext()

if __name__ == "__main__":
    # Demo telemetry system
    print("🔍 DAWN Unified Telemetry System Demo")
    
    # Initialize with development profile
    telemetry = initialize_telemetry_system(profile="development")
    
    try:
        # Start telemetry
        telemetry.start()
        
        # Demo logging
        telemetry.log_event('demo', 'test', 'system_demo', TelemetryLevel.INFO, {'demo': True})
        
        # Demo performance context
        with telemetry.create_performance_context('demo', 'test', 'demo_operation') as ctx:
            ctx.add_metadata('demo_metadata', 'test_value')
            time.sleep(0.1)  # Simulate work
        
        # Demo error logging
        try:
            raise ValueError("Demo error")
        except ValueError as e:
            telemetry.log_error('demo', 'test', e, {'context': 'demo'})
        
        # Wait a bit for processing
        time.sleep(2)
        
        # Show metrics
        metrics = telemetry.get_system_metrics()
        print(f"\n📊 System Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        # Show health
        health = telemetry.get_health_summary()
        print(f"\n🏥 Health Summary:")
        for key, value in health.items():
            print(f"   {key}: {value}")
        
    finally:
        telemetry.stop()
        shutdown_telemetry_system()
    
    print("\n✅ Telemetry system demo complete")

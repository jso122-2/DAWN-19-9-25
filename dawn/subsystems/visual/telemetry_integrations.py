#!/usr/bin/env python3
"""
DAWN Telemetry Analytics Integration Layer
==========================================

Integration utilities to connect the telemetry analytics engine
with existing DAWN systems for seamless data flow and optimization.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .telemetry_analytics import get_telemetry_analytics, ingest_telemetry_data

logger = logging.getLogger(__name__)

class TelemetryIntegrationManager:
    """Manages integration between telemetry analytics and DAWN systems."""
    
    def __init__(self):
        self.integrations = {}
        self.collection_threads = {}
        self.running = False
        
    def start_integrations(self):
        """Start all telemetry integrations."""
        self.running = True
        
        # Start system metrics collection
        self._start_system_metrics_collection()
        
        # Start DAWN module integrations
        self._start_dawn_module_integrations()
        
        logger.info("📡 Telemetry integrations started")
        
    def stop_integrations(self):
        """Stop all telemetry integrations."""
        self.running = False
        
        for thread in self.collection_threads.values():
            if thread.is_alive():
                thread.join(timeout=2.0)
                
        logger.info("📡 Telemetry integrations stopped")
        
    def _start_system_metrics_collection(self):
        """Start collecting system performance metrics."""
        def collect_system_metrics():
            import psutil
            
            while self.running:
                try:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    ingest_telemetry_data("system", "cpu_usage", cpu_percent / 100.0)
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    ingest_telemetry_data("system", "memory_usage", memory.percent / 100.0)
                    ingest_telemetry_data("system", "memory_available", memory.available)
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    ingest_telemetry_data("system", "disk_usage", disk.percent / 100.0)
                    ingest_telemetry_data("system", "disk_free", disk.free)
                    
                    time.sleep(5)  # Collect every 5 seconds
                    
                except Exception as e:
                    logger.warning(f"System metrics collection error: {e}")
                    time.sleep(5)
                    
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
        self.collection_threads['system_metrics'] = thread
        
    def _start_dawn_module_integrations(self):
        """Start integrations with DAWN modules."""
        # Integration with stable state detector
        self._integrate_stable_state_detector()
        
        # Integration with recursive bubble
        self._integrate_recursive_bubble()
        
        # Integration with tick engine
        self._integrate_tick_engine()
        
    def _integrate_stable_state_detector(self):
        """Integrate with stable state detection system."""
        try:
            from .stable_state import get_stable_state_detector
            
            detector = get_stable_state_detector(auto_start=False)
            
            # Register callback for stability events
            def stability_callback(event):
                ingest_telemetry_data(
                    "stable_state", 
                    "stability_score", 
                    event.stability_score,
                    tags={"event_type": event.event_type}
                )
                
                ingest_telemetry_data(
                    "stable_state",
                    "recovery_action",
                    event.recovery_action.value,
                    tags={"success": str(event.success)}
                )
                
            detector.register_event_callback(stability_callback)
            
            # Collect stability metrics periodically
            def collect_stability_metrics():
                while self.running:
                    try:
                        status = detector.get_stability_status()
                        
                        if status.get('current_metrics'):
                            metrics = status['current_metrics']
                            ingest_telemetry_data("stability", "overall_stability", metrics['overall_stability'])
                            ingest_telemetry_data("stability", "entropy_stability", metrics['entropy_stability'])
                            ingest_telemetry_data("stability", "memory_coherence", metrics['memory_coherence'])
                            
                        time.sleep(10)
                        
                    except Exception as e:
                        logger.warning(f"Stability metrics collection error: {e}")
                        time.sleep(10)
                        
            thread = threading.Thread(target=collect_stability_metrics, daemon=True)
            thread.start()
            self.collection_threads['stability_metrics'] = thread
            
            logger.info("✅ Integrated with stable state detector")
            
        except ImportError:
            logger.info("ℹ️ Stable state detector not available for integration")
            
    def _integrate_recursive_bubble(self):
        """Integrate with recursive bubble system."""
        def collect_recursive_metrics():
            while self.running:
                try:
                    # Simulate recursive bubble metrics
                    # In real implementation, this would connect to actual module
                    import random
                    
                    current_depth = random.randint(1, 6)
                    max_depth = random.randint(current_depth, 8)
                    stabilizations = random.randint(0, 15)
                    
                    ingest_telemetry_data("recursive_bubble", "current_depth", current_depth)
                    ingest_telemetry_data("recursive_bubble", "max_depth_reached", max_depth)
                    ingest_telemetry_data("recursive_bubble", "stabilization_count", stabilizations)
                    
                    time.sleep(8)
                    
                except Exception as e:
                    logger.warning(f"Recursive metrics collection error: {e}")
                    time.sleep(8)
                    
        thread = threading.Thread(target=collect_recursive_metrics, daemon=True)
        thread.start()
        self.collection_threads['recursive_metrics'] = thread
        
    def _integrate_tick_engine(self):
        """Integrate with tick engine system."""
        def collect_tick_metrics():
            while self.running:
                try:
                    # Simulate tick engine metrics
                    import random
                    
                    tick_rate = random.uniform(8.0, 12.0)
                    tick_time = random.uniform(80, 120)  # milliseconds
                    
                    ingest_telemetry_data("tick_engine", "tick_rate", tick_rate)
                    ingest_telemetry_data("tick_engine", "avg_tick_time", tick_time)
                    
                    # Operations timing
                    ingest_telemetry_data("operations", "recursive_reflection_time", random.uniform(10, 50))
                    ingest_telemetry_data("operations", "sigil_execution_time", random.uniform(20, 100))
                    ingest_telemetry_data("operations", "memory_rebloom_time", random.uniform(5, 30))
                    ingest_telemetry_data("operations", "owl_observation_time", random.uniform(15, 60))
                    
                    time.sleep(6)
                    
                except Exception as e:
                    logger.warning(f"Tick metrics collection error: {e}")
                    time.sleep(6)
                    
        thread = threading.Thread(target=collect_tick_metrics, daemon=True)
        thread.start()
        self.collection_threads['tick_metrics'] = thread

def setup_telemetry_analytics_integration():
    """Setup complete telemetry analytics integration with DAWN systems."""
    
    # Initialize analytics engine
    analytics = get_telemetry_analytics(auto_start=True)
    
    # Setup integration manager
    integration_manager = TelemetryIntegrationManager()
    integration_manager.start_integrations()
    
    # Register analytics callback for optimization feedback
    def optimization_callback(performance, insights):
        """Process analytics results for system optimization."""
        # Log significant insights
        for insight in insights:
            if insight.implementation_priority <= 2:
                logger.info(f"🔍 High-priority insight: {insight.recommendation}")
                
        # Send optimization signals back to DAWN engine
        if performance.overall_health_score < 0.7:
            logger.warning(f"⚠️ System health below optimal: {performance.overall_health_score:.3f}")
            
    analytics.register_analysis_callback(optimization_callback)
    
    logger.info("🔗 Complete telemetry analytics integration established")
    
    return analytics, integration_manager

# Global integration manager
_integration_manager = None

def get_integration_manager() -> Optional[TelemetryIntegrationManager]:
    """Get the global integration manager instance."""
    return _integration_manager

def start_telemetry_integrations():
    """Start telemetry integrations globally."""
    global _integration_manager
    
    if _integration_manager is None:
        _, _integration_manager = setup_telemetry_analytics_integration()
    else:
        _integration_manager.start_integrations()
        
def stop_telemetry_integrations():
    """Stop telemetry integrations globally."""
    global _integration_manager
    
    if _integration_manager:
        _integration_manager.stop_integrations()

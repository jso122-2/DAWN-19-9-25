#!/usr/bin/env python3
"""
DAWN Tracer System
==================

Core tracing and telemetry system for DAWN's consciousness monitoring.
Provides real-time instrumentation of consciousness operations.
"""

import time
import uuid
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class TraceEvent:
    """Individual trace event record"""
    trace_id: str
    parent_id: Optional[str]
    operation: str
    module: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Union[int, float]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    status: str = "active"  # active, completed, error

@dataclass
class TracingSession:
    """Active tracing session context"""
    trace_id: str
    module: str
    operation: str
    start_time: float
    tracer: 'DAWNTracer'
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Union[int, float]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def log_metric(self, name: str, value: Union[int, float]):
        """Log a metric during this trace"""
        self.metrics[name] = value
        
    def log_error(self, error_type: str, message: str):
        """Log an error during this trace"""
        error_entry = f"{error_type}: {message}"
        self.errors.append(error_entry)
        logger.error(f"Trace {self.trace_id}: {error_entry}")
        
    def log_metadata(self, key: str, value: Any):
        """Log metadata during this trace"""
        self.metadata[key] = value

class DAWNTracer:
    """DAWN's core tracing and telemetry system"""
    
    def __init__(self, buffer_size: int = 10000):
        self.tracer_id = str(uuid.uuid4())
        self.buffer_size = buffer_size
        self.trace_buffer = deque(maxlen=buffer_size)
        self.active_traces = {}
        self.metrics = {
            "total_traces": 0,
            "active_traces": 0,
            "completed_traces": 0,
            "error_traces": 0,
            "total_duration_ms": 0.0
        }
        self.module_stats = defaultdict(lambda: {
            "trace_count": 0,
            "total_duration_ms": 0.0,
            "error_count": 0
        })
        
        # Thread safety
        self._lock = threading.Lock()
        self._background_collection = False
        self._collection_thread = None
        
        logger.info(f"🔗 DAWNTracer initialized: {self.tracer_id}")
    
    @contextmanager
    def trace(self, module: str, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations"""
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create tracing session
        session = TracingSession(
            trace_id=trace_id,
            module=module,
            operation=operation,
            start_time=start_time,
            tracer=self,
            metadata=metadata or {}
        )
        
        # Register active trace
        with self._lock:
            self.active_traces[trace_id] = session
            self.metrics["active_traces"] += 1
            self.metrics["total_traces"] += 1
        
        try:
            yield session
            status = "completed"
            
        except Exception as e:
            session.log_error("trace_exception", str(e))
            status = "error"
            with self._lock:
                self.metrics["error_traces"] += 1
            raise
            
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Create trace event
            trace_event = TraceEvent(
                trace_id=trace_id,
                parent_id=None,  # Could implement trace hierarchy later
                operation=operation,
                module=module,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                metadata=session.metadata.copy(),
                metrics=session.metrics.copy(),
                errors=session.errors.copy(),
                status=status
            )
            
            # Update metrics and storage
            with self._lock:
                self.trace_buffer.append(trace_event)
                self.active_traces.pop(trace_id, None)
                self.metrics["active_traces"] -= 1
                self.metrics["completed_traces"] += 1
                self.metrics["total_duration_ms"] += duration_ms
                
                # Update module stats
                self.module_stats[module]["trace_count"] += 1
                self.module_stats[module]["total_duration_ms"] += duration_ms
                if status == "error":
                    self.module_stats[module]["error_count"] += 1
            
            logger.debug(f"Trace completed: {module}.{operation} ({duration_ms:.2f}ms)")
    
    def start_background_collection(self):
        """Start background trace collection and processing"""
        if self._background_collection:
            return
            
        self._background_collection = True
        self._collection_thread = threading.Thread(
            target=self._background_collection_loop,
            name="dawn_tracer_collection",
            daemon=True
        )
        self._collection_thread.start()
        logger.info("🔗 Tracer background collection started")
    
    def stop_background_collection(self):
        """Stop background trace collection"""
        if not self._background_collection:
            return
            
        self._background_collection = False
        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=5.0)
        logger.info("🔗 Tracer background collection stopped")
    
    def _background_collection_loop(self):
        """Background loop for processing traces"""
        while self._background_collection:
            try:
                # Process traces, perform cleanup, etc.
                self._cleanup_old_traces()
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in tracer background loop: {e}")
                time.sleep(5.0)  # Brief pause on error
    
    def _cleanup_old_traces(self):
        """Clean up old traces and manage memory"""
        # This could implement more sophisticated cleanup logic
        # For now, the deque automatically handles size limits
        pass
    
    def get_trace_summary(self, module: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of tracing activity"""
        with self._lock:
            if module:
                module_data = self.module_stats.get(module, {})
                return {
                    "module": module,
                    "trace_count": module_data.get("trace_count", 0),
                    "total_duration_ms": module_data.get("total_duration_ms", 0),
                    "avg_duration_ms": (
                        module_data.get("total_duration_ms", 0) / 
                        max(1, module_data.get("trace_count", 1))
                    ),
                    "error_count": module_data.get("error_count", 0),
                    "error_rate": (
                        module_data.get("error_count", 0) / 
                        max(1, module_data.get("trace_count", 1))
                    )
                }
            else:
                return {
                    "tracer_id": self.tracer_id,
                    "global_metrics": self.metrics.copy(),
                    "buffer_utilization": len(self.trace_buffer) / self.buffer_size,
                    "active_traces": len(self.active_traces),
                    "modules_tracked": len(self.module_stats)
                }
    
    def get_recent_traces(self, limit: int = 100, module: Optional[str] = None) -> List[TraceEvent]:
        """Get recent trace events"""
        with self._lock:
            traces = list(self.trace_buffer)
            
        if module:
            traces = [t for t in traces if t.module == module]
            
        return traces[-limit:]
    
    def get_trace_metrics(self) -> Dict[str, Any]:
        """Get comprehensive trace metrics"""
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "tracer_id": self.tracer_id,
                "global_metrics": self.metrics.copy(),
                "module_statistics": {
                    module: stats.copy() 
                    for module, stats in self.module_stats.items()
                },
                "buffer_status": {
                    "size": len(self.trace_buffer),
                    "capacity": self.buffer_size,
                    "utilization": len(self.trace_buffer) / self.buffer_size
                },
                "active_traces": len(self.active_traces)
            }

# Global tracer instance
_global_tracer = None
_tracer_lock = threading.Lock()

def get_global_tracer() -> DAWNTracer:
    """Get the global tracer instance"""
    global _global_tracer
    
    with _tracer_lock:
        if _global_tracer is None:
            _global_tracer = DAWNTracer()
        return _global_tracer

def trace_operation(module: str, operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator for tracing function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            with tracer.trace(module, operation, metadata) as t:
                t.log_metadata("function", func.__name__)
                t.log_metadata("args_count", len(args))
                t.log_metadata("kwargs_count", len(kwargs))
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Convenience function for quick tracing
def trace_quick(module: str, operation: str, func: callable, *args, **kwargs):
    """Quick trace a function call"""
    tracer = get_global_tracer()
    with tracer.trace(module, operation) as t:
        result = func(*args, **kwargs)
        t.log_metadata("result_type", type(result).__name__)
        return result

if __name__ == "__main__":
    # Demo the tracer system
    logging.basicConfig(level=logging.INFO)
    
    tracer = DAWNTracer()
    print(f"🔗 Testing DAWNTracer: {tracer.tracer_id}")
    
    # Test basic tracing
    with tracer.trace("test_module", "test_operation") as t:
        time.sleep(0.1)
        t.log_metric("test_metric", 42)
        t.log_metadata("test_data", "example")
    
    # Test error tracing
    try:
        with tracer.trace("test_module", "error_operation") as t:
            raise ValueError("Test error")
    except ValueError:
        pass
    
    # Show results
    summary = tracer.get_trace_summary()
    print(f"✅ Trace summary: {summary}")
    
    module_summary = tracer.get_trace_summary("test_module")
    print(f"📊 Module summary: {module_summary}")
    
    print("🔗 DAWNTracer test complete!")

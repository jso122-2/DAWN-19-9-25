#!/usr/bin/env python3
"""
DAWN State Publisher
===================

Publishes live DAWN state to shared memory/files for monitoring tools.
This module should be integrated into DAWN engines to broadcast their state.
"""

import time
import threading
from pathlib import Path
from typing import Optional

try:
    from .shared_state_reader import SharedStateManager, SharedTickState
except ImportError:
    # Fallback for standalone usage
    import sys
    sys.path.append(str(Path(__file__).parent))
    from shared_state_reader import SharedStateManager, SharedTickState

class StatePublisher:
    """Publishes DAWN engine state to shared storage"""
    
    def __init__(self, publish_interval: float = 0.5):
        self.state_manager = SharedStateManager()
        self.publish_interval = publish_interval
        self.running = False
        self.thread = None
        
        # Cached state for efficient publishing
        self._last_published = 0
        self._current_state = None
        
    def start(self):
        """Start publishing state in background thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop publishing state"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def update_state(self, 
                    tick_count: int,
                    current_phase: str = "UNKNOWN",
                    phase_duration: float = 0.0,
                    cycle_time: float = 0.0,
                    consciousness_level: float = 0.5,
                    unity_score: float = 0.5,
                    awareness_delta: float = 0.5,
                    processing_load: float = 0.0,
                    active_modules: int = 0,
                    error_count: int = 0,
                    scup_value: float = 0.0,
                    heat_level: float = 0.0,
                    engine_status: str = "RUNNING"):
        """Update current state (called by DAWN engine)"""
        
        self._current_state = SharedTickState(
            timestamp=time.time(),
            tick_count=tick_count,
            current_phase=current_phase,
            phase_duration=phase_duration,
            cycle_time=cycle_time,
            consciousness_level=consciousness_level,
            unity_score=unity_score,
            awareness_delta=awareness_delta,
            processing_load=processing_load,
            active_modules=active_modules,
            error_count=error_count,
            scup_value=scup_value,
            heat_level=heat_level,
            engine_status=engine_status
        )
        
    def _publish_loop(self):
        """Background publishing loop"""
        while self.running:
            try:
                if self._current_state is not None:
                    # Only publish if enough time has passed
                    now = time.time()
                    if now - self._last_published >= self.publish_interval:
                        self.state_manager.write_state(self._current_state)
                        self._last_published = now
                        
                time.sleep(0.1)  # Small sleep to prevent CPU spinning
                
            except Exception as e:
                # Silently continue on publish errors
                time.sleep(1.0)
                
    def cleanup(self):
        """Clean up shared state files"""
        self.stop()
        self.state_manager.cleanup()

# Global publisher instance for easy integration
_global_publisher: Optional[StatePublisher] = None
_publisher_lock = threading.Lock()

def get_state_publisher() -> StatePublisher:
    """Get or create global state publisher"""
    global _global_publisher
    
    with _publisher_lock:
        if _global_publisher is None:
            _global_publisher = StatePublisher()
            _global_publisher.start()
    
    return _global_publisher

def publish_dawn_state(**kwargs):
    """Convenience function to publish DAWN state"""
    publisher = get_state_publisher()
    publisher.update_state(**kwargs)

def cleanup_publisher():
    """Clean up global publisher"""
    global _global_publisher
    
    with _publisher_lock:
        if _global_publisher is not None:
            _global_publisher.cleanup()
            _global_publisher = None

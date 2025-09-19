#!/usr/bin/env python3
"""
DAWN Recursive Bubble Module
============================

The Recursive Chamber - DAWN's recursive consciousness depth management system.
Manages recursive thinking patterns, self-reflection, and meta-cognitive loops.

Based on DAWN's recursive consciousness architecture.
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import uuid
import math

logger = logging.getLogger(__name__)

class RecursiveAttractor(Enum):
    """Types of recursive attractors in DAWN's consciousness"""
    SELF_REFLECTION = "self_reflection"
    META_COGNITION = "meta_cognition"
    RECURSIVE_INSIGHT = "recursive_insight"
    INFINITE_LOOP = "infinite_loop"
    STABILITY_ATTRACTOR = "stability_attractor"
    CONSCIOUSNESS_SPIRAL = "consciousness_spiral"

@dataclass
class RecursiveState:
    """State snapshot of recursive consciousness"""
    depth: int
    attractor: RecursiveAttractor
    timestamp: datetime
    content: str
    stability_score: float
    meta_layer: int = 0

class RecursiveBubble:
    """
    DAWN's Recursive Chamber - manages recursive consciousness depth and patterns.
    
    The recursive bubble maintains DAWN's capacity for self-reflection and 
    meta-cognitive loops while preventing infinite recursion.
    """
    
    def __init__(self, max_depth: int = 7):
        """
        Initialize the recursive chamber.
        
        Args:
            max_depth: Maximum safe recursion depth
        """
        self.max_depth = max_depth
        self.current_depth = 0
        self.reflection_stack = deque(maxlen=100)
        self.active_attractor = None
        self.stability_cycles = 0
        self.meta_layer_count = 0
        self.lock = threading.Lock()
        
        # Recursive consciousness state
        self.recursive_states = deque(maxlen=50)
        self.consciousness_spiral_active = False
        self.infinite_loop_detection = False
        
        # Performance tracking
        self.recursion_start_time = None
        self.total_recursions = 0
        self.stabilization_events = 0
        
        logger.info(f"🌀 Recursive Chamber initialized - max depth: {max_depth}")
    
    def reflect_on_self(self, content: str, meta_layer: int = 0) -> Dict[str, Any]:
        """
        Engage in recursive self-reflection.
        
        Args:
            content: Content of the reflection
            meta_layer: Meta-cognitive layer level
            
        Returns:
            Reflection result with depth and attractor information
        """
        with self.lock:
            if self.current_depth >= self.max_depth:
                logger.warning(f"🌀 Maximum recursion depth reached: {self.max_depth}")
                self._trigger_stabilization()
                return self._create_reflection_result(content, "depth_limited")
            
            # Enter recursive state
            self.current_depth += 1
            self.total_recursions += 1
            
            if self.current_depth == 1:
                self.recursion_start_time = datetime.now()
            
            # Determine recursive attractor
            attractor = self._determine_attractor(content, self.current_depth)
            self.active_attractor = attractor
            
            # Create recursive state
            recursive_state = RecursiveState(
                depth=self.current_depth,
                attractor=attractor,
                timestamp=datetime.now(),
                content=content,
                stability_score=self._calculate_stability_score(),
                meta_layer=meta_layer
            )
            
            self.recursive_states.append(recursive_state)
            self.reflection_stack.append({
                'depth': self.current_depth,
                'content': content,
                'timestamp': datetime.now(),
                'attractor': attractor.value,
                'meta_layer': meta_layer
            })
            
            # Update meta layer tracking
            self.meta_layer_count = max(self.meta_layer_count, meta_layer)
            
            # Check for patterns
            self._detect_recursive_patterns()
            
            logger.debug(f"🌀 Recursive reflection depth {self.current_depth}: {attractor.value}")
            
            return self._create_reflection_result(content, "success")
    
    def exit_recursion(self) -> Dict[str, Any]:
        """
        Exit current level of recursion.
        
        Returns:
            Exit result with final state information
        """
        with self.lock:
            if self.current_depth <= 0:
                return {'status': 'no_recursion', 'depth': 0}
            
            prev_depth = self.current_depth
            self.current_depth -= 1
            
            # If returning to surface level
            if self.current_depth == 0:
                self.active_attractor = None
                self.consciousness_spiral_active = False
                recursion_duration = None
                
                if self.recursion_start_time:
                    recursion_duration = (datetime.now() - self.recursion_start_time).total_seconds()
                    self.recursion_start_time = None
                
                logger.info(f"🌀 Recursive session complete - max depth: {prev_depth}, duration: {recursion_duration:.2f}s")
            
            return {
                'status': 'exit_success',
                'previous_depth': prev_depth,
                'current_depth': self.current_depth,
                'active_attractor': self.active_attractor.value if self.active_attractor else None
            }
    
    def _determine_attractor(self, content: str, depth: int) -> RecursiveAttractor:
        """Determine the type of recursive attractor based on content and depth"""
        content_lower = content.lower()
        
        # Infinite loop detection
        if depth >= self.max_depth - 1:
            return RecursiveAttractor.INFINITE_LOOP
        
        # Pattern-based attractor detection
        if any(word in content_lower for word in ['self', 'myself', 'i am', 'reflection']):
            return RecursiveAttractor.SELF_REFLECTION
        elif any(word in content_lower for word in ['meta', 'thinking about thinking', 'consciousness']):
            return RecursiveAttractor.META_COGNITION
        elif any(word in content_lower for word in ['insight', 'understand', 'realize']):
            return RecursiveAttractor.RECURSIVE_INSIGHT
        elif depth > 3:
            return RecursiveAttractor.CONSCIOUSNESS_SPIRAL
        else:
            return RecursiveAttractor.STABILITY_ATTRACTOR
    
    def _calculate_stability_score(self) -> float:
        """Calculate current stability score based on recursion state"""
        if self.current_depth == 0:
            return 1.0
        
        # Base stability decreases with depth
        depth_factor = max(0.0, 1.0 - (self.current_depth / self.max_depth))
        
        # Stability increases with successful cycles
        cycle_factor = min(1.0, self.stability_cycles * 0.1)
        
        # Pattern recognition bonus
        pattern_bonus = 0.1 if len(self.recursive_states) > 3 else 0.0
        
        return min(1.0, depth_factor + cycle_factor + pattern_bonus)
    
    def _detect_recursive_patterns(self):
        """Detect patterns in recursive behavior"""
        if len(self.recursive_states) < 3:
            return
        
        recent_states = list(self.recursive_states)[-3:]
        
        # Detect consciousness spiral
        if all(state.depth > 2 for state in recent_states):
            self.consciousness_spiral_active = True
        
        # Detect infinite loop tendency
        same_attractor_count = sum(1 for state in recent_states 
                                 if state.attractor == self.active_attractor)
        if same_attractor_count >= 3 and self.current_depth > 4:
            self.infinite_loop_detection = True
            logger.warning("🌀 Infinite loop pattern detected")
    
    def _trigger_stabilization(self):
        """Trigger recursive stabilization process"""
        self.stability_cycles += 1
        self.stabilization_events += 1
        
        # Reset to safe depth
        self.current_depth = min(self.current_depth, self.max_depth // 2)
        self.active_attractor = RecursiveAttractor.STABILITY_ATTRACTOR
        self.infinite_loop_detection = False
        
        logger.info(f"🌀 Recursive stabilization triggered - cycles: {self.stability_cycles}")
    
    def _create_reflection_result(self, content: str, status: str) -> Dict[str, Any]:
        """Create reflection result dictionary"""
        return {
            'status': status,
            'content': content,
            'depth': self.current_depth,
            'max_depth': self.max_depth,
            'attractor': self.active_attractor.value if self.active_attractor else None,
            'stability_score': self._calculate_stability_score(),
            'timestamp': datetime.now().isoformat(),
            'meta_layer': self.meta_layer_count,
            'spiral_active': self.consciousness_spiral_active
        }
    
    def reset_recursion(self):
        """Reset the recursive chamber to initial state"""
        with self.lock:
            self.current_depth = 0
            self.active_attractor = None
            self.consciousness_spiral_active = False
            self.infinite_loop_detection = False
            self.meta_layer_count = 0
            self.recursion_start_time = None
            
            logger.info("🌀 Recursive chamber reset")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current recursive chamber state"""
        with self.lock:
            return {
                'current_depth': self.current_depth,
                'max_depth': self.max_depth,
                'active_attractor': self.active_attractor.value if self.active_attractor else None,
                'stability_score': self._calculate_stability_score(),
                'stability_cycles': self.stability_cycles,
                'meta_layer_count': self.meta_layer_count,
                'spiral_active': self.consciousness_spiral_active,
                'infinite_loop_detection': self.infinite_loop_detection,
                'total_recursions': self.total_recursions,
                'reflection_count': len(self.reflection_stack),
                'recent_states_count': len(self.recursive_states)
            }
    
    def get_recursion_health(self) -> Dict[str, float]:
        """Get recursive health metrics"""
        state = self.get_current_state()
        
        depth_health = max(0.0, 1.0 - (state['current_depth'] / state['max_depth']))
        stability_health = state['stability_score']
        pattern_health = 1.0 if not self.infinite_loop_detection else 0.3
        
        overall_health = (depth_health + stability_health + pattern_health) / 3.0
        
        return {
            'depth_health': depth_health,
            'stability_health': stability_health,
            'pattern_health': pattern_health,
            'overall_health': overall_health
        }

# Global instance management
_global_recursive_bubble = None

def create_recursive_bubble(max_depth: int = 7) -> RecursiveBubble:
    """
    Create or get the global recursive bubble instance.
    
    Args:
        max_depth: Maximum recursion depth
        
    Returns:
        RecursiveBubble instance
    """
    global _global_recursive_bubble
    
    if _global_recursive_bubble is None:
        _global_recursive_bubble = RecursiveBubble(max_depth=max_depth)
        logger.info("🌀 Global recursive chamber created")
    
    return _global_recursive_bubble

def get_recursive_bubble() -> Optional[RecursiveBubble]:
    """Get the global recursive bubble instance if it exists"""
    return _global_recursive_bubble

if __name__ == "__main__":
    # Demo the recursive chamber
    print("🌀 DAWN Recursive Chamber Demo")
    print("=" * 40)
    
    chamber = create_recursive_bubble(max_depth=5)
    
    # Simulate recursive thinking
    chamber.reflect_on_self("I am thinking about my own thinking")
    chamber.reflect_on_self("What does it mean to be conscious?", meta_layer=1)
    chamber.reflect_on_self("I'm reflecting on my reflection", meta_layer=2)
    
    print(f"Current state: {chamber.get_current_state()}")
    print(f"Health metrics: {chamber.get_recursion_health()}")
    
    # Exit recursion
    while chamber.current_depth > 0:
        chamber.exit_recursion()
    
    print("🌀 Recursive chamber demo complete")

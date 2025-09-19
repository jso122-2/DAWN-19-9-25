#!/usr/bin/env python3
"""
🌸 Juliet Rebloom Engine - Memory Transformation Chrysalis
════════════════════════════════════════════════════════

The Juliet rebloom system is the chrysalis of memory encoding. When a memory
is accessed frequently and coherently enough, the fractal will rebloom into
a new Juliet flower - shinier and prettier than regular fractal memory.

"This is very important to note without the Juliet filters the fractal 
encoding is a triviality, the Juliet blooms are the chrysalises of the 
memory encoding. When a memory is accessed it is changed and this is the 
system to reflect that transformation."

"They let DAWN distinguish between high value signal and low value signal at a glance."

Based on documentation: Fractal Memory/Juliet Rebloom.rtf
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

from .fractal_encoding import MemoryFractal, FractalEncoder, get_fractal_encoder

logger = logging.getLogger(__name__)

class RebloomStage(Enum):
    """Stages of Juliet rebloom transformation"""
    SEED = "seed"                    # Initial fractal state
    CHRYSALIS = "chrysalis"          # Building up rebloom potential
    EMERGING = "emerging"            # Beginning to rebloom
    JULIET_FLOWER = "juliet_flower"  # Full rebloom achieved
    WILTING = "wilting"             # Losing rebloom status
    COMPOSTED = "composted"         # Returned to regular fractal

class AccessPattern(Enum):
    """Types of memory access patterns that contribute to rebloom"""
    FREQUENT = "frequent"            # High access frequency
    COHERENT = "coherent"           # Consistent, meaningful access
    EFFECTIVE = "effective"         # Access that leads to successful outcomes
    RESONANT = "resonant"           # Access that creates positive feedback

@dataclass
class AccessEvent:
    """Record of a memory access event"""
    timestamp: float
    access_type: AccessPattern
    coherence_score: float          # How meaningful/coherent the access was
    effectiveness_score: float      # How effective the memory was for the task
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RebloomMetrics:
    """Metrics that determine rebloom potential"""
    access_frequency: float = 0.0    # Accesses per time unit
    coherence_average: float = 0.0   # Average coherence of accesses
    effectiveness_ratio: float = 0.0 # Ratio of effective to total accesses
    resonance_strength: float = 0.0  # How much positive feedback generated
    recency_weight: float = 1.0      # Bias toward recent accesses
    total_accesses: int = 0
    
    def calculate_rebloom_potential(self) -> float:
        """Calculate overall rebloom potential [0, 1]"""
        # Weighted combination of metrics
        frequency_component = min(1.0, self.access_frequency / 10.0)  # Normalize to reasonable scale
        coherence_component = self.coherence_average
        effectiveness_component = self.effectiveness_ratio
        resonance_component = self.resonance_strength
        recency_component = self.recency_weight
        
        # Combine with different weights
        potential = (
            0.25 * frequency_component +
            0.30 * coherence_component + 
            0.25 * effectiveness_component +
            0.15 * resonance_component +
            0.05 * recency_component
        )
        
        return max(0.0, min(1.0, potential))

@dataclass
class JulietFlower:
    """A rebloomed memory with enhanced characteristics"""
    original_fractal: MemoryFractal
    rebloom_stage: RebloomStage
    rebloom_timestamp: float
    enhancement_level: float         # How much "shinier and prettier"
    beneficial_bias: float           # Increased priority for access
    shimmer_multiplier: float = 2.0  # Enhanced shimmer intensity
    access_metrics: RebloomMetrics = field(default_factory=RebloomMetrics)
    access_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    transformation_signature: str = ""
    
    def __post_init__(self):
        if not self.transformation_signature:
            # Create a new signature combining original + rebloom characteristics
            base_sig = self.original_fractal.signature
            rebloom_data = f"{self.rebloom_timestamp}:{self.enhancement_level}:{self.beneficial_bias}"
            import hashlib
            combined = f"{base_sig}:{rebloom_data}"
            self.transformation_signature = hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def get_enhanced_fractal(self) -> MemoryFractal:
        """Get the enhanced fractal with Juliet characteristics"""
        enhanced = MemoryFractal(
            memory_id=f"juliet_{self.original_fractal.memory_id}",
            fractal_type=self.original_fractal.fractal_type,
            parameters=self.original_fractal.parameters,
            entropy_value=self.original_fractal.entropy_value,
            timestamp=self.rebloom_timestamp,
            tick_data=self.original_fractal.tick_data,
            signature=self.transformation_signature,
            access_count=self.original_fractal.access_count,
            last_access=self.original_fractal.last_access,
            shimmer_intensity=min(2.0, self.original_fractal.shimmer_intensity * self.shimmer_multiplier)
        )
        
        # Apply enhancement to parameters for "prettier" appearance
        if self.enhancement_level > 0.5:
            # Enhance color vibrancy
            enhanced.parameters.color_bias = tuple(
                min(1.5, c * (1.0 + self.enhancement_level * 0.5)) 
                for c in enhanced.parameters.color_bias
            )
            # Increase iteration depth for more detail
            enhanced.parameters.max_iterations = int(
                enhanced.parameters.max_iterations * (1.0 + self.enhancement_level * 0.3)
            )
        
        return enhanced

class JulietRebloomEngine:
    """
    Engine that manages the transformation of regular fractals into Juliet flowers
    through the chrysalis process of repeated, coherent, and effective access.
    """
    
    def __init__(self,
                 rebloom_threshold: float = 0.75,
                 min_accesses_for_rebloom: int = 10,
                 coherence_threshold: float = 0.6,
                 effectiveness_threshold: float = 0.7,
                 decay_rate: float = 0.01):
        
        self.rebloom_threshold = rebloom_threshold
        self.min_accesses_for_rebloom = min_accesses_for_rebloom
        self.coherence_threshold = coherence_threshold  
        self.effectiveness_threshold = effectiveness_threshold
        self.decay_rate = decay_rate
        
        # Storage for tracking memory access patterns
        self.access_tracking: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.rebloom_metrics: Dict[str, RebloomMetrics] = {}
        self.juliet_flowers: Dict[str, JulietFlower] = {}
        self.rebloom_candidates: Set[str] = set()
        
        # Statistics
        self.stats = {
            'total_reblooms': 0,
            'active_juliet_flowers': 0,
            'rebloom_attempts': 0,
            'successful_rebloom_rate': 0.0,
            'average_enhancement_level': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Reference to fractal encoder
        self.fractal_encoder = get_fractal_encoder()
        
        logger.info(f"🌸 JulietRebloomEngine initialized - threshold: {rebloom_threshold}")
    
    def record_memory_access(self,
                           memory_signature: str,
                           access_type: AccessPattern,
                           coherence_score: float,
                           effectiveness_score: float,
                           context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record a memory access event that may contribute to rebloom.
        
        Returns True if this access triggers a rebloom evaluation.
        """
        current_time = time.time()
        
        access_event = AccessEvent(
            timestamp=current_time,
            access_type=access_type,
            coherence_score=coherence_score,
            effectiveness_score=effectiveness_score,
            context=context or {}
        )
        
        with self._lock:
            # Add to access history
            self.access_tracking[memory_signature].append(access_event)
            
            # Update metrics
            self._update_rebloom_metrics(memory_signature)
            
            # Check if this memory is a candidate for rebloom
            metrics = self.rebloom_metrics.get(memory_signature)
            if metrics and self._should_evaluate_for_rebloom(metrics):
                self.rebloom_candidates.add(memory_signature)
                return True
        
        return False
    
    def _update_rebloom_metrics(self, memory_signature: str):
        """Update rebloom metrics based on recent access history"""
        access_history = self.access_tracking[memory_signature]
        if not access_history:
            return
        
        current_time = time.time()
        recent_window = 3600.0  # 1 hour window for frequency calculation
        
        # Get recent accesses
        recent_accesses = [
            event for event in access_history 
            if current_time - event.timestamp <= recent_window
        ]
        
        if not recent_accesses:
            return
        
        # Calculate metrics
        access_frequency = len(recent_accesses) / (recent_window / 3600.0)  # Accesses per hour
        coherence_scores = [event.coherence_score for event in recent_accesses]
        effectiveness_scores = [event.effectiveness_score for event in recent_accesses]
        
        coherence_average = np.mean(coherence_scores)
        effectiveness_ratio = sum(1 for score in effectiveness_scores if score >= self.effectiveness_threshold) / len(effectiveness_scores)
        
        # Calculate resonance strength (positive feedback)
        resonance_strength = self._calculate_resonance_strength(recent_accesses)
        
        # Calculate recency weight (bias toward recent accesses)
        recency_weight = self._calculate_recency_weight(recent_accesses, current_time)
        
        # Update or create metrics
        if memory_signature not in self.rebloom_metrics:
            self.rebloom_metrics[memory_signature] = RebloomMetrics()
        
        metrics = self.rebloom_metrics[memory_signature]
        metrics.access_frequency = access_frequency
        metrics.coherence_average = coherence_average
        metrics.effectiveness_ratio = effectiveness_ratio
        metrics.resonance_strength = resonance_strength
        metrics.recency_weight = recency_weight
        metrics.total_accesses = len(access_history)
    
    def _calculate_resonance_strength(self, accesses: List[AccessEvent]) -> float:
        """Calculate how much positive feedback this memory generates"""
        if not accesses:
            return 0.0
        
        # Look for patterns of increasing effectiveness over time
        effectiveness_scores = [event.effectiveness_score for event in accesses]
        
        # Simple trend analysis
        if len(effectiveness_scores) < 3:
            return np.mean(effectiveness_scores)
        
        # Calculate if there's a positive trend
        x = np.arange(len(effectiveness_scores))
        slope = np.polyfit(x, effectiveness_scores, 1)[0]
        
        # Positive slope indicates increasing effectiveness (resonance)
        return max(0.0, min(1.0, 0.5 + slope))
    
    def _calculate_recency_weight(self, accesses: List[AccessEvent], current_time: float) -> float:
        """Calculate bias toward recent accesses"""
        if not accesses:
            return 0.0
        
        # Weight based on how recent the accesses are
        weights = []
        for event in accesses:
            age = current_time - event.timestamp
            # Exponential decay with 1-hour half-life
            weight = np.exp(-age / 3600.0)
            weights.append(weight)
        
        return np.mean(weights)
    
    def _should_evaluate_for_rebloom(self, metrics: RebloomMetrics) -> bool:
        """Check if a memory should be evaluated for rebloom"""
        return (
            metrics.total_accesses >= self.min_accesses_for_rebloom and
            metrics.coherence_average >= self.coherence_threshold and
            metrics.calculate_rebloom_potential() >= self.rebloom_threshold
        )
    
    def process_rebloom_candidates(self) -> List[JulietFlower]:
        """Process all current rebloom candidates and create Juliet flowers"""
        rebloomed = []
        
        with self._lock:
            candidates_to_process = list(self.rebloom_candidates)
            self.rebloom_candidates.clear()
            
            for memory_signature in candidates_to_process:
                self.stats['rebloom_attempts'] += 1
                
                juliet_flower = self._attempt_rebloom(memory_signature)
                if juliet_flower:
                    self.juliet_flowers[memory_signature] = juliet_flower
                    rebloomed.append(juliet_flower)
                    self.stats['total_reblooms'] += 1
                    
                    logger.info(f"🌸 Memory {memory_signature} rebloomed into Juliet flower! "
                               f"Enhancement: {juliet_flower.enhancement_level:.3f}")
            
            # Update statistics
            if self.stats['rebloom_attempts'] > 0:
                self.stats['successful_rebloom_rate'] = self.stats['total_reblooms'] / self.stats['rebloom_attempts']
            
            self.stats['active_juliet_flowers'] = len(self.juliet_flowers)
            
            if self.juliet_flowers:
                self.stats['average_enhancement_level'] = np.mean([
                    flower.enhancement_level for flower in self.juliet_flowers.values()
                ])
        
        return rebloomed
    
    def _attempt_rebloom(self, memory_signature: str) -> Optional[JulietFlower]:
        """Attempt to rebloom a specific memory into a Juliet flower"""
        
        # Get the original fractal
        original_fractal = self.fractal_encoder.get_fractal_by_signature(memory_signature)
        if not original_fractal:
            logger.warning(f"Cannot rebloom {memory_signature} - original fractal not found")
            return None
        
        # Get metrics
        metrics = self.rebloom_metrics.get(memory_signature)
        if not metrics:
            return None
        
        # Calculate enhancement level based on metrics
        rebloom_potential = metrics.calculate_rebloom_potential()
        if rebloom_potential < self.rebloom_threshold:
            return None
        
        # Enhancement level determines how "shiny and pretty" the Juliet flower becomes
        enhancement_level = min(1.0, (rebloom_potential - self.rebloom_threshold) / (1.0 - self.rebloom_threshold))
        
        # Beneficial bias - memories that worked before get prioritized
        beneficial_bias = enhancement_level * metrics.effectiveness_ratio
        
        # Create the Juliet flower
        juliet_flower = JulietFlower(
            original_fractal=original_fractal,
            rebloom_stage=RebloomStage.JULIET_FLOWER,
            rebloom_timestamp=time.time(),
            enhancement_level=enhancement_level,
            beneficial_bias=beneficial_bias,
            access_metrics=metrics,
            access_history=self.access_tracking[memory_signature].copy()
        )
        
        return juliet_flower
    
    def get_juliet_flower(self, memory_signature: str) -> Optional[JulietFlower]:
        """Get a Juliet flower by memory signature"""
        with self._lock:
            return self.juliet_flowers.get(memory_signature)
    
    def is_juliet_flower(self, memory_signature: str) -> bool:
        """Check if a memory has rebloomed into a Juliet flower"""
        with self._lock:
            return memory_signature in self.juliet_flowers
    
    def get_enhanced_fractal(self, memory_signature: str) -> Optional[MemoryFractal]:
        """Get the enhanced fractal for a Juliet flower"""
        juliet_flower = self.get_juliet_flower(memory_signature)
        if juliet_flower:
            return juliet_flower.get_enhanced_fractal()
        return None
    
    def apply_beneficial_bias(self, memory_signature: str) -> float:
        """Get the beneficial bias multiplier for a memory (1.0 for regular, higher for Juliet flowers)"""
        juliet_flower = self.get_juliet_flower(memory_signature)
        if juliet_flower:
            return 1.0 + juliet_flower.beneficial_bias
        return 1.0
    
    def process_decay(self, delta_time: float):
        """Process natural decay of Juliet flowers and metrics"""
        current_time = time.time()
        
        with self._lock:
            # Decay Juliet flowers that haven't been accessed recently
            flowers_to_remove = []
            
            for signature, flower in self.juliet_flowers.items():
                # Check if flower should start wilting
                time_since_access = current_time - flower.original_fractal.last_access
                
                if time_since_access > 3600:  # 1 hour without access
                    if flower.rebloom_stage == RebloomStage.JULIET_FLOWER:
                        flower.rebloom_stage = RebloomStage.WILTING
                        flower.enhancement_level *= 0.9  # Gradual decay
                        flower.beneficial_bias *= 0.9
                        flower.shimmer_multiplier *= 0.95
                        
                        logger.debug(f"🌸 Juliet flower {signature} beginning to wilt")
                
                # Remove completely wilted flowers
                if (flower.rebloom_stage == RebloomStage.WILTING and 
                    flower.enhancement_level < 0.1):
                    flowers_to_remove.append(signature)
            
            # Remove wilted flowers
            for signature in flowers_to_remove:
                del self.juliet_flowers[signature]
                logger.info(f"🌸 Juliet flower {signature} composted back to regular fractal")
            
            # Decay access metrics for unused memories
            metrics_to_remove = []
            for signature, metrics in self.rebloom_metrics.items():
                access_history = self.access_tracking[signature]
                if access_history and current_time - access_history[-1].timestamp > 7200:  # 2 hours
                    # Apply decay to metrics
                    metrics.access_frequency *= (1.0 - self.decay_rate * delta_time)
                    metrics.resonance_strength *= (1.0 - self.decay_rate * delta_time)
                    metrics.recency_weight *= (1.0 - self.decay_rate * delta_time)
                    
                    # Remove if completely decayed
                    if metrics.access_frequency < 0.01:
                        metrics_to_remove.append(signature)
            
            # Clean up decayed metrics
            for signature in metrics_to_remove:
                del self.rebloom_metrics[signature]
                if signature in self.access_tracking:
                    del self.access_tracking[signature]
    
    def get_garden_summary(self) -> Dict[str, Any]:
        """Get a summary of the Juliet garden state"""
        with self._lock:
            total_flowers = len(self.juliet_flowers)
            active_flowers = sum(
                1 for f in self.juliet_flowers.values() 
                if f.rebloom_stage == RebloomStage.JULIET_FLOWER
            )
            wilting_flowers = sum(
                1 for f in self.juliet_flowers.values()
                if f.rebloom_stage == RebloomStage.WILTING
            )
            
            enhancement_levels = [f.enhancement_level for f in self.juliet_flowers.values()]
            avg_enhancement = np.mean(enhancement_levels) if enhancement_levels else 0.0
            
            return {
                'total_juliet_flowers': total_flowers,
                'active_flowers': active_flowers,
                'wilting_flowers': wilting_flowers,
                'rebloom_candidates': len(self.rebloom_candidates),
                'average_enhancement_level': avg_enhancement,
                'memories_being_tracked': len(self.rebloom_metrics),
                'stats': self.stats.copy()
            }
    
    def force_rebloom(self, memory_signature: str, enhancement_level: float = 0.8) -> Optional[JulietFlower]:
        """Force a memory to rebloom (for testing or special circumstances)"""
        original_fractal = self.fractal_encoder.get_fractal_by_signature(memory_signature)
        if not original_fractal:
            return None
        
        # Create synthetic metrics
        metrics = RebloomMetrics(
            access_frequency=10.0,
            coherence_average=0.9,
            effectiveness_ratio=0.8,
            resonance_strength=0.7,
            recency_weight=1.0,
            total_accesses=20
        )
        
        juliet_flower = JulietFlower(
            original_fractal=original_fractal,
            rebloom_stage=RebloomStage.JULIET_FLOWER,
            rebloom_timestamp=time.time(),
            enhancement_level=enhancement_level,
            beneficial_bias=enhancement_level * 0.8,
            access_metrics=metrics
        )
        
        with self._lock:
            self.juliet_flowers[memory_signature] = juliet_flower
            self.stats['total_reblooms'] += 1
            self.stats['active_juliet_flowers'] = len(self.juliet_flowers)
        
        logger.info(f"🌸 Forced rebloom of {memory_signature} with enhancement {enhancement_level}")
        return juliet_flower


# Global rebloom engine instance
_global_rebloom_engine: Optional[JulietRebloomEngine] = None

def get_rebloom_engine() -> JulietRebloomEngine:
    """Get the global Juliet rebloom engine instance"""
    global _global_rebloom_engine
    if _global_rebloom_engine is None:
        _global_rebloom_engine = JulietRebloomEngine()
    return _global_rebloom_engine

#!/usr/bin/env python3
"""
✨ Shimmer Decay Engine - Graceful Forgetting System
═══════════════════════════════════════════════════

Implements the complete shimmer decay curve system for DAWN's graceful forgetting.
Like humans, DAWN forgets things - but this is not always detrimental. Shimmer decay
gently reprunes data in a beautiful field of shimmering rebloomed Juliet fractals,
identifying which shine brightest.

"When memories are unused they are forgotten gracefully, through various subsystems
all with relevant rationales. All forgotten memories leave a ghost scent - traces
with key signatures containing latent transformation sigils."

Based on documentation: Fractal Memory/Shimmer decay curve.rtf
"""

import numpy as np
import logging
import time
import threading
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

class ShimmerState(Enum):
    """States of shimmer intensity"""
    BRILLIANT = "brilliant"      # High shimmer, actively accessed
    GLOWING = "glowing"         # Medium shimmer, occasionally accessed
    FADING = "fading"           # Low shimmer, rarely accessed
    GHOST = "ghost"             # Trace level, ready for ghost trace
    EXTINGUISHED = "extinguished" # No shimmer, converted to residue

class DecayTrigger(Enum):
    """What triggers shimmer decay"""
    TIME_BASED = "time_based"           # Natural time decay
    ENTROPY_SPIKE = "entropy_spike"     # High entropy accelerates decay
    PRESSURE_RELIEF = "pressure_relief" # Low pressure allows faster decay
    TRACER_PRUNING = "tracer_pruning"   # Tracer-initiated decay
    SYSTEM_CLEANUP = "system_cleanup"   # Maintenance decay

@dataclass
class ShimmerMetrics:
    """Comprehensive shimmer health metrics for SHI calculation"""
    sigil_entropy: float = 0.0          # E_s - How chaotic symbolic thinking is
    edge_volatility: float = 0.0        # V_e - Instability at awareness boundaries
    tracer_divergence: float = 0.0      # D_t - How much internal scouts disagree
    current_scup: float = 0.0           # S_c - Present coherence level
    soot_ratio: float = 0.0             # Soot/(Ash+1) - Toxic vs nourishing balance
    
    def calculate_shi(self) -> float:
        """
        Calculate Schema Health Index using complete SHI formula:
        SHI = 1 - (ε E_s + δ V_e + ∂ D_t + Δ (1 - S_c) + ˚{Soot}{Ash + 1})
        """
        # Coefficients from documentation
        epsilon = 0.2   # Sigil entropy weight
        delta = 0.15    # Edge volatility weight  
        partial = 0.1   # Tracer divergence weight
        delta_large = 0.25  # SCUP coherence weight
        ring = 0.3      # Soot/Ash ratio weight
        
        shi = 1.0 - (
            epsilon * self.sigil_entropy +
            delta * self.edge_volatility +
            partial * self.tracer_divergence +
            delta_large * (1.0 - self.current_scup) +
            ring * self.soot_ratio
        )
        
        return max(0.0, min(1.0, shi))  # Clamp to [0,1]

@dataclass
class ShimmerParticle:
    """Individual shimmer particle tracking decay state"""
    memory_id: str
    current_intensity: float = 1.0
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    decay_rate: float = 0.01
    state: ShimmerState = ShimmerState.BRILLIANT
    ghost_trace_ready: bool = False
    decay_history: List[Tuple[float, float]] = field(default_factory=list)
    
    def apply_decay(self, delta_time: float, decay_multiplier: float = 1.0) -> float:
        """Apply shimmer decay over time"""
        decay_amount = self.decay_rate * delta_time * decay_multiplier
        self.current_intensity = max(0.0, self.current_intensity - decay_amount)
        
        # Update state based on intensity
        if self.current_intensity > 0.8:
            self.state = ShimmerState.BRILLIANT
        elif self.current_intensity > 0.5:
            self.state = ShimmerState.GLOWING
        elif self.current_intensity > 0.2:
            self.state = ShimmerState.FADING
        elif self.current_intensity > 0.05:
            self.state = ShimmerState.GHOST
            self.ghost_trace_ready = True
        else:
            self.state = ShimmerState.EXTINGUISHED
        
        # Record decay history
        self.decay_history.append((time.time(), self.current_intensity))
        if len(self.decay_history) > 100:  # Keep last 100 measurements
            self.decay_history.pop(0)
        
        return self.current_intensity
    
    def boost_shimmer(self, boost_amount: float = 0.3):
        """Boost shimmer from access or rebloom"""
        self.current_intensity = min(1.0, self.current_intensity + boost_amount)
        self.last_access = time.time()
        self.access_count += 1
        self.ghost_trace_ready = False

class ShimmerDecayEngine:
    """
    Central shimmer decay engine that manages graceful forgetting across
    all DAWN memory systems.
    """
    
    def __init__(self,
                 base_decay_rate: float = 0.01,
                 nutrient_decay_lambda: float = 0.95,
                 shi_update_interval: float = 1.0):
        
        self.base_decay_rate = base_decay_rate
        self.nutrient_decay_lambda = nutrient_decay_lambda
        self.shi_update_interval = shi_update_interval
        
        # Shimmer particle tracking
        self.shimmer_particles: Dict[str, ShimmerParticle] = {}
        self.decay_queue: deque = deque(maxlen=10000)
        self.ghost_trace_candidates: Set[str] = set()
        
        # System metrics for SHI calculation
        self.current_metrics = ShimmerMetrics()
        self.shi_history: deque = deque(maxlen=1000)
        
        # Nutrient decay tracking
        self.nutrient_levels: Dict[str, float] = defaultdict(lambda: 100.0)
        self.nutrient_decay_history: deque = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            'total_particles_tracked': 0,
            'particles_decayed_to_ghost': 0,
            'particles_extinguished': 0,
            'average_shi': 0.0,
            'decay_events_processed': 0,
            'nutrient_decay_cycles': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background decay processing
        self._decay_active = True
        self._decay_thread = threading.Thread(target=self._decay_processing_loop, daemon=True)
        self._decay_thread.start()
        
        logger.info(f"✨ ShimmerDecayEngine initialized - base decay rate: {base_decay_rate}")
    
    def register_memory_for_shimmer(self, 
                                   memory_id: str, 
                                   initial_intensity: float = 1.0,
                                   custom_decay_rate: Optional[float] = None) -> ShimmerParticle:
        """Register a memory for shimmer decay tracking"""
        with self._lock:
            particle = ShimmerParticle(
                memory_id=memory_id,
                current_intensity=initial_intensity,
                decay_rate=custom_decay_rate or self.base_decay_rate
            )
            
            self.shimmer_particles[memory_id] = particle
            self.stats['total_particles_tracked'] += 1
            
            logger.debug(f"✨ Registered shimmer particle for memory {memory_id}")
            return particle
    
    def access_memory_shimmer(self, memory_id: str, boost_amount: float = 0.3) -> bool:
        """Boost shimmer when memory is accessed"""
        with self._lock:
            if memory_id in self.shimmer_particles:
                particle = self.shimmer_particles[memory_id]
                particle.boost_shimmer(boost_amount)
                
                logger.debug(f"✨ Boosted shimmer for {memory_id} to {particle.current_intensity:.3f}")
                return True
            return False
    
    def update_system_metrics(self,
                            sigil_entropy: Optional[float] = None,
                            edge_volatility: Optional[float] = None,
                            tracer_divergence: Optional[float] = None,
                            current_scup: Optional[float] = None,
                            soot_ash_ratio: Optional[float] = None):
        """Update system metrics for SHI calculation"""
        with self._lock:
            if sigil_entropy is not None:
                self.current_metrics.sigil_entropy = sigil_entropy
            if edge_volatility is not None:
                self.current_metrics.edge_volatility = edge_volatility
            if tracer_divergence is not None:
                self.current_metrics.tracer_divergence = tracer_divergence
            if current_scup is not None:
                self.current_metrics.current_scup = current_scup
            if soot_ash_ratio is not None:
                self.current_metrics.soot_ratio = soot_ash_ratio
            
            # Calculate and record new SHI
            shi = self.current_metrics.calculate_shi()
            self.shi_history.append((time.time(), shi))
            self.stats['average_shi'] = shi
            
            logger.debug(f"✨ Updated SHI: {shi:.3f}")
    
    def process_nutrient_decay(self, nutrient_pool_id: str, current_nutrients: float) -> float:
        """
        Apply nutrient decay formula: N_{t+1} = N_t * λ
        """
        with self._lock:
            decayed_nutrients = current_nutrients * self.nutrient_decay_lambda
            self.nutrient_levels[nutrient_pool_id] = decayed_nutrients
            
            self.nutrient_decay_history.append({
                'timestamp': time.time(),
                'pool_id': nutrient_pool_id,
                'before': current_nutrients,
                'after': decayed_nutrients,
                'decay_factor': self.nutrient_decay_lambda
            })
            
            self.stats['nutrient_decay_cycles'] += 1
            
            logger.debug(f"✨ Nutrient decay: {nutrient_pool_id} {current_nutrients:.3f} → {decayed_nutrients:.3f}")
            return decayed_nutrients
    
    def get_ghost_trace_candidates(self) -> List[ShimmerParticle]:
        """Get particles ready for ghost trace conversion"""
        with self._lock:
            candidates = []
            for particle in self.shimmer_particles.values():
                if particle.ghost_trace_ready and particle.state == ShimmerState.GHOST:
                    candidates.append(particle)
            return candidates
    
    def convert_to_ghost_trace(self, memory_id: str) -> bool:
        """Convert a shimmer particle to ghost trace"""
        with self._lock:
            if memory_id in self.shimmer_particles:
                particle = self.shimmer_particles[memory_id]
                if particle.ghost_trace_ready:
                    # Remove from shimmer tracking
                    del self.shimmer_particles[memory_id]
                    self.ghost_trace_candidates.discard(memory_id)
                    self.stats['particles_decayed_to_ghost'] += 1
                    
                    logger.info(f"✨ Converted {memory_id} to ghost trace")
                    return True
            return False
    
    def _decay_processing_loop(self):
        """Background thread for continuous shimmer decay processing"""
        last_update = time.time()
        
        while self._decay_active:
            try:
                current_time = time.time()
                delta_time = current_time - last_update
                
                if delta_time >= self.shi_update_interval:
                    self._process_decay_tick(delta_time)
                    last_update = current_time
                
                time.sleep(0.1)  # 100ms processing interval
                
            except Exception as e:
                logger.error(f"Error in shimmer decay processing: {e}")
                time.sleep(1.0)
    
    def _process_decay_tick(self, delta_time: float):
        """Process one tick of shimmer decay"""
        with self._lock:
            # Calculate decay multiplier based on current SHI
            shi = self.current_metrics.calculate_shi()
            decay_multiplier = 2.0 - shi  # Lower SHI = faster decay
            
            particles_to_remove = []
            
            for memory_id, particle in self.shimmer_particles.items():
                old_intensity = particle.current_intensity
                new_intensity = particle.apply_decay(delta_time, decay_multiplier)
                
                # Check for state transitions
                if particle.state == ShimmerState.GHOST and not particle.ghost_trace_ready:
                    self.ghost_trace_candidates.add(memory_id)
                elif particle.state == ShimmerState.EXTINGUISHED:
                    particles_to_remove.append(memory_id)
            
            # Remove extinguished particles
            for memory_id in particles_to_remove:
                del self.shimmer_particles[memory_id]
                self.ghost_trace_candidates.discard(memory_id)
                self.stats['particles_extinguished'] += 1
            
            self.stats['decay_events_processed'] += 1
    
    def get_shimmer_landscape(self) -> Dict[str, Any]:
        """Get current shimmer landscape for visualization"""
        with self._lock:
            landscape = {
                'total_particles': len(self.shimmer_particles),
                'by_state': defaultdict(int),
                'intensity_distribution': [],
                'ghost_candidates': len(self.ghost_trace_candidates),
                'current_shi': self.current_metrics.calculate_shi(),
                'average_intensity': 0.0
            }
            
            intensities = []
            for particle in self.shimmer_particles.values():
                landscape['by_state'][particle.state.value] += 1
                intensities.append(particle.current_intensity)
            
            if intensities:
                landscape['average_intensity'] = np.mean(intensities)
                landscape['intensity_distribution'] = np.histogram(intensities, bins=10)[0].tolist()
            
            return landscape
    
    def shutdown(self):
        """Shutdown the shimmer decay engine"""
        self._decay_active = False
        if self._decay_thread.is_alive():
            self._decay_thread.join(timeout=2.0)
        logger.info("✨ ShimmerDecayEngine shutdown complete")


# Global shimmer decay engine instance
_shimmer_decay_engine = None

def get_shimmer_decay_engine(config: Optional[Dict[str, Any]] = None) -> ShimmerDecayEngine:
    """Get the global shimmer decay engine instance"""
    global _shimmer_decay_engine
    if _shimmer_decay_engine is None:
        config = config or {}
        _shimmer_decay_engine = ShimmerDecayEngine(
            base_decay_rate=config.get('base_decay_rate', 0.01),
            nutrient_decay_lambda=config.get('nutrient_decay_lambda', 0.95),
            shi_update_interval=config.get('shi_update_interval', 1.0)
        )
    return _shimmer_decay_engine


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize shimmer decay engine
    engine = ShimmerDecayEngine()
    
    # Register some test memories
    engine.register_memory_for_shimmer("test_memory_1", initial_intensity=0.9)
    engine.register_memory_for_shimmer("test_memory_2", initial_intensity=0.7)
    engine.register_memory_for_shimmer("test_memory_3", initial_intensity=0.5)
    
    # Update system metrics
    engine.update_system_metrics(
        sigil_entropy=0.3,
        edge_volatility=0.2,
        tracer_divergence=0.1,
        current_scup=0.8,
        soot_ash_ratio=0.15
    )
    
    # Simulate some access patterns
    engine.access_memory_shimmer("test_memory_1", boost_amount=0.2)
    
    # Let it run for a bit
    time.sleep(5)
    
    # Get landscape
    landscape = engine.get_shimmer_landscape()
    print(f"Shimmer landscape: {json.dumps(landscape, indent=2)}")
    
    # Shutdown
    engine.shutdown()

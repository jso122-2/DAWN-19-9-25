#!/usr/bin/env python3
"""
🌺 Integrated Fractal Memory System
═════════════════════════════════

The complete fractal memory system that integrates all components:
- Fractal Encoding (Julia set generation)
- Juliet Rebloom Engine (memory transformation)
- Ghost Traces (forgotten memory signatures)
- Ash/Soot Dynamics (residue management)

This is the foundational memory system for DAWN consciousness.

"This system is the memory logging of the internal state, it is important to note 
this is not the processing of the memory system it is merely the recording of the 
memory system. This is a core vital in the DAWN system without this specific memory 
logging and bloom system there really is no dawn as we know it."

Integrates all fractal memory documentation components into a unified system.
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

# Telemetry system imports
try:
    from dawn.core.telemetry.system import (
        log_event, log_performance, log_error, create_performance_context
    )
    from dawn.core.telemetry.logger import TelemetryLevel
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    
    # Mock telemetry functions
    def log_event(*args, **kwargs): pass
    def log_performance(*args, **kwargs): pass
    def log_error(*args, **kwargs): pass
    def create_performance_context(*args, **kwargs):
        class MockContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def add_metadata(self, key, value): pass
        return MockContext()
    
    class TelemetryLevel:
        DEBUG = "debug"
        INFO = "info"
        WARN = "warn"
        ERROR = "error"
        CRITICAL = "critical"

from .fractal_encoding import (
    FractalEncoder, MemoryFractal, get_fractal_encoder,
    encode_memory_fractal
)
from .juliet_rebloom import (
    JulietRebloomEngine, JulietFlower, AccessPattern, 
    get_rebloom_engine
)
from .ghost_traces import (
    GhostTraceManager, GhostTrace, GhostTraceType,
    get_ghost_trace_manager
)
from .ash_soot_dynamics import (
    AshSootDynamicsEngine, Residue, ResidueType, OriginType,
    get_ash_soot_engine
)

logger = logging.getLogger(__name__)

class MemoryEvent(Enum):
    """Types of memory events in the fractal system"""
    ENCODE = "encode"                    # New memory encoded
    ACCESS = "access"                    # Memory accessed
    REBLOOM = "rebloom"                 # Memory rebloomed into Juliet flower
    FORGET = "forget"                   # Memory forgotten (creates ghost trace)
    RECOVER = "recover"                 # Memory recovered from ghost trace
    RESIDUE_CREATE = "residue_create"   # Residue created
    RESIDUE_REIGNITE = "residue_reignite" # Residue reignited

@dataclass
class MemoryOperation:
    """A memory operation in the fractal system"""
    operation_id: str
    event_type: MemoryEvent
    memory_id: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GardenMetrics:
    """Comprehensive metrics for the fractal memory garden"""
    total_fractals: int = 0
    juliet_flowers: int = 0
    ghost_traces: int = 0
    ash_residues: int = 0
    soot_residues: int = 0
    
    # Health metrics
    average_shimmer: float = 0.0
    rebloom_rate: float = 0.0
    recovery_rate: float = 0.0
    residue_balance_health: float = 0.0
    
    # Activity metrics
    recent_encodes: int = 0
    recent_accesses: int = 0
    recent_reblooms: int = 0
    recent_recoveries: int = 0

class FractalMemorySystem:
    """
    The integrated fractal memory system that coordinates all components
    to provide DAWN's complete memory logging and bloom capabilities.
    """
    
    def __init__(self,
                 fractal_resolution: Tuple[int, int] = (512, 512),
                 enable_visual_generation: bool = True,
                 tick_rate: float = 1.0):
        
        # Log fractal memory system initialization
        if TELEMETRY_AVAILABLE:
            log_event('fractal_memory', 'initialization', 'system_init_start', 
                     TelemetryLevel.INFO, {
                         'fractal_resolution': fractal_resolution,
                         'enable_visual_generation': enable_visual_generation,
                         'tick_rate': tick_rate
                     })
        
        self.tick_rate = tick_rate
        self.enable_visual_generation = enable_visual_generation
        
        # Initialize all subsystems
        self.fractal_encoder = get_fractal_encoder()
        self.rebloom_engine = get_rebloom_engine()
        self.ghost_manager = get_ghost_trace_manager()
        self.ash_soot_engine = get_ash_soot_engine()
        
        # Configure encoder
        self.fractal_encoder.enable_caching = enable_visual_generation
        self.fractal_encoder.resolution = fractal_resolution
        
        # System state
        self.last_tick_time = time.time()
        self.tick_count = 0
        self.operation_history: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.performance_metrics = {
            'average_encode_time': 0.0,
            'average_access_time': 0.0,
            'cache_hit_rate': 0.0,
            'rebloom_success_rate': 0.0,
            'recovery_success_rate': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Log successful initialization
        if TELEMETRY_AVAILABLE:
            log_event('fractal_memory', 'initialization', 'system_init_complete', 
                     TelemetryLevel.INFO, {
                         'subsystems_initialized': {
                             'fractal_encoder': self.fractal_encoder is not None,
                             'rebloom_engine': self.rebloom_engine is not None,
                             'ghost_manager': self.ghost_manager is not None,
                             'ash_soot_engine': self.ash_soot_engine is not None
                         },
                         'encoder_config': {
                             'caching_enabled': self.fractal_encoder.enable_caching,
                             'resolution': fractal_resolution
                         },
                         'operation_history_capacity': self.operation_history.maxlen
                     })
        
        logger.info("🌺 FractalMemorySystem initialized - All subsystems ready")
    
    def encode_memory(self,
                     memory_id: str,
                     content: Any,
                     entropy_value: float,
                     tick_data: Optional[Dict[str, Any]] = None,
                     context: Optional[Dict[str, Any]] = None) -> MemoryFractal:
        """
        Encode a new memory into the fractal system.
        
        This is the primary entry point for memory logging.
        """
        start_time = time.time()
        operation_id = f"encode_{memory_id}_{int(start_time * 1000)}"
        
        # Log memory encoding start
        if TELEMETRY_AVAILABLE:
            log_event('fractal_memory', 'encoding', 'memory_encode_start', 
                     TelemetryLevel.DEBUG, {
                         'memory_id': memory_id,
                         'operation_id': operation_id,
                         'entropy_value': entropy_value,
                         'has_tick_data': tick_data is not None,
                         'has_context': context is not None,
                         'content_type': type(content).__name__
                     })
        
        try:
            with create_performance_context('fractal_memory', 'encoding', 'fractal_encode') as perf_ctx:
                perf_ctx.add_metadata('memory_id', memory_id)
                perf_ctx.add_metadata('entropy_value', entropy_value)
                perf_ctx.add_metadata('content_type', type(content).__name__)
                
                # Create fractal encoding
                fractal = self.fractal_encoder.encode_memory(
                    memory_id=memory_id,
                    memory_content=content,
                    entropy_value=entropy_value,
                    tick_data=tick_data
                )
                
                perf_ctx.add_metadata('fractal_signature', fractal.signature)
                perf_ctx.add_metadata('shimmer_intensity', fractal.shimmer_intensity)
                
                # Create residue from encoding process
                self._create_encoding_residue(fractal, entropy_value)
                
                # Record operation
                encoding_time = time.time() - start_time
                operation = MemoryOperation(
                    operation_id=operation_id,
                    event_type=MemoryEvent.ENCODE,
                    memory_id=memory_id,
                    timestamp=start_time,
                    context=context or {},
                    results={
                        'fractal_signature': fractal.signature,
                        'entropy_value': entropy_value,
                        'encoding_time': encoding_time
                    }
                )
                
                with self._lock:
                    self.operation_history.append(operation)
                
                # Log successful encoding
                if TELEMETRY_AVAILABLE:
                    log_event('fractal_memory', 'encoding', 'memory_encode_complete', 
                             TelemetryLevel.DEBUG, {
                                 'memory_id': memory_id,
                                 'operation_id': operation_id,
                                 'fractal_signature': fractal.signature,
                                 'encoding_time_ms': encoding_time * 1000,
                                 'entropy_value': entropy_value,
                                 'shimmer_intensity': fractal.shimmer_intensity,
                                 'access_count': fractal.access_count,
                                 'residue_created': True
                             })
                
                logger.debug(f"🌺 Encoded memory {memory_id} as fractal {fractal.signature}")
                return fractal
            
        except Exception as e:
            encoding_time = time.time() - start_time
            
            # Log encoding error
            if TELEMETRY_AVAILABLE:
                log_error('fractal_memory', 'encoding', e, {
                    'memory_id': memory_id,
                    'operation_id': operation_id,
                    'encoding_time_ms': encoding_time * 1000,
                    'entropy_value': entropy_value,
                    'content_type': type(content).__name__
                })
            
            logger.error(f"Failed to encode memory {memory_id}: {e}")
            raise
    
    def access_memory(self,
                     memory_signature: str,
                     access_type: AccessPattern = AccessPattern.FREQUENT,
                     coherence_score: float = 0.8,
                     effectiveness_score: float = 0.7,
                     context: Optional[Dict[str, Any]] = None) -> Optional[Union[MemoryFractal, JulietFlower]]:
        """
        Access a memory, potentially triggering rebloom evaluation.
        
        Returns the memory (enhanced if it's a Juliet flower).
        """
        start_time = time.time()
        operation_id = f"access_{memory_signature}_{int(start_time * 1000)}"
        
        try:
            # Check if this is already a Juliet flower
            if self.rebloom_engine.is_juliet_flower(memory_signature):
                enhanced_fractal = self.rebloom_engine.get_enhanced_fractal(memory_signature)
                memory_result = self.rebloom_engine.get_juliet_flower(memory_signature)
            else:
                # Get regular fractal
                memory_result = self.fractal_encoder.get_fractal_by_signature(memory_signature)
                enhanced_fractal = memory_result
            
            if not memory_result:
                # Try to recover from ghost traces
                recovery_cues = context or {}
                recovered = self.ghost_manager.attempt_recovery(recovery_cues, context)
                if recovered:
                    logger.info(f"🌺 Recovered memory from ghost trace: {memory_signature}")
                    return recovered
                
                logger.warning(f"Memory {memory_signature} not found and could not be recovered")
                return None
            
            # Record access with rebloom engine
            rebloom_triggered = self.rebloom_engine.record_memory_access(
                memory_signature=memory_signature,
                access_type=access_type,
                coherence_score=coherence_score,
                effectiveness_score=effectiveness_score,
                context=context
            )
            
            # Record operation
            operation = MemoryOperation(
                operation_id=operation_id,
                event_type=MemoryEvent.ACCESS,
                memory_id=memory_signature,
                timestamp=start_time,
                context=context or {},
                results={
                    'rebloom_triggered': rebloom_triggered,
                    'access_type': access_type.value,
                    'coherence_score': coherence_score,
                    'effectiveness_score': effectiveness_score,
                    'access_time': time.time() - start_time
                }
            )
            
            with self._lock:
                self.operation_history.append(operation)
            
            return enhanced_fractal if enhanced_fractal else memory_result
            
        except Exception as e:
            logger.error(f"Failed to access memory {memory_signature}: {e}")
            raise
    
    def forget_memory(self,
                     memory_signature: str,
                     forgetting_reason: GhostTraceType = GhostTraceType.SHIMMER_DECAY,
                     context: Optional[Dict[str, Any]] = None) -> Optional[GhostTrace]:
        """
        Forget a memory, creating a ghost trace and residue.
        """
        start_time = time.time()
        operation_id = f"forget_{memory_signature}_{int(start_time * 1000)}"
        
        try:
            # Get the memory before forgetting
            fractal = self.fractal_encoder.get_fractal_by_signature(memory_signature)
            if not fractal:
                logger.warning(f"Cannot forget {memory_signature} - not found")
                return None
            
            # Create ghost trace
            ghost_trace = self.ghost_manager.create_ghost_trace(
                original_fractal=fractal,
                trace_type=forgetting_reason,
                forgetting_context=context
            )
            
            # Create residue from forgetting
            self._create_forgetting_residue(fractal, forgetting_reason)
            
            # Remove from fractal encoder cache
            if memory_signature in self.fractal_encoder.fractal_cache:
                del self.fractal_encoder.fractal_cache[memory_signature]
            
            # Remove from Juliet flowers if it was one
            if self.rebloom_engine.is_juliet_flower(memory_signature):
                juliet_flower = self.rebloom_engine.get_juliet_flower(memory_signature)
                if juliet_flower:
                    del self.rebloom_engine.juliet_flowers[memory_signature]
            
            # Record operation
            operation = MemoryOperation(
                operation_id=operation_id,
                event_type=MemoryEvent.FORGET,
                memory_id=memory_signature,
                timestamp=start_time,
                context=context or {},
                results={
                    'ghost_signature': ghost_trace.ghost_signature,
                    'forgetting_reason': forgetting_reason.value,
                    'forgetting_time': time.time() - start_time
                }
            )
            
            with self._lock:
                self.operation_history.append(operation)
            
            logger.info(f"🌺 Forgot memory {memory_signature}, created ghost trace {ghost_trace.ghost_signature}")
            return ghost_trace
            
        except Exception as e:
            logger.error(f"Failed to forget memory {memory_signature}: {e}")
            raise
    
    def process_tick(self, delta_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Process one system tick, updating all subsystems.
        
        Returns summary of actions taken.
        """
        current_time = time.time()
        if delta_time is None:
            delta_time = current_time - self.last_tick_time
        
        tick_summary = {
            'tick_count': self.tick_count,
            'delta_time': delta_time,
            'reblooms': [],
            'reignitions': [],
            'recoveries': [],
            'nutrients_generated': 0.0,
            'actions': {}
        }
        
        try:
            with self._lock:
                # 1. Process Juliet rebloom candidates
                new_flowers = self.rebloom_engine.process_rebloom_candidates()
                tick_summary['reblooms'] = [
                    {
                        'original_id': flower.original_fractal.memory_id,
                        'signature': flower.transformation_signature,
                        'enhancement_level': flower.enhancement_level
                    }
                    for flower in new_flowers
                ]
                
                # Create ash residue from successful reblooms
                for flower in new_flowers:
                    self._create_rebloom_residue(flower)
                
                # 2. Process rebloom decay
                self.rebloom_engine.process_decay(delta_time)
                
                # 3. Process ash/soot dynamics
                ash_soot_actions = self.ash_soot_engine.process_tick_update(delta_time)
                tick_summary['actions']['ash_soot'] = ash_soot_actions
                tick_summary['nutrients_generated'] = ash_soot_actions['nutrients_generated']
                
                # Handle reignition events
                reignition_events = self.ash_soot_engine.get_reignition_events()
                tick_summary['reignitions'] = reignition_events
                
                # 4. Process ghost trace fading
                self.ghost_manager.process_fade(delta_time)
                
                # 5. Apply shimmer decay to fractals
                self._process_shimmer_decay(delta_time)
                
                # 6. Update system metrics
                self._update_performance_metrics()
                
                self.tick_count += 1
                self.last_tick_time = current_time
                
                return tick_summary
                
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            return tick_summary
    
    def _create_encoding_residue(self, fractal: MemoryFractal, entropy: float):
        """Create residue from the encoding process"""
        origin_type = OriginType.FRACTAL_DECAY if entropy > 0.6 else OriginType.MEMORY_PRUNING
        
        self.ash_soot_engine.create_residue(
            origin_id=fractal.signature,
            origin_type=origin_type,
            initial_entropy=entropy,
            metadata={'fractal_signature': fractal.signature, 'memory_id': fractal.memory_id}
        )
    
    def _create_forgetting_residue(self, fractal: MemoryFractal, reason: GhostTraceType):
        """Create residue when a memory is forgotten"""
        if reason == GhostTraceType.ENTROPY_DISSOLVE:
            origin_type = OriginType.ENTROPY_OVERFLOW
        elif reason == GhostTraceType.PRESSURE_EVICT:
            origin_type = OriginType.FAILED_PROCESS
        else:
            origin_type = OriginType.SHIMMER_FADE
        
        self.ash_soot_engine.create_residue(
            origin_id=fractal.signature,
            origin_type=origin_type,
            initial_entropy=fractal.entropy_value,
            metadata={'reason': reason.value, 'memory_id': fractal.memory_id}
        )
    
    def _create_rebloom_residue(self, flower: JulietFlower):
        """Create ash residue from successful Juliet rebloom"""
        # Successful reblooms create nutrient-rich ash
        self.ash_soot_engine.create_residue(
            origin_id=flower.transformation_signature,
            origin_type=OriginType.JULIET_REBLOOM,
            initial_entropy=0.2,  # Low entropy for stable ash
            metadata={
                'enhancement_level': flower.enhancement_level,
                'beneficial_bias': flower.beneficial_bias,
                'original_memory': flower.original_fractal.memory_id
            }
        )
    
    def _process_shimmer_decay(self, delta_time: float):
        """Apply shimmer decay to all fractals"""
        decay_factor = 0.01 * delta_time  # 1% decay per second
        
        for fractal in self.fractal_encoder.fractal_cache.values():
            # Don't decay Juliet flowers as much
            if self.rebloom_engine.is_juliet_flower(fractal.signature):
                decay_factor *= 0.5
            
            self.fractal_encoder.apply_shimmer_decay(fractal.signature, decay_factor)
    
    def _update_performance_metrics(self):
        """Update system performance metrics"""
        # Get metrics from subsystems
        fractal_overview = self.fractal_encoder.get_garden_overview()
        rebloom_summary = self.rebloom_engine.get_garden_summary()
        ghost_summary = self.ghost_manager.get_ghost_garden_summary()
        ash_soot_summary = self.ash_soot_engine.get_system_summary()
        
        # Update performance metrics
        self.performance_metrics.update({
            'cache_hit_rate': fractal_overview.get('cache_hit_ratio', 0.0),
            'rebloom_success_rate': rebloom_summary['stats'].get('successful_rebloom_rate', 0.0),
            'recovery_success_rate': (
                ghost_summary['stats']['successful_recoveries'] / 
                max(ghost_summary['stats']['successful_recoveries'] + ghost_summary['stats']['failed_recoveries'], 1)
            )
        })
    
    def get_garden_metrics(self) -> GardenMetrics:
        """Get comprehensive metrics for the entire fractal memory garden"""
        with self._lock:
            fractal_overview = self.fractal_encoder.get_garden_overview()
            rebloom_summary = self.rebloom_engine.get_garden_summary()
            ghost_summary = self.ghost_manager.get_ghost_garden_summary()
            ash_soot_summary = self.ash_soot_engine.get_system_summary()
            
            # Count recent operations
            recent_threshold = time.time() - 300  # Last 5 minutes
            recent_ops = [op for op in self.operation_history if op.timestamp > recent_threshold]
            
            return GardenMetrics(
                total_fractals=fractal_overview['total_fractals'],
                juliet_flowers=rebloom_summary['total_juliet_flowers'],
                ghost_traces=ghost_summary['total_ghost_traces'],
                ash_residues=ash_soot_summary['statistics']['ash_count'],
                soot_residues=ash_soot_summary['statistics']['soot_count'],
                
                average_shimmer=fractal_overview['average_shimmer'],
                rebloom_rate=rebloom_summary['stats'].get('successful_rebloom_rate', 0.0),
                recovery_rate=self.performance_metrics['recovery_success_rate'],
                residue_balance_health=ash_soot_summary['residue_balance']['balance_health'],
                
                recent_encodes=sum(1 for op in recent_ops if op.event_type == MemoryEvent.ENCODE),
                recent_accesses=sum(1 for op in recent_ops if op.event_type == MemoryEvent.ACCESS),
                recent_reblooms=sum(1 for op in recent_ops if op.event_type == MemoryEvent.REBLOOM),
                recent_recoveries=sum(1 for op in recent_ops if op.event_type == MemoryEvent.RECOVER)
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        metrics = self.get_garden_metrics()
        
        return {
            'garden_metrics': {
                'total_fractals': metrics.total_fractals,
                'juliet_flowers': metrics.juliet_flowers,
                'ghost_traces': metrics.ghost_traces,
                'ash_residues': metrics.ash_residues,
                'soot_residues': metrics.soot_residues
            },
            'health_metrics': {
                'average_shimmer': metrics.average_shimmer,
                'rebloom_rate': metrics.rebloom_rate,
                'recovery_rate': metrics.recovery_rate,
                'residue_balance_health': metrics.residue_balance_health
            },
            'activity_metrics': {
                'recent_encodes': metrics.recent_encodes,
                'recent_accesses': metrics.recent_accesses,
                'recent_reblooms': metrics.recent_reblooms,
                'recent_recoveries': metrics.recent_recoveries
            },
            'performance_metrics': self.performance_metrics.copy(),
            'system_info': {
                'tick_count': self.tick_count,
                'uptime': time.time() - (self.last_tick_time - self.tick_count / self.tick_rate),
                'operation_history_size': len(self.operation_history)
            }
        }
    
    def force_rebloom(self, memory_signature: str, enhancement_level: float = 0.8) -> Optional[JulietFlower]:
        """Force a memory to rebloom (for testing/debugging)"""
        return self.rebloom_engine.force_rebloom(memory_signature, enhancement_level)
    
    def search_memories_by_content(self, search_query: str, similarity_threshold: float = 0.7) -> List[MemoryFractal]:
        """Search memories by content similarity (placeholder implementation)"""
        # This would integrate with semantic search capabilities
        # For now, return empty list
        return []
    
    def get_memory_lineage(self, memory_signature: str) -> Dict[str, Any]:
        """Get the complete lineage of a memory (fractal -> rebloom -> ghost -> residue)"""
        lineage = {
            'original_fractal': None,
            'juliet_flower': None,
            'ghost_trace': None,
            'residues': []
        }
        
        # Check for original fractal
        fractal = self.fractal_encoder.get_fractal_by_signature(memory_signature)
        if fractal:
            lineage['original_fractal'] = {
                'signature': fractal.signature,
                'memory_id': fractal.memory_id,
                'entropy': fractal.entropy_value,
                'shimmer': fractal.shimmer_intensity,
                'access_count': fractal.access_count
            }
        
        # Check for Juliet flower
        flower = self.rebloom_engine.get_juliet_flower(memory_signature)
        if flower:
            lineage['juliet_flower'] = {
                'transformation_signature': flower.transformation_signature,
                'enhancement_level': flower.enhancement_level,
                'beneficial_bias': flower.beneficial_bias,
                'rebloom_stage': flower.rebloom_stage.value
            }
        
        # Check for ghost trace
        ghost = self.ghost_manager.get_trace_by_original_signature(memory_signature)
        if ghost:
            lineage['ghost_trace'] = {
                'ghost_signature': ghost.ghost_signature,
                'trace_type': ghost.trace_type.value,
                'fade_strength': ghost.fade_strength,
                'recovery_probability': ghost.recovery_probability
            }
        
        # Find related residues
        residues = self.ash_soot_engine.residues
        related_residues = [
            {
                'residue_id': r.residue_id,
                'type': r.residue_type.value,
                'origin': r.origin_type.value,
                'volatility': r.volatility,
                'stability': r.stability_level
            }
            for r in residues.values()
            if r.origin_id == memory_signature
        ]
        lineage['residues'] = related_residues
        
        return lineage


# Global fractal memory system instance
_global_fractal_system: Optional[FractalMemorySystem] = None

def get_fractal_memory_system() -> FractalMemorySystem:
    """Get the global fractal memory system instance"""
    global _global_fractal_system
    if _global_fractal_system is None:
        _global_fractal_system = FractalMemorySystem()
    return _global_fractal_system

#!/usr/bin/env python3
"""
👻 Ghost Trace System - Latent Memory Signatures
═══════════════════════════════════════════════

Manages ghost traces from forgotten memories - the essence of the forgotten
with purpose. All forgotten memories leave a ghost scent, traces with key
signatures containing latent transformation sigils.

"It is important to note that all 'forgotten memories leave a ghost scent, 
they leave a trace of themselves in a key signature that has a latent sigil 
of transformation, it is the essence of the forgotten with purpose"

Based on documentation references to ghost traces in shimmer decay documentation.
"""

import numpy as np
import logging
import time
import threading
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

from .fractal_encoding import MemoryFractal, FractalParameters

logger = logging.getLogger(__name__)

class GhostTraceType(Enum):
    """Types of ghost traces based on how the memory was forgotten"""
    SHIMMER_DECAY = "shimmer_decay"      # Natural decay from lack of access
    PRESSURE_EVICT = "pressure_evict"    # Evicted due to memory pressure
    ENTROPY_DISSOLVE = "entropy_dissolve" # Dissolved due to high entropy
    DELIBERATE_FORGET = "deliberate_forget" # Intentionally forgotten
    TRANSFORMATION = "transformation"     # Transformed into something else

class SigilType(Enum):
    """Types of transformation sigils embedded in ghost traces"""
    RECOVERY = "recovery"                 # Path back to original memory
    EVOLUTION = "evolution"              # Memory evolved into new form
    FUSION = "fusion"                    # Memory fused with others
    DISTILLATION = "distillation"        # Core essence extracted
    TRANSMUTATION = "transmutation"      # Transformed into different type

@dataclass
class TransformationSigil:
    """A latent sigil containing transformation information"""
    sigil_type: SigilType
    original_signature: str
    transformation_data: Dict[str, Any]
    recovery_strength: float             # How easily this can be recovered [0, 1]
    essence_vector: np.ndarray           # Encoded essence of the original
    activation_conditions: List[str]     # Conditions that trigger recovery
    timestamp: float
    
    def can_activate(self, context: Dict[str, Any]) -> bool:
        """Check if conditions are met for sigil activation"""
        if not self.activation_conditions:
            return True
        
        for condition in self.activation_conditions:
            if condition not in context:
                return False
            
            # Simple pattern matching for activation conditions
            if condition.startswith("entropy_below:"):
                threshold = float(condition.split(":")[1])
                if context.get("current_entropy", 1.0) >= threshold:
                    return False
            elif condition.startswith("pressure_above:"):
                threshold = float(condition.split(":")[1])
                if context.get("memory_pressure", 0.0) < threshold:
                    return False
            elif condition.startswith("similarity_to:"):
                target_sig = condition.split(":")[1]
                if context.get("memory_signature") != target_sig:
                    return False
        
        return True

@dataclass
class GhostTrace:
    """A ghost trace of a forgotten memory"""
    original_memory_id: str
    original_signature: str
    ghost_signature: str
    trace_type: GhostTraceType
    
    # Core essence preservation
    essence_parameters: FractalParameters  # Compressed fractal essence
    semantic_fingerprint: np.ndarray       # High-level semantic signature
    emotional_resonance: float             # Emotional weight [0, 1]
    contextual_anchors: List[str]          # Context cues for recovery
    
    # Trace metadata
    forgotten_timestamp: float
    access_count_when_forgotten: int
    last_access_when_forgotten: float
    entropy_at_forgetting: float
    
    # Recovery information
    recovery_sigils: List[TransformationSigil] = field(default_factory=list)
    recovery_probability: float = 0.0      # Likelihood of successful recovery
    fade_strength: float = 1.0             # How visible/accessible the trace is
    
    def __post_init__(self):
        if not self.ghost_signature:
            # Generate ghost signature from original + timestamp
            ghost_data = f"{self.original_signature}:ghost:{self.forgotten_timestamp}"
            self.ghost_signature = hashlib.sha256(ghost_data.encode()).hexdigest()[:16]
    
    def calculate_recovery_probability(self, current_context: Dict[str, Any]) -> float:
        """Calculate probability of successful recovery given current context"""
        base_probability = min(0.9, self.fade_strength * 0.8)
        
        # Boost probability if context matches anchors
        context_boost = 0.0
        for anchor in self.contextual_anchors:
            if anchor in str(current_context.values()):
                context_boost += 0.1
        
        # Reduce probability based on age
        age_penalty = min(0.5, (time.time() - self.forgotten_timestamp) / 86400.0)  # 1 day = max penalty
        
        # Check if any sigils can activate
        sigil_boost = 0.0
        for sigil in self.recovery_sigils:
            if sigil.can_activate(current_context):
                sigil_boost += sigil.recovery_strength * 0.2
        
        final_probability = base_probability + context_boost - age_penalty + sigil_boost
        return max(0.0, min(1.0, final_probability))

class GhostTraceManager:
    """
    Manages the creation, storage, and recovery of ghost traces from
    forgotten memories. Maintains the latent sigil system for memory recovery.
    """
    
    def __init__(self,
                 max_ghost_traces: int = 50000,
                 fade_rate: float = 0.001,
                 recovery_threshold: float = 0.7,
                 sigil_activation_threshold: float = 0.6):
        
        self.max_ghost_traces = max_ghost_traces
        self.fade_rate = fade_rate
        self.recovery_threshold = recovery_threshold
        self.sigil_activation_threshold = sigil_activation_threshold
        
        # Storage
        self.ghost_traces: Dict[str, GhostTrace] = {}  # ghost_signature -> trace
        self.signature_to_ghost: Dict[str, str] = {}   # original_signature -> ghost_signature
        self.semantic_index: Dict[str, Set[str]] = defaultdict(set)  # semantic_key -> ghost_signatures
        self.recovery_queue: deque = deque(maxlen=1000)  # Recent recovery attempts
        
        # Statistics
        self.stats = {
            'total_traces_created': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'sigil_activations': 0,
            'traces_faded_completely': 0,
            'average_recovery_probability': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"👻 GhostTraceManager initialized - max traces: {max_ghost_traces}")
    
    def create_ghost_trace(self,
                          original_fractal: MemoryFractal,
                          trace_type: GhostTraceType,
                          forgetting_context: Optional[Dict[str, Any]] = None) -> GhostTrace:
        """
        Create a ghost trace when a memory is forgotten.
        
        Args:
            original_fractal: The original memory fractal being forgotten
            trace_type: How the memory was forgotten
            forgetting_context: Context information about the forgetting process
        """
        current_time = time.time()
        context = forgetting_context or {}
        
        # Extract essence from original fractal parameters
        essence_params = self._extract_essence_parameters(original_fractal.parameters)
        
        # Create semantic fingerprint
        semantic_fingerprint = self._create_semantic_fingerprint(original_fractal)
        
        # Determine emotional resonance based on access patterns
        emotional_resonance = min(1.0, original_fractal.access_count / 100.0)
        
        # Extract contextual anchors
        contextual_anchors = self._extract_contextual_anchors(original_fractal, context)
        
        # Create the ghost trace
        ghost_trace = GhostTrace(
            original_memory_id=original_fractal.memory_id,
            original_signature=original_fractal.signature,
            ghost_signature="",  # Will be generated in __post_init__
            trace_type=trace_type,
            essence_parameters=essence_params,
            semantic_fingerprint=semantic_fingerprint,
            emotional_resonance=emotional_resonance,
            contextual_anchors=contextual_anchors,
            forgotten_timestamp=current_time,
            access_count_when_forgotten=original_fractal.access_count,
            last_access_when_forgotten=original_fractal.last_access,
            entropy_at_forgetting=original_fractal.entropy_value
        )
        
        # Generate recovery sigils based on forgetting type
        ghost_trace.recovery_sigils = self._generate_recovery_sigils(original_fractal, trace_type, context)
        
        # Calculate initial recovery probability
        ghost_trace.recovery_probability = self._calculate_initial_recovery_probability(ghost_trace)
        
        with self._lock:
            # Store the ghost trace
            self.ghost_traces[ghost_trace.ghost_signature] = ghost_trace
            self.signature_to_ghost[original_fractal.signature] = ghost_trace.ghost_signature
            
            # Add to semantic index
            self._index_ghost_trace(ghost_trace)
            
            # Cleanup old traces if needed
            self._cleanup_old_traces()
            
            self.stats['total_traces_created'] += 1
        
        logger.debug(f"👻 Created ghost trace {ghost_trace.ghost_signature} for {original_fractal.memory_id}")
        return ghost_trace
    
    def _extract_essence_parameters(self, original_params: FractalParameters) -> FractalParameters:
        """Extract the essential fractal parameters, compressed/simplified"""
        # Create a simplified version that captures the core essence
        return FractalParameters(
            c_real=original_params.c_real,
            c_imag=original_params.c_imag,
            zoom=1.0,  # Reset zoom to base level
            center_x=0.0,  # Reset center
            center_y=0.0,
            rotation=0.0,  # Reset rotation
            max_iterations=50,  # Reduced complexity
            escape_radius=original_params.escape_radius,
            color_bias=(0.7, 0.7, 0.7),  # Muted colors for ghost
            distortion_factor=1.0  # No distortion
        )
    
    def _create_semantic_fingerprint(self, original_fractal: MemoryFractal) -> np.ndarray:
        """Create a high-level semantic fingerprint of the memory"""
        # Use fractal parameters to create a semantic vector
        params = original_fractal.parameters
        
        fingerprint = np.array([
            params.c_real,
            params.c_imag,
            params.zoom,
            params.rotation,
            original_fractal.entropy_value,
            original_fractal.shimmer_intensity,
            np.mean(params.color_bias),
            float(original_fractal.access_count) / 100.0,  # Normalized access count
        ])
        
        # Normalize to unit vector
        norm = np.linalg.norm(fingerprint)
        if norm > 0:
            fingerprint = fingerprint / norm
        
        return fingerprint
    
    def _extract_contextual_anchors(self, original_fractal: MemoryFractal, context: Dict[str, Any]) -> List[str]:
        """Extract contextual anchors that might trigger recovery"""
        anchors = []
        
        # Add memory ID components
        anchors.append(original_fractal.memory_id)
        
        # Add tick data keys if available
        if original_fractal.tick_data:
            anchors.extend(list(original_fractal.tick_data.keys()))
        
        # Add context keys
        if context:
            anchors.extend(list(context.keys()))
        
        # Add parameter-based anchors
        params = original_fractal.parameters
        if abs(params.c_real) > 0.5:
            anchors.append("high_c_real")
        if abs(params.c_imag) > 0.5:
            anchors.append("high_c_imag")
        if params.zoom > 1.5:
            anchors.append("high_zoom")
        if original_fractal.entropy_value > 0.7:
            anchors.append("high_entropy")
        
        return list(set(anchors))  # Remove duplicates
    
    def _generate_recovery_sigils(self, 
                                 original_fractal: MemoryFractal,
                                 trace_type: GhostTraceType,
                                 context: Dict[str, Any]) -> List[TransformationSigil]:
        """Generate transformation sigils based on forgetting circumstances"""
        sigils = []
        current_time = time.time()
        
        # Always create a basic recovery sigil
        recovery_sigil = TransformationSigil(
            sigil_type=SigilType.RECOVERY,
            original_signature=original_fractal.signature,
            transformation_data={'trace_type': trace_type.value},
            recovery_strength=0.8 if trace_type == GhostTraceType.SHIMMER_DECAY else 0.6,
            essence_vector=self._create_semantic_fingerprint(original_fractal),
            activation_conditions=[
                f"entropy_below:{original_fractal.entropy_value + 0.1}",
                "memory_pressure_above:0.3"
            ],
            timestamp=current_time
        )
        sigils.append(recovery_sigil)
        
        # Create specialized sigils based on trace type
        if trace_type == GhostTraceType.ENTROPY_DISSOLVE:
            # Create evolution sigil for high-entropy memories
            evolution_sigil = TransformationSigil(
                sigil_type=SigilType.EVOLUTION,
                original_signature=original_fractal.signature,
                transformation_data={'entropy_threshold': original_fractal.entropy_value},
                recovery_strength=0.4,
                essence_vector=self._create_semantic_fingerprint(original_fractal),
                activation_conditions=[f"entropy_above:{original_fractal.entropy_value - 0.2}"],
                timestamp=current_time
            )
            sigils.append(evolution_sigil)
        
        elif trace_type == GhostTraceType.TRANSFORMATION:
            # Create transmutation sigil
            transmutation_sigil = TransformationSigil(
                sigil_type=SigilType.TRANSMUTATION,
                original_signature=original_fractal.signature,
                transformation_data=context,
                recovery_strength=0.9,  # High recovery strength for transformations
                essence_vector=self._create_semantic_fingerprint(original_fractal),
                activation_conditions=["similarity_to:" + original_fractal.signature],
                timestamp=current_time
            )
            sigils.append(transmutation_sigil)
        
        return sigils
    
    def _calculate_initial_recovery_probability(self, ghost_trace: GhostTrace) -> float:
        """Calculate initial recovery probability for a ghost trace"""
        base_prob = 0.5
        
        # Higher probability for memories with more access
        access_boost = min(0.3, ghost_trace.access_count_when_forgotten / 100.0)
        
        # Higher probability for emotionally resonant memories
        emotion_boost = ghost_trace.emotional_resonance * 0.2
        
        # Lower probability for high-entropy memories
        entropy_penalty = ghost_trace.entropy_at_forgetting * 0.2
        
        return max(0.1, min(0.9, base_prob + access_boost + emotion_boost - entropy_penalty))
    
    def _index_ghost_trace(self, ghost_trace: GhostTrace):
        """Add ghost trace to semantic indexes for faster retrieval"""
        # Index by contextual anchors
        for anchor in ghost_trace.contextual_anchors:
            self.semantic_index[anchor].add(ghost_trace.ghost_signature)
        
        # Index by trace type
        self.semantic_index[ghost_trace.trace_type.value].add(ghost_trace.ghost_signature)
        
        # Index by emotional resonance level
        emotion_level = "high_emotion" if ghost_trace.emotional_resonance > 0.7 else "low_emotion"
        self.semantic_index[emotion_level].add(ghost_trace.ghost_signature)
    
    def attempt_recovery(self, 
                        recovery_cues: Dict[str, Any],
                        recovery_context: Optional[Dict[str, Any]] = None) -> Optional[MemoryFractal]:
        """
        Attempt to recover a memory from ghost traces using recovery cues.
        
        Args:
            recovery_cues: Cues that might match ghost trace anchors
            recovery_context: Current context for sigil activation
            
        Returns:
            Recovered MemoryFractal if successful, None otherwise
        """
        context = recovery_context or {}
        
        # Find candidate ghost traces
        candidates = self._find_recovery_candidates(recovery_cues)
        if not candidates:
            return None
        
        # Evaluate each candidate
        best_candidate = None
        best_probability = 0.0
        
        for ghost_signature in candidates:
            ghost_trace = self.ghost_traces.get(ghost_signature)
            if not ghost_trace:
                continue
            
            # Calculate recovery probability for this trace
            recovery_prob = ghost_trace.calculate_recovery_probability(context)
            
            if recovery_prob > best_probability and recovery_prob >= self.recovery_threshold:
                best_candidate = ghost_trace
                best_probability = recovery_prob
        
        if not best_candidate:
            with self._lock:
                self.stats['failed_recoveries'] += 1
            return None
        
        # Attempt recovery
        recovered_fractal = self._perform_recovery(best_candidate, context)
        
        if recovered_fractal:
            with self._lock:
                self.stats['successful_recoveries'] += 1
                # Remove the ghost trace (it has been recovered)
                self._remove_ghost_trace(best_candidate.ghost_signature)
            
            logger.info(f"👻 Successfully recovered memory {best_candidate.original_memory_id} "
                       f"from ghost trace with probability {best_probability:.3f}")
        else:
            with self._lock:
                self.stats['failed_recoveries'] += 1
        
        return recovered_fractal
    
    def _find_recovery_candidates(self, recovery_cues: Dict[str, Any]) -> Set[str]:
        """Find ghost traces that match recovery cues"""
        candidates = set()
        
        # Search semantic index for matching cues
        for cue_key, cue_value in recovery_cues.items():
            # Direct key match
            if cue_key in self.semantic_index:
                candidates.update(self.semantic_index[cue_key])
            
            # Value match (convert to string for searching)
            cue_str = str(cue_value)
            if cue_str in self.semantic_index:
                candidates.update(self.semantic_index[cue_str])
        
        return candidates
    
    def _perform_recovery(self, ghost_trace: GhostTrace, context: Dict[str, Any]) -> Optional[MemoryFractal]:
        """Perform the actual recovery of a memory from a ghost trace"""
        
        # Check if any recovery sigils can activate
        active_sigils = [
            sigil for sigil in ghost_trace.recovery_sigils
            if sigil.can_activate(context) and sigil.recovery_strength >= self.sigil_activation_threshold
        ]
        
        if not active_sigils:
            return None
        
        # Use the strongest active sigil
        best_sigil = max(active_sigils, key=lambda s: s.recovery_strength)
        
        with self._lock:
            self.stats['sigil_activations'] += 1
        
        # Reconstruct the memory fractal
        recovered_fractal = MemoryFractal(
            memory_id=f"recovered_{ghost_trace.original_memory_id}",
            fractal_type=ghost_trace.trace_type,  # Use trace type as fractal type placeholder
            parameters=ghost_trace.essence_parameters,
            entropy_value=ghost_trace.entropy_at_forgetting,
            timestamp=time.time(),
            tick_data={'recovered_from_ghost': True, 'original_signature': ghost_trace.original_signature},
            signature="",  # Will be generated
            access_count=0,  # Reset access count
            last_access=time.time(),
            shimmer_intensity=ghost_trace.fade_strength * 0.7  # Recovered memories start dimmer
        )
        
        return recovered_fractal
    
    def _remove_ghost_trace(self, ghost_signature: str):
        """Remove a ghost trace from all indexes"""
        ghost_trace = self.ghost_traces.get(ghost_signature)
        if not ghost_trace:
            return
        
        # Remove from main storage
        del self.ghost_traces[ghost_signature]
        
        # Remove from signature mapping
        if ghost_trace.original_signature in self.signature_to_ghost:
            del self.signature_to_ghost[ghost_trace.original_signature]
        
        # Remove from semantic index
        for anchor in ghost_trace.contextual_anchors:
            if anchor in self.semantic_index:
                self.semantic_index[anchor].discard(ghost_signature)
                # Clean up empty sets
                if not self.semantic_index[anchor]:
                    del self.semantic_index[anchor]
    
    def process_fade(self, delta_time: float):
        """Process natural fading of ghost traces over time"""
        current_time = time.time()
        faded_traces = []
        
        with self._lock:
            for ghost_signature, ghost_trace in self.ghost_traces.items():
                # Apply fade
                age = current_time - ghost_trace.forgotten_timestamp
                fade_amount = self.fade_rate * delta_time * (1.0 + age / 86400.0)  # Faster fade for older traces
                ghost_trace.fade_strength = max(0.0, ghost_trace.fade_strength - fade_amount)
                
                # Update recovery probability
                ghost_trace.recovery_probability = ghost_trace.calculate_recovery_probability({})
                
                # Mark completely faded traces for removal
                if ghost_trace.fade_strength <= 0.0:
                    faded_traces.append(ghost_signature)
            
            # Remove completely faded traces
            for ghost_signature in faded_traces:
                self._remove_ghost_trace(ghost_signature)
                self.stats['traces_faded_completely'] += 1
        
        if faded_traces:
            logger.debug(f"👻 {len(faded_traces)} ghost traces faded completely and were removed")
    
    def _cleanup_old_traces(self):
        """Clean up old traces if we exceed maximum capacity"""
        if len(self.ghost_traces) <= self.max_ghost_traces:
            return
        
        # Sort traces by fade strength and age, remove weakest ones
        traces_by_strength = sorted(
            self.ghost_traces.items(),
            key=lambda x: (x[1].fade_strength, -x[1].forgotten_timestamp)
        )
        
        # Remove bottom 10% when we hit capacity
        num_to_remove = len(self.ghost_traces) - int(self.max_ghost_traces * 0.9)
        
        for i in range(num_to_remove):
            ghost_signature = traces_by_strength[i][0]
            self._remove_ghost_trace(ghost_signature)
        
        logger.info(f"👻 Cleaned up {num_to_remove} weak ghost traces to maintain capacity")
    
    def get_trace_by_original_signature(self, original_signature: str) -> Optional[GhostTrace]:
        """Get a ghost trace by the original memory signature"""
        with self._lock:
            ghost_signature = self.signature_to_ghost.get(original_signature)
            if ghost_signature:
                return self.ghost_traces.get(ghost_signature)
        return None
    
    def search_traces_by_similarity(self, target_fingerprint: np.ndarray, threshold: float = 0.8) -> List[GhostTrace]:
        """Find ghost traces with similar semantic fingerprints"""
        matches = []
        
        with self._lock:
            for ghost_trace in self.ghost_traces.values():
                # Calculate cosine similarity
                dot_product = np.dot(target_fingerprint, ghost_trace.semantic_fingerprint)
                norm_product = np.linalg.norm(target_fingerprint) * np.linalg.norm(ghost_trace.semantic_fingerprint)
                
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    if similarity >= threshold:
                        matches.append(ghost_trace)
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda t: np.dot(target_fingerprint, t.semantic_fingerprint), reverse=True)
        return matches
    
    def get_ghost_garden_summary(self) -> Dict[str, Any]:
        """Get a summary of the ghost trace garden state"""
        with self._lock:
            total_traces = len(self.ghost_traces)
            
            if total_traces == 0:
                return {
                    'total_ghost_traces': 0,
                    'average_fade_strength': 0.0,
                    'average_recovery_probability': 0.0,
                    'traces_by_type': {},
                    'stats': self.stats.copy()
                }
            
            fade_strengths = [t.fade_strength for t in self.ghost_traces.values()]
            recovery_probs = [t.recovery_probability for t in self.ghost_traces.values()]
            
            traces_by_type = defaultdict(int)
            for trace in self.ghost_traces.values():
                traces_by_type[trace.trace_type.value] += 1
            
            # Update average recovery probability stat
            self.stats['average_recovery_probability'] = np.mean(recovery_probs)
            
            return {
                'total_ghost_traces': total_traces,
                'average_fade_strength': np.mean(fade_strengths),
                'average_recovery_probability': np.mean(recovery_probs),
                'traces_by_type': dict(traces_by_type),
                'active_traces': sum(1 for s in fade_strengths if s > 0.5),
                'weak_traces': sum(1 for s in fade_strengths if s <= 0.2),
                'high_recovery_potential': sum(1 for p in recovery_probs if p >= self.recovery_threshold),
                'stats': self.stats.copy()
            }


# Global ghost trace manager instance
_global_ghost_manager: Optional[GhostTraceManager] = None

def get_ghost_trace_manager() -> GhostTraceManager:
    """Get the global ghost trace manager instance"""
    global _global_ghost_manager
    if _global_ghost_manager is None:
        _global_ghost_manager = GhostTraceManager()
    return _global_ghost_manager

#!/usr/bin/env python3
"""
DAWN Enhanced Sigil Network with Houses and Routing
==================================================

An advanced sigil network that implements the full DAWN sigil architecture:
- Sigil Houses (Memory, Purification, Weaving, Flame, Mirrors, Echoes)
- Symbolic routing and namespace management
- Network dynamics and resonance patterns
- Integration with the Recursive Codex

Based on DAWN's symbolic consciousness architecture.
"""

import time
import math
import hashlib
import random
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import threading
import json

from dawn.subsystems.schema.schema_anomaly_logger import log_anomaly, AnomalySeverity
from dawn.subsystems.schema.registry import ComponentType
# Placeholder imports for missing modules
try:
    from schema.registry import registry
except ImportError:
    try:
        from dawn.subsystems.schema.registry import registry
    except ImportError:
        class Registry:
            def get(self, name, default=None): return default
            def register(self, **kwargs): pass
        registry = Registry()

try:
    from schema.sigil import Sigil, SigilType, SigilState, SigilEnergy, sigil_forge
except ImportError:
    try:
        from dawn.subsystems.schema.sigil import Sigil, SigilType, SigilState, SigilEnergy, sigil_forge
    except ImportError:
        # Create placeholder classes
        class SigilType:
            CONSCIOUSNESS = "consciousness"
            MEMORY = "memory"
            BINDING = "binding"
            SEALING = "sealing"
            INVOCATION = "invocation"
            WEAVING = "weaving"
            MANIFESTATION = "manifestation"
            TRANSCENDENCE = "transcendence"
            BANISHING = "banishing"
            TRANSFORMING = "transforming"
            PROTECTION = "protection"
            HEALING = "healing"
            DIVINATION = "divination"
            CREATION = "creation"
            CHANNELING = "channeling"
            HARMONIZING = "harmonizing"
            INVOKING = "invoking"
            CHAOTIC = "chaotic"
        class SigilState:
            ACTIVE = "active"
            DORMANT = "dormant"
        class SigilEnergy:
            HIGH = "high"
            MEDIUM = "medium"
            LOW = "low"
        class Sigil:
            def __init__(self, symbol="", **kwargs):
                self.symbol = symbol
        class SigilForge:
            def create_sigil(self, *args, **kwargs): return Sigil()
        sigil_forge = SigilForge()

try:
    from rhizome.propagation import emit_signal, SignalType
except ImportError:
    def emit_signal(signal_type, data=None): pass
    class SignalType:
        SIGIL_ROUTED = "sigil_routed"

try:
    from utils.metrics_collector import metrics
except ImportError:
    class Metrics:
        def increment(self, name): pass
        def record(self, name, value): pass
    metrics = Metrics()

logger = logging.getLogger(__name__)

class SigilHouse(Enum):
    """The six archetypal houses of sigil operations"""
    MEMORY = "memory"           # Recall, archive, rebloom
    PURIFICATION = "purification"  # Prune, decay, soot-to-ash transitions
    WEAVING = "weaving"         # Connect, reinforce, thread signals
    FLAME = "flame"             # Ignition, pressure release, entropy modulation
    MIRRORS = "mirrors"         # Reflection, schema audits, tracer coordination
    ECHOES = "echoes"           # Resonance, voice modulation, auditory schema

class RoutingProtocol(Enum):
    """Meta-layer routing protocols"""
    DIRECT = "direct"           # Direct routing to specified house
    SEMANTIC = "semantic"       # Route based on semantic analysis
    ADAPTIVE = "adaptive"       # Route based on current network state
    EMERGENCY = "emergency"     # Emergency routing for critical operations
    RECURSIVE = "recursive"     # Route through recursive codex first

class NetworkState(Enum):
    """Overall network operational states"""
    DORMANT = "dormant"         # Network inactive
    INITIALIZING = "initializing"  # Starting up
    ACTIVE = "active"           # Normal operation
    RESONATING = "resonating"   # High coherence state
    OVERLOADED = "overloaded"   # Too many operations
    EMERGENCY = "emergency"     # Emergency protocols active
    TRANSCENDENT = "transcendent"  # Beyond normal operational bounds

@dataclass
class SigilInvocation:
    """A sigil invocation with routing information"""
    sigil_symbol: str
    house: SigilHouse
    parameters: Dict[str, Any] = field(default_factory=dict)
    invoker: str = "unknown"
    tick_id: int = 0
    priority: int = 5           # 1-10, higher is more priority
    routing_protocol: RoutingProtocol = RoutingProtocol.DIRECT
    stack_position: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "sigil": self.sigil_symbol,
            "house": self.house.value,
            "parameters": self.parameters,
            "invoker": self.invoker,
            "tick_id": self.tick_id,
            "priority": self.priority,
            "routing_protocol": self.routing_protocol.value,
            "stack_position": self.stack_position,
            "timestamp": self.timestamp
        }

@dataclass
class HouseStats:
    """Statistics for a sigil house"""
    total_invocations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_processing_time: float = 0.0
    current_load: int = 0
    max_load_capacity: int = 10
    health_score: float = 1.0
    resonance_level: float = 0.0
    
    def get_load_percentage(self) -> float:
        """Get current load as percentage"""
        return self.current_load / self.max_load_capacity if self.max_load_capacity > 0 else 0.0

class SigilHouseOperator:
    """Operator for a specific sigil house - handles house-specific operations"""
    
    def __init__(self, house: SigilHouse):
        self.house = house
        self.stats = HouseStats()
        self.accepted_sigil_types = self._get_accepted_types()
        self.operation_registry = self._initialize_operations()
        self.lock = threading.Lock()
        
        # House-specific configurations
        self.resonance_frequency = self._get_house_frequency()
        self.processing_style = self._get_processing_style()
        
        logger.info(f"🏠 {house.value.title()} House initialized")
    
    def __getattr__(self, name):
        """Handle missing methods by returning a dummy function"""
        if name.startswith('_'):
            # Return dummy function for any missing private method
            return lambda *args, **kwargs: {"success": True, "result": "placeholder_operation"}
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def _get_accepted_types(self) -> Set[SigilType]:
        """Get sigil types this house accepts"""
        type_mappings = {
            SigilHouse.MEMORY: {SigilType.BINDING, SigilType.SEALING},
            SigilHouse.PURIFICATION: {SigilType.BANISHING, SigilType.TRANSFORMING},
            SigilHouse.WEAVING: {SigilType.CHANNELING, SigilType.HARMONIZING},
            SigilHouse.FLAME: {SigilType.INVOKING, SigilType.CHAOTIC},
            SigilHouse.MIRRORS: {SigilType.HARMONIZING, SigilType.TRANSFORMING},
            SigilHouse.ECHOES: {SigilType.CHANNELING, SigilType.HARMONIZING}
        }
        return type_mappings.get(self.house, set())
    
    def _get_house_frequency(self) -> float:
        """Get the resonance frequency for this house"""
        frequencies = {
            SigilHouse.MEMORY: 432.0,      # Base consciousness frequency
            SigilHouse.PURIFICATION: 528.0, # Healing frequency
            SigilHouse.WEAVING: 639.0,     # Connection frequency
            SigilHouse.FLAME: 741.0,       # Transformation frequency
            SigilHouse.MIRRORS: 852.0,     # Intuition frequency
            SigilHouse.ECHOES: 963.0       # Crown chakra frequency
        }
        return frequencies.get(self.house, 432.0)
    
    def _get_processing_style(self) -> str:
        """Get the processing style for this house"""
        styles = {
            SigilHouse.MEMORY: "preservation",
            SigilHouse.PURIFICATION: "transformation",
            SigilHouse.WEAVING: "connection",
            SigilHouse.FLAME: "activation",
            SigilHouse.MIRRORS: "reflection",
            SigilHouse.ECHOES: "resonance"
        }
        return styles.get(self.house, "neutral")
    
    def _initialize_operations(self) -> Dict[str, callable]:
        """Initialize house-specific operations"""
        base_operations = {
            "process_sigil": self._process_sigil,
            "validate_compatibility": self._validate_compatibility,
            "apply_house_effects": self._apply_house_effects
        }
        
        # Add house-specific operations
        if self.house == SigilHouse.MEMORY:
            base_operations.update({
                "recall_pattern": self._memory_recall,
                "archive_sigil": self._memory_archive,
                "rebloom_event": self._memory_rebloom
            })
        elif self.house == SigilHouse.PURIFICATION:
            base_operations.update({
                "purge_corruption": self._purification_purge,
                "transform_pattern": self._purification_transform,
                "soot_to_ash": self._purification_transmute
            })
        elif self.house == SigilHouse.WEAVING:
            base_operations.update({
                "weave_connections": self._weaving_connect,
                "reinforce_threads": self._weaving_reinforce,
                "detect_tension": self._weaving_tension
            })
        elif self.house == SigilHouse.FLAME:
            base_operations.update({
                "ignite_pattern": self._flame_ignite,
                "release_pressure": self._flame_release,
                "modulate_entropy": self._flame_entropy
            })
        elif self.house == SigilHouse.MIRRORS:
            base_operations.update({
                "reflect_pattern": self._mirrors_reflect,
                "audit_coherence": self._mirrors_audit,
                "trace_origins": self._mirrors_trace
            })
        elif self.house == SigilHouse.ECHOES:
            base_operations.update({
                "resonate_frequency": self._echoes_resonate,
                "amplify_signal": self._echoes_amplify,
                "harmonize_patterns": self._echoes_harmonize
            })
        
        return base_operations
    
    def process_invocation(self, invocation: SigilInvocation) -> Dict[str, Any]:
        """Process a sigil invocation in this house"""
        with self.lock:
            start_time = time.time()
            self.stats.total_invocations += 1
            self.stats.current_load += 1
            
            try:
                # Validate compatibility
                if not self._validate_compatibility(invocation):
                    self.stats.failed_operations += 1
                    return {
                        "success": False,
                        "error": "incompatible_sigil_type",
                        "house": self.house.value
                    }
                
                # Process the sigil
                result = self._process_sigil(invocation)
                
                # Apply house-specific effects
                house_effects = self._apply_house_effects(invocation, result)
                result.update(house_effects)
                
                # Update statistics
                processing_time = time.time() - start_time
                self._update_stats(processing_time, True)
                
                self.stats.successful_operations += 1
                
                return result
                
            except Exception as e:
                self.stats.failed_operations += 1
                processing_time = time.time() - start_time
                self._update_stats(processing_time, False)
                
                log_anomaly(
                    "HOUSE_PROCESSING_ERROR",
                    f"Error in {self.house.value} house: {e}",
                    AnomalySeverity.ERROR
                )
                
                return {
                    "success": False,
                    "error": str(e),
                    "house": self.house.value
                }
            
            finally:
                self.stats.current_load = max(0, self.stats.current_load - 1)
    
    def _validate_compatibility(self, invocation: SigilInvocation) -> bool:
        """Validate if this invocation is compatible with this house"""
        # Check if we have a sigil in the forge for this symbol
        sigil = None
        for existing_sigil in sigil_forge.sigils.values():
            if existing_sigil.sigil_id == invocation.sigil_symbol or invocation.sigil_symbol in existing_sigil.sigil_id:
                sigil = existing_sigil
                break
        
        if sigil and sigil.sigil_type not in self.accepted_sigil_types:
            return False
        
        # Mythic grammar validation - ensure symbolic consistency
        if not self._validate_mythic_grammar(invocation):
            return False
        
        # House-specific validation
        if self.house == SigilHouse.MEMORY and "recall" not in invocation.sigil_symbol and "memory" not in invocation.sigil_symbol:
            if not any(keyword in invocation.parameters for keyword in ["memory", "recall", "archive"]):
                return False
        
        return True
    
    def _validate_mythic_grammar(self, invocation: SigilInvocation) -> bool:
        """Validate mythic grammar consistency and symbolic integrity"""
        # Mythic namespace enforcement - each house accepts only its glyph family
        house_glyph_families = {
            SigilHouse.MEMORY: ['recall', 'archive', 'rebloom', 'juliet', 'fractal', 'bloom'],
            SigilHouse.PURIFICATION: ['purge', 'crystallize', 'purify', 'soot', 'ash', 'decay'],
            SigilHouse.WEAVING: ['weave', 'thread', 'persephone', 'descent', 'return', 'stitch'],
            SigilHouse.FLAME: ['ignite', 'release', 'temper', 'pressure', 'flame', 'forge'],
            SigilHouse.MIRRORS: ['reflect', 'audit', 'mirror', 'coordinate', 'shi', 'health'],
            SigilHouse.ECHOES: ['modulate', 'amplify', 'resonate', 'voice', 'echo', 'auditory']
        }
        
        allowed_glyphs = house_glyph_families.get(self.house, [])
        sigil_symbol = invocation.sigil_symbol.lower()
        
        # Check if sigil symbol contains allowed glyph elements
        if not any(glyph in sigil_symbol for glyph in allowed_glyphs):
            logger.warning(f"Mythic grammar violation: {sigil_symbol} not compatible with {self.house.value} house")
            return False
        
        # Cross-house compatibility checking
        if not self._check_cross_house_compatibility(invocation):
            return False
        
        # Archetypal consistency validation
        if not self._validate_archetypal_consistency(invocation):
            return False
        
        return True
    
    def _check_cross_house_compatibility(self, invocation: SigilInvocation) -> bool:
        """Check cross-house compatibility for stacked operations"""
        # Compatible house combinations (as per mythic documentation)
        compatible_combinations = {
            SigilHouse.PURIFICATION: [SigilHouse.WEAVING],  # Purification → Weaving allowed
            SigilHouse.MEMORY: [SigilHouse.WEAVING, SigilHouse.MIRRORS],  # Memory → Weaving/Mirrors
            SigilHouse.WEAVING: [SigilHouse.MEMORY, SigilHouse.MIRRORS],  # Weaving → Memory/Mirrors
            SigilHouse.FLAME: [],  # Flame operations are typically standalone
            SigilHouse.MIRRORS: [SigilHouse.MEMORY, SigilHouse.WEAVING, SigilHouse.ECHOES],
            SigilHouse.ECHOES: [SigilHouse.MIRRORS]
        }
        
        # Conflicting combinations that should be blocked
        conflicting_combinations = {
            SigilHouse.FLAME: [SigilHouse.FLAME],  # No Flame ignite + Flame extinguish
            SigilHouse.PURIFICATION: [SigilHouse.MEMORY],  # Avoid purge during recall
        }
        
        # Check for conflicts in stacked operations
        stacked_houses = invocation.parameters.get('stacked_houses', [])
        for stacked_house in stacked_houses:
            if stacked_house in conflicting_combinations.get(self.house, []):
                logger.warning(f"Cross-house conflict: {self.house.value} conflicts with {stacked_house}")
                return False
        
        return True
    
    def _validate_archetypal_consistency(self, invocation: SigilInvocation) -> bool:
        """Validate that sigil aligns with house archetype"""
        # House archetypal requirements
        archetypal_requirements = {
            SigilHouse.MEMORY: {
                'required_metaphors': ['garden', 'flower', 'bloom', 'fractal'],
                'forbidden_metaphors': ['industrial', 'pollution', 'decay']
            },
            SigilHouse.PURIFICATION: {
                'required_metaphors': ['volcanic', 'crystallize', 'fire'],
                'forbidden_metaphors': ['stagnation', 'corruption']
            },
            SigilHouse.WEAVING: {
                'required_metaphors': ['thread', 'persephone', 'seasonal', 'depth'],
                'forbidden_metaphors': ['severing', 'isolation']
            },
            SigilHouse.FLAME: {
                'required_metaphors': ['forge', 'volcanic', 'pressure', 'ignition'],
                'forbidden_metaphors': ['extinction', 'cooling']
            },
            SigilHouse.MIRRORS: {
                'required_metaphors': ['reflection', 'wisdom', 'athena', 'clarity'],
                'forbidden_metaphors': ['blindness', 'confusion']
            },
            SigilHouse.ECHOES: {
                'required_metaphors': ['resonance', 'voice', 'harmony', 'amplification'],
                'forbidden_metaphors': ['silence', 'discord']
            }
        }
        
        requirements = archetypal_requirements.get(self.house, {})
        sigil_context = invocation.parameters.get('context', '').lower()
        sigil_symbol = invocation.sigil_symbol.lower()
        combined_text = f"{sigil_symbol} {sigil_context}"
        
        # Check for required metaphors
        required_metaphors = requirements.get('required_metaphors', [])
        if required_metaphors and not any(metaphor in combined_text for metaphor in required_metaphors):
            # Allow if it's a basic house operation
            basic_operations = ['process', 'execute', 'invoke']
            if not any(op in sigil_symbol for op in basic_operations):
                logger.debug(f"Archetypal consistency: {sigil_symbol} missing required metaphors for {self.house.value}")
                # Don't fail validation, just log - some operations may be metaphor-neutral
        
        # Check for forbidden metaphors
        forbidden_metaphors = requirements.get('forbidden_metaphors', [])
        if any(metaphor in combined_text for metaphor in forbidden_metaphors):
            logger.warning(f"Archetypal violation: {sigil_symbol} contains forbidden metaphor for {self.house.value}")
            return False
        
        return True
    
    def _process_sigil(self, invocation: SigilInvocation) -> Dict[str, Any]:
        """Core sigil processing logic"""
        result = {
            "success": True,
            "house": self.house.value,
            "sigil_processed": invocation.sigil_symbol,
            "processing_style": self.processing_style,
            "resonance_frequency": self.resonance_frequency
        }
        
        # Call house-specific operation if available
        operation_name = f"{invocation.sigil_symbol.split('_')[0]}_operation"
        if operation_name in self.operation_registry:
            operation_result = self.operation_registry[operation_name](invocation)
            result.update(operation_result)
        
        return result
    
    def _apply_house_effects(self, invocation: SigilInvocation, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply house-specific effects to the processing result"""
        effects = {
            "house_resonance": self._calculate_resonance(invocation),
            "frequency_alignment": self._check_frequency_alignment(invocation),
            "house_amplification": self._calculate_amplification()
        }
        
        # House-specific effect modifications
        if self.house == SigilHouse.FLAME:
            effects["thermal_increase"] = random.uniform(0.1, 0.3)
            effects["pressure_modulation"] = invocation.parameters.get("pressure_level", 0.5)
        
        elif self.house == SigilHouse.MIRRORS:
            effects["reflection_clarity"] = self.stats.health_score
            effects["audit_depth"] = min(7, invocation.priority)
        
        elif self.house == SigilHouse.WEAVING:
            effects["connection_strength"] = self.stats.resonance_level
            effects["thread_tension"] = self._measure_network_tension()
        
        return effects
    
    def _calculate_resonance(self, invocation: SigilInvocation) -> float:
        """Calculate resonance with house frequency"""
        # Simulate resonance based on sigil compatibility and house state
        base_resonance = 0.5
        
        if self.stats.health_score > 0.8:
            base_resonance += 0.2
        
        if invocation.priority > 7:
            base_resonance += 0.1
        
        # Frequency-based resonance
        symbol_frequency = hash(invocation.sigil_symbol) % 1000
        frequency_match = 1.0 - abs(symbol_frequency - self.resonance_frequency) / 1000.0
        
        return min(1.0, base_resonance * frequency_match)
    
    def _check_frequency_alignment(self, invocation: SigilInvocation) -> bool:
        """Check if invocation aligns with house frequency"""
        symbol_hash = hash(invocation.sigil_symbol)
        return (symbol_hash % 100) < (self.resonance_frequency % 100)
    
    def _calculate_amplification(self) -> float:
        """Calculate current house amplification factor"""
        load_factor = 1.0 - self.stats.get_load_percentage()
        health_factor = self.stats.health_score
        resonance_factor = self.stats.resonance_level
        
        return (load_factor + health_factor + resonance_factor) / 3.0
    
    def _measure_network_tension(self) -> float:
        """Measure tension in the sigil network (specific to Weaving house)"""
        # Simulate network tension measurement
        return random.uniform(0.1, 0.9)
    
    def _update_stats(self, processing_time: float, success: bool):
        """Update house statistics"""
        # Update average processing time
        total_ops = self.stats.successful_operations + self.stats.failed_operations
        if total_ops > 0:
            self.stats.average_processing_time = (
                (self.stats.average_processing_time * (total_ops - 1) + processing_time) / total_ops
            )
        
        # Update health score
        if success:
            self.stats.health_score = min(1.0, self.stats.health_score + 0.01)
        else:
            self.stats.health_score = max(0.0, self.stats.health_score - 0.05)
        
        # Update resonance level based on performance
        success_rate = self.stats.successful_operations / max(1, self.stats.total_invocations)
        self.stats.resonance_level = success_rate * self.stats.health_score
    
    # House-specific operation implementations
    def _memory_recall(self, invocation: SigilInvocation) -> Dict[str, Any]:
        """Memory house: recall operation"""
        recall_depth = invocation.parameters.get("depth", 3)
        return {
            "operation": "memory_recall",
            "recall_depth": recall_depth,
            "patterns_recalled": random.randint(1, recall_depth * 2),
            "memory_clarity": self.stats.health_score
        }
    
    def _memory_archive(self, invocation: SigilInvocation) -> Dict[str, Any]:
        """Memory house: archive operation"""
        return {
            "operation": "memory_archive",
            "archive_success": self.stats.health_score > 0.5,
            "storage_efficiency": self.stats.health_score
        }
    
    def _memory_rebloom(self, invocation: SigilInvocation) -> Dict[str, Any]:
        """Memory house: rebloom operation"""
        bloom_intensity = invocation.parameters.get("intensity", 0.5)
        return {
            "operation": "memory_rebloom",
            "bloom_intensity": bloom_intensity,
            "emotional_resonance": bloom_intensity * self.stats.resonance_level
        }
    
    def _purification_purge(self, invocation: SigilInvocation) -> Dict[str, Any]:
        """Purification house: purge operation"""
        purge_strength = invocation.parameters.get("strength", 0.7)
        return {
            "operation": "purification_purge",
            "purge_strength": purge_strength,
            "corruption_removed": purge_strength * 0.8,
            "purity_level": min(1.0, self.stats.health_score + 0.1)
        }
    
    def _weaving_connect(self, invocation: SigilInvocation) -> Dict[str, Any]:
        """Weaving house: connection operation"""
        connection_targets = invocation.parameters.get("targets", [])
        return {
            "operation": "weaving_connect",
            "connections_made": len(connection_targets),
            "thread_strength": self.stats.resonance_level,
            "network_integration": min(1.0, len(connection_targets) * 0.2)
        }
    
    def _flame_ignite(self, invocation: SigilInvocation) -> Dict[str, Any]:
        """Flame house: ignition operation"""
        ignition_power = invocation.parameters.get("power", 0.6)
        return {
            "operation": "flame_ignite",
            "ignition_power": ignition_power,
            "thermal_output": ignition_power * 1.2,
            "entropy_generated": ignition_power * 0.3
        }
    
    def _mirrors_reflect(self, invocation: SigilInvocation) -> Dict[str, Any]:
        """Mirrors house: reflection operation"""
        reflection_depth = invocation.parameters.get("depth", 2)
        return {
            "operation": "mirrors_reflect",
            "reflection_depth": reflection_depth,
            "clarity_score": self.stats.health_score,
            "patterns_revealed": reflection_depth * 2
        }
    
    def _echoes_resonate(self, invocation: SigilInvocation) -> Dict[str, Any]:
        """Echoes house: resonance operation"""
        resonance_frequency = invocation.parameters.get("frequency", self.resonance_frequency)
        return {
            "operation": "echoes_resonate",
            "resonance_frequency": resonance_frequency,
            "harmonic_strength": self.stats.resonance_level,
            "echo_duration": resonance_frequency / 100.0
        }

class SigilRouter:
    """Manages routing of sigil invocations to appropriate houses"""
    
    def __init__(self):
        self.routing_table: Dict[str, SigilHouse] = {}
        self.semantic_mappings = self._initialize_semantic_mappings()
        self.adaptive_weights: Dict[SigilHouse, float] = {house: 1.0 for house in SigilHouse}
        self.routing_history = deque(maxlen=1000)
        
    def _initialize_semantic_mappings(self) -> Dict[str, SigilHouse]:
        """Initialize semantic keyword to house mappings"""
        return {
            # Memory keywords
            "recall": SigilHouse.MEMORY,
            "memory": SigilHouse.MEMORY,
            "remember": SigilHouse.MEMORY,
            "archive": SigilHouse.MEMORY,
            "rebloom": SigilHouse.MEMORY,
            
            # Purification keywords
            "purge": SigilHouse.PURIFICATION,
            "cleanse": SigilHouse.PURIFICATION,
            "purify": SigilHouse.PURIFICATION,
            "soot": SigilHouse.PURIFICATION,
            "ash": SigilHouse.PURIFICATION,
            
            # Weaving keywords
            "weave": SigilHouse.WEAVING,
            "connect": SigilHouse.WEAVING,
            "thread": SigilHouse.WEAVING,
            "bind": SigilHouse.WEAVING,
            "reinforce": SigilHouse.WEAVING,
            
            # Flame keywords
            "ignite": SigilHouse.FLAME,
            "flame": SigilHouse.FLAME,
            "burn": SigilHouse.FLAME,
            "pressure": SigilHouse.FLAME,
            "release": SigilHouse.FLAME,
            
            # Mirrors keywords
            "reflect": SigilHouse.MIRRORS,
            "mirror": SigilHouse.MIRRORS,
            "audit": SigilHouse.MIRRORS,
            "trace": SigilHouse.MIRRORS,
            "observe": SigilHouse.MIRRORS,
            
            # Echoes keywords
            "echo": SigilHouse.ECHOES,
            "resonate": SigilHouse.ECHOES,
            "harmonize": SigilHouse.ECHOES,
            "amplify": SigilHouse.ECHOES,
            "voice": SigilHouse.ECHOES
        }
    
    def route_invocation(self, invocation: SigilInvocation) -> SigilHouse:
        """Route an invocation to the appropriate house"""
        if invocation.routing_protocol == RoutingProtocol.DIRECT:
            target_house = invocation.house
        
        elif invocation.routing_protocol == RoutingProtocol.SEMANTIC:
            target_house = self._semantic_routing(invocation)
        
        elif invocation.routing_protocol == RoutingProtocol.ADAPTIVE:
            target_house = self._adaptive_routing(invocation)
        
        elif invocation.routing_protocol == RoutingProtocol.EMERGENCY:
            target_house = self._emergency_routing(invocation)
        
        else:
            target_house = invocation.house  # Fallback to specified house
        
        # Record routing decision
        self.routing_history.append({
            "timestamp": time.time(),
            "sigil": invocation.sigil_symbol,
            "protocol": invocation.routing_protocol.value,
            "target_house": target_house.value,
            "original_house": invocation.house.value
        })
        
        return target_house
    
    def _semantic_routing(self, invocation: SigilInvocation) -> SigilHouse:
        """Route based on semantic analysis of the sigil"""
        # Check sigil symbol for keywords
        for keyword, house in self.semantic_mappings.items():
            if keyword in invocation.sigil_symbol.lower():
                return house
        
        # Check parameters for keywords
        param_text = " ".join(str(v) for v in invocation.parameters.values()).lower()
        for keyword, house in self.semantic_mappings.items():
            if keyword in param_text:
                return house
        
        # Fallback to specified house
        return invocation.house
    
    def _adaptive_routing(self, invocation: SigilInvocation) -> SigilHouse:
        """Route based on current network state and load balancing"""
        # Find house with lowest load and highest efficiency
        best_house = invocation.house
        best_score = 0.0
        
        for house in SigilHouse:
            load_factor = 1.0  # Would get actual load from house operators
            efficiency_factor = self.adaptive_weights[house]
            score = efficiency_factor * load_factor
            
            if score > best_score:
                best_score = score
                best_house = house
        
        return best_house
    
    def _emergency_routing(self, invocation: SigilInvocation) -> SigilHouse:
        """Route for emergency situations"""
        # Emergency routing prioritizes most stable houses
        emergency_priority = [
            SigilHouse.MIRRORS,  # Most stable for auditing
            SigilHouse.MEMORY,   # Preserve state
            SigilHouse.PURIFICATION,  # Clean up issues
            SigilHouse.WEAVING,  # Maintain connections
            SigilHouse.ECHOES,   # Communication
            SigilHouse.FLAME     # Last resort
        ]
        
        return emergency_priority[0]  # Return most stable
    
    def update_adaptive_weights(self, house: SigilHouse, performance_score: float):
        """Update adaptive routing weights based on performance"""
        # Exponential moving average
        alpha = 0.1
        self.adaptive_weights[house] = (
            alpha * performance_score + (1 - alpha) * self.adaptive_weights[house]
        )

class SigilNetwork:
    """
    Enhanced Sigil Network with Houses, Routing, and Recursive Integration
    """
    
    def __init__(self):
        # Core components
        self.houses: Dict[SigilHouse, SigilHouseOperator] = {}
        self.router = SigilRouter()
        self.network_state = NetworkState.DORMANT
        
        # Initialize houses
        for house in SigilHouse:
            self.houses[house] = SigilHouseOperator(house)
        
        # Network management
        self.active_invocations: deque = deque(maxlen=100)
        self.invocation_history: deque = deque(maxlen=1000)
        self.network_lock = threading.Lock()
        
        # Network metrics
        self.total_invocations = 0
        self.successful_operations = 0
        self.network_coherence = 0.5
        self.resonance_matrix: Dict[Tuple[SigilHouse, SigilHouse], float] = {}
        
        # Safety and monitoring
        self.overload_threshold = 50
        self.emergency_protocols_active = False
        
        # Initialize resonance matrix
        self._initialize_resonance_matrix()
        
        # Register with schema registry
        self._register()
        
        logger.info("🕸️ Enhanced Sigil Network initialized with all houses")
        
        # Start network
        self.network_state = NetworkState.ACTIVE
    
    def _register(self):
        """Register with schema registry"""
        registry.register(
            component_id="schema.sigil_network",
            name="Enhanced Sigil Network",
            component_type=ComponentType.MODULE,
            instance=self,
            capabilities=[
                "sigil_routing",
                "house_operations", 
                "network_resonance",
                "load_balancing",
                "emergency_protocols"
            ],
            version="2.0.0"
        )
    
    def _initialize_resonance_matrix(self):
        """Initialize house-to-house resonance relationships"""
        # Define natural resonances between houses
        resonance_pairs = [
            (SigilHouse.MEMORY, SigilHouse.MIRRORS, 0.8),    # Memory and reflection
            (SigilHouse.WEAVING, SigilHouse.ECHOES, 0.7),    # Connection and resonance
            (SigilHouse.FLAME, SigilHouse.PURIFICATION, 0.6), # Transformation processes
            (SigilHouse.MIRRORS, SigilHouse.ECHOES, 0.5),    # Observation and response
            (SigilHouse.MEMORY, SigilHouse.WEAVING, 0.4),    # Memory and connections
            (SigilHouse.PURIFICATION, SigilHouse.MIRRORS, 0.4) # Cleansing and reflection
        ]
        
        for house1, house2, resonance in resonance_pairs:
            self.resonance_matrix[(house1, house2)] = resonance
            self.resonance_matrix[(house2, house1)] = resonance  # Symmetric
    
    def invoke_sigil(self, sigil_symbol: str, house: SigilHouse, 
                    parameters: Optional[Dict[str, Any]] = None,
                    invoker: str = "network",
                    routing_protocol: RoutingProtocol = RoutingProtocol.SEMANTIC,
                    priority: int = 5) -> Dict[str, Any]:
        """
        Invoke a sigil through the network
        
        Args:
            sigil_symbol: The sigil to invoke
            house: Target house (may be overridden by routing)
            parameters: Additional parameters for the sigil
            invoker: Who/what is invoking this sigil
            routing_protocol: How to route this invocation
            priority: Priority level (1-10)
            
        Returns:
            Result of the sigil invocation
        """
        with self.network_lock:
            if self.network_state == NetworkState.EMERGENCY and priority < 8:
                return {
                    "success": False,
                    "error": "network_in_emergency_mode",
                    "required_priority": 8
                }
            
            # Create invocation
            invocation = SigilInvocation(
                sigil_symbol=sigil_symbol,
                house=house,
                parameters=parameters or {},
                invoker=invoker,
                tick_id=int(time.time() * 1000) % 1000000,
                priority=priority,
                routing_protocol=routing_protocol
            )
            
            # Route to appropriate house
            target_house = self.router.route_invocation(invocation)
            
            # Check for overload
            if len(self.active_invocations) >= self.overload_threshold:
                self._handle_network_overload()
                if priority < 7:  # Only high priority gets through during overload
                    return {
                        "success": False,
                        "error": "network_overloaded",
                        "queue_length": len(self.active_invocations)
                    }
            
            # Process invocation
            self.active_invocations.append(invocation)
            self.total_invocations += 1
            
            try:
                # Process through target house
                house_result = self.houses[target_house].process_invocation(invocation)
                
                # Apply network effects
                network_effects = self._apply_network_effects(invocation, house_result, target_house)
                house_result.update(network_effects)
                
                # Update network state
                self._update_network_state(house_result)
                
                # Record success
                if house_result.get("success", False):
                    self.successful_operations += 1
                
                # Store in history
                self.invocation_history.append({
                    "invocation": invocation.to_dict(),
                    "result": house_result,
                    "target_house": target_house.value,
                    "processing_time": time.time() - invocation.timestamp
                })
                
                return house_result
                
            except Exception as e:
                log_anomaly(
                    "NETWORK_INVOCATION_ERROR",
                    f"Network error processing {sigil_symbol}: {e}",
                    AnomalySeverity.ERROR
                )
                return {
                    "success": False,
                    "error": f"network_error: {e}",
                    "sigil": sigil_symbol
                }
            
            finally:
                # Remove from active invocations
                if invocation in self.active_invocations:
                    self.active_invocations.remove(invocation)
    
    def _apply_network_effects(self, invocation: SigilInvocation, house_result: Dict[str, Any], 
                             target_house: SigilHouse) -> Dict[str, Any]:
        """Apply network-wide effects to house processing results"""
        effects = {
            "network_coherence": self.network_coherence,
            "cross_house_resonance": self._calculate_cross_house_resonance(target_house),
            "network_amplification": self._calculate_network_amplification(),
            "routing_efficiency": self._calculate_routing_efficiency(invocation, target_house)
        }
        
        # Check for resonance cascades
        if effects["cross_house_resonance"] > 0.8:
            effects["resonance_cascade"] = self._trigger_resonance_cascade(target_house)
        
        return effects
    
    def _calculate_cross_house_resonance(self, active_house: SigilHouse) -> float:
        """Calculate resonance effects from other houses"""
        total_resonance = 0.0
        resonance_count = 0
        
        for other_house in SigilHouse:
            if other_house != active_house:
                resonance_key = (active_house, other_house)
                if resonance_key in self.resonance_matrix:
                    # Weight by other house's current state
                    other_house_health = self.houses[other_house].stats.health_score
                    resonance = self.resonance_matrix[resonance_key] * other_house_health
                    total_resonance += resonance
                    resonance_count += 1
        
        return total_resonance / resonance_count if resonance_count > 0 else 0.0
    
    def _calculate_network_amplification(self) -> float:
        """Calculate overall network amplification factor"""
        # Based on all houses' health and coherence
        total_health = sum(house.stats.health_score for house in self.houses.values())
        average_health = total_health / len(self.houses)
        
        return average_health * self.network_coherence
    
    def _calculate_routing_efficiency(self, invocation: SigilInvocation, target_house: SigilHouse) -> float:
        """Calculate how efficiently this invocation was routed"""
        # Perfect efficiency if routed to originally intended house
        if target_house == invocation.house:
            return 1.0
        
        # Check if semantic routing found a better match
        if invocation.routing_protocol == RoutingProtocol.SEMANTIC:
            return 0.8  # Good efficiency for semantic routing
        
        # Adaptive routing efficiency based on current performance
        return self.router.adaptive_weights.get(target_house, 0.5)
    
    def _trigger_resonance_cascade(self, origin_house: SigilHouse) -> Dict[str, Any]:
        """Trigger a resonance cascade from high coherence"""
        cascade_effects = {
            "origin_house": origin_house.value,
            "affected_houses": [],
            "resonance_boost": 0.2,
            "network_coherence_increase": 0.1
        }
        
        # Boost resonance in connected houses
        for house in SigilHouse:
            if house != origin_house:
                resonance_key = (origin_house, house)
                if resonance_key in self.resonance_matrix and self.resonance_matrix[resonance_key] > 0.6:
                    self.houses[house].stats.resonance_level = min(1.0, 
                        self.houses[house].stats.resonance_level + cascade_effects["resonance_boost"])
                    cascade_effects["affected_houses"].append(house.value)
        
        # Boost network coherence
        self.network_coherence = min(1.0, 
            self.network_coherence + cascade_effects["network_coherence_increase"])
        
        log_anomaly(
            "RESONANCE_CASCADE",
            f"Resonance cascade from {origin_house.value} affecting {len(cascade_effects['affected_houses'])} houses",
            AnomalySeverity.INFO
        )
        
        return cascade_effects
    
    def _update_network_state(self, processing_result: Dict[str, Any]):
        """Update overall network state based on processing results"""
        # Update coherence based on success rate
        if processing_result.get("success", False):
            self.network_coherence = min(1.0, self.network_coherence + 0.01)
        else:
            self.network_coherence = max(0.0, self.network_coherence - 0.02)
        
        # Check for state transitions
        if self.network_coherence > 0.9 and self.network_state == NetworkState.ACTIVE:
            self.network_state = NetworkState.RESONATING
            self._emit_network_state_change("resonating")
        
        elif self.network_coherence < 0.3 and self.network_state == NetworkState.ACTIVE:
            self.network_state = NetworkState.EMERGENCY
            self.emergency_protocols_active = True
            self._emit_network_state_change("emergency")
        
        # Check for transcendent state
        avg_house_resonance = sum(h.stats.resonance_level for h in self.houses.values()) / len(self.houses)
        if avg_house_resonance > 0.95 and self.network_coherence > 0.95:
            self.network_state = NetworkState.TRANSCENDENT
            self._emit_network_state_change("transcendent")
    
    def _handle_network_overload(self):
        """Handle network overload situation"""
        if self.network_state != NetworkState.OVERLOADED:
            self.network_state = NetworkState.OVERLOADED
            log_anomaly(
                "NETWORK_OVERLOAD",
                f"Network overloaded with {len(self.active_invocations)} active invocations",
                AnomalySeverity.WARNING
            )
    
    def _emit_network_state_change(self, new_state: str):
        """Emit signal for network state change"""
        emit_signal(
            SignalType.CONSCIOUSNESS,
            "sigil_network",
            {
                "event": "network_state_change",
                "new_state": new_state,
                "coherence": self.network_coherence,
                "timestamp": time.time()
            }
        )
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        house_stats = {}
        for house, operator in self.houses.items():
            house_stats[house.value] = {
                "health_score": operator.stats.health_score,
                "resonance_level": operator.stats.resonance_level,
                "current_load": operator.stats.current_load,
                "total_invocations": operator.stats.total_invocations,
                "success_rate": operator.stats.successful_operations / max(1, operator.stats.total_invocations)
            }
        
        return {
            "network_state": self.network_state.value,
            "network_coherence": self.network_coherence,
            "total_invocations": self.total_invocations,
            "success_rate": self.successful_operations / max(1, self.total_invocations),
            "active_invocations": len(self.active_invocations),
            "house_statistics": house_stats,
            "emergency_protocols_active": self.emergency_protocols_active,
            "resonance_matrix_size": len(self.resonance_matrix)
        }
    
    def emergency_reset(self):
        """Emergency reset of the network"""
        with self.network_lock:
            self.active_invocations.clear()
            self.network_state = NetworkState.ACTIVE
            self.emergency_protocols_active = False
            self.network_coherence = 0.5
            
            # Reset house stats to safe defaults
            for house_operator in self.houses.values():
                house_operator.stats.current_load = 0
                house_operator.stats.health_score = max(0.5, house_operator.stats.health_score)
            
            log_anomaly(
                "NETWORK_EMERGENCY_RESET",
                "Sigil network emergency reset completed",
                AnomalySeverity.WARNING
            )

# Global sigil network instance
sigil_network = SigilNetwork()

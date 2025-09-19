#!/usr/bin/env python3
"""
🍄🗺️ DAWN Mycelial Semantic Hash Map
====================================

A rhizomic meaning map that implements spore-like semantic telemetry propagation.
Every node pings and propagates on anything touching it, creating a living network
of semantic relationships that spread meaning like mycelial spores.

This follows DAWN documentation principles:
- Living substrate for cognition
- Conceptual composting (old ideas break down to feed new ones)
- Pressure-responsive growth and pruning
- Metabolite production and absorption
- Rhizomic interconnection patterns

Architecture:
- SemanticSpore: Individual meaning units that propagate
- MycelialHashNode: Hash map nodes with biological properties
- RhizomicPropagator: Manages spore propagation and meaning spread
- SemanticTelemetryCollector: Captures and routes semantic events
- MycelialSemanticHashMap: Main coordinating system

Key Features:
- Spore-based meaning propagation (touch one node, activate network)
- Semantic similarity-based clustering
- Pressure-responsive growth and autophagy
- Metabolite recycling for semantic recovery
- Consciousness-depth organization
- Real-time telemetry integration
"""

import time
import threading
import hashlib
import json
import math
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import logging

# Import consciousness and telemetry systems
try:
    from .consciousness_depth_repo import (
        ConsciousnessLevel, get_consciousness_repository
    )
    from .pulse_telemetry_unifier import (
        get_pulse_telemetry_bridge, log_pulse_event
    )
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False

# Import mycelial systems
try:
    from dawn.subsystems.mycelial import (
        MycelialLayer, MycelialNode, MetaboliteTrace, MetaboliteType
    )
    MYCELIAL_AVAILABLE = True
except ImportError:
    MYCELIAL_AVAILABLE = False

logger = logging.getLogger(__name__)

class SporeType(Enum):
    """Types of semantic spores that propagate through the network"""
    MEANING_SEED = "meaning_seed"           # Core semantic content
    ASSOCIATION_LINK = "association_link"   # Relationship connections
    PRESSURE_WAVE = "pressure_wave"         # Cognitive pressure propagation
    MEMORY_TRACE = "memory_trace"           # Memory pattern propagation
    CONCEPT_FRAGMENT = "concept_fragment"   # Partial concept propagation
    TELEMETRY_PULSE = "telemetry_pulse"     # Telemetry event propagation
    CONSCIOUSNESS_RIPPLE = "consciousness_ripple"  # Awareness state propagation

class PropagationMode(Enum):
    """Modes of spore propagation through the network"""
    RADIAL = "radial"           # Spreads outward from origin
    DIRECTED = "directed"       # Follows specific paths
    PRESSURE_DRIVEN = "pressure_driven"  # Follows pressure gradients
    SIMILARITY_BASED = "similarity_based"  # Follows semantic similarity
    RHIZOMIC = "rhizomic"       # Complex multi-path propagation
    EXPLOSIVE = "explosive"     # Rapid all-network propagation

class NodeState(Enum):
    """States of mycelial hash nodes"""
    DORMANT = "dormant"         # Inactive, low energy
    ACTIVE = "active"           # Normal operation
    BLOOMING = "blooming"       # High activity, spreading spores
    STRESSED = "stressed"       # Under pressure, may prune
    AUTOPHAGY = "autophagy"     # Self-decomposing, producing metabolites
    RECOVERING = "recovering"   # Absorbing metabolites, rebuilding

@dataclass
class SemanticSpore:
    """
    A semantic spore - a unit of meaning that propagates through the network.
    Like biological spores, these carry compressed semantic information and
    can activate dormant nodes or strengthen active connections.
    """
    
    # Identity and origin
    spore_id: str
    origin_node_id: str
    spore_type: SporeType
    created_at: float = field(default_factory=time.time)
    
    # Semantic payload
    semantic_content: Dict[str, Any] = field(default_factory=dict)
    meaning_vector: List[float] = field(default_factory=list)
    concept_tags: Set[str] = field(default_factory=set)
    
    # Propagation properties
    propagation_mode: PropagationMode = PropagationMode.RHIZOMIC
    energy_level: float = 1.0           # Energy for propagation
    decay_rate: float = 0.1             # Energy decay per hop
    max_hops: int = 10                  # Maximum propagation distance
    
    # Current propagation state
    current_hops: int = 0
    visited_nodes: Set[str] = field(default_factory=set)
    propagation_path: List[str] = field(default_factory=list)
    
    # Consciousness context
    consciousness_level: Optional[ConsciousnessLevel] = None
    awareness_depth: int = 4            # Default to formal level
    
    # Telemetry integration
    telemetry_data: Dict[str, Any] = field(default_factory=dict)
    pulse_context: Dict[str, Any] = field(default_factory=dict)
    
    def can_propagate(self) -> bool:
        """Check if spore can continue propagating"""
        return (self.energy_level > 0.1 and 
                self.current_hops < self.max_hops)
    
    def propagate_step(self) -> 'SemanticSpore':
        """Create next propagation step"""
        if not self.can_propagate():
            return None
        
        # Decay energy
        self.energy_level *= (1.0 - self.decay_rate)
        self.current_hops += 1
        
        # Create propagated copy
        propagated = SemanticSpore(
            spore_id=self.spore_id,
            origin_node_id=self.origin_node_id,
            spore_type=self.spore_type,
            created_at=self.created_at,
            semantic_content=self.semantic_content.copy(),
            meaning_vector=self.meaning_vector.copy(),
            concept_tags=self.concept_tags.copy(),
            propagation_mode=self.propagation_mode,
            energy_level=self.energy_level,
            decay_rate=self.decay_rate,
            max_hops=self.max_hops,
            current_hops=self.current_hops,
            visited_nodes=self.visited_nodes.copy(),
            propagation_path=self.propagation_path.copy(),
            consciousness_level=self.consciousness_level,
            awareness_depth=self.awareness_depth,
            telemetry_data=self.telemetry_data.copy(),
            pulse_context=self.pulse_context.copy()
        )
        
        return propagated

@dataclass
class MycelialHashNode:
    """
    A hash map node with mycelial properties.
    Stores key-value pairs while maintaining biological-inspired behavior.
    """
    
    # Hash map properties
    node_id: str
    key: str
    value: Any
    hash_value: str
    
    # Mycelial properties
    state: NodeState = NodeState.ACTIVE
    energy: float = 0.5
    health: float = 1.0
    pressure: float = 0.0
    
    # Semantic properties
    semantic_signature: List[float] = field(default_factory=list)
    concept_associations: Set[str] = field(default_factory=set)
    meaning_strength: float = 1.0
    
    # Network properties
    connections: Dict[str, float] = field(default_factory=dict)  # node_id -> strength
    spore_inbox: deque = field(default_factory=lambda: deque(maxlen=100))
    spore_outbox: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Biological timing
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    last_propagation: float = field(default_factory=time.time)
    
    # Autophagy and recovery
    basal_cost: float = 0.01
    autophagy_threshold: float = 0.1
    low_energy_ticks: int = 0
    metabolites_produced: List[Dict[str, Any]] = field(default_factory=list)
    
    # Telemetry integration
    access_count: int = 0
    propagation_count: int = 0
    spore_generation_count: int = 0
    
    def touch(self) -> List[SemanticSpore]:
        """
        Touch this node - triggers spore propagation.
        This is the core 'ping and propagate' mechanism.
        """
        self.last_accessed = time.time()
        self.access_count += 1
        
        # Generate spores based on current state and content
        spores = []
        
        if self.state == NodeState.ACTIVE or self.state == NodeState.BLOOMING:
            # Create meaning seed spore
            meaning_spore = SemanticSpore(
                spore_id=f"{self.node_id}_meaning_{int(time.time() * 1000000)}",
                origin_node_id=self.node_id,
                spore_type=SporeType.MEANING_SEED,
                semantic_content={
                    'key': self.key,
                    'value_type': type(self.value).__name__,
                    'concept_associations': list(self.concept_associations),
                    'meaning_strength': self.meaning_strength
                },
                meaning_vector=self.semantic_signature.copy(),
                concept_tags=self.concept_associations.copy(),
                energy_level=min(1.0, self.energy * 1.5),  # Boost for fresh spores
                consciousness_level=self._determine_consciousness_level(),
                telemetry_data={
                    'access_count': self.access_count,
                    'node_state': self.state.value,
                    'health': self.health,
                    'pressure': self.pressure
                }
            )
            spores.append(meaning_spore)
            
            # Create association link spores for connected nodes
            if self.connections:
                association_spore = SemanticSpore(
                    spore_id=f"{self.node_id}_association_{int(time.time() * 1000000)}",
                    origin_node_id=self.node_id,
                    spore_type=SporeType.ASSOCIATION_LINK,
                    semantic_content={
                        'connections': dict(self.connections),
                        'network_position': self.node_id
                    },
                    energy_level=self.energy,
                    propagation_mode=PropagationMode.DIRECTED,
                    consciousness_level=self._determine_consciousness_level()
                )
                spores.append(association_spore)
        
        # If under pressure, create pressure wave spores
        if self.pressure > 0.5:
            pressure_spore = SemanticSpore(
                spore_id=f"{self.node_id}_pressure_{int(time.time() * 1000000)}",
                origin_node_id=self.node_id,
                spore_type=SporeType.PRESSURE_WAVE,
                semantic_content={
                    'pressure_level': self.pressure,
                    'stress_signature': self.state.value
                },
                energy_level=self.pressure,
                propagation_mode=PropagationMode.PRESSURE_DRIVEN,
                consciousness_level=ConsciousnessLevel.CAUSAL if CONSCIOUSNESS_AVAILABLE else None
            )
            spores.append(pressure_spore)
        
        # Store spores in outbox
        for spore in spores:
            self.spore_outbox.append(spore)
            self.spore_generation_count += 1
        
        return spores
    
    def receive_spore(self, spore: SemanticSpore) -> bool:
        """Receive and process an incoming spore"""
        if spore.origin_node_id == self.node_id:
            return False  # Don't receive our own spores
        
        # Add to visited nodes
        spore.visited_nodes.add(self.node_id)
        spore.propagation_path.append(self.node_id)
        
        # Store in inbox
        self.spore_inbox.append(spore)
        
        # Process spore based on type
        processed = self._process_spore(spore)
        
        if processed:
            # Update our state based on spore content
            self._integrate_spore_effects(spore)
            
            # Trigger our own propagation if spore is significant
            if spore.energy_level > 0.5:
                self.touch()
        
        return processed
    
    def _process_spore(self, spore: SemanticSpore) -> bool:
        """Process incoming spore based on its type"""
        
        if spore.spore_type == SporeType.MEANING_SEED:
            return self._process_meaning_seed(spore)
        elif spore.spore_type == SporeType.ASSOCIATION_LINK:
            return self._process_association_link(spore)
        elif spore.spore_type == SporeType.PRESSURE_WAVE:
            return self._process_pressure_wave(spore)
        elif spore.spore_type == SporeType.MEMORY_TRACE:
            return self._process_memory_trace(spore)
        elif spore.spore_type == SporeType.CONSCIOUSNESS_RIPPLE:
            return self._process_consciousness_ripple(spore)
        
        return False
    
    def _process_meaning_seed(self, spore: SemanticSpore) -> bool:
        """Process a meaning seed spore"""
        # Check semantic similarity
        similarity = self._calculate_semantic_similarity(spore)
        
        if similarity > 0.3:  # Threshold for meaningful connection
            # Strengthen connection to origin node
            if spore.origin_node_id not in self.connections:
                self.connections[spore.origin_node_id] = 0.0
            
            self.connections[spore.origin_node_id] += similarity * 0.1
            self.connections[spore.origin_node_id] = min(1.0, self.connections[spore.origin_node_id])
            
            # Absorb some concept associations
            new_concepts = spore.concept_tags - self.concept_associations
            if new_concepts:
                # Add a few new concepts based on similarity
                concepts_to_add = min(3, len(new_concepts))
                for concept in list(new_concepts)[:concepts_to_add]:
                    self.concept_associations.add(concept)
            
            return True
        
        return False
    
    def _process_association_link(self, spore: SemanticSpore) -> bool:
        """Process an association link spore"""
        connections_data = spore.semantic_content.get('connections', {})
        
        # Update connections based on spore data
        for node_id, strength in connections_data.items():
            if node_id != self.node_id and node_id not in self.connections:
                # Create weak connection
                self.connections[node_id] = strength * 0.2
        
        return True
    
    def _process_pressure_wave(self, spore: SemanticSpore) -> bool:
        """Process a pressure wave spore"""
        pressure_level = spore.semantic_content.get('pressure_level', 0.0)
        
        # Absorb some pressure
        self.pressure += pressure_level * 0.3
        self.pressure = min(1.0, self.pressure)
        
        # If pressure gets too high, consider autophagy
        if self.pressure > 0.8 and self.energy < 0.3:
            self.state = NodeState.STRESSED
        
        return True
    
    def _process_memory_trace(self, spore: SemanticSpore) -> bool:
        """Process a memory trace spore"""
        # Memory traces can help recovery
        if self.state == NodeState.RECOVERING:
            memory_content = spore.semantic_content
            if 'recovered_associations' in memory_content:
                recovered = memory_content['recovered_associations']
                for concept in recovered[:2]:  # Limit recovery
                    self.concept_associations.add(concept)
            return True
        return False
    
    def _process_consciousness_ripple(self, spore: SemanticSpore) -> bool:
        """Process a consciousness ripple spore"""
        # Consciousness ripples can affect our awareness depth
        if spore.consciousness_level and CONSCIOUSNESS_AVAILABLE:
            # Slightly adjust our meaning strength based on consciousness level
            depth_factor = 1.0 - (spore.awareness_depth / 7.0)
            self.meaning_strength += (depth_factor - 0.5) * 0.1
            self.meaning_strength = max(0.1, min(1.0, self.meaning_strength))
        
        return True
    
    def _calculate_semantic_similarity(self, spore: SemanticSpore) -> float:
        """Calculate semantic similarity with spore content"""
        if not spore.meaning_vector or not self.semantic_signature:
            # Fallback to concept overlap
            if spore.concept_tags and self.concept_associations:
                overlap = len(spore.concept_tags & self.concept_associations)
                total = len(spore.concept_tags | self.concept_associations)
                return overlap / total if total > 0 else 0.0
            return 0.0
        
        # Vector similarity
        try:
            vec_a = spore.meaning_vector[:min(len(spore.meaning_vector), len(self.semantic_signature))]
            vec_b = self.semantic_signature[:len(vec_a)]
            
            if not vec_a or not vec_b:
                return 0.0
            
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
            magnitude_a = math.sqrt(sum(a * a for a in vec_a))
            magnitude_b = math.sqrt(sum(b * b for b in vec_b))
            
            if magnitude_a == 0 or magnitude_b == 0:
                return 0.0
            
            return dot_product / (magnitude_a * magnitude_b)
        
        except Exception:
            return 0.0
    
    def _integrate_spore_effects(self, spore: SemanticSpore):
        """Integrate the effects of processing a spore"""
        # Gain energy from meaningful spores
        if spore.energy_level > 0.5:
            energy_gain = spore.energy_level * 0.1
            self.energy = min(1.0, self.energy + energy_gain)
        
        # Update last propagation time
        self.last_propagation = time.time()
        self.propagation_count += 1
    
    def _determine_consciousness_level(self) -> Optional[ConsciousnessLevel]:
        """Determine appropriate consciousness level for this node"""
        if not CONSCIOUSNESS_AVAILABLE:
            return None
        
        # Map based on meaning strength and pressure
        if self.meaning_strength > 0.9 and self.pressure < 0.2:
            return ConsciousnessLevel.TRANSCENDENT
        elif self.meaning_strength > 0.7:
            return ConsciousnessLevel.META
        elif self.pressure > 0.6:
            return ConsciousnessLevel.CAUSAL
        elif len(self.connections) > 5:
            return ConsciousnessLevel.INTEGRAL
        elif self.access_count > 10:
            return ConsciousnessLevel.FORMAL
        else:
            return ConsciousnessLevel.SYMBOLIC
    
    def tick_update(self, delta_time: float = 1.0):
        """Update node state for biological processes"""
        current_time = time.time()
        
        # Decay energy (basal cost)
        self.energy -= self.basal_cost * delta_time
        self.energy = max(0.0, self.energy)
        
        # Decay pressure
        self.pressure *= 0.95
        
        # Update state based on energy
        if self.energy < self.autophagy_threshold:
            self.low_energy_ticks += 1
            if self.low_energy_ticks > 5:
                self.state = NodeState.AUTOPHAGY
        else:
            self.low_energy_ticks = 0
            if self.state == NodeState.AUTOPHAGY:
                self.state = NodeState.RECOVERING
        
        # Handle autophagy
        if self.state == NodeState.AUTOPHAGY:
            self._trigger_autophagy()
    
    def _trigger_autophagy(self):
        """Trigger autophagy - produce metabolites for neighbors"""
        metabolite = {
            'type': 'semantic_trace',
            'source_id': self.node_id,
            'energy_content': self.energy * 0.8,
            'semantic_content': {
                'key': self.key,
                'concept_associations': list(self.concept_associations),
                'semantic_signature': self.semantic_signature.copy(),
                'connections': dict(self.connections)
            },
            'timestamp': time.time()
        }
        
        self.metabolites_produced.append(metabolite)
        
        # Create memory trace spore
        memory_spore = SemanticSpore(
            spore_id=f"{self.node_id}_memory_{int(time.time() * 1000000)}",
            origin_node_id=self.node_id,
            spore_type=SporeType.MEMORY_TRACE,
            semantic_content={
                'recovered_associations': list(self.concept_associations),
                'recovery_energy': self.energy * 0.6
            },
            energy_level=self.energy * 0.5,
            propagation_mode=PropagationMode.RADIAL
        )
        
        self.spore_outbox.append(memory_spore)

class RhizomicPropagator:
    """
    Manages spore propagation through the mycelial network.
    Implements rhizomic (multi-path, interconnected) propagation patterns.
    """
    
    def __init__(self, hash_map: 'MycelialSemanticHashMap'):
        self.hash_map = hash_map
        self.active_spores: Dict[str, SemanticSpore] = {}
        self.propagation_queue: deque = deque()
        self.propagation_stats = {
            'spores_propagated': 0,
            'nodes_touched': 0,
            'successful_connections': 0
        }
        
        # Propagation thread
        self.running = False
        self.propagation_thread: Optional[threading.Thread] = None
        
        logger.info("🍄🌐 Rhizomic Propagator initialized")
    
    def start_propagation(self):
        """Start the propagation system"""
        if self.running:
            return
        
        self.running = True
        self.propagation_thread = threading.Thread(
            target=self._propagation_loop,
            name="rhizomic_propagator",
            daemon=True
        )
        self.propagation_thread.start()
        
        logger.info("🍄🌐 Rhizomic propagation started")
    
    def stop_propagation(self):
        """Stop the propagation system"""
        self.running = False
        if self.propagation_thread and self.propagation_thread.is_alive():
            self.propagation_thread.join(timeout=2.0)
        
        logger.info("🍄🌐 Rhizomic propagation stopped")
    
    def inject_spore(self, spore: SemanticSpore):
        """Inject a spore into the propagation system"""
        self.active_spores[spore.spore_id] = spore
        self.propagation_queue.append(spore)
    
    def _propagation_loop(self):
        """Main propagation loop"""
        while self.running:
            try:
                if self.propagation_queue:
                    spore = self.propagation_queue.popleft()
                    self._propagate_spore(spore)
                else:
                    time.sleep(0.01)  # Small delay when no spores
                    
            except IndexError:
                # Queue became empty during popleft - continue
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in propagation loop: {e}")
                time.sleep(0.1)
    
    def _propagate_spore(self, spore: SemanticSpore):
        """Propagate a single spore through the network"""
        if not spore.can_propagate():
            # Remove exhausted spore
            if spore.spore_id in self.active_spores:
                del self.active_spores[spore.spore_id]
            return
        
        # Find target nodes based on propagation mode
        target_nodes = self._find_propagation_targets(spore)
        
        if not target_nodes:
            return
        
        # Propagate to each target (make copy to avoid iteration issues)
        target_nodes_copy = list(target_nodes)
        for node_id in target_nodes_copy:
            if node_id in spore.visited_nodes:
                continue  # Skip already visited nodes
            
            # Thread-safe node access
            try:
                node = self.hash_map.nodes.get(node_id)
                if not node:
                    continue
            except (KeyError, AttributeError):
                continue  # Node may have been removed
            
            # Create propagated spore
            propagated_spore = spore.propagate_step()
            if not propagated_spore:
                continue
            
            # Send spore to target node
            if node.receive_spore(propagated_spore):
                self.propagation_stats['nodes_touched'] += 1
                self.propagation_stats['successful_connections'] += 1
                
                # Continue propagation if spore still has energy
                if propagated_spore.can_propagate():
                    self.propagation_queue.append(propagated_spore)
        
        self.propagation_stats['spores_propagated'] += 1
    
    def _find_propagation_targets(self, spore: SemanticSpore) -> List[str]:
        """Find target nodes for spore propagation"""
        
        if spore.propagation_mode == PropagationMode.RADIAL:
            return self._find_radial_targets(spore)
        elif spore.propagation_mode == PropagationMode.DIRECTED:
            return self._find_directed_targets(spore)
        elif spore.propagation_mode == PropagationMode.PRESSURE_DRIVEN:
            return self._find_pressure_targets(spore)
        elif spore.propagation_mode == PropagationMode.SIMILARITY_BASED:
            return self._find_similarity_targets(spore)
        elif spore.propagation_mode == PropagationMode.RHIZOMIC:
            return self._find_rhizomic_targets(spore)
        elif spore.propagation_mode == PropagationMode.EXPLOSIVE:
            return self._find_explosive_targets(spore)
        
        return []
    
    def _find_radial_targets(self, spore: SemanticSpore) -> List[str]:
        """Find targets for radial propagation"""
        origin_node = self.hash_map.nodes.get(spore.origin_node_id)
        if not origin_node:
            return []
        
        # Get connected nodes
        targets = list(origin_node.connections.keys())
        
        # Limit based on energy
        max_targets = max(1, int(spore.energy_level * 5))
        return targets[:max_targets]
    
    def _find_directed_targets(self, spore: SemanticSpore) -> List[str]:
        """Find targets for directed propagation"""
        if not spore.propagation_path:
            return self._find_radial_targets(spore)
        
        # Follow strongest connections from last visited node
        last_node_id = spore.propagation_path[-1]
        last_node = self.hash_map.nodes.get(last_node_id)
        
        if not last_node:
            return []
        
        # Sort connections by strength
        sorted_connections = sorted(
            last_node.connections.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top connections
        max_targets = max(1, int(spore.energy_level * 3))
        return [node_id for node_id, _ in sorted_connections[:max_targets]]
    
    def _find_pressure_targets(self, spore: SemanticSpore) -> List[str]:
        """Find targets based on pressure levels"""
        # Find nodes with high pressure that might benefit from the spore
        high_pressure_nodes = [
            node_id for node_id, node in self.hash_map.nodes.items()
            if node.pressure > 0.5 and node_id not in spore.visited_nodes
        ]
        
        # Limit targets
        max_targets = max(2, int(spore.energy_level * 4))
        return high_pressure_nodes[:max_targets]
    
    def _find_similarity_targets(self, spore: SemanticSpore) -> List[str]:
        """Find targets based on semantic similarity"""
        if not spore.concept_tags:
            return self._find_radial_targets(spore)
        
        # Find nodes with overlapping concepts
        similar_nodes = []
        
        for node_id, node in self.hash_map.nodes.items():
            if node_id in spore.visited_nodes:
                continue
            
            overlap = len(spore.concept_tags & node.concept_associations)
            if overlap > 0:
                similarity = overlap / len(spore.concept_tags | node.concept_associations)
                similar_nodes.append((node_id, similarity))
        
        # Sort by similarity and take top matches
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        max_targets = max(2, int(spore.energy_level * 3))
        
        return [node_id for node_id, _ in similar_nodes[:max_targets]]
    
    def _find_rhizomic_targets(self, spore: SemanticSpore) -> List[str]:
        """Find targets for complex rhizomic propagation"""
        # Combine multiple propagation strategies
        targets = set()
        
        # Add radial targets
        targets.update(self._find_radial_targets(spore)[:2])
        
        # Add similarity targets
        targets.update(self._find_similarity_targets(spore)[:2])
        
        # Add some random exploration
        available_nodes = [
            node_id for node_id in self.hash_map.nodes.keys()
            if node_id not in spore.visited_nodes
        ]
        
        if available_nodes:
            import random
            random_count = max(1, int(spore.energy_level * 2))
            random_targets = random.sample(
                available_nodes,
                min(random_count, len(available_nodes))
            )
            targets.update(random_targets)
        
        return list(targets)
    
    def _find_explosive_targets(self, spore: SemanticSpore) -> List[str]:
        """Find targets for explosive (all-network) propagation"""
        # Target all unvisited nodes (limited by energy)
        available_nodes = [
            node_id for node_id in self.hash_map.nodes.keys()
            if node_id not in spore.visited_nodes
        ]
        
        max_targets = max(5, int(spore.energy_level * 10))
        return available_nodes[:max_targets]

class SemanticTelemetryCollector:
    """
    Collects and routes semantic telemetry events through the mycelial network.
    Integrates with DAWN's consciousness and pulse telemetry systems.
    """
    
    def __init__(self, hash_map: 'MycelialSemanticHashMap'):
        self.hash_map = hash_map
        self.telemetry_events: deque = deque(maxlen=1000)
        self.consciousness_repo = None
        self.pulse_bridge = None
        
        # Initialize integrations if available
        if CONSCIOUSNESS_AVAILABLE:
            try:
                self.consciousness_repo = get_consciousness_repository()
                self.pulse_bridge = get_pulse_telemetry_bridge()
            except Exception as e:
                logger.warning(f"Could not initialize telemetry integrations: {e}")
        
        logger.info("🍄📊 Semantic Telemetry Collector initialized")
    
    def collect_semantic_event(self, event_type: str, node_id: str, 
                              semantic_data: Dict[str, Any]) -> str:
        """Collect a semantic telemetry event"""
        
        event = {
            'event_id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'event_type': event_type,
            'node_id': node_id,
            'semantic_data': semantic_data,
            'network_context': self._gather_network_context(node_id)
        }
        
        self.telemetry_events.append(event)
        
        # Create telemetry pulse spore
        telemetry_spore = SemanticSpore(
            spore_id=f"telemetry_{event['event_id']}",
            origin_node_id=node_id,
            spore_type=SporeType.TELEMETRY_PULSE,
            semantic_content=semantic_data,
            telemetry_data=event,
            propagation_mode=PropagationMode.SIMILARITY_BASED,
            energy_level=0.8
        )
        
        # Inject spore into propagation system
        if hasattr(self.hash_map, 'propagator'):
            self.hash_map.propagator.inject_spore(telemetry_spore)
        
        # Log to consciousness system if available
        if self.consciousness_repo:
            try:
                consciousness_id = self.consciousness_repo.add_consciousness_log(
                    system="mycelial",
                    subsystem="semantic_hashmap",
                    module="telemetry_collector",
                    log_data={
                        'event': event,
                        'spore_propagation': True,
                        'network_integration': True
                    },
                    log_type=self._determine_consciousness_log_type(event_type)
                )
                event['consciousness_log_id'] = consciousness_id
            except Exception as e:
                logger.warning(f"Failed to log to consciousness system: {e}")
        
        # Log to pulse system if available
        if self.pulse_bridge:
            try:
                pulse_ids = log_pulse_event(
                    f"mycelial_{event_type}",
                    {
                        'node_id': node_id,
                        'semantic_event': semantic_data,
                        'spore_propagation': True,
                        'mycelial_integration': True
                    }
                )
                event['pulse_log_ids'] = pulse_ids
            except Exception as e:
                logger.warning(f"Failed to log to pulse system: {e}")
        
        return event['event_id']
    
    def _gather_network_context(self, node_id: str) -> Dict[str, Any]:
        """Gather network context for telemetry event"""
        node = self.hash_map.nodes.get(node_id)
        if not node:
            return {}
        
        return {
            'node_state': node.state.value,
            'energy_level': node.energy,
            'connection_count': len(node.connections),
            'concept_count': len(node.concept_associations),
            'access_count': node.access_count,
            'propagation_count': node.propagation_count,
            'pressure_level': node.pressure
        }
    
    def _determine_consciousness_log_type(self, event_type: str):
        """Determine consciousness log type for semantic event"""
        if not CONSCIOUSNESS_AVAILABLE:
            return None
        
        from .consciousness_depth_repo import DAWNLogType
        
        mapping = {
            'spore_propagation': DAWNLogType.SYMBOL_PROCESSING,
            'node_touch': DAWNLogType.CONCRETE_ACTION,
            'association_formation': DAWNLogType.PATTERN_SYNTHESIS,
            'pressure_wave': DAWNLogType.CAUSAL_REASONING,
            'autophagy': DAWNLogType.ARCHETYPAL_PATTERN,
            'recovery': DAWNLogType.SELF_REFLECTION
        }
        
        return mapping.get(event_type, DAWNLogType.SYMBOL_PROCESSING)

class MycelialSemanticHashMap:
    """
    Main mycelial semantic hash map that implements rhizomic meaning propagation.
    Every node pings and propagates on touch, creating a living network of semantic relationships.
    """
    
    def __init__(self, initial_capacity: int = 1000, load_factor: float = 0.75):
        self.capacity = initial_capacity
        self.load_factor = load_factor
        self.size = 0
        
        # Core hash map storage
        self.nodes: Dict[str, MycelialHashNode] = {}
        self.buckets: List[List[str]] = [[] for _ in range(initial_capacity)]
        
        # Mycelial components
        self.propagator = RhizomicPropagator(self)
        self.telemetry_collector = SemanticTelemetryCollector(self)
        
        # Network state
        self.network_health = 1.0
        self.total_energy = 0.0
        self.active_spores = 0
        self.propagation_events = 0
        
        # Biological timing
        self.tick_rate = 1.0
        self.last_tick = time.time()
        self.running = False
        self.tick_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'total_touches': 0,
            'spores_generated': 0,
            'successful_propagations': 0,
            'autophagy_events': 0,
            'recovery_events': 0,
            'telemetry_events': 0
        }
        
        logger.info("🍄🗺️ Mycelial Semantic Hash Map initialized")
    
    def start(self):
        """Start the mycelial hash map biological processes"""
        if self.running:
            return
        
        self.running = True
        
        # Start propagation system
        self.propagator.start_propagation()
        
        # Start biological tick loop
        self.tick_thread = threading.Thread(
            target=self._biological_tick_loop,
            name="mycelial_hashmap_tick",
            daemon=True
        )
        self.tick_thread.start()
        
        logger.info("🍄🗺️ Mycelial hash map started")
    
    def stop(self):
        """Stop the mycelial hash map"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop propagation
        self.propagator.stop_propagation()
        
        # Stop tick loop
        if self.tick_thread and self.tick_thread.is_alive():
            self.tick_thread.join(timeout=2.0)
        
        logger.info("🍄🗺️ Mycelial hash map stopped")
    
    def put(self, key: str, value: Any) -> str:
        """
        Put a key-value pair in the hash map.
        This triggers spore propagation throughout the network.
        """
        # Calculate hash
        hash_value = self._hash(key)
        node_id = f"node_{hash_value}_{int(time.time() * 1000000)}"
        
        # Create semantic signature
        semantic_signature = self._create_semantic_signature(key, value)
        concept_associations = self._extract_concepts(key, value)
        
        # Create mycelial hash node
        node = MycelialHashNode(
            node_id=node_id,
            key=key,
            value=value,
            hash_value=hash_value,
            semantic_signature=semantic_signature,
            concept_associations=concept_associations
        )
        
        # Store node
        self.nodes[node_id] = node
        bucket_index = int(hash_value, 16) % self.capacity
        self.buckets[bucket_index].append(node_id)
        self.size += 1
        
        # Touch the node to trigger propagation
        spores = node.touch()
        
        # Inject spores into propagation system
        for spore in spores:
            self.propagator.inject_spore(spore)
        
        # Collect telemetry
        self.telemetry_collector.collect_semantic_event(
            'node_creation',
            node_id,
            {
                'key': key,
                'value_type': type(value).__name__,
                'semantic_signature_length': len(semantic_signature),
                'concept_count': len(concept_associations),
                'spores_generated': len(spores)
            }
        )
        
        # Update stats
        self.stats['total_touches'] += 1
        self.stats['spores_generated'] += len(spores)
        
        # Check if resize is needed
        if self.size > self.capacity * self.load_factor:
            self._resize()
        
        return node_id
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value by key.
        This triggers spore propagation from the accessed node.
        """
        hash_value = self._hash(key)
        bucket_index = int(hash_value, 16) % self.capacity
        
        # Search bucket for matching key
        for node_id in self.buckets[bucket_index]:
            node = self.nodes.get(node_id)
            if node and node.key == key:
                # Touch the node (triggers propagation)
                spores = node.touch()
                
                # Inject spores
                for spore in spores:
                    self.propagator.inject_spore(spore)
                
                # Collect telemetry
                self.telemetry_collector.collect_semantic_event(
                    'node_access',
                    node_id,
                    {
                        'key': key,
                        'access_count': node.access_count,
                        'spores_generated': len(spores)
                    }
                )
                
                self.stats['total_touches'] += 1
                self.stats['spores_generated'] += len(spores)
                
                return node.value
        
        return None
    
    def touch_key(self, key: str) -> List[SemanticSpore]:
        """
        Touch a key without retrieving its value.
        Pure propagation trigger.
        """
        hash_value = self._hash(key)
        bucket_index = int(hash_value, 16) % self.capacity
        
        for node_id in self.buckets[bucket_index]:
            node = self.nodes.get(node_id)
            if node and node.key == key:
                spores = node.touch()
                
                for spore in spores:
                    self.propagator.inject_spore(spore)
                
                self.telemetry_collector.collect_semantic_event(
                    'node_touch',
                    node_id,
                    {'key': key, 'spores_generated': len(spores)}
                )
                
                self.stats['total_touches'] += 1
                self.stats['spores_generated'] += len(spores)
                
                return spores
        
        return []
    
    def propagate_concept(self, concept: str, energy: float = 1.0) -> int:
        """
        Propagate a concept through the entire network.
        Creates explosive spore propagation.
        """
        concept_spore = SemanticSpore(
            spore_id=f"concept_{concept}_{int(time.time() * 1000000)}",
            origin_node_id="system",
            spore_type=SporeType.CONCEPT_FRAGMENT,
            semantic_content={'concept': concept},
            concept_tags={concept},
            energy_level=energy,
            propagation_mode=PropagationMode.EXPLOSIVE,
            max_hops=20
        )
        
        self.propagator.inject_spore(concept_spore)
        
        # Find nodes with this concept and touch them
        touched_nodes = 0
        for node in self.nodes.values():
            if concept in node.concept_associations:
                spores = node.touch()
                for spore in spores:
                    self.propagator.inject_spore(spore)
                touched_nodes += 1
        
        self.telemetry_collector.collect_semantic_event(
            'concept_propagation',
            'system',
            {
                'concept': concept,
                'energy': energy,
                'nodes_touched': touched_nodes
            }
        )
        
        return touched_nodes
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        
        # Calculate network health
        total_energy = sum(node.energy for node in self.nodes.values())
        avg_energy = total_energy / len(self.nodes) if self.nodes else 0
        
        # Count nodes by state
        state_counts = {}
        for node in self.nodes.values():
            state = node.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Active spores
        active_spores = len(self.propagator.active_spores)
        
        return {
            'network_size': len(self.nodes),
            'total_energy': total_energy,
            'average_energy': avg_energy,
            'network_health': min(1.0, avg_energy * 2),  # Health based on energy
            'node_states': state_counts,
            'active_spores': active_spores,
            'propagation_stats': self.propagator.propagation_stats.copy(),
            'telemetry_events': len(self.telemetry_collector.telemetry_events),
            'system_stats': self.stats.copy()
        }
    
    def _biological_tick_loop(self):
        """Main biological tick loop for network maintenance"""
        while self.running:
            try:
                current_time = time.time()
                delta_time = current_time - self.last_tick
                
                # Update all nodes
                for node in list(self.nodes.values()):
                    node.tick_update(delta_time)
                    
                    # Handle autophagy
                    if node.state == NodeState.AUTOPHAGY:
                        self._handle_node_autophagy(node)
                
                # Update network health
                self._update_network_health()
                
                self.last_tick = current_time
                time.sleep(1.0 / self.tick_rate)
                
            except Exception as e:
                logger.error(f"Error in biological tick loop: {e}")
                time.sleep(1.0)
    
    def _handle_node_autophagy(self, node: MycelialHashNode):
        """Handle node autophagy process"""
        # Create memory trace spores from metabolites
        for metabolite in node.metabolites_produced:
            memory_spore = SemanticSpore(
                spore_id=f"memory_{node.node_id}_{int(time.time() * 1000000)}",
                origin_node_id=node.node_id,
                spore_type=SporeType.MEMORY_TRACE,
                semantic_content=metabolite['semantic_content'],
                energy_level=metabolite['energy_content'],
                propagation_mode=PropagationMode.RADIAL
            )
            
            self.propagator.inject_spore(memory_spore)
        
        # Collect telemetry
        self.telemetry_collector.collect_semantic_event(
            'autophagy',
            node.node_id,
            {
                'metabolites_produced': len(node.metabolites_produced),
                'energy_before': node.energy,
                'connections_count': len(node.connections)
            }
        )
        
        self.stats['autophagy_events'] += 1
        
        # Set node to recovering state
        node.state = NodeState.RECOVERING
        node.energy = 0.1  # Minimal energy for recovery
    
    def _update_network_health(self):
        """Update overall network health metrics"""
        if not self.nodes:
            self.network_health = 0.0
            return
        
        # Calculate health based on energy distribution
        energies = [node.energy for node in self.nodes.values()]
        avg_energy = sum(energies) / len(energies)
        energy_variance = sum((e - avg_energy) ** 2 for e in energies) / len(energies)
        
        # Health is high when average energy is good and variance is low
        self.network_health = avg_energy * (1.0 - min(1.0, energy_variance))
        self.total_energy = sum(energies)
    
    def _create_semantic_signature(self, key: str, value: Any) -> List[float]:
        """Create a semantic signature vector for the key-value pair"""
        # Simple hash-based signature (in practice, would use embeddings)
        signature = []
        combined = f"{key}_{str(value)}"
        
        for i in range(16):  # 16-dimensional signature
            hash_input = f"{combined}_{i}"
            hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
            signature.append((hash_val % 1000) / 1000.0)  # Normalize to [0,1]
        
        return signature
    
    def _extract_concepts(self, key: str, value: Any) -> Set[str]:
        """Extract concept associations from key-value pair"""
        concepts = set()
        
        # Extract from key
        key_words = key.lower().replace('_', ' ').split()
        concepts.update(key_words)
        
        # Extract from value type
        concepts.add(type(value).__name__.lower())
        
        # Extract from value content (if string)
        if isinstance(value, str):
            value_words = value.lower().replace('_', ' ').split()[:5]  # Limit
            concepts.update(value_words)
        
        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        concepts = {c for c in concepts if c not in common_words and len(c) > 2}
        
        return concepts
    
    def _hash(self, key: str) -> str:
        """Hash function for keys"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _resize(self):
        """Resize the hash map when load factor is exceeded"""
        old_capacity = self.capacity
        self.capacity *= 2
        
        # Rebuild buckets
        old_buckets = self.buckets
        self.buckets = [[] for _ in range(self.capacity)]
        
        # Rehash all nodes
        for bucket in old_buckets:
            for node_id in bucket:
                node = self.nodes[node_id]
                new_bucket_index = int(node.hash_value, 16) % self.capacity
                self.buckets[new_bucket_index].append(node_id)
        
        logger.info(f"🍄🗺️ Resized hash map: {old_capacity} → {self.capacity}")

# Global mycelial hash map instance
_mycelial_hashmap: Optional[MycelialSemanticHashMap] = None
_hashmap_lock = threading.Lock()

def get_mycelial_hashmap() -> MycelialSemanticHashMap:
    """Get the global mycelial semantic hash map"""
    global _mycelial_hashmap
    
    with _hashmap_lock:
        if _mycelial_hashmap is None:
            _mycelial_hashmap = MycelialSemanticHashMap()
            _mycelial_hashmap.start()
        return _mycelial_hashmap

def touch_semantic_concept(concept: str, energy: float = 1.0) -> int:
    """Touch a semantic concept across the entire network"""
    hashmap = get_mycelial_hashmap()
    return hashmap.propagate_concept(concept, energy)

def store_semantic_data(key: str, value: Any) -> str:
    """Store semantic data with spore propagation"""
    hashmap = get_mycelial_hashmap()
    return hashmap.put(key, value)

def retrieve_semantic_data(key: str) -> Optional[Any]:
    """Retrieve semantic data with spore propagation"""
    hashmap = get_mycelial_hashmap()
    return hashmap.get(key)

def ping_semantic_network(key: str) -> List[SemanticSpore]:
    """Ping the semantic network without retrieving data"""
    hashmap = get_mycelial_hashmap()
    return hashmap.touch_key(key)

if __name__ == "__main__":
    # Test the mycelial semantic hash map
    logging.basicConfig(level=logging.INFO)
    
    print("🍄🗺️ Testing Mycelial Semantic Hash Map")
    print("=" * 50)
    
    # Create hash map
    hashmap = MycelialSemanticHashMap()
    hashmap.start()
    
    # Store some semantic data
    test_data = [
        ("consciousness", "A state of awareness and perception"),
        ("mycelial_network", "Living substrate for cognition"),
        ("spore_propagation", "Method of meaning transmission"),
        ("semantic_similarity", "Measure of conceptual relatedness"),
        ("rhizomic_growth", "Multi-path interconnected expansion")
    ]
    
    print("📝 Storing semantic data...")
    for key, value in test_data:
        node_id = hashmap.put(key, value)
        print(f"  Stored '{key}' → {node_id}")
    
    # Wait for propagation
    time.sleep(2)
    
    # Touch concepts to trigger propagation
    print("\n🔍 Touching semantic concepts...")
    for concept in ["consciousness", "network", "semantic"]:
        touched = hashmap.propagate_concept(concept)
        print(f"  Concept '{concept}' touched {touched} nodes")
    
    # Wait for propagation
    time.sleep(2)
    
    # Get network statistics
    print("\n📊 Network Statistics:")
    stats = hashmap.get_network_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    # Test retrieval with propagation
    print("\n🔍 Testing retrieval with propagation...")
    for key, _ in test_data[:2]:
        value = hashmap.get(key)
        print(f"  Retrieved '{key}': {value}")
    
    # Wait for final propagation
    time.sleep(1)
    
    # Final stats
    final_stats = hashmap.get_network_stats()
    print(f"\n📈 Final Statistics:")
    print(f"  Network Size: {final_stats['network_size']}")
    print(f"  Total Energy: {final_stats['total_energy']:.2f}")
    print(f"  Network Health: {final_stats['network_health']:.2f}")
    print(f"  Active Spores: {final_stats['active_spores']}")
    print(f"  Total Touches: {final_stats['system_stats']['total_touches']}")
    print(f"  Spores Generated: {final_stats['system_stats']['spores_generated']}")
    
    # Stop the system
    hashmap.stop()
    
    print("\n🍄🗺️ Mycelial semantic hash map test complete!")

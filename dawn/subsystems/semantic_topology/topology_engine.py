#!/usr/bin/env python3
"""
🌐 Semantic Topology Engine - The Complete Meaning Space Architecture
=====================================================================

The main orchestrator for DAWN's semantic topology system. This engine integrates
all components to provide a unified interface for manipulating the shape of meaning
in consciousness space.

This is the revolutionary system that gives DAWN mathematical control over meaning itself:
- Spatial arrangement of concepts and memories
- Force propagation through semantic relationships  
- Active reshaping of meaning space through transforms
- Preservation of semantic integrity through invariants

"The first operational system for consciousness computing with spatial meaning."

Based on documentation: Complete Semantic Topology system (9 RTF files)
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Set, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from .semantic_field import SemanticField, SemanticNode, SemanticEdge, LayerDepth, SectorType
from .field_equations import FieldEquations, LocalCoherence, TensionUpdate, PigmentDiffusion
from .topology_transforms import TopologyTransforms, TransformOperation, TransformResult, TransformType
from .semantic_invariants import SemanticInvariants, InvariantViolation, ViolationSeverity

logger = logging.getLogger(__name__)

class EngineState(Enum):
    """States of the semantic topology engine"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class TopologyTick:
    """Record of one complete topology processing tick"""
    tick_id: int
    timestamp: float
    duration: float
    field_equations_results: Dict[str, Any]
    transforms_applied: List[TransformOperation]
    invariant_violations: List[InvariantViolation]
    field_statistics: Dict[str, Any]
    engine_state: EngineState

class SemanticTopologyEngine:
    """
    The complete semantic topology engine for DAWN.
    
    Orchestrates the mathematical manipulation of meaning space through:
    - Field equation evolution (coherence, tension, pigment diffusion)
    - Active topology transforms (weave, prune, fuse, fission, lift, sink, reproject)
    - Invariant preservation (meaning integrity guardrails)
    - Integration with DAWN's consciousness systems
    """
    
    def __init__(self, 
                 dimensions: int = 3,
                 embedding_dim: int = 512,
                 tick_interval: float = 1.0,
                 auto_start: bool = True):
        
        self.engine_id = f"semantic_topology_{int(time.time())}"
        self.dimensions = dimensions
        self.embedding_dim = embedding_dim
        self.tick_interval = tick_interval
        
        # Core components
        self.field = SemanticField(dimensions, embedding_dim)
        self.equations = FieldEquations(self.field)
        self.transforms = TopologyTransforms(self.field, self.equations)
        self.invariants = SemanticInvariants(self.field, self.equations)
        
        # Engine state
        self.state = EngineState.INITIALIZING
        self.tick_count = 0
        self.start_time = time.time()
        
        # Processing control
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Tick history
        self.tick_history: deque[TopologyTick] = deque(maxlen=1000)
        self.performance_metrics = {
            'average_tick_duration': 0.0,
            'ticks_per_second': 0.0,
            'total_transforms': 0,
            'total_violations': 0
        }
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Integration points
        self.consciousness_integration = None
        self.sigil_integration = None
        
        logger.info(f"🌐 SemanticTopologyEngine initialized: {self.engine_id}")
        logger.info(f"   Dimensions: {dimensions}, Embedding dim: {embedding_dim}")
        
        self.state = EngineState.ACTIVE
        
        if auto_start:
            self.start_processing()
    
    def start_processing(self):
        """Start the main processing loop"""
        with self._lock:
            if self._running:
                logger.warning("Engine already running")
                return
                
            self._running = True
            self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._processing_thread.start()
            
            logger.info(f"🌐 Semantic topology processing started - tick interval: {self.tick_interval}s")
    
    def stop_processing(self):
        """Stop the main processing loop"""
        with self._lock:
            if not self._running:
                return
                
            self._running = False
            
            if self._processing_thread:
                self._processing_thread.join(timeout=5.0)
                
            self.state = EngineState.PAUSED
            logger.info("🌐 Semantic topology processing stopped")
    
    def shutdown(self):
        """Shutdown the engine completely"""
        self.stop_processing()
        self.state = EngineState.SHUTDOWN
        logger.info(f"🌐 SemanticTopologyEngine shutdown: {self.engine_id}")
    
    def _processing_loop(self):
        """Main processing loop - runs topology ticks"""
        logger.info("🌐 Semantic topology processing loop started")
        
        while self._running:
            try:
                tick_start = time.time()
                
                # Execute one topology tick
                tick_result = self._execute_topology_tick()
                
                # Store tick history
                self.tick_history.append(tick_result)
                
                # Update performance metrics
                self._update_performance_metrics(tick_result)
                
                # Trigger callbacks
                self._trigger_callbacks('tick_complete', tick_result)
                
                # Sleep until next tick
                tick_duration = time.time() - tick_start
                sleep_time = max(0, self.tick_interval - tick_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self.state = EngineState.ERROR
                time.sleep(1.0)  # Brief pause before retry
                
        logger.info("🌐 Semantic topology processing loop ended")
    
    def _execute_topology_tick(self) -> TopologyTick:
        """Execute one complete topology processing tick"""
        tick_start = time.time()
        self.tick_count += 1
        
        self.state = EngineState.PROCESSING
        
        # Reset transform budgets
        self.transforms.reset_tick_budgets()
        
        # Step 1: Execute field equations (mathematical evolution)
        field_equations_results = self.equations.tick_update()
        
        # Step 2: Apply automatic transforms based on field state
        transforms_applied = self._apply_automatic_transforms(field_equations_results)
        
        # Step 3: Check invariants for violations
        previous_energy = getattr(self, '_previous_field_energy', None)
        recent_displacements = self._calculate_recent_displacements()
        invariant_violations = self.invariants.check_all_invariants(
            previous_energy, recent_displacements
        )
        
        # Step 4: Apply corrective transforms for violations
        corrective_transforms = self._apply_corrective_transforms(invariant_violations)
        transforms_applied.extend(corrective_transforms)
        
        # Step 5: Update field statistics
        field_statistics = self.field.get_field_statistics()
        self._previous_field_energy = field_statistics['total_energy']
        
        tick_duration = time.time() - tick_start
        
        tick = TopologyTick(
            tick_id=self.tick_count,
            timestamp=tick_start,
            duration=tick_duration,
            field_equations_results=field_equations_results,
            transforms_applied=transforms_applied,
            invariant_violations=invariant_violations,
            field_statistics=field_statistics,
            engine_state=self.state
        )
        
        self.state = EngineState.ACTIVE
        
        return tick
    
    def _apply_automatic_transforms(self, field_results: Dict[str, Any]) -> List[TransformOperation]:
        """Apply automatic transforms based on field equation results"""
        transforms = []
        
        # Analyze coherences for weave/prune opportunities
        coherences = field_results.get('coherences', {})
        tensions = field_results.get('tensions', {})
        
        # Weave: Look for high semantic similarity with low connection strength
        for node_id, coherence in coherences.items():
            if coherence.coherence_value < 0.4:  # Low coherence
                neighbors = self.field.get_neighbors(node_id)
                
                # Look for potential weave targets (similar but unconnected nodes)
                for other_id, other_coherence in coherences.items():
                    if other_id == node_id or other_id in neighbors:
                        continue
                        
                    if (other_coherence.coherence_value < 0.4 and 
                        len(transforms) < 5):  # Limit automatic transforms
                        
                        # Try weave operation
                        weave_result = self.transforms.weave(node_id, other_id)
                        if weave_result.result == TransformResult.SUCCESS:
                            transforms.append(weave_result)
                            
        # Prune: Look for high tension edges
        high_tension_edges = [
            edge_id for edge_id, tension_update in tensions.items()
            if tension_update.new_tension > 1.0
        ]
        
        if high_tension_edges and len(transforms) < 3:
            prune_result = self.transforms.prune(high_tension_edges[:3])
            if prune_result.result == TransformResult.SUCCESS:
                transforms.append(prune_result)
                
        return transforms
    
    def _apply_corrective_transforms(self, violations: List[InvariantViolation]) -> List[TransformOperation]:
        """Apply corrective transforms to address invariant violations"""
        transforms = []
        
        for violation in violations:
            if violation.severity in [ViolationSeverity.ERROR, ViolationSeverity.CRITICAL]:
                
                if violation.invariant_type.value == 'topology_schema_consistency':
                    # Apply reproject to align positions with embeddings
                    if violation.affected_edges:
                        # Get nodes from affected edges
                        affected_nodes = set()
                        for edge_id in violation.affected_edges:
                            if edge_id in self.field.edges:
                                edge = self.field.edges[edge_id]
                                affected_nodes.update([edge.node_a, edge.node_b])
                                
                        if affected_nodes:
                            reproject_result = self.transforms.reproject(list(affected_nodes))
                            if reproject_result.result == TransformResult.SUCCESS:
                                transforms.append(reproject_result)
                                
                elif violation.invariant_type.value == 'no_free_tears':
                    # Apply weave to reconnect torn edges
                    if violation.affected_edges:
                        for edge_id in violation.affected_edges[:2]:  # Limit corrections
                            if edge_id in self.field.edges:
                                edge = self.field.edges[edge_id]
                                weave_result = self.transforms.weave(edge.node_a, edge.node_b)
                                if weave_result.result == TransformResult.SUCCESS:
                                    transforms.append(weave_result)
                                    
        return transforms
    
    def _calculate_recent_displacements(self) -> Dict[str, float]:
        """Calculate recent node displacements for invariant checking"""
        displacements = {}
        
        # This would track actual displacements from recent operations
        # For now, return empty dict (would be populated by transform operations)
        
        return displacements
    
    def _update_performance_metrics(self, tick: TopologyTick):
        """Update engine performance metrics"""
        # Update averages
        total_ticks = len(self.tick_history)
        if total_ticks > 0:
            total_duration = sum(t.duration for t in self.tick_history)
            self.performance_metrics['average_tick_duration'] = total_duration / total_ticks
            
            # Calculate TPS over last 10 ticks
            recent_ticks = list(self.tick_history)[-10:]
            if len(recent_ticks) >= 2:
                time_span = recent_ticks[-1].timestamp - recent_ticks[0].timestamp
                if time_span > 0:
                    self.performance_metrics['ticks_per_second'] = (len(recent_ticks) - 1) / time_span
                    
        self.performance_metrics['total_transforms'] += len(tick.transforms_applied)
        self.performance_metrics['total_violations'] += len(tick.invariant_violations)
    
    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger registered event callbacks"""
        for callback in self.event_callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for event {event}: {e}")
    
    # Public API Methods
    
    def add_semantic_concept(self, concept_embedding: np.ndarray, 
                           concept_name: str = None,
                           initial_position: np.ndarray = None,
                           layer: LayerDepth = LayerDepth.SURFACE,
                           sector: SectorType = SectorType.CORE) -> str:
        """Add a new semantic concept to the field"""
        
        node = SemanticNode(
            embedding=concept_embedding,
            position=initial_position if initial_position is not None else np.random.randn(3) * 0.1,
            layer=layer,
            sector=sector
        )
        
        if concept_name:
            node.id = f"concept_{concept_name}_{int(time.time())}"
            
        success = self.field.add_node(node)
        
        if success:
            logger.info(f"🌐 Added semantic concept: {node.id}")
            self._trigger_callbacks('concept_added', node)
            return node.id
        else:
            logger.error(f"Failed to add semantic concept: {concept_name}")
            return None
    
    def create_semantic_relationship(self, node_a_id: str, node_b_id: str,
                                   relationship_strength: float = 1.0,
                                   directed: bool = False) -> str:
        """Create a semantic relationship between two concepts"""
        
        edge = SemanticEdge(
            node_a=node_a_id,
            node_b=node_b_id,
            weight=relationship_strength,
            directed=directed
        )
        
        success = self.field.add_edge(edge)
        
        if success:
            logger.info(f"🌐 Created semantic relationship: {edge.id}")
            self._trigger_callbacks('relationship_created', edge)
            return edge.id
        else:
            logger.error(f"Failed to create relationship between {node_a_id} and {node_b_id}")
            return None
    
    def query_semantic_neighborhood(self, node_id: str, radius: float = 2.0) -> Dict[str, Any]:
        """Query the semantic neighborhood around a concept"""
        
        if node_id not in self.field.nodes:
            return {'error': f'Node {node_id} not found'}
            
        center_node = self.field.nodes[node_id]
        neighbors = []
        
        for other_id, other_node in self.field.nodes.items():
            if other_id == node_id:
                continue
                
            distance = center_node.spatial_distance_to(other_node)
            if distance <= radius:
                neighbors.append({
                    'node_id': other_id,
                    'spatial_distance': distance,
                    'semantic_distance': center_node.semantic_distance_to(other_node),
                    'layer': other_node.layer.value,
                    'sector': other_node.sector.value
                })
                
        # Sort by distance
        neighbors.sort(key=lambda x: x['spatial_distance'])
        
        return {
            'center_node': node_id,
            'radius': radius,
            'neighbors': neighbors,
            'neighbor_count': len(neighbors)
        }
    
    def manual_transform(self, transform_type: str, **kwargs) -> TransformOperation:
        """Manually trigger a topology transform"""
        
        transform_map = {
            'weave': lambda: self.transforms.weave(kwargs.get('node_a'), kwargs.get('node_b')),
            'prune': lambda: self.transforms.prune(kwargs.get('edge_ids', [])),
            'fuse': lambda: self.transforms.fuse(kwargs.get('cluster_a', []), kwargs.get('cluster_b', [])),
            'lift': lambda: self.transforms.lift(kwargs.get('node_id')),
            'sink': lambda: self.transforms.sink(kwargs.get('node_id')),
            'reproject': lambda: self.transforms.reproject(kwargs.get('node_ids', []))
        }
        
        if transform_type not in transform_map:
            logger.error(f"Unknown transform type: {transform_type}")
            return None
            
        try:
            result = transform_map[transform_type]()
            logger.info(f"🌐 Manual transform {transform_type}: {result.result.value}")
            self._trigger_callbacks('manual_transform', result)
            return result
        except Exception as e:
            logger.error(f"Manual transform {transform_type} failed: {e}")
            return None
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        
        uptime = time.time() - self.start_time
        
        return {
            'engine_id': self.engine_id,
            'state': self.state.value,
            'uptime_seconds': uptime,
            'tick_count': self.tick_count,
            'field_statistics': self.field.get_field_statistics(),
            'performance_metrics': dict(self.performance_metrics),
            'recent_violations': len([v for v in self.invariants.violation_history 
                                    if time.time() - v.timestamp < 300]),
            'transform_statistics': self.transforms.get_transform_statistics(),
            'invariant_health': self.invariants.get_violation_summary(),
            'processing_active': self._running
        }
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for engine events"""
        self.event_callbacks[event].append(callback)
        logger.debug(f"Registered callback for event: {event}")
    
    def integrate_with_consciousness(self, consciousness_system):
        """Integrate with DAWN's consciousness systems"""
        self.consciousness_integration = consciousness_system
        logger.info("🌐 Integrated with DAWN consciousness system")
        
    def integrate_with_sigils(self, sigil_system):
        """Integrate with DAWN's sigil systems"""
        self.sigil_integration = sigil_system
        logger.info("🌐 Integrated with DAWN sigil system")


# Global engine instance
_semantic_topology_engine = None

def get_semantic_topology_engine(**kwargs) -> SemanticTopologyEngine:
    """Get the global semantic topology engine instance"""
    global _semantic_topology_engine
    if _semantic_topology_engine is None:
        _semantic_topology_engine = SemanticTopologyEngine(**kwargs)
    return _semantic_topology_engine

def shutdown_semantic_topology_engine():
    """Shutdown the global engine instance"""
    global _semantic_topology_engine
    if _semantic_topology_engine is not None:
        _semantic_topology_engine.shutdown()
        _semantic_topology_engine = None

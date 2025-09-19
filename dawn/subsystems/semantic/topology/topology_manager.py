"""
Semantic Topology Manager
=========================

Main integration system for DAWN's semantic topology based on RTF specifications.
Coordinates topology state, field equations, transforms, and integration with
the broader DAWN consciousness system.
"""

import numpy as np
import time
import json
import threading
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
import logging

from .primitives import (
    SemanticNode, SemanticEdge, TopologyLayer, TopologyFrame, 
    NodeCoordinates, TopologyMetrics, TopologySector,
    create_semantic_node, create_semantic_edge
)
from .field_equations import FieldEquationEngine, FieldParameters
from .transforms import TopologyTransforms, TransformResult, TransformOperation

logger = logging.getLogger(__name__)


@dataclass
class TopologySnapshot:
    """Complete snapshot of topology state for serialization/analysis"""
    tick: int
    timestamp: float
    
    # Core topology data
    nodes: Dict[str, Dict[str, Any]]
    edges: Dict[str, Dict[str, Any]]
    layers: Dict[int, Dict[str, Any]]
    
    # Metrics and statistics
    metrics: Dict[str, Any]
    field_statistics: Dict[str, Any]
    transform_statistics: Dict[str, Any]
    
    # Frame state
    active_frames: List[Dict[str, Any]]
    
    def to_json(self) -> str:
        """Serialize snapshot to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                serializable_data[key] = self._convert_numpy_to_lists(value)
            elif isinstance(value, list):
                serializable_data[key] = [self._convert_numpy_to_lists(item) if isinstance(item, dict) else item for item in value]
            else:
                serializable_data[key] = value
        
        return json.dumps(serializable_data, indent=2)
    
    def _convert_numpy_to_lists(self, obj):
        """Recursively convert numpy arrays to lists"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_lists(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


@dataclass 
class TopologyState:
    """Current state of the semantic topology system"""
    tick_count: int = 0
    last_update_time: float = 0.0
    
    # Core topology structures
    nodes: Dict[str, SemanticNode] = field(default_factory=dict)
    edges: Dict[str, SemanticEdge] = field(default_factory=dict)
    layers: Dict[int, TopologyLayer] = field(default_factory=dict)
    frames: List[TopologyFrame] = field(default_factory=list)
    
    # System state
    total_energy: float = 100.0
    system_coherence: float = 0.5
    global_tension: float = 0.0
    
    # History tracking
    snapshot_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    coherence_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def get_node_count(self) -> int:
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        return len(self.edges)
    
    def get_layer_distribution(self) -> Dict[int, int]:
        layer_counts = {}
        for node in self.nodes.values():
            layer = node.coordinates.layer
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        return layer_counts
    
    def get_sector_distribution(self) -> Dict[str, int]:
        sector_counts = {}
        for node in self.nodes.values():
            sector = node.coordinates.sector.value
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return sector_counts


class SemanticTopologyManager:
    """
    Main manager for DAWN's semantic topology system.
    
    Integrates all topology components and provides unified interface
    for topology operations, field updates, and system monitoring.
    """
    
    def __init__(self, 
                 field_params: Optional[FieldParameters] = None,
                 transform_energy_budget: float = 5.0,
                 auto_update_enabled: bool = True,
                 snapshot_interval: int = 100):
        
        # Core system state
        self.state = TopologyState()
        self.running = False
        self.auto_update_enabled = auto_update_enabled
        self.snapshot_interval = snapshot_interval
        
        # Initialize field equation engine
        self.field_engine = FieldEquationEngine(field_params)
        
        # Initialize transform system
        self.transforms = TopologyTransforms(transform_energy_budget)
        
        # Residue data tracking (from external systems)
        self.residue_data: Dict[str, Dict[str, float]] = {}
        
        # Threading for auto-updates
        self.update_thread: Optional[threading.Thread] = None
        self.update_lock = threading.RLock()
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics = {
            'update_times': deque(maxlen=100),
            'field_update_times': deque(maxlen=100),
            'transform_times': deque(maxlen=100),
            'total_updates': 0
        }
        
        # Initialize default layer 0
        self.state.layers[0] = TopologyLayer(depth=0)
        
        logger.info("🗺️ Semantic Topology Manager initialized")
    
    def start(self) -> bool:
        """Start the topology manager with optional auto-updates"""
        if self.running:
            logger.warning("Topology manager already running")
            return False
        
        self.running = True
        
        if self.auto_update_enabled:
            self.update_thread = threading.Thread(
                target=self._auto_update_loop,
                name="topology_auto_update",
                daemon=True
            )
            self.update_thread.start()
            logger.info("🗺️ Auto-update thread started")
        
        self._emit_event('topology_started', {'timestamp': time.time()})
        logger.info("🗺️ Semantic Topology Manager started")
        return True
    
    def stop(self) -> bool:
        """Stop the topology manager"""
        if not self.running:
            return False
        
        self.running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        self._emit_event('topology_stopped', {'timestamp': time.time()})
        logger.info("🗺️ Semantic Topology Manager stopped")
        return True
    
    def add_semantic_node(self, content: str, embedding: np.ndarray,
                         position: Optional[np.ndarray] = None,
                         sector: TopologySector = TopologySector.PERIPHERAL,
                         layer: int = 0) -> str:
        """Add a new semantic node to the topology"""
        with self.update_lock:
            node = create_semantic_node(content, embedding, position, sector, layer)
            
            # Add to topology state
            self.state.nodes[node.id] = node
            
            # Add to appropriate layer
            if layer not in self.state.layers:
                self.state.layers[layer] = TopologyLayer(depth=layer)
            
            self.state.layers[layer].add_node(node.id)
            
            # Update system metrics
            self._update_system_metrics()
            
            self._emit_event('node_added', {
                'node_id': node.id,
                'content': content,
                'layer': layer,
                'sector': sector.value
            })
            
            logger.debug(f"Added semantic node: {node.id} ({content[:30]}...)")
            return node.id
    
    def add_semantic_edge(self, source_id: str, target_id: str,
                         weight: float = 0.5, directed: bool = False) -> Optional[str]:
        """Add a new semantic edge between nodes"""
        with self.update_lock:
            if source_id not in self.state.nodes or target_id not in self.state.nodes:
                logger.warning(f"Cannot create edge: missing nodes {source_id} or {target_id}")
                return None
            
            edge = create_semantic_edge(source_id, target_id, weight, directed)
            
            # Check if edge already exists
            reverse_id = f"{target_id}-{source_id}"
            if edge.id in self.state.edges or reverse_id in self.state.edges:
                logger.debug(f"Edge already exists: {edge.id}")
                return edge.id
            
            self.state.edges[edge.id] = edge
            
            self._emit_event('edge_added', {
                'edge_id': edge.id,
                'source_id': source_id,
                'target_id': target_id,
                'weight': weight,
                'directed': directed
            })
            
            logger.debug(f"Added semantic edge: {edge.id}")
            return edge.id
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges from the topology"""
        with self.update_lock:
            if node_id not in self.state.nodes:
                return False
            
            node = self.state.nodes[node_id]
            
            # Remove from layer
            if node.coordinates.layer in self.state.layers:
                self.state.layers[node.coordinates.layer].remove_node(node_id)
            
            # Remove all edges connected to this node
            edges_to_remove = []
            for edge_id, edge in self.state.edges.items():
                if edge.source_id == node_id or edge.target_id == node_id:
                    edges_to_remove.append(edge_id)
            
            for edge_id in edges_to_remove:
                del self.state.edges[edge_id]
            
            # Remove node
            del self.state.nodes[node_id]
            
            self._emit_event('node_removed', {
                'node_id': node_id,
                'edges_removed': len(edges_to_remove)
            })
            
            logger.debug(f"Removed node {node_id} and {len(edges_to_remove)} edges")
            return True
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the topology"""
        with self.update_lock:
            if edge_id not in self.state.edges:
                return False
            
            edge = self.state.edges[edge_id]
            del self.state.edges[edge_id]
            
            self._emit_event('edge_removed', {
                'edge_id': edge_id,
                'source_id': edge.source_id,
                'target_id': edge.target_id
            })
            
            logger.debug(f"Removed edge {edge_id}")
            return True
    
    def update_topology_tick(self, external_residue_data: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Main topology update cycle (called per DAWN tick)
        
        Implements the RTF specification:
        1. Update: Ingest data, recompute metrics, soft-reproject
        2. Decide: Determine necessary transforms
        3. Transform: Apply topology operations
        4. Monitor: Check invariants and system health
        """
        if not self.running:
            return {'error': 'Topology manager not running'}
        
        start_time = time.time()
        
        with self.update_lock:
            try:
                self.state.tick_count += 1
                
                # Update residue data if provided
                if external_residue_data:
                    self.residue_data.update(external_residue_data)
                
                # Phase 1: Update field equations
                field_start = time.time()
                field_results = self.field_engine.update_all_fields(
                    self.state.nodes, self.state.edges, self.residue_data
                )
                field_time = time.time() - field_start
                self.performance_metrics['field_update_times'].append(field_time)
                
                # Phase 2: Decide on transforms
                transform_decisions = self._decide_transforms()
                
                # Phase 3: Apply transforms
                transform_start = time.time()
                transform_results = self._apply_transforms(transform_decisions)
                transform_time = time.time() - transform_start
                self.performance_metrics['transform_times'].append(transform_time)
                
                # Phase 4: Update system metrics
                self._update_system_metrics()
                
                # Phase 5: Monitor invariants
                invariant_violations = self._check_invariants()
                
                # Create snapshot if interval reached
                if self.state.tick_count % self.snapshot_interval == 0:
                    self._create_snapshot()
                
                # Update performance tracking
                total_time = time.time() - start_time
                self.performance_metrics['update_times'].append(total_time)
                self.performance_metrics['total_updates'] += 1
                self.state.last_update_time = time.time()
                
                # Emit update event
                self._emit_event('topology_updated', {
                    'tick': self.state.tick_count,
                    'update_time_ms': total_time * 1000,
                    'field_time_ms': field_time * 1000,
                    'transform_time_ms': transform_time * 1000,
                    'node_count': len(self.state.nodes),
                    'edge_count': len(self.state.edges)
                })
                
                return {
                    'success': True,
                    'tick': self.state.tick_count,
                    'update_time_ms': total_time * 1000,
                    'field_results': field_results,
                    'transform_results': transform_results,
                    'invariant_violations': invariant_violations,
                    'system_coherence': self.state.system_coherence,
                    'node_count': len(self.state.nodes),
                    'edge_count': len(self.state.edges)
                }
                
            except Exception as e:
                logger.error(f"Error in topology tick update: {e}")
                return {'error': str(e), 'tick': self.state.tick_count}
    
    def _decide_transforms(self) -> List[Dict[str, Any]]:
        """Decide which topology transforms to apply based on current state"""
        decisions = []
        
        # Auto-prune problematic edges
        for edge_id, edge in self.state.edges.items():
            if edge.tension > 0.7 or edge.volatility > 0.8 or edge.reliability < 0.3:
                decisions.append({
                    'type': 'prune',
                    'target': edge_id,
                    'strength': min(1.0, edge.tension + edge.volatility),
                    'reason': f'High tension/volatility or low reliability'
                })
        
        # Auto-lift high-energy nodes
        for node_id, node in self.state.nodes.items():
            if (node.energy > 0.7 and node.health > 0.6 and 
                node.coordinates.layer > 0 and node.local_coherence > 0.6):
                decisions.append({
                    'type': 'lift',
                    'target': node_id,
                    'reason': 'High energy and coherence - rebloom candidate'
                })
        
        # Auto-sink low-health nodes
        for node_id, node in self.state.nodes.items():
            if node.health < 0.3 and node.energy < 0.2 and node.coordinates.layer < 3:
                decisions.append({
                    'type': 'sink',
                    'target': node_id,
                    'reason': 'Low health and energy - decay candidate'
                })
        
        # Reproject if average coherence is low
        if (len(self.state.nodes) > 5 and 
            self.state.system_coherence < 0.4 and 
            self.state.tick_count % 50 == 0):  # Every 50 ticks
            decisions.append({
                'type': 'reproject',
                'target': 'all_nodes',
                'method': 'pca',
                'reason': 'Low system coherence - spatial reoptimization needed'
            })
        
        return decisions
    
    def _apply_transforms(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply decided topology transforms"""
        results = {
            'applied': [],
            'failed': [],
            'total_energy_used': 0.0
        }
        
        for decision in decisions:
            try:
                transform_type = decision['type']
                target = decision['target']
                
                if transform_type == 'prune':
                    if target in self.state.edges:
                        edge = self.state.edges[target]
                        strength = decision.get('strength', 0.5)
                        result = self.transforms.prune.prune(edge, strength, self.state.edges, remove_completely=True)
                        
                        if result == TransformResult.SUCCESS:
                            results['applied'].append(decision)
                        else:
                            results['failed'].append({'decision': decision, 'result': result.value})
                
                elif transform_type == 'lift':
                    if target in self.state.nodes:
                        node = self.state.nodes[target]
                        result = self.transforms.lift.lift(node, self.state.layers)
                        
                        if result == TransformResult.SUCCESS:
                            results['applied'].append(decision)
                        else:
                            results['failed'].append({'decision': decision, 'result': result.value})
                
                elif transform_type == 'sink':
                    if target in self.state.nodes:
                        node = self.state.nodes[target]
                        result = self.transforms.sink.sink(node, self.state.layers)
                        
                        if result == TransformResult.SUCCESS:
                            results['applied'].append(decision)
                        else:
                            results['failed'].append({'decision': decision, 'result': result.value})
                
                elif transform_type == 'reproject':
                    method = decision.get('method', 'pca')
                    result = self.transforms.reproject.reproject(self.state.nodes, self.state.edges, method)
                    
                    if result == TransformResult.SUCCESS:
                        results['applied'].append(decision)
                    else:
                        results['failed'].append({'decision': decision, 'result': result.value})
                
            except Exception as e:
                logger.error(f"Error applying transform {decision}: {e}")
                results['failed'].append({'decision': decision, 'error': str(e)})
        
        # Get energy usage
        energy_status = self.transforms.get_energy_status()
        results['energy_status'] = energy_status
        
        return results
    
    def _update_system_metrics(self):
        """Update overall system coherence and health metrics"""
        if not self.state.nodes:
            self.state.system_coherence = 0.0
            return
        
        # Calculate average coherence
        coherences = [node.local_coherence for node in self.state.nodes.values()]
        self.state.system_coherence = np.mean(coherences) if coherences else 0.0
        
        # Calculate global tension
        if self.state.edges:
            tensions = [edge.tension for edge in self.state.edges.values()]
            self.state.global_tension = np.mean(tensions)
        else:
            self.state.global_tension = 0.0
        
        # Update coherence history
        self.state.coherence_history.append(self.state.system_coherence)
    
    def _check_invariants(self) -> List[str]:
        """Check topology invariants and return violations"""
        violations = []
        
        # Invariant 1: Topology ↔ Schema Consistency
        high_tension_edges = [e for e in self.state.edges.values() if e.tension > 0.8]
        if len(high_tension_edges) > len(self.state.edges) * 0.2:  # More than 20% high tension
            violations.append(f"High tension in {len(high_tension_edges)} edges - topology-schema inconsistency")
        
        # Invariant 2: No Free Tears (high-reliability edges shouldn't be too long)
        for edge in self.state.edges.values():
            if edge.reliability > 0.8:
                if (edge.source_id in self.state.nodes and edge.target_id in self.state.nodes):
                    source_node = self.state.nodes[edge.source_id]
                    target_node = self.state.nodes[edge.target_id]
                    distance = source_node.coordinates.spatial_distance(target_node.coordinates)
                    
                    if distance > 3.0:  # Arbitrary threshold for "too long"
                        violations.append(f"High-reliability edge {edge.id} spans large distance ({distance:.2f})")
        
        # Invariant 3: Layer Ordering (energy should generally decrease with depth)
        layer_energies = {}
        for node in self.state.nodes.values():
            layer = node.coordinates.layer
            if layer not in layer_energies:
                layer_energies[layer] = []
            layer_energies[layer].append(node.energy)
        
        # Check if deeper layers have significantly higher average energy
        for layer_depth in sorted(layer_energies.keys()):
            if layer_depth > 0:
                current_avg = np.mean(layer_energies[layer_depth])
                surface_avg = np.mean(layer_energies.get(0, [0.5]))
                
                if current_avg > surface_avg + 0.3:  # Deep layer much higher energy
                    violations.append(f"Layer {layer_depth} has higher average energy ({current_avg:.2f}) than surface ({surface_avg:.2f})")
        
        return violations
    
    def _create_snapshot(self) -> TopologySnapshot:
        """Create a complete snapshot of current topology state"""
        # Convert nodes to serializable format
        nodes_dict = {}
        for node_id, node in self.state.nodes.items():
            nodes_dict[node_id] = {
                'id': node.id,
                'position': node.coordinates.position.tolist(),
                'embedding': node.coordinates.embedding.tolist(),
                'layer': node.coordinates.layer,
                'sector': node.coordinates.sector.value,
                'tint': node.tint.tolist(),
                'health': node.health,
                'energy': node.energy,
                'local_coherence': node.local_coherence,
                'residue_pressure': node.residue_pressure,
                'content': node.content,
                'last_touch': node.last_touch
            }
        
        # Convert edges to serializable format
        edges_dict = {}
        for edge_id, edge in self.state.edges.items():
            edges_dict[edge_id] = {
                'id': edge.id,
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'weight': edge.weight,
                'tension': edge.tension,
                'volatility': edge.volatility,
                'reliability': edge.reliability,
                'directed': edge.directed,
                'nutrient_flow': edge.nutrient_flow,
                'rattling_phase': edge.rattling_phase
            }
        
        # Convert layers to serializable format
        layers_dict = {}
        for layer_depth, layer in self.state.layers.items():
            layers_dict[layer_depth] = {
                'depth': layer.depth,
                'node_ids': list(layer.node_ids),
                'capacity': layer.capacity,
                'occupancy_ratio': layer.get_occupancy_ratio(),
                'decay_rate': layer.decay_rate,
                'rebloom_threshold': layer.rebloom_threshold
            }
        
        # Get current metrics
        metrics = TopologyMetrics(
            total_nodes=len(self.state.nodes),
            total_edges=len(self.state.edges),
            average_coherence=self.state.system_coherence,
            average_tension=self.state.global_tension,
            layer_occupancy=self.state.get_layer_distribution(),
            sector_distribution=self.state.get_sector_distribution()
        )
        
        # Create snapshot
        snapshot = TopologySnapshot(
            tick=self.state.tick_count,
            timestamp=time.time(),
            nodes=nodes_dict,
            edges=edges_dict,
            layers=layers_dict,
            metrics=metrics.to_dict(),
            field_statistics=self.field_engine.get_field_statistics(self.state.nodes, self.state.edges),
            transform_statistics=self.transforms.get_transform_statistics(),
            active_frames=[self._frame_to_dict(frame) for frame in self.state.frames]
        )
        
        self.state.snapshot_history.append(snapshot)
        return snapshot
    
    def _frame_to_dict(self, frame: TopologyFrame) -> Dict[str, Any]:
        """Convert topology frame to dictionary"""
        return {
            'center': frame.center.tolist(),
            'radius': frame.radius,
            'layer': frame.layer,
            'filters': frame.filters,
            'motion_budget': frame.motion_budget,
            'energy_budget': frame.energy_budget
        }
    
    def _auto_update_loop(self):
        """Auto-update loop for continuous topology maintenance"""
        while self.running:
            try:
                self.update_topology_tick()
                time.sleep(0.1)  # 10 Hz update rate
            except Exception as e:
                logger.error(f"Error in auto-update loop: {e}")
                time.sleep(1.0)  # Back off on errors
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register callback for topology events"""
        self.event_callbacks[event_type].append(callback)
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered callbacks"""
        for callback in self.event_callbacks[event_type]:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback for {event_type}: {e}")
    
    def get_topology_state(self) -> Dict[str, Any]:
        """Get current topology state summary"""
        return {
            'running': self.running,
            'tick_count': self.state.tick_count,
            'node_count': len(self.state.nodes),
            'edge_count': len(self.state.edges),
            'layer_count': len(self.state.layers),
            'system_coherence': self.state.system_coherence,
            'global_tension': self.state.global_tension,
            'total_energy': self.state.total_energy,
            'layer_distribution': self.state.get_layer_distribution(),
            'sector_distribution': self.state.get_sector_distribution(),
            'last_update_time': self.state.last_update_time,
            'performance_summary': {
                'total_updates': self.performance_metrics['total_updates'],
                'average_update_time_ms': np.mean(self.performance_metrics['update_times']) * 1000 if self.performance_metrics['update_times'] else 0,
                'average_field_time_ms': np.mean(self.performance_metrics['field_update_times']) * 1000 if self.performance_metrics['field_update_times'] else 0
            }
        }
    
    def get_latest_snapshot(self) -> Optional[TopologySnapshot]:
        """Get the most recent topology snapshot"""
        return self.state.snapshot_history[-1] if self.state.snapshot_history else None
    
    def export_topology_data(self, filepath: str) -> bool:
        """Export current topology state to file"""
        try:
            snapshot = self._create_snapshot()
            
            with open(filepath, 'w') as f:
                f.write(snapshot.to_json())
            
            logger.info(f"Exported topology data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export topology data: {e}")
            return False
    
    def recharge_transform_energy(self, amount: Optional[float] = None):
        """Recharge energy budgets for transform operations"""
        self.transforms.recharge_energy(amount)
        logger.debug("Transform energy budgets recharged")
    
    def get_node_neighbors(self, node_id: str, max_neighbors: int = 10) -> List[str]:
        """Get neighboring node IDs for a given node"""
        if node_id not in self.state.nodes:
            return []
        
        node = self.state.nodes[node_id]
        all_nodes = list(self.state.nodes.values())
        
        neighbors = self.field_engine.coherence_calculator.find_neighbors(node, all_nodes)
        neighbor_ids = [n.id for n in neighbors[:max_neighbors]]
        
        return neighbor_ids
    
    def force_topology_operation(self, operation_type: str, **kwargs) -> Dict[str, Any]:
        """Force a specific topology operation (for external control)"""
        try:
            if operation_type == 'weave' and 'source_id' in kwargs and 'target_id' in kwargs:
                source_node = self.state.nodes.get(kwargs['source_id'])
                target_node = self.state.nodes.get(kwargs['target_id'])
                
                if source_node and target_node:
                    alpha = kwargs.get('alpha', 0.5)
                    result, edge = self.transforms.weave.weave(
                        source_node, target_node, alpha, self.state.edges
                    )
                    return {'success': result == TransformResult.SUCCESS, 'result': result.value, 'edge_id': edge.id if edge else None}
            
            elif operation_type == 'prune' and 'edge_id' in kwargs:
                edge = self.state.edges.get(kwargs['edge_id'])
                if edge:
                    kappa = kwargs.get('kappa', 0.5)
                    result = self.transforms.prune.prune(edge, kappa, self.state.edges, kwargs.get('remove_completely', False))
                    return {'success': result == TransformResult.SUCCESS, 'result': result.value}
            
            elif operation_type == 'lift' and 'node_id' in kwargs:
                node = self.state.nodes.get(kwargs['node_id'])
                if node:
                    result = self.transforms.lift.lift(node, self.state.layers, kwargs.get('target_layer'))
                    return {'success': result == TransformResult.SUCCESS, 'result': result.value}
            
            elif operation_type == 'sink' and 'node_id' in kwargs:
                node = self.state.nodes.get(kwargs['node_id'])
                if node:
                    result = self.transforms.sink.sink(node, self.state.layers, kwargs.get('target_layer'))
                    return {'success': result == TransformResult.SUCCESS, 'result': result.value}
            
            elif operation_type == 'reproject':
                method = kwargs.get('method', 'pca')
                result = self.transforms.reproject.reproject(self.state.nodes, self.state.edges, method)
                return {'success': result == TransformResult.SUCCESS, 'result': result.value}
            
            else:
                return {'success': False, 'error': f'Unknown operation type: {operation_type}'}
                
        except Exception as e:
            logger.error(f"Error in forced topology operation {operation_type}: {e}")
            return {'success': False, 'error': str(e)}


# Global topology manager instance
_global_topology_manager: Optional[SemanticTopologyManager] = None

def get_topology_manager(field_params: Optional[FieldParameters] = None,
                        transform_energy_budget: float = 5.0,
                        auto_start: bool = True) -> SemanticTopologyManager:
    """Get the global semantic topology manager instance"""
    global _global_topology_manager
    
    if _global_topology_manager is None:
        _global_topology_manager = SemanticTopologyManager(
            field_params=field_params,
            transform_energy_budget=transform_energy_budget
        )
        
        if auto_start:
            _global_topology_manager.start()
    
    return _global_topology_manager

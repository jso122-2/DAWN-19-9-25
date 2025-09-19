"""
🧠 Integrated Mycelial Layer System
===================================

Complete integration of all mycelial layer components into a unified system
that integrates with DAWN's tick loop. This system orchestrates:

1. Core mycelial layer with nodes and edges
2. Nutrient economy with demand calculation and allocation
3. Energy flows (passive diffusion and active transport)
4. Growth gates and autophagy mechanisms
5. Metabolite production and recycling
6. Cluster fusion and fission dynamics

The system "lives inside the tick loop, reacting to real-time pressures"
as specified in the DAWN documentation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import logging
import json

# Import all mycelial components
from .core import MycelialLayer, MycelialNode, MycelialEdge
from .nutrient_economy import NutrientEconomy
from .energy_flows import EnergyFlowManager
from .growth_gates import GrowthGate, AutophagyManager
from .metabolites import MetaboliteManager
from .cluster_dynamics import ClusterManager

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """Overall system states"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    STRESSED = "stressed"
    CRITICAL = "critical"
    RECOVERY = "recovery"
    DORMANT = "dormant"

@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""
    
    # Core metrics
    total_nodes: int = 0
    total_edges: int = 0
    total_energy: float = 0.0
    avg_node_health: float = 0.0
    
    # Economy metrics
    nutrient_efficiency: float = 0.0
    demand_satisfaction: float = 0.0
    energy_generation_rate: float = 0.0
    
    # Flow metrics
    energy_flows_active: int = 0
    flow_efficiency: float = 0.0
    
    # Growth metrics
    growth_approval_rate: float = 0.0
    autophagy_rate: float = 0.0
    
    # Metabolite metrics
    active_metabolites: int = 0
    metabolite_absorption_rate: float = 0.0
    
    # Cluster metrics
    active_clusters: int = 0
    clustering_efficiency: float = 0.0
    fusion_fission_activity: float = 0.0
    
    # System health
    overall_health: float = 0.0
    stress_level: float = 0.0
    adaptation_rate: float = 0.0

class IntegratedMycelialSystem:
    """
    The complete integrated mycelial layer system that orchestrates all components
    and integrates with DAWN's consciousness architecture.
    """
    
    def __init__(self, 
                 max_nodes: int = 10000,
                 max_edges_per_node: int = 50,
                 tick_rate: float = 1.0,
                 enable_nutrient_economy: bool = True,
                 enable_energy_flows: bool = True,
                 enable_growth_gates: bool = True,
                 enable_metabolites: bool = True,
                 enable_clustering: bool = True):
        
        # Core system
        self.mycelial_layer = MycelialLayer(
            max_nodes=max_nodes,
            max_edges_per_node=max_edges_per_node,
            tick_rate=tick_rate
        )
        
        # Component systems (optional)
        self.nutrient_economy = NutrientEconomy(self.mycelial_layer) if enable_nutrient_economy else None
        self.energy_flow_manager = EnergyFlowManager(self.mycelial_layer) if enable_energy_flows else None
        self.growth_gate = GrowthGate() if enable_growth_gates else None
        self.autophagy_manager = AutophagyManager() if enable_growth_gates else None
        self.metabolite_manager = MetaboliteManager(self.mycelial_layer) if enable_metabolites else None
        self.cluster_manager = ClusterManager(self.mycelial_layer) if enable_clustering else None
        
        # System state
        self.state = SystemState.INITIALIZING
        self.metrics = SystemMetrics()
        self.tick_count = 0
        self.last_update = time.time()
        
        # Performance tracking
        self.tick_times: deque = deque(maxlen=100)
        self.component_times: Dict[str, deque] = {
            'core': deque(maxlen=100),
            'nutrient_economy': deque(maxlen=100),
            'energy_flows': deque(maxlen=100),
            'growth_autophagy': deque(maxlen=100),
            'metabolites': deque(maxlen=100),
            'clustering': deque(maxlen=100)
        }
        
        # Integration parameters
        self.stress_threshold = 0.7
        self.critical_threshold = 0.9
        self.recovery_threshold = 0.3
        
        # External interfaces
        self.consciousness_interface = None  # For integration with DAWN consciousness
        self.sensor_interfaces: Dict[str, Any] = {}  # For external sensors
        self.actuator_interfaces: Dict[str, Any] = {}  # For external actuators
        
        # Event system
        self.event_handlers: Dict[str, List[callable]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Mark as initialized
        self.state = SystemState.HEALTHY
        
        enabled_count = sum([
            enable_nutrient_economy, enable_energy_flows, enable_growth_gates,
            enable_metabolites, enable_clustering
        ])
        logger.info(f"IntegratedMycelialSystem initialized with {enabled_count} enabled subsystems")
    
    def tick_update(self, delta_time: Optional[float] = None,
                   external_pressures: Optional[Dict[str, float]] = None,
                   consciousness_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main tick update that coordinates all subsystems.
        
        This is where the mycelial layer "lives inside the tick loop,
        reacting to real-time pressures" as per DAWN documentation.
        """
        tick_start = time.time()
        
        if delta_time is None:
            delta_time = tick_start - self.last_update
        
        with self._lock:
            self.tick_count += 1
            
            # Apply external pressures to the system
            if external_pressures:
                self._apply_external_pressures(external_pressures)
            
            # Integrate with consciousness state
            if consciousness_state:
                self._integrate_consciousness_state(consciousness_state)
            
            # Component updates in dependency order
            component_results = {}
            
            # 1. Core mycelial layer update
            core_start = time.time()
            self.mycelial_layer.tick_update(delta_time)
            self.component_times['core'].append(time.time() - core_start)
            
            # 2. Nutrient economy (provides energy to nodes)
            if self.nutrient_economy:
                economy_start = time.time()
                economy_results = self.nutrient_economy.tick_update(delta_time)
                component_results['nutrient_economy'] = economy_results
                self.component_times['nutrient_economy'].append(time.time() - economy_start)
            
            # 3. Energy flows (redistributes energy between nodes)
            if self.energy_flow_manager:
                flow_start = time.time()
                flow_results = self.energy_flow_manager.tick_update(delta_time)
                component_results['energy_flows'] = flow_results
                self.component_times['energy_flows'].append(time.time() - flow_start)
            
            # 4. Growth gates and autophagy (structural changes)
            if self.growth_gate and self.autophagy_manager:
                growth_start = time.time()
                growth_results = self._update_growth_and_autophagy(delta_time)
                component_results['growth_autophagy'] = growth_results
                self.component_times['growth_autophagy'].append(time.time() - growth_start)
            
            # 5. Metabolite management (recycles dead nodes)
            if self.metabolite_manager:
                metabolite_start = time.time()
                metabolite_results = self.metabolite_manager.tick_update(delta_time)
                component_results['metabolites'] = metabolite_results
                self.component_times['metabolites'].append(time.time() - metabolite_start)
            
            # 6. Cluster dynamics (high-level organization)
            if self.cluster_manager:
                cluster_start = time.time()
                cluster_results = self.cluster_manager.tick_update(delta_time)
                component_results['clustering'] = cluster_results
                self.component_times['clustering'].append(time.time() - cluster_start)
            
            # Update system metrics and state
            self._update_system_metrics()
            self._update_system_state()
            
            # Emit system events
            self._emit_system_events(component_results)
            
            # Record timing
            total_time = time.time() - tick_start
            self.tick_times.append(total_time)
            self.last_update = tick_start
            
            return {
                'tick_count': self.tick_count,
                'system_state': self.state.value,
                'system_metrics': self._serialize_metrics(),
                'component_results': component_results,
                'timing': {
                    'total_time': total_time,
                    'component_times': {k: v[-1] if v else 0.0 for k, v in self.component_times.items()}
                }
            }
    
    def _apply_external_pressures(self, pressures: Dict[str, float]):
        """Apply external pressures to nodes in the mycelial layer"""
        
        # Global pressure affects all nodes
        global_pressure = pressures.get('global_pressure', 0.0)
        
        # Specific pressures can target node subsets
        cognitive_pressure = pressures.get('cognitive_pressure', 0.0)
        emotional_pressure = pressures.get('emotional_pressure', 0.0)
        memory_pressure = pressures.get('memory_pressure', 0.0)
        
        for node in self.mycelial_layer.nodes.values():
            # Apply global pressure
            node.pressure += global_pressure * 0.1
            
            # Apply specific pressures based on node characteristics
            if hasattr(node, 'node_type'):
                if node.node_type == 'cognitive':
                    node.pressure += cognitive_pressure * 0.2
                elif node.node_type == 'emotional':
                    node.pressure += emotional_pressure * 0.2
                elif node.node_type == 'memory':
                    node.pressure += memory_pressure * 0.2
            
            # Clamp pressure values
            node.pressure = max(0.0, min(1.0, node.pressure))
    
    def _integrate_consciousness_state(self, consciousness_state: Dict[str, Any]):
        """Integrate with DAWN consciousness state"""
        
        # Extract relevant consciousness metrics
        consciousness_level = consciousness_state.get('consciousness_level', 0.5)
        coherence = consciousness_state.get('coherence', 0.5)
        unity = consciousness_state.get('unity', 0.5)
        awareness = consciousness_state.get('awareness', 0.5)
        
        # Affect global mycelial parameters
        if self.nutrient_economy:
            # Higher consciousness = more efficient nutrient allocation
            efficiency_boost = consciousness_level * 0.1
            self.nutrient_economy.budget.base_efficiency += efficiency_boost
            
            # Coherence affects allocation weights
            coherence_factor = coherence - 0.5  # [-0.5, 0.5]
            self.nutrient_economy.budget.pressure_weight += coherence_factor * 0.1
        
        # Unity affects cluster dynamics
        if self.cluster_manager:
            unity_factor = unity - 0.5
            # Higher unity encourages fusion, lower unity encourages fission
            self.cluster_manager.fusion_fission_engine.fusion_health_threshold += unity_factor * 0.2
        
        # Awareness affects growth gates
        if self.growth_gate:
            awareness_factor = awareness - 0.5
            self.growth_gate.config.similarity_threshold += awareness_factor * 0.1
    
    def _update_growth_and_autophagy(self, delta_time: float) -> Dict[str, Any]:
        """Coordinate growth gate and autophagy processes"""
        results = {
            'growth_attempts': 0,
            'growth_approvals': 0,
            'autophagy_triggered': 0,
            'metabolites_from_autophagy': 0
        }
        
        # 1. Evaluate autophagy candidates
        autophagy_candidates = self.autophagy_manager.evaluate_autophagy_candidates(self.mycelial_layer)
        
        # 2. Trigger autophagy for qualified nodes
        for node_id, trigger in autophagy_candidates:
            if self.autophagy_manager.initiate_autophagy(node_id, trigger, self.mycelial_layer):
                results['autophagy_triggered'] += 1
        
        # 3. Process ongoing autophagy
        metabolites = self.autophagy_manager.process_autophagy(self.mycelial_layer, delta_time)
        results['metabolites_from_autophagy'] = len(metabolites)
        
        # 4. Add metabolites to metabolite manager
        if self.metabolite_manager and metabolites:
            for metabolite_data in metabolites:
                # Convert autophagy metabolites to MetaboliteTrace objects
                source_node = self.mycelial_layer.nodes.get(metabolite_data['source_id'])
                if source_node:
                    from .metabolites import MetaboliteType
                    self.metabolite_manager.produce_metabolite(
                        source_node=source_node,
                        metabolite_type=MetaboliteType.SEMANTIC_TRACE,
                        energy_content=metabolite_data.get('energy_content', 0.0),
                        semantic_content=metabolite_data.get('semantic_content', {})
                    )
        
        # 5. Evaluate potential new connections (growth)
        # Sample some node pairs for growth evaluation
        node_ids = list(self.mycelial_layer.nodes.keys())
        if len(node_ids) >= 2:
            # Evaluate a subset of potential connections
            max_evaluations = min(20, len(node_ids) * 2)  # Limit for performance
            
            for _ in range(max_evaluations):
                # Random node pair selection
                source_id = np.random.choice(node_ids)
                target_id = np.random.choice(node_ids)
                
                if source_id == target_id:
                    continue
                
                source_node = self.mycelial_layer.nodes[source_id]
                target_node = self.mycelial_layer.nodes[target_id]
                
                # Evaluate growth proposal
                decision, evaluation = self.growth_gate.evaluate_growth_proposal(
                    source_node, target_node, self.mycelial_layer
                )
                
                results['growth_attempts'] += 1
                
                # If approved, create the edge
                if decision.value == 'approved':
                    edge = self.mycelial_layer.add_edge(source_id, target_id)
                    if edge:
                        results['growth_approvals'] += 1
        
        return results
    
    def _update_system_metrics(self):
        """Update comprehensive system metrics"""
        layer_stats = self.mycelial_layer.get_stats()
        
        # Core metrics
        self.metrics.total_nodes = layer_stats['node_count']
        self.metrics.total_edges = layer_stats['edge_count']
        self.metrics.total_energy = layer_stats['total_energy']
        self.metrics.avg_node_health = layer_stats.get('avg_energy', 0.0)
        
        # Component metrics
        if self.nutrient_economy:
            economy_stats = self.nutrient_economy.get_economy_stats()
            self.metrics.nutrient_efficiency = economy_stats['distribution_efficiency']
            self.metrics.demand_satisfaction = economy_stats['satisfied_demand_ratio']
            self.metrics.energy_generation_rate = economy_stats['energy_generation_rate']
        
        if self.energy_flow_manager:
            flow_stats = self.energy_flow_manager.get_flow_statistics()
            self.metrics.energy_flows_active = flow_stats['current_stats']['total_transfers']
            self.metrics.flow_efficiency = flow_stats['system'].get('flow_efficiency', 0.0)
        
        if self.growth_gate:
            growth_stats = self.growth_gate.get_growth_statistics()
            self.metrics.growth_approval_rate = growth_stats['approval_rate']
        
        if self.autophagy_manager:
            autophagy_stats = self.autophagy_manager.get_autophagy_statistics()
            total_nodes = max(1, self.metrics.total_nodes)
            self.metrics.autophagy_rate = autophagy_stats['active_autophagies'] / total_nodes
        
        if self.metabolite_manager:
            metabolite_stats = self.metabolite_manager.get_metabolite_statistics()
            self.metrics.active_metabolites = metabolite_stats['totals']['active_metabolites']
            self.metrics.metabolite_absorption_rate = metabolite_stats['efficiency']['absorption_rate']
        
        if self.cluster_manager:
            cluster_stats = self.cluster_manager.get_cluster_statistics()
            self.metrics.active_clusters = cluster_stats['total_clusters']
            self.metrics.clustering_efficiency = cluster_stats['clustering_efficiency']
            fusion_fission = cluster_stats['fusion_fission_stats']
            recent_activity = fusion_fission['recent_fusion_events'] + fusion_fission['recent_fission_events']
            self.metrics.fusion_fission_activity = recent_activity / max(1, self.metrics.active_clusters)
        
        # Compute overall health and stress
        self._compute_system_health()
    
    def _compute_system_health(self):
        """Compute overall system health and stress metrics"""
        health_factors = []
        
        # Energy health
        if self.metrics.total_nodes > 0:
            energy_health = min(1.0, self.metrics.avg_node_health)
            health_factors.append(energy_health)
        
        # Economic health
        if self.nutrient_economy:
            economic_health = (self.metrics.nutrient_efficiency + self.metrics.demand_satisfaction) / 2.0
            health_factors.append(economic_health)
        
        # Structural health
        if self.metrics.total_nodes > 0:
            edge_density = self.metrics.total_edges / max(1, self.metrics.total_nodes)
            structural_health = min(1.0, edge_density / 5.0)  # Normalize to reasonable range
            health_factors.append(structural_health)
        
        # Growth health
        if self.growth_gate:
            growth_health = self.metrics.growth_approval_rate
            health_factors.append(growth_health)
        
        # Cluster health
        if self.cluster_manager:
            cluster_health = self.metrics.clustering_efficiency
            health_factors.append(cluster_health)
        
        # Overall health
        if health_factors:
            self.metrics.overall_health = sum(health_factors) / len(health_factors)
        else:
            self.metrics.overall_health = 0.5
        
        # Stress level (inverse of health with additional factors)
        stress_factors = [
            1.0 - self.metrics.overall_health,
            self.metrics.autophagy_rate,
            max(0.0, self.metrics.fusion_fission_activity - 0.1)  # High activity = stress
        ]
        
        self.metrics.stress_level = sum(stress_factors) / len(stress_factors)
        
        # Adaptation rate (how quickly system responds to changes)
        if len(self.tick_times) > 10:
            recent_times = list(self.tick_times)[-10:]
            time_variance = np.var(recent_times)
            self.metrics.adaptation_rate = max(0.0, 1.0 - time_variance * 10)  # Lower variance = better adaptation
        else:
            self.metrics.adaptation_rate = 0.5
    
    def _update_system_state(self):
        """Update overall system state based on metrics"""
        health = self.metrics.overall_health
        stress = self.metrics.stress_level
        
        if health > 0.8 and stress < 0.3:
            self.state = SystemState.HEALTHY
        elif health > 0.6 and stress < 0.6:
            # Check if recovering from critical state
            if self.state == SystemState.CRITICAL and health > self.recovery_threshold:
                self.state = SystemState.RECOVERY
            else:
                self.state = SystemState.HEALTHY
        elif health > 0.4 or stress > self.stress_threshold:
            self.state = SystemState.STRESSED
        elif stress > self.critical_threshold:
            self.state = SystemState.CRITICAL
        elif health < 0.2:
            self.state = SystemState.DORMANT
        
        # Recovery state transitions
        if self.state == SystemState.RECOVERY and health > 0.7:
            self.state = SystemState.HEALTHY
    
    def _emit_system_events(self, component_results: Dict[str, Any]):
        """Emit system-level events based on component results and state changes"""
        events = []
        
        # State change events
        if hasattr(self, '_previous_state') and self._previous_state != self.state:
            events.append({
                'type': 'state_change',
                'from_state': self._previous_state.value,
                'to_state': self.state.value,
                'timestamp': time.time(),
                'health': self.metrics.overall_health,
                'stress': self.metrics.stress_level
            })
        
        # High activity events
        if self.metrics.fusion_fission_activity > 0.5:
            events.append({
                'type': 'high_cluster_activity',
                'activity_level': self.metrics.fusion_fission_activity,
                'active_clusters': self.metrics.active_clusters,
                'timestamp': time.time()
            })
        
        # Growth events
        growth_results = component_results.get('growth_autophagy', {})
        if growth_results.get('growth_approvals', 0) > 5:
            events.append({
                'type': 'rapid_growth',
                'approvals': growth_results['growth_approvals'],
                'approval_rate': self.metrics.growth_approval_rate,
                'timestamp': time.time()
            })
        
        # Autophagy events
        if growth_results.get('autophagy_triggered', 0) > 0:
            events.append({
                'type': 'autophagy_wave',
                'nodes_affected': growth_results['autophagy_triggered'],
                'metabolites_produced': growth_results['metabolites_from_autophagy'],
                'timestamp': time.time()
            })
        
        # Process events
        for event in events:
            self._process_event(event)
        
        self._previous_state = self.state
    
    def _process_event(self, event: Dict[str, Any]):
        """Process a system event"""
        self.event_history.append(event)
        
        # Call registered event handlers
        event_type = event['type']
        for handler in self.event_handlers[event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler failed for {event_type}: {e}")
    
    def register_event_handler(self, event_type: str, handler: callable):
        """Register an event handler for a specific event type"""
        self.event_handlers[event_type].append(handler)
    
    def add_node(self, node_id: str, node_type: str = "semantic", **kwargs) -> bool:
        """Add a new node to the mycelial layer"""
        node = self.mycelial_layer.add_node(node_id, **kwargs)
        if node and hasattr(node, 'node_type'):
            node.node_type = node_type
        return node is not None
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the mycelial layer"""
        if node_id in self.mycelial_layer.nodes:
            self.mycelial_layer._remove_node(node_id)
            return True
        return False
    
    def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status for a specific node"""
        if node_id not in self.mycelial_layer.nodes:
            return None
        
        node = self.mycelial_layer.nodes[node_id]
        
        status = {
            'id': node_id,
            'energy': node.energy,
            'health': node.health,
            'pressure': node.pressure,
            'state': str(node.state) if hasattr(node, 'state') else 'unknown',
            'connections': len(node.connections),
            'cluster_id': node.cluster_id if hasattr(node, 'cluster_id') else None
        }
        
        # Add component-specific status
        if self.nutrient_economy:
            nutrient_status = self.nutrient_economy.get_node_nutrient_status(node_id)
            if nutrient_status:
                status['nutrients'] = nutrient_status
        
        if self.energy_flow_manager:
            flow_status = self.energy_flow_manager.get_node_flow_status(node_id)
            if flow_status:
                status['energy_flows'] = flow_status
        
        if self.metabolite_manager:
            metabolite_status = self.metabolite_manager.get_node_metabolite_status(node_id)
            if metabolite_status:
                status['metabolites'] = metabolite_status
        
        return status
    
    def _serialize_metrics(self) -> Dict[str, Any]:
        """Serialize system metrics for external consumption"""
        return {
            'total_nodes': self.metrics.total_nodes,
            'total_edges': self.metrics.total_edges,
            'total_energy': self.metrics.total_energy,
            'avg_node_health': self.metrics.avg_node_health,
            'nutrient_efficiency': self.metrics.nutrient_efficiency,
            'demand_satisfaction': self.metrics.demand_satisfaction,
            'energy_generation_rate': self.metrics.energy_generation_rate,
            'energy_flows_active': self.metrics.energy_flows_active,
            'flow_efficiency': self.metrics.flow_efficiency,
            'growth_approval_rate': self.metrics.growth_approval_rate,
            'autophagy_rate': self.metrics.autophagy_rate,
            'active_metabolites': self.metrics.active_metabolites,
            'metabolite_absorption_rate': self.metrics.metabolite_absorption_rate,
            'active_clusters': self.metrics.active_clusters,
            'clustering_efficiency': self.metrics.clustering_efficiency,
            'fusion_fission_activity': self.metrics.fusion_fission_activity,
            'overall_health': self.metrics.overall_health,
            'stress_level': self.metrics.stress_level,
            'adaptation_rate': self.metrics.adaptation_rate
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'state': self.state.value,
            'tick_count': self.tick_count,
            'metrics': self._serialize_metrics(),
            'components': {
                'mycelial_layer': bool(self.mycelial_layer),
                'nutrient_economy': bool(self.nutrient_economy),
                'energy_flows': bool(self.energy_flow_manager),
                'growth_gates': bool(self.growth_gate),
                'autophagy': bool(self.autophagy_manager),
                'metabolites': bool(self.metabolite_manager),
                'clustering': bool(self.cluster_manager)
            },
            'performance': {
                'avg_tick_time': sum(self.tick_times) / max(1, len(self.tick_times)) if self.tick_times else 0.0,
                'component_times': {k: sum(v) / max(1, len(v)) if v else 0.0 
                                  for k, v in self.component_times.items()}
            },
            'recent_events': list(self.event_history)[-10:]  # Last 10 events
        }

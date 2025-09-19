"""
🔄 Cluster Fusion & Fission Dynamics
====================================

Implements cluster-level dynamics for the mycelial layer:
1. Cluster Fusion: High-health, co-firing clusters can fuse, pooling mitochondrial 
   capacity and increasing energy conversion efficiency
2. Cluster Fission: High-entropy hubs can fission, sending resources out to 
   peripheral nodes to prevent stagnation

Based on DAWN documentation specifications for maintaining optimal cluster sizes
and preventing both overcentralization and fragmentation.
"""

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import logging

try:
    from scipy.spatial.distance import pdist, squareform
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

class ClusterEventType(Enum):
    """Types of cluster dynamics events"""
    FUSION = "fusion"
    FISSION = "fission"
    FORMATION = "formation"
    DISSOLUTION = "dissolution"
    REBALANCING = "rebalancing"

class ClusterState(Enum):
    """States of cluster health and activity"""
    HEALTHY = "healthy"        # Optimal size and activity
    OVERGROWN = "overgrown"    # Too large, candidate for fission
    STAGNANT = "stagnant"      # Low activity, needs stimulation
    FRAGMENTED = "fragmented"  # Too small, candidate for fusion
    DYING = "dying"           # Very low health, may dissolve

@dataclass
class ClusterMetrics:
    """Metrics for evaluating cluster health and dynamics"""
    
    # Size metrics
    node_count: int = 0
    edge_density: float = 0.0
    avg_connection_degree: float = 0.0
    
    # Energy metrics
    total_energy: float = 0.0
    avg_energy: float = 0.0
    energy_variance: float = 0.0
    energy_flow_rate: float = 0.0
    
    # Activity metrics
    co_firing_rate: float = 0.0    # How often nodes activate together
    pressure_coherence: float = 0.0  # Alignment of pressure values
    entropy_level: float = 0.0      # Overall cluster entropy
    
    # Stability metrics
    structural_integrity: float = 0.0  # How well-connected the cluster is
    temporal_stability: float = 0.0    # How stable over time
    fusion_potential: float = 0.0      # Readiness to fuse with others
    fission_pressure: float = 0.0      # Pressure to split

@dataclass
class MycelialCluster:
    """
    Represents a cluster of nodes in the mycelial layer with shared dynamics.
    """
    
    # Identity
    id: str
    created_at: float = field(default_factory=time.time)
    
    # Membership
    node_ids: Set[str] = field(default_factory=set)
    edge_ids: Set[str] = field(default_factory=set)
    
    # Cluster properties
    center_position: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    radius: float = 1.0
    
    # State and health
    state: ClusterState = ClusterState.HEALTHY
    health: float = 1.0
    stability: float = 1.0
    
    # Mitochondrial capacity (shared energy processing)
    mitochondrial_capacity: float = 10.0
    energy_conversion_efficiency: float = 0.9
    shared_energy_pool: float = 0.0
    
    # Dynamics tracking
    metrics: ClusterMetrics = field(default_factory=ClusterMetrics)
    last_update: float = field(default_factory=time.time)
    
    # Event history
    fusion_history: List[Dict[str, Any]] = field(default_factory=list)
    fission_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed properties"""
        if isinstance(self.center_position, (list, tuple)):
            self.center_position = torch.tensor(self.center_position, dtype=torch.float32)
    
    def add_node(self, node_id: str):
        """Add a node to this cluster"""
        self.node_ids.add(node_id)
    
    def remove_node(self, node_id: str):
        """Remove a node from this cluster"""
        self.node_ids.discard(node_id)
    
    def compute_center(self, mycelial_layer) -> torch.Tensor:
        """Compute the geometric center of the cluster"""
        if not self.node_ids:
            return self.center_position
        
        positions = []
        for node_id in self.node_ids:
            node = mycelial_layer.nodes.get(node_id)
            if node and hasattr(node, 'position'):
                positions.append(node.position)
        
        if positions:
            self.center_position = torch.stack(positions).mean(dim=0)
        
        return self.center_position
    
    def compute_radius(self, mycelial_layer) -> float:
        """Compute the radius of the cluster"""
        center = self.compute_center(mycelial_layer)
        max_distance = 0.0
        
        for node_id in self.node_ids:
            node = mycelial_layer.nodes.get(node_id)
            if node and hasattr(node, 'position'):
                distance = torch.norm(node.position - center).item()
                max_distance = max(max_distance, distance)
        
        self.radius = max_distance
        return self.radius

class FusionFissionEngine:
    """
    Manages cluster fusion and fission events based on health, activity, and size.
    """
    
    def __init__(self, mycelial_layer):
        self.mycelial_layer = mycelial_layer
        
        # Configuration
        self.min_cluster_size = 3
        self.max_cluster_size = 50
        self.fusion_health_threshold = 0.8
        self.fission_size_threshold = 30
        self.fission_entropy_threshold = 0.7
        
        # Fusion parameters
        self.fusion_distance_threshold = 2.0
        self.fusion_coherence_threshold = 0.6
        self.fusion_energy_bonus = 1.2
        
        # Fission parameters
        self.fission_split_ratio = 0.6  # Ratio for primary/secondary split
        self.fission_energy_cost = 0.8
        self.min_fission_distance = 1.5
        
        # Event tracking
        self.fusion_events: deque = deque(maxlen=500)
        self.fission_events: deque = deque(maxlen=500)
        self.rebalancing_events: deque = deque(maxlen=200)
        
        # Statistics
        self.total_fusions = 0
        self.total_fissions = 0
        self.total_rebalances = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("FusionFissionEngine initialized")
    
    def evaluate_fusion_candidates(self, clusters: Dict[str, MycelialCluster]) -> List[Tuple[str, str, float]]:
        """
        Find pairs of clusters that are candidates for fusion.
        Returns list of (cluster1_id, cluster2_id, fusion_score) tuples.
        """
        candidates = []
        cluster_ids = list(clusters.keys())
        
        for i, cluster1_id in enumerate(cluster_ids):
            for j, cluster2_id in enumerate(cluster_ids[i+1:], i+1):
                cluster1 = clusters[cluster1_id]
                cluster2 = clusters[cluster2_id]
                
                # Check basic requirements
                if not self._can_fuse(cluster1, cluster2):
                    continue
                
                # Compute fusion score
                fusion_score = self._compute_fusion_score(cluster1, cluster2)
                
                if fusion_score > 0.5:  # Minimum fusion threshold
                    candidates.append((cluster1_id, cluster2_id, fusion_score))
        
        # Sort by fusion score (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates
    
    def _can_fuse(self, cluster1: MycelialCluster, cluster2: MycelialCluster) -> bool:
        """Check if two clusters can fuse"""
        
        # Size check - don't create overly large clusters
        combined_size = len(cluster1.node_ids) + len(cluster2.node_ids)
        if combined_size > self.max_cluster_size:
            return False
        
        # Health check - both clusters should be reasonably healthy
        if cluster1.health < 0.5 or cluster2.health < 0.5:
            return False
        
        # State check - avoid fusing problematic clusters
        if (cluster1.state in [ClusterState.DYING, ClusterState.OVERGROWN] or
            cluster2.state in [ClusterState.DYING, ClusterState.OVERGROWN]):
            return False
        
        # Distance check
        distance = torch.norm(cluster1.center_position - cluster2.center_position).item()
        if distance > self.fusion_distance_threshold:
            return False
        
        return True
    
    def _compute_fusion_score(self, cluster1: MycelialCluster, cluster2: MycelialCluster) -> float:
        """Compute how beneficial fusing two clusters would be"""
        score = 0.0
        
        # Health compatibility (both high health = good fusion)
        health_score = (cluster1.health + cluster2.health) / 2.0
        score += 0.3 * health_score
        
        # Size compatibility (prefer similar sizes)
        size1, size2 = len(cluster1.node_ids), len(cluster2.node_ids)
        size_ratio = min(size1, size2) / max(size1, size2)
        score += 0.2 * size_ratio
        
        # Energy compatibility
        energy_diff = abs(cluster1.metrics.avg_energy - cluster2.metrics.avg_energy)
        energy_score = max(0.0, 1.0 - energy_diff)
        score += 0.2 * energy_score
        
        # Pressure coherence
        if hasattr(cluster1.metrics, 'pressure_coherence') and hasattr(cluster2.metrics, 'pressure_coherence'):
            coherence_score = (cluster1.metrics.pressure_coherence + cluster2.metrics.pressure_coherence) / 2.0
            score += 0.2 * coherence_score
        
        # Proximity bonus
        distance = torch.norm(cluster1.center_position - cluster2.center_position).item()
        proximity_score = max(0.0, 1.0 - distance / self.fusion_distance_threshold)
        score += 0.1 * proximity_score
        
        return min(1.0, score)
    
    def execute_fusion(self, cluster1_id: str, cluster2_id: str, clusters: Dict[str, MycelialCluster]) -> Optional[str]:
        """
        Execute fusion of two clusters into one.
        Returns the ID of the new merged cluster, or None if fusion failed.
        """
        with self._lock:
            cluster1 = clusters.get(cluster1_id)
            cluster2 = clusters.get(cluster2_id)
            
            if not cluster1 or not cluster2:
                return None
            
            # Create new merged cluster
            new_cluster_id = f"cluster_fusion_{time.time():.3f}"
            merged_cluster = MycelialCluster(
                id=new_cluster_id,
                created_at=time.time()
            )
            
            # Merge node membership
            merged_cluster.node_ids = cluster1.node_ids.union(cluster2.node_ids)
            merged_cluster.edge_ids = cluster1.edge_ids.union(cluster2.edge_ids)
            
            # Compute merged properties
            merged_cluster.compute_center(self.mycelial_layer)
            merged_cluster.compute_radius(self.mycelial_layer)
            
            # Merge health and stability (weighted average)
            size1, size2 = len(cluster1.node_ids), len(cluster2.node_ids)
            total_size = size1 + size2
            
            merged_cluster.health = (cluster1.health * size1 + cluster2.health * size2) / total_size
            merged_cluster.stability = (cluster1.stability * size1 + cluster2.stability * size2) / total_size
            
            # Pool mitochondrial capacity with efficiency bonus
            merged_cluster.mitochondrial_capacity = (
                cluster1.mitochondrial_capacity + cluster2.mitochondrial_capacity
            ) * self.fusion_energy_bonus
            
            merged_cluster.energy_conversion_efficiency = max(
                cluster1.energy_conversion_efficiency,
                cluster2.energy_conversion_efficiency
            ) + 0.05  # Small efficiency bonus from fusion
            
            # Merge shared energy pools
            merged_cluster.shared_energy_pool = (
                cluster1.shared_energy_pool + cluster2.shared_energy_pool
            )
            
            # Update node cluster assignments
            for node_id in merged_cluster.node_ids:
                node = self.mycelial_layer.nodes.get(node_id)
                if node:
                    node.cluster_id = new_cluster_id
            
            # Record fusion event
            fusion_event = {
                'timestamp': time.time(),
                'event_type': ClusterEventType.FUSION.value,
                'cluster1_id': cluster1_id,
                'cluster2_id': cluster2_id,
                'new_cluster_id': new_cluster_id,
                'merged_size': len(merged_cluster.node_ids),
                'fusion_score': self._compute_fusion_score(cluster1, cluster2),
                'energy_bonus': self.fusion_energy_bonus
            }
            
            self.fusion_events.append(fusion_event)
            self.total_fusions += 1
            
            # Add merged cluster history
            merged_cluster.fusion_history.append(fusion_event)
            
            # Remove old clusters and add new one
            clusters[new_cluster_id] = merged_cluster
            del clusters[cluster1_id]
            del clusters[cluster2_id]
            
            logger.info(f"Fused clusters {cluster1_id} and {cluster2_id} into {new_cluster_id}")
            return new_cluster_id
    
    def evaluate_fission_candidates(self, clusters: Dict[str, MycelialCluster]) -> List[Tuple[str, float]]:
        """
        Find clusters that are candidates for fission.
        Returns list of (cluster_id, fission_urgency) tuples.
        """
        candidates = []
        
        for cluster_id, cluster in clusters.items():
            if self._should_fission(cluster):
                urgency = self._compute_fission_urgency(cluster)
                candidates.append((cluster_id, urgency))
        
        # Sort by urgency (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _should_fission(self, cluster: MycelialCluster) -> bool:
        """Check if a cluster should undergo fission"""
        
        # Size check
        if len(cluster.node_ids) < self.min_cluster_size * 2:
            return False
        
        # Entropy check
        if cluster.metrics.entropy_level > self.fission_entropy_threshold:
            return True
        
        # Size threshold check
        if len(cluster.node_ids) > self.fission_size_threshold:
            return True
        
        # Stagnation check
        if cluster.state == ClusterState.STAGNANT and cluster.health < 0.6:
            return True
        
        return False
    
    def _compute_fission_urgency(self, cluster: MycelialCluster) -> float:
        """Compute how urgently a cluster needs to undergo fission"""
        urgency = 0.0
        
        # Size pressure
        size = len(cluster.node_ids)
        if size > self.fission_size_threshold:
            size_pressure = (size - self.fission_size_threshold) / self.fission_size_threshold
            urgency += 0.4 * min(1.0, size_pressure)
        
        # Entropy pressure
        entropy_pressure = max(0.0, cluster.metrics.entropy_level - 0.5) * 2.0
        urgency += 0.3 * entropy_pressure
        
        # Stagnation pressure
        if cluster.state == ClusterState.STAGNANT:
            stagnation_pressure = 1.0 - cluster.health
            urgency += 0.2 * stagnation_pressure
        
        # Low coherence pressure
        coherence_pressure = max(0.0, 0.5 - cluster.metrics.pressure_coherence) * 2.0
        urgency += 0.1 * coherence_pressure
        
        return min(1.0, urgency)
    
    def execute_fission(self, cluster_id: str, clusters: Dict[str, MycelialCluster]) -> Optional[Tuple[str, str]]:
        """
        Execute fission of a cluster into two smaller clusters.
        Returns (cluster1_id, cluster2_id) of new clusters, or None if fission failed.
        """
        with self._lock:
            cluster = clusters.get(cluster_id)
            if not cluster:
                return None
            
            # Get cluster nodes for analysis
            cluster_nodes = []
            for node_id in cluster.node_ids:
                node = self.mycelial_layer.nodes.get(node_id)
                if node:
                    cluster_nodes.append((node_id, node))
            
            if len(cluster_nodes) < self.min_cluster_size * 2:
                return None
            
            # Split the cluster using spatial clustering
            split_groups = self._split_cluster_nodes(cluster_nodes)
            
            if len(split_groups) != 2:
                return None
            
            # Create new clusters
            new_cluster1_id = f"cluster_fission1_{time.time():.3f}"
            new_cluster2_id = f"cluster_fission2_{time.time():.3f}"
            
            new_cluster1 = self._create_fission_cluster(new_cluster1_id, split_groups[0], cluster)
            new_cluster2 = self._create_fission_cluster(new_cluster2_id, split_groups[1], cluster)
            
            # Distribute shared resources
            self._distribute_fission_resources(cluster, new_cluster1, new_cluster2)
            
            # Update node cluster assignments
            for node_id in new_cluster1.node_ids:
                node = self.mycelial_layer.nodes.get(node_id)
                if node:
                    node.cluster_id = new_cluster1_id
            
            for node_id in new_cluster2.node_ids:
                node = self.mycelial_layer.nodes.get(node_id)
                if node:
                    node.cluster_id = new_cluster2_id
            
            # Record fission event
            fission_event = {
                'timestamp': time.time(),
                'event_type': ClusterEventType.FISSION.value,
                'original_cluster_id': cluster_id,
                'new_cluster1_id': new_cluster1_id,
                'new_cluster2_id': new_cluster2_id,
                'original_size': len(cluster.node_ids),
                'split1_size': len(new_cluster1.node_ids),
                'split2_size': len(new_cluster2.node_ids),
                'fission_urgency': self._compute_fission_urgency(cluster)
            }
            
            self.fission_events.append(fission_event)
            self.total_fissions += 1
            
            # Add to cluster histories
            new_cluster1.fission_history.append(fission_event)
            new_cluster2.fission_history.append(fission_event)
            
            # Replace old cluster with new ones
            clusters[new_cluster1_id] = new_cluster1
            clusters[new_cluster2_id] = new_cluster2
            del clusters[cluster_id]
            
            logger.info(f"Split cluster {cluster_id} into {new_cluster1_id} and {new_cluster2_id}")
            return new_cluster1_id, new_cluster2_id
    
    def _split_cluster_nodes(self, cluster_nodes: List[Tuple[str, Any]]) -> List[List[Tuple[str, Any]]]:
        """Split cluster nodes into groups using spatial clustering"""
        if len(cluster_nodes) < 4:
            return []
        
        # Extract positions for clustering
        positions = []
        for node_id, node in cluster_nodes:
            if hasattr(node, 'position'):
                positions.append(node.position.numpy())
            else:
                positions.append([0, 0, 0])  # Default position
        
        positions = np.array(positions)
        
        if HAS_SKLEARN:
            try:
                # Use AgglomerativeClustering for binary split
                clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
                labels = clustering.fit_predict(positions)
                
                # Group nodes by cluster label
                group1 = [cluster_nodes[i] for i in range(len(cluster_nodes)) if labels[i] == 0]
                group2 = [cluster_nodes[i] for i in range(len(cluster_nodes)) if labels[i] == 1]
                
                # Ensure minimum sizes
                if len(group1) >= self.min_cluster_size and len(group2) >= self.min_cluster_size:
                    return [group1, group2]
                
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
        
        # Fallback: simple spatial split
        center_pos = positions.mean(axis=0)
        distances = [np.linalg.norm(pos - center_pos) for pos in positions]
        median_distance = np.median(distances)
        
        group1 = [cluster_nodes[i] for i in range(len(cluster_nodes)) if distances[i] <= median_distance]
        group2 = [cluster_nodes[i] for i in range(len(cluster_nodes)) if distances[i] > median_distance]
        
        if len(group1) >= self.min_cluster_size and len(group2) >= self.min_cluster_size:
            return [group1, group2]
        
        return []
    
    def _create_fission_cluster(self, cluster_id: str, node_group: List[Tuple[str, Any]], 
                               parent_cluster: MycelialCluster) -> MycelialCluster:
        """Create a new cluster from a group of nodes after fission"""
        new_cluster = MycelialCluster(
            id=cluster_id,
            created_at=time.time()
        )
        
        # Add nodes
        for node_id, node in node_group:
            new_cluster.add_node(node_id)
        
        # Compute spatial properties
        new_cluster.compute_center(self.mycelial_layer)
        new_cluster.compute_radius(self.mycelial_layer)
        
        # Inherit scaled properties from parent
        size_ratio = len(node_group) / len(parent_cluster.node_ids)
        
        new_cluster.health = parent_cluster.health * 0.9  # Slight health penalty
        new_cluster.stability = parent_cluster.stability * 0.8  # Reduced stability initially
        
        # Scale mitochondrial capacity
        new_cluster.mitochondrial_capacity = parent_cluster.mitochondrial_capacity * size_ratio
        new_cluster.energy_conversion_efficiency = parent_cluster.energy_conversion_efficiency
        
        return new_cluster
    
    def _distribute_fission_resources(self, parent_cluster: MycelialCluster, 
                                    cluster1: MycelialCluster, cluster2: MycelialCluster):
        """Distribute shared resources between fission products"""
        total_energy = parent_cluster.shared_energy_pool
        
        # Apply energy cost for fission
        remaining_energy = total_energy * self.fission_energy_cost
        
        # Distribute proportionally by size
        size1, size2 = len(cluster1.node_ids), len(cluster2.node_ids)
        total_size = size1 + size2
        
        if total_size > 0:
            cluster1.shared_energy_pool = remaining_energy * (size1 / total_size)
            cluster2.shared_energy_pool = remaining_energy * (size2 / total_size)

class ClusterManager:
    """
    Main manager for cluster dynamics in the mycelial layer.
    Orchestrates cluster formation, maintenance, fusion, and fission.
    """
    
    def __init__(self, mycelial_layer):
        self.mycelial_layer = mycelial_layer
        self.fusion_fission_engine = FusionFissionEngine(mycelial_layer)
        
        # Cluster storage
        self.clusters: Dict[str, MycelialCluster] = {}
        self.cluster_counter = 0
        
        # Configuration
        self.auto_clustering_enabled = True
        self.clustering_interval = 10  # Ticks between clustering updates
        self.metrics_update_interval = 5  # Ticks between metrics updates
        
        # State tracking
        self.last_clustering_tick = 0
        self.last_metrics_update = 0
        
        # Statistics
        self.total_clusters_created = 0
        self.total_clusters_dissolved = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("ClusterManager initialized")
    
    def tick_update(self, delta_time: float = 1.0) -> Dict[str, Any]:
        """Perform one tick of cluster management"""
        with self._lock:
            current_tick = self.mycelial_layer.tick_count
            
            results = {
                'clusters_active': len(self.clusters),
                'fusion_events': 0,
                'fission_events': 0,
                'clusters_formed': 0,
                'clusters_dissolved': 0
            }
            
            # Update cluster metrics periodically
            if current_tick - self.last_metrics_update >= self.metrics_update_interval:
                self._update_all_cluster_metrics()
                self.last_metrics_update = current_tick
            
            # Perform clustering operations periodically
            if current_tick - self.last_clustering_tick >= self.clustering_interval:
                if self.auto_clustering_enabled:
                    cluster_results = self._perform_clustering_operations()
                    results.update(cluster_results)
                self.last_clustering_tick = current_tick
            
            # Clean up empty or invalid clusters
            dissolved = self._cleanup_invalid_clusters()
            results['clusters_dissolved'] += dissolved
            
            return results
    
    def _update_all_cluster_metrics(self):
        """Update metrics for all active clusters"""
        for cluster in self.clusters.values():
            self._update_cluster_metrics(cluster)
    
    def _update_cluster_metrics(self, cluster: MycelialCluster):
        """Update metrics for a single cluster"""
        if not cluster.node_ids:
            return
        
        # Collect node data
        cluster_nodes = []
        total_energy = 0.0
        total_pressure = 0.0
        total_entropy = 0.0
        connection_counts = []
        
        for node_id in cluster.node_ids:
            node = self.mycelial_layer.nodes.get(node_id)
            if node:
                cluster_nodes.append(node)
                total_energy += node.energy
                total_pressure += node.pressure
                total_entropy += node.entropy
                connection_counts.append(len(node.connections))
        
        if not cluster_nodes:
            return
        
        node_count = len(cluster_nodes)
        
        # Basic metrics
        cluster.metrics.node_count = node_count
        cluster.metrics.total_energy = total_energy
        cluster.metrics.avg_energy = total_energy / node_count
        cluster.metrics.avg_connection_degree = sum(connection_counts) / node_count
        
        # Energy variance
        energy_variance = sum((node.energy - cluster.metrics.avg_energy) ** 2 for node in cluster_nodes) / node_count
        cluster.metrics.energy_variance = energy_variance
        
        # Pressure coherence (how aligned pressures are)
        avg_pressure = total_pressure / node_count
        pressure_variance = sum((node.pressure - avg_pressure) ** 2 for node in cluster_nodes) / node_count
        cluster.metrics.pressure_coherence = max(0.0, 1.0 - pressure_variance)
        
        # Entropy level
        cluster.metrics.entropy_level = total_entropy / node_count
        
        # Edge density calculation
        cluster_edges = self._count_cluster_edges(cluster)
        max_possible_edges = node_count * (node_count - 1) / 2
        cluster.metrics.edge_density = cluster_edges / max(1, max_possible_edges)
        
        # Update cluster state based on metrics
        self._update_cluster_state(cluster)
        
        cluster.last_update = time.time()
    
    def _count_cluster_edges(self, cluster: MycelialCluster) -> int:
        """Count edges within a cluster"""
        edge_count = 0
        
        for node_id in cluster.node_ids:
            node = self.mycelial_layer.nodes.get(node_id)
            if node:
                # Count connections to other nodes in the same cluster
                for connected_id in node.connections.keys():
                    if connected_id in cluster.node_ids:
                        edge_count += 1
        
        return edge_count // 2  # Each edge counted twice
    
    def _update_cluster_state(self, cluster: MycelialCluster):
        """Update cluster state based on current metrics"""
        size = cluster.metrics.node_count
        
        if size == 0:
            cluster.state = ClusterState.DYING
        elif size < self.fusion_fission_engine.min_cluster_size:
            cluster.state = ClusterState.FRAGMENTED
        elif size > self.fusion_fission_engine.fission_size_threshold:
            cluster.state = ClusterState.OVERGROWN
        elif cluster.metrics.entropy_level > 0.8:
            cluster.state = ClusterState.STAGNANT
        elif cluster.health > 0.7 and cluster.metrics.pressure_coherence > 0.6:
            cluster.state = ClusterState.HEALTHY
        else:
            cluster.state = ClusterState.STAGNANT
        
        # Update health based on metrics
        health_factors = [
            cluster.metrics.pressure_coherence,
            min(1.0, cluster.metrics.avg_energy),
            1.0 - cluster.metrics.entropy_level,
            min(1.0, cluster.metrics.edge_density * 2)  # Boost for connectivity
        ]
        
        cluster.health = sum(health_factors) / len(health_factors)
    
    def _perform_clustering_operations(self) -> Dict[str, Any]:
        """Perform fusion, fission, and other clustering operations"""
        results = {
            'fusion_events': 0,
            'fission_events': 0,
            'clusters_formed': 0
        }
        
        # 1. Evaluate and execute fusions
        fusion_candidates = self.fusion_fission_engine.evaluate_fusion_candidates(self.clusters)
        
        for cluster1_id, cluster2_id, fusion_score in fusion_candidates[:3]:  # Limit to top 3
            if cluster1_id in self.clusters and cluster2_id in self.clusters:
                new_cluster_id = self.fusion_fission_engine.execute_fusion(
                    cluster1_id, cluster2_id, self.clusters
                )
                if new_cluster_id:
                    results['fusion_events'] += 1
        
        # 2. Evaluate and execute fissions
        fission_candidates = self.fusion_fission_engine.evaluate_fission_candidates(self.clusters)
        
        for cluster_id, urgency in fission_candidates[:2]:  # Limit to top 2
            if cluster_id in self.clusters and urgency > 0.7:
                new_cluster_ids = self.fusion_fission_engine.execute_fission(
                    cluster_id, self.clusters
                )
                if new_cluster_ids:
                    results['fission_events'] += 1
        
        # 3. Form new clusters for unclustered nodes
        unclustered_nodes = self._find_unclustered_nodes()
        if len(unclustered_nodes) >= self.fusion_fission_engine.min_cluster_size:
            new_clusters = self._form_new_clusters(unclustered_nodes)
            results['clusters_formed'] += len(new_clusters)
        
        return results
    
    def _find_unclustered_nodes(self) -> List[str]:
        """Find nodes that don't belong to any cluster"""
        clustered_nodes = set()
        for cluster in self.clusters.values():
            clustered_nodes.update(cluster.node_ids)
        
        all_nodes = set(self.mycelial_layer.nodes.keys())
        unclustered = all_nodes - clustered_nodes
        
        return list(unclustered)
    
    def _form_new_clusters(self, unclustered_nodes: List[str]) -> List[str]:
        """Form new clusters from unclustered nodes"""
        if len(unclustered_nodes) < self.fusion_fission_engine.min_cluster_size:
            return []
        
        # Use spatial clustering to group nearby nodes
        node_positions = []
        valid_nodes = []
        
        for node_id in unclustered_nodes:
            node = self.mycelial_layer.nodes.get(node_id)
            if node and hasattr(node, 'position'):
                node_positions.append(node.position.numpy())
                valid_nodes.append(node_id)
        
        if len(valid_nodes) < self.fusion_fission_engine.min_cluster_size:
            return []
        
        if HAS_SKLEARN:
            try:
                # Use DBSCAN for adaptive clustering
                clustering = DBSCAN(eps=1.5, min_samples=self.fusion_fission_engine.min_cluster_size)
                labels = clustering.fit_predict(node_positions)
                
                # Group nodes by cluster label
                cluster_groups = defaultdict(list)
                for i, label in enumerate(labels):
                    if label != -1:  # -1 is noise in DBSCAN
                        cluster_groups[label].append(valid_nodes[i])
                
                # Create clusters for each group
                new_cluster_ids = []
                for group_nodes in cluster_groups.values():
                    if len(group_nodes) >= self.fusion_fission_engine.min_cluster_size:
                        cluster_id = self._create_cluster_from_nodes(group_nodes)
                        if cluster_id:
                            new_cluster_ids.append(cluster_id)
                
                return new_cluster_ids
                
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
        
        # Fallback: simple distance-based clustering
        return []
    
    def _create_cluster_from_nodes(self, node_ids: List[str]) -> Optional[str]:
        """Create a new cluster from a list of nodes"""
        if len(node_ids) < self.fusion_fission_engine.min_cluster_size:
            return None
        
        self.cluster_counter += 1
        cluster_id = f"cluster_{self.cluster_counter}_{time.time():.3f}"
        
        cluster = MycelialCluster(
            id=cluster_id,
            created_at=time.time()
        )
        
        # Add nodes to cluster
        for node_id in node_ids:
            cluster.add_node(node_id)
            node = self.mycelial_layer.nodes.get(node_id)
            if node:
                node.cluster_id = cluster_id
        
        # Initialize cluster properties
        cluster.compute_center(self.mycelial_layer)
        cluster.compute_radius(self.mycelial_layer)
        cluster.mitochondrial_capacity = len(node_ids) * 2.0  # Base capacity
        
        # Store cluster
        self.clusters[cluster_id] = cluster
        self.total_clusters_created += 1
        
        logger.info(f"Created new cluster {cluster_id} with {len(node_ids)} nodes")
        return cluster_id
    
    def _cleanup_invalid_clusters(self) -> int:
        """Remove empty or invalid clusters"""
        to_remove = []
        
        for cluster_id, cluster in self.clusters.items():
            # Remove clusters with no valid nodes
            valid_nodes = [nid for nid in cluster.node_ids 
                          if nid in self.mycelial_layer.nodes]
            
            if len(valid_nodes) == 0:
                to_remove.append(cluster_id)
            elif len(valid_nodes) != len(cluster.node_ids):
                # Update cluster with only valid nodes
                cluster.node_ids = set(valid_nodes)
        
        # Remove empty clusters
        for cluster_id in to_remove:
            del self.clusters[cluster_id]
            self.total_clusters_dissolved += 1
        
        return len(to_remove)
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics"""
        if not self.clusters:
            return {
                'total_clusters': 0,
                'total_nodes_clustered': 0,
                'clustering_efficiency': 0.0
            }
        
        # Analyze cluster distribution
        cluster_sizes = [len(cluster.node_ids) for cluster in self.clusters.values()]
        cluster_healths = [cluster.health for cluster in self.clusters.values()]
        
        state_counts = defaultdict(int)
        for cluster in self.clusters.values():
            state_counts[cluster.state.value] += 1
        
        total_nodes_clustered = sum(cluster_sizes)
        total_nodes = len(self.mycelial_layer.nodes)
        clustering_efficiency = total_nodes_clustered / max(1, total_nodes)
        
        return {
            'total_clusters': len(self.clusters),
            'total_nodes_clustered': total_nodes_clustered,
            'clustering_efficiency': clustering_efficiency,
            'cluster_size_stats': {
                'min': min(cluster_sizes) if cluster_sizes else 0,
                'max': max(cluster_sizes) if cluster_sizes else 0,
                'avg': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
            },
            'cluster_health_stats': {
                'avg': sum(cluster_healths) / len(cluster_healths) if cluster_healths else 0,
                'min': min(cluster_healths) if cluster_healths else 0,
                'max': max(cluster_healths) if cluster_healths else 0
            },
            'state_distribution': dict(state_counts),
            'fusion_fission_stats': {
                'total_fusions': self.fusion_fission_engine.total_fusions,
                'total_fissions': self.fusion_fission_engine.total_fissions,
                'recent_fusion_events': len(self.fusion_fission_engine.fusion_events),
                'recent_fission_events': len(self.fusion_fission_engine.fission_events)
            },
            'lifecycle_stats': {
                'clusters_created': self.total_clusters_created,
                'clusters_dissolved': self.total_clusters_dissolved,
                'net_clusters': self.total_clusters_created - self.total_clusters_dissolved
            }
        }

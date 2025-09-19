"""
Bee Tracer - Schema Pollinator

Cross-cluster pollination tracer that moves between nodes, clusters, and layers,
transferring signals and nutrients. Ensures ideas, traces, and schema states
cross boundaries rather than staying isolated. Prevents cognitive siloing.
"""

from typing import Dict, Any, List, Tuple, Optional
from .base_tracer import BaseTracer, TracerType, TracerReport, AlertSeverity, TracerSpawnConditions
import random
import logging

logger = logging.getLogger(__name__)


class BeeTracer(BaseTracer):
    """Cross-cluster pollination tracer - the pollinator of DAWN"""
    
    def __init__(self, tracer_id: str = None):
        super().__init__(tracer_id)
        self.collected_payload = {
            'nutrients': 0.0,
            'pigment_bias': [0.0, 0.0, 0.0],  # RGB
            'trace_ids': [],
            'pattern_fragments': [],
            'coherence_factor': 0.0
        }
        self.origin_cluster = None
        self.target_cluster = None
        self.pollination_phase = "collection"  # collection -> transfer -> distribution
        self.pollination_complete = False
        
    @property
    def tracer_type(self) -> TracerType:
        return TracerType.BEE
    
    @property
    def base_lifespan(self) -> int:
        return 6  # 3-10 ticks average - efficient and focused
    
    @property
    def base_nutrient_cost(self) -> float:
        return 0.2  # Low, scalable
    
    @property
    def archetype_description(self) -> str:
        return "Schema pollinator - transfers fragments/pigments between clusters"
    
    def spawn_conditions_met(self, context: Dict[str, Any]) -> bool:
        """Spawn when clusters are isolated or imbalanced"""
        # Check for cluster isolation
        if TracerSpawnConditions.cluster_isolation(context, ratio_threshold=0.3):
            return True
            
        # Check for nutrient imbalances between clusters
        clusters = context.get('schema_clusters', [])
        if len(clusters) >= 2:
            nutrient_levels = [c.get('nutrient_level', 0) for c in clusters]
            if nutrient_levels:
                nutrient_variance = max(nutrient_levels) - min(nutrient_levels)
                if nutrient_variance > 0.5:
                    return True
        
        # SHI requests for pollination
        if context.get('pollination_requested', False):
            return True
            
        # Schema fragmentation indicators
        fragmentation_score = context.get('schema_fragmentation', 0.0)
        if fragmentation_score > 0.6:
            return True
            
        # Diversity loss in clusters
        clusters = context.get('schema_clusters', [])
        low_diversity_clusters = [c for c in clusters if c.get('diversity_score', 1.0) < 0.4]
        if len(low_diversity_clusters) > 1:
            return True
            
        return False
    
    def observe(self, context: Dict[str, Any]) -> List[TracerReport]:
        """Execute pollination process: collect -> transfer -> distribute"""
        reports = []
        current_tick = context.get('tick_id', 0)
        
        if self.pollination_phase == "collection":
            success = self._collection_phase(context)
            if success:
                self.pollination_phase = "transfer"
                reports.append(self._create_phase_report(context, "collection_complete"))
                
        elif self.pollination_phase == "transfer":
            success = self._transfer_phase(context)
            if success:
                self.pollination_phase = "distribution"
                reports.append(self._create_phase_report(context, "transfer_complete"))
                
        elif self.pollination_phase == "distribution":
            success = self._distribution_phase(context)
            if success:
                self.pollination_complete = True
                reports.append(self._create_pollination_report(context))
                
        return reports
    
    def _collection_phase(self, context: Dict[str, Any]) -> bool:
        """Phase 1: Select clusters and collect payload from source"""
        clusters = context.get('schema_clusters', [])
        
        if not clusters:
            return False
            
        # Select origin and target clusters
        self._select_clusters(clusters)
        
        if self.origin_cluster and self.target_cluster:
            # Collect payload from origin cluster
            return self._collect_payload(clusters)
            
        return False
    
    def _transfer_phase(self, context: Dict[str, Any]) -> bool:
        """Phase 2: Process and prepare payload for distribution"""
        if not self.collected_payload['nutrients'] and not self.collected_payload['trace_ids']:
            return False
            
        # Enhance payload during transfer
        self._enhance_payload(context)
        
        # Simulate travel time/processing
        return True
    
    def _distribution_phase(self, context: Dict[str, Any]) -> bool:
        """Phase 3: Distribute payload to target cluster"""
        clusters = context.get('schema_clusters', [])
        return self._distribute_payload(clusters)
    
    def _select_clusters(self, clusters: List[Dict[str, Any]]) -> None:
        """Select origin (nutrient-rich) and target (nutrient-poor) clusters"""
        if len(clusters) < 2:
            return
            
        # Sort clusters by nutrient level and connectivity
        cluster_scores = []
        for cluster in clusters:
            nutrient_level = cluster.get('nutrient_level', 0)
            connectivity = cluster.get('connectivity', 0)
            diversity = cluster.get('diversity_score', 0)
            
            # Rich clusters: high nutrients, good connectivity
            richness_score = nutrient_level * (1 + connectivity) * (1 + diversity)
            
            # Poor clusters: low nutrients, low diversity (need help)
            neediness_score = (1 - nutrient_level) * (1 - diversity) * (1 + connectivity)
            
            cluster_scores.append((cluster, richness_score, neediness_score))
        
        # Select richest cluster as origin
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        self.origin_cluster = cluster_scores[0][0].get('id')
        
        # Select neediest cluster as target (but not the same as origin)
        cluster_scores.sort(key=lambda x: x[2], reverse=True)
        for cluster, _, _ in cluster_scores:
            if cluster.get('id') != self.origin_cluster:
                self.target_cluster = cluster.get('id')
                break
    
    def _collect_payload(self, clusters: List[Dict[str, Any]]) -> bool:
        """Collect payload from origin cluster"""
        origin = next((c for c in clusters if c.get('id') == self.origin_cluster), None)
        
        if not origin:
            return False
            
        # Collect nutrients (take 10% of origin's nutrients)
        origin_nutrients = origin.get('nutrient_level', 0)
        self.collected_payload['nutrients'] = origin_nutrients * 0.1
        
        # Collect pigment bias
        self.collected_payload['pigment_bias'] = origin.get('pigment_bias', [0, 0, 0])[:3]
        
        # Collect recent traces
        self.collected_payload['trace_ids'] = origin.get('recent_traces', [])[:5]
        
        # Collect pattern fragments from recent Ant reports
        self.collected_payload['pattern_fragments'] = origin.get('pattern_fragments', [])[:3]
        
        # Collect coherence factor
        self.collected_payload['coherence_factor'] = origin.get('coherence', 0)
        
        return True
    
    def _enhance_payload(self, context: Dict[str, Any]) -> None:
        """Enhance payload during transfer phase"""
        # Cross-link patterns with global context
        global_entropy = context.get('entropy', 0)
        global_pressure = context.get('pressure', 0)
        
        # Adjust nutrient value based on global conditions
        if global_entropy > 0.7:  # High entropy - nutrients more valuable
            self.collected_payload['nutrients'] *= 1.2
        elif global_entropy < 0.3:  # Low entropy - standard value
            self.collected_payload['nutrients'] *= 0.9
            
        # Enhance pigment bias with global pressure influence
        if global_pressure > 0.6:
            # High pressure - bias toward blue (stability)
            self.collected_payload['pigment_bias'][2] += 0.1
        elif global_pressure < 0.4:
            # Low pressure - bias toward red (activation)
            self.collected_payload['pigment_bias'][0] += 0.1
    
    def _distribute_payload(self, clusters: List[Dict[str, Any]]) -> bool:
        """Distribute payload to target cluster"""
        target = next((c for c in clusters if c.get('id') == self.target_cluster), None)
        
        if not target or self.collected_payload['nutrients'] <= 0:
            return False
            
        # This would integrate with the actual schema system
        # For now, we simulate successful distribution
        logger.debug(f"Bee {self.tracer_id} distributing {self.collected_payload['nutrients']:.3f} "
                    f"nutrients from {self.origin_cluster} to {self.target_cluster}")
        
        return True
    
    def _create_phase_report(self, context: Dict[str, Any], phase: str) -> TracerReport:
        """Create a report for phase completion"""
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.INFO,
            report_type="pollination_phase",
            metadata={
                "phase": phase,
                "origin_cluster": self.origin_cluster,
                "target_cluster": self.target_cluster,
                "payload_size": self.collected_payload['nutrients'],
                "phase_duration": context.get('tick_id', 0) - self.spawn_tick
            }
        )
    
    def _create_pollination_report(self, context: Dict[str, Any]) -> TracerReport:
        """Create final pollination completion report"""
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.INFO,
            report_type="pollination",
            metadata={
                "origin_cluster": self.origin_cluster,
                "destination_cluster": self.target_cluster,
                "payload": self.collected_payload.copy(),
                "pollination_duration": context.get('tick_id', 0) - self.spawn_tick,
                "success": True,
                "cross_link_strength": self._calculate_cross_link_strength()
            }
        )
    
    def _calculate_cross_link_strength(self) -> float:
        """Calculate the strength of the cross-link created"""
        base_strength = 0.5
        
        # Strengthen based on payload diversity
        payload_diversity = 0.0
        if self.collected_payload['trace_ids']:
            payload_diversity += 0.2
        if sum(self.collected_payload['pigment_bias']) > 0:
            payload_diversity += 0.2
        if self.collected_payload['pattern_fragments']:
            payload_diversity += 0.2
            
        # Strengthen based on coherence transfer
        coherence_bonus = self.collected_payload['coherence_factor'] * 0.3
        
        return min(1.0, base_strength + payload_diversity + coherence_bonus)
    
    def should_retire(self, context: Dict[str, Any]) -> bool:
        """Retire after successful pollination or lifespan"""
        current_tick = context.get('tick_id', 0)
        age = self.get_age(current_tick)
        
        # Retire after lifespan
        if age >= self.base_lifespan:
            return True
            
        # Retire after successful pollination
        if self.pollination_complete:
            return True
            
        # Early retirement if clusters not found
        if age > 2 and not self.origin_cluster:
            return True
            
        # Early retirement if payload collection failed
        if (age > 3 and self.pollination_phase == "collection" and 
            self.collected_payload['nutrients'] == 0):
            return True
            
        return False

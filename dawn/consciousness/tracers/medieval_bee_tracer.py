"""
Medieval Bee Tracer - Historical Anchor & Legacy Carrier

Specialized pollinator dedicated to anchoring DAWN in historical memory.
Unlike the regular Bee (light, present-focused pollinator), the Medieval Bee
carries archaic, long-lived traces across epochs, ensuring deep schema patterns
and "ancestral DNA" in cognition remain woven into present thought.
"""

from typing import Dict, Any, List, Optional
from .base_tracer import BaseTracer, TracerType, TracerReport, AlertSeverity
import logging

logger = logging.getLogger(__name__)


class MedievalBeeTracer(BaseTracer):
    """Historical anchor and legacy carrier - the heritage keeper of DAWN"""
    
    def __init__(self, tracer_id: str = None):
        super().__init__(tracer_id)
        self.heritage_payload = {
            'archival_traces': [],
            'pigment_legacy': [0.0, 0.0, 0.0],  # RGB heritage
            'schema_links': [],
            'epochal_memories': [],
            'ancestral_patterns': [],
            'temporal_anchors': []
        }
        
        # Mythic Archetypal Properties
        self.mythic_role = "Pollinator of Time"  # As per mythic documentation
        self.archetypal_bias = "heritage_continuity"
        self.medieval_wisdom = 0.0  # Accumulated wisdom from medieval cosmology
        self.order_memory_strength = 0.0  # Bees symbolized order and memory
        self.pigment_transport_capacity = []  # Pigment vectors being transported
        self.cross_epochal_links = {}  # Links between different epochs
        self.collection_epoch = None
        self.target_epoch = None
        self.heritage_phase = "identification"  # identification -> collection -> preservation -> integration
        self.integration_complete = False
        self.heritage_significance = 0.0
        
    @property
    def tracer_type(self) -> TracerType:
        return TracerType.MEDIEVAL_BEE
    
    @property
    def base_lifespan(self) -> int:
        return 75  # Long-lived, epoch scale
    
    @property
    def base_nutrient_cost(self) -> float:
        return 2.5  # Very high cost - rare spawning
    
    @property
    def archetype_description(self) -> str:
        return "Historical anchor - carries archaic traces across epochs"
    
    def spawn_conditions_met(self, context: Dict[str, Any]) -> bool:
        """Spawn at epoch boundaries or when heritage drift is detected"""
        current_tick = context.get('tick_id', 0)
        
        # Primary trigger: Epoch boundaries (every 1000 ticks)
        if current_tick % 1000 == 0 and current_tick > 0:
            return True
        
        # Schema drift threatens core anchors
        heritage_drift = context.get('heritage_drift', 0.0)
        if heritage_drift > context.get('heritage_drift_threshold', 0.7):
            return True
        
        # Manual operator flags for legacy retention
        if context.get('preserve_heritage_requested', False):
            return True
        
        # Extreme schema instability requiring historical anchoring
        schema_instability = context.get('schema_instability', 0.0)
        if schema_instability > 0.8:
            return True
        
        # Memory consolidation events requiring heritage integration
        consolidation_event = context.get('major_consolidation_event', False)
        if consolidation_event:
            return True
        
        # Long periods without heritage preservation (emergency spawn)
        last_heritage_preservation = context.get('last_heritage_preservation_tick', 0)
        if current_tick - last_heritage_preservation > 2000:  # 2000 ticks without heritage work
            return True
        
        return False
    
    def observe(self, context: Dict[str, Any]) -> List[TracerReport]:
        """Execute heritage preservation process: identify -> collect -> preserve -> integrate"""
        reports = []
        current_tick = context.get('tick_id', 0)
        
        if self.heritage_phase == "identification":
            success = self._identification_phase(context)
            if success:
                self.heritage_phase = "collection"
                reports.append(self._create_identification_report(context))
                
        elif self.heritage_phase == "collection":
            success = self._collection_phase(context)
            if success:
                self.heritage_phase = "preservation"
                reports.append(self._create_collection_report(context))
                
        elif self.heritage_phase == "preservation":
            success = self._preservation_phase(context)
            if success:
                self.heritage_phase = "integration"
                reports.append(self._create_preservation_report(context))
                
        elif self.heritage_phase == "integration":
            success = self._integration_phase(context)
            if success:
                self.integration_complete = True
                reports.append(self._create_integration_report(context))
        
        return reports
    
    def _identification_phase(self, context: Dict[str, Any]) -> bool:
        """Phase 1: Identify heritage sources and target epoch"""
        current_tick = context.get('tick_id', 0)
        
        # Determine collection epoch (source of heritage)
        self.collection_epoch = self._identify_collection_epoch(context)
        
        # Determine target epoch (where heritage will be integrated)
        self.target_epoch = current_tick
        
        # Assess heritage significance
        self.heritage_significance = self._assess_heritage_significance(context)
        
        return self.collection_epoch is not None
    
    def _collection_phase(self, context: Dict[str, Any]) -> bool:
        """Phase 2: Collect archival traces and heritage patterns"""
        # Collect archival traces from Owl summaries
        self._collect_archival_traces(context)
        
        # Collect long-persisting schema edges (heritage links)
        self._collect_heritage_schema_links(context)
        
        # Collect epochal memories (significant past events)
        self._collect_epochal_memories(context)
        
        # Collect ancestral patterns (deep recurring motifs)
        self._collect_ancestral_patterns(context)
        
        # Collect temporal anchors (fixed reference points)
        self._collect_temporal_anchors(context)
        
        # Extract pigment legacy from historical ash
        self._extract_pigment_legacy(context)
        
        return len(self.heritage_payload['archival_traces']) > 0
    
    def _preservation_phase(self, context: Dict[str, Any]) -> bool:
        """Phase 3: Process and prepare heritage for long-term preservation"""
        # Validate heritage integrity
        integrity_check = self._validate_heritage_integrity()
        
        if not integrity_check:
            logger.warning(f"Medieval Bee {self.tracer_id} detected heritage corruption")
            return False
        
        # Compress heritage data for efficient storage
        self._compress_heritage_payload()
        
        # Add preservation metadata
        self._add_preservation_metadata(context)
        
        # Create heritage checksums for integrity verification
        self._create_heritage_checksums()
        
        return True
    
    def _integration_phase(self, context: Dict[str, Any]) -> bool:
        """Phase 4: Integrate heritage into current schema and memory systems"""
        # Integrate archival traces into current memory
        traces_integrated = self._integrate_archival_traces(context)
        
        # Reinforce heritage schema links
        links_reinforced = self._reinforce_heritage_links(context)
        
        # Deposit epochal memories into long-term storage
        memories_deposited = self._deposit_epochal_memories(context)
        
        # Weave ancestral patterns into current schema
        patterns_woven = self._weave_ancestral_patterns(context)
        
        # Establish temporal anchors
        anchors_established = self._establish_temporal_anchors(context)
        
        # Apply pigment legacy to current ash
        pigment_applied = self._apply_pigment_legacy(context)
        
        # Heritage successfully integrated if most operations succeeded
        success_count = sum([traces_integrated, links_reinforced, memories_deposited, 
                           patterns_woven, anchors_established, pigment_applied])
        
        return success_count >= 4  # At least 4 out of 6 operations successful
    
    def _identify_collection_epoch(self, context: Dict[str, Any]) -> Optional[int]:
        """Identify the source epoch for heritage collection"""
        current_tick = context.get('tick_id', 0)
        
        # Look for significant past epochs with high stability
        epoch_history = context.get('epoch_history', [])
        
        if not epoch_history:
            # Default to looking back 1000 ticks if no epoch history
            return max(0, current_tick - 1000)
        
        # Find epochs with high stability and significance
        stable_epochs = [
            epoch for epoch in epoch_history 
            if epoch.get('stability_score', 0) > 0.7 and epoch.get('significance', 0) > 0.6
        ]
        
        if stable_epochs:
            # Select the most recent stable epoch
            stable_epochs.sort(key=lambda x: x.get('end_tick', 0), reverse=True)
            return stable_epochs[0].get('start_tick')
        
        # Fallback: use the most recent epoch with any recorded data
        if epoch_history:
            return epoch_history[-1].get('start_tick')
        
        return None
    
    def _assess_heritage_significance(self, context: Dict[str, Any]) -> float:
        """Assess the significance of available heritage"""
        significance_factors = []
        
        # Schema stability in collection epoch
        epoch_stability = context.get('collection_epoch_stability', 0.5)
        significance_factors.append(epoch_stability)
        
        # Presence of major decisions or events
        major_events = context.get('major_historical_events', [])
        event_significance = min(1.0, len(major_events) / 10.0)  # Normalize to 10 events
        significance_factors.append(event_significance)
        
        # Schema pattern persistence
        pattern_persistence = context.get('pattern_persistence_score', 0.5)
        significance_factors.append(pattern_persistence)
        
        # Cultural/cognitive artifacts present
        artifact_richness = context.get('cognitive_artifact_richness', 0.5)
        significance_factors.append(artifact_richness)
        
        return sum(significance_factors) / len(significance_factors)
    
    def _collect_archival_traces(self, context: Dict[str, Any]) -> None:
        """Collect archival traces from Owl summaries and historical logs"""
        # Simulated collection from Owl archives
        owl_archives = context.get('owl_archives', [])
        
        for archive in owl_archives:
            if archive.get('epoch_start', 0) == self.collection_epoch:
                # Extract significant traces from this epoch
                epoch_summary = archive.get('epoch_summary', {})
                
                if epoch_summary.get('significance', 0) > 0.6:
                    self.heritage_payload['archival_traces'].append({
                        'source': 'owl_archive',
                        'epoch': self.collection_epoch,
                        'summary': epoch_summary,
                        'significance': epoch_summary.get('significance', 0.5),
                        'preservation_priority': 'high' if epoch_summary.get('significance', 0) > 0.8 else 'medium'
                    })
        
        # Collect traces from fractal memory if available
        fractal_traces = context.get('fractal_memory_traces', [])
        for trace in fractal_traces:
            if (trace.get('epoch_origin', 0) == self.collection_epoch and 
                trace.get('persistence_score', 0) > 0.7):
                
                self.heritage_payload['archival_traces'].append({
                    'source': 'fractal_memory',
                    'epoch': self.collection_epoch,
                    'trace_id': trace.get('id'),
                    'persistence_score': trace.get('persistence_score'),
                    'preservation_priority': 'high'
                })
    
    def _collect_heritage_schema_links(self, context: Dict[str, Any]) -> None:
        """Collect long-persisting schema edges that represent heritage connections"""
        historical_schema = context.get('historical_schema_data', {})
        
        if self.collection_epoch in historical_schema:
            epoch_schema = historical_schema[self.collection_epoch]
            persistent_edges = epoch_schema.get('persistent_edges', [])
            
            for edge in persistent_edges:
                if edge.get('persistence_duration', 0) > 500:  # Long-lived edges
                    self.heritage_payload['schema_links'].append({
                        'edge_id': edge.get('id'),
                        'source': edge.get('source'),
                        'target': edge.get('target'),
                        'strength': edge.get('strength', 0.5),
                        'persistence_duration': edge.get('persistence_duration'),
                        'heritage_value': edge.get('heritage_value', 0.5)
                    })
    
    def _collect_epochal_memories(self, context: Dict[str, Any]) -> None:
        """Collect significant memories from the heritage epoch"""
        memory_archives = context.get('memory_archives', [])
        
        for memory in memory_archives:
            if (memory.get('epoch', 0) == self.collection_epoch and 
                memory.get('significance', 0) > 0.7):
                
                self.heritage_payload['epochal_memories'].append({
                    'memory_id': memory.get('id'),
                    'content_summary': memory.get('summary', ''),
                    'significance': memory.get('significance'),
                    'emotional_weight': memory.get('emotional_weight', 0.5),
                    'decision_influence': memory.get('decision_influence', 0.5),
                    'preservation_reason': memory.get('preservation_reason', 'high_significance')
                })
    
    def _collect_ancestral_patterns(self, context: Dict[str, Any]) -> None:
        """Collect deep recurring patterns that represent cognitive ancestry"""
        pattern_archives = context.get('pattern_archives', [])
        
        for pattern in pattern_archives:
            if (pattern.get('first_emergence_epoch', 0) <= self.collection_epoch and
                pattern.get('recurrence_count', 0) > 10):  # Well-established patterns
                
                self.heritage_payload['ancestral_patterns'].append({
                    'pattern_id': pattern.get('id'),
                    'pattern_type': pattern.get('type'),
                    'recurrence_count': pattern.get('recurrence_count'),
                    'stability_score': pattern.get('stability_score', 0.5),
                    'influence_scope': pattern.get('influence_scope', []),
                    'evolutionary_stage': pattern.get('evolutionary_stage', 'mature')
                })
    
    def _collect_temporal_anchors(self, context: Dict[str, Any]) -> None:
        """Collect temporal reference points that provide historical continuity"""
        anchor_points = context.get('temporal_anchor_points', [])
        
        for anchor in anchor_points:
            if anchor.get('epoch', 0) == self.collection_epoch:
                self.heritage_payload['temporal_anchors'].append({
                    'anchor_id': anchor.get('id'),
                    'anchor_type': anchor.get('type'),  # decision_point, paradigm_shift, etc.
                    'temporal_coordinates': anchor.get('coordinates'),
                    'reference_strength': anchor.get('strength', 0.5),
                    'cultural_significance': anchor.get('cultural_significance', 0.5),
                    'continuity_value': anchor.get('continuity_value', 0.5)
                })
    
    def _extract_pigment_legacy(self, context: Dict[str, Any]) -> None:
        """Extract heritage pigment patterns from historical ash"""
        historical_ash = context.get('historical_ash_data', {})
        
        if self.collection_epoch in historical_ash:
            epoch_ash = historical_ash[self.collection_epoch]
            avg_pigment = epoch_ash.get('average_pigment', [0.0, 0.0, 0.0])
            
            # Weight by significance and stability
            epoch_significance = epoch_ash.get('epoch_significance', 0.5)
            stability_factor = epoch_ash.get('stability_factor', 0.5)
            
            heritage_weight = epoch_significance * stability_factor
            
            self.heritage_payload['pigment_legacy'] = [
                avg_pigment[0] * heritage_weight,
                avg_pigment[1] * heritage_weight,
                avg_pigment[2] * heritage_weight
            ]
    
    def _validate_heritage_integrity(self) -> bool:
        """Validate integrity of collected heritage"""
        # Check for minimum viable heritage
        if (len(self.heritage_payload['archival_traces']) == 0 and 
            len(self.heritage_payload['schema_links']) == 0 and
            len(self.heritage_payload['epochal_memories']) == 0):
            return False
        
        # Check for corruption indicators
        for trace in self.heritage_payload['archival_traces']:
            if trace.get('significance', 0) < 0 or trace.get('significance', 0) > 1:
                return False
        
        # Check pigment legacy validity
        pigment_sum = sum(abs(p) for p in self.heritage_payload['pigment_legacy'])
        if pigment_sum > 3.0:  # Unrealistic pigment values
            return False
        
        return True
    
    def _compress_heritage_payload(self) -> None:
        """Compress heritage data for efficient long-term storage"""
        # Remove redundant archival traces
        unique_traces = []
        seen_summaries = set()
        
        for trace in self.heritage_payload['archival_traces']:
            summary_key = str(trace.get('summary', {}))
            if summary_key not in seen_summaries:
                unique_traces.append(trace)
                seen_summaries.add(summary_key)
        
        self.heritage_payload['archival_traces'] = unique_traces
        
        # Consolidate similar schema links
        consolidated_links = []
        link_groups = {}
        
        for link in self.heritage_payload['schema_links']:
            group_key = f"{link.get('source')}-{link.get('target')}"
            if group_key not in link_groups:
                link_groups[group_key] = []
            link_groups[group_key].append(link)
        
        for group in link_groups.values():
            if len(group) == 1:
                consolidated_links.append(group[0])
            else:
                # Merge similar links
                merged_link = group[0].copy()
                merged_link['strength'] = max(link.get('strength', 0) for link in group)
                merged_link['persistence_duration'] = max(link.get('persistence_duration', 0) for link in group)
                consolidated_links.append(merged_link)
        
        self.heritage_payload['schema_links'] = consolidated_links
    
    def _add_preservation_metadata(self, context: Dict[str, Any]) -> None:
        """Add metadata for heritage preservation"""
        preservation_metadata = {
            'preservation_tick': context.get('tick_id', 0),
            'medieval_bee_id': self.tracer_id,
            'collection_epoch': self.collection_epoch,
            'target_epoch': self.target_epoch,
            'heritage_significance': self.heritage_significance,
            'preservation_method': 'medieval_bee_archival',
            'integrity_verified': True,
            'compression_applied': True
        }
        
        self.heritage_payload['preservation_metadata'] = preservation_metadata
    
    def _create_heritage_checksums(self) -> None:
        """Create checksums for heritage integrity verification"""
        # Simple checksum based on content hashing
        checksums = {}
        
        checksums['archival_traces'] = hash(str(self.heritage_payload['archival_traces']))
        checksums['schema_links'] = hash(str(self.heritage_payload['schema_links']))
        checksums['epochal_memories'] = hash(str(self.heritage_payload['epochal_memories']))
        checksums['ancestral_patterns'] = hash(str(self.heritage_payload['ancestral_patterns']))
        checksums['temporal_anchors'] = hash(str(self.heritage_payload['temporal_anchors']))
        
        self.heritage_payload['checksums'] = checksums
    
    def _integrate_archival_traces(self, context: Dict[str, Any]) -> bool:
        """Integrate archival traces into current memory systems"""
        if not self.heritage_payload['archival_traces']:
            return True  # Nothing to integrate
        
        # This would integrate with the actual memory system
        # For now, we simulate successful integration
        integration_success_rate = 0.8  # 80% success rate
        
        successful_integrations = 0
        for trace in self.heritage_payload['archival_traces']:
            # Simulate integration based on significance and priority
            significance = trace.get('significance', 0.5)
            priority = trace.get('preservation_priority', 'medium')
            
            success_probability = significance * (1.2 if priority == 'high' else 1.0 if priority == 'medium' else 0.8)
            
            if success_probability > 0.6:  # Threshold for successful integration
                successful_integrations += 1
        
        success_rate = successful_integrations / len(self.heritage_payload['archival_traces'])
        return success_rate >= 0.5  # At least 50% integration success
    
    def _reinforce_heritage_links(self, context: Dict[str, Any]) -> bool:
        """Reinforce heritage schema links in current schema"""
        if not self.heritage_payload['schema_links']:
            return True
        
        # Simulate reinforcement of heritage links
        reinforced_count = 0
        
        for link in self.heritage_payload['schema_links']:
            heritage_value = link.get('heritage_value', 0.5)
            persistence_duration = link.get('persistence_duration', 0)
            
            # Links with high heritage value and long persistence are easier to reinforce
            reinforcement_strength = heritage_value * min(1.0, persistence_duration / 1000.0)
            
            if reinforcement_strength > 0.4:
                reinforced_count += 1
        
        return reinforced_count >= len(self.heritage_payload['schema_links']) * 0.6
    
    def _deposit_epochal_memories(self, context: Dict[str, Any]) -> bool:
        """Deposit epochal memories into long-term storage"""
        if not self.heritage_payload['epochal_memories']:
            return True
        
        # Simulate memory deposition
        deposited_count = 0
        
        for memory in self.heritage_payload['epochal_memories']:
            significance = memory.get('significance', 0.5)
            emotional_weight = memory.get('emotional_weight', 0.5)
            
            # Memories with high significance and emotional weight are easier to deposit
            deposition_strength = (significance + emotional_weight) / 2
            
            if deposition_strength > 0.5:
                deposited_count += 1
        
        return deposited_count >= len(self.heritage_payload['epochal_memories']) * 0.7
    
    def _weave_ancestral_patterns(self, context: Dict[str, Any]) -> bool:
        """Weave ancestral patterns into current schema"""
        if not self.heritage_payload['ancestral_patterns']:
            return True
        
        # Simulate pattern weaving
        woven_count = 0
        
        for pattern in self.heritage_payload['ancestral_patterns']:
            stability_score = pattern.get('stability_score', 0.5)
            recurrence_count = pattern.get('recurrence_count', 0)
            
            # Stable patterns with high recurrence are easier to weave
            weaving_strength = stability_score * min(1.0, recurrence_count / 20.0)
            
            if weaving_strength > 0.4:
                woven_count += 1
        
        return woven_count >= len(self.heritage_payload['ancestral_patterns']) * 0.6
    
    def _establish_temporal_anchors(self, context: Dict[str, Any]) -> bool:
        """Establish temporal anchors for historical continuity"""
        if not self.heritage_payload['temporal_anchors']:
            return True
        
        # Simulate anchor establishment
        established_count = 0
        
        for anchor in self.heritage_payload['temporal_anchors']:
            reference_strength = anchor.get('reference_strength', 0.5)
            continuity_value = anchor.get('continuity_value', 0.5)
            
            # Strong references with high continuity value are easier to establish
            establishment_strength = (reference_strength + continuity_value) / 2
            
            if establishment_strength > 0.5:
                established_count += 1
        
        return established_count >= len(self.heritage_payload['temporal_anchors']) * 0.7
    
    def _apply_pigment_legacy(self, context: Dict[str, Any]) -> bool:
        """Apply heritage pigment patterns to current ash"""
        pigment_legacy = self.heritage_payload['pigment_legacy']
        
        if sum(abs(p) for p in pigment_legacy) < 0.1:
            return True  # No significant pigment legacy to apply
        
        # This would integrate with the actual ash/pigment system
        # For now, we simulate successful application
        return True
    
    def _create_identification_report(self, context: Dict[str, Any]) -> TracerReport:
        """Create report for heritage identification phase"""
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.INFO,
            report_type="heritage_identification",
            metadata={
                "collection_epoch": self.collection_epoch,
                "target_epoch": self.target_epoch,
                "heritage_significance": self.heritage_significance,
                "phase": "identification_complete"
            }
        )
    
    def _create_collection_report(self, context: Dict[str, Any]) -> TracerReport:
        """Create report for heritage collection phase"""
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.INFO,
            report_type="heritage_collection",
            metadata={
                "collection_epoch": self.collection_epoch,
                "collected_data": {
                    "archival_traces": len(self.heritage_payload['archival_traces']),
                    "schema_links": len(self.heritage_payload['schema_links']),
                    "epochal_memories": len(self.heritage_payload['epochal_memories']),
                    "ancestral_patterns": len(self.heritage_payload['ancestral_patterns']),
                    "temporal_anchors": len(self.heritage_payload['temporal_anchors'])
                },
                "pigment_legacy": self.heritage_payload['pigment_legacy'],
                "phase": "collection_complete"
            }
        )
    
    def _create_preservation_report(self, context: Dict[str, Any]) -> TracerReport:
        """Create report for heritage preservation phase"""
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.INFO,
            report_type="heritage_preservation",
            metadata={
                "preservation_metadata": self.heritage_payload.get('preservation_metadata', {}),
                "integrity_verified": True,
                "compression_applied": True,
                "checksums_created": True,
                "phase": "preservation_complete"
            }
        )
    
    def _create_integration_report(self, context: Dict[str, Any]) -> TracerReport:
        """Create final heritage integration report"""
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.INFO,
            report_type="heritage_pollination",
            metadata={
                "origin_epoch": self.collection_epoch,
                "destination_epoch": self.target_epoch,
                "payload": {
                    "archival_traces": len(self.heritage_payload['archival_traces']),
                    "schema_links": len(self.heritage_payload['schema_links']),
                    "epochal_memories": len(self.heritage_payload['epochal_memories']),
                    "ancestral_patterns": len(self.heritage_payload['ancestral_patterns']),
                    "temporal_anchors": len(self.heritage_payload['temporal_anchors']),
                    "pigment_legacy": self.heritage_payload['pigment_legacy']
                },
                "heritage_significance": self.heritage_significance,
                "integration_success": True,
                "continuity_preserved": True,
                "phase": "integration_complete"
            }
        )
    
    def should_retire(self, context: Dict[str, Any]) -> bool:
        """Retire when heritage integration is complete or lifespan exceeded"""
        current_tick = context.get('tick_id', 0)
        age = self.get_age(current_tick)
        
        # Retire when integration is complete
        if self.integration_complete:
            return True
        
        # Retire after lifespan
        if age >= self.base_lifespan:
            return True
        
        # Emergency retirement if heritage collection fails
        if (age > 15 and self.heritage_phase == "identification" and 
            self.collection_epoch is None):
            logger.warning(f"Medieval Bee {self.tracer_id} emergency retirement: no heritage source found")
            return True
        
        # Emergency retirement if heritage validation fails
        if (age > 20 and self.heritage_phase == "preservation" and 
            not self._validate_heritage_integrity()):
            logger.warning(f"Medieval Bee {self.tracer_id} emergency retirement: heritage integrity failed")
            return True
        
        return False
    
    def transport_pigment_legacy(self, source_pigment: List[float], destination_cluster: str) -> bool:
        """Transport pigment vectors across clusters as heritage continuity (mythic behavior)"""
        if len(self.pigment_transport_capacity) >= 3:  # Limited transport capacity
            logger.debug(f"Medieval Bee {self.tracer_id} pigment transport at capacity")
            return False
        
        # Apply archetypal bias - Medieval Bee enhances green (balance/preservation)
        enhanced_pigment = source_pigment.copy()
        enhanced_pigment[1] += 0.2  # Green channel enhancement for preservation
        
        # Normalize to maintain vector integrity
        import numpy as np
        pigment_magnitude = np.linalg.norm(enhanced_pigment)
        if pigment_magnitude > 0:
            enhanced_pigment = [p / pigment_magnitude for p in enhanced_pigment]
        
        transport_package = {
            'original_pigment': source_pigment,
            'enhanced_pigment': enhanced_pigment,
            'destination': destination_cluster,
            'transport_timestamp': self.last_observation_tick,
            'heritage_significance': self.heritage_significance,
            'medieval_blessing': self.medieval_wisdom * 0.1
        }
        
        self.pigment_transport_capacity.append(transport_package)
        logger.info(f"Medieval Bee {self.tracer_id} transporting pigment legacy to {destination_cluster}")
        return True
    
    def weave_cross_epochal_links(self, epoch_a: int, epoch_b: int, link_strength: float) -> str:
        """Create links between different epochs to prevent amnesia (mythic behavior)"""
        link_id = f"epoch_link_{epoch_a}_{epoch_b}_{self.tracer_id}"
        
        # Medieval Bees create stronger links due to their temporal anchoring role
        enhanced_strength = min(1.0, link_strength + self.order_memory_strength * 0.2)
        
        self.cross_epochal_links[link_id] = {
            'epoch_a': epoch_a,
            'epoch_b': epoch_b,
            'link_strength': enhanced_strength,
            'created_tick': self.last_observation_tick,
            'mythic_archetype': 'medieval_bee_temporal_anchor',
            'order_memory_contribution': self.order_memory_strength
        }
        
        logger.info(f"Medieval Bee {self.tracer_id} wove cross-epochal link: {link_id}")
        return link_id
    
    def get_archetypal_status(self) -> Dict[str, Any]:
        """Return current archetypal status for mythic monitoring"""
        return {
            'mythic_role': self.mythic_role,
            'archetypal_bias': self.archetypal_bias,
            'medieval_wisdom': self.medieval_wisdom,
            'order_memory_strength': self.order_memory_strength,
            'pigment_transport_load': len(self.pigment_transport_capacity),
            'cross_epochal_links': len(self.cross_epochal_links),
            'heritage_phase': self.heritage_phase,
            'temporal_anchoring_active': len(self.cross_epochal_links) > 0,
            'heritage_significance': self.heritage_significance,
            'amnesia_prevention_score': self.medieval_wisdom * self.order_memory_strength
        }

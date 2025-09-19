"""
Beetle Tracer - Decay & Residue Recycler

Crawls through cognitive residue - forgotten fragments, soot buildup, decaying
schema edges - ensuring nothing toxic lingers too long. Where Ant builds,
Spider senses, and Crow scouts, Beetle cleans and recycles.
"""

from typing import Dict, Any, List, Tuple
from .base_tracer import BaseTracer, TracerType, TracerReport, AlertSeverity, TracerSpawnConditions
import logging

logger = logging.getLogger(__name__)


class BeetleTracer(BaseTracer):
    """Decay processing and nutrient recycling tracer - the recycler of DAWN"""
    
    def __init__(self, tracer_id: str = None):
        super().__init__(tracer_id)
        self.target_residues = []
        self.nutrients_recovered = 0.0
        self.processing_queue = []
        self.processing_complete = False
        self.toxicity_detected = False
        self.recycling_efficiency = 0.0
        
    @property
    def tracer_type(self) -> TracerType:
        return TracerType.BEETLE
    
    @property
    def base_lifespan(self) -> int:
        return 999  # Until task completion
    
    @property
    def base_nutrient_cost(self) -> float:
        return 0.4  # Medium cost, offset by recovery
    
    @property
    def archetype_description(self) -> str:
        return "Decay recycler - breaks down residues, recycles nutrients"
    
    def spawn_conditions_met(self, context: Dict[str, Any]) -> bool:
        """Spawn when soot accumulation or decay targets detected"""
        # Adjusted for DAWN's ranges - soot accumulation
        if TracerSpawnConditions.soot_accumulation(context, threshold=0.2):  # Was 0.4
            return True
            
        # Schema edges flagged for pruning
        schema_edges = context.get('schema_edges', [])
        prunable_edges = [e for e in schema_edges if e.get('marked_for_pruning', False)]
        if len(prunable_edges) > 2:
            return True
            
        # Residual fragments detected
        residual_fragments = context.get('residual_fragments', [])
        toxic_residues = [r for r in residual_fragments if r.get('toxicity', 0) > 0.6]
        if len(toxic_residues) > 1:
            return True
            
        # Ash fragments that have stagnated
        ash_fragments = context.get('ash_fragments', [])
        stagnant_ash = [a for a in ash_fragments if a.get('age', 0) > 20 and a.get('crystallization', 0) < 0.2]
        if len(stagnant_ash) > 5:
            return True
            
        # Memory pressure indicators
        memory_pressure = context.get('memory_pressure', 0.0)
        if memory_pressure > 0.7:
            return True
            
        # Bloom debris accumulation
        bloom_debris = context.get('bloom_debris', [])
        if len(bloom_debris) > 10:
            return True
            
        return False
    
    def observe(self, context: Dict[str, Any]) -> List[TracerReport]:
        """Process decay and recycle nutrients"""
        reports = []
        
        # Phase 1: Identify and categorize targets
        if not self.target_residues and not self.processing_queue:
            self._identify_decay_targets(context)
            if self.target_residues:
                reports.append(self._create_identification_report(context))
                
        # Phase 2: Process residues
        elif self.target_residues or self.processing_queue:
            processed = self._process_residues(context)
            
            for residue_id, action, recovered, risk_level in processed:
                reports.append(TracerReport(
                    tracer_id=self.tracer_id,
                    tracer_type=self.tracer_type,
                    tick_id=context.get('tick_id', 0),
                    timestamp=context.get('timestamp', 0),
                    severity=AlertSeverity.WARN if risk_level == "high" else AlertSeverity.INFO,
                    report_type="decay_event",
                    metadata={
                        "residue_id": residue_id,
                        "action": action,
                        "nutrient_recovered": recovered,
                        "risk_level": risk_level,
                        "processing_efficiency": self.recycling_efficiency
                    }
                ))
                
                self.nutrients_recovered += recovered
            
            # Check if processing is complete
            if not self.target_residues and not self.processing_queue:
                self.processing_complete = True
                
                # Final summary report
                reports.append(self._create_completion_report(context))
        
        return reports
    
    def _identify_decay_targets(self, context: Dict[str, Any]) -> None:
        """Identify and prioritize residues for processing"""
        # High-entropy soot fragments (priority: high)
        soot_fragments = context.get('soot_fragments', [])
        for fragment in soot_fragments:
            entropy = fragment.get('entropy', 0)
            age = fragment.get('age', 0)
            volatility = fragment.get('volatility', 0)
            
            if entropy > 0.7 and age > 5:
                priority = "high" if volatility > 0.6 else "medium"
                self.target_residues.append({
                    'id': fragment.get('id'),
                    'type': 'soot',
                    'entropy': entropy,
                    'age': age,
                    'volatility': volatility,
                    'priority': priority,
                    'toxicity': min(1.0, entropy * volatility)
                })
        
        # Abandoned bloom traces (priority: medium)
        bloom_traces = context.get('bloom_traces', [])
        for trace in bloom_traces:
            if trace.get('abandoned', False) and trace.get('decay_rate', 0) < 0.1:
                self.target_residues.append({
                    'id': trace.get('id'),
                    'type': 'bloom_trace',
                    'decay_rate': trace.get('decay_rate'),
                    'age': trace.get('age', 0),
                    'priority': "medium",
                    'toxicity': 0.3  # Generally low toxicity
                })
        
        # Schema debris (priority: low to medium)
        schema_debris = context.get('schema_debris', [])
        for debris in schema_debris:
            redundancy = debris.get('redundancy_score', 0)
            if debris.get('redundant', False) or redundancy > 0.7:
                priority = "high" if redundancy > 0.9 else "medium"
                self.target_residues.append({
                    'id': debris.get('id'),
                    'type': 'schema_debris',
                    'redundancy_score': redundancy,
                    'priority': priority,
                    'toxicity': redundancy * 0.5
                })
        
        # Stagnant ash fragments (priority: low)
        ash_fragments = context.get('ash_fragments', [])
        for ash in ash_fragments:
            age = ash.get('age', 0)
            crystallization = ash.get('crystallization', 0)
            
            if age > 20 and crystallization < 0.2:  # Old but not crystallizing
                self.target_residues.append({
                    'id': ash.get('id'),
                    'type': 'stagnant_ash',
                    'age': age,
                    'crystallization': crystallization,
                    'priority': "low",
                    'toxicity': 0.2  # Generally safe
                })
        
        # Toxic residual fragments (priority: critical)
        residual_fragments = context.get('residual_fragments', [])
        for residue in residual_fragments:
            toxicity = residue.get('toxicity', 0)
            if toxicity > 0.6:
                self.target_residues.append({
                    'id': residue.get('id'),
                    'type': 'toxic_residue',
                    'toxicity': toxicity,
                    'priority': "critical",
                    'source': residue.get('source', 'unknown')
                })
                self.toxicity_detected = True
        
        # Sort by priority: critical > high > medium > low
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        self.target_residues.sort(key=lambda x: (priority_order.get(x['priority'], 4), -x.get('toxicity', 0)))
    
    def _process_residues(self, context: Dict[str, Any]) -> List[Tuple[str, str, float, str]]:
        """Process and recycle residues"""
        processed = []
        processing_capacity = 3  # Process up to 3 residues per tick
        
        # Move high-priority items to processing queue
        while (len(self.processing_queue) < processing_capacity and 
               self.target_residues and 
               self.target_residues[0]['priority'] in ["critical", "high"]):
            self.processing_queue.append(self.target_residues.pop(0))
        
        # Fill remaining capacity with other items
        while (len(self.processing_queue) < processing_capacity and self.target_residues):
            self.processing_queue.append(self.target_residues.pop(0))
        
        # Process items in queue
        remaining_queue = []
        for target in self.processing_queue:
            if len(processed) >= processing_capacity:
                remaining_queue.append(target)
                continue
                
            residue_id = target['id']
            residue_type = target['type']
            priority = target['priority']
            
            # Determine action and efficiency based on residue properties
            action, recovered, risk_level = self._determine_processing_action(target)
            
            processed.append((residue_id, action, recovered, risk_level))
            
            # Update recycling efficiency
            self._update_recycling_efficiency(target, recovered)
        
        self.processing_queue = remaining_queue
        return processed
    
    def _determine_processing_action(self, target: Dict[str, Any]) -> Tuple[str, float, str]:
        """Determine the best processing action for a residue"""
        residue_type = target['type']
        priority = target['priority']
        toxicity = target.get('toxicity', 0)
        
        if residue_type == 'toxic_residue':
            if toxicity > 0.8:
                return "neutralize", 0.1, "high"  # High toxicity, low recovery
            else:
                return "breakdown", 0.2, "medium"
                
        elif residue_type == 'soot':
            entropy = target.get('entropy', 0)
            volatility = target.get('volatility', 0)
            
            if entropy > 0.8 and volatility > 0.7:
                return "breakdown", 0.3, "high"  # Volatile soot, moderate recovery
            elif entropy > 0.6:
                return "recycle", 0.4, "medium"  # Stable soot, good recovery
            else:
                return "compost", 0.5, "low"  # Low entropy, excellent recovery
                
        elif residue_type == 'bloom_trace':
            decay_rate = target.get('decay_rate', 0)
            if decay_rate < 0.05:
                return "recycle", 0.5, "low"  # Stable traces, good nutrients
            else:
                return "breakdown", 0.3, "medium"
                
        elif residue_type == 'schema_debris':
            redundancy = target.get('redundancy_score', 0)
            if redundancy > 0.8:
                return "discard", 0.1, "low"  # Highly redundant, little value
            else:
                return "recycle", 0.4, "medium"  # Some structural value
                
        elif residue_type == 'stagnant_ash':
            crystallization = target.get('crystallization', 0)
            if crystallization < 0.1:
                return "breakdown", 0.2, "low"  # Poor crystallization
            else:
                return "recycle", 0.3, "low"  # Some crystallization value
                
        else:
            return "breakdown", 0.2, "medium"  # Default action
    
    def _update_recycling_efficiency(self, target: Dict[str, Any], recovered: float) -> None:
        """Update overall recycling efficiency metrics"""
        # Calculate efficiency based on recovered nutrients vs processing cost
        processing_cost = 0.1  # Base processing cost per item
        
        if target['priority'] == "critical":
            processing_cost *= 2.0  # Higher cost for toxic materials
        elif target['priority'] == "high":
            processing_cost *= 1.5
            
        efficiency = recovered / processing_cost if processing_cost > 0 else 0
        
        # Update running average
        if self.recycling_efficiency == 0:
            self.recycling_efficiency = efficiency
        else:
            self.recycling_efficiency = (self.recycling_efficiency * 0.8 + efficiency * 0.2)
    
    def _create_identification_report(self, context: Dict[str, Any]) -> TracerReport:
        """Create report for target identification phase"""
        priority_counts = {}
        toxicity_levels = []
        
        for target in self.target_residues:
            priority = target['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            toxicity_levels.append(target.get('toxicity', 0))
        
        avg_toxicity = sum(toxicity_levels) / len(toxicity_levels) if toxicity_levels else 0
        
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.WARN if self.toxicity_detected else AlertSeverity.INFO,
            report_type="decay_identification",
            metadata={
                "total_targets": len(self.target_residues),
                "priority_breakdown": priority_counts,
                "avg_toxicity": avg_toxicity,
                "toxicity_detected": self.toxicity_detected,
                "estimated_recovery": self._estimate_total_recovery()
            }
        )
    
    def _create_completion_report(self, context: Dict[str, Any]) -> TracerReport:
        """Create final processing completion report"""
        processing_duration = context.get('tick_id', 0) - self.spawn_tick
        
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.INFO,
            report_type="recycling_complete",
            metadata={
                "total_nutrients_recovered": self.nutrients_recovered,
                "processing_duration": processing_duration,
                "recycling_efficiency": self.recycling_efficiency,
                "toxicity_handled": self.toxicity_detected,
                "performance_rating": self._get_performance_rating()
            }
        )
    
    def _estimate_total_recovery(self) -> float:
        """Estimate total nutrient recovery from all targets"""
        total_estimate = 0.0
        
        for target in self.target_residues:
            _, estimated_recovery, _ = self._determine_processing_action(target)
            total_estimate += estimated_recovery
            
        return total_estimate
    
    def _get_performance_rating(self) -> str:
        """Get performance rating based on efficiency and speed"""
        if self.recycling_efficiency > 2.0:
            return "excellent"
        elif self.recycling_efficiency > 1.5:
            return "good"
        elif self.recycling_efficiency > 1.0:
            return "adequate"
        elif self.recycling_efficiency > 0.5:
            return "poor"
        else:
            return "inefficient"
    
    def should_retire(self, context: Dict[str, Any]) -> bool:
        """Retire when processing is complete or no targets found"""
        current_tick = context.get('tick_id', 0)
        age = self.get_age(current_tick)
        
        # Retire when processing complete
        if self.processing_complete:
            return True
            
        # Early retirement if no targets found after reasonable search
        if age > 3 and not self.target_residues and not self.processing_queue:
            return True
            
        # Starvation protection - retire if running too long without progress
        if age > 25 and self.nutrients_recovered == 0:
            return True
            
        # Emergency retirement if toxicity levels are too high to handle safely
        if (self.toxicity_detected and age > 15 and 
            any(t.get('toxicity', 0) > 0.9 for t in self.target_residues)):
            logger.warning(f"Beetle {self.tracer_id} emergency retirement due to extreme toxicity")
            return True
            
        return False

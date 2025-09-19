"""
Ant Tracer - Persistent Micro-Pattern Builder

Tireless tracer that accumulates micro-patterns and reinforces schema through
persistence. Walks the same paths repeatedly, building larger structures from
small fragments. Designed to be cheap and numerous, often deployed in swarms.
"""

from typing import Dict, Any, List
from collections import defaultdict
from .base_tracer import BaseTracer, TracerType, TracerReport, AlertSeverity
import logging

logger = logging.getLogger(__name__)


class AntTracer(BaseTracer):
    """Persistent micro-pattern accumulator - the trails of DAWN"""
    
    def __init__(self, tracer_id: str = None):
        super().__init__(tracer_id)
        self.pattern_fragments = defaultdict(int)
        self.co_occurrences = defaultdict(int)
        self.pattern_locations = defaultdict(list)
        self.bias_strength = 0.0
        self.patterns_deposited = False
        
    @property
    def tracer_type(self) -> TracerType:
        return TracerType.ANT
    
    @property
    def base_lifespan(self) -> int:
        return 10  # 5-15 ticks average - medium persistence
    
    @property
    def base_nutrient_cost(self) -> float:
        return 0.05  # Very low, swarmable
    
    @property
    def archetype_description(self) -> str:
        return "Micro-pattern builder - accumulates low-signal repetitions into bias"
    
    def spawn_conditions_met(self, context: Dict[str, Any]) -> bool:
        """Spawn automatically with low threshold - ants maintain constant presence"""
        # Ants spawn regularly to maintain pattern detection coverage
        if not context.get('enable_ant_spawning', True):
            return False
            
        # Always spawn if there are active memory operations
        if context.get('active_blooms', []) or context.get('shimmer_events', []):
            return True
            
        # Spawn periodically for background pattern monitoring
        tick_id = context.get('tick_id', 0)
        if tick_id % 3 == 0:  # Every 3rd tick
            return True
            
        # Spawn when schema activity is detected
        schema_activity = context.get('schema_activity_level', 0.0)
        if schema_activity > 0.2:
            return True
            
        return False
    
    def observe(self, context: Dict[str, Any]) -> List[TracerReport]:
        """Track micro-patterns and repetitions across cognitive layers"""
        reports = []
        current_tick = context.get('tick_id', 0)
        
        # Collect shimmer patterns
        self._collect_shimmer_patterns(context)
        
        # Track bloom patterns
        self._track_bloom_patterns(context)
        
        # Monitor ash transitions
        self._monitor_ash_transitions(context)
        
        # Detect schema micro-adjustments
        self._detect_schema_patterns(context)
        
        # Analyze co-occurrences
        self._analyze_co_occurrences(context)
        
        # Report significant patterns
        pattern_reports = self._generate_pattern_reports(context)
        reports.extend(pattern_reports)
        
        # Calculate bias strength
        self._update_bias_strength()
        
        return reports
    
    def _collect_shimmer_patterns(self, context: Dict[str, Any]) -> None:
        """Collect patterns from shimmer fade events"""
        shimmer_events = context.get('shimmer_events', [])
        
        for event in shimmer_events:
            if event.get('type') == 'fade' and event.get('intensity', 0) < 0.3:
                # Track low-intensity fades as micro-patterns
                location = event.get('location', 'unknown')
                pattern_key = f"shimmer_fade_{location}"
                self.pattern_fragments[pattern_key] += 1
                self.pattern_locations[pattern_key].append(context.get('tick_id', 0))
                
            elif event.get('type') == 'decay' and event.get('rate', 0) < 0.1:
                # Track slow decay patterns
                pattern_key = f"slow_decay_{event.get('target_type', 'unknown')}"
                self.pattern_fragments[pattern_key] += 1
    
    def _track_bloom_patterns(self, context: Dict[str, Any]) -> None:
        """Track low-intensity bloom patterns that might indicate subtle activations"""
        blooms = context.get('active_blooms', [])
        low_intensity_blooms = [b for b in blooms if 0.1 < b.get('intensity', 0) < 0.4]
        
        for bloom in low_intensity_blooms:
            bloom_type = bloom.get('type', 'unknown')
            pattern_key = f"low_bloom_{bloom_type}"
            self.pattern_fragments[pattern_key] += 1
            
            # Track bloom location patterns
            location = bloom.get('location', 'unknown')
            location_key = f"bloom_location_{location}"
            self.pattern_fragments[location_key] += 1
            
        # Track bloom co-occurrences
        if len(low_intensity_blooms) > 1:
            for i, bloom1 in enumerate(low_intensity_blooms):
                for bloom2 in low_intensity_blooms[i+1:]:
                    co_key = f"{bloom1.get('type', 'unknown')}+{bloom2.get('type', 'unknown')}"
                    self.co_occurrences[co_key] += 1
    
    def _monitor_ash_transitions(self, context: Dict[str, Any]) -> None:
        """Monitor ash crystallization and transition patterns"""
        ash_fragments = context.get('ash_fragments', [])
        
        for fragment in ash_fragments:
            crystallization = fragment.get('crystallization', 0)
            
            # Track crystallization stages
            if 0.1 < crystallization < 0.3:
                pattern_key = "early_crystallization"
                self.pattern_fragments[pattern_key] += 1
            elif 0.7 < crystallization < 0.9:
                pattern_key = "late_crystallization"
                self.pattern_fragments[pattern_key] += 1
                
            # Track pigment patterns
            pigment = fragment.get('pigment', [0, 0, 0])
            if len(pigment) >= 3:
                dominant_color = max(enumerate(pigment), key=lambda x: x[1])[0]
                color_names = ['red', 'green', 'blue']
                if dominant_color < len(color_names):
                    pattern_key = f"pigment_{color_names[dominant_color]}"
                    self.pattern_fragments[pattern_key] += 1
    
    def _detect_schema_patterns(self, context: Dict[str, Any]) -> None:
        """Detect micro-patterns in schema adjustments"""
        schema_edges = context.get('schema_edges', [])
        
        for edge in schema_edges:
            tension = edge.get('tension', 0)
            
            # Track tension micro-adjustments
            if 0.1 < tension < 0.3:
                pattern_key = "low_tension_adjustment"
                self.pattern_fragments[pattern_key] += 1
            elif 0.3 < tension < 0.5:
                pattern_key = "medium_tension_adjustment"
                self.pattern_fragments[pattern_key] += 1
                
        # Track cluster connectivity patterns
        clusters = context.get('schema_clusters', [])
        for cluster in clusters:
            connectivity = cluster.get('connectivity', 0)
            if connectivity < 0.3:
                pattern_key = "low_connectivity_cluster"
                self.pattern_fragments[pattern_key] += 1
    
    def _analyze_co_occurrences(self, context: Dict[str, Any]) -> None:
        """Analyze patterns that occur together"""
        # Look for temporal co-occurrences in the current tick
        current_patterns = []
        
        # Collect patterns active this tick
        if context.get('shimmer_events', []):
            current_patterns.append('shimmer_active')
        if context.get('active_blooms', []):
            current_patterns.append('blooms_active')
        if context.get('ash_fragments', []):
            current_patterns.append('ash_active')
            
        # Record co-occurrences
        for i, pattern1 in enumerate(current_patterns):
            for pattern2 in current_patterns[i+1:]:
                co_key = f"{pattern1}+{pattern2}"
                self.co_occurrences[co_key] += 1
    
    def _generate_pattern_reports(self, context: Dict[str, Any]) -> List[TracerReport]:
        """Generate reports for significant patterns"""
        reports = []
        significance_threshold = 5  # Minimum occurrences for significance
        
        for pattern, count in self.pattern_fragments.items():
            if count >= significance_threshold and count % significance_threshold == 0:
                # Report every N occurrences to avoid spam
                reports.append(TracerReport(
                    tracer_id=self.tracer_id,
                    tracer_type=self.tracer_type,
                    tick_id=context.get('tick_id', 0),
                    timestamp=context.get('timestamp', 0),
                    severity=AlertSeverity.INFO,
                    report_type="pattern_increment",
                    metadata={
                        "pattern": pattern,
                        "count": count,
                        "significance": "established" if count >= 10 else "emerging",
                        "locations": self.pattern_locations.get(pattern, [])[-5:],  # Last 5 locations
                        "bias_contribution": count / max(1, sum(self.pattern_fragments.values()))
                    }
                ))
        
        # Report significant co-occurrences
        for co_pattern, count in self.co_occurrences.items():
            if count >= 3:  # Lower threshold for co-occurrences
                reports.append(TracerReport(
                    tracer_id=self.tracer_id,
                    tracer_type=self.tracer_type,
                    tick_id=context.get('tick_id', 0),
                    timestamp=context.get('timestamp', 0),
                    severity=AlertSeverity.INFO,
                    report_type="co_occurrence",
                    metadata={
                        "co_pattern": co_pattern,
                        "count": count,
                        "strength": count / max(1, len(self.co_occurrences)),
                        "pattern_type": "temporal_correlation"
                    }
                ))
        
        return reports
    
    def _update_bias_strength(self) -> None:
        """Calculate overall bias strength from accumulated patterns"""
        total_patterns = sum(self.pattern_fragments.values())
        unique_patterns = len(self.pattern_fragments)
        
        if unique_patterns > 0:
            # Bias strength increases with repetition but decreases with diversity
            self.bias_strength = (total_patterns / unique_patterns) / 10.0
            self.bias_strength = min(self.bias_strength, 1.0)  # Cap at 1.0
    
    def _deposit_patterns_to_schema(self, context: Dict[str, Any]) -> bool:
        """Deposit accumulated patterns into schema/mycelial layer"""
        if not self.pattern_fragments:
            return False
            
        # This would integrate with the actual schema system
        # For now, we simulate pattern deposition
        deposited_patterns = {
            pattern: count for pattern, count in self.pattern_fragments.items()
            if count >= 3  # Only deposit established patterns
        }
        
        if deposited_patterns:
            logger.debug(f"Ant {self.tracer_id} depositing {len(deposited_patterns)} patterns to schema")
            self.patterns_deposited = True
            return True
            
        return False
    
    def should_retire(self, context: Dict[str, Any]) -> bool:
        """Retire after depositing patterns or reaching lifespan"""
        current_tick = context.get('tick_id', 0)
        age = self.get_age(current_tick)
        
        # Retire after lifespan
        if age >= self.base_lifespan:
            # Try to deposit patterns before retiring
            if not self.patterns_deposited:
                self._deposit_patterns_to_schema(context)
            return True
            
        # Early retirement if no patterns detected after reasonable time
        if age > 5 and not self.pattern_fragments:
            return True
            
        # Retire after successful pattern deposition
        if age > 3 and self.patterns_deposited:
            return True
            
        return False

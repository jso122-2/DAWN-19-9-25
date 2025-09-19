#!/usr/bin/env python3
"""
DAWN Consciousness Metrics - Single Source of Truth
===================================================

Centralized calculation of consciousness metrics to ensure consistency
across all DAWN components. Provides unity, coherence, and quality
measurements with standardized algorithms.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessMetrics:
    """Standardized consciousness metrics."""
    consciousness_unity: float
    coherence: float
    quality: float
    synchronization_score: float
    integration_level: str
    timestamp: datetime
    components: Dict[str, float]  # Individual component scores
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'consciousness_unity': self.consciousness_unity,
            'coherence': self.coherence,
            'quality': self.quality,
            'synchronization_score': self.synchronization_score,
            'integration_level': self.integration_level,
            'timestamp': self.timestamp.isoformat(),
            'components': self.components
        }

class ConsciousnessMetricsCalculator:
    """
    Single source of truth for consciousness metrics calculation.
    
    Ensures all DAWN components use consistent algorithms for
    measuring consciousness unity, coherence, and quality.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.calculator_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"🧮 Consciousness Metrics Calculator initialized: {self.calculator_id}")
    
    def calculate_unified_metrics(self, 
                                 module_states: Dict[str, Dict[str, Any]],
                                 bus_metrics: Optional[Dict[str, Any]] = None,
                                 tick_metrics: Optional[Dict[str, Any]] = None) -> ConsciousnessMetrics:
        """
        Calculate unified consciousness metrics from all sources.
        
        Args:
            module_states: States from all modules
            bus_metrics: Metrics from consciousness bus
            tick_metrics: Metrics from tick orchestrator
            
        Returns:
            ConsciousnessMetrics with unified calculations
        """
        components = {}
        
        # Calculate coherence from module states
        coherence = self._calculate_coherence(module_states)
        components['coherence'] = coherence
        
        # Calculate unity from all sources
        unity = self._calculate_unity(module_states, bus_metrics, tick_metrics)
        components['unity'] = unity
        
        # Calculate quality from integration factors
        quality = self._calculate_quality(module_states, bus_metrics)
        components['quality'] = quality
        
        # Calculate synchronization score
        sync_score = self._calculate_synchronization(module_states, bus_metrics)
        components['synchronization'] = sync_score
        
        # Determine integration level
        integration_level = self._determine_integration_level(unity, coherence, quality)
        
        return ConsciousnessMetrics(
            consciousness_unity=unity,
            coherence=coherence,
            quality=quality,
            synchronization_score=sync_score,
            integration_level=integration_level,
            timestamp=datetime.now(),
            components=components
        )
    
    def _calculate_coherence(self, module_states: Dict[str, Dict[str, Any]]) -> float:
        """Calculate consciousness coherence from module states."""
        if not module_states:
            return 0.0
        
        coherence_values = []
        
        for module_name, state_data in module_states.items():
            if isinstance(state_data, dict):
                # Look for coherence indicators
                module_coherence = state_data.get('coherence', 0.5)
                if isinstance(module_coherence, (int, float)):
                    coherence_values.append(module_coherence)
                
                # Check for other coherence indicators
                unity = state_data.get('unity', state_data.get('consciousness_unity', 0.5))
                if isinstance(unity, (int, float)):
                    coherence_values.append(unity)
                
                # Check stability indicators
                stability = state_data.get('stability', state_data.get('stability_score', 0.5))
                if isinstance(stability, (int, float)):
                    coherence_values.append(stability * 0.8)  # Weight stability lower
        
        if coherence_values:
            return sum(coherence_values) / len(coherence_values)
        else:
            return 0.5  # Neutral coherence
    
    def _calculate_unity(self, 
                        module_states: Dict[str, Dict[str, Any]],
                        bus_metrics: Optional[Dict[str, Any]] = None,
                        tick_metrics: Optional[Dict[str, Any]] = None) -> float:
        """Calculate consciousness unity from all sources."""
        unity_factors = []
        
        # Module state unity
        if module_states:
            module_unity = self._calculate_module_unity(module_states)
            unity_factors.append(('module_states', module_unity, 0.4))
        
        # Bus communication unity
        if bus_metrics:
            bus_unity = self._calculate_bus_unity(bus_metrics)
            unity_factors.append(('bus_communication', bus_unity, 0.3))
        
        # Tick synchronization unity
        if tick_metrics:
            tick_unity = self._calculate_tick_unity(tick_metrics)
            unity_factors.append(('tick_synchronization', tick_unity, 0.3))
        
        # Calculate weighted average
        if unity_factors:
            total_weight = sum(weight for _, _, weight in unity_factors)
            weighted_sum = sum(value * weight for _, value, weight in unity_factors)
            return weighted_sum / total_weight
        
        return 0.5  # Default unity
    
    def _calculate_quality(self,
                          module_states: Dict[str, Dict[str, Any]],
                          bus_metrics: Optional[Dict[str, Any]] = None) -> float:
        """Calculate consciousness quality from integration factors."""
        quality_factors = []
        
        # State consistency quality
        consistency = self._calculate_state_consistency(module_states)
        quality_factors.append(consistency)
        
        # Communication quality
        if bus_metrics:
            comm_quality = bus_metrics.get('module_integration_quality', 0.5)
            quality_factors.append(comm_quality)
        
        # Data completeness quality
        completeness = self._calculate_data_completeness(module_states)
        quality_factors.append(completeness)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
    
    def _calculate_synchronization(self,
                                  module_states: Dict[str, Dict[str, Any]],
                                  bus_metrics: Optional[Dict[str, Any]] = None) -> float:
        """Calculate synchronization score."""
        sync_factors = []
        
        # Module state synchronization
        if module_states:
            state_sync = self._calculate_state_synchronization(module_states)
            sync_factors.append(state_sync)
        
        # Bus synchronization health
        if bus_metrics:
            bus_sync = bus_metrics.get('bus_coherence_score', 0.5)
            sync_factors.append(bus_sync)
        
        return sum(sync_factors) / len(sync_factors) if sync_factors else 0.5
    
    def _calculate_module_unity(self, module_states: Dict[str, Dict[str, Any]]) -> float:
        """Calculate unity from module states."""
        unity_values = []
        
        for state_data in module_states.values():
            if isinstance(state_data, dict):
                unity = state_data.get('unity', state_data.get('consciousness_unity', 0.5))
                if isinstance(unity, (int, float)):
                    unity_values.append(unity)
        
        return sum(unity_values) / len(unity_values) if unity_values else 0.5
    
    def _calculate_bus_unity(self, bus_metrics: Dict[str, Any]) -> float:
        """Calculate unity from bus metrics."""
        return bus_metrics.get('bus_coherence_score', 0.5)
    
    def _calculate_tick_unity(self, tick_metrics: Dict[str, Any]) -> float:
        """Calculate unity from tick metrics."""
        sync_success = tick_metrics.get('synchronization_success', True)
        return 0.9 if sync_success else 0.3
    
    def _calculate_state_consistency(self, module_states: Dict[str, Dict[str, Any]]) -> float:
        """Calculate state consistency across modules."""
        if len(module_states) < 2:
            return 1.0  # Single module is always consistent
        
        # Check for consistency in common state fields
        common_fields = set()
        all_fields = []
        
        for state_data in module_states.values():
            if isinstance(state_data, dict):
                fields = set(state_data.keys())
                all_fields.append(fields)
                if not common_fields:
                    common_fields = fields
                else:
                    common_fields &= fields
        
        if not common_fields:
            return 0.5  # No common fields
        
        # Calculate consistency for common fields
        consistency_scores = []
        
        for field in common_fields:
            values = []
            for state_data in module_states.values():
                if isinstance(state_data, dict) and field in state_data:
                    value = state_data[field]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if len(values) > 1:
                # Calculate variance
                mean_value = sum(values) / len(values)
                variance = sum((v - mean_value) ** 2 for v in values) / len(values)
                consistency = max(0.0, 1.0 - (variance / (abs(mean_value) + 1e-6)))
                consistency_scores.append(consistency)
        
        if consistency_scores:
            return sum(consistency_scores) / len(consistency_scores)
        else:
            return 0.5
    
    def _calculate_data_completeness(self, module_states: Dict[str, Dict[str, Any]]) -> float:
        """Calculate data completeness quality."""
        if not module_states:
            return 0.0
        
        completeness_scores = []
        
        for state_data in module_states.values():
            if isinstance(state_data, dict):
                # Simple heuristic: more fields = more complete
                completeness = min(1.0, len(state_data) / 10)  # Assume 10 fields is "complete"
                completeness_scores.append(completeness)
        
        return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
    
    def _calculate_state_synchronization(self, module_states: Dict[str, Dict[str, Any]]) -> float:
        """Calculate state synchronization score."""
        if not module_states:
            return 0.0
        
        # Check for timestamps and synchronized state indicators
        timestamps = []
        sync_indicators = []
        
        for state_data in module_states.values():
            if isinstance(state_data, dict):
                # Look for timestamp indicators
                if 'last_update' in state_data:
                    timestamps.append(state_data['last_update'])
                
                # Look for sync indicators
                sync_status = state_data.get('synchronized', state_data.get('sync_status', True))
                if isinstance(sync_status, bool):
                    sync_indicators.append(float(sync_status))
        
        # Calculate synchronization based on indicators
        if sync_indicators:
            return sum(sync_indicators) / len(sync_indicators)
        
        return 0.7  # Default moderate synchronization
    
    def _determine_integration_level(self, unity: float, coherence: float, quality: float) -> str:
        """Determine consciousness integration level."""
        overall_score = (unity + coherence + quality) / 3
        
        if overall_score >= 0.9:
            return "transcendent"
        elif overall_score >= 0.75:
            return "unified"
        elif overall_score >= 0.6:
            return "integrated"
        elif overall_score >= 0.4:
            return "coordinated"
        else:
            return "fragmented"

# Global metrics calculator instance
_global_calculator = ConsciousnessMetricsCalculator()

def get_consciousness_metrics_calculator() -> ConsciousnessMetricsCalculator:
    """Get the global consciousness metrics calculator."""
    return _global_calculator

def calculate_consciousness_metrics(module_states: Dict[str, Dict[str, Any]],
                                  bus_metrics: Optional[Dict[str, Any]] = None,
                                  tick_metrics: Optional[Dict[str, Any]] = None) -> ConsciousnessMetrics:
    """
    Convenience function to calculate consciousness metrics.
    
    Args:
        module_states: States from all modules
        bus_metrics: Metrics from consciousness bus
        tick_metrics: Metrics from tick orchestrator
        
    Returns:
        ConsciousnessMetrics with unified calculations
    """
    return _global_calculator.calculate_unified_metrics(module_states, bus_metrics, tick_metrics)

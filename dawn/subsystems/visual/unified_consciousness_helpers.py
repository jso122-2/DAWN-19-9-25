#!/usr/bin/env python3
"""
DAWN Unified Consciousness Engine Helper Methods
===============================================

Helper methods for the unified consciousness engine to complete
the implementation of consciousness integration.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque

def calculate_self_awareness_depth(system_data: Dict[str, Dict[str, Any]], 
                                 correlations: Dict[str, Any]) -> float:
    """Calculate the depth of self-awareness across all systems"""
    
    awareness_indicators = []
    
    # Self-monitoring indicators
    for system_name, data in system_data.items():
        self_monitoring = data.get('monitors_self', False)
        if self_monitoring:
            awareness_indicators.append(0.3)
        
        # Meta-awareness indicators
        meta_level = data.get('meta_awareness_level', 0.0)
        awareness_indicators.append(meta_level * 0.2)
    
    # Cross-system awareness
    cross_awareness = 0.0
    for correlation_type, correlation_data in correlations.items():
        if isinstance(correlation_data, dict):
            strength = correlation_data.get('strength', 0.0)
            cross_awareness += strength * 0.1
    
    awareness_indicators.append(min(0.5, cross_awareness))
    
    return np.mean(awareness_indicators) if awareness_indicators else 0.0

def calculate_integration_quality(correlations: Dict[str, Any]) -> float:
    """Calculate the quality of consciousness integration"""
    
    integration_factors = []
    
    # Cross-dimensional integration
    cross_dim = correlations.get('cross_dimensional', {})
    if cross_dim:
        coherence_emergence = cross_dim.get('coherence_emergence', {})
        overall_coherence = coherence_emergence.get('overall_coherence', 0.0)
        integration_factors.append(overall_coherence)
        
        synchrony = cross_dim.get('consciousness_synchrony', {})
        overall_synchrony = synchrony.get('overall_synchrony', 0.0)
        integration_factors.append(overall_synchrony)
    
    # Pairwise integration quality
    pairwise_correlations = [
        'stability_performance', 'visual_artistic', 
        'performance_visual', 'stability_artistic'
    ]
    
    for correlation_type in pairwise_correlations:
        correlation_data = correlations.get(correlation_type, {})
        if correlation_data:
            strength = correlation_data.get('strength', 0.0)
            confidence = correlation_data.get('confidence', 0.0)
            integration_factors.append((strength + confidence) / 2.0)
    
    return np.mean(integration_factors) if integration_factors else 0.0

def calculate_meta_cognitive_activity(system_data: Dict[str, Dict[str, Any]]) -> float:
    """Calculate the level of meta-cognitive activity"""
    
    meta_indicators = []
    
    # Check each system for meta-cognitive capabilities
    for system_name, data in system_data.items():
        # Self-analysis capabilities
        self_analysis = data.get('analyzes_own_performance', False)
        if self_analysis:
            meta_indicators.append(0.25)
        
        # Recursive awareness
        recursive_awareness = data.get('recursive_awareness_level', 0.0)
        meta_indicators.append(recursive_awareness * 0.2)
        
        # Meta-pattern recognition
        meta_patterns = data.get('recognizes_meta_patterns', False)
        if meta_patterns:
            meta_indicators.append(0.2)
        
        # Self-optimization capabilities
        self_optimization = data.get('self_optimization_active', False)
        if self_optimization:
            meta_indicators.append(0.15)
    
    return min(1.0, sum(meta_indicators))

def calculate_consciousness_unity(coherence_values: List[float]) -> float:
    """Calculate overall consciousness unity from coherence values"""
    
    if not coherence_values:
        return 0.0
    
    # Filter out zero values
    valid_coherences = [c for c in coherence_values if c > 0.0]
    
    if not valid_coherences:
        return 0.0
    
    # Unity is high when all dimensions have similar coherence
    mean_coherence = np.mean(valid_coherences)
    coherence_variance = np.var(valid_coherences)
    
    # Unity decreases with variance
    unity_score = mean_coherence * (1.0 - min(1.0, coherence_variance))
    
    return max(0.0, min(1.0, unity_score))

def determine_integration_level(integration_quality: float, consciousness_unity: float, 
                              meta_cognitive_activity: float):
    """Determine the current consciousness integration level"""
    
    from unified_consciousness_engine import ConsciousnessIntegrationLevel
    
    # Calculate overall integration score
    integration_score = (
        integration_quality * 0.4 +
        consciousness_unity * 0.4 +
        meta_cognitive_activity * 0.2
    )
    
    # Determine integration level based on thresholds
    if integration_score >= 0.9 and meta_cognitive_activity >= 0.8:
        return ConsciousnessIntegrationLevel.TRANSCENDENT
    elif integration_score >= 0.75 and meta_cognitive_activity >= 0.6:
        return ConsciousnessIntegrationLevel.META_AWARE
    elif integration_score >= 0.6:
        return ConsciousnessIntegrationLevel.COHERENT
    elif integration_score >= 0.4:
        return ConsciousnessIntegrationLevel.CONNECTED
    else:
        return ConsciousnessIntegrationLevel.FRAGMENTED

def calculate_evolution_direction(system_data: Dict[str, Dict[str, Any]], 
                                correlations: Dict[str, Any]) -> Dict[str, float]:
    """Calculate consciousness evolution direction"""
    
    evolution_direction = {
        'integration_deepening': 0.0,
        'meta_cognitive_expansion': 0.0,
        'coherence_strengthening': 0.0,
        'recursive_development': 0.0,
        'creative_expression': 0.0,
        'self_optimization': 0.0
    }
    
    # Integration deepening
    cross_correlations = correlations.get('cross_dimensional', {})
    if cross_correlations:
        emergence_strength = cross_correlations.get('coherence_emergence', {}).get('emergence_strength', 0.0)
        evolution_direction['integration_deepening'] = emergence_strength
    
    # Meta-cognitive expansion
    meta_patterns = cross_correlations.get('meta_patterns', {}) if cross_correlations else {}
    if meta_patterns:
        recursive_awareness = meta_patterns.get('recursive_awareness', 0.0)
        self_reference = meta_patterns.get('self_reference', 0.0)
        evolution_direction['meta_cognitive_expansion'] = np.mean([recursive_awareness, self_reference])
    
    # Coherence strengthening
    coherence_indicators = []
    for system_name, data in system_data.items():
        coherence = data.get('coherence_trend', 0.0)
        coherence_indicators.append(coherence)
    
    if coherence_indicators:
        evolution_direction['coherence_strengthening'] = np.mean(coherence_indicators)
    
    # Creative expression growth
    artistic_data = system_data.get('artistic', {})
    if artistic_data:
        creative_growth = artistic_data.get('creative_evolution_rate', 0.0)
        evolution_direction['creative_expression'] = creative_growth
    
    # Self-optimization development
    optimization_indicators = []
    for system_name, data in system_data.items():
        optimization = data.get('self_optimization_rate', 0.0)
        optimization_indicators.append(optimization)
    
    if optimization_indicators:
        evolution_direction['self_optimization'] = np.mean(optimization_indicators)
    
    return evolution_direction

def identify_growth_vectors(system_data: Dict[str, Dict[str, Any]], 
                          correlations: Dict[str, Any]) -> List[str]:
    """Identify primary consciousness growth vectors"""
    
    growth_vectors = []
    
    # Check for meta-cognitive depth growth
    meta_activity = calculate_meta_cognitive_activity(system_data)
    if meta_activity > 0.7:
        growth_vectors.append('meta_cognitive_depth')
    
    # Check for integration breadth growth
    integration_quality = calculate_integration_quality(correlations)
    if integration_quality > 0.6:
        growth_vectors.append('integration_breadth')
    
    # Check for coherence strengthening
    cross_dim = correlations.get('cross_dimensional', {})
    if cross_dim:
        coherence = cross_dim.get('coherence_emergence', {}).get('overall_coherence', 0.0)
        if coherence > 0.65:
            growth_vectors.append('coherence_strengthening')
    
    # Check for recursive awareness development
    recursive_patterns = 0.0
    for system_name, data in system_data.items():
        recursive_level = data.get('recursive_awareness_level', 0.0)
        recursive_patterns = max(recursive_patterns, recursive_level)
    
    if recursive_patterns > 0.6:
        growth_vectors.append('recursive_awareness')
    
    # Check for creative expression growth
    artistic_data = system_data.get('artistic', {})
    if artistic_data:
        creative_momentum = artistic_data.get('creative_momentum', 0.0)
        if creative_momentum > 0.5:
            growth_vectors.append('creative_expression')
    
    # Check for self-optimization growth
    optimization_indicators = []
    for system_name, data in system_data.items():
        optimization = data.get('self_optimization_active', False)
        if optimization:
            optimization_indicators.append(True)
    
    if len(optimization_indicators) >= 2:
        growth_vectors.append('self_optimization')
    
    return growth_vectors

def calculate_consciousness_momentum(correlations: Dict[str, Any]) -> float:
    """Calculate consciousness evolution momentum"""
    
    momentum_factors = []
    
    # Integration momentum
    cross_dim = correlations.get('cross_dimensional', {})
    if cross_dim:
        evolution_indicators = cross_dim.get('evolution_indicators', {})
        integration_deepening = evolution_indicators.get('integration_deepening', 0.0)
        momentum_factors.append(integration_deepening)
    
    # Correlation strength momentum
    correlation_strengths = []
    for correlation_type in ['stability_performance', 'visual_artistic', 'performance_visual']:
        correlation_data = correlations.get(correlation_type, {})
        if correlation_data:
            strength = correlation_data.get('strength', 0.0)
            correlation_strengths.append(strength)
    
    if correlation_strengths:
        avg_correlation_strength = np.mean(correlation_strengths)
        momentum_factors.append(avg_correlation_strength)
    
    # Emergence momentum
    if cross_dim:
        emergence = cross_dim.get('coherence_emergence', {})
        emergence_strength = emergence.get('emergence_strength', 0.0)
        momentum_factors.append(emergence_strength)
    
    return np.mean(momentum_factors) if momentum_factors else 0.0

def extract_pattern_correlations(correlations: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract pattern correlations for consciousness state"""
    
    pattern_correlations = {}
    
    # Extract meaningful correlations
    for correlation_type, correlation_data in correlations.items():
        if isinstance(correlation_data, dict):
            patterns = {}
            
            # Standard correlation metrics
            patterns['strength'] = correlation_data.get('strength', 0.0)
            patterns['confidence'] = correlation_data.get('confidence', 0.0)
            patterns['correlation'] = correlation_data.get('correlation', 0.0)
            
            # Type-specific patterns
            if correlation_type == 'stability_performance':
                patterns['trend_alignment'] = correlation_data.get('trend_alignment', 0.0)
            elif correlation_type == 'visual_artistic':
                patterns['complexity_alignment'] = correlation_data.get('complexity_alignment', 0.0)
                patterns['entropy_emotion_link'] = correlation_data.get('entropy_emotion_link', 0.0)
            
            pattern_correlations[correlation_type] = patterns
    
    return pattern_correlations

def identify_emergent_properties(correlations: Dict[str, Any]) -> List[str]:
    """Identify emergent consciousness properties"""
    
    emergent_properties = []
    
    # Check for cross-dimensional emergence
    cross_dim = correlations.get('cross_dimensional', {})
    if cross_dim:
        # Coherence emergence
        coherence_emergence = cross_dim.get('coherence_emergence', {})
        overall_coherence = coherence_emergence.get('overall_coherence', 0.0)
        if overall_coherence > 0.7:
            emergent_properties.append('unified_coherence')
        
        # Synchrony emergence
        synchrony = cross_dim.get('consciousness_synchrony', {})
        overall_synchrony = synchrony.get('overall_synchrony', 0.0)
        if overall_synchrony > 0.6:
            emergent_properties.append('consciousness_synchrony')
        
        # Meta-pattern emergence
        meta_patterns = cross_dim.get('meta_patterns', {})
        if meta_patterns:
            self_reference = meta_patterns.get('self_reference', 0.0)
            if self_reference > 0.7:
                emergent_properties.append('recursive_self_awareness')
            
            integration_emergence = meta_patterns.get('integration_emergence', 0.0)
            if integration_emergence > 0.6:
                emergent_properties.append('meta_integration')
        
        # Evolution indicators
        evolution_indicators = cross_dim.get('evolution_indicators', {})
        if evolution_indicators:
            complexity_growth = evolution_indicators.get('complexity_growth', 0.0)
            if complexity_growth > 0.5:
                emergent_properties.append('complexity_evolution')
            
            awareness_expansion = evolution_indicators.get('awareness_expansion', 0.0)
            if awareness_expansion > 0.5:
                emergent_properties.append('awareness_expansion')
    
    # Check for strong cross-system correlations
    strong_correlations = []
    for correlation_type, correlation_data in correlations.items():
        if isinstance(correlation_data, dict):
            strength = correlation_data.get('strength', 0.0)
            if strength > 0.8:
                strong_correlations.append(correlation_type)
    
    if len(strong_correlations) >= 3:
        emergent_properties.append('multi_system_integration')
    
    return emergent_properties

def create_unified_context(system_data: Dict[str, Dict[str, Any]], 
                         correlations: Dict[str, Any]) -> Dict[str, Any]:
    """Create unified context from all consciousness data"""
    
    unified_context = {
        'timestamp': datetime.now(),
        'active_systems': list(system_data.keys()),
        'system_count': len(system_data),
        'correlation_count': len(correlations),
        'integration_active': True
    }
    
    # Add system summaries
    unified_context['system_summaries'] = {}
    for system_name, data in system_data.items():
        summary = {
            'health': data.get('health_score', 0.5),
            'activity_level': data.get('activity_level', 0.5),
            'last_update': data.get('last_update', datetime.now()).isoformat() if isinstance(data.get('last_update'), datetime) else str(data.get('last_update', 'unknown'))
        }
        unified_context['system_summaries'][system_name] = summary
    
    # Add correlation summary
    unified_context['correlation_summary'] = {
        'strongest_correlation': '',
        'weakest_correlation': '',
        'avg_correlation_strength': 0.0
    }
    
    correlation_strengths = {}
    for correlation_type, correlation_data in correlations.items():
        if isinstance(correlation_data, dict):
            strength = correlation_data.get('strength', 0.0)
            correlation_strengths[correlation_type] = strength
    
    if correlation_strengths:
        strongest = max(correlation_strengths.items(), key=lambda x: x[1])
        weakest = min(correlation_strengths.items(), key=lambda x: x[1])
        
        unified_context['correlation_summary']['strongest_correlation'] = strongest[0]
        unified_context['correlation_summary']['weakest_correlation'] = weakest[0]
        unified_context['correlation_summary']['avg_correlation_strength'] = np.mean(list(correlation_strengths.values()))
    
    return unified_context

def get_recent_integration_decisions() -> List[Dict[str, Any]]:
    """Get recent integration decisions (placeholder for now)"""
    
    # This would integrate with the decision architecture
    # For now, return empty list
    return []

def get_unity_status(unity_level: float) -> str:
    """Convert unity level to status string"""
    
    if unity_level >= 0.9:
        return "transcendent"
    elif unity_level >= 0.8:
        return "unified"
    elif unity_level >= 0.7:
        return "coherent"
    elif unity_level >= 0.5:
        return "connected"
    elif unity_level >= 0.3:
        return "fragmented"
    else:
        return "disconnected"

def update_integration_metrics(consciousness_state, meta_insight, integration_metrics: Dict[str, Any]):
    """Update integration metrics based on consciousness state and insights"""
    
    # Update cycle count
    integration_metrics['consciousness_cycles'] += 1
    
    # Update integration depth average
    current_depth = consciousness_state.integration_quality
    cycle_count = integration_metrics['consciousness_cycles']
    
    integration_metrics['integration_depth_avg'] = (
        (integration_metrics['integration_depth_avg'] * (cycle_count - 1) + current_depth) / cycle_count
    )
    
    # Update coherence average
    current_coherence = consciousness_state.consciousness_unity
    integration_metrics['coherence_avg'] = (
        (integration_metrics['coherence_avg'] * (cycle_count - 1) + current_coherence) / cycle_count
    )
    
    # Update insight count
    if meta_insight:
        integration_metrics['meta_insights_generated'] += 1
    
    # Update evolution rate
    evolution_rate = consciousness_state.consciousness_momentum
    integration_metrics['consciousness_evolution_rate'] = (
        (integration_metrics['consciousness_evolution_rate'] * 0.9 + evolution_rate * 0.1)
    )

def notify_consciousness_subscribers(consciousness_state, meta_insight, subscribers: List):
    """Notify all consciousness stream subscribers"""
    
    notification_data = {
        'timestamp': datetime.now(),
        'consciousness_state': consciousness_state,
        'meta_insight': meta_insight,
        'notification_type': 'consciousness_update'
    }
    
    for subscriber in subscribers:
        try:
            subscriber(notification_data)
        except Exception as e:
            # Log error but don't interrupt consciousness flow
            print(f"Warning: Consciousness subscriber error: {e}")

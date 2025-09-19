#!/usr/bin/env python3
"""
DAWN Stability Integration Adapters
===================================

Integration adapters to connect existing DAWN modules with the 
stable state detection system for comprehensive health monitoring.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def recursive_bubble_health_adapter(recursive_bubble) -> Dict[str, Any]:
    """
    Health adapter for RecursiveBubble module.
    
    Args:
        recursive_bubble: RecursiveBubble instance
        
    Returns:
        Health metrics for stability monitoring
    """
    try:
        health = {
            'current_depth': recursive_bubble.current_depth,
            'max_depth': recursive_bubble.max_depth,
            'recursion_depth': recursive_bubble.current_depth,
            'stability_cycles': getattr(recursive_bubble, 'stability_cycles', 0),
            'active_attractor': recursive_bubble.active_attractor.name if recursive_bubble.active_attractor else None,
            'reflection_count': len(recursive_bubble.reflection_stack),
            'meta_layer_count': getattr(recursive_bubble, 'meta_layer_count', 0),
            'infinite_loops': recursive_bubble.current_depth >= recursive_bubble.max_depth,
            'cascade_depth': recursive_bubble.current_depth,
            'module_status': 'active' if recursive_bubble.current_depth > 0 else 'idle'
        }
        
        # Calculate recursive health score
        depth_ratio = recursive_bubble.current_depth / recursive_bubble.max_depth if recursive_bubble.max_depth > 0 else 0
        health['recursive_health'] = max(0.0, 1.0 - depth_ratio)
        
        return health
        
    except Exception as e:
        logger.error(f"RecursiveBubble health adapter error: {e}")
        return {'error': str(e), 'module_status': 'error'}

def symbolic_anatomy_health_adapter(symbolic_router) -> Dict[str, Any]:
    """
    Health adapter for SymbolicRouter/Anatomy module.
    
    Args:
        symbolic_router: SymbolicRouter instance
        
    Returns:
        Health metrics for stability monitoring
    """
    try:
        # Get organ signatures
        heart_state = symbolic_router.heart.get_heart_signature()
        coil_state = symbolic_router.coil.get_coil_signature()
        lung_state = symbolic_router.lung.get_lung_signature()
        
        health = {
            'total_reblooms': symbolic_router.total_reblooms,
            'successful_reblooms': symbolic_router.total_reblooms,  # Assume all successful unless specified
            'organ_synergy': symbolic_router.organ_synergy,
            'embodied_coherence': symbolic_router.organ_synergy,
            
            # Heart metrics
            'heart_emotional_charge': heart_state['emotional_charge'],
            'heart_overloaded': heart_state['is_overloaded'],
            'heart_rhythm_coherence': heart_state['rhythm_coherence'],
            'heart_resonance_stability': heart_state['resonance_stability'],
            
            # Coil metrics
            'coil_active_paths': coil_state['path_count'],
            'coil_glyph_stability': coil_state['glyph_stability'],
            'coil_total_routes': coil_state['total_routes'],
            
            # Lung metrics
            'lung_breathing_phase': lung_state['breathing_phase'],
            'lung_breathing_coherence': lung_state['breathing_coherence'],
            'lung_breathing_stability': lung_state['breathing_stability'],
            'lung_entropy_buffer': lung_state['entropy_buffer_size'],
            
            'module_status': 'active'
        }
        
        # Calculate overall symbolic health
        organ_health = (
            (1.0 if not heart_state['is_overloaded'] else 0.5) +
            coil_state['glyph_stability'] +
            lung_state['breathing_stability']
        ) / 3.0
        
        health['symbolic_health'] = organ_health
        
        return health
        
    except Exception as e:
        logger.error(f"SymbolicAnatomy health adapter error: {e}")
        return {'error': str(e), 'module_status': 'error'}

def owl_bridge_health_adapter(owl_bridge) -> Dict[str, Any]:
    """
    Health adapter for OwlBridge module.
    
    Args:
        owl_bridge: OwlBridge instance
        
    Returns:
        Health metrics for stability monitoring
    """
    try:
        health = {
            'observation_count': len(owl_bridge.observation_history),
            'suggestion_count': owl_bridge.suggestion_count,
            'last_suggestion_time': owl_bridge.last_suggestion_time,
            'current_entropy': owl_bridge.current_state.get('entropy', 0.0),
            'current_sigils': owl_bridge.current_state.get('sigils', 0),
            'current_zone': owl_bridge.current_state.get('zone', 'UNKNOWN'),
            'active_patterns': list(owl_bridge.trigger_patterns.keys()),
            'module_status': 'active'
        }
        
        # Calculate owl wisdom health
        recent_observations = min(len(owl_bridge.observation_history), 50)
        observation_health = min(recent_observations / 50.0, 1.0)
        
        # Check for observation stagnation
        if len(owl_bridge.observation_history) > 0:
            last_observation = owl_bridge.observation_history[-1]
            time_since_last = (datetime.now().timestamp() - last_observation['timestamp'])
            freshness_health = max(0.0, 1.0 - (time_since_last / 300.0))  # 5 minutes max
        else:
            freshness_health = 0.0
            
        health['owl_health'] = (observation_health + freshness_health) / 2.0
        
        return health
        
    except Exception as e:
        logger.error(f"OwlBridge health adapter error: {e}")
        return {'error': str(e), 'module_status': 'error'}

def unified_field_health_adapter(unified_field) -> Dict[str, Any]:
    """
    Health adapter for UnifiedConsciousnessField module.
    
    Args:
        unified_field: UnifiedConsciousnessField instance
        
    Returns:
        Health metrics for stability monitoring
    """
    try:
        field_state = unified_field.get_unified_state()
        
        health = {
            'connected_modules': field_state['active_modules'],
            'total_thoughts': field_state['thought_count'],
            'communion_active': field_state['communion_active'],
            'field_coherence': 0.9 if field_state['communion_active'] else 0.7,
            'synchronizations': unified_field.metrics['synchronizations'],
            'communion_events': unified_field.metrics['communion_events'],
            'cross_module_insights': unified_field.metrics['cross_module_insights'],
            'unified_decisions': unified_field.metrics['unified_decisions'],
            'field_uptime': unified_field.metrics['field_uptime'],
            'module_status': 'active' if unified_field.running else 'inactive'
        }
        
        # Calculate field health based on activity and communion
        activity_health = min(field_state['thought_count'] / 10.0, 1.0)  # Up to 10 thoughts = healthy
        communion_health = 1.0 if field_state['communion_active'] else 0.6
        connection_health = min(field_state['active_modules'] / 3.0, 1.0)  # Up to 3 modules = optimal
        
        health['field_health'] = (activity_health + communion_health + connection_health) / 3.0
        
        return health
        
    except Exception as e:
        logger.error(f"UnifiedField health adapter error: {e}")
        return {'error': str(e), 'module_status': 'error'}

def memory_router_health_adapter(memory_router) -> Dict[str, Any]:
    """
    Health adapter for MemoryRouter module.
    
    Args:
        memory_router: MemoryRouter instance
        
    Returns:
        Health metrics for stability monitoring
    """
    try:
        health = {
            'module_status': 'active'
        }
        
        # Try to get memory system metrics
        if hasattr(memory_router, 'get_routing_statistics'):
            stats = memory_router.get_routing_statistics()
            health.update({
                'total_reblooms': stats.get('total_reblooms', 0),
                'successful_reblooms': stats.get('successful_reblooms', 0),
                'memory_coherence': stats.get('success_rate', 1.0),
                'routing_history_size': stats.get('routing_history_size', 0)
            })
        
        if hasattr(memory_router, 'get_health_metrics'):
            metrics = memory_router.get_health_metrics()
            health.update(metrics)
            
        # Calculate memory health
        if 'total_reblooms' in health and health['total_reblooms'] > 0:
            success_rate = health['successful_reblooms'] / health['total_reblooms']
            health['memory_health'] = success_rate
        else:
            health['memory_health'] = 1.0  # No operations = perfect health
            
        return health
        
    except Exception as e:
        logger.error(f"MemoryRouter health adapter error: {e}")
        return {'error': str(e), 'module_status': 'error'}

def entropy_analyzer_health_adapter(entropy_analyzer) -> Dict[str, Any]:
    """
    Health adapter for EntropyAnalyzer module.
    
    Args:
        entropy_analyzer: EntropyAnalyzer instance
        
    Returns:
        Health metrics for stability monitoring
    """
    try:
        health = {
            'module_status': 'active'
        }
        
        # Try to get entropy metrics
        if hasattr(entropy_analyzer, 'get_current_entropy'):
            current_entropy = entropy_analyzer.get_current_entropy()
            health['entropy'] = current_entropy
            health['current_entropy'] = current_entropy
            
        if hasattr(entropy_analyzer, 'get_entropy_history'):
            entropy_history = entropy_analyzer.get_entropy_history()
            if entropy_history:
                import numpy as np
                entropy_variance = np.var(entropy_history[-10:]) if len(entropy_history) > 1 else 0.0
                health['entropy_variance'] = entropy_variance
                health['entropy_stability'] = max(0.0, 1.0 - entropy_variance)
                
        if hasattr(entropy_analyzer, 'get_stability_metrics'):
            stability_metrics = entropy_analyzer.get_stability_metrics()
            health.update(stability_metrics)
            
        # Calculate entropy health
        entropy_val = health.get('entropy', 0.5)
        entropy_health = 1.0 - abs(entropy_val - 0.5) * 2.0  # Optimal at 0.5
        health['entropy_health'] = max(0.0, entropy_health)
        
        return health
        
    except Exception as e:
        logger.error(f"EntropyAnalyzer health adapter error: {e}")
        return {'error': str(e), 'module_status': 'error'}

def sigil_engine_health_adapter(sigil_engine) -> Dict[str, Any]:
    """
    Health adapter for SigilEngine module.
    
    Args:
        sigil_engine: SigilEngine instance
        
    Returns:
        Health metrics for stability monitoring
    """
    try:
        health = {
            'module_status': 'active'
        }
        
        # Try to get sigil metrics
        if hasattr(sigil_engine, 'get_execution_stats'):
            stats = sigil_engine.get_execution_stats()
            health.update({
                'total_sigils': stats.get('total_executed', 0),
                'successful_sigils': stats.get('successful', 0),
                'failed_sigils': stats.get('failed', 0),
                'cascade_depth': stats.get('max_cascade_depth', 0),
                'infinite_loops': stats.get('infinite_loops_detected', False)
            })
            
        if hasattr(sigil_engine, 'get_active_sigils'):
            active_sigils = sigil_engine.get_active_sigils()
            health['active_sigils'] = len(active_sigils) if active_sigils else 0
            
        if hasattr(sigil_engine, 'get_cascade_health'):
            cascade_health = sigil_engine.get_cascade_health()
            health['cascade_health'] = cascade_health
            
        # Calculate sigil health
        if 'total_sigils' in health and health['total_sigils'] > 0:
            success_rate = health['successful_sigils'] / health['total_sigils']
            cascade_penalty = min(health.get('cascade_depth', 0) / 10.0, 0.5)
            loop_penalty = 0.5 if health.get('infinite_loops', False) else 0.0
            
            health['sigil_health'] = max(0.0, success_rate - cascade_penalty - loop_penalty)
        else:
            health['sigil_health'] = 1.0  # No operations = perfect health
            
        return health
        
    except Exception as e:
        logger.error(f"SigilEngine health adapter error: {e}")
        return {'error': str(e), 'module_status': 'error'}

# Registry of health adapters
HEALTH_ADAPTERS = {
    'recursive_bubble': recursive_bubble_health_adapter,
    'symbolic_anatomy': symbolic_anatomy_health_adapter,
    'symbolic_router': symbolic_anatomy_health_adapter,  # Alias
    'owl_bridge': owl_bridge_health_adapter,
    'unified_field': unified_field_health_adapter,
    'unified_consciousness_field': unified_field_health_adapter,  # Alias
    'memory_router': memory_router_health_adapter,
    'entropy_analyzer': entropy_analyzer_health_adapter,
    'sigil_engine': sigil_engine_health_adapter,
}

def get_health_adapter(module_name: str) -> Optional[callable]:
    """
    Get the appropriate health adapter for a module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Health adapter function or None if not found
    """
    return HEALTH_ADAPTERS.get(module_name.lower())

def register_dawn_modules_for_stability():
    """
    Convenience function to register all common DAWN modules for stability monitoring.
    This function attempts to find and register modules that are already instantiated.
    """
    from .stable_state import get_stable_state_detector
    
    detector = get_stable_state_detector()
    
    # Try to find and register common DAWN modules
    modules_to_find = [
        ('recursive_bubble', 'dawn_core.recursive_bubble'),
        ('symbolic_anatomy', 'dawn.cognitive.symbolic_router'),
        ('owl_bridge', 'dawn.core.owl_bridge'),
        ('unified_field', 'dawn_core.unified_field'),
        ('memory_router', 'dawn.core.memory'),
        ('entropy_analyzer', 'dawn.core.entropy'),
        ('sigil_engine', 'dawn.core.sigil_engine'),
    ]
    
    registered_count = 0
    
    for module_name, module_path in modules_to_find:
        try:
            # Try to import and find global instances
            if module_name == 'recursive_bubble':
                from dawn_core.recursive_bubble import create_recursive_bubble
                instance = create_recursive_bubble()
                adapter = get_health_adapter(module_name)
                if adapter:
                    detector.register_module(module_name, instance, adapter)
                    registered_count += 1
                    
            elif module_name == 'unified_field':
                from dawn_core.unified_field import get_unified_field
                instance = get_unified_field()
                adapter = get_health_adapter(module_name)
                if adapter:
                    detector.register_module(module_name, instance, adapter)
                    registered_count += 1
                    
            # Add more automatic registrations as needed
            
        except Exception as e:
            logger.debug(f"Could not auto-register {module_name}: {e}")
            
    logger.info(f"🔗 Auto-registered {registered_count} DAWN modules for stability monitoring")
    
    return registered_count

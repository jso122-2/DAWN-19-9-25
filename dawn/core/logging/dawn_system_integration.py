#!/usr/bin/env python3
"""
🔗 DAWN System Universal JSON Logging Integration
=================================================

This module provides comprehensive integration of universal JSON logging
into all major DAWN systems, engines, and components. It ensures that
every process and module writes its state to disk in JSON/JSONL format.

Major Systems Integrated:
- Consciousness Engine (primary_engine.py)
- Tick Orchestrator (orchestrator.py)
- All Subsystem Engines and Managers
- Memory Systems (fractal, shimmer, rebloom)
- Visual Systems (consciousness, rendering)
- Thermal/Pulse Systems
- Schema and Sigil Systems
- Mycelial Layer Systems
- Self-Modification Systems
"""

import sys
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from pathlib import Path
import importlib

# Import our logging infrastructure
try:
    from .universal_json_logger import get_universal_logger, log_object_state, register_for_logging
    from .auto_integration import get_integration_manager, start_universal_logging
except ImportError:
    # Handle relative imports
    sys.path.append(str(Path(__file__).parent))
    from universal_json_logger import get_universal_logger, log_object_state, register_for_logging
    from auto_integration import get_integration_manager, start_universal_logging

logger = logging.getLogger(__name__)

class DAWNSystemIntegrator:
    """Comprehensive integrator for all DAWN systems"""
    
    def __init__(self):
        self.integrated_systems: Set[str] = set()
        self.system_objects: Dict[str, Any] = {}
        self.integration_callbacks: Dict[str, Callable] = {}
        
        # Get universal logger
        self.universal_logger = get_universal_logger()
        
        # Statistics
        self.stats = {
            'systems_integrated': 0,
            'objects_registered': 0,
            'integration_errors': 0,
            'start_time': time.time()
        }
    
    def integrate_all_dawn_systems(self) -> Dict[str, bool]:
        """Integrate all major DAWN systems with universal JSON logging"""
        integration_results = {}
        
        # Core consciousness systems
        integration_results.update(self._integrate_consciousness_systems())
        
        # Processing systems
        integration_results.update(self._integrate_processing_systems())
        
        # Subsystem engines
        integration_results.update(self._integrate_subsystem_engines())
        
        # Memory systems
        integration_results.update(self._integrate_memory_systems())
        
        # Visual systems
        integration_results.update(self._integrate_visual_systems())
        
        # Thermal/Pulse systems
        integration_results.update(self._integrate_thermal_systems())
        
        # Schema systems
        integration_results.update(self._integrate_schema_systems())
        
        # Self-modification systems
        integration_results.update(self._integrate_self_mod_systems())
        
        # Mycelial systems
        integration_results.update(self._integrate_mycelial_systems())
        
        logger.info(f"🔗 Integrated {len(integration_results)} DAWN systems")
        return integration_results
    
    def _integrate_consciousness_systems(self) -> Dict[str, bool]:
        """Integrate consciousness engines and systems"""
        results = {}
        
        try:
            # Primary consciousness engine
            from dawn.consciousness.engines.core.primary_engine import DAWNEngine
            results['consciousness.primary_engine'] = self._integrate_class(
                DAWNEngine, 'consciousness.primary_engine',
                methods_to_hook=['tick', 'register_module', 'unregister_module']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate primary engine: {e}")
            results['consciousness.primary_engine'] = False
        
        try:
            # Advanced consciousness tracer
            from dawn.consciousness.advanced_consciousness_tracer import AdvancedConsciousnessTracer
            results['consciousness.advanced_tracer'] = self._integrate_class(
                AdvancedConsciousnessTracer, 'consciousness.advanced_tracer',
                methods_to_hook=['tick', 'process_consciousness_data']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate consciousness tracer: {e}")
            results['consciousness.advanced_tracer'] = False
        
        try:
            # Unified pulse consciousness
            from dawn.consciousness.unified_pulse_consciousness import UnifiedPulseConsciousnessSystem
            results['consciousness.unified_pulse'] = self._integrate_class(
                UnifiedPulseConsciousnessSystem, 'consciousness.unified_pulse',
                methods_to_hook=['start_autonomous_consciousness', 'process_consciousness_tick']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate pulse consciousness: {e}")
            results['consciousness.unified_pulse'] = False
        
        return results
    
    def _integrate_processing_systems(self) -> Dict[str, bool]:
        """Integrate processing engines and orchestrators"""
        results = {}
        
        try:
            # Tick orchestrator
            from dawn.processing.engines.tick.synchronous.orchestrator import TickOrchestrator
            results['processing.tick_orchestrator'] = self._integrate_class(
                TickOrchestrator, 'processing.tick_orchestrator',
                methods_to_hook=['execute_unified_tick', 'register_module', 'tick_cycle']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate tick orchestrator: {e}")
            results['processing.tick_orchestrator'] = False
        
        try:
            # Communication consensus
            from dawn.core.communication.consensus import ConsensusEngine
            results['processing.consensus_engine'] = self._integrate_class(
                ConsensusEngine, 'processing.consensus_engine',
                methods_to_hook=['process_decision', 'update_consensus']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate consensus engine: {e}")
            results['processing.consensus_engine'] = False
        
        return results
    
    def _integrate_subsystem_engines(self) -> Dict[str, bool]:
        """Integrate all subsystem engines"""
        results = {}
        
        # Forecasting engine
        try:
            from dawn.subsystems.forecasting.unified_forecasting_engine import UnifiedForecastingEngine
            results['subsystems.forecasting_engine'] = self._integrate_class(
                UnifiedForecastingEngine, 'subsystems.forecasting_engine',
                methods_to_hook=['generate_forecast', 'update_model']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate forecasting engine: {e}")
            results['subsystems.forecasting_engine'] = False
        
        # Mood systems
        try:
            from dawn.subsystems.mood.tension_engine import TensionEngine
            results['subsystems.tension_engine'] = self._integrate_class(
                TensionEngine, 'subsystems.tension_engine',
                methods_to_hook=['process_tension', 'update_mood_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate tension engine: {e}")
            results['subsystems.tension_engine'] = False
        
        try:
            from dawn.subsystems.mood.emotional_override import EmotionalOverrideSystem
            results['subsystems.emotional_override'] = self._integrate_class(
                EmotionalOverrideSystem, 'subsystems.emotional_override',
                methods_to_hook=['process_override', 'update_emotional_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate emotional override: {e}")
            results['subsystems.emotional_override'] = False
        
        # Semantic systems
        try:
            from dawn.subsystems.semantic.semantic_context_engine import SemanticContextEngine
            results['subsystems.semantic_engine'] = self._integrate_class(
                SemanticContextEngine, 'subsystems.semantic_engine',
                methods_to_hook=['process_context', 'update_semantic_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate semantic engine: {e}")
            results['subsystems.semantic_engine'] = False
        
        return results
    
    def _integrate_memory_systems(self) -> Dict[str, bool]:
        """Integrate memory-related systems"""
        results = {}
        
        try:
            from dawn.subsystems.memory.fractal_memory_system import FractalMemorySystem
            results['memory.fractal_system'] = self._integrate_class(
                FractalMemorySystem, 'memory.fractal_system',
                methods_to_hook=['encode_memory', 'retrieve_memory', 'process_fractal']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate fractal memory: {e}")
            results['memory.fractal_system'] = False
        
        try:
            from dawn.subsystems.memory.shimmer_decay_engine import ShimmerDecayEngine
            results['memory.shimmer_decay'] = self._integrate_class(
                ShimmerDecayEngine, 'memory.shimmer_decay',
                methods_to_hook=['process_decay', 'update_shimmer_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate shimmer decay: {e}")
            results['memory.shimmer_decay'] = False
        
        try:
            from dawn.subsystems.memory.juliet_rebloom import JulietRebloomEngine
            results['memory.juliet_rebloom'] = self._integrate_class(
                JulietRebloomEngine, 'memory.juliet_rebloom',
                methods_to_hook=['trigger_rebloom', 'process_juliet_flower']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate juliet rebloom: {e}")
            results['memory.juliet_rebloom'] = False
        
        try:
            from dawn.subsystems.memory.carrin_hash_map import CarrinHashMap
            results['memory.carrin_hashmap'] = self._integrate_class(
                CarrinHashMap, 'memory.carrin_hashmap',
                methods_to_hook=['put', 'get', 'update_cache_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate carrin hashmap: {e}")
            results['memory.carrin_hashmap'] = False
        
        return results
    
    def _integrate_visual_systems(self) -> Dict[str, bool]:
        """Integrate visual consciousness and rendering systems"""
        results = {}
        
        try:
            from dawn.subsystems.visual.visual_consciousness import VisualConsciousnessEngine
            results['visual.consciousness_engine'] = self._integrate_class(
                VisualConsciousnessEngine, 'visual.consciousness_engine',
                methods_to_hook=['process_visual_data', 'update_visual_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate visual consciousness: {e}")
            results['visual.consciousness_engine'] = False
        
        try:
            from dawn.subsystems.visual.unified_consciousness_engine import UnifiedConsciousnessEngine
            results['visual.unified_engine'] = self._integrate_class(
                UnifiedConsciousnessEngine, 'visual.unified_engine',
                methods_to_hook=['render_consciousness', 'update_unified_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate unified consciousness engine: {e}")
            results['visual.unified_engine'] = False
        
        return results
    
    def _integrate_thermal_systems(self) -> Dict[str, bool]:
        """Integrate thermal and pulse systems"""
        results = {}
        
        try:
            from dawn.subsystems.thermal.pulse.pulse_engine import PulseEngine
            results['thermal.pulse_engine'] = self._integrate_class(
                PulseEngine, 'thermal.pulse_engine',
                methods_to_hook=['process_pulse', 'update_thermal_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate pulse engine: {e}")
            results['thermal.pulse_engine'] = False
        
        try:
            from dawn.subsystems.thermal.pulse.unified_pulse_system import UnifiedPulseSystem
            results['thermal.unified_pulse'] = self._integrate_class(
                UnifiedPulseSystem, 'thermal.unified_pulse',
                methods_to_hook=['process_unified_pulse', 'update_pulse_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate unified pulse system: {e}")
            results['thermal.unified_pulse'] = False
        
        return results
    
    def _integrate_schema_systems(self) -> Dict[str, bool]:
        """Integrate schema and sigil systems"""
        results = {}
        
        try:
            from dawn.subsystems.schema.core_schema_system import CoreSchemaSystem
            results['schema.core_system'] = self._integrate_class(
                CoreSchemaSystem, 'schema.core_system',
                methods_to_hook=['process_schema', 'update_schema_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate core schema system: {e}")
            results['schema.core_system'] = False
        
        try:
            from dawn.subsystems.schema.enhanced_scup_system import EnhancedSCUPSystem
            results['schema.enhanced_scup'] = self._integrate_class(
                EnhancedSCUPSystem, 'schema.enhanced_scup',
                methods_to_hook=['calculate_scup', 'update_scup_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate enhanced SCUP system: {e}")
            results['schema.enhanced_scup'] = False
        
        return results
    
    def _integrate_self_mod_systems(self) -> Dict[str, bool]:
        """Integrate self-modification systems"""
        results = {}
        
        try:
            from dawn.subsystems.self_mod.recursive_controller import RecursiveController
            results['self_mod.recursive_controller'] = self._integrate_class(
                RecursiveController, 'self_mod.recursive_controller',
                methods_to_hook=['process_modification', 'update_recursive_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate recursive controller: {e}")
            results['self_mod.recursive_controller'] = False
        
        try:
            from dawn.subsystems.self_mod.recursive_snapshots import RecursiveSnapshots
            results['self_mod.recursive_snapshots'] = self._integrate_class(
                RecursiveSnapshots, 'self_mod.recursive_snapshots',
                methods_to_hook=['take_snapshot', 'process_snapshot']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate recursive snapshots: {e}")
            results['self_mod.recursive_snapshots'] = False
        
        return results
    
    def _integrate_mycelial_systems(self) -> Dict[str, bool]:
        """Integrate mycelial layer systems"""
        results = {}
        
        try:
            from dawn.subsystems.mycelial.integrated_system import IntegratedMycelialSystem
            results['mycelial.integrated_system'] = self._integrate_class(
                IntegratedMycelialSystem, 'mycelial.integrated_system',
                methods_to_hook=['process_mycelial_tick', 'update_mycelial_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate mycelial system: {e}")
            results['mycelial.integrated_system'] = False
        
        try:
            from dawn.subsystems.mycelial.cluster_dynamics import ClusterManager
            results['mycelial.cluster_manager'] = self._integrate_class(
                ClusterManager, 'mycelial.cluster_manager',
                methods_to_hook=['process_clusters', 'update_cluster_state']
            )
        except ImportError as e:
            logger.debug(f"Could not integrate cluster manager: {e}")
            results['mycelial.cluster_manager'] = False
        
        return results
    
    def _integrate_class(self, cls: type, system_name: str, methods_to_hook: List[str]) -> bool:
        """Integrate a specific class with universal JSON logging"""
        try:
            # Store original methods
            original_methods = {}
            for method_name in methods_to_hook:
                if hasattr(cls, method_name):
                    original_methods[method_name] = getattr(cls, method_name)
            
            # Hook __init__ for automatic registration
            if hasattr(cls, '__init__'):
                original_init = cls.__init__
                
                def logged_init(self, *args, **kwargs):
                    original_init(self, *args, **kwargs)
                    # Register with universal logger
                    object_id = register_for_logging(self, f"{system_name}_{id(self)}")
                    self._universal_logging_id = object_id
                    self._universal_logging_system = system_name
                    
                    # Store reference for batch operations
                    self.integrator.system_objects[object_id] = self
                
                cls.__init__ = logged_init
            
            # Hook specified methods
            for method_name, original_method in original_methods.items():
                def create_logged_method(orig_method, method_name):
                    def logged_method(self, *args, **kwargs):
                        # Execute original method
                        result = orig_method(self, *args, **kwargs)
                        
                        # Log state after execution
                        try:
                            if hasattr(self, '_universal_logging_id'):
                                log_object_state(self, 
                                               custom_metadata={
                                                   'method_called': method_name,
                                                   'system_name': system_name,
                                                   'call_timestamp': time.time()
                                               })
                        except Exception as e:
                            logger.debug(f"Failed to log state after {method_name}: {e}")
                        
                        return result
                    
                    return logged_method
                
                setattr(cls, method_name, create_logged_method(original_method, method_name))
            
            # Mark class as integrated
            cls._universal_logging_integrated = True
            cls._universal_logging_system = system_name
            
            # Store reference to integrator
            cls.integrator = self
            
            self.integrated_systems.add(system_name)
            self.stats['systems_integrated'] += 1
            
            logger.debug(f"✅ Integrated {system_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to integrate {system_name}: {e}")
            self.stats['integration_errors'] += 1
            return False
    
    def log_all_system_states(self, tick_number: Optional[int] = None) -> int:
        """Log state of all integrated system objects"""
        logged_count = 0
        
        for object_id, obj in self.system_objects.items():
            try:
                if log_object_state(obj, custom_metadata={'tick_number': tick_number}):
                    logged_count += 1
            except Exception as e:
                logger.debug(f"Failed to log {object_id}: {e}")
        
        return logged_count
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        return {
            **self.stats,
            'runtime_seconds': time.time() - self.stats['start_time'],
            'integrated_systems': list(self.integrated_systems),
            'system_objects_count': len(self.system_objects),
            'universal_logger_stats': self.universal_logger.get_stats()
        }

# Global integrator instance
_dawn_integrator: Optional[DAWNSystemIntegrator] = None
_integrator_lock = threading.Lock()

def get_dawn_integrator() -> DAWNSystemIntegrator:
    """Get the global DAWN system integrator"""
    global _dawn_integrator
    
    with _integrator_lock:
        if _dawn_integrator is None:
            _dawn_integrator = DAWNSystemIntegrator()
        return _dawn_integrator

def integrate_all_dawn_systems() -> Dict[str, bool]:
    """Integrate all DAWN systems with universal JSON logging"""
    integrator = get_dawn_integrator()
    return integrator.integrate_all_dawn_systems()

def log_all_dawn_system_states(tick_number: Optional[int] = None) -> int:
    """Log state of all integrated DAWN systems"""
    integrator = get_dawn_integrator()
    return integrator.log_all_system_states(tick_number)

def get_dawn_integration_stats() -> Dict[str, Any]:
    """Get DAWN integration statistics"""
    integrator = get_dawn_integrator()
    return integrator.get_integration_stats()

# Convenience function to start complete DAWN logging
def start_complete_dawn_logging():
    """Start complete universal JSON logging for all DAWN systems"""
    logger.info("🚀 Starting complete DAWN universal JSON logging...")
    
    # Start universal logging infrastructure
    start_universal_logging()
    
    # Integrate all DAWN systems
    integration_results = integrate_all_dawn_systems()
    
    # Report results
    successful = sum(1 for success in integration_results.values() if success)
    total = len(integration_results)
    
    logger.info(f"✅ Complete DAWN logging started: {successful}/{total} systems integrated")
    
    return integration_results

if __name__ == "__main__":
    # Test the DAWN system integration
    logging.basicConfig(level=logging.INFO)
    
    # Start complete DAWN logging
    results = start_complete_dawn_logging()
    
    print("Integration Results:")
    for system, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {system}")
    
    # Get stats
    stats = get_dawn_integration_stats()
    print(f"\nIntegration Stats: {stats}")

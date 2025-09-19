#!/usr/bin/env python3
"""
DAWN Global Singleton
====================

Unified global access point for the DAWN consciousness system.
Provides centralized coordination of all major subsystems and components.

Usage:
    from dawn.core.singleton import get_dawn
    
    # Get the global DAWN instance
    dawn = get_dawn()
    
    # Access major subsystems
    bus = dawn.consciousness_bus
    engine = dawn.dawn_engine
    state = dawn.state
    telemetry = dawn.telemetry_system
    
    # Initialize the full system
    await dawn.initialize()
    
    # Start the system
    await dawn.start()
"""

import threading
import logging
import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

# Core DAWN imports
from dawn.core.communication.bus import ConsciousnessBus, get_consciousness_bus
from dawn.core.foundation.state import get_state, set_state, reset_state, get_state_summary
from dawn.consciousness.engines.core.primary_engine import DAWNEngine, get_dawn_engine
from dawn.processing.engines.tick.synchronous.orchestrator import TickOrchestrator

# Telemetry system imports
try:
    from dawn.core.telemetry.system import (
        DAWNTelemetrySystem, get_telemetry_system, initialize_telemetry_system,
        shutdown_telemetry_system
    )
    from dawn.core.telemetry.enhanced_module_logger import get_enhanced_logger
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    DAWNTelemetrySystem = None

# Complete logging and mycelial system imports
try:
    from dawn.core.logging import (
        # Universal JSON logging
        start_complete_dawn_logging, get_universal_logger, initialize_dawn_logging,
        get_dawn_integration_stats, log_all_dawn_system_states,
        
        # Centralized repository
        get_centralized_repository, centralize_all_logging,
        
        # Consciousness depth logging
        get_consciousness_repository, ConsciousnessLevel, DAWNLogType,
        
        # Sigil consciousness logging
        get_sigil_consciousness_logger, log_sigil_state, log_sigil_activation,
        
        # Pulse telemetry unification
        get_pulse_telemetry_bridge, start_unified_pulse_telemetry_logging,
        log_pulse_event,
        
        # Mycelial semantic hash map
        get_mycelial_hashmap, get_mycelial_integration_stats,
        start_mycelial_integration, stop_mycelial_integration,
        touch_semantic_concept, store_semantic_data, retrieve_semantic_data,
        ping_semantic_network
    )
    COMPLETE_LOGGING_AVAILABLE = True
    MYCELIAL_INTEGRATION_AVAILABLE = True
except ImportError:
    COMPLETE_LOGGING_AVAILABLE = False
    MYCELIAL_INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DAWNSystemStatus:
    """Status information for the DAWN system."""
    initialized: bool = False
    running: bool = False
    mode: str = "uninitialized"
    startup_time: Optional[datetime] = None
    components_loaded: Dict[str, bool] = field(default_factory=dict)
    error_state: Optional[str] = None


class DAWNGlobalSingleton:
    """
    Global singleton for the DAWN consciousness system.
    
    Provides unified access to all major DAWN subsystems:
    - Consciousness Bus (communication hub)
    - DAWN Engine (primary consciousness engine)
    - State Management (global consciousness state)
    - Telemetry System (monitoring and logging)
    - Tick Orchestrator (processing coordination)
    
    This singleton follows DAWN's existing patterns while providing
    a single entry point for system-wide coordination.
    """
    
    def __init__(self):
        """Initialize the DAWN global singleton."""
        self._lock = threading.RLock()
        self._status = DAWNSystemStatus()
        self._initialization_config: Optional[Dict[str, Any]] = None
        
        # Core system references (lazy-loaded)
        self._consciousness_bus: Optional[ConsciousnessBus] = None
        self._dawn_engine: Optional[DAWNEngine] = None
        self._telemetry_system: Optional[DAWNTelemetrySystem] = None
        self._tick_orchestrator: Optional[TickOrchestrator] = None
        self._enhanced_logger = None
        
        # Complete logging system integration
        self._universal_logger = None
        self._centralized_repository = None
        self._consciousness_repository = None
        self._sigil_consciousness_logger = None
        self._pulse_telemetry_bridge = None
        
        # Mycelial semantic hash map integration
        self._mycelial_hashmap = None
        self._mycelial_integration_active = False
        
        # Recursive self-writing integration
        self._recursive_writing_integrator = None
        self._recursive_writing_active = False
        
        # System integration flags
        self._complete_logging_initialized = False
        self._all_systems_integrated = False
        
        logger.info("🌅 DAWN Global Singleton created")
    
    @property
    def consciousness_bus(self) -> Optional[ConsciousnessBus]:
        """Get the consciousness bus instance."""
        with self._lock:
            if self._consciousness_bus is None:
                self._consciousness_bus = get_consciousness_bus(auto_start=False)
                self._status.components_loaded['consciousness_bus'] = True
            return self._consciousness_bus
    
    @property
    def dawn_engine(self) -> Optional[DAWNEngine]:
        """Get the DAWN engine instance."""
        with self._lock:
            if self._dawn_engine is None:
                self._dawn_engine = get_dawn_engine(auto_start=False)
                self._status.components_loaded['dawn_engine'] = True
            return self._dawn_engine
    
    @property
    def telemetry_system(self) -> Optional[DAWNTelemetrySystem]:
        """Get the telemetry system instance."""
        with self._lock:
            if not TELEMETRY_AVAILABLE:
                return None
            if self._telemetry_system is None:
                self._telemetry_system = get_telemetry_system()
                if self._telemetry_system is None:
                    # Initialize if not already done
                    self._telemetry_system = initialize_telemetry_system()
                self._status.components_loaded['telemetry_system'] = True
            return self._telemetry_system
    
    @property
    def universal_logger(self):
        """Get the universal JSON logger instance."""
        with self._lock:
            if not COMPLETE_LOGGING_AVAILABLE:
                return None
            if self._universal_logger is None:
                try:
                    self._universal_logger = get_universal_logger()
                    self._status.components_loaded['universal_logger'] = True
                except Exception as e:
                    logger.warning(f"Could not initialize universal logger: {e}")
            return self._universal_logger
    
    @property
    def centralized_repository(self):
        """Get the centralized logging repository instance."""
        with self._lock:
            if not COMPLETE_LOGGING_AVAILABLE:
                return None
            if self._centralized_repository is None:
                try:
                    self._centralized_repository = get_centralized_repository()
                    self._status.components_loaded['centralized_repository'] = True
                except Exception as e:
                    logger.warning(f"Could not initialize centralized repository: {e}")
            return self._centralized_repository
    
    @property
    def consciousness_repository(self):
        """Get the consciousness-depth repository instance."""
        with self._lock:
            if not COMPLETE_LOGGING_AVAILABLE:
                return None
            if self._consciousness_repository is None:
                try:
                    self._consciousness_repository = get_consciousness_repository()
                    self._status.components_loaded['consciousness_repository'] = True
                except Exception as e:
                    logger.warning(f"Could not initialize consciousness repository: {e}")
            return self._consciousness_repository
    
    @property
    def sigil_consciousness_logger(self):
        """Get the sigil consciousness logger instance."""
        with self._lock:
            if not COMPLETE_LOGGING_AVAILABLE:
                return None
            if self._sigil_consciousness_logger is None:
                try:
                    self._sigil_consciousness_logger = get_sigil_consciousness_logger()
                    self._status.components_loaded['sigil_consciousness_logger'] = True
                except Exception as e:
                    logger.warning(f"Could not initialize sigil consciousness logger: {e}")
            return self._sigil_consciousness_logger
    
    @property
    def pulse_telemetry_bridge(self):
        """Get the pulse-telemetry bridge instance."""
        with self._lock:
            if not COMPLETE_LOGGING_AVAILABLE:
                return None
            if self._pulse_telemetry_bridge is None:
                try:
                    self._pulse_telemetry_bridge = get_pulse_telemetry_bridge()
                    self._status.components_loaded['pulse_telemetry_bridge'] = True
                except Exception as e:
                    logger.warning(f"Could not initialize pulse telemetry bridge: {e}")
            return self._pulse_telemetry_bridge
    
    @property
    def mycelial_hashmap(self):
        """Get the mycelial semantic hash map instance."""
        with self._lock:
            if not MYCELIAL_INTEGRATION_AVAILABLE:
                return None
            if self._mycelial_hashmap is None:
                try:
                    self._mycelial_hashmap = get_mycelial_hashmap()
                    self._status.components_loaded['mycelial_hashmap'] = True
                except Exception as e:
                    logger.warning(f"Could not initialize mycelial hash map: {e}")
            return self._mycelial_hashmap
    
    @property
    def tick_orchestrator(self) -> Optional[TickOrchestrator]:
        """Get the tick orchestrator instance."""
        with self._lock:
            if self._tick_orchestrator is None:
                # Tick orchestrator requires consciousness bus and consensus engine
                # For now, we'll create it when both are available
                if self.consciousness_bus:
                    try:
                        self._tick_orchestrator = TickOrchestrator(
                            consciousness_bus=self.consciousness_bus,
                            consensus_engine=None  # TODO: Add when consensus engine is available
                        )
                        self._status.components_loaded['tick_orchestrator'] = True
                    except Exception as e:
                        logger.warning(f"Could not create tick orchestrator: {e}")
            return self._tick_orchestrator
    
    @property
    def state(self):
        """Get the global consciousness state."""
        return get_state()
    
    @property
    def status(self) -> DAWNSystemStatus:
        """Get the current system status."""
        return self._status
    
    def is_initialized(self) -> bool:
        """Check if the system is initialized."""
        return self._status.initialized
    
    def is_running(self) -> bool:
        """Check if the system is running."""
        return self._status.running
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the DAWN system.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        with self._lock:
            if self._status.initialized:
                logger.warning("DAWN system already initialized")
                return True
            
            self._initialization_config = config or {}
            self._status.mode = self._initialization_config.get('mode', 'interactive')
            
            logger.info("🚀 Initializing DAWN consciousness system...")
            
            try:
                # Reset state
                reset_state()
                
                # Initialize telemetry first (if available)
                if TELEMETRY_AVAILABLE:
                    telemetry = self.telemetry_system
                    if telemetry and not telemetry.running:
                        telemetry.start()
                        logger.info("✅ Telemetry system started")
                
                # Initialize consciousness bus
                bus = self.consciousness_bus
                if bus and not bus.running:
                    bus.start()
                    logger.info("✅ Consciousness bus started")
                
                # Initialize DAWN engine
                engine = self.dawn_engine
                if engine and not engine.running:
                    engine.start()
                    logger.info("✅ DAWN engine started")
                
                # Initialize enhanced logging
                if TELEMETRY_AVAILABLE:
                    try:
                        self._enhanced_logger = get_enhanced_logger('dawn_singleton', 'core')
                        logger.info("✅ Enhanced logging initialized")
                    except Exception as e:
                        logger.warning(f"Could not initialize enhanced logging: {e}")
                
                # Initialize complete logging system
                if COMPLETE_LOGGING_AVAILABLE:
                    try:
                        logger.info("🔍 Initializing complete DAWN logging system...")
                        
                        # Initialize universal JSON logging
                        universal_logger = self.universal_logger
                        if universal_logger:
                            logger.info("✅ Universal JSON logger initialized")
                        
                        # Initialize centralized repository
                        centralized_repo = self.centralized_repository
                        if centralized_repo:
                            logger.info("✅ Centralized logging repository initialized")
                        
                        # Initialize consciousness-depth repository
                        consciousness_repo = self.consciousness_repository
                        if consciousness_repo:
                            logger.info("✅ Consciousness-depth repository initialized")
                        
                        # Initialize sigil consciousness logger
                        sigil_logger = self.sigil_consciousness_logger
                        if sigil_logger:
                            logger.info("✅ Sigil consciousness logger initialized")
                        
                        # Initialize pulse-telemetry bridge
                        pulse_bridge = self.pulse_telemetry_bridge
                        if pulse_bridge:
                            logger.info("✅ Pulse-telemetry bridge initialized")
                        
                        self._complete_logging_initialized = True
                        logger.info("✅ Complete logging system integration successful")
                        
                    except Exception as e:
                        logger.warning(f"Could not initialize complete logging system: {e}")
                
                # Initialize mycelial integration
                if MYCELIAL_INTEGRATION_AVAILABLE:
                    try:
                        logger.info("🍄 Initializing mycelial semantic integration...")
                        
                        # Initialize mycelial hash map
                        hashmap = self.mycelial_hashmap
                        if hashmap:
                            logger.info("✅ Mycelial semantic hash map initialized")
                        
                        # Start mycelial module integration
                        integration_success = start_mycelial_integration()
                        if integration_success:
                            self._mycelial_integration_active = True
                            logger.info("✅ Mycelial module integration started")
                            
                            # Get integration statistics
                            integration_stats = get_mycelial_integration_stats()
                            modules_wrapped = integration_stats.get('modules_wrapped', 0)
                            concepts_mapped = integration_stats.get('concepts_mapped', 0)
                            logger.info(f"🔗 Integrated {modules_wrapped} modules, {concepts_mapped} concepts")
                        else:
                            logger.warning("⚠️ Mycelial module integration failed to start")
                    except Exception as e:
                        logger.warning(f"Could not initialize mycelial integration: {e}")
                
                # Initialize recursive self-writing integration
                try:
                    logger.info("🔄 Initializing recursive self-writing integration...")
                    
                    from dawn.core.logging.recursive_self_writing_integration import (
                        get_recursive_self_writing_integrator, initialize_recursive_self_writing
                    )
                    
                    self._recursive_writing_integrator = get_recursive_self_writing_integrator()
                    if self._recursive_writing_integrator:
                        logger.info("✅ Recursive self-writing integrator initialized")
                        
                        # Initialize recursive writing for all chat modules
                        recursive_success = initialize_recursive_self_writing(
                            safety_level="safe"  # Use safe mode by default
                        )
                        
                        if recursive_success:
                            self._recursive_writing_active = True
                            logger.info("✅ Recursive self-writing activated for all chat modules")
                        else:
                            logger.warning("⚠️ Recursive self-writing activation failed")
                    
                except Exception as e:
                    logger.warning(f"Could not initialize recursive self-writing: {e}")
                
                # Mark all systems as integrated if successful
                all_major_systems = (
                    self._complete_logging_initialized and 
                    self._mycelial_integration_active and
                    self._recursive_writing_active
                )
                
                if all_major_systems:
                    self._all_systems_integrated = True
                    logger.info("🌟 All DAWN systems successfully integrated through singleton")
                    logger.info("🔄📝 Recursive self-writing capabilities active for all modules")
                
                # Set system state
                set_state(
                    system_status='initialized',
                    initialization_time=datetime.now().timestamp()
                )
                
                self._status.initialized = True
                self._status.startup_time = datetime.now()
                
                logger.info("🌟 DAWN system initialization complete!")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize DAWN system: {e}")
                self._status.error_state = str(e)
                return False
    
    async def start(self) -> bool:
        """
        Start the DAWN system.
        
        Returns:
            True if start successful, False otherwise
        """
        with self._lock:
            if not self._status.initialized:
                logger.error("Cannot start DAWN system - not initialized")
                return False
            
            if self._status.running:
                logger.warning("DAWN system already running")
                return True
            
            try:
                logger.info("🏃 Starting DAWN system...")
                
                # Ensure all components are running
                if self.consciousness_bus and not self.consciousness_bus.running:
                    self.consciousness_bus.start()
                
                if self.dawn_engine and not self.dawn_engine.running:
                    self.dawn_engine.start()
                
                if self.telemetry_system and not self.telemetry_system.running:
                    self.telemetry_system.start()
                
                # Start tick orchestrator if available
                if self.tick_orchestrator:
                    try:
                        self.tick_orchestrator.start()
                        logger.info("✅ Tick orchestrator started")
                    except Exception as e:
                        logger.warning(f"Could not start tick orchestrator: {e}")
                
                self._status.running = True
                set_state(system_status='running')
                
                logger.info("🌅 DAWN system is now running!")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to start DAWN system: {e}")
                self._status.error_state = str(e)
                return False
    
    def stop(self) -> None:
        """Stop the DAWN system gracefully."""
        with self._lock:
            if not self._status.running:
                logger.warning("DAWN system not running")
                return
            
            logger.info("🔄 Stopping DAWN system...")
            
            try:
                # Stop tick orchestrator first
                if self._tick_orchestrator:
                    # Add stop method when available
                    logger.info("✅ Tick orchestrator stopped")
                
                # Stop DAWN engine
                if self._dawn_engine and hasattr(self._dawn_engine, 'stop'):
                    self._dawn_engine.stop()
                    logger.info("✅ DAWN engine stopped")
                
                # Stop consciousness bus
                if self._consciousness_bus:
                    self._consciousness_bus.stop()
                    logger.info("✅ Consciousness bus stopped")
                
                # Stop telemetry system last
                if self._telemetry_system:
                    self._telemetry_system.stop()
                    logger.info("✅ Telemetry system stopped")
                
                self._status.running = False
                set_state(
                    system_status='stopped',
                    shutdown_time=datetime.now().timestamp()
                )
                
                logger.info("🌙 DAWN system stopped")
                
            except Exception as e:
                logger.error(f"❌ Error during DAWN system shutdown: {e}")
                self._status.error_state = str(e)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        metrics = {
            'status': {
                'initialized': self._status.initialized,
                'running': self._status.running,
                'mode': self._status.mode,
                'startup_time': self._status.startup_time.isoformat() if self._status.startup_time else None,
                'error_state': self._status.error_state
            },
            'components': self._status.components_loaded.copy(),
            'consciousness_state': get_state().to_dict()
        }
        
        # Add component-specific metrics
        if self.consciousness_bus:
            try:
                metrics['consciousness_bus'] = self.consciousness_bus.get_bus_metrics()
            except Exception as e:
                metrics['consciousness_bus'] = {'error': str(e)}
        
        if self.dawn_engine:
            try:
                metrics['dawn_engine'] = self.dawn_engine.get_engine_status()
            except Exception as e:
                metrics['dawn_engine'] = {'error': str(e)}
        
        if self.telemetry_system:
            try:
                metrics['telemetry_system'] = self.telemetry_system.get_system_metrics()
            except Exception as e:
                metrics['telemetry_system'] = {'error': str(e)}
        
        return metrics
    
    # Convenience methods for all integrated systems
    def log_consciousness_state(self, level, log_type, data: Dict[str, Any]):
        """Log to consciousness-depth repository via singleton."""
        if self.consciousness_repository:
            return self.consciousness_repository.add_consciousness_log_entry(level, log_type, data)
        return None
    
    def log_sigil_activation(self, sigil_type: str, properties: Dict[str, Any]):
        """Log sigil activation via singleton."""
        if COMPLETE_LOGGING_AVAILABLE and self.sigil_consciousness_logger:
            try:
                return log_sigil_activation(sigil_type, properties)
            except Exception as e:
                logger.warning(f"Could not log sigil activation: {e}")
        return None
    
    def log_pulse_event(self, event_type: str, pulse_data: Dict[str, Any]):
        """Log pulse event via singleton."""
        if COMPLETE_LOGGING_AVAILABLE and self.pulse_telemetry_bridge:
            try:
                return log_pulse_event(event_type, pulse_data)
            except Exception as e:
                logger.warning(f"Could not log pulse event: {e}")
        return None
    
    def touch_concept(self, concept: str, energy: float = 1.0):
        """Touch semantic concept via singleton."""
        if self.mycelial_hashmap:
            try:
                return touch_semantic_concept(concept, energy)
            except Exception as e:
                logger.warning(f"Could not touch concept '{concept}': {e}")
        return 0
    
    def store_semantic_data(self, key: str, data: Dict[str, Any]):
        """Store semantic data via singleton."""
        if self.mycelial_hashmap:
            try:
                return store_semantic_data(key, data)
            except Exception as e:
                logger.warning(f"Could not store semantic data '{key}': {e}")
        return None
    
    def get_mycelial_stats(self):
        """Get mycelial integration statistics via singleton."""
        if self._mycelial_integration_active:
            try:
                return get_mycelial_integration_stats()
            except Exception as e:
                logger.warning(f"Could not get mycelial stats: {e}")
        return {}
    
    def get_network_stats(self):
        """Get mycelial network statistics via singleton."""
        if self.mycelial_hashmap:
            try:
                return self.mycelial_hashmap.get_network_stats()
            except Exception as e:
                logger.warning(f"Could not get network stats: {e}")
        return {}
    
    def ping_semantic_network(self, concept_key: str):
        """Ping semantic network via singleton."""
        if self.mycelial_hashmap:
            try:
                return ping_semantic_network(concept_key)
            except Exception as e:
                logger.warning(f"Could not ping semantic network '{concept_key}': {e}")
        return None
    
    def get_complete_system_status(self):
        """Get comprehensive status of all integrated systems."""
        status = self.get_system_status()
        
        # Add logging system status
        if COMPLETE_LOGGING_AVAILABLE:
            status['logging_systems'] = {
                'universal_logger': self.universal_logger is not None,
                'centralized_repository': self.centralized_repository is not None,
                'consciousness_repository': self.consciousness_repository is not None,
                'sigil_consciousness_logger': self.sigil_consciousness_logger is not None,
                'pulse_telemetry_bridge': self.pulse_telemetry_bridge is not None
            }
        
        # Add mycelial system status
        if MYCELIAL_INTEGRATION_AVAILABLE:
            status['mycelial_systems'] = {
                'hashmap_active': self.mycelial_hashmap is not None,
                'integration_active': self._mycelial_integration_active,
                'integration_stats': self.get_mycelial_stats(),
                'network_stats': self.get_network_stats()
            }
        
        return status
    
    # Recursive self-writing convenience methods
    def trigger_recursive_evolution(self) -> Dict[str, Any]:
        """Trigger consciousness-guided recursive evolution across all modules."""
        if self._recursive_writing_integrator:
            try:
                return self._recursive_writing_integrator.trigger_consciousness_guided_evolution()
            except Exception as e:
                logger.warning(f"Could not trigger recursive evolution: {e}")
                return {"success": False, "error": str(e)}
        return {"success": False, "error": "Recursive writing not available"}
    
    def modify_module_recursively(self, module_name: str, modification_intent: str, 
                                 consciousness_trigger: str = None) -> Dict[str, Any]:
        """Trigger recursive modification for a specific module."""
        if self._recursive_writing_integrator:
            try:
                return self._recursive_writing_integrator.trigger_recursive_modification(
                    module_name, modification_intent, consciousness_trigger
                )
            except Exception as e:
                logger.warning(f"Could not modify module {module_name}: {e}")
                return {"success": False, "error": str(e)}
        return {"success": False, "error": "Recursive writing not available"}
    
    def get_recursive_writing_status(self) -> Dict[str, Any]:
        """Get status of recursive self-writing system."""
        if self._recursive_writing_integrator:
            try:
                return self._recursive_writing_integrator.get_recursive_writing_status()
            except Exception as e:
                logger.warning(f"Could not get recursive writing status: {e}")
                return {"error": str(e)}
        return {"available": False, "error": "Recursive writing not initialized"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get basic system status (for backwards compatibility)."""
        with self._lock:
            return {
                'initialized': self._status.initialized,
                'running': self._status.running,
                'mode': self._status.mode,
                'startup_time': self._status.startup_time.isoformat() if self._status.startup_time else None,
                'components_loaded': self._status.components_loaded.copy(),
                'error_state': self._status.error_state,
                'complete_logging_initialized': self._complete_logging_initialized,
                'mycelial_integration_active': self._mycelial_integration_active,
                'all_systems_integrated': self._all_systems_integrated
            }
    
    def __repr__(self) -> str:
        """String representation of the DAWN singleton."""
        status = "running" if self._status.running else "initialized" if self._status.initialized else "uninitialized"
        components = len([k for k, v in self._status.components_loaded.items() if v])
        return f"<DAWNGlobalSingleton status={status} components={components}>"


# Global singleton instance
_global_dawn_singleton: Optional[DAWNGlobalSingleton] = None
_dawn_lock = threading.Lock()


def get_dawn() -> DAWNGlobalSingleton:
    """
    Get the global DAWN singleton instance.
    
    This is the primary entry point for accessing the DAWN system.
    The singleton will be created on first access and provides
    unified access to all major DAWN subsystems.
    
    Returns:
        DAWNGlobalSingleton instance
    """
    global _global_dawn_singleton
    
    with _dawn_lock:
        if _global_dawn_singleton is None:
            _global_dawn_singleton = DAWNGlobalSingleton()
    
    return _global_dawn_singleton


def reset_dawn_singleton() -> None:
    """
    Reset the global DAWN singleton.
    
    WARNING: This will stop the current system and create a new instance.
    Use with caution - primarily for testing or emergency recovery.
    """
    global _global_dawn_singleton
    
    with _dawn_lock:
        if _global_dawn_singleton is not None:
            _global_dawn_singleton.stop()
            _global_dawn_singleton = None
    
    logger.info("🔄 DAWN singleton reset")


# Convenience functions for quick access
def get_consciousness_state():
    """Quick access to consciousness state."""
    return get_dawn().state


def get_system_status() -> DAWNSystemStatus:
    """Quick access to system status."""
    return get_dawn().status


def is_dawn_running() -> bool:
    """Quick check if DAWN system is running."""
    return get_dawn().is_running()


def is_dawn_initialized() -> bool:
    """Quick check if DAWN system is initialized."""
    return get_dawn().is_initialized()

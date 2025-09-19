#!/usr/bin/env python3
"""
🔍 DAWN Universal JSON Logging System
====================================

Complete universal JSON logging system that ensures ABSOLUTELY EVERY process 
and module in the DAWN system writes its state to disk in JSON/JSONL format.

This package provides:
- Universal JSON logger for any Python object
- Automatic integration with all DAWN systems
- Zero-code-change integration via import hooks
- High-performance JSONL streaming
- State diffing and change detection
- Comprehensive system coverage

Usage:
    # Automatic integration (recommended)
    from dawn.core.logging import start_complete_dawn_logging
    start_complete_dawn_logging()
    
    # Manual object logging
    from dawn.core.logging import log_object_state
    log_object_state(my_object, name="my_object")
    
    # Register object for automatic logging
    from dawn.core.logging import register_for_logging
    register_for_logging(my_object, "my_object")
"""

# Core logging components
from .universal_json_logger import (
    UniversalJsonLogger,
    get_universal_logger,
    log_object_state,
    register_for_logging,
    auto_log_state,
    BatchLogging,
    LoggingConfig,
    LogFormat,
    StateScope
)

# Auto-integration system
from .auto_integration import (
    AutoIntegrationManager,
    get_integration_manager,
    start_universal_logging,
    stop_universal_logging,
    integrate_object,
    integrate_class,
    log_dawn_system_state,
    DAWNSystemBatchLogging
)

# DAWN system-specific integration
from .dawn_system_integration import (
    DAWNSystemIntegrator,
    get_dawn_integrator,
    integrate_all_dawn_systems,
    log_all_dawn_system_states,
    get_dawn_integration_stats,
    start_complete_dawn_logging
)

# Import TelemetryLevel for external use
try:
    from dawn.core.telemetry import TelemetryLevel
    TELEMETRY_LEVEL_AVAILABLE = True
except ImportError:
    from enum import Enum
    class TelemetryLevel(Enum):
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARN = "WARN"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"
    TELEMETRY_LEVEL_AVAILABLE = False

# Centralized deep repository
from .centralized_repo import (
    CentralizedLoggingRepository,
    get_centralized_repository,
    centralize_all_logging,
    LogEntry,
    RepositoryStats
)

# Consciousness-depth repository
from .consciousness_depth_repo import (
    ConsciousnessDepthRepository,
    ConsciousnessLevel,
    DAWNLogType,
    ConsciousnessLogEntry,
    ConsciousnessPathMapper,
    get_consciousness_repository
)

# Sigil consciousness logging
from .sigil_consciousness_logger import (
    SigilConsciousnessLogger,
    SigilConsciousnessType,
    SigilConsciousnessState,
    get_sigil_consciousness_logger,
    log_sigil_state,
    log_sigil_activation,
    log_archetypal_emergence
)

# Pulse-telemetry unification
from .pulse_telemetry_unifier import (
    PulseTelemetryBridge,
    ConsciousnessPulseLogger,
    TelemetryPulseCollector,
    PulseTelemetryEvent,
    UnifiedPulseMetrics,
    PulsePhase,
    PulseConsciousnessMapping,
    get_pulse_telemetry_bridge,
    start_unified_pulse_telemetry_logging,
    log_pulse_event
)

# Mycelial semantic hash map
from .mycelial_semantic_hashmap import (
    MycelialSemanticHashMap,
    SemanticSpore,
    MycelialHashNode,
    RhizomicPropagator,
    SemanticTelemetryCollector,
    SporeType,
    PropagationMode,
    NodeState,
    get_mycelial_hashmap,
    touch_semantic_concept,
    store_semantic_data,
    retrieve_semantic_data,
    ping_semantic_network
)

# Mycelial module integration
from .mycelial_module_integration import (
    MycelialModuleIntegrator,
    ModuleSemanticWrapper,
    SemanticPathMapper,
    CrossModulePropagator,
    FolderTopologyMapper,
    ModuleSemanticContext,
    SemanticExecutionTrace,
    get_mycelial_integrator,
    start_mycelial_integration,
    stop_mycelial_integration,
    integrate_module_with_mycelial,
    get_mycelial_integration_stats
)

# Recursive self-writing integration
from .recursive_self_writing_integration import (
    RecursiveSelfWritingIntegrator,
    RecursiveSelfWritingStatus,
    get_recursive_self_writing_integrator,
    initialize_recursive_self_writing,
    trigger_consciousness_evolution,
    get_recursive_writing_status
)

# Convenience imports
__all__ = [
    # Core logger
    'UniversalJsonLogger',
    'get_universal_logger',
    'log_object_state',
    'register_for_logging',
    'auto_log_state',
    'BatchLogging',
    'LoggingConfig',
    'LogFormat',
    'StateScope',
    
    # Auto-integration
    'AutoIntegrationManager',
    'get_integration_manager',
    'start_universal_logging',
    'stop_universal_logging',
    'integrate_object',
    'integrate_class',
    'log_dawn_system_state',
    'DAWNSystemBatchLogging',
    
    # DAWN system integration
    'DAWNSystemIntegrator',
    'get_dawn_integrator',
    'integrate_all_dawn_systems',
    'log_all_dawn_system_states',
    'get_dawn_integration_stats',
    'start_complete_dawn_logging',
    
    # Centralized repository
    'CentralizedLoggingRepository',
    'get_centralized_repository',
    'centralize_all_logging',
    'LogEntry',
    'RepositoryStats',
    
    # Consciousness-depth repository
    'ConsciousnessDepthRepository',
    'ConsciousnessLevel',
    'DAWNLogType',
    'ConsciousnessLogEntry',
    'ConsciousnessPathMapper',
    'get_consciousness_repository',
    
    # Sigil consciousness logging
    'SigilConsciousnessLogger',
    'SigilConsciousnessType', 
    'SigilConsciousnessState',
    'get_sigil_consciousness_logger',
    'log_sigil_state',
    'log_sigil_activation',
    'log_archetypal_emergence',
    
    # Pulse-telemetry unification
    'PulseTelemetryBridge',
    'ConsciousnessPulseLogger',
    'TelemetryPulseCollector',
    'PulseTelemetryEvent',
    'UnifiedPulseMetrics',
    'PulsePhase',
    'PulseConsciousnessMapping',
    'get_pulse_telemetry_bridge',
    'start_unified_pulse_telemetry_logging',
    'log_pulse_event',
    
    # Mycelial semantic hash map
    'MycelialSemanticHashMap',
    'SemanticSpore',
    'MycelialHashNode',
    'RhizomicPropagator',
    'SemanticTelemetryCollector',
    'SporeType',
    'PropagationMode',
    'NodeState',
    'get_mycelial_hashmap',
    'touch_semantic_concept',
    'store_semantic_data',
    'retrieve_semantic_data',
    'ping_semantic_network',
    
    # Mycelial module integration
    'MycelialModuleIntegrator',
    'ModuleSemanticWrapper',
    'SemanticPathMapper',
    'CrossModulePropagator',
    'FolderTopologyMapper',
    'ModuleSemanticContext',
    'SemanticExecutionTrace',
    'get_mycelial_integrator',
    'start_mycelial_integration',
    'stop_mycelial_integration',
    'integrate_module_with_mycelial',
    'get_mycelial_integration_stats',
    
    # Recursive self-writing integration
    'RecursiveSelfWritingIntegrator',
    'RecursiveSelfWritingStatus', 
    'get_recursive_self_writing_integrator',
    'initialize_recursive_self_writing',
    'trigger_consciousness_evolution',
    'get_recursive_writing_status',
    
    # Telemetry support
    'TelemetryLevel'
]

# Version info
__version__ = "1.0.0"
__author__ = "DAWN Universal Logging System"
__description__ = "Complete universal JSON logging for all DAWN processes and modules"

import logging
logger = logging.getLogger(__name__)

def initialize_dawn_logging(auto_start: bool = True, config: LoggingConfig = None, 
                           centralized: bool = True, repo_path: str = "logs_repository",
                           mycelial_integration: bool = True):
    """Initialize DAWN universal logging system with centralized repository and mycelial integration"""
    if auto_start:
        logger.info("🔍 Initializing DAWN Universal JSON Logging System")
        
        # Start universal logging
        result = start_complete_dawn_logging()
        
        # Initialize centralized repository if requested
        if centralized:
            logger.info("🗂️ Initializing centralized deep logging repository")
            centralize_all_logging(repo_path)
        
        # Initialize mycelial integration if requested
        if mycelial_integration:
            logger.info("🍄 Initializing mycelial module integration")
            try:
                mycelial_success = start_mycelial_integration()
                if mycelial_success:
                    logger.info("🍄✅ Mycelial integration active - all modules wired for spore propagation")
                else:
                    logger.warning("🍄⚠️ Mycelial integration failed to start")
            except Exception as e:
                logger.warning(f"🍄❌ Could not start mycelial integration: {e}")
        
        return result
    else:
        logger.info("🔍 DAWN Universal JSON Logging System available (manual start)")
        return None

# Auto-initialize if environment variable is set
import os
# DISABLED: Auto-initialization causes verbose output in CLI
# To enable, set DAWN_AUTO_LOGGING=1 environment variable
if os.getenv('DAWN_AUTO_LOGGING') == '1':
    try:
        initialize_dawn_logging(auto_start=True)
    except Exception as e:
        logger.warning(f"Failed to auto-initialize DAWN logging: {e}")

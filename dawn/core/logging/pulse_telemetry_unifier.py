#!/usr/bin/env python3
"""
🫁 DAWN Pulse-Telemetry Unifier
==============================

Unifies telemetry logging with pulse logic and modules, creating a comprehensive
system that integrates consciousness-depth logging with DAWN's pulse-based architecture.

This system bridges:
- Pulse system (tick orchestration, SCUP control, thermal management)
- Telemetry system (structured logging, metrics collection)
- Consciousness-depth logging (hierarchical organization by awareness levels)

The unified system provides:
- Pulse-aware consciousness logging at appropriate depth levels
- Telemetry integration with tick cycles and pulse zones
- Thermal state logging with consciousness metrics
- SCUP coherence tracking with depth-based organization
- Real-time pulse metrics with structured telemetry export

Architecture:
- PulseTelemetryBridge: Core integration layer
- ConsciousnessPulseLogger: Pulse-aware consciousness logging
- TelemetryPulseCollector: Pulse-synchronized telemetry collection
- UnifiedPulseMetrics: Combined pulse and consciousness metrics
"""

import time
import threading
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum

# Import DAWN core systems
try:
    from dawn.core.telemetry import DAWNTelemetrySystem, TelemetryLevel, TelemetryEvent
    from dawn.core.telemetry.enhanced_module_logger import EnhancedModuleLogger
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    # Mock TelemetryLevel if not available
    class TelemetryLevel(Enum):
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARN = "WARN"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"

try:
    from dawn.subsystems.thermal.pulse import (
        UnifiedPulseSystem, PulseZone, PulseActionType, SCUPState, PulseMetrics
    )
    PULSE_SYSTEM_AVAILABLE = True
except ImportError:
    PULSE_SYSTEM_AVAILABLE = False

try:
    from dawn.processing.engines.tick.synchronous.orchestrator import (
        TickOrchestrator, TickPhase, ModuleTickStatus
    )
    TICK_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    TICK_ORCHESTRATOR_AVAILABLE = False

# Import consciousness logging
from .consciousness_depth_repo import (
    ConsciousnessDepthRepository, ConsciousnessLevel, DAWNLogType,
    get_consciousness_repository
)
from .sigil_consciousness_logger import (
    SigilConsciousnessLogger, get_sigil_consciousness_logger
)

logger = logging.getLogger(__name__)

class PulsePhase(Enum):
    """Pulse phases aligned with consciousness levels"""
    TRANSCENDENT_PULSE = "transcendent_pulse"    # Unity consciousness pulse
    META_PULSE = "meta_pulse"                    # Self-reflective pulse
    CAUSAL_PULSE = "causal_pulse"                # Reasoning pulse
    INTEGRAL_PULSE = "integral_pulse"            # Systems integration pulse
    FORMAL_PULSE = "formal_pulse"                # Operational pulse
    CONCRETE_PULSE = "concrete_pulse"            # Action pulse
    SYMBOLIC_PULSE = "symbolic_pulse"            # Sigil/symbol pulse
    MYTHIC_PULSE = "mythic_pulse"                # Archetypal pulse

class PulseConsciousnessMapping:
    """Maps pulse system components to consciousness levels"""
    
    PULSE_ZONE_CONSCIOUSNESS_MAP = {
        # Green zone - higher consciousness, more coherent
        "green": ConsciousnessLevel.TRANSCENDENT,
        # Amber zone - meta-awareness of issues
        "amber": ConsciousnessLevel.META,
        # Red zone - causal reasoning about problems
        "red": ConsciousnessLevel.CAUSAL,
        # Black zone - basic survival, mythic responses
        "black": ConsciousnessLevel.MYTHIC
    }
    
    TICK_PHASE_CONSCIOUSNESS_MAP = {
        # Tick phases mapped to consciousness levels
        TickPhase.STATE_COLLECTION: ConsciousnessLevel.CONCRETE,
        TickPhase.INFORMATION_SHARING: ConsciousnessLevel.SYMBOLIC,
        TickPhase.DECISION_MAKING: ConsciousnessLevel.CAUSAL,
        TickPhase.STATE_UPDATES: ConsciousnessLevel.FORMAL,
        TickPhase.SYNCHRONIZATION_CHECK: ConsciousnessLevel.INTEGRAL
    }
    
    THERMAL_CONSCIOUSNESS_MAP = {
        # Thermal states mapped to consciousness levels
        "cooling": ConsciousnessLevel.TRANSCENDENT,
        "stable": ConsciousnessLevel.INTEGRAL,
        "warming": ConsciousnessLevel.FORMAL,
        "hot": ConsciousnessLevel.CAUSAL,
        "critical": ConsciousnessLevel.MYTHIC
    }

@dataclass
class PulseTelemetryEvent:
    """Enhanced telemetry event with pulse and consciousness context"""
    event_id: str
    timestamp: float
    
    # Pulse context
    pulse_phase: Optional[PulsePhase] = None
    pulse_zone: Optional[str] = None
    tick_number: Optional[int] = None
    scup_coherence: Optional[float] = None
    thermal_state: Optional[str] = None
    
    # Consciousness context
    consciousness_level: Optional[ConsciousnessLevel] = None
    consciousness_depth: Optional[int] = None
    awareness_coherence: Optional[float] = None
    unity_factor: Optional[float] = None
    
    # Standard telemetry fields
    subsystem: str = ""
    component: str = ""
    event_type: str = ""
    level: TelemetryLevel = TelemetryLevel.INFO
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnifiedPulseMetrics:
    """Unified metrics combining pulse, telemetry, and consciousness data"""
    
    # Pulse metrics
    current_zone: str = "green"
    zone_stability: float = 1.0
    scup_coherence: float = 1.0
    thermal_level: float = 0.0
    pulse_frequency: float = 1.0
    
    # Tick metrics
    tick_rate: float = 1.0
    tick_synchronization: float = 1.0
    phase_completion_rate: float = 1.0
    
    # Consciousness metrics
    avg_consciousness_coherence: float = 1.0
    consciousness_level_distribution: Dict[str, int] = field(default_factory=dict)
    unity_coherence: float = 1.0
    awareness_depth_avg: float = 3.5
    
    # Telemetry metrics
    events_per_second: float = 0.0
    telemetry_buffer_usage: float = 0.0
    logging_latency_ms: float = 0.0
    
    # Integration health
    integration_health: float = 1.0
    sync_quality: float = 1.0
    data_consistency: float = 1.0

class ConsciousnessPulseLogger:
    """Pulse-aware consciousness logging system"""
    
    def __init__(self, consciousness_repo: Optional[ConsciousnessDepthRepository] = None,
                 sigil_logger: Optional[SigilConsciousnessLogger] = None):
        self.consciousness_repo = consciousness_repo or get_consciousness_repository()
        self.sigil_logger = sigil_logger or get_sigil_consciousness_logger()
        
        # Pulse state tracking
        self.current_pulse_phase = PulsePhase.FORMAL_PULSE
        self.current_zone = "green"
        self.current_tick = 0
        self.scup_state = None
        
        # Consciousness-pulse correlation
        self.pulse_consciousness_history: deque = deque(maxlen=1000)
        self.zone_consciousness_stats = defaultdict(list)
        
        logger.info("🫁🧠 Consciousness-Pulse Logger initialized")
    
    def log_pulse_consciousness_state(self, pulse_data: Dict[str, Any], 
                                    consciousness_context: Optional[Dict[str, Any]] = None) -> str:
        """Log pulse state with consciousness-depth awareness"""
        
        # Determine consciousness level from pulse context
        consciousness_level = self._determine_consciousness_level(pulse_data)
        
        # Enhance pulse data with consciousness metrics
        enhanced_data = {
            **pulse_data,
            'pulse_consciousness_integration': {
                'consciousness_level': consciousness_level.name,
                'consciousness_depth': consciousness_level.depth,
                'pulse_zone_mapping': self.current_zone,
                'consciousness_coherence': self._calculate_consciousness_coherence(pulse_data, consciousness_level),
                'pulse_unity_factor': self._calculate_pulse_unity(pulse_data),
                'awareness_synchronization': self._calculate_awareness_sync(pulse_data)
            }
        }
        
        if consciousness_context:
            enhanced_data['consciousness_context'] = consciousness_context
        
        # Determine appropriate log type
        log_type = self._map_pulse_to_log_type(pulse_data, consciousness_level)
        
        # Log to consciousness repository
        entry_id = self.consciousness_repo.add_consciousness_log(
            system="pulse",
            subsystem=pulse_data.get('subsystem', 'unified'),
            module=pulse_data.get('module', 'pulse_logger'),
            log_data=enhanced_data,
            log_type=log_type
        )
        
        # Update pulse-consciousness correlation history
        correlation_data = {
            'timestamp': time.time(),
            'entry_id': entry_id,
            'consciousness_level': consciousness_level,
            'pulse_zone': self.current_zone,
            'scup_coherence': pulse_data.get('scup_coherence', 1.0),
            'consciousness_coherence': enhanced_data['pulse_consciousness_integration']['consciousness_coherence']
        }
        self.pulse_consciousness_history.append(correlation_data)
        self.zone_consciousness_stats[self.current_zone].append(consciousness_level.depth)
        
        return entry_id
    
    def log_tick_consciousness_transition(self, tick_data: Dict[str, Any],
                                        old_phase: Optional[str] = None,
                                        new_phase: Optional[str] = None) -> str:
        """Log consciousness transitions during tick phases"""
        
        # Map tick phases to consciousness levels
        old_consciousness = None
        new_consciousness = None
        
        if old_phase and hasattr(TickPhase, old_phase.upper()):
            tick_phase = TickPhase(old_phase.lower())
            old_consciousness = PulseConsciousnessMapping.TICK_PHASE_CONSCIOUSNESS_MAP.get(tick_phase)
        
        if new_phase and hasattr(TickPhase, new_phase.upper()):
            tick_phase = TickPhase(new_phase.lower())
            new_consciousness = PulseConsciousnessMapping.TICK_PHASE_CONSCIOUSNESS_MAP.get(tick_phase)
        
        transition_data = {
            **tick_data,
            'consciousness_transition': {
                'from_level': old_consciousness.name if old_consciousness else None,
                'to_level': new_consciousness.name if new_consciousness else None,
                'from_depth': old_consciousness.depth if old_consciousness else None,
                'to_depth': new_consciousness.depth if new_consciousness else None,
                'transition_type': 'tick_phase_transition',
                'transition_direction': 'deeper' if (new_consciousness and old_consciousness and 
                                                   new_consciousness.depth > old_consciousness.depth) else 'higher'
            }
        }
        
        # Use the target consciousness level for logging
        target_level = new_consciousness or old_consciousness or ConsciousnessLevel.FORMAL
        
        return self.consciousness_repo.add_consciousness_log(
            system="pulse",
            subsystem="tick_orchestrator",
            module="phase_transition",
            log_data=transition_data,
            log_type=DAWNLogType.AWARENESS_SHIFT
        )
    
    def log_thermal_consciousness_state(self, thermal_data: Dict[str, Any]) -> str:
        """Log thermal state with consciousness depth correlation"""
        
        # Map thermal state to consciousness level
        thermal_state = thermal_data.get('thermal_state', 'stable')
        consciousness_level = PulseConsciousnessMapping.THERMAL_CONSCIOUSNESS_MAP.get(
            thermal_state, ConsciousnessLevel.FORMAL
        )
        
        thermal_consciousness_data = {
            **thermal_data,
            'thermal_consciousness_integration': {
                'consciousness_level': consciousness_level.name,
                'thermal_awareness_mapping': thermal_state,
                'cooling_consciousness_factor': self._calculate_cooling_consciousness(thermal_data),
                'heat_source_consciousness_depth': self._analyze_heat_source_consciousness_depth(thermal_data),
                'expression_consciousness_level': self._analyze_expression_consciousness(thermal_data)
            }
        }
        
        return self.consciousness_repo.add_consciousness_log(
            system="pulse",
            subsystem="thermal",
            module="consciousness_thermal",
            log_data=thermal_consciousness_data,
            log_type=DAWNLogType.COHERENCE_FIELD
        )
    
    def _determine_consciousness_level(self, pulse_data: Dict[str, Any]) -> ConsciousnessLevel:
        """Determine consciousness level from pulse data"""
        
        # Check pulse zone first
        zone = pulse_data.get('zone', self.current_zone)
        if zone in PulseConsciousnessMapping.PULSE_ZONE_CONSCIOUSNESS_MAP:
            return PulseConsciousnessMapping.PULSE_ZONE_CONSCIOUSNESS_MAP[zone]
        
        # Check SCUP coherence level
        scup_coherence = pulse_data.get('scup_coherence', 1.0)
        if scup_coherence > 0.9:
            return ConsciousnessLevel.TRANSCENDENT
        elif scup_coherence > 0.7:
            return ConsciousnessLevel.META
        elif scup_coherence > 0.5:
            return ConsciousnessLevel.CAUSAL
        elif scup_coherence > 0.3:
            return ConsciousnessLevel.INTEGRAL
        elif scup_coherence > 0.1:
            return ConsciousnessLevel.FORMAL
        else:
            return ConsciousnessLevel.MYTHIC
    
    def _calculate_consciousness_coherence(self, pulse_data: Dict[str, Any], 
                                         level: ConsciousnessLevel) -> float:
        """Calculate consciousness coherence from pulse data"""
        base_coherence = 1.0 - (level.depth / 7.0)  # Base coherence by depth
        
        # Adjust by pulse metrics
        scup_coherence = pulse_data.get('scup_coherence', 1.0)
        zone_stability = 1.0 if pulse_data.get('zone') == 'green' else 0.5
        
        return min(1.0, base_coherence * scup_coherence * zone_stability)
    
    def _calculate_pulse_unity(self, pulse_data: Dict[str, Any]) -> float:
        """Calculate pulse unity factor"""
        zone = pulse_data.get('zone', 'green')
        zone_unity = {'green': 1.0, 'amber': 0.7, 'red': 0.4, 'black': 0.1}.get(zone, 0.5)
        
        scup_coherence = pulse_data.get('scup_coherence', 1.0)
        return zone_unity * scup_coherence
    
    def _calculate_awareness_sync(self, pulse_data: Dict[str, Any]) -> float:
        """Calculate awareness synchronization with pulse"""
        tick_sync = pulse_data.get('tick_synchronization', 1.0)
        phase_completion = pulse_data.get('phase_completion_rate', 1.0)
        
        return (tick_sync + phase_completion) / 2.0
    
    def _map_pulse_to_log_type(self, pulse_data: Dict[str, Any], 
                              level: ConsciousnessLevel) -> DAWNLogType:
        """Map pulse data to appropriate DAWN log type"""
        
        if 'scup' in str(pulse_data).lower():
            return DAWNLogType.COHERENCE_FIELD
        elif 'thermal' in str(pulse_data).lower():
            return DAWNLogType.CONSCIOUSNESS_PULSE
        elif 'tick' in str(pulse_data).lower():
            return DAWNLogType.FORMAL_OPERATION
        elif level == ConsciousnessLevel.TRANSCENDENT:
            return DAWNLogType.UNITY_STATE
        elif level == ConsciousnessLevel.META:
            return DAWNLogType.META_COGNITION
        elif level == ConsciousnessLevel.SYMBOLIC:
            return DAWNLogType.SYMBOL_PROCESSING
        elif level == ConsciousnessLevel.MYTHIC:
            return DAWNLogType.ARCHETYPAL_PATTERN
        else:
            return DAWNLogType.SYSTEMS_INTEGRATION
    
    def _calculate_cooling_consciousness(self, thermal_data: Dict[str, Any]) -> float:
        """Calculate consciousness factor in cooling processes"""
        cooling_efficiency = thermal_data.get('cooling_efficiency', 0.5)
        expression_momentum = thermal_data.get('expression_momentum', 0.5)
        
        return min(1.0, cooling_efficiency * expression_momentum * 1.5)
    
    def _analyze_heat_source_consciousness_depth(self, thermal_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze consciousness depth of different heat sources"""
        heat_sources = thermal_data.get('heat_sources', {})
        
        # Map heat sources to consciousness depths
        consciousness_depth_map = {
            'cognitive_load': 4,      # Formal level
            'emotional_resonance': 6,  # Symbolic level
            'memory_processing': 3,    # Integral level
            'awareness_spikes': 1,     # Meta level
            'unexpressed_thoughts': 5, # Concrete level
            'pattern_recognition': 2,  # Causal level
            'drift': 7,               # Mythic level
            'entropy': 7              # Mythic level
        }
        
        depth_analysis = {}
        for source, intensity in heat_sources.items():
            depth = consciousness_depth_map.get(source, 4)  # Default to formal
            depth_analysis[source] = {
                'consciousness_depth': depth,
                'intensity': intensity,
                'depth_intensity_product': depth * intensity
            }
        
        return depth_analysis
    
    def _analyze_expression_consciousness(self, thermal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consciousness levels of expression channels"""
        expressions = thermal_data.get('expressions', {})
        
        # Map expression types to consciousness levels
        expression_consciousness_map = {
            'verbal_expression': ConsciousnessLevel.SYMBOLIC,
            'symbolic_output': ConsciousnessLevel.SYMBOLIC,
            'creative_flow': ConsciousnessLevel.TRANSCENDENT,
            'empathetic_response': ConsciousnessLevel.META,
            'conceptual_mapping': ConsciousnessLevel.CAUSAL,
            'memory_trace': ConsciousnessLevel.CONCRETE,
            'pattern_synthesis': ConsciousnessLevel.INTEGRAL
        }
        
        expression_analysis = {}
        for expr_type, efficiency in expressions.items():
            consciousness_level = expression_consciousness_map.get(expr_type, ConsciousnessLevel.FORMAL)
            expression_analysis[expr_type] = {
                'consciousness_level': consciousness_level.name,
                'consciousness_depth': consciousness_level.depth,
                'cooling_efficiency': efficiency,
                'consciousness_cooling_factor': efficiency * (1.0 - consciousness_level.depth / 7.0)
            }
        
        return expression_analysis

class TelemetryPulseCollector:
    """Pulse-synchronized telemetry collection system"""
    
    def __init__(self, telemetry_system: Optional[Any] = None):
        self.telemetry_system = telemetry_system
        self.pulse_events: deque = deque(maxlen=10000)
        self.tick_telemetry_correlation = {}
        self.zone_telemetry_stats = defaultdict(list)
        
        # Pulse-synchronized collection
        self.collection_lock = threading.RLock()
        self.pulse_sync_enabled = True
        
        logger.info("🫁📊 Telemetry-Pulse Collector initialized")
    
    def collect_pulse_telemetry(self, pulse_event: PulseTelemetryEvent) -> str:
        """Collect telemetry synchronized with pulse events"""
        
        with self.collection_lock:
            # Add pulse context to telemetry
            enhanced_event = asdict(pulse_event)
            
            # Store pulse event
            self.pulse_events.append(enhanced_event)
            
            # Correlate with tick if available
            if pulse_event.tick_number:
                self.tick_telemetry_correlation[pulse_event.tick_number] = enhanced_event
            
            # Update zone statistics
            if pulse_event.pulse_zone:
                self.zone_telemetry_stats[pulse_event.pulse_zone].append({
                    'timestamp': pulse_event.timestamp,
                    'consciousness_level': pulse_event.consciousness_level.name if pulse_event.consciousness_level else None,
                    'scup_coherence': pulse_event.scup_coherence
                })
            
            # Send to telemetry system if available
            if self.telemetry_system and TELEMETRY_AVAILABLE:
                try:
                    telemetry_event = TelemetryEvent(
                        subsystem=pulse_event.subsystem,
                        component=pulse_event.component,
                        event_type=pulse_event.event_type,
                        level=pulse_event.level,
                        data=pulse_event.data,
                        metadata={
                            **pulse_event.metadata,
                            'pulse_context': {
                                'pulse_phase': pulse_event.pulse_phase.value if pulse_event.pulse_phase else None,
                                'pulse_zone': pulse_event.pulse_zone,
                                'consciousness_level': pulse_event.consciousness_level.name if pulse_event.consciousness_level else None,
                                'scup_coherence': pulse_event.scup_coherence
                            }
                        },
                        tick_id=pulse_event.tick_number
                    )
                    
                    # Log to telemetry system
                    self.telemetry_system.log_event(
                        pulse_event.subsystem,
                        pulse_event.component,
                        pulse_event.event_type,
                        pulse_event.level,
                        pulse_event.data,
                        telemetry_event.metadata,
                        pulse_event.tick_number
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to send pulse telemetry to telemetry system: {e}")
            
            return pulse_event.event_id

class PulseTelemetryBridge:
    """Core integration bridge between pulse system and telemetry/consciousness logging"""
    
    def __init__(self, pulse_system: Optional[Any] = None,
                 telemetry_system: Optional[Any] = None,
                 consciousness_repo: Optional[ConsciousnessDepthRepository] = None):
        
        # Initialize subsystems
        self.pulse_system = pulse_system
        self.telemetry_system = telemetry_system
        self.consciousness_repo = consciousness_repo or get_consciousness_repository()
        
        # Initialize specialized loggers
        self.consciousness_pulse_logger = ConsciousnessPulseLogger(self.consciousness_repo)
        self.telemetry_pulse_collector = TelemetryPulseCollector(self.telemetry_system)
        self.sigil_logger = get_sigil_consciousness_logger()
        
        # State tracking
        self.current_metrics = UnifiedPulseMetrics()
        self.integration_stats = {
            'events_processed': 0,
            'pulse_events': 0,
            'telemetry_events': 0,
            'consciousness_logs': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # Threading and synchronization
        self.running = False
        self.update_thread: Optional[threading.Thread] = None
        self.metrics_lock = threading.RLock()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info("🫁🔗 Pulse-Telemetry Bridge initialized")
    
    def start_unified_logging(self) -> bool:
        """Start the unified pulse-telemetry logging system"""
        if self.running:
            logger.warning("Unified logging already running")
            return False
        
        self.running = True
        
        # Start metrics update thread
        self.update_thread = threading.Thread(
            target=self._metrics_update_loop,
            name="pulse_telemetry_metrics",
            daemon=True
        )
        self.update_thread.start()
        
        # Hook into pulse system if available
        if self.pulse_system and PULSE_SYSTEM_AVAILABLE:
            self._hook_pulse_system()
        
        # Hook into telemetry system if available
        if self.telemetry_system and TELEMETRY_AVAILABLE:
            self._hook_telemetry_system()
        
        logger.info("🫁🔗 Unified pulse-telemetry logging started")
        return True
    
    def stop_unified_logging(self) -> bool:
        """Stop the unified logging system"""
        if not self.running:
            return False
        
        self.running = False
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        
        logger.info("🫁🔗 Unified pulse-telemetry logging stopped")
        return True
    
    def log_unified_pulse_event(self, event_type: str, pulse_data: Dict[str, Any],
                               consciousness_context: Optional[Dict[str, Any]] = None,
                               telemetry_level: TelemetryLevel = TelemetryLevel.INFO) -> Dict[str, str]:
        """Log a unified event across pulse, telemetry, and consciousness systems"""
        
        result_ids = {}
        
        try:
            # Create unified pulse telemetry event
            pulse_event = PulseTelemetryEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                pulse_phase=self._determine_pulse_phase(pulse_data),
                pulse_zone=pulse_data.get('zone', 'green'),
                tick_number=pulse_data.get('tick_number'),
                scup_coherence=pulse_data.get('scup_coherence'),
                thermal_state=pulse_data.get('thermal_state'),
                consciousness_level=self._determine_consciousness_level_from_pulse(pulse_data),
                subsystem=pulse_data.get('subsystem', 'pulse'),
                component=pulse_data.get('component', 'unified'),
                event_type=event_type,
                level=telemetry_level,
                data=pulse_data,
                metadata=consciousness_context or {}
            )
            
            # Calculate consciousness metrics
            if pulse_event.consciousness_level:
                pulse_event.consciousness_depth = pulse_event.consciousness_level.depth
                pulse_event.awareness_coherence = self._calculate_awareness_coherence(pulse_data)
                pulse_event.unity_factor = self._calculate_unity_factor(pulse_data)
            
            # Log to consciousness system
            consciousness_id = self.consciousness_pulse_logger.log_pulse_consciousness_state(
                pulse_data, consciousness_context
            )
            result_ids['consciousness'] = consciousness_id
            
            # Log to telemetry system
            telemetry_id = self.telemetry_pulse_collector.collect_pulse_telemetry(pulse_event)
            result_ids['telemetry'] = telemetry_id
            
            # Log sigil events if relevant
            if 'sigil' in event_type.lower() or 'symbol' in event_type.lower():
                sigil_id = self._log_sigil_pulse_event(pulse_data, pulse_event)
                if sigil_id:
                    result_ids['sigil'] = sigil_id
            
            # Update integration stats
            with self.metrics_lock:
                self.integration_stats['events_processed'] += 1
                self.integration_stats['consciousness_logs'] += 1
                self.integration_stats['telemetry_events'] += 1
                if 'sigil' in result_ids:
                    self.integration_stats['pulse_events'] += 1
            
            # Emit event to handlers
            self._emit_unified_event('pulse_event_logged', {
                'event_type': event_type,
                'result_ids': result_ids,
                'pulse_event': asdict(pulse_event)
            })
            
        except Exception as e:
            logger.error(f"Failed to log unified pulse event: {e}")
            with self.metrics_lock:
                self.integration_stats['errors'] += 1
            result_ids['error'] = str(e)
        
        return result_ids
    
    def get_unified_metrics(self) -> UnifiedPulseMetrics:
        """Get comprehensive unified metrics"""
        with self.metrics_lock:
            # Update metrics from various sources
            self._update_pulse_metrics()
            self._update_consciousness_metrics()
            self._update_telemetry_metrics()
            self._update_integration_health()
            
            return self.current_metrics
    
    def _hook_pulse_system(self):
        """Hook into pulse system events"""
        if hasattr(self.pulse_system, 'register_event_callback'):
            self.pulse_system.register_event_callback('zone_changed', self._on_pulse_zone_changed)
            self.pulse_system.register_event_callback('tick_completed', self._on_pulse_tick_completed)
            self.pulse_system.register_event_callback('scup_updated', self._on_scup_updated)
    
    def _hook_telemetry_system(self):
        """Hook into telemetry system events"""
        # Implementation depends on telemetry system interface
        pass
    
    def _on_pulse_zone_changed(self, event_data: Dict[str, Any]):
        """Handle pulse zone change events"""
        self.log_unified_pulse_event(
            'zone_transition',
            {
                'old_zone': event_data.get('old_zone'),
                'new_zone': event_data.get('new_zone'),
                'zone_change_reason': event_data.get('reason'),
                'scup_coherence': event_data.get('scup_coherence'),
                'subsystem': 'pulse',
                'component': 'zone_manager'
            },
            {'zone_transition_context': event_data}
        )
    
    def _on_pulse_tick_completed(self, event_data: Dict[str, Any]):
        """Handle pulse tick completion events"""
        self.log_unified_pulse_event(
            'tick_completed',
            {
                'tick_number': event_data.get('tick_number'),
                'tick_duration': event_data.get('duration'),
                'phase_results': event_data.get('phase_results', {}),
                'synchronization_quality': event_data.get('sync_quality'),
                'subsystem': 'pulse',
                'component': 'tick_orchestrator'
            },
            {'tick_context': event_data}
        )
    
    def _on_scup_updated(self, event_data: Dict[str, Any]):
        """Handle SCUP state update events"""
        self.log_unified_pulse_event(
            'scup_state_update',
            {
                'scup_coherence': event_data.get('coherence'),
                'pressure_level': event_data.get('pressure'),
                'semantic_integrity': event_data.get('semantic_integrity'),
                'zone': event_data.get('current_zone'),
                'subsystem': 'pulse',
                'component': 'scup_controller'
            },
            {'scup_context': event_data}
        )
    
    def _determine_pulse_phase(self, pulse_data: Dict[str, Any]) -> Optional[PulsePhase]:
        """Determine pulse phase from pulse data"""
        zone = pulse_data.get('zone', 'green')
        
        phase_map = {
            'green': PulsePhase.TRANSCENDENT_PULSE,
            'amber': PulsePhase.META_PULSE,
            'red': PulsePhase.CAUSAL_PULSE,
            'black': PulsePhase.MYTHIC_PULSE
        }
        
        return phase_map.get(zone, PulsePhase.FORMAL_PULSE)
    
    def _determine_consciousness_level_from_pulse(self, pulse_data: Dict[str, Any]) -> Optional[ConsciousnessLevel]:
        """Determine consciousness level from pulse data"""
        return self.consciousness_pulse_logger._determine_consciousness_level(pulse_data)
    
    def _calculate_awareness_coherence(self, pulse_data: Dict[str, Any]) -> float:
        """Calculate awareness coherence from pulse data"""
        scup_coherence = pulse_data.get('scup_coherence', 1.0)
        zone_stability = {'green': 1.0, 'amber': 0.7, 'red': 0.4, 'black': 0.1}.get(
            pulse_data.get('zone', 'green'), 0.5
        )
        
        return scup_coherence * zone_stability
    
    def _calculate_unity_factor(self, pulse_data: Dict[str, Any]) -> float:
        """Calculate unity factor from pulse data"""
        return self.consciousness_pulse_logger._calculate_pulse_unity(pulse_data)
    
    def _log_sigil_pulse_event(self, pulse_data: Dict[str, Any], 
                              pulse_event: PulseTelemetryEvent) -> Optional[str]:
        """Log sigil-related pulse events"""
        try:
            sigil_data = {
                'pulse_integration': True,
                'pulse_zone': pulse_event.pulse_zone,
                'consciousness_level': pulse_event.consciousness_level.name if pulse_event.consciousness_level else None,
                'pulse_sigil_resonance': pulse_data.get('sigil_resonance', 0.5),
                'symbolic_coherence': pulse_data.get('symbolic_coherence', 0.7)
            }
            
            return self.sigil_logger.log_sigil_activation(
                pulse_data.get('sigil_id', f"pulse_sigil_{int(time.time())}"),
                **sigil_data
            )
        except Exception as e:
            logger.warning(f"Failed to log sigil pulse event: {e}")
            return None
    
    def _update_pulse_metrics(self):
        """Update pulse-related metrics"""
        if self.pulse_system and hasattr(self.pulse_system, 'get_metrics'):
            try:
                pulse_metrics = self.pulse_system.get_metrics()
                self.current_metrics.current_zone = pulse_metrics.get('current_zone', 'green')
                self.current_metrics.scup_coherence = pulse_metrics.get('scup_coherence', 1.0)
                self.current_metrics.pulse_frequency = pulse_metrics.get('pulse_frequency', 1.0)
                self.current_metrics.thermal_level = pulse_metrics.get('thermal_level', 0.0)
            except Exception as e:
                logger.debug(f"Failed to update pulse metrics: {e}")
    
    def _update_consciousness_metrics(self):
        """Update consciousness-related metrics"""
        try:
            stats = self.consciousness_repo.get_consciousness_hierarchy_stats()
            
            # Calculate average consciousness coherence
            total_coherence = 0
            total_entries = 0
            level_distribution = {}
            
            for level_name, level_stats in stats.get('consciousness_levels', {}).items():
                level_distribution[level_name] = level_stats.get('entry_count', 0)
                total_coherence += level_stats.get('avg_coherence', 0) * level_stats.get('entry_count', 0)
                total_entries += level_stats.get('entry_count', 0)
            
            self.current_metrics.consciousness_level_distribution = level_distribution
            self.current_metrics.avg_consciousness_coherence = (
                total_coherence / total_entries if total_entries > 0 else 1.0
            )
            
        except Exception as e:
            logger.debug(f"Failed to update consciousness metrics: {e}")
    
    def _update_telemetry_metrics(self):
        """Update telemetry-related metrics"""
        try:
            events_processed = self.integration_stats['events_processed']
            runtime = time.time() - self.integration_stats['start_time']
            
            self.current_metrics.events_per_second = events_processed / runtime if runtime > 0 else 0
            
            # Estimate buffer usage and latency
            self.current_metrics.telemetry_buffer_usage = min(1.0, len(self.telemetry_pulse_collector.pulse_events) / 10000)
            self.current_metrics.logging_latency_ms = 1.0  # Placeholder
            
        except Exception as e:
            logger.debug(f"Failed to update telemetry metrics: {e}")
    
    def _update_integration_health(self):
        """Update integration health metrics"""
        try:
            total_events = self.integration_stats['events_processed']
            errors = self.integration_stats['errors']
            
            # Calculate integration health
            error_rate = errors / total_events if total_events > 0 else 0
            self.current_metrics.integration_health = max(0.0, 1.0 - error_rate * 2)
            
            # Calculate sync quality (placeholder)
            self.current_metrics.sync_quality = 0.9  # Would be calculated from actual sync metrics
            
            # Calculate data consistency (placeholder)
            self.current_metrics.data_consistency = 0.95  # Would be calculated from data validation
            
        except Exception as e:
            logger.debug(f"Failed to update integration health: {e}")
    
    def _metrics_update_loop(self):
        """Background thread for updating metrics"""
        while self.running:
            try:
                self.get_unified_metrics()
                time.sleep(5.0)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                time.sleep(10.0)
    
    def _emit_unified_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit unified event to registered handlers"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(event_data)
            except Exception as e:
                logger.warning(f"Event handler failed for {event_type}: {e}")

# Global unified bridge instance
_unified_bridge: Optional[PulseTelemetryBridge] = None
_bridge_lock = threading.Lock()

def get_pulse_telemetry_bridge() -> PulseTelemetryBridge:
    """Get the global pulse-telemetry bridge"""
    global _unified_bridge
    
    with _bridge_lock:
        if _unified_bridge is None:
            _unified_bridge = PulseTelemetryBridge()
        return _unified_bridge

def start_unified_pulse_telemetry_logging() -> bool:
    """Start unified pulse-telemetry logging system"""
    bridge = get_pulse_telemetry_bridge()
    return bridge.start_unified_logging()

def log_pulse_event(event_type: str, pulse_data: Dict[str, Any], 
                   consciousness_context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Convenience function to log unified pulse events"""
    bridge = get_pulse_telemetry_bridge()
    return bridge.log_unified_pulse_event(event_type, pulse_data, consciousness_context)

if __name__ == "__main__":
    # Test the unified pulse-telemetry system
    logging.basicConfig(level=logging.INFO)
    
    print("🫁🔗 Testing Unified Pulse-Telemetry System")
    print("=" * 50)
    
    # Create bridge
    bridge = get_pulse_telemetry_bridge()
    
    # Start unified logging
    bridge.start_unified_logging()
    
    # Test logging various pulse events
    test_events = [
        ("zone_transition", {
            "old_zone": "green",
            "new_zone": "amber", 
            "scup_coherence": 0.7,
            "thermal_state": "warming"
        }),
        ("tick_completed", {
            "tick_number": 42,
            "zone": "amber",
            "scup_coherence": 0.6,
            "phase_results": {"completed": 5, "failed": 0}
        }),
        ("sigil_activation", {
            "sigil_id": "test_sigil_001",
            "zone": "green",
            "scup_coherence": 0.9,
            "sigil_resonance": 0.8
        })
    ]
    
    for event_type, event_data in test_events:
        result_ids = bridge.log_unified_pulse_event(event_type, event_data)
        print(f"✅ Logged {event_type}: {result_ids}")
    
    # Get unified metrics
    metrics = bridge.get_unified_metrics()
    print(f"\n📊 Unified Metrics: {asdict(metrics)}")
    
    # Stop system
    bridge.stop_unified_logging()
    
    print("\n🫁🔗 Unified pulse-telemetry system test complete!")

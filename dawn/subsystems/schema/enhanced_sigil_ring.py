#!/usr/bin/env python3
"""
DAWN Enhanced Sigil Ring - Complete Symbolic Execution Environment
=================================================================

Implementation of the complete DAWN Sigil Ring as documented.
Provides casting circle mechanics, containment boundary enforcement,
stack ordering, and visual representation for the symbolic execution layer.

Based on DAWN's documented Sigil Ring architecture.
"""

import time
import math
import threading
import logging
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json

from core.schema_anomaly_logger import log_anomaly, AnomalySeverity
from schema.registry import registry
from schema.sigil_glyph_codex import sigil_glyph_codex, SigilGlyph, GlyphCategory, SigilHouse
from schema.sigil_network import SigilNetwork, SigilInvocation, sigil_network
from rhizome.propagation import emit_signal, SignalType
from utils.metrics_collector import metrics

logger = logging.getLogger(__name__)

class RingState(Enum):
    """States of the Sigil Ring"""
    DORMANT = "dormant"         # Ring inactive
    FORMING = "forming"         # Ring being established per tick
    ACTIVE = "active"           # Normal operation with casting circle
    STACKING = "stacking"       # Processing stacked invocations
    RESONATING = "resonating"   # High coherence operation
    OVERLOADED = "overloaded"   # Too many operations per tick
    CONTAINMENT_BREACH = "containment_breach"  # Safety breach
    EMERGENCY_SEAL = "emergency_seal"  # Emergency containment

class InvokerPriority(Enum):
    """Priority levels for stack ordering as documented"""
    CORE = 1            # Core system operations (highest priority)
    TRACER = 2          # Tracer feedback operations
    OPERATOR = 3        # Human operator commands (lowest priority)

class ContainmentLevel(Enum):
    """Levels of containment for sigil operations"""
    OPEN = "open"           # No containment
    BASIC = "basic"         # Basic safety checks
    SECURED = "secured"     # Enhanced containment
    SEALED = "sealed"       # Maximum containment
    QUARANTINE = "quarantine"  # Isolated execution

@dataclass
class ContainmentBoundary:
    """Containment boundary for the ring"""
    level: ContainmentLevel
    max_invocations_per_tick: int = 10
    max_stack_depth: int = 3
    max_execution_time: float = 5.0
    breach_threshold: int = 3
    active_breaches: int = 0
    
    def is_breach_critical(self) -> bool:
        """Check if breach count is critical"""
        return self.active_breaches >= self.breach_threshold

@dataclass
class CastingCircle:
    """The casting circle formed each tick"""
    tick_id: int
    formation_time: float
    active_invocations: List[SigilInvocation] = field(default_factory=list)
    house_nodes: Dict[SigilHouse, List[str]] = field(default_factory=lambda: defaultdict(list))
    stack_order: List[str] = field(default_factory=list)  # Ordered by priority
    containment_boundary: Optional[ContainmentBoundary] = None
    circle_coherence: float = 1.0
    
    def add_invocation(self, invocation: SigilInvocation):
        """Add invocation to the circle"""
        self.active_invocations.append(invocation)
        if invocation.house:
            self.house_nodes[invocation.house].append(invocation.sigil_symbol)
    
    def get_invocation_count(self) -> int:
        """Get total invocation count"""
        return len(self.active_invocations)
    
    def get_house_activity(self) -> Dict[SigilHouse, int]:
        """Get activity count per house"""
        return {house: len(invocations) for house, invocations in self.house_nodes.items()}

@dataclass
class SigilStack:
    """A stack of sigil invocations with priority ordering"""
    stack_id: str
    invocations: List[SigilInvocation] = field(default_factory=list)
    invoker_priority: InvokerPriority = InvokerPriority.OPERATOR
    execution_priority: int = 5
    containment_level: ContainmentLevel = ContainmentLevel.BASIC
    creation_time: float = field(default_factory=time.time)
    execution_start: Optional[float] = None
    execution_end: Optional[float] = None
    
    def add_invocation(self, invocation: SigilInvocation):
        """Add invocation to stack"""
        invocation.stack_position = len(self.invocations)
        self.invocations.append(invocation)
        
        # Update execution priority based on glyph priority
        glyph_priority = sigil_glyph_codex.get_execution_priority([invocation.sigil_symbol])
        self.execution_priority = max(self.execution_priority, glyph_priority)
    
    def get_execution_time(self) -> Optional[float]:
        """Get total execution time if completed"""
        if self.execution_start and self.execution_end:
            return self.execution_end - self.execution_start
        return None
    
    def get_layered_meaning(self) -> str:
        """Get the combined meaning of stacked glyphs"""
        symbols = [inv.sigil_symbol for inv in self.invocations]
        return sigil_glyph_codex.resolve_layered_meaning(symbols)

@dataclass
class RingMetrics:
    """Comprehensive metrics for ring performance"""
    total_circles_formed: int = 0
    total_stacks_processed: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    containment_breaches: int = 0
    emergency_seals_triggered: int = 0
    average_execution_time: float = 0.0
    peak_invocations_per_tick: int = 0
    total_invocations_processed: int = 0
    ring_overloads: int = 0
    
    def get_success_rate(self) -> float:
        """Get success rate percentage"""
        total = self.successful_executions + self.failed_executions
        return (self.successful_executions / total * 100) if total > 0 else 0.0
    
    def update_execution_time(self, execution_time: float):
        """Update average execution time"""
        total_executions = self.successful_executions + self.failed_executions
        if total_executions > 0:
            self.average_execution_time = (
                (self.average_execution_time * (total_executions - 1) + execution_time) / total_executions
            )

class RingTelemetry:
    """Comprehensive telemetry for the Sigil Ring"""
    
    def __init__(self):
        self.ring_log = deque(maxlen=1000)  # Every invocation, house route, invoker, tick ID
        self.stack_traces = deque(maxlen=500)  # Sequence of glyphs resolved each tick
        self.conflict_alerts = deque(maxlen=200)  # When Houses clash inside ring
        self.circle_formations = deque(maxlen=100)  # Circle formation events
        
    def log_invocation(self, invocation: SigilInvocation, house_route: SigilHouse, tick_id: int):
        """Log individual invocation"""
        self.ring_log.append({
            "timestamp": time.time(),
            "tick_id": tick_id,
            "sigil": invocation.sigil_symbol,
            "house_route": house_route.value,
            "invoker": invocation.invoker,
            "priority": invocation.priority
        })
    
    def log_stack_trace(self, stack: SigilStack, tick_id: int):
        """Log stack execution trace"""
        self.stack_traces.append({
            "timestamp": time.time(),
            "tick_id": tick_id,
            "stack_id": stack.stack_id,
            "glyph_sequence": [inv.sigil_symbol for inv in stack.invocations],
            "layered_meaning": stack.get_layered_meaning(),
            "execution_time": stack.get_execution_time(),
            "success": stack.execution_end is not None
        })
    
    def log_conflict_alert(self, conflict_type: str, details: str, tick_id: int):
        """Log house conflicts"""
        self.conflict_alerts.append({
            "timestamp": time.time(),
            "tick_id": tick_id,
            "conflict_type": conflict_type,
            "details": details
        })
    
    def log_circle_formation(self, circle: CastingCircle):
        """Log casting circle formation"""
        self.circle_formations.append({
            "timestamp": time.time(),
            "tick_id": circle.tick_id,
            "invocation_count": circle.get_invocation_count(),
            "house_activity": circle.get_house_activity(),
            "coherence": circle.circle_coherence
        })

class EnhancedSigilRing:
    """
    Enhanced Sigil Ring - Complete Symbolic Execution Environment
    
    Implements the full documented Sigil Ring with:
    - Casting circle formation per tick
    - Containment boundary enforcement
    - Stack ordering by priority (Core > Tracer > Operator)
    - Ring overload protection with spillover management
    - Visual representation support
    """
    
    def __init__(self, network: Optional[SigilNetwork] = None):
        self.network = network or sigil_network
        self.ring_state = RingState.DORMANT
        self.ring_id = str(uuid.uuid4())[:8]
        
        # Core ring components
        self.current_circle: Optional[CastingCircle] = None
        self.active_stacks: Dict[str, SigilStack] = {}
        self.execution_queue: deque = deque()
        self.spillover_queue: deque = deque()  # For overload management
        
        # Containment and safety
        self.containment_boundary = ContainmentBoundary(ContainmentLevel.BASIC)
        self.emergency_seal_active = False
        
        # Monitoring and telemetry
        self.metrics = RingMetrics()
        self.telemetry = RingTelemetry()
        
        # Thread safety
        self.ring_lock = threading.RLock()
        
        # Current tick tracking
        self.current_tick = 0
        
        # Register with schema registry
        self._register()
        
        logger.info(f"💍 Enhanced Sigil Ring {self.ring_id} initialized")
        logger.info(f"   Containment level: {self.containment_boundary.level.value}")
        logger.info(f"   Max invocations per tick: {self.containment_boundary.max_invocations_per_tick}")
    
    def _register(self):
        """Register with schema registry"""
        registry.register(
            component_id=f"schema.enhanced_sigil_ring.{self.ring_id}",
            name="Enhanced Sigil Ring",
            component_type="SYMBOLIC_EXECUTION_ENVIRONMENT",
            instance=self,
            capabilities=[
                "casting_circle_formation",
                "containment_boundary_enforcement", 
                "priority_stack_ordering",
                "ring_overload_protection",
                "symbolic_execution",
                "visual_representation"
            ],
            version="2.0.0"
        )
    
    def activate_ring(self):
        """Activate the Sigil Ring for operation"""
        with self.ring_lock:
            if self.ring_state == RingState.DORMANT:
                self.ring_state = RingState.FORMING
                
                if self._validate_ring_integrity():
                    self.ring_state = RingState.ACTIVE
                    
                    emit_signal(
                        SignalType.CONSCIOUSNESS,
                        "enhanced_sigil_ring",
                        {
                            "event": "ring_activated",
                            "ring_id": self.ring_id,
                            "containment_level": self.containment_boundary.level.value
                        }
                    )
                    
                    logger.info(f"💍 Enhanced Ring {self.ring_id} activated for symbolic execution")
                else:
                    self.ring_state = RingState.DORMANT
                    logger.error(f"💍 Ring {self.ring_id} failed integrity validation")
    
    def _validate_ring_integrity(self) -> bool:
        """Validate ring integrity before activation"""
        # Check network connection
        if not self.network:
            return False
        
        # Check glyph codex availability
        if not sigil_glyph_codex:
            return False
        
        # Check containment boundary
        if self.containment_boundary.is_breach_critical():
            return False
        
        return True
    
    def form_casting_circle(self, tick_id: int) -> CastingCircle:
        """Form casting circle for the current tick"""
        with self.ring_lock:
            if self.ring_state not in [RingState.ACTIVE, RingState.RESONATING]:
                raise RuntimeError(f"Cannot form casting circle in state: {self.ring_state}")
            
            self.ring_state = RingState.FORMING
            
            # Create new casting circle
            circle = CastingCircle(
                tick_id=tick_id,
                formation_time=time.time(),
                containment_boundary=self.containment_boundary
            )
            
            # Process spillover from previous tick
            while self.spillover_queue and len(circle.active_invocations) < self.containment_boundary.max_invocations_per_tick:
                spillover_stack = self.spillover_queue.popleft()
                for invocation in spillover_stack.invocations:
                    circle.add_invocation(invocation)
            
            self.current_circle = circle
            self.current_tick = tick_id
            self.ring_state = RingState.ACTIVE
            
            # Update metrics
            self.metrics.total_circles_formed += 1
            
            # Log circle formation
            self.telemetry.log_circle_formation(circle)
            
            logger.debug(f"💍 Casting circle formed for tick {tick_id}")
            
            return circle
    
    def cast_sigil_stack(self, invocations: List[SigilInvocation], 
                        invoker: str = "ring",
                        invoker_priority: InvokerPriority = InvokerPriority.OPERATOR,
                        containment: Optional[ContainmentLevel] = None) -> Dict[str, Any]:
        """
        Cast a stack of sigil invocations in the ring with full documentation compliance
        """
        if self.ring_state not in [RingState.ACTIVE, RingState.RESONATING]:
            return {
                "success": False,
                "error": f"ring_not_active: {self.ring_state.value}",
                "ring_id": self.ring_id
            }
        
        if self.emergency_seal_active:
            return {
                "success": False,
                "error": "emergency_seal_active",
                "ring_id": self.ring_id
            }
        
        # Validate glyph layering
        glyph_symbols = [inv.sigil_symbol for inv in invocations]
        is_valid, violations = sigil_glyph_codex.validate_layering(glyph_symbols)
        if not is_valid:
            return {
                "success": False,
                "error": "invalid_glyph_layering",
                "violations": violations,
                "symbols": glyph_symbols
            }
        
        # Create stack with priority
        stack = SigilStack(
            stack_id=str(uuid.uuid4())[:12],
            invoker_priority=invoker_priority,
            containment_level=containment or self._determine_containment_level(invocations)
        )
        
        # Add invocations to stack
        for invocation in invocations:
            stack.add_invocation(invocation)
        
        # Check for ring overload
        if not self.current_circle:
            self.form_casting_circle(self.current_tick)
        
        current_load = self.current_circle.get_invocation_count()
        if current_load + len(invocations) > self.containment_boundary.max_invocations_per_tick:
            # Handle ring overload with spillover
            return self._handle_ring_overload(stack)
        
        # Add to current circle
        for invocation in invocations:
            self.current_circle.add_invocation(invocation)
            
            # Log each invocation
            self.telemetry.log_invocation(invocation, invocation.house, self.current_tick)
        
        # Execute stack with containment
        return self._execute_stack_with_containment(stack)
    
    def _handle_ring_overload(self, stack: SigilStack) -> Dict[str, Any]:
        """Handle ring overload with spillover management"""
        self.ring_state = RingState.OVERLOADED
        self.metrics.ring_overloads += 1
        
        # Add to spillover queue for next tick
        self.spillover_queue.append(stack)
        
        log_anomaly(
            "RING_OVERLOAD",
            f"Ring overloaded, stack {stack.stack_id} moved to spillover",
            AnomalySeverity.WARNING
        )
        
        return {
            "success": False,
            "error": "ring_overloaded",
            "stack_id": stack.stack_id,
            "spillover_position": len(self.spillover_queue),
            "current_load": self.current_circle.get_invocation_count() if self.current_circle else 0,
            "max_capacity": self.containment_boundary.max_invocations_per_tick
        }
    
    def _execute_stack_with_containment(self, stack: SigilStack) -> Dict[str, Any]:
        """Execute stack with full containment enforcement"""
        stack.execution_start = time.time()
        self.ring_state = RingState.STACKING
        
        try:
            # Enforce containment boundary
            containment_result = self._enforce_containment_boundary(stack)
            if not containment_result["allowed"]:
                return {
                    "success": False,
                    "error": "containment_violation",
                    "details": containment_result["violations"],
                    "stack_id": stack.stack_id
                }
            
            # Order stack by priority (Core > Tracer > Operator)
            ordered_invocations = self._order_stack_by_priority(stack.invocations)
            
            # Execute invocations in order
            execution_results = []
            overall_success = True
            
            for invocation in ordered_invocations:
                # Route through network with house resolution
                result = self.network.invoke_sigil(
                    invocation.sigil_symbol,
                    invocation.house,
                    invocation.parameters,
                    invocation.invoker
                )
                
                execution_results.append(result)
                
                if not result.get("success", False):
                    overall_success = False
                    
                    # Check for containment breach
                    if self._is_containment_breach(result):
                        self._handle_containment_breach(stack, result)
                        break
            
            # Compile final result
            final_result = {
                "success": overall_success,
                "stack_id": stack.stack_id,
                "layered_meaning": stack.get_layered_meaning(),
                "invocations_executed": len(execution_results),
                "total_invocations": len(stack.invocations),
                "execution_results": execution_results,
                "containment_level": stack.containment_level.value,
                "ring_id": self.ring_id
            }
            
            # Update metrics
            if overall_success:
                self.metrics.successful_executions += 1
            else:
                self.metrics.failed_executions += 1
            
            self.metrics.total_invocations_processed += len(stack.invocations)
            
            return final_result
            
        except Exception as e:
            log_anomaly(
                "RING_EXECUTION_ERROR",
                f"Error executing stack {stack.stack_id}: {e}",
                AnomalySeverity.ERROR
            )
            
            return {
                "success": False,
                "error": f"execution_error: {e}",
                "stack_id": stack.stack_id,
                "ring_id": self.ring_id
            }
        
        finally:
            # Clean up
            stack.execution_end = time.time()
            
            if stack.stack_id in self.active_stacks:
                del self.active_stacks[stack.stack_id]
            
            # Update metrics
            if stack.get_execution_time():
                self.metrics.update_execution_time(stack.get_execution_time())
            
            # Log stack trace
            self.telemetry.log_stack_trace(stack, self.current_tick)
            
            # Return to active state
            if self.ring_state == RingState.STACKING:
                self.ring_state = RingState.ACTIVE
    
    def _enforce_containment_boundary(self, stack: SigilStack) -> Dict[str, Any]:
        """Enforce containment boundary rules"""
        violations = []
        
        # Check stack depth
        if len(stack.invocations) > self.containment_boundary.max_stack_depth:
            violations.append(f"Stack depth {len(stack.invocations)} exceeds maximum {self.containment_boundary.max_stack_depth}")
        
        # Check for dangerous glyph combinations
        glyph_symbols = [inv.sigil_symbol for inv in stack.invocations]
        dangerous_patterns = ["⟁", "/X-"]  # Contradiction Break, Schema Restart Call
        
        if any(pattern in glyph_symbols for pattern in dangerous_patterns):
            if stack.containment_level in [ContainmentLevel.OPEN, ContainmentLevel.BASIC]:
                violations.append("Dangerous glyphs require higher containment level")
        
        return {
            "allowed": len(violations) == 0,
            "violations": violations
        }
    
    def _order_stack_by_priority(self, invocations: List[SigilInvocation]) -> List[SigilInvocation]:
        """Order stack by priority: Core > Tracer > Operator"""
        def get_priority_score(invocation: SigilInvocation) -> int:
            # Get invoker priority
            invoker_scores = {
                "core": 100,
                "tracer": 50,
                "operator": 10
            }
            
            invoker_score = 0
            for key, score in invoker_scores.items():
                if key in invocation.invoker.lower():
                    invoker_score = score
                    break
            
            # Get glyph priority
            glyph_priority = sigil_glyph_codex.get_execution_priority([invocation.sigil_symbol])
            
            return invoker_score + glyph_priority
        
        return sorted(invocations, key=get_priority_score, reverse=True)
    
    def _determine_containment_level(self, invocations: List[SigilInvocation]) -> ContainmentLevel:
        """Determine appropriate containment level"""
        danger_score = 0
        
        for invocation in invocations:
            glyph = sigil_glyph_codex.get_glyph(invocation.sigil_symbol)
            if not glyph:
                continue
            
            # High-risk glyphs
            if glyph.symbol in ["⟁", "/X-", "Z~"]:  # Contradiction Break, Schema Restart, Fusion Under Pressure
                danger_score += 3
            
            # Medium-risk glyphs
            if glyph.category == GlyphCategory.COMPOSITE:
                danger_score += 2
            
            # Core minimal glyphs are high priority but stable
            if glyph.category == GlyphCategory.CORE_MINIMAL:
                danger_score += 1
        
        # Determine containment based on danger score
        if danger_score >= 8:
            return ContainmentLevel.QUARANTINE
        elif danger_score >= 5:
            return ContainmentLevel.SEALED
        elif danger_score >= 3:
            return ContainmentLevel.SECURED
        else:
            return ContainmentLevel.BASIC
    
    def _is_containment_breach(self, result: Dict[str, Any]) -> bool:
        """Check if execution result indicates containment breach"""
        if not result.get("success", True):
            error_str = str(result.get("error", "")).lower()
            breach_indicators = ["breach", "overflow", "corruption", "rogue", "leak"]
            return any(indicator in error_str for indicator in breach_indicators)
        return False
    
    def _handle_containment_breach(self, stack: SigilStack, result: Dict[str, Any]):
        """Handle containment breach"""
        self.containment_boundary.active_breaches += 1
        self.metrics.containment_breaches += 1
        
        self.ring_state = RingState.CONTAINMENT_BREACH
        
        self.telemetry.log_conflict_alert(
            "containment_breach",
            f"Stack {stack.stack_id} breached containment: {result.get('error', 'unknown')}",
            self.current_tick
        )
        
        # Check for emergency seal trigger
        if self.containment_boundary.is_breach_critical():
            self._trigger_emergency_seal()
        
        log_anomaly(
            "CONTAINMENT_BREACH",
            f"Containment breach in stack {stack.stack_id}",
            AnomalySeverity.CRITICAL
        )
    
    def _trigger_emergency_seal(self):
        """Trigger emergency seal of the ring"""
        self.ring_state = RingState.EMERGENCY_SEAL
        self.emergency_seal_active = True
        self.metrics.emergency_seals_triggered += 1
        
        # Clear all active operations
        with self.ring_lock:
            self.active_stacks.clear()
            self.execution_queue.clear()
            self.spillover_queue.clear()
            self.current_circle = None
        
        log_anomaly(
            "RING_EMERGENCY_SEAL",
            f"Emergency seal triggered for ring {self.ring_id}",
            AnomalySeverity.CRITICAL
        )
        
        emit_signal(
            SignalType.ENTROPY,
            "enhanced_sigil_ring",
            {
                "event": "emergency_seal",
                "ring_id": self.ring_id,
                "breach_count": self.containment_boundary.active_breaches
            }
        )
    
    def tick(self, tick_number: int):
        """Process a tick in the ring"""
        if self.emergency_seal_active:
            return
        
        # Form new casting circle for this tick
        if self.ring_state == RingState.ACTIVE:
            try:
                self.form_casting_circle(tick_number)
            except RuntimeError as e:
                logger.error(f"Failed to form casting circle for tick {tick_number}: {e}")
    
    def get_visual_representation(self) -> Dict[str, Any]:
        """Get visual representation data for GUI"""
        if not self.current_circle:
            return {"active": False, "ring_id": self.ring_id}
        
        # House nodes with activity
        house_nodes = {}
        for house, glyphs in self.current_circle.house_nodes.items():
            house_nodes[house.value] = {
                "active": len(glyphs) > 0,
                "glyph_count": len(glyphs),
                "glyphs": glyphs,
                "emoji": self._get_house_emoji(house)
            }
        
        # Orbiting glyphs
        orbiting_glyphs = []
        for invocation in self.current_circle.active_invocations:
            glyph = sigil_glyph_codex.get_glyph(invocation.sigil_symbol)
            orbiting_glyphs.append({
                "symbol": invocation.sigil_symbol,
                "name": glyph.name if glyph else "Unknown",
                "house": invocation.house.value if invocation.house else None,
                "priority": sigil_glyph_codex.get_execution_priority([invocation.sigil_symbol])
            })
        
        return {
            "active": True,
            "ring_id": self.ring_id,
            "tick_id": self.current_circle.tick_id,
            "state": self.ring_state.value,
            "containment_level": self.containment_boundary.level.value,
            "house_nodes": house_nodes,
            "orbiting_glyphs": orbiting_glyphs,
            "circle_coherence": self.current_circle.circle_coherence,
            "invocation_count": self.current_circle.get_invocation_count(),
            "max_capacity": self.containment_boundary.max_invocations_per_tick,
            "spillover_count": len(self.spillover_queue)
        }
    
    def _get_house_emoji(self, house: SigilHouse) -> str:
        """Get emoji representation for house"""
        house_emojis = {
            SigilHouse.MEMORY: "🌸",
            SigilHouse.PURIFICATION: "🔥", 
            SigilHouse.WEAVING: "🕸️",
            SigilHouse.FLAME: "⚡",
            SigilHouse.MIRRORS: "🪞",
            SigilHouse.ECHOES: "🔊"
        }
        return house_emojis.get(house, "❓")
    
    def get_ring_status(self) -> Dict[str, Any]:
        """Get comprehensive ring status"""
        return {
            "ring_id": self.ring_id,
            "ring_state": self.ring_state.value,
            "emergency_seal_active": self.emergency_seal_active,
            "current_tick": self.current_tick,
            "active_circle": self.current_circle is not None,
            "spillover_queue_size": len(self.spillover_queue),
            "containment_boundary": {
                "level": self.containment_boundary.level.value,
                "active_breaches": self.containment_boundary.active_breaches,
                "is_critical": self.containment_boundary.is_breach_critical()
            },
            "metrics": {
                "total_circles_formed": self.metrics.total_circles_formed,
                "success_rate": self.metrics.get_success_rate(),
                "average_execution_time": self.metrics.average_execution_time,
                "containment_breaches": self.metrics.containment_breaches,
                "emergency_seals": self.metrics.emergency_seals_triggered,
                "ring_overloads": self.metrics.ring_overloads
            }
        }

# Global enhanced ring instance
enhanced_sigil_ring = EnhancedSigilRing()

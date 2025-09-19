#!/usr/bin/env python3
"""
DAWN Unified Pulse System
========================

Complete implementation of DAWN's pulse system based on RTF specifications.
Provides the central nervous system for DAWN - the information highway that
carries tick and recession data throughout the consciousness architecture.

Core Components:
- Pulse Scheduler: Executes micro-actions every tick
- SCUP Controller: Maintains semantic coherence under pressure
- Zone Management: Adaptive control policies based on system state
- Sigil Integration: Routes pulse actions through sigil ring system

Based on DAWN-docs/SCUP + Pulse/ specifications.
"""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
from datetime import datetime
import logging
import json
import uuid

logger = logging.getLogger(__name__)


class PulseZone(Enum):
    """Operating zones with distinct control policies"""
    GREEN = "green"      # F* < 0.5, SHI >= θ_ok - Observe & conserve
    AMBER = "amber"      # 0.5 <= F* < 1.0 - Pre-emptive stabilization  
    RED = "red"          # 1.0 <= F* < 1.5 - Active intervention
    BLACK = "black"      # F* >= 1.5 or SHI < θ_crit - Fail-soft containment


class PulseActionType(Enum):
    """Types of pulse actions that can be scheduled"""
    PURGE_SOOT = "purge_soot"
    REINFORCE_THREAD = "reinforce_thread"
    FLAME_VENT = "flame_vent"
    WHALE_SCAN = "whale_scan"
    BEETLE_SWEEP = "beetle_sweep"
    BEE_POLLINATE = "bee_pollinate"
    ANT_SWARM = "ant_swarm"
    WEAVE_THREAD = "weave_thread"
    PRUNE_EDGE = "prune_edge"
    LIFT_NODE = "lift_node"
    SINK_NODE = "sink_node"
    OWL_AUDIT = "owl_audit"


class SigilHouse(Enum):
    """Sigil houses for routing pulse actions"""
    PURIFICATION = "purification"
    WEAVING = "weaving"
    FLAME = "flame"
    MIRRORS = "mirrors"
    ECHO = "echo"


@dataclass
class PulseAction:
    """Individual pulse action to be executed"""
    action_type: PulseActionType
    target: str                    # Target ID (node, edge, cluster, etc.)
    priority: float = 0.5         # Action priority [0,1]
    energy_cost: float = 0.1      # Energy budget required
    house: Optional[SigilHouse] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    scheduled_tick: int = 0
    executed: bool = False
    success: bool = False
    execution_time: float = 0.0
    result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SCUPState:
    """SCUP (Semantic Coherence Under Pressure) controller state"""
    current_shi: float = 0.8       # Current SHI (semantic coherence)
    target_shi: float = 0.8        # Target SHI setpoint
    forecast_index: float = 0.3    # Smoothed forecast index F*
    pressure: float = 0.0          # Current system pressure
    
    # PID controller state
    error_integral: float = 0.0
    previous_error: float = 0.0
    actuation_budget: float = 0.0  # Current tick's actuation budget [0,1]
    
    # Zone state
    current_zone: PulseZone = PulseZone.GREEN
    zone_entry_time: float = 0.0
    zone_stability: float = 1.0
    
    # Control parameters
    kp: float = 0.6               # Proportional gain
    ki: float = 0.1               # Integral gain  
    kd: float = 0.2               # Derivative gain


@dataclass
class PulseMetrics:
    """Performance metrics for pulse system"""
    total_actions_scheduled: int = 0
    total_actions_executed: int = 0
    total_actions_successful: int = 0
    
    # Per-zone statistics
    zone_time_distribution: Dict[str, float] = field(default_factory=dict)
    zone_action_counts: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    average_execution_time: float = 0.0
    success_rate: float = 1.0
    energy_efficiency: float = 1.0
    
    # SCUP metrics
    shi_stability: float = 1.0
    zone_stability: float = 1.0
    pressure_variance: float = 0.0


class PulseScheduler:
    """
    Core pulse scheduler that executes micro-actions every tick.
    Translates SCUP actuation budget into concrete interventions.
    """
    
    def __init__(self):
        self.rate_limits = {
            PulseActionType.PURGE_SOOT: 2,
            PulseActionType.WEAVE_THREAD: 1,
            PulseActionType.FLAME_VENT: 1,
            PulseActionType.WHALE_SCAN: 1,
            PulseActionType.BEETLE_SWEEP: 1,
            PulseActionType.BEE_POLLINATE: 2,
            PulseActionType.ANT_SWARM: 3
        }
        
        # Zone-based priority orders
        self.zone_priorities = {
            PulseZone.GREEN: [
                PulseActionType.ANT_SWARM,
                PulseActionType.BEE_POLLINATE,
                PulseActionType.WEAVE_THREAD
            ],
            PulseZone.AMBER: [
                PulseActionType.PURGE_SOOT,
                PulseActionType.WEAVE_THREAD,
                PulseActionType.BEE_POLLINATE,
                PulseActionType.WHALE_SCAN
            ],
            PulseZone.RED: [
                PulseActionType.PURGE_SOOT,
                PulseActionType.WEAVE_THREAD,
                PulseActionType.FLAME_VENT,
                PulseActionType.WHALE_SCAN,
                PulseActionType.BEETLE_SWEEP
            ],
            PulseZone.BLACK: [
                PulseActionType.PURGE_SOOT,
                PulseActionType.FLAME_VENT,
                PulseActionType.WHALE_SCAN,
                PulseActionType.OWL_AUDIT
            ]
        }
        
        # Backoff tracking
        self.backoff_state = defaultdict(float)
        self.backoff_decay = 0.95
        
        # Action execution tracking
        self.current_tick_actions: List[PulseAction] = []
        self.execution_history: deque = deque(maxlen=1000)
        
    def plan_actions(self, scup_state: SCUPState, hotlists: Dict[str, List[str]]) -> List[PulseAction]:
        """
        Plan pulse actions for current tick based on SCUP state and hotlists.
        
        Args:
            scup_state: Current SCUP controller state
            hotlists: Dictionary of problematic elements by category
            
        Returns:
            List of planned pulse actions
        """
        actions = []
        zone = scup_state.current_zone
        budget = scup_state.actuation_budget
        
        # Get zone-appropriate action priorities
        priority_order = self.zone_priorities.get(zone, [])
        
        # Plan actions based on hotlists and zone policy
        for action_type in priority_order:
            if budget <= 0:
                break
                
            # Check rate limits
            current_count = len([a for a in actions if a.action_type == action_type])
            if current_count >= self.rate_limits.get(action_type, 1):
                continue
                
            # Check backoff state
            if self.backoff_state[action_type] > 0.5:
                continue
                
            # Plan specific actions based on type and hotlists
            new_actions = self._plan_specific_actions(
                action_type, zone, hotlists, budget
            )
            
            for action in new_actions:
                if budget >= action.energy_cost:
                    actions.append(action)
                    budget -= action.energy_cost
        
        return actions
    
    def _plan_specific_actions(self, action_type: PulseActionType, 
                             zone: PulseZone, hotlists: Dict[str, List[str]], 
                             budget: float) -> List[PulseAction]:
        """Plan specific actions based on type and current conditions"""
        actions = []
        
        if action_type == PulseActionType.PURGE_SOOT:
            # Target soot clusters from hotlist
            soot_targets = hotlists.get('soot_clusters', [])[:2]  # Max 2 per tick
            for target in soot_targets:
                actions.append(PulseAction(
                    action_type=action_type,
                    target=target,
                    priority=0.8,
                    energy_cost=0.15,
                    house=SigilHouse.PURIFICATION,
                    parameters={'intensity': 'micro' if zone == PulseZone.AMBER else 'batch'}
                ))
        
        elif action_type == PulseActionType.REINFORCE_THREAD:
            # Target high-tension edges
            tension_edges = hotlists.get('tension_edges', [])[:1]  # Max 1 per tick
            for target in tension_edges:
                actions.append(PulseAction(
                    action_type=action_type,
                    target=target,
                    priority=0.7,
                    energy_cost=0.1,
                    house=SigilHouse.WEAVING,
                    parameters={'strength': 0.3}
                ))
        
        elif action_type == PulseActionType.FLAME_VENT:
            # Emergency thermal release
            if zone in [PulseZone.RED, PulseZone.BLACK]:
                actions.append(PulseAction(
                    action_type=action_type,
                    target="system",
                    priority=0.9,
                    energy_cost=0.2,
                    house=SigilHouse.FLAME,
                    parameters={'scope': 'emergency' if zone == PulseZone.BLACK else 'local'}
                ))
        
        elif action_type == PulseActionType.WHALE_SCAN:
            # Drift anomaly scanning
            drift_targets = hotlists.get('drift_anomalies', ['global'])[:1]
            for target in drift_targets:
                actions.append(PulseAction(
                    action_type=action_type,
                    target=target,
                    priority=0.6,
                    energy_cost=0.05,
                    house=SigilHouse.MIRRORS,
                    parameters={'window': 50, 'depth': 'shallow'}
                ))
        
        elif action_type == PulseActionType.BEE_POLLINATE:
            # Pigment balance restoration
            imbalance_areas = hotlists.get('pigment_imbalances', [])[:2]
            for target in imbalance_areas:
                actions.append(PulseAction(
                    action_type=action_type,
                    target=target,
                    priority=0.5,
                    energy_cost=0.08,
                    house=SigilHouse.ECHO,
                    parameters={'pollination_strength': 0.4}
                ))
        
        elif action_type == PulseActionType.ANT_SWARM:
            # Normalization operations
            if zone == PulseZone.GREEN:
                actions.append(PulseAction(
                    action_type=action_type,
                    target="global",
                    priority=0.3,
                    energy_cost=0.03,
                    house=SigilHouse.ECHO,
                    parameters={'swarm_size': 'small'}
                ))
        
        return actions
    
    def execute_actions(self, actions: List[PulseAction], tick: int) -> Dict[str, Any]:
        """
        Execute planned actions and track results.
        
        Args:
            actions: List of actions to execute
            tick: Current tick number
            
        Returns:
            Execution results and metrics
        """
        results = {
            'tick': tick,
            'planned': len(actions),
            'executed': 0,
            'successful': 0,
            'skipped': 0,
            'total_energy_used': 0.0,
            'execution_time': 0.0,
            'actions': []
        }
        
        start_time = time.time()
        
        for action in actions:
            action.scheduled_tick = tick
            
            try:
                # Simulate action execution (replace with actual implementation)
                execution_start = time.time()
                success = self._execute_single_action(action)
                execution_time = time.time() - execution_start
                
                action.executed = True
                action.success = success
                action.execution_time = execution_time
                
                results['executed'] += 1
                results['total_energy_used'] += action.energy_cost
                
                if success:
                    results['successful'] += 1
                    # Reduce backoff for successful actions
                    self.backoff_state[action.action_type] *= 0.8
                else:
                    # Increase backoff for failed actions
                    self.backoff_state[action.action_type] = min(1.0, 
                        self.backoff_state[action.action_type] + 0.3)
                
                results['actions'].append({
                    'type': action.action_type.value,
                    'target': action.target,
                    'success': success,
                    'energy_cost': action.energy_cost,
                    'execution_time': execution_time
                })
                
            except Exception as e:
                logger.error(f"Error executing pulse action {action.action_type}: {e}")
                action.executed = True
                action.success = False
                results['skipped'] += 1
        
        results['execution_time'] = time.time() - start_time
        
        # Update backoff decay
        for action_type in self.backoff_state:
            self.backoff_state[action_type] *= self.backoff_decay
        
        # Store in execution history
        self.execution_history.append(results)
        
        return results
    
    def _execute_single_action(self, action: PulseAction) -> bool:
        """
        Execute a single pulse action.
        
        This is a placeholder that simulates action execution.
        In the full implementation, this would route through the sigil system.
        """
        # Simulate execution time and success probability
        time.sleep(0.001)  # Simulate processing time
        
        # Success probability based on action type and current conditions
        base_success_rate = 0.85
        
        # Adjust success rate based on backoff state
        backoff_penalty = self.backoff_state[action.action_type] * 0.2
        success_rate = base_success_rate - backoff_penalty
        
        # Simulate execution
        success = np.random.random() < success_rate
        
        if success:
            # Simulate positive system effects
            action.result = {
                'shi_delta': np.random.uniform(0.01, 0.05),
                'tension_reduction': np.random.uniform(0.02, 0.08),
                'soot_reduction': np.random.uniform(0.01, 0.06)
            }
        else:
            action.result = {'error': 'execution_failed'}
        
        return success


class SCUPController:
    """
    SCUP (Semantic Coherence Under Pressure) controller.
    Maintains system coherence through PID-based actuation budget control.
    """
    
    def __init__(self, target_shi: float = 0.8):
        self.state = SCUPState(target_shi=target_shi)
        self.shi_history: deque = deque(maxlen=100)
        self.forecast_history: deque = deque(maxlen=100)
        
        # Zone transition hysteresis
        self.hysteresis_margin = 0.05
        self.zone_thresholds = {
            'green_amber': 0.5,
            'amber_red': 1.0,
            'red_black': 1.5
        }
        
        # SHI thresholds
        self.shi_thresholds = {
            'ok': 0.70,
            'warn': 0.55,
            'crit': 0.40
        }
    
    def update_state(self, current_shi: float, forecast_index: float, 
                    pressure: float = 0.0) -> SCUPState:
        """
        Update SCUP controller state with new measurements.
        
        Args:
            current_shi: Current semantic coherence (SHI)
            forecast_index: Smoothed forecast index F*
            pressure: System pressure measurement
            
        Returns:
            Updated SCUP state
        """
        # Update measurements
        self.state.current_shi = current_shi
        self.state.forecast_index = forecast_index
        self.state.pressure = pressure
        
        # Store history
        self.shi_history.append(current_shi)
        self.forecast_history.append(forecast_index)
        
        # Update zone based on forecast index and SHI
        self._update_zone()
        
        # Calculate PID control signal
        self._calculate_actuation_budget()
        
        return self.state
    
    def _update_zone(self):
        """Update current operating zone with hysteresis"""
        f_star = self.state.forecast_index
        shi = self.state.current_shi
        current_zone = self.state.current_zone
        
        # Determine target zone based on thresholds
        target_zone = PulseZone.GREEN
        
        # Check SHI-based zone escalation
        if shi < self.shi_thresholds['crit']:
            target_zone = PulseZone.BLACK
        elif shi < self.shi_thresholds['warn']:
            target_zone = PulseZone.RED
        elif shi < self.shi_thresholds['ok']:
            target_zone = PulseZone.AMBER
        
        # Check forecast-based zone escalation
        if f_star >= self.zone_thresholds['red_black']:
            target_zone = max(target_zone, PulseZone.BLACK, key=lambda z: z.value)
        elif f_star >= self.zone_thresholds['amber_red']:
            target_zone = max(target_zone, PulseZone.RED, key=lambda z: z.value)
        elif f_star >= self.zone_thresholds['green_amber']:
            target_zone = max(target_zone, PulseZone.AMBER, key=lambda z: z.value)
        
        # Apply hysteresis for zone transitions
        if target_zone != current_zone:
            # Check if we should transition
            should_transition = True
            
            # Hysteresis: require margin to move to lower zone
            zone_values = {'green': 0, 'amber': 1, 'red': 2, 'black': 3}
            if zone_values[target_zone.value] < zone_values[current_zone.value]:
                # Moving to lower zone - check hysteresis
                if target_zone == PulseZone.GREEN:
                    should_transition = f_star < (self.zone_thresholds['green_amber'] - self.hysteresis_margin)
                elif target_zone == PulseZone.AMBER:
                    should_transition = f_star < (self.zone_thresholds['amber_red'] - self.hysteresis_margin)
                elif target_zone == PulseZone.RED:
                    should_transition = f_star < (self.zone_thresholds['red_black'] - self.hysteresis_margin)
            
            if should_transition:
                logger.info(f"SCUP zone transition: {current_zone.value} -> {target_zone.value}")
                self.state.current_zone = target_zone
                self.state.zone_entry_time = time.time()
    
    def _calculate_actuation_budget(self):
        """Calculate actuation budget using PID controller"""
        target = self.state.target_shi
        current = self.state.current_shi
        
        # Calculate error
        error = target - current
        
        # PID terms
        p_term = self.state.kp * error
        i_term = self.state.ki * self.state.error_integral
        d_term = self.state.kd * (error - self.state.previous_error)
        
        # Update integral and previous error
        self.state.error_integral += error
        self.state.previous_error = error
        
        # Calculate actuation budget
        raw_budget = p_term + i_term + d_term
        
        # Clamp to [0,1] and apply zone-based scaling
        zone_scaling = {
            PulseZone.GREEN: 0.3,
            PulseZone.AMBER: 0.6,
            PulseZone.RED: 0.9,
            PulseZone.BLACK: 1.0
        }
        
        max_budget = zone_scaling[self.state.current_zone]
        self.state.actuation_budget = max(0.0, min(max_budget, raw_budget))
    
    def get_zone_statistics(self) -> Dict[str, Any]:
        """Get statistics about zone behavior"""
        if len(self.shi_history) < 2:
            return {}
        
        return {
            'current_zone': self.state.current_zone.value,
            'zone_stability': self.state.zone_stability,
            'shi_mean': np.mean(list(self.shi_history)),
            'shi_std': np.std(list(self.shi_history)),
            'forecast_mean': np.mean(list(self.forecast_history)),
            'forecast_std': np.std(list(self.forecast_history)),
            'actuation_budget': self.state.actuation_budget,
            'time_in_zone': time.time() - self.state.zone_entry_time
        }


class UnifiedPulseSystem:
    """
    Unified pulse system manager integrating all pulse components.
    Provides the central nervous system for DAWN consciousness.
    """
    
    def __init__(self):
        self.scheduler = PulseScheduler()
        self.scup_controller = SCUPController()
        
        # System state
        self.running = False
        self.tick_count = 0
        self.last_tick_time = time.time()
        
        # Threading
        self.update_thread: Optional[threading.Thread] = None
        self.update_lock = threading.RLock()
        
        # Metrics and monitoring
        self.metrics = PulseMetrics()
        self.performance_history: deque = deque(maxlen=1000)
        
        # Hotlist tracking (would be populated by other systems)
        self.hotlists = {
            'soot_clusters': [],
            'tension_edges': [],
            'drift_anomalies': [],
            'pigment_imbalances': []
        }
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info("🫁 Unified Pulse System initialized")
    
    def start(self) -> bool:
        """Start the pulse system"""
        if self.running:
            logger.warning("Pulse system already running")
            return False
        
        self.running = True
        
        # Start background update thread
        self.update_thread = threading.Thread(
            target=self._pulse_loop,
            name="pulse_system_loop",
            daemon=True
        )
        self.update_thread.start()
        
        self._emit_event('pulse_started', {'timestamp': time.time()})
        logger.info("🫁 Pulse system started")
        return True
    
    def stop(self) -> bool:
        """Stop the pulse system"""
        if not self.running:
            return False
        
        self.running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        self._emit_event('pulse_stopped', {'timestamp': time.time()})
        logger.info("🫁 Pulse system stopped")
        return True
    
    def pulse_tick(self, current_shi: float = 0.8, forecast_index: float = 0.3,
                  pressure: float = 0.0, external_hotlists: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Execute a single pulse tick - the core heartbeat of DAWN.
        
        This implements the RTF specification:
        1. Ingest: Pull inputs from forecasting, SCUP, hotlists
        2. Plan: Build slot list of micro-actions  
        3. Route: Convert to sigil invocations
        4. Execute: Fire sigil stacks in priority order
        5. Evaluate: Measure outcomes and update controller
        
        Args:
            current_shi: Current semantic coherence measurement
            forecast_index: Smoothed forecast index F*
            pressure: System pressure measurement
            external_hotlists: External hotlists from other subsystems
            
        Returns:
            Pulse tick results and metrics
        """
        with self.update_lock:
            tick_start = time.time()
            self.tick_count += 1
            
            try:
                # 1. Ingest: Update SCUP state
                scup_state = self.scup_controller.update_state(
                    current_shi, forecast_index, pressure
                )
                
                # Update hotlists from external sources
                if external_hotlists:
                    self.hotlists.update(external_hotlists)
                
                # 2. Plan: Generate pulse actions
                planned_actions = self.scheduler.plan_actions(scup_state, self.hotlists)
                
                # 3. Route & 4. Execute: Execute planned actions
                execution_results = self.scheduler.execute_actions(planned_actions, self.tick_count)
                
                # 5. Evaluate: Update metrics and controller feedback
                self._update_metrics(execution_results, scup_state)
                
                # Performance tracking
                tick_time = time.time() - tick_start
                self.last_tick_time = time.time()
                
                # Compile results
                results = {
                    'success': True,
                    'tick': self.tick_count,
                    'tick_time_ms': tick_time * 1000,
                    'scup_state': {
                        'zone': scup_state.current_zone.value,
                        'shi': scup_state.current_shi,
                        'forecast_index': scup_state.forecast_index,
                        'actuation_budget': scup_state.actuation_budget
                    },
                    'execution': execution_results,
                    'metrics': {
                        'success_rate': self.metrics.success_rate,
                        'energy_efficiency': self.metrics.energy_efficiency,
                        'zone_stability': self.metrics.zone_stability
                    }
                }
                
                # Store performance data
                self.performance_history.append(results)
                
                # Emit tick event
                self._emit_event('pulse_tick', results)
                
                return results
                
            except Exception as e:
                logger.error(f"Error in pulse tick: {e}")
                return {
                    'success': False,
                    'tick': self.tick_count,
                    'error': str(e)
                }
    
    def _pulse_loop(self):
        """Background pulse loop running at ~10Hz"""
        while self.running:
            try:
                # Execute pulse tick with default parameters
                self.pulse_tick()
                
                # Sleep for ~100ms (10Hz rhythm)
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in pulse loop: {e}")
                time.sleep(0.5)  # Back off on errors
    
    def _update_metrics(self, execution_results: Dict[str, Any], scup_state: SCUPState):
        """Update system metrics based on execution results"""
        # Update basic counters
        self.metrics.total_actions_scheduled += execution_results['planned']
        self.metrics.total_actions_executed += execution_results['executed']
        self.metrics.total_actions_successful += execution_results['successful']
        
        # Update success rate
        if self.metrics.total_actions_executed > 0:
            self.metrics.success_rate = (
                self.metrics.total_actions_successful / self.metrics.total_actions_executed
            )
        
        # Update zone statistics
        zone_name = scup_state.current_zone.value
        if zone_name not in self.metrics.zone_time_distribution:
            self.metrics.zone_time_distribution[zone_name] = 0.0
            self.metrics.zone_action_counts[zone_name] = 0
        
        self.metrics.zone_time_distribution[zone_name] += 0.1  # 100ms per tick
        self.metrics.zone_action_counts[zone_name] += execution_results['executed']
        
        # Update performance metrics
        if execution_results['execution_time'] > 0:
            # Simple moving average
            alpha = 0.1
            self.metrics.average_execution_time = (
                alpha * execution_results['execution_time'] + 
                (1 - alpha) * self.metrics.average_execution_time
            )
        
        # Energy efficiency (successful actions per energy unit)
        if execution_results['total_energy_used'] > 0:
            current_efficiency = execution_results['successful'] / execution_results['total_energy_used']
            self.metrics.energy_efficiency = (
                0.1 * current_efficiency + 0.9 * self.metrics.energy_efficiency
            )
    
    def update_hotlists(self, new_hotlists: Dict[str, List[str]]):
        """Update hotlists from external systems"""
        with self.update_lock:
            self.hotlists.update(new_hotlists)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        scup_stats = self.scup_controller.get_zone_statistics()
        
        return {
            'running': self.running,
            'tick_count': self.tick_count,
            'uptime_seconds': time.time() - (self.last_tick_time - self.tick_count * 0.1),
            'scup_state': scup_stats,
            'metrics': {
                'total_actions_scheduled': self.metrics.total_actions_scheduled,
                'total_actions_executed': self.metrics.total_actions_executed,
                'success_rate': self.metrics.success_rate,
                'energy_efficiency': self.metrics.energy_efficiency,
                'average_execution_time_ms': self.metrics.average_execution_time * 1000,
                'zone_distribution': self.metrics.zone_time_distribution,
                'zone_action_counts': self.metrics.zone_action_counts
            },
            'hotlists_status': {k: len(v) for k, v in self.hotlists.items()},
            'performance': {
                'last_tick_time': self.last_tick_time,
                'recent_performance': list(self.performance_history)[-10:] if self.performance_history else []
            }
        }
    
    def force_zone_transition(self, target_zone: PulseZone) -> bool:
        """Force transition to specific zone (for testing/emergency)"""
        try:
            with self.update_lock:
                self.scup_controller.state.current_zone = target_zone
                self.scup_controller.state.zone_entry_time = time.time()
                
                logger.warning(f"Forced zone transition to {target_zone.value}")
                self._emit_event('forced_zone_transition', {
                    'target_zone': target_zone.value,
                    'timestamp': time.time()
                })
                return True
        except Exception as e:
            logger.error(f"Error forcing zone transition: {e}")
            return False
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register callback for pulse events"""
        self.event_callbacks[event_type].append(callback)
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered callbacks"""
        for callback in self.event_callbacks[event_type]:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback for {event_type}: {e}")
    
    def export_pulse_data(self, filepath: str) -> bool:
        """Export pulse system data for analysis"""
        try:
            export_data = {
                'system_status': self.get_system_status(),
                'scup_history': {
                    'shi_history': list(self.scup_controller.shi_history),
                    'forecast_history': list(self.scup_controller.forecast_history)
                },
                'execution_history': list(self.scheduler.execution_history),
                'performance_history': list(self.performance_history)
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported pulse data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export pulse data: {e}")
            return False


# Global pulse system instance
_global_pulse_system: Optional[UnifiedPulseSystem] = None

def get_pulse_system(auto_start: bool = True) -> UnifiedPulseSystem:
    """Get the global unified pulse system instance"""
    global _global_pulse_system
    
    if _global_pulse_system is None:
        _global_pulse_system = UnifiedPulseSystem()
        
        if auto_start:
            _global_pulse_system.start()
    
    return _global_pulse_system

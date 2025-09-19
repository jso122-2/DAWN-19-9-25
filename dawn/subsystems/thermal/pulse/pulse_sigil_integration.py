#!/usr/bin/env python3
"""
Pulse-Sigil Integration Layer
============================

Integrates DAWN's pulse system with the sigil ring architecture.
Routes pulse actions through appropriate sigil houses and manages
sigil stack construction for pulse operations.

Based on RTF specifications for pulse routing and sigil composability.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .unified_pulse_system import PulseAction, PulseActionType, SigilHouse

logger = logging.getLogger(__name__)


class SigilGlyph(Enum):
    """Sigil glyphs for pulse operations"""
    # Purification House
    PURGE_SOOT = "purge_soot"
    CLEANSE_RESIDUE = "cleanse_residue"
    
    # Weaving House  
    REINFORCE_THREAD = "reinforce_thread"
    WEAVE_CONNECTION = "weave_connection"
    PRUNE_EDGE = "prune_edge"
    
    # Flame House
    IGNITE = "ignite"
    VENT = "vent"
    EXTINGUISH = "extinguish"
    
    # Mirrors House
    SCAN = "scan"
    REFLECT = "reflect"
    ANCHOR = "anchor"
    
    # Echo House
    POLLINATE = "pollinate"
    AMPLIFY = "amplify"
    NORMALIZE = "normalize"


@dataclass
class SigilStack:
    """A stack of sigil glyphs to be executed together"""
    house: SigilHouse
    glyphs: List[SigilGlyph]
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    energy_cost: float = 0.1
    
    # Execution state
    executed: bool = False
    success: bool = False
    execution_time: float = 0.0
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SigilCompatibilityRule:
    """Rules for sigil glyph compatibility"""
    conflicting_glyphs: Set[Tuple[SigilGlyph, SigilGlyph]]
    required_sequences: Dict[SigilGlyph, List[SigilGlyph]]
    house_restrictions: Dict[SigilHouse, Set[SigilGlyph]]


class PulseSigilRouter:
    """
    Routes pulse actions through the sigil ring system.
    Handles sigil stack construction and compatibility checking.
    """
    
    def __init__(self):
        # Action to sigil mapping
        self.action_to_sigil = {
            PulseActionType.PURGE_SOOT: (SigilHouse.PURIFICATION, SigilGlyph.PURGE_SOOT),
            PulseActionType.REINFORCE_THREAD: (SigilHouse.WEAVING, SigilGlyph.REINFORCE_THREAD),
            PulseActionType.FLAME_VENT: (SigilHouse.FLAME, SigilGlyph.VENT),
            PulseActionType.WHALE_SCAN: (SigilHouse.MIRRORS, SigilGlyph.SCAN),
            PulseActionType.BEETLE_SWEEP: (SigilHouse.PURIFICATION, SigilGlyph.CLEANSE_RESIDUE),
            PulseActionType.BEE_POLLINATE: (SigilHouse.ECHO, SigilGlyph.POLLINATE),
            PulseActionType.ANT_SWARM: (SigilHouse.ECHO, SigilGlyph.NORMALIZE),
            PulseActionType.WEAVE_THREAD: (SigilHouse.WEAVING, SigilGlyph.WEAVE_CONNECTION),
            PulseActionType.PRUNE_EDGE: (SigilHouse.WEAVING, SigilGlyph.PRUNE_EDGE),
            PulseActionType.OWL_AUDIT: (SigilHouse.MIRRORS, SigilGlyph.REFLECT)
        }
        
        # Compatibility rules
        self.compatibility = SigilCompatibilityRule(
            conflicting_glyphs={
                (SigilGlyph.IGNITE, SigilGlyph.EXTINGUISH),
                (SigilGlyph.WEAVE_CONNECTION, SigilGlyph.PRUNE_EDGE),
                (SigilGlyph.AMPLIFY, SigilGlyph.NORMALIZE)
            },
            required_sequences={
                SigilGlyph.IGNITE: [SigilGlyph.VENT, SigilGlyph.EXTINGUISH],
                SigilGlyph.PURGE_SOOT: [SigilGlyph.CLEANSE_RESIDUE]
            },
            house_restrictions={
                SigilHouse.PURIFICATION: {
                    SigilGlyph.PURGE_SOOT, SigilGlyph.CLEANSE_RESIDUE
                },
                SigilHouse.WEAVING: {
                    SigilGlyph.REINFORCE_THREAD, SigilGlyph.WEAVE_CONNECTION, SigilGlyph.PRUNE_EDGE
                },
                SigilHouse.FLAME: {
                    SigilGlyph.IGNITE, SigilGlyph.VENT, SigilGlyph.EXTINGUISH
                },
                SigilHouse.MIRRORS: {
                    SigilGlyph.SCAN, SigilGlyph.REFLECT, SigilGlyph.ANCHOR
                },
                SigilHouse.ECHO: {
                    SigilGlyph.POLLINATE, SigilGlyph.AMPLIFY, SigilGlyph.NORMALIZE
                }
            }
        )
        
        # Execution tracking
        self.stack_history: List[SigilStack] = []
        self.house_load: Dict[SigilHouse, float] = defaultdict(float)
        
    def route_actions(self, actions: List[PulseAction]) -> List[SigilStack]:
        """
        Route pulse actions to sigil stacks.
        
        Args:
            actions: List of pulse actions to route
            
        Returns:
            List of sigil stacks ready for execution
        """
        stacks = []
        
        # Group actions by house
        house_actions: Dict[SigilHouse, List[PulseAction]] = defaultdict(list)
        
        for action in actions:
            if action.action_type in self.action_to_sigil:
                house, glyph = self.action_to_sigil[action.action_type]
                house_actions[house].append(action)
        
        # Create stacks for each house
        for house, house_action_list in house_actions.items():
            stack = self._create_stack_for_house(house, house_action_list)
            if stack:
                stacks.append(stack)
        
        # Validate stack compatibility
        valid_stacks = self._validate_stack_compatibility(stacks)
        
        return valid_stacks
    
    def _create_stack_for_house(self, house: SigilHouse, 
                               actions: List[PulseAction]) -> Optional[SigilStack]:
        """Create a sigil stack for a specific house"""
        if not actions:
            return None
        
        glyphs = []
        total_energy = 0.0
        max_priority = 0.0
        combined_params = {}
        targets = []
        
        for action in actions:
            if action.action_type in self.action_to_sigil:
                house_check, glyph = self.action_to_sigil[action.action_type]
                if house_check == house:
                    glyphs.append(glyph)
                    total_energy += action.energy_cost
                    max_priority = max(max_priority, action.priority)
                    combined_params.update(action.parameters)
                    targets.append(action.target)
        
        if not glyphs:
            return None
        
        # Check house restrictions
        allowed_glyphs = self.compatibility.house_restrictions.get(house, set())
        if allowed_glyphs:
            glyphs = [g for g in glyphs if g in allowed_glyphs]
        
        if not glyphs:
            logger.warning(f"No valid glyphs for house {house.value}")
            return None
        
        # Create stack
        stack = SigilStack(
            house=house,
            glyphs=glyphs,
            target=";".join(targets),  # Combine multiple targets
            parameters=combined_params,
            priority=max_priority,
            energy_cost=total_energy
        )
        
        return stack
    
    def _validate_stack_compatibility(self, stacks: List[SigilStack]) -> List[SigilStack]:
        """Validate that sigil stacks are compatible with each other"""
        valid_stacks = []
        all_glyphs = []
        
        # Collect all glyphs from all stacks
        for stack in stacks:
            all_glyphs.extend(stack.glyphs)
        
        # Check for conflicts
        for stack in stacks:
            stack_valid = True
            
            for glyph in stack.glyphs:
                # Check for conflicting glyphs
                for conflict_pair in self.compatibility.conflicting_glyphs:
                    if glyph in conflict_pair:
                        other_glyph = conflict_pair[0] if glyph == conflict_pair[1] else conflict_pair[1]
                        if other_glyph in all_glyphs:
                            logger.warning(f"Conflicting glyphs: {glyph.value} and {other_glyph.value}")
                            stack_valid = False
                            break
                
                if not stack_valid:
                    break
            
            if stack_valid:
                valid_stacks.append(stack)
            else:
                logger.warning(f"Rejecting invalid stack for house {stack.house.value}")
        
        return valid_stacks
    
    def execute_stacks(self, stacks: List[SigilStack]) -> Dict[str, Any]:
        """
        Execute sigil stacks in priority order.
        
        Args:
            stacks: List of sigil stacks to execute
            
        Returns:
            Execution results and metrics
        """
        results = {
            'total_stacks': len(stacks),
            'executed': 0,
            'successful': 0,
            'failed': 0,
            'total_energy_used': 0.0,
            'execution_time': 0.0,
            'house_results': defaultdict(list)
        }
        
        start_time = time.time()
        
        # Sort stacks by priority
        sorted_stacks = sorted(stacks, key=lambda s: s.priority, reverse=True)
        
        for stack in sorted_stacks:
            try:
                execution_start = time.time()
                success = self._execute_single_stack(stack)
                execution_time = time.time() - execution_start
                
                stack.executed = True
                stack.success = success
                stack.execution_time = execution_time
                
                results['executed'] += 1
                results['total_energy_used'] += stack.energy_cost
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                
                # Record house-specific results
                results['house_results'][stack.house.value].append({
                    'glyphs': [g.value for g in stack.glyphs],
                    'target': stack.target,
                    'success': success,
                    'energy_cost': stack.energy_cost,
                    'execution_time': execution_time
                })
                
                # Update house load tracking
                self.house_load[stack.house] += stack.energy_cost
                
            except Exception as e:
                logger.error(f"Error executing sigil stack for {stack.house.value}: {e}")
                stack.executed = True
                stack.success = False
                results['failed'] += 1
        
        results['execution_time'] = time.time() - start_time
        
        # Store in history
        self.stack_history.extend(sorted_stacks)
        
        # Decay house load
        for house in self.house_load:
            self.house_load[house] *= 0.95
        
        return results
    
    def _execute_single_stack(self, stack: SigilStack) -> bool:
        """
        Execute a single sigil stack.
        
        This is a placeholder implementation that simulates sigil execution.
        In the full system, this would interface with the actual sigil ring.
        """
        # Simulate execution time based on stack complexity
        glyph_count = len(stack.glyphs)
        base_time = 0.001 * glyph_count
        time.sleep(base_time)
        
        # Simulate success probability based on house load and glyph complexity
        base_success_rate = 0.9
        
        # House load penalty
        load_penalty = self.house_load[stack.house] * 0.1
        
        # Glyph complexity penalty
        complexity_penalty = (glyph_count - 1) * 0.05
        
        success_rate = base_success_rate - load_penalty - complexity_penalty
        success = np.random.random() < max(0.5, success_rate)
        
        if success:
            # Simulate positive effects
            stack.results = {
                'glyphs_executed': len(stack.glyphs),
                'energy_consumed': stack.energy_cost,
                'house_efficiency': 1.0 - load_penalty,
                'system_effects': self._generate_system_effects(stack)
            }
        else:
            stack.results = {
                'error': 'execution_failed',
                'house_overload': load_penalty > 0.3,
                'complexity_overload': complexity_penalty > 0.2
            }
        
        return success
    
    def _generate_system_effects(self, stack: SigilStack) -> Dict[str, float]:
        """Generate simulated system effects for successful stack execution"""
        effects = {}
        
        # House-specific effects
        if stack.house == SigilHouse.PURIFICATION:
            effects['soot_reduction'] = np.random.uniform(0.05, 0.15)
            effects['residue_cleared'] = np.random.uniform(0.03, 0.08)
        
        elif stack.house == SigilHouse.WEAVING:
            effects['connection_strength'] = np.random.uniform(0.02, 0.06)
            effects['topology_stability'] = np.random.uniform(0.01, 0.04)
        
        elif stack.house == SigilHouse.FLAME:
            effects['thermal_release'] = np.random.uniform(0.1, 0.3)
            effects['pressure_relief'] = np.random.uniform(0.05, 0.12)
        
        elif stack.house == SigilHouse.MIRRORS:
            effects['drift_detection'] = np.random.uniform(0.02, 0.05)
            effects['anomaly_identification'] = np.random.uniform(0.01, 0.03)
        
        elif stack.house == SigilHouse.ECHO:
            effects['pigment_balance'] = np.random.uniform(0.03, 0.07)
            effects['harmonic_resonance'] = np.random.uniform(0.02, 0.05)
        
        return effects
    
    def get_house_status(self) -> Dict[str, Any]:
        """Get status of all sigil houses"""
        return {
            'house_loads': dict(self.house_load),
            'recent_executions': len([s for s in self.stack_history[-100:] if s.executed]),
            'success_rates': self._calculate_house_success_rates(),
            'glyph_usage': self._calculate_glyph_usage()
        }
    
    def _calculate_house_success_rates(self) -> Dict[str, float]:
        """Calculate success rates by house"""
        rates = {}
        recent_stacks = self.stack_history[-100:]  # Last 100 executions
        
        for house in SigilHouse:
            house_stacks = [s for s in recent_stacks if s.house == house and s.executed]
            if house_stacks:
                successful = sum(1 for s in house_stacks if s.success)
                rates[house.value] = successful / len(house_stacks)
            else:
                rates[house.value] = 1.0
        
        return rates
    
    def _calculate_glyph_usage(self) -> Dict[str, int]:
        """Calculate glyph usage statistics"""
        usage = defaultdict(int)
        recent_stacks = self.stack_history[-100:]
        
        for stack in recent_stacks:
            for glyph in stack.glyphs:
                usage[glyph.value] += 1
        
        return dict(usage)


class PulseSigilIntegration:
    """
    Main integration class connecting pulse system with sigil routing.
    Provides unified interface for pulse-sigil operations.
    """
    
    def __init__(self):
        self.router = PulseSigilRouter()
        self.integration_active = False
        
        # Performance tracking
        self.routing_times = []
        self.execution_times = []
        self.total_operations = 0
        
        logger.info("🔗 Pulse-Sigil Integration initialized")
    
    def activate_integration(self) -> bool:
        """Activate pulse-sigil integration"""
        try:
            self.integration_active = True
            logger.info("🔗 Pulse-Sigil Integration activated")
            return True
        except Exception as e:
            logger.error(f"Failed to activate pulse-sigil integration: {e}")
            return False
    
    def process_pulse_actions(self, actions: List[PulseAction]) -> Dict[str, Any]:
        """
        Process pulse actions through the sigil system.
        
        This is the main interface that the pulse system calls to execute actions.
        
        Args:
            actions: List of pulse actions to process
            
        Returns:
            Combined routing and execution results
        """
        if not self.integration_active:
            logger.warning("Pulse-sigil integration not active")
            return {'error': 'integration_not_active'}
        
        try:
            # Route actions to sigil stacks
            routing_start = time.time()
            stacks = self.router.route_actions(actions)
            routing_time = time.time() - routing_start
            
            # Execute sigil stacks
            execution_start = time.time()
            execution_results = self.router.execute_stacks(stacks)
            execution_time = time.time() - execution_start
            
            # Track performance
            self.routing_times.append(routing_time)
            self.execution_times.append(execution_time)
            self.total_operations += 1
            
            # Keep only recent performance data
            if len(self.routing_times) > 100:
                self.routing_times = self.routing_times[-100:]
                self.execution_times = self.execution_times[-100:]
            
            # Combine results
            results = {
                'routing': {
                    'actions_received': len(actions),
                    'stacks_created': len(stacks),
                    'routing_time_ms': routing_time * 1000
                },
                'execution': execution_results,
                'total_time_ms': (routing_time + execution_time) * 1000,
                'integration_active': self.integration_active
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing pulse actions through sigil system: {e}")
            return {'error': str(e)}
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        house_status = self.router.get_house_status()
        
        avg_routing_time = np.mean(self.routing_times) if self.routing_times else 0
        avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0
        
        return {
            'active': self.integration_active,
            'total_operations': self.total_operations,
            'performance': {
                'average_routing_time_ms': avg_routing_time * 1000,
                'average_execution_time_ms': avg_execution_time * 1000,
                'total_average_time_ms': (avg_routing_time + avg_execution_time) * 1000
            },
            'sigil_system': house_status,
            'stack_history_size': len(self.router.stack_history)
        }
    
    def reset_integration(self):
        """Reset integration state and statistics"""
        self.router.stack_history.clear()
        self.router.house_load.clear()
        self.routing_times.clear()
        self.execution_times.clear()
        self.total_operations = 0
        
        logger.info("🔗 Pulse-Sigil Integration reset")


# Global integration instance
_global_pulse_sigil_integration: Optional[PulseSigilIntegration] = None

def get_pulse_sigil_integration() -> PulseSigilIntegration:
    """Get the global pulse-sigil integration instance"""
    global _global_pulse_sigil_integration
    
    if _global_pulse_sigil_integration is None:
        _global_pulse_sigil_integration = PulseSigilIntegration()
        _global_pulse_sigil_integration.activate_integration()
    
    return _global_pulse_sigil_integration

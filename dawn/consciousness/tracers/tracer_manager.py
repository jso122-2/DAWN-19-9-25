"""
Tracer Manager for DAWN consciousness system.

Manages the complete tracer ecosystem including spawning, monitoring,
retirement, and nutrient budget allocation.
"""

from typing import Dict, List, Type, Any, Optional, Tuple
from collections import defaultdict, deque
import logging
import time
from .base_tracer import BaseTracer, TracerType, TracerStatus, TracerReport

logger = logging.getLogger(__name__)


class TracerEcosystemMetrics:
    """Metrics and telemetry for the tracer ecosystem"""
    
    def __init__(self):
        self.spawn_counts = defaultdict(int)
        self.retire_counts = defaultdict(int)
        self.active_counts = defaultdict(int)
        self.nutrient_usage_history = deque(maxlen=100)
        self.report_counts = defaultdict(int)
        self.alert_counts = defaultdict(int)
        self.total_reports_generated = 0
        self.total_tracers_spawned = 0
        self.total_tracers_retired = 0
        
    def update_spawn(self, tracer_type: TracerType):
        """Record a tracer spawn"""
        self.spawn_counts[tracer_type] += 1
        self.total_tracers_spawned += 1
        
    def update_retire(self, tracer_type: TracerType):
        """Record a tracer retirement"""
        self.retire_counts[tracer_type] += 1
        self.total_tracers_retired += 1
        
    def update_active_counts(self, active_tracers: Dict[str, BaseTracer]):
        """Update active tracer counts"""
        self.active_counts.clear()
        for tracer in active_tracers.values():
            self.active_counts[tracer.tracer_type] += 1
            
    def update_nutrient_usage(self, usage: float):
        """Record current nutrient usage"""
        self.nutrient_usage_history.append(usage)
        
    def record_reports(self, reports: List[TracerReport]):
        """Record generated reports"""
        for report in reports:
            self.report_counts[report.tracer_type] += 1
            self.alert_counts[report.severity] += 1
            self.total_reports_generated += 1
            
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            "active_tracers": dict(self.active_counts),
            "spawn_counts": dict(self.spawn_counts),
            "retire_counts": dict(self.retire_counts),
            "report_counts": dict(self.report_counts),
            "alert_counts": dict(self.alert_counts),
            "total_tracers_spawned": self.total_tracers_spawned,
            "total_tracers_retired": self.total_tracers_retired,
            "total_reports_generated": self.total_reports_generated,
            "avg_nutrient_usage": sum(self.nutrient_usage_history) / len(self.nutrient_usage_history) 
                                if self.nutrient_usage_history else 0.0
        }


class TracerManager:
    """
    Manages the complete tracer ecosystem.
    
    Responsibilities:
    - Spawning tracers based on conditions and budget
    - Managing active tracer lifecycle
    - Enforcing rate limits and safeguards
    - Nutrient budget allocation and recycling
    - Ecosystem metrics and telemetry
    """
    
    def __init__(self, nutrient_budget: float = 100.0):
        self.active_tracers: Dict[str, BaseTracer] = {}
        self.tracer_classes: Dict[TracerType, Type[BaseTracer]] = {}
        self.nutrient_budget = nutrient_budget
        self.current_nutrient_usage = 0.0
        self.recycled_energy_pool = 0.0
        self.metrics = TracerEcosystemMetrics()
        
        # Rate limits per tick (flood protection)
        self.rate_limits = {
            TracerType.CROW: 10,        # Fast, lightweight
            TracerType.ANT: 20,         # Swarmable
            TracerType.BEE: 15,         # Scalable
            TracerType.SPIDER: 5,       # Medium cost
            TracerType.BEETLE: 3,       # Medium cost
            TracerType.WHALE: 1,        # High cost
            TracerType.OWL: 1,          # High cost
            TracerType.MEDIEVAL_BEE: 1  # High cost, rare
        }
        
        # Spawn cooldowns (prevent excessive spawning)
        self.spawn_cooldowns = {
            TracerType.WHALE: 10,       # Expensive, cooldown needed
            TracerType.OWL: 15,         # Archival overhead
            TracerType.MEDIEVAL_BEE: 100 # Epoch-scale spawning
        }
        
        self.last_spawn_tick = defaultdict(int)
        
    def register_tracer_class(self, tracer_type: TracerType, tracer_class: Type[BaseTracer]):
        """Register a tracer implementation"""
        self.tracer_classes[tracer_type] = tracer_class
        logger.info(f"Registered tracer class: {tracer_type.value}")
        
    def get_available_budget(self) -> float:
        """Get current available nutrient budget including recycled energy"""
        return self.nutrient_budget - self.current_nutrient_usage + self.recycled_energy_pool
        
    def evaluate_spawning(self, tick_id: int, context: Dict[str, Any]) -> List[BaseTracer]:
        """Evaluate which tracers should spawn based on conditions and constraints"""
        new_tracers = []
        spawn_counts_this_tick = defaultdict(int)
        
        for tracer_type, tracer_class in self.tracer_classes.items():
            # Check rate limits
            if spawn_counts_this_tick[tracer_type] >= self.rate_limits[tracer_type]:
                continue
                
            # Check cooldowns
            if tracer_type in self.spawn_cooldowns:
                if tick_id - self.last_spawn_tick[tracer_type] < self.spawn_cooldowns[tracer_type]:
                    continue
            
            # Create temporary instance to check conditions and costs
            temp_tracer = tracer_class()
            
            # Check nutrient budget
            available_budget = self.get_available_budget()
            if temp_tracer.base_nutrient_cost > available_budget:
                logger.debug(f"Insufficient budget for {tracer_type.value}: need {temp_tracer.base_nutrient_cost}, have {available_budget}")
                continue
                
            # Check spawn conditions
            try:
                if temp_tracer.spawn_conditions_met(context):
                    # Budget check passed, conditions met - spawn the tracer
                    temp_tracer.spawn(tick_id, context)
                    new_tracers.append(temp_tracer)
                    spawn_counts_this_tick[tracer_type] += 1
                    self.last_spawn_tick[tracer_type] = tick_id
                    
                    # Reserve nutrients
                    if temp_tracer.base_nutrient_cost <= self.recycled_energy_pool:
                        # Use recycled energy first
                        self.recycled_energy_pool -= temp_tracer.base_nutrient_cost
                    else:
                        # Use remaining recycled energy and main budget
                        remaining_cost = temp_tracer.base_nutrient_cost - self.recycled_energy_pool
                        self.recycled_energy_pool = 0.0
                        self.current_nutrient_usage += remaining_cost
                    
                    self.metrics.update_spawn(tracer_type)
                    logger.info(f"Spawned {tracer_type.value} tracer {temp_tracer.tracer_id}")
                    
            except Exception as e:
                logger.error(f"Error evaluating spawn conditions for {tracer_type.value}: {e}")
                continue
                
        return new_tracers
        
    def process_active_tracers(self, tick_id: int, context: Dict[str, Any]) -> Tuple[List[TracerReport], List[str]]:
        """Process all active tracers and return reports and retiring tracer IDs"""
        all_reports = []
        retiring_tracers = []
        
        for tracer_id, tracer in self.active_tracers.items():
            try:
                reports = tracer.tick(tick_id, context)
                all_reports.extend(reports)
                
                if tracer.status == TracerStatus.RETIRING:
                    retiring_tracers.append(tracer_id)
                    
            except Exception as e:
                logger.error(f"Error processing tracer {tracer_id} ({tracer.tracer_type.value}): {e}")
                # Force retirement on error
                tracer.status = TracerStatus.RETIRING
                retiring_tracers.append(tracer_id)
                
        return all_reports, retiring_tracers
        
    def retire_tracers(self, tick_id: int, retiring_tracer_ids: List[str]) -> float:
        """Retire specified tracers and reclaim nutrients"""
        total_recycled = 0.0
        
        for tracer_id in retiring_tracer_ids:
            if tracer_id in self.active_tracers:
                tracer = self.active_tracers.pop(tracer_id)
                recycled_energy = tracer.retire(tick_id)
                
                # Return nutrients to budget
                self.current_nutrient_usage = max(0, self.current_nutrient_usage - tracer.current_nutrient_cost)
                self.recycled_energy_pool += recycled_energy
                total_recycled += recycled_energy
                
                self.metrics.update_retire(tracer.tracer_type)
                
        return total_recycled
        
    def apply_safeguards(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ecosystem safeguards and generate warnings"""
        safeguard_actions = {}
        
        # Tracer flood detection
        total_active = len(self.active_tracers)
        if total_active > 50:  # Threshold for flood
            safeguard_actions['tracer_flood'] = {
                'action': 'reduce_spawn_rates',
                'severity': 'warn',
                'details': f'High tracer count: {total_active}'
            }
            
        # Budget depletion warning
        budget_usage_ratio = self.current_nutrient_usage / self.nutrient_budget
        if budget_usage_ratio > 0.9:
            safeguard_actions['budget_depletion'] = {
                'action': 'emergency_retirement',
                'severity': 'critical',
                'details': f'Budget usage: {budget_usage_ratio:.1%}'
            }
            
        # Stagnant tracers (running too long without reports)
        stagnant_count = 0
        for tracer in self.active_tracers.values():
            age = tracer.get_age(context.get('tick_id', 0))
            if age > 20 and len(tracer.reports) == 0:
                stagnant_count += 1
                
        if stagnant_count > 5:
            safeguard_actions['stagnant_tracers'] = {
                'action': 'force_retirement',
                'severity': 'warn', 
                'details': f'Stagnant tracers: {stagnant_count}'
            }
            
        return safeguard_actions
        
    def tick(self, tick_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one tick of the tracer ecosystem"""
        tick_start_time = time.time()
        
        # Update active tracer counts
        self.metrics.update_active_counts(self.active_tracers)
        
        # Evaluate new spawns
        new_tracers = self.evaluate_spawning(tick_id, context)
        for tracer in new_tracers:
            self.active_tracers[tracer.tracer_id] = tracer
            
        # Process all active tracers
        all_reports, retiring_tracer_ids = self.process_active_tracers(tick_id, context)
        
        # Retire tracers and reclaim nutrients
        total_recycled = self.retire_tracers(tick_id, retiring_tracer_ids)
        
        # Apply ecosystem safeguards
        safeguard_actions = self.apply_safeguards(context)
        
        # Update metrics
        self.metrics.update_nutrient_usage(self.current_nutrient_usage)
        self.metrics.record_reports(all_reports)
        
        # Prepare tick summary
        tick_duration = time.time() - tick_start_time
        
        tick_summary = {
            'tick_id': tick_id,
            'timestamp': time.time(),
            'duration_ms': tick_duration * 1000,
            'reports': [report.to_dict() for report in all_reports],
            'ecosystem_state': {
                'active_tracers': len(self.active_tracers),
                'spawned_this_tick': len(new_tracers),
                'retired_this_tick': len(retiring_tracer_ids),
                'nutrient_usage': self.current_nutrient_usage,
                'recycled_energy_pool': self.recycled_energy_pool,
                'budget_utilization': self.current_nutrient_usage / self.nutrient_budget
            },
            'safeguard_actions': safeguard_actions,
            'metrics': self.metrics.get_summary()
        }
        
        if len(new_tracers) > 0 or len(retiring_tracer_ids) > 0:
            logger.info(f"Tick {tick_id}: spawned {len(new_tracers)}, retired {len(retiring_tracer_ids)}, "
                       f"active: {len(self.active_tracers)}, reports: {len(all_reports)}")
        
        return tick_summary
        
    def get_tracer_status(self, tracer_id: str = None) -> Dict[str, Any]:
        """Get status information for specific tracer or all tracers"""
        if tracer_id:
            if tracer_id in self.active_tracers:
                return self.active_tracers[tracer_id].get_status_info()
            else:
                return {"error": f"Tracer {tracer_id} not found"}
        else:
            return {
                tracer_id: tracer.get_status_info() 
                for tracer_id, tracer in self.active_tracers.items()
            }
            
    def force_retire_tracer(self, tracer_id: str, tick_id: int) -> bool:
        """Force retirement of a specific tracer"""
        if tracer_id in self.active_tracers:
            tracer = self.active_tracers[tracer_id]
            tracer.status = TracerStatus.RETIRING
            self.retire_tracers(tick_id, [tracer_id])
            logger.info(f"Force retired tracer {tracer_id}")
            return True
        return False
        
    def emergency_budget_recovery(self, tick_id: int) -> Dict[str, Any]:
        """Emergency procedure to recover budget by retiring expensive tracers"""
        recovery_info = {
            'tracers_retired': 0,
            'energy_recovered': 0.0,
            'actions_taken': []
        }
        
        # Retire expensive tracers first
        expensive_tracers = [
            (tid, tracer) for tid, tracer in self.active_tracers.items()
            if tracer.current_nutrient_cost > 1.0
        ]
        
        expensive_tracers.sort(key=lambda x: x[1].current_nutrient_cost, reverse=True)
        
        for tracer_id, tracer in expensive_tracers[:3]:  # Retire up to 3 expensive tracers
            tracer.status = TracerStatus.RETIRING
            energy_recovered = self.retire_tracers(tick_id, [tracer_id])
            recovery_info['tracers_retired'] += 1
            recovery_info['energy_recovered'] += energy_recovered
            recovery_info['actions_taken'].append(f"Retired {tracer.tracer_type.value} tracer {tracer_id}")
            
        logger.warning(f"Emergency budget recovery: retired {recovery_info['tracers_retired']} tracers, "
                      f"recovered {recovery_info['energy_recovered']:.3f} energy")
                      
        return recovery_info

"""
Tracer Integration Module

Integrates the tracer ecosystem with DAWN's core consciousness subsystems.
Provides context adapters, data bridges, and system hooks for seamless
operation within the broader DAWN architecture.
"""

from typing import Dict, Any, List, Optional, Callable
import logging
from .tracer_manager import TracerManager
from .base_tracer import TracerReport, AlertSeverity

logger = logging.getLogger(__name__)


class TracerSystemIntegration:
    """
    Integration layer between tracer ecosystem and DAWN consciousness systems.
    
    Responsibilities:
    - Context translation between DAWN systems and tracers
    - Event routing and notification
    - Subsystem health monitoring
    - Integration safeguards and error handling
    """
    
    def __init__(self, tracer_manager: TracerManager):
        self.tracer_manager = tracer_manager
        self.subsystem_hooks = {}
        self.context_adapters = {}
        self.alert_handlers = {}
        self.integration_metrics = {
            'context_updates': 0,
            'alerts_processed': 0,
            'integration_errors': 0,
            'subsystem_responses': 0
        }
        
    def register_subsystem_hook(self, subsystem_name: str, hook_function: Callable):
        """Register a hook function for a DAWN subsystem"""
        self.subsystem_hooks[subsystem_name] = hook_function
        logger.info(f"Registered subsystem hook: {subsystem_name}")
        
    def register_context_adapter(self, adapter_name: str, adapter_function: Callable):
        """Register a context adapter for data translation"""
        self.context_adapters[adapter_name] = adapter_function
        logger.info(f"Registered context adapter: {adapter_name}")
        
    def register_alert_handler(self, alert_type: str, handler_function: Callable):
        """Register a handler for specific alert types"""
        self.alert_handlers[alert_type] = handler_function
        logger.info(f"Registered alert handler: {alert_type}")
        
    def build_tracer_context(self, dawn_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build tracer-compatible context from DAWN's internal state.
        
        Args:
            dawn_state: Current DAWN system state
            
        Returns:
            dict: Tracer-compatible context
        """
        context = {
            'tick_id': dawn_state.get('current_tick', 0),
            'timestamp': dawn_state.get('timestamp', 0.0)
        }
        
        try:
            # Apply context adapters
            for adapter_name, adapter_func in self.context_adapters.items():
                try:
                    adapter_data = adapter_func(dawn_state)
                    context.update(adapter_data)
                except Exception as e:
                    logger.error(f"Context adapter {adapter_name} failed: {e}")
                    self.integration_metrics['integration_errors'] += 1
            
            # Add default fallbacks for required fields
            self._add_context_fallbacks(context, dawn_state)
            
            self.integration_metrics['context_updates'] += 1
            
        except Exception as e:
            logger.error(f"Failed to build tracer context: {e}")
            self.integration_metrics['integration_errors'] += 1
            # Return minimal context to prevent system failure
            context = {
                'tick_id': dawn_state.get('current_tick', 0),
                'timestamp': dawn_state.get('timestamp', 0.0),
                'error_state': True
            }
            
        return context
        
    def process_tracer_reports(self, reports: List[TracerReport], dawn_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process tracer reports and trigger appropriate system responses.
        
        Args:
            reports: List of tracer reports
            dawn_state: Current DAWN system state
            
        Returns:
            dict: Summary of actions taken
        """
        action_summary = {
            'alerts_processed': 0,
            'subsystem_actions': [],
            'critical_alerts': 0,
            'recommendations': []
        }
        
        try:
            for report in reports:
                # Process individual report
                self._process_single_report(report, dawn_state, action_summary)
                
            self.integration_metrics['alerts_processed'] += len(reports)
            
        except Exception as e:
            logger.error(f"Failed to process tracer reports: {e}")
            self.integration_metrics['integration_errors'] += 1
            
        return action_summary
        
    def _process_single_report(self, report: TracerReport, dawn_state: Dict[str, Any], 
                              action_summary: Dict[str, Any]) -> None:
        """Process a single tracer report"""
        action_summary['alerts_processed'] += 1
        
        # Count critical alerts
        if report.severity == AlertSeverity.CRITICAL:
            action_summary['critical_alerts'] += 1
            
        # Route to specific alert handlers
        report_type = report.report_type
        if report_type in self.alert_handlers:
            try:
                handler_result = self.alert_handlers[report_type](report, dawn_state)
                if handler_result:
                    action_summary['subsystem_actions'].append(handler_result)
                    self.integration_metrics['subsystem_responses'] += 1
            except Exception as e:
                logger.error(f"Alert handler {report_type} failed: {e}")
                self.integration_metrics['integration_errors'] += 1
        
        # Extract recommendations from report metadata
        recommendations = report.metadata.get('recommendations', [])
        if recommendations:
            action_summary['recommendations'].extend(recommendations)
            
        # Trigger subsystem hooks based on report content
        self._trigger_relevant_hooks(report, dawn_state, action_summary)
        
    def _trigger_relevant_hooks(self, report: TracerReport, dawn_state: Dict[str, Any],
                               action_summary: Dict[str, Any]) -> None:
        """Trigger relevant subsystem hooks based on report content"""
        
        # Map report types to subsystem hooks
        hook_mappings = {
            'bloom_anomaly': ['bloom_system', 'schema_health'],
            'soot_volatility': ['ash_soot_system', 'beetle_spawner'],
            'tension_alert': ['schema_system', 'spider_spawner'],
            'pollination': ['bee_system', 'schema_system'],
            'decay_event': ['memory_system', 'nutrient_recycler'],
            'context_scan': ['forecasting_system', 'health_monitor'],
            'long_memory_audit': ['memory_system', 'decision_system'],
            'heritage_pollination': ['schema_system', 'memory_system']
        }
        
        report_type = report.report_type
        relevant_hooks = hook_mappings.get(report_type, [])
        
        for hook_name in relevant_hooks:
            if hook_name in self.subsystem_hooks:
                try:
                    hook_result = self.subsystem_hooks[hook_name](report, dawn_state)
                    if hook_result:
                        action_summary['subsystem_actions'].append({
                            'subsystem': hook_name,
                            'action': hook_result,
                            'triggered_by': report.tracer_id
                        })
                        self.integration_metrics['subsystem_responses'] += 1
                except Exception as e:
                    logger.error(f"Subsystem hook {hook_name} failed: {e}")
                    self.integration_metrics['integration_errors'] += 1
                    
    def _add_context_fallbacks(self, context: Dict[str, Any], dawn_state: Dict[str, Any]) -> None:
        """Add fallback values for required context fields"""
        
        # Essential fields with fallbacks
        fallbacks = {
            'entropy': 0.5,
            'pressure': 0.5,
            'drift_magnitude': 0.0,
            'soot_ratio': 0.3,
            'avg_schema_coherence': 0.7,
            'memory_pressure': 0.3,
            'entropy_history': [],
            'pressure_history': [],
            'drift_history': [],
            'active_blooms': [],
            'soot_fragments': [],
            'schema_edges': [],
            'schema_clusters': [],
            'ash_fragments': [],
            'mycelial_flows': [],
            'enable_ant_spawning': True,
            'shi_variance': 0.0
        }
        
        for field, fallback_value in fallbacks.items():
            if field not in context:
                context[field] = fallback_value


class DAWNContextAdapter:
    """
    Default context adapter for translating DAWN state to tracer context.
    Provides standard mappings for common DAWN subsystems.
    """
    
    @staticmethod
    def consciousness_state_adapter(dawn_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt consciousness state data for tracers"""
        consciousness = dawn_state.get('consciousness', {})
        
        return {
            'entropy': consciousness.get('entropy_level', 0.5),
            'pressure': consciousness.get('cognitive_pressure', 0.5),
            'coherence': consciousness.get('coherence_score', 0.7),
            'awareness_level': consciousness.get('awareness_level', 0.5)
        }
    
    @staticmethod  
    def memory_system_adapter(dawn_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt memory system data for tracers"""
        memory = dawn_state.get('memory', {})
        
        return {
            'memory_pressure': memory.get('pressure', 0.3),
            'active_blooms': memory.get('active_blooms', []),
            'bloom_traces': memory.get('bloom_traces', []),
            'rebloom_events': memory.get('rebloom_events', []),
            'memory_fragmentation': memory.get('fragmentation', 0.2),
            'memory_metrics': memory.get('metrics', {})
        }
    
    @staticmethod
    def schema_system_adapter(dawn_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt schema system data for tracers"""
        schema = dawn_state.get('schema', {})
        
        return {
            'schema_clusters': schema.get('clusters', []),
            'schema_edges': schema.get('edges', []),
            'avg_schema_coherence': schema.get('avg_coherence', 0.7),
            'schema_complexity': schema.get('complexity', 0.5),
            'drift_magnitude': schema.get('drift_magnitude', 0.0),
            'drift_history': schema.get('drift_history', []),
            'schema_change_rate': schema.get('change_rate', 0.3)
        }
    
    @staticmethod
    def ash_soot_adapter(dawn_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt ash/soot system data for tracers"""
        ash_soot = dawn_state.get('ash_soot', {})
        
        return {
            'ash_fragments': ash_soot.get('ash_fragments', []),
            'soot_fragments': ash_soot.get('soot_fragments', []),
            'soot_ratio': ash_soot.get('soot_ratio', 0.3),
            'ash_ratio': ash_soot.get('ash_ratio', 0.7),
            'residual_fragments': ash_soot.get('residual_fragments', []),
            'crystallization_rate': ash_soot.get('crystallization_rate', 0.5)
        }
    
    @staticmethod
    def mycelial_adapter(dawn_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt mycelial system data for tracers"""
        mycelial = dawn_state.get('mycelial', {})
        
        return {
            'mycelial_flows': mycelial.get('flows', []),
            'nutrient_variance': mycelial.get('nutrient_variance', 0.2),
            'flow_efficiency': mycelial.get('avg_efficiency', 0.7),
            'nutrient_allocation': mycelial.get('allocation', {}),
            'cluster_isolation': mycelial.get('cluster_isolation', False)
        }
    
    @staticmethod
    def historical_adapter(dawn_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt historical data for tracers"""
        history = dawn_state.get('history', {})
        
        return {
            'entropy_history': history.get('entropy_history', []),
            'pressure_history': history.get('pressure_history', []),
            'drift_history': history.get('drift_history', []),
            'significant_events': history.get('significant_events', []),
            'epoch_history': history.get('epoch_history', []),
            'last_archival_tick': history.get('last_archival_tick', 0)
        }


class TracerAlertHandlers:
    """
    Standard alert handlers for common tracer report types.
    """
    
    @staticmethod
    def bloom_anomaly_handler(report: TracerReport, dawn_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle bloom anomaly alerts"""
        if report.severity == AlertSeverity.CRITICAL:
            return {
                'action': 'emergency_bloom_stabilization',
                'bloom_id': report.metadata.get('bloom_id'),
                'priority': 'critical'
            }
        elif report.severity == AlertSeverity.WARN:
            return {
                'action': 'bloom_monitoring_increase',
                'bloom_id': report.metadata.get('bloom_id'),
                'priority': 'medium'
            }
        return None
    
    @staticmethod
    def tension_alert_handler(report: TracerReport, dawn_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle schema tension alerts"""
        tension_level = report.metadata.get('tension', 0)
        
        if tension_level > 0.9:
            return {
                'action': 'emergency_edge_support',
                'edge_id': report.metadata.get('edge_id'),
                'priority': 'critical'
            }
        elif tension_level > 0.7:
            return {
                'action': 'reinforce_schema_edge',
                'edge_id': report.metadata.get('edge_id'),
                'priority': 'high'
            }
        return None
    
    @staticmethod
    def decay_event_handler(report: TracerReport, dawn_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle decay event reports"""
        action = report.metadata.get('action')
        
        if action == 'neutralize':
            return {
                'action': 'toxic_residue_alert',
                'residue_id': report.metadata.get('residue_id'),
                'priority': 'high'
            }
        elif action == 'recycle':
            nutrients_recovered = report.metadata.get('nutrient_recovered', 0)
            if nutrients_recovered > 0.5:
                return {
                    'action': 'nutrient_windfall',
                    'amount': nutrients_recovered,
                    'priority': 'low'
                }
        return None
    
    @staticmethod
    def context_scan_handler(report: TracerReport, dawn_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle deep context scan reports"""
        findings = report.metadata.get('findings', {})
        recommendations = report.metadata.get('recommendations', [])
        
        if recommendations:
            return {
                'action': 'implement_whale_recommendations',
                'recommendations': recommendations,
                'priority': 'medium'
            }
        return None


def create_standard_integration(tracer_manager: TracerManager) -> TracerSystemIntegration:
    """
    Create a standard tracer integration with default adapters and handlers.
    
    Args:
        tracer_manager: Tracer manager instance
        
    Returns:
        TracerSystemIntegration: Configured integration instance
    """
    integration = TracerSystemIntegration(tracer_manager)
    
    # Register standard context adapters
    integration.register_context_adapter('consciousness', DAWNContextAdapter.consciousness_state_adapter)
    integration.register_context_adapter('memory', DAWNContextAdapter.memory_system_adapter)
    integration.register_context_adapter('schema', DAWNContextAdapter.schema_system_adapter)
    integration.register_context_adapter('ash_soot', DAWNContextAdapter.ash_soot_adapter)
    integration.register_context_adapter('mycelial', DAWNContextAdapter.mycelial_adapter)
    integration.register_context_adapter('historical', DAWNContextAdapter.historical_adapter)
    
    # Register standard alert handlers
    integration.register_alert_handler('bloom_anomaly', TracerAlertHandlers.bloom_anomaly_handler)
    integration.register_alert_handler('tension_alert', TracerAlertHandlers.tension_alert_handler)
    integration.register_alert_handler('decay_event', TracerAlertHandlers.decay_event_handler)
    integration.register_alert_handler('context_scan', TracerAlertHandlers.context_scan_handler)
    
    return integration

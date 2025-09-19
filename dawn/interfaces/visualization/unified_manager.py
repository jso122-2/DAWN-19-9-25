#!/usr/bin/env python3
"""
🚀 DAWN Unified Visualization Manager
====================================

Comprehensive visualization management system that coordinates all DAWN
visualization components, data sources, and GUI integrations.

Features:
- Centralized visualization coordination
- Multi-source data aggregation
- Real-time subsystem monitoring
- GUI framework integration
- CUDA acceleration management
- Automatic data collection
- DAWN singleton integration

"The central nervous system of consciousness visualization."
"""

import logging
import threading
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import uuid
import queue
import json

# DAWN core imports
from dawn.core.singleton import get_dawn
from .cuda_matplotlib_engine import get_cuda_matplotlib_engine, VisualizationConfig
from .gui_integration import get_visualization_gui_manager, GUIVisualizationConfig

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    source_id: str
    subsystem: str
    data_type: str
    collection_interval: float = 1.0  # seconds
    enabled: bool = True
    transform_func: Optional[Callable] = None
    cache_size: int = 100


@dataclass
class VisualizationBinding:
    """Binding between data source and visualization"""
    binding_id: str
    data_source_id: str
    visualization_name: str
    gui_widget_id: Optional[str] = None
    auto_update: bool = True
    transform_func: Optional[Callable] = None


class DataCollector:
    """Collects data from DAWN subsystems for visualization"""
    
    def __init__(self):
        self.collector_id = str(uuid.uuid4())
        self.dawn = get_dawn()
        self.collection_running = False
        self.collection_thread = None
        self.data_sources: Dict[str, DataSourceConfig] = {}
        self.collected_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()
        
        logger.info(f"📊 Data Collector initialized: {self.collector_id}")
    
    def register_data_source(self, config: DataSourceConfig):
        """Register a data source for collection"""
        with self._lock:
            self.data_sources[config.source_id] = config
            logger.info(f"📈 Registered data source: {config.source_id} ({config.subsystem}.{config.data_type})")
    
    def start_collection(self):
        """Start data collection from all registered sources"""
        if self.collection_running:
            return
        
        self.collection_running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            name="dawn_data_collector",
            daemon=True
        )
        self.collection_thread.start()
        logger.info("🚀 Started data collection")
    
    def stop_collection(self):
        """Stop data collection"""
        if not self.collection_running:
            return
        
        self.collection_running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        logger.info("🛑 Stopped data collection")
    
    def _collection_loop(self):
        """Main data collection loop"""
        last_collection_times = defaultdict(float)
        
        while self.collection_running:
            try:
                current_time = time.time()
                
                for source_id, config in list(self.data_sources.items()):
                    if not config.enabled:
                        continue
                    
                    # Check if it's time to collect from this source
                    if current_time - last_collection_times[source_id] >= config.collection_interval:
                        try:
                            data = self._collect_from_source(config)
                            if data is not None:
                                # Apply transform if specified
                                if config.transform_func:
                                    data = config.transform_func(data)
                                
                                # Store collected data
                                with self._lock:
                                    self.collected_data[source_id].append({
                                        'timestamp': current_time,
                                        'data': data
                                    })
                                
                                last_collection_times[source_id] = current_time
                                
                        except Exception as e:
                            logger.error(f"Error collecting from {source_id}: {e}")
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(1.0)
    
    def _collect_from_source(self, config: DataSourceConfig) -> Optional[Dict[str, Any]]:
        """Collect data from a specific source"""
        try:
            if config.subsystem == 'tracer_ecosystem':
                return self._collect_tracer_data()
            elif config.subsystem == 'consciousness':
                return self._collect_consciousness_data()
            elif config.subsystem == 'semantic_topology':
                return self._collect_semantic_data()
            elif config.subsystem == 'self_modification':
                return self._collect_self_mod_data()
            elif config.subsystem == 'memory_system':
                return self._collect_memory_data()
            elif config.subsystem == 'telemetry':
                return self._collect_telemetry_data()
            elif config.subsystem == 'logging':
                return self._collect_logging_data()
            else:
                logger.warning(f"Unknown subsystem: {config.subsystem}")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting from {config.subsystem}: {e}")
            return None
    
    def _collect_tracer_data(self) -> Optional[Dict[str, Any]]:
        """Collect tracer ecosystem data"""
        try:
            # Try to get tracer manager from DAWN
            if hasattr(self.dawn, 'tracer_manager'):
                manager = self.dawn.tracer_manager
                
                # Get active tracers
                tracers = {}
                positions = {}
                
                for tracer_id, tracer in manager.active_tracers.items():
                    tracers[tracer_id] = {
                        'tracer_type': tracer.tracer_type.value,
                        'status': tracer.status.value,
                        'activity_level': getattr(tracer, 'activity_level', 0.5),
                        'age': tracer.get_age(0)
                    }
                    
                    # Get position from GPU if available
                    if hasattr(manager, 'cuda_tracer_engine') and manager.cuda_tracer_engine:
                        gpu_positions = manager.cuda_tracer_engine.get_tracer_positions()
                        if tracer_id in gpu_positions:
                            positions[tracer_id] = gpu_positions[tracer_id]
                        else:
                            # Default position
                            positions[tracer_id] = [0.5, 0.5, 0.5]
                    else:
                        positions[tracer_id] = [0.5, 0.5, 0.5]
                
                # Get nutrient field if available
                nutrient_field = None
                if hasattr(manager, 'cuda_tracer_engine') and manager.cuda_tracer_engine:
                    nutrient_field = manager.cuda_tracer_engine.get_nutrient_field()
                
                return {
                    'tracers': tracers,
                    'positions': positions,
                    'nutrient_field': nutrient_field,
                    'ecosystem_metrics': manager.metrics.get_summary() if hasattr(manager, 'metrics') else {}
                }
            else:
                # Generate sample data for demonstration
                return self._generate_sample_tracer_data()
                
        except Exception as e:
            logger.error(f"Error collecting tracer data: {e}")
            return self._generate_sample_tracer_data()
    
    def _collect_consciousness_data(self) -> Optional[Dict[str, Any]]:
        """Collect consciousness state data"""
        try:
            if self.dawn.is_initialized:
                # Get current consciousness state
                state = self.dawn.state if hasattr(self.dawn, 'state') else {}
                
                # Get engine metrics if available
                engine_metrics = {}
                if hasattr(self.dawn, 'dawn_engine') and self.dawn.dawn_engine:
                    engine_metrics = getattr(self.dawn.dawn_engine, 'metrics', {})
                
                return {
                    'current_state': {
                        'coherence': state.get('coherence', 0.7),
                        'unity': state.get('unity', 0.6),
                        'pressure': state.get('pressure', 0.4),
                        'entropy': state.get('entropy', 0.3),
                        'awareness': state.get('awareness', 0.8),
                        'integration': state.get('integration', 0.7)
                    },
                    'engine_metrics': engine_metrics,
                    'system_status': self.dawn.get_status() if hasattr(self.dawn, 'get_status') else {}
                }
            else:
                return self._generate_sample_consciousness_data()
                
        except Exception as e:
            logger.error(f"Error collecting consciousness data: {e}")
            return self._generate_sample_consciousness_data()
    
    def _collect_semantic_data(self) -> Optional[Dict[str, Any]]:
        """Collect semantic topology data"""
        try:
            # Try to get semantic topology engine
            semantic_engine = getattr(self.dawn, 'semantic_topology_engine', None)
            
            if semantic_engine:
                return {
                    'semantic_field': getattr(semantic_engine, 'semantic_field', None),
                    'clusters': getattr(semantic_engine, 'clusters', []),
                    'edges': getattr(semantic_engine, 'edges', []),
                    'invariants': getattr(semantic_engine, 'invariants', {}),
                    'topology_metrics': getattr(semantic_engine, 'metrics', {})
                }
            else:
                return self._generate_sample_semantic_data()
                
        except Exception as e:
            logger.error(f"Error collecting semantic data: {e}")
            return self._generate_sample_semantic_data()
    
    def _collect_self_mod_data(self) -> Optional[Dict[str, Any]]:
        """Collect self-modification data"""
        try:
            # Try to get self-modification system
            self_mod_system = getattr(self.dawn, 'self_modification_system', None)
            
            if self_mod_system:
                return {
                    'modifications': getattr(self_mod_system, 'active_modifications', []),
                    'permission_matrix': getattr(self_mod_system, 'permission_matrix', None),
                    'depth_data': getattr(self_mod_system, 'depth_tracking', []),
                    'safety_metrics': getattr(self_mod_system, 'safety_metrics', {})
                }
            else:
                return self._generate_sample_self_mod_data()
                
        except Exception as e:
            logger.error(f"Error collecting self-mod data: {e}")
            return self._generate_sample_self_mod_data()
    
    def _collect_memory_data(self) -> Optional[Dict[str, Any]]:
        """Collect memory system data"""
        try:
            # Try to get memory system
            memory_system = getattr(self.dawn, 'memory_system', None)
            
            if memory_system:
                return {
                    'memories': getattr(memory_system, 'active_memories', []),
                    'palace_structure': getattr(memory_system, 'palace_structure', {}),
                    'bloom_history': getattr(memory_system, 'bloom_history', []),
                    'active_blooms': getattr(memory_system, 'active_blooms', []),
                    'ash_history': getattr(memory_system, 'ash_history', []),
                    'soot_history': getattr(memory_system, 'soot_history', [])
                }
            else:
                return self._generate_sample_memory_data()
                
        except Exception as e:
            logger.error(f"Error collecting memory data: {e}")
            return self._generate_sample_memory_data()
    
    def _collect_telemetry_data(self) -> Optional[Dict[str, Any]]:
        """Collect telemetry data"""
        try:
            if self.dawn.telemetry_system:
                return {
                    'metrics': getattr(self.dawn.telemetry_system, 'current_metrics', {}),
                    'performance_history': getattr(self.dawn.telemetry_system, 'performance_history', []),
                    'system_stats': getattr(self.dawn.telemetry_system, 'system_stats', {}),
                    'health_indicators': getattr(self.dawn.telemetry_system, 'health_indicators', {})
                }
            else:
                return self._generate_sample_telemetry_data()
                
        except Exception as e:
            logger.error(f"Error collecting telemetry data: {e}")
            return self._generate_sample_telemetry_data()
    
    def _collect_logging_data(self) -> Optional[Dict[str, Any]]:
        """Collect logging system data"""
        try:
            # Try to get logging statistics
            return {
                'log_flows': [],  # Would be populated by actual log flow analysis
                'modules': ['consciousness', 'tracers', 'semantic', 'memory'],
                'error_history': [],
                'network_stats': {'nodes': [], 'edges': []}
            }
            
        except Exception as e:
            logger.error(f"Error collecting logging data: {e}")
            return {}
    
    # Sample data generators for testing/fallback
    def _generate_sample_tracer_data(self) -> Dict[str, Any]:
        """Generate sample tracer data for testing"""
        import numpy as np
        
        return {
            'tracers': {
                'crow_1': {'tracer_type': 'crow', 'activity_level': 0.8, 'age': 5},
                'whale_1': {'tracer_type': 'whale', 'activity_level': 0.6, 'age': 15},
                'spider_1': {'tracer_type': 'spider', 'activity_level': 0.7, 'age': 8}
            },
            'positions': {
                'crow_1': [0.3 + np.random.random()*0.1, 0.7 + np.random.random()*0.1, 0.5],
                'whale_1': [0.8 + np.random.random()*0.1, 0.2 + np.random.random()*0.1, 0.6],
                'spider_1': [0.5 + np.random.random()*0.1, 0.5 + np.random.random()*0.1, 0.4]
            },
            'nutrient_field': np.random.rand(32, 32, 32) * 0.5,
            'trails': {
                'crow_1': [[0.25, 0.65, 0.45], [0.28, 0.68, 0.48], [0.3, 0.7, 0.5]]
            }
        }
    
    def _generate_sample_consciousness_data(self) -> Dict[str, Any]:
        """Generate sample consciousness data"""
        import numpy as np
        
        return {
            'current_state': {
                'coherence': 0.7 + np.random.random()*0.2,
                'unity': 0.6 + np.random.random()*0.2,
                'pressure': 0.4 + np.random.random()*0.2,
                'entropy': 0.3 + np.random.random()*0.2,
                'awareness': 0.8 + np.random.random()*0.1,
                'integration': 0.7 + np.random.random()*0.2
            },
            'consciousness_history': [
                {'coherence': 0.7, 'unity': 0.6, 'pressure': 0.4},
                {'coherence': 0.8, 'unity': 0.7, 'pressure': 0.5},
                {'coherence': 0.6, 'unity': 0.5, 'pressure': 0.6}
            ]
        }
    
    def _generate_sample_semantic_data(self) -> Dict[str, Any]:
        """Generate sample semantic data"""
        import numpy as np
        
        return {
            'semantic_field': np.random.rand(16, 16, 16),
            'clusters': [
                {'center': [0.3, 0.7, 0.5], 'coherence': 0.8},
                {'center': [0.8, 0.2, 0.6], 'coherence': 0.6}
            ],
            'edges': [
                {'start': [0.3, 0.7, 0.5], 'end': [0.8, 0.2, 0.6], 'strength': 0.7}
            ]
        }
    
    def _generate_sample_self_mod_data(self) -> Dict[str, Any]:
        """Generate sample self-modification data"""
        import numpy as np
        
        return {
            'modifications': [
                {'id': 'mod_1', 'name': 'Behavior Update', 'status': 'approved', 'depth': 0, 'parent_id': None},
                {'id': 'mod_2', 'name': 'Safety Check', 'status': 'pending', 'depth': 1, 'parent_id': 'mod_1'}
            ],
            'permission_matrix': np.random.rand(5, 5),
            'depth_data': [
                {'depth': 0, 'angle': 0, 'type': 'structural', 'impact': 0.7},
                {'depth': 1, 'angle': 1.57, 'type': 'behavioral', 'impact': 0.5}
            ]
        }
    
    def _generate_sample_memory_data(self) -> Dict[str, Any]:
        """Generate sample memory data"""
        import numpy as np
        
        return {
            'memories': [
                {'location': [0.3, 0.7, 0.5], 'importance': 0.8, 'type': 'episodic'},
                {'location': [0.8, 0.2, 0.6], 'importance': 0.6, 'type': 'semantic'}
            ],
            'palace_structure': {
                'rooms': [
                    {'center': [0.5, 0.5, 0.5], 'size': 0.2}
                ]
            },
            'bloom_history': [
                {'total_intensity': 0.7, 'bloom_count': 3},
                {'total_intensity': 0.8, 'bloom_count': 4}
            ],
            'active_blooms': [
                {'intensity': 0.6, 'type': 'memory', 'age': 5}
            ],
            'ash_history': [0.3, 0.4, 0.5, 0.4, 0.3],
            'soot_history': [0.2, 0.3, 0.4, 0.3, 0.2]
        }
    
    def _generate_sample_telemetry_data(self) -> Dict[str, Any]:
        """Generate sample telemetry data"""
        import numpy as np
        
        return {
            'performance_history': [
                {'cpu_usage': 45, 'memory_usage': 60, 'gpu_usage': 30},
                {'cpu_usage': 50, 'memory_usage': 65, 'gpu_usage': 35}
            ],
            'health_indicators': {
                'CPU Health': 0.8,
                'Memory Health': 0.7,
                'GPU Health': 0.9,
                'Network Health': 0.6
            },
            'resource_usage': {
                'CPU': 45,
                'Memory': 60,
                'GPU': 30,
                'Network': 20
            }
        }
    
    def get_latest_data(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest data from a source"""
        with self._lock:
            if source_id in self.collected_data and self.collected_data[source_id]:
                return self.collected_data[source_id][-1]['data']
            return None
    
    def get_data_history(self, source_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical data from a source"""
        with self._lock:
            if source_id in self.collected_data:
                return list(self.collected_data[source_id])[-limit:]
            return []


class UnifiedVisualizationManager:
    """
    Unified manager for all DAWN visualization components.
    
    Coordinates data collection, visualization generation, and GUI integration
    across all DAWN subsystems.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.manager_id = str(uuid.uuid4())
        self.config = config or VisualizationConfig()
        
        # DAWN integration
        self.dawn = get_dawn()
        
        # Component managers
        self.data_collector = DataCollector()
        self.viz_engine = get_cuda_matplotlib_engine(config)
        self.gui_manager = get_visualization_gui_manager()
        
        # Visualization bindings
        self.bindings: Dict[str, VisualizationBinding] = {}
        self.active_widgets: Dict[str, Any] = {}
        
        # State
        self.running = False
        self.update_thread = None
        self._lock = threading.RLock()
        
        logger.info(f"🎨 Unified Visualization Manager initialized: {self.manager_id}")
        
        # Initialize default data sources and bindings
        self._setup_default_configuration()
    
    def _setup_default_configuration(self):
        """Set up default data sources and visualization bindings"""
        # Register default data sources
        default_sources = [
            DataSourceConfig('tracer_ecosystem', 'tracer_ecosystem', 'live_data', 0.5),
            DataSourceConfig('consciousness_state', 'consciousness', 'state_data', 1.0),
            DataSourceConfig('semantic_topology', 'semantic_topology', 'topology_data', 2.0),
            DataSourceConfig('self_modification', 'self_modification', 'mod_data', 5.0),
            DataSourceConfig('memory_system', 'memory_system', 'memory_data', 3.0),
            DataSourceConfig('telemetry_system', 'telemetry', 'metrics_data', 1.0),
            DataSourceConfig('logging_system', 'logging', 'log_data', 10.0)
        ]
        
        for source_config in default_sources:
            self.data_collector.register_data_source(source_config)
        
        # Create default visualization bindings
        default_bindings = [
            VisualizationBinding('tracer_3d', 'tracer_ecosystem', 'tracer_ecosystem_3d'),
            VisualizationBinding('tracer_interactions', 'tracer_ecosystem', 'tracer_interactions'),
            VisualizationBinding('nutrient_field', 'tracer_ecosystem', 'tracer_nutrient_field'),
            VisualizationBinding('consciousness_flow', 'consciousness_state', 'consciousness_flow'),
            VisualizationBinding('scup_metrics', 'consciousness_state', 'scup_metrics'),
            VisualizationBinding('semantic_3d', 'semantic_topology', 'semantic_topology_3d'),
            VisualizationBinding('semantic_heatmap', 'semantic_topology', 'semantic_field_heatmap'),
            VisualizationBinding('self_mod_tree', 'self_modification', 'self_mod_tree'),
            VisualizationBinding('memory_palace', 'memory_system', 'memory_palace_3d'),
            VisualizationBinding('bloom_dynamics', 'memory_system', 'bloom_dynamics'),
            VisualizationBinding('telemetry_dashboard', 'telemetry_system', 'telemetry_dashboard'),
            VisualizationBinding('system_health', 'telemetry_system', 'system_health_radar')
        ]
        
        for binding in default_bindings:
            self.bindings[binding.binding_id] = binding
        
        logger.info(f"✅ Configured {len(default_sources)} data sources and {len(default_bindings)} bindings")
    
    def start_system(self):
        """Start the unified visualization system"""
        if self.running:
            return
        
        self.running = True
        
        # Start data collection
        self.data_collector.start_collection()
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._update_loop,
            name="unified_viz_manager",
            daemon=True
        )
        self.update_thread.start()
        
        logger.info("🚀 Started unified visualization system")
    
    def stop_system(self):
        """Stop the unified visualization system"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop data collection
        self.data_collector.stop_collection()
        
        # Stop update thread
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        
        # Cleanup GUI widgets
        self.gui_manager.cleanup_widgets()
        
        logger.info("🛑 Stopped unified visualization system")
    
    def _update_loop(self):
        """Main update loop for visualization system"""
        while self.running:
            try:
                # Update all active bindings
                for binding_id, binding in list(self.bindings.items()):
                    if binding.auto_update:
                        self._update_binding(binding)
                
                # Sleep briefly
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(1.0)
    
    def _update_binding(self, binding: VisualizationBinding):
        """Update a specific visualization binding"""
        try:
            # Get latest data
            data = self.data_collector.get_latest_data(binding.data_source_id)
            if data is None:
                return
            
            # Apply transform if specified
            if binding.transform_func:
                data = binding.transform_func(data)
            
            # Update GUI widget if bound
            if binding.gui_widget_id and binding.gui_widget_id in self.active_widgets:
                widget = self.active_widgets[binding.gui_widget_id]
                if hasattr(widget, 'set_data'):
                    widget.set_data(data)
            
        except Exception as e:
            logger.error(f"Error updating binding {binding.binding_id}: {e}")
    
    def create_gui_widget(self, framework: str, parent, binding_id: str, 
                         config: Optional[GUIVisualizationConfig] = None):
        """Create a GUI widget for a visualization binding"""
        if binding_id not in self.bindings:
            logger.error(f"Binding not found: {binding_id}")
            return None
        
        binding = self.bindings[binding_id]
        
        try:
            if framework.lower() == 'tkinter':
                widget = self.gui_manager.create_tkinter_widget(
                    parent, binding.visualization_name, config
                )
            elif framework.lower() == 'qt':
                widget = self.gui_manager.create_qt_widget(
                    parent, binding.visualization_name, config
                )
            else:
                logger.error(f"Unknown framework: {framework}")
                return None
            
            if widget:
                widget_id = f"{framework}_{binding_id}_{widget.widget_id}"
                self.active_widgets[widget_id] = widget
                binding.gui_widget_id = widget_id
                
                # Set initial data
                data = self.data_collector.get_latest_data(binding.data_source_id)
                if data:
                    widget.set_data(data)
                
                logger.info(f"✅ Created GUI widget: {widget_id}")
                return widget
            
        except Exception as e:
            logger.error(f"Failed to create GUI widget: {e}")
            return None
    
    def get_available_visualizations(self) -> Dict[str, List[str]]:
        """Get available visualizations organized by category"""
        all_viz = self.viz_engine.get_available_visualizations()
        
        categories = {
            'Tracer Ecosystem': [v for v in all_viz if 'tracer' in v],
            'Consciousness': [v for v in all_viz if 'consciousness' in v or 'scup' in v or 'entropy' in v],
            'Semantic Topology': [v for v in all_viz if 'semantic' in v],
            'Self-Modification': [v for v in all_viz if 'self_mod' in v or 'recursive' in v or 'permission' in v],
            'Memory System': [v for v in all_viz if 'memory' in v or 'bloom' in v or 'ash' in v],
            'System Telemetry': [v for v in all_viz if 'telemetry' in v or 'health' in v or 'log' in v]
        }
        
        return categories
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'manager_id': self.manager_id,
            'running': self.running,
            'data_collection': {
                'active': self.data_collector.collection_running,
                'sources': len(self.data_collector.data_sources),
                'data_points': sum(len(cache) for cache in self.data_collector.collected_data.values())
            },
            'visualizations': {
                'available': len(self.viz_engine.get_available_visualizations()),
                'bindings': len(self.bindings),
                'active_widgets': len(self.active_widgets)
            },
            'gui_integration': self.gui_manager.get_manager_summary(),
            'dawn_integration': {
                'connected': self.dawn is not None,
                'initialized': self.dawn.is_initialized if self.dawn else False
            }
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration"""
        return {
            'data_sources': {
                source_id: {
                    'subsystem': config.subsystem,
                    'data_type': config.data_type,
                    'collection_interval': config.collection_interval,
                    'enabled': config.enabled
                }
                for source_id, config in self.data_collector.data_sources.items()
            },
            'bindings': {
                binding_id: {
                    'data_source_id': binding.data_source_id,
                    'visualization_name': binding.visualization_name,
                    'auto_update': binding.auto_update
                }
                for binding_id, binding in self.bindings.items()
            }
        }
    
    def import_configuration(self, config_data: Dict[str, Any]):
        """Import configuration from data"""
        try:
            # Import data sources
            if 'data_sources' in config_data:
                for source_id, source_data in config_data['data_sources'].items():
                    config = DataSourceConfig(
                        source_id=source_id,
                        subsystem=source_data['subsystem'],
                        data_type=source_data['data_type'],
                        collection_interval=source_data.get('collection_interval', 1.0),
                        enabled=source_data.get('enabled', True)
                    )
                    self.data_collector.register_data_source(config)
            
            # Import bindings
            if 'bindings' in config_data:
                for binding_id, binding_data in config_data['bindings'].items():
                    binding = VisualizationBinding(
                        binding_id=binding_id,
                        data_source_id=binding_data['data_source_id'],
                        visualization_name=binding_data['visualization_name'],
                        auto_update=binding_data.get('auto_update', True)
                    )
                    self.bindings[binding_id] = binding
            
            logger.info("✅ Configuration imported successfully")
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")


# Global manager instance
_global_unified_manager: Optional[UnifiedVisualizationManager] = None
_unified_manager_lock = threading.Lock()


def get_unified_visualization_manager(config: Optional[VisualizationConfig] = None) -> UnifiedVisualizationManager:
    """Get the global unified visualization manager instance"""
    global _global_unified_manager
    
    with _unified_manager_lock:
        if _global_unified_manager is None:
            _global_unified_manager = UnifiedVisualizationManager(config)
    
    return _global_unified_manager


def reset_unified_visualization_manager():
    """Reset the global unified visualization manager"""
    global _global_unified_manager
    
    with _unified_manager_lock:
        if _global_unified_manager:
            _global_unified_manager.stop_system()
        _global_unified_manager = None


if __name__ == "__main__":
    # Demo the unified visualization manager
    logging.basicConfig(level=logging.INFO)
    
    print("🚀" * 50)
    print("🧠 DAWN UNIFIED VISUALIZATION MANAGER DEMO")
    print("🚀" * 50)
    
    # Create manager
    manager = get_unified_visualization_manager()
    
    # Show system status
    status = manager.get_system_status()
    print(f"✅ System Status: {json.dumps(status, indent=2)}")
    
    # Show available visualizations
    available_viz = manager.get_available_visualizations()
    print(f"\n🎨 Available Visualizations:")
    for category, visualizations in available_viz.items():
        print(f"   {category}: {len(visualizations)} visualizations")
        for viz in visualizations[:3]:  # Show first 3
            print(f"     - {viz}")
        if len(visualizations) > 3:
            print(f"     ... and {len(visualizations) - 3} more")
    
    # Start system
    print("\n🚀 Starting unified visualization system...")
    manager.start_system()
    
    # Let it run for a few seconds
    print("⏱️  Running for 10 seconds...")
    time.sleep(10)
    
    # Show updated status
    status = manager.get_system_status()
    print(f"\n📊 Updated Status:")
    print(f"   Data Collection Active: {status['data_collection']['active']}")
    print(f"   Data Points Collected: {status['data_collection']['data_points']}")
    print(f"   Active Widgets: {status['visualizations']['active_widgets']}")
    
    # Stop system
    print("\n🛑 Stopping system...")
    manager.stop_system()
    
    print("\n🚀 Unified Visualization Manager demo complete!")

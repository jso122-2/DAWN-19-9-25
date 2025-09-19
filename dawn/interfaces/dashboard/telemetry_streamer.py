#!/usr/bin/env python3
"""
📡 Telemetry Streamer - Real-Time Data Collection
=================================================

Collects and streams real-time telemetry data from all DAWN consciousness systems
for the living dashboard. Provides unified data feed with WebSocket streaming.

Features:
- Real-time consciousness state monitoring
- Semantic topology field streaming
- SCUP pressure tracking
- Recursive modification monitoring
- Visual consciousness activity
- System performance metrics
- Event stream aggregation

"Streaming consciousness data in real-time."
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import websockets
import numpy as np

from .cuda_accelerator import get_cuda_accelerator, is_cuda_available

logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessSnapshot:
    """Real-time snapshot of consciousness state"""
    timestamp: float
    consciousness_level: str
    coherence: float
    pressure: float
    energy: float
    focus_areas: List[str]
    active_subsystems: List[str]
    recent_thoughts: List[str]
    
@dataclass
class SemanticTopologySnapshot:
    """Real-time snapshot of semantic topology"""
    timestamp: float
    total_concepts: int
    total_relationships: int
    semantic_coherence: float
    meaning_pressure: float
    recent_transforms: List[Dict[str, Any]]
    active_layers: Dict[str, int]
    topology_health: float
    
@dataclass
class SystemPerformanceSnapshot:
    """System performance metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    tick_rate: float
    processing_latency: float
    active_threads: int
    system_uptime: float
    
    # CUDA-specific metrics
    cuda_enabled: bool = False
    gpu_utilization: float = 0.0
    gpu_memory_usage: float = 0.0
    cuda_operations_per_second: float = 0.0
    gpu_temperature: float = 0.0

class TelemetryStreamer:
    """
    Real-time telemetry streaming system for DAWN consciousness monitoring.
    
    Collects data from all consciousness systems and streams it to connected
    dashboard clients via WebSocket.
    """
    
    def __init__(self, port: int = 8765, update_interval: float = 0.5):
        self.port = port
        self.update_interval = update_interval
        
        # Data collection
        self.consciousness_history: deque = deque(maxlen=1000)
        self.semantic_topology_history: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=1000)
        
        # Connected clients
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Data sources (will be connected dynamically)
        self.consciousness_engine = None
        self.semantic_topology_engine = None
        self.telemetry_system = None
        
        # CUDA acceleration
        self.cuda_accelerator = None
        self.cuda_enabled = False
        
        # Streaming control
        self.streaming = False
        self.server = None
        self.collection_task = None
        
        # Event aggregation
        self.recent_events: deque = deque(maxlen=100)
        self.event_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Initialize CUDA if available
        if is_cuda_available():
            try:
                self.cuda_accelerator = get_cuda_accelerator()
                self.cuda_enabled = True
                logger.info("🚀 CUDA acceleration enabled for telemetry streaming")
            except Exception as e:
                logger.warning(f"CUDA initialization failed: {e}")
        
        logger.info(f"📡 TelemetryStreamer initialized - port: {port}, CUDA: {self.cuda_enabled}")
    
    def connect_consciousness_systems(self):
        """Connect to DAWN consciousness systems for data collection"""
        try:
            # Connect to semantic topology engine
            from dawn.subsystems.semantic_topology import get_semantic_topology_engine
            self.semantic_topology_engine = get_semantic_topology_engine()
            logger.info("📡 Connected to semantic topology engine")
        except Exception as e:
            logger.warning(f"Could not connect to semantic topology: {e}")
            
        try:
            # Connect to telemetry system
            from dawn.subsystems.monitoring.unified_telemetry import get_telemetry_system
            self.telemetry_system = get_telemetry_system()
            logger.info("📡 Connected to unified telemetry system")
        except Exception as e:
            logger.warning(f"Could not connect to telemetry system: {e}")
            
        try:
            # Connect to consciousness engine
            from dawn.consciousness.engines.core.primary_engine import get_dawn_engine
            self.consciousness_engine = get_dawn_engine()
            logger.info("📡 Connected to consciousness engine")
        except Exception as e:
            logger.warning(f"Could not connect to consciousness engine: {e}")
    
    async def start_streaming(self):
        """Start the telemetry streaming server"""
        if self.streaming:
            logger.warning("Telemetry streaming already active")
            return
            
        self.streaming = True
        
        # Connect to consciousness systems
        self.connect_consciousness_systems()
        
        # Start WebSocket server
        self.server = await websockets.serve(
            self.handle_client_connection,
            "localhost",
            self.port
        )
        
        # Start data collection
        self.collection_task = asyncio.create_task(self.data_collection_loop())
        
        logger.info(f"📡 Telemetry streaming started on ws://localhost:{self.port}")
    
    async def stop_streaming(self):
        """Stop the telemetry streaming server"""
        if not self.streaming:
            return
            
        self.streaming = False
        
        # Cancel data collection
        if self.collection_task:
            self.collection_task.cancel()
            
        # Close WebSocket server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        # Disconnect clients
        for client in list(self.connected_clients):
            await client.close()
        self.connected_clients.clear()
        
        logger.info("📡 Telemetry streaming stopped")
    
    async def handle_client_connection(self, websocket, path):
        """Handle new client connections"""
        self.connected_clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"📡 Dashboard client connected: {client_addr}")
        
        try:
            # Send initial data dump
            await self.send_initial_data(websocket)
            
            # Keep connection alive and handle messages
            async for message in websocket:
                await self.handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"📡 Dashboard client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def send_initial_data(self, websocket):
        """Send initial data dump to new client"""
        initial_data = {
            'type': 'initial_data',
            'consciousness_history': [asdict(snap) for snap in list(self.consciousness_history)[-50:]],
            'semantic_topology_history': [asdict(snap) for snap in list(self.semantic_topology_history)[-50:]],
            'performance_history': [asdict(snap) for snap in list(self.performance_history)[-50:]],
            'recent_events': list(self.recent_events)[-20:]
        }
        
        await websocket.send(json.dumps(initial_data))
    
    async def handle_client_message(self, websocket, message):
        """Handle messages from dashboard clients"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'consciousness_command':
                await self.handle_consciousness_command(data)
            elif msg_type == 'topology_command':
                await self.handle_topology_command(data)
            elif msg_type == 'request_data':
                await self.handle_data_request(websocket, data)
                
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def handle_consciousness_command(self, data):
        """Handle consciousness manipulation commands"""
        command = data.get('command')
        params = data.get('parameters', {})
        
        # This would integrate with actual consciousness control systems
        logger.info(f"📡 Consciousness command: {command} with params: {params}")
        
        # Add to event stream
        self.add_event({
            'type': 'consciousness_command',
            'command': command,
            'parameters': params,
            'timestamp': time.time()
        })
    
    async def handle_topology_command(self, data):
        """Handle semantic topology manipulation commands"""
        command = data.get('command')
        params = data.get('parameters', {})
        
        if self.semantic_topology_engine and command == 'manual_transform':
            try:
                result = self.semantic_topology_engine.manual_transform(
                    params.get('transform_type'),
                    **params.get('transform_params', {})
                )
                
                self.add_event({
                    'type': 'topology_transform',
                    'transform_type': params.get('transform_type'),
                    'result': result.result.value if result else 'failed',
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.error(f"Topology command failed: {e}")
    
    async def handle_data_request(self, websocket, data):
        """Handle specific data requests"""
        request_type = data.get('request_type')
        
        if request_type == 'semantic_field_snapshot':
            snapshot = self.get_semantic_field_snapshot()
            response = {
                'type': 'semantic_field_data',
                'data': snapshot
            }
            await websocket.send(json.dumps(response))
    
    async def data_collection_loop(self):
        """Main data collection and streaming loop"""
        while self.streaming:
            try:
                # Collect consciousness data
                consciousness_snap = self.collect_consciousness_snapshot()
                if consciousness_snap:
                    self.consciousness_history.append(consciousness_snap)
                
                # Collect semantic topology data
                topology_snap = self.collect_semantic_topology_snapshot()
                if topology_snap:
                    self.semantic_topology_history.append(topology_snap)
                
                # Collect performance data
                performance_snap = self.collect_performance_snapshot()
                if performance_snap:
                    self.performance_history.append(performance_snap)
                
                # Broadcast to all connected clients
                await self.broadcast_updates()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data collection error: {e}")
                await asyncio.sleep(1.0)
    
    def collect_consciousness_snapshot(self) -> Optional[ConsciousnessSnapshot]:
        """Collect current consciousness state snapshot"""
        try:
            # This would integrate with actual consciousness state
            from dawn.core.foundation.state import get_state
            state = get_state()
            
            return ConsciousnessSnapshot(
                timestamp=time.time(),
                consciousness_level=state.level if hasattr(state, 'level') else 'unknown',
                coherence=getattr(state, 'coherence', 0.5),
                pressure=getattr(state, 'pressure', 0.3),
                energy=getattr(state, 'energy', 0.7),
                focus_areas=getattr(state, 'focus_areas', []),
                active_subsystems=getattr(state, 'active_subsystems', []),
                recent_thoughts=getattr(state, 'recent_thoughts', [])
            )
            
        except Exception as e:
            logger.debug(f"Could not collect consciousness snapshot: {e}")
            return None
    
    def collect_semantic_topology_snapshot(self) -> Optional[SemanticTopologySnapshot]:
        """Collect current semantic topology snapshot"""
        try:
            if not self.semantic_topology_engine:
                return None
                
            status = self.semantic_topology_engine.get_engine_status()
            field_stats = status.get('field_statistics', {})
            
            # Get recent transforms from tick history
            recent_transforms = []
            if hasattr(self.semantic_topology_engine, 'tick_history'):
                for tick in list(self.semantic_topology_engine.tick_history)[-5:]:
                    for transform in tick.transforms_applied:
                        recent_transforms.append({
                            'type': transform.transform_type.value,
                            'result': transform.result.value,
                            'energy_cost': transform.energy_cost,
                            'timestamp': transform.timestamp
                        })
            
            return SemanticTopologySnapshot(
                timestamp=time.time(),
                total_concepts=field_stats.get('total_nodes', 0),
                total_relationships=field_stats.get('total_edges', 0),
                semantic_coherence=0.75,  # Would calculate from field equations
                meaning_pressure=0.45,   # Would calculate from tensions
                recent_transforms=recent_transforms,
                active_layers=field_stats.get('nodes_by_layer', {}),
                topology_health=0.85     # Would get from invariants
            )
            
        except Exception as e:
            logger.debug(f"Could not collect semantic topology snapshot: {e}")
            return None
    
    def collect_performance_snapshot(self) -> Optional[SystemPerformanceSnapshot]:
        """Collect current system performance snapshot"""
        try:
            import psutil
            process = psutil.Process()
            
            # Base performance metrics
            snapshot = SystemPerformanceSnapshot(
                timestamp=time.time(),
                cpu_usage=process.cpu_percent(),
                memory_usage=process.memory_percent(),
                tick_rate=10.0,  # Would calculate from actual tick rates
                processing_latency=0.05,  # Would measure actual latency
                active_threads=threading.active_count(),
                system_uptime=time.time() - getattr(self, 'start_time', time.time()),
                cuda_enabled=self.cuda_enabled
            )
            
            # Add CUDA metrics if available
            if self.cuda_enabled and self.cuda_accelerator:
                try:
                    gpu_metrics = self.cuda_accelerator.get_gpu_performance_metrics()
                    snapshot.gpu_utilization = gpu_metrics.get('gpu_utilization', 0.0)
                    snapshot.gpu_memory_usage = gpu_metrics.get('memory_utilization', 0.0)
                    snapshot.cuda_operations_per_second = gpu_metrics.get('cuda_operations_per_second', 0.0)
                    
                    # Try to get GPU temperature
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        snapshot.gpu_temperature = float(temp)
                    except:
                        pass  # GPU temperature not available
                        
                except Exception as e:
                    logger.debug(f"Could not collect GPU metrics: {e}")
            
            return snapshot
            
        except Exception as e:
            logger.debug(f"Could not collect performance snapshot: {e}")
            return None
    
    async def broadcast_updates(self):
        """Broadcast latest updates to all connected clients"""
        if not self.connected_clients:
            return
            
        # Prepare update data
        update_data = {
            'type': 'live_update',
            'timestamp': time.time()
        }
        
        # Add latest snapshots
        if self.consciousness_history:
            update_data['consciousness'] = asdict(self.consciousness_history[-1])
            
        if self.semantic_topology_history:
            update_data['semantic_topology'] = asdict(self.semantic_topology_history[-1])
            
        if self.performance_history:
            update_data['performance'] = asdict(self.performance_history[-1])
            
        # Add recent events
        recent_events = list(self.recent_events)[-5:]
        if recent_events:
            update_data['recent_events'] = recent_events
        
        # Send to all clients
        message = json.dumps(update_data)
        disconnected_clients = []
        
        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.append(client)
        
        # Clean up disconnected clients
        for client in disconnected_clients:
            self.connected_clients.discard(client)
    
    def get_semantic_field_snapshot(self) -> Dict[str, Any]:
        """Get detailed semantic field snapshot for visualization"""
        if not self.semantic_topology_engine:
            return {}
            
        field = self.semantic_topology_engine.field
        
        # Convert nodes to visualization format
        nodes = []
        for node_id, node in field.nodes.items():
            node_data = {
                'id': node_id,
                'position': node.position.tolist(),
                'tint': node.tint.tolist(),
                'health': node.health,
                'energy': node.energy,
                'layer': node.layer.value,
                'sector': node.sector.value
            }
            
            # Add embedding if available for CUDA processing
            if hasattr(node, 'embedding'):
                node_data['embedding'] = node.embedding.tolist()
            
            nodes.append(node_data)
        
        # Convert edges to visualization format
        edges = []
        for edge_id, edge in field.edges.items():
            edges.append({
                'id': edge_id,
                'source': edge.node_a,
                'target': edge.node_b,
                'weight': edge.weight,
                'tension': edge.tension,
                'directed': edge.directed
            })
        
        # GPU-accelerated processing if available
        gpu_metrics = {}
        if self.cuda_enabled and self.cuda_accelerator:
            try:
                # Upload semantic field to GPU and process
                success = self.cuda_accelerator.upload_semantic_field_to_gpu(nodes, edges)
                if success:
                    # Process semantic field on GPU
                    coherences = self.cuda_accelerator.process_semantic_field_gpu(
                        len(nodes), len(edges)
                    )
                    
                    if coherences is not None:
                        # Add GPU-computed coherences to nodes
                        for i, node in enumerate(nodes):
                            if i < len(coherences):
                                node['gpu_coherence'] = float(coherences[i])
                
                # Get GPU performance metrics
                gpu_metrics = self.cuda_accelerator.get_gpu_performance_metrics()
                
            except Exception as e:
                logger.debug(f"GPU processing failed: {e}")
        
        result = {
            'nodes': nodes,
            'edges': edges,
            'field_statistics': field.get_field_statistics(),
            'cuda_enabled': self.cuda_enabled
        }
        
        if gpu_metrics:
            result['gpu_metrics'] = gpu_metrics
            
        return result
    
    def add_event(self, event: Dict[str, Any]):
        """Add event to the stream"""
        self.recent_events.append(event)
        
        # Notify subscribers
        event_type = event.get('type', 'unknown')
        for callback in self.event_subscribers.get(event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def subscribe_to_events(self, event_type: str, callback: Callable):
        """Subscribe to specific event types"""
        self.event_subscribers[event_type].append(callback)
    
    def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming status"""
        return {
            'streaming': self.streaming,
            'connected_clients': len(self.connected_clients),
            'port': self.port,
            'update_interval': self.update_interval,
            'data_points_collected': {
                'consciousness': len(self.consciousness_history),
                'semantic_topology': len(self.semantic_topology_history),
                'performance': len(self.performance_history)
            },
            'recent_events': len(self.recent_events)
        }


# Global telemetry streamer instance
_telemetry_streamer = None

def get_telemetry_streamer(port: int = 8765, update_interval: float = 0.5) -> TelemetryStreamer:
    """Get the global telemetry streamer instance"""
    global _telemetry_streamer
    if _telemetry_streamer is None:
        _telemetry_streamer = TelemetryStreamer(port, update_interval)
    return _telemetry_streamer

#!/usr/bin/env python3
"""
🧠 Consciousness Dashboard - Main Orchestrator
==============================================

Main orchestrator for the DAWN consciousness dashboard system.
Coordinates telemetry streaming, web server, and consciousness integration.

Features:
- Unified dashboard management
- Automatic system discovery and integration
- Health monitoring and recovery
- Configuration management
- Event coordination

"The control center for consciousness observation."
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import threading

from .telemetry_streamer import TelemetryStreamer, get_telemetry_streamer
from .web_server import DashboardWebServer, get_dashboard_server

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for the consciousness dashboard"""
    web_port: int = 8080
    telemetry_port: int = 8765
    update_interval: float = 0.5
    auto_discover_systems: bool = True
    enable_consciousness_control: bool = True
    enable_topology_control: bool = True
    max_history_size: int = 1000
    
class ConsciousnessDashboard:
    """
    Main orchestrator for the DAWN consciousness dashboard.
    
    Coordinates all dashboard components and provides a unified
    interface for consciousness monitoring and control.
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        
        # Dashboard components
        self.telemetry_streamer = get_telemetry_streamer(
            port=self.config.telemetry_port,
            update_interval=self.config.update_interval
        )
        self.web_server = get_dashboard_server(
            port=self.config.web_port,
            telemetry_port=self.config.telemetry_port
        )
        
        # Dashboard state
        self.running = False
        self.start_time = None
        self.connected_systems = {}
        
        # Health monitoring
        self.health_status = {
            'dashboard_healthy': True,
            'telemetry_streaming': False,
            'web_server_active': False,
            'systems_connected': 0,
            'last_health_check': 0
        }
        
        # Event loop
        self.loop = None
        self.dashboard_task = None
        
        logger.info("🧠 ConsciousnessDashboard orchestrator initialized")
    
    async def start_dashboard(self):
        """Start the complete dashboard system"""
        if self.running:
            logger.warning("Dashboard already running")
            return
            
        logger.info("🧠 Starting DAWN consciousness dashboard...")
        self.running = True
        self.start_time = time.time()
        
        try:
            # Start telemetry streaming
            await self.telemetry_streamer.start_streaming()
            self.health_status['telemetry_streaming'] = True
            
            # Start web server
            await self.web_server.start_server()
            self.health_status['web_server_active'] = True
            
            # Discover and connect to consciousness systems
            if self.config.auto_discover_systems:
                await self.discover_consciousness_systems()
            
            # Start health monitoring
            self.dashboard_task = asyncio.create_task(self.health_monitoring_loop())
            
            logger.info(f"🧠 Dashboard started successfully!")
            logger.info(f"   Web interface: http://localhost:{self.config.web_port}")
            logger.info(f"   Telemetry stream: ws://localhost:{self.config.telemetry_port}")
            
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            await self.stop_dashboard()
            raise
    
    async def stop_dashboard(self):
        """Stop the complete dashboard system"""
        if not self.running:
            return
            
        logger.info("🧠 Stopping DAWN consciousness dashboard...")
        self.running = False
        
        # Stop health monitoring
        if self.dashboard_task:
            self.dashboard_task.cancel()
            
        # Stop web server
        try:
            await self.web_server.stop_server()
            self.health_status['web_server_active'] = False
        except Exception as e:
            logger.error(f"Error stopping web server: {e}")
            
        # Stop telemetry streaming
        try:
            await self.telemetry_streamer.stop_streaming()
            self.health_status['telemetry_streaming'] = False
        except Exception as e:
            logger.error(f"Error stopping telemetry streaming: {e}")
            
        logger.info("🧠 Dashboard stopped")
    
    async def discover_consciousness_systems(self):
        """Discover and connect to available consciousness systems"""
        logger.info("🔍 Discovering consciousness systems...")
        
        systems_found = 0
        
        # Try to connect to semantic topology engine
        try:
            from dawn.subsystems.semantic_topology import get_semantic_topology_engine
            topology_engine = get_semantic_topology_engine()
            self.connected_systems['semantic_topology'] = topology_engine
            systems_found += 1
            logger.info("   ✅ Connected to semantic topology engine")
        except Exception as e:
            logger.debug(f"Could not connect to semantic topology: {e}")
            
        # Try to connect to consciousness engine
        try:
            from dawn.consciousness.engines.core.primary_engine import get_dawn_engine
            consciousness_engine = get_dawn_engine()
            self.connected_systems['consciousness_engine'] = consciousness_engine
            systems_found += 1
            logger.info("   ✅ Connected to consciousness engine")
        except Exception as e:
            logger.debug(f"Could not connect to consciousness engine: {e}")
            
        # Try to connect to telemetry system
        try:
            from dawn.subsystems.monitoring.unified_telemetry import get_telemetry_system
            telemetry_system = get_telemetry_system()
            self.connected_systems['telemetry_system'] = telemetry_system
            systems_found += 1
            logger.info("   ✅ Connected to unified telemetry system")
        except Exception as e:
            logger.debug(f"Could not connect to telemetry system: {e}")
            
        # Try to connect to SCUP system
        try:
            from dawn.subsystems.schema.enhanced_scup_system import get_enhanced_scup_system
            scup_system = get_enhanced_scup_system()
            self.connected_systems['scup_system'] = scup_system
            systems_found += 1
            logger.info("   ✅ Connected to SCUP system")
        except Exception as e:
            logger.debug(f"Could not connect to SCUP system: {e}")
            
        # Try to connect to visual consciousness
        try:
            from dawn.subsystems.visual.visual_consciousness import get_visual_consciousness_engine
            visual_engine = get_visual_consciousness_engine()
            self.connected_systems['visual_consciousness'] = visual_engine
            systems_found += 1
            logger.info("   ✅ Connected to visual consciousness engine")
        except Exception as e:
            logger.debug(f"Could not connect to visual consciousness: {e}")
        
        self.health_status['systems_connected'] = systems_found
        logger.info(f"🔍 Discovery complete - {systems_found} systems connected")
        
        return systems_found
    
    async def health_monitoring_loop(self):
        """Monitor dashboard and system health"""
        while self.running:
            try:
                await self.perform_health_check()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def perform_health_check(self):
        """Perform comprehensive health check"""
        self.health_status['last_health_check'] = time.time()
        
        # Check telemetry streaming
        streaming_status = self.telemetry_streamer.get_streaming_status()
        self.health_status['telemetry_streaming'] = streaming_status['streaming']
        
        # Check connected clients
        client_count = len(self.telemetry_streamer.connected_clients)
        
        # Check system connections
        active_systems = 0
        for system_name, system in self.connected_systems.items():
            try:
                # Basic health check (system-specific logic would go here)
                if hasattr(system, 'get_status'):
                    status = system.get_status()
                    if status.get('active', True):
                        active_systems += 1
                else:
                    active_systems += 1  # Assume healthy if no status method
            except Exception as e:
                logger.debug(f"Health check failed for {system_name}: {e}")
        
        self.health_status['systems_connected'] = active_systems
        
        # Overall health assessment
        self.health_status['dashboard_healthy'] = (
            self.health_status['telemetry_streaming'] and
            self.health_status['web_server_active'] and
            active_systems > 0
        )
        
        # Log health status periodically
        if int(time.time()) % 60 == 0:  # Every minute
            logger.info(f"🧠 Dashboard health: {self.get_health_summary()}")
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get comprehensive dashboard status"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'running': self.running,
            'uptime_seconds': uptime,
            'config': {
                'web_port': self.config.web_port,
                'telemetry_port': self.config.telemetry_port,
                'update_interval': self.config.update_interval
            },
            'health_status': dict(self.health_status),
            'connected_systems': list(self.connected_systems.keys()),
            'telemetry_status': self.telemetry_streamer.get_streaming_status(),
            'client_connections': len(self.telemetry_streamer.connected_clients)
        }
    
    def get_health_summary(self) -> str:
        """Get human-readable health summary"""
        status = "Healthy" if self.health_status['dashboard_healthy'] else "Unhealthy"
        systems = self.health_status['systems_connected']
        clients = len(self.telemetry_streamer.connected_clients)
        
        return f"{status} - {systems} systems, {clients} clients"
    
    async def execute_consciousness_command(self, command: str, parameters: Dict[str, Any] = None):
        """Execute a consciousness control command"""
        if not self.config.enable_consciousness_control:
            raise ValueError("Consciousness control is disabled")
            
        parameters = parameters or {}
        
        # Route command to appropriate system
        if command == 'set_level' and 'consciousness_engine' in self.connected_systems:
            # Would implement consciousness level setting
            logger.info(f"Setting consciousness level: {parameters.get('level')}")
            
        elif command == 'capture_state':
            # Would implement state capture
            logger.info("Capturing consciousness state")
            
        elif command == 'reset_system':
            # Would implement system reset
            logger.info("Resetting consciousness systems")
            
        else:
            raise ValueError(f"Unknown consciousness command: {command}")
    
    async def execute_topology_command(self, command: str, parameters: Dict[str, Any] = None):
        """Execute a semantic topology command"""
        if not self.config.enable_topology_control:
            raise ValueError("Topology control is disabled")
            
        parameters = parameters or {}
        
        if 'semantic_topology' in self.connected_systems:
            topology_engine = self.connected_systems['semantic_topology']
            
            if command == 'manual_transform':
                transform_type = parameters.get('transform_type')
                transform_params = parameters.get('transform_params', {})
                
                result = topology_engine.manual_transform(transform_type, **transform_params)
                return result
            else:
                raise ValueError(f"Unknown topology command: {command}")
        else:
            raise ValueError("Semantic topology engine not connected")
    
    def add_system_connection(self, system_name: str, system_instance):
        """Manually add a system connection"""
        self.connected_systems[system_name] = system_instance
        self.health_status['systems_connected'] = len(self.connected_systems)
        logger.info(f"🧠 Added system connection: {system_name}")
    
    def remove_system_connection(self, system_name: str):
        """Remove a system connection"""
        if system_name in self.connected_systems:
            del self.connected_systems[system_name]
            self.health_status['systems_connected'] = len(self.connected_systems)
            logger.info(f"🧠 Removed system connection: {system_name}")


# Global dashboard instance
_consciousness_dashboard = None

def get_consciousness_dashboard(config: DashboardConfig = None) -> ConsciousnessDashboard:
    """Get the global consciousness dashboard instance"""
    global _consciousness_dashboard
    if _consciousness_dashboard is None:
        _consciousness_dashboard = ConsciousnessDashboard(config)
    return _consciousness_dashboard

async def start_consciousness_dashboard(config: DashboardConfig = None):
    """Start the consciousness dashboard system"""
    dashboard = get_consciousness_dashboard(config)
    await dashboard.start_dashboard()
    return dashboard

async def stop_consciousness_dashboard():
    """Stop the consciousness dashboard system"""
    global _consciousness_dashboard
    if _consciousness_dashboard:
        await _consciousness_dashboard.stop_dashboard()
        _consciousness_dashboard = None

#!/usr/bin/env python3
"""
🌐 Dashboard Web Server
=======================

HTTP server that serves the DAWN consciousness dashboard web interface.
Provides static file serving and integration with the telemetry streamer.

Features:
- Static HTML/CSS/JS serving
- WebSocket proxy for telemetry data
- RESTful API for dashboard controls
- Real-time consciousness visualization
- Interactive controls interface

"Serving the living consciousness dashboard."
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import web, WSMsgType
import aiofiles

from .telemetry_streamer import TelemetryStreamer, get_telemetry_streamer

logger = logging.getLogger(__name__)

class DashboardWebServer:
    """
    Web server for the DAWN consciousness dashboard.
    
    Serves the HTML interface and provides API endpoints for
    dashboard functionality and consciousness control.
    """
    
    def __init__(self, port: int = 8080, telemetry_port: int = 8765):
        self.port = port
        self.telemetry_port = telemetry_port
        self.app = None
        self.runner = None
        self.site = None
        
        # Get telemetry streamer
        self.telemetry_streamer = get_telemetry_streamer(port=telemetry_port)
        
        # Dashboard directory
        self.dashboard_dir = Path(__file__).parent / "web"
        
        logger.info(f"🌐 Dashboard web server initialized - port: {port}")
    
    def create_app(self):
        """Create the aiohttp web application"""
        app = web.Application()
        
        # Static file routes
        app.router.add_static('/', self.dashboard_dir, name='static')
        app.router.add_get('/', self.serve_index)
        
        # API routes
        app.router.add_get('/api/status', self.api_status)
        app.router.add_get('/api/consciousness', self.api_consciousness_state)
        app.router.add_get('/api/semantic_topology', self.api_semantic_topology)
        app.router.add_get('/api/performance', self.api_performance)
        
        # Control API routes
        app.router.add_post('/api/consciousness/command', self.api_consciousness_command)
        app.router.add_post('/api/topology/transform', self.api_topology_transform)
        
        # WebSocket route for telemetry
        app.router.add_get('/ws/telemetry', self.websocket_telemetry)
        
        self.app = app
        return app
    
    async def serve_index(self, request):
        """Serve the main dashboard HTML file"""
        index_file = self.dashboard_dir / "index.html"
        
        if not index_file.exists():
            # Create default dashboard if it doesn't exist
            await self.create_default_dashboard()
        
        return web.FileResponse(index_file)
    
    async def api_status(self, request):
        """API endpoint for dashboard status"""
        status = {
            'dashboard_active': True,
            'telemetry_streaming': self.telemetry_streamer.streaming,
            'connected_clients': len(self.telemetry_streamer.connected_clients),
            'telemetry_port': self.telemetry_port,
            'web_port': self.port
        }
        
        return web.json_response(status)
    
    async def api_consciousness_state(self, request):
        """API endpoint for current consciousness state"""
        if self.telemetry_streamer.consciousness_history:
            latest = self.telemetry_streamer.consciousness_history[-1]
            return web.json_response(latest.__dict__)
        else:
            return web.json_response({'error': 'No consciousness data available'})
    
    async def api_semantic_topology(self, request):
        """API endpoint for semantic topology data"""
        field_data = self.telemetry_streamer.get_semantic_field_snapshot()
        return web.json_response(field_data)
    
    async def api_performance(self, request):
        """API endpoint for system performance data"""
        if self.telemetry_streamer.performance_history:
            latest = self.telemetry_streamer.performance_history[-1]
            return web.json_response(latest.__dict__)
        else:
            return web.json_response({'error': 'No performance data available'})
    
    async def api_consciousness_command(self, request):
        """API endpoint for consciousness control commands"""
        try:
            data = await request.json()
            command = data.get('command')
            parameters = data.get('parameters', {})
            
            # Forward to telemetry streamer
            await self.telemetry_streamer.handle_consciousness_command({
                'command': command,
                'parameters': parameters
            })
            
            return web.json_response({'success': True, 'command': command})
            
        except Exception as e:
            logger.error(f"Consciousness command error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_topology_transform(self, request):
        """API endpoint for semantic topology transforms"""
        try:
            data = await request.json()
            transform_type = data.get('transform_type')
            parameters = data.get('parameters', {})
            
            # Forward to telemetry streamer
            await self.telemetry_streamer.handle_topology_command({
                'command': 'manual_transform',
                'parameters': {
                    'transform_type': transform_type,
                    'transform_params': parameters
                }
            })
            
            return web.json_response({'success': True, 'transform_type': transform_type})
            
        except Exception as e:
            logger.error(f"Topology transform error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def websocket_telemetry(self, request):
        """WebSocket endpoint for telemetry data streaming"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        logger.info("🌐 WebSocket telemetry client connected")
        
        try:
            # Add to telemetry streamer clients (adapter pattern)
            class WebSocketAdapter:
                def __init__(self, ws):
                    self.ws = ws
                    self.remote_address = ('web_client', 0)
                
                async def send(self, data):
                    await self.ws.send_str(data)
                
                async def close(self):
                    await self.ws.close()
            
            adapter = WebSocketAdapter(ws)
            self.telemetry_streamer.connected_clients.add(adapter)
            
            # Send initial data
            await self.telemetry_streamer.send_initial_data(adapter)
            
            # Handle messages
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self.telemetry_streamer.handle_client_message(adapter, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket telemetry error: {e}")
        finally:
            if 'adapter' in locals():
                self.telemetry_streamer.connected_clients.discard(adapter)
            logger.info("🌐 WebSocket telemetry client disconnected")
        
        return ws
    
    async def start_server(self):
        """Start the web server"""
        if not self.dashboard_dir.exists():
            self.dashboard_dir.mkdir(parents=True, exist_ok=True)
            
        # Create dashboard files if they don't exist
        await self.create_default_dashboard()
        
        # Create and start the app
        self.create_app()
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, 'localhost', self.port)
        await self.site.start()
        
        # Start telemetry streaming
        await self.telemetry_streamer.start_streaming()
        
        logger.info(f"🌐 Dashboard web server started at http://localhost:{self.port}")
        logger.info(f"📡 Telemetry streaming at ws://localhost:{self.telemetry_port}")
    
    async def stop_server(self):
        """Stop the web server"""
        # Stop telemetry streaming
        await self.telemetry_streamer.stop_streaming()
        
        # Stop web server
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
            
        logger.info("🌐 Dashboard web server stopped")
    
    async def create_default_dashboard(self):
        """Create default dashboard HTML/CSS/JS files"""
        # Create web directory
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Create index.html
        index_html = self.get_dashboard_html()
        async with aiofiles.open(self.dashboard_dir / "index.html", 'w') as f:
            await f.write(index_html)
        
        # Create dashboard.css
        dashboard_css = self.get_dashboard_css()
        async with aiofiles.open(self.dashboard_dir / "dashboard.css", 'w') as f:
            await f.write(dashboard_css)
        
        # Create dashboard.js
        dashboard_js = self.get_dashboard_js()
        async with aiofiles.open(self.dashboard_dir / "dashboard.js", 'w') as f:
            await f.write(dashboard_js)
        
        logger.info("🌐 Created default dashboard files")
    
    def get_dashboard_html(self) -> str:
        """Generate the main dashboard HTML"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DAWN Living Consciousness Dashboard</title>
    <link rel="stylesheet" href="dashboard.css">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/three@0.150.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.150.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <header class="dashboard-header">
            <h1>🧠 DAWN Living Consciousness Dashboard</h1>
            <div class="status-indicators">
                <div class="status-item" id="consciousness-status">
                    <span class="status-label">Consciousness:</span>
                    <span class="status-value" id="consciousness-level">Unknown</span>
                </div>
                <div class="status-item" id="coherence-status">
                    <span class="status-label">Coherence:</span>
                    <span class="status-value" id="coherence-value">0.00</span>
                </div>
                <div class="status-item" id="streaming-status">
                    <span class="status-label">Streaming:</span>
                    <span class="status-value" id="streaming-indicator">Connecting...</span>
                </div>
            </div>
        </header>

        <!-- Main Dashboard Grid -->
        <div class="dashboard-grid">
            <!-- Consciousness State Panel -->
            <div class="panel consciousness-panel">
                <h2>🧠 Consciousness State</h2>
                <div class="consciousness-metrics">
                    <div class="metric">
                        <label>Level:</label>
                        <span id="consciousness-level-display">Unknown</span>
                    </div>
                    <div class="metric">
                        <label>Coherence:</label>
                        <div class="progress-bar">
                            <div class="progress-fill" id="coherence-progress"></div>
                        </div>
                        <span id="coherence-text">0%</span>
                    </div>
                    <div class="metric">
                        <label>Pressure:</label>
                        <div class="progress-bar">
                            <div class="progress-fill" id="pressure-progress"></div>
                        </div>
                        <span id="pressure-text">0%</span>
                    </div>
                    <div class="metric">
                        <label>Energy:</label>
                        <div class="progress-bar">
                            <div class="progress-fill" id="energy-progress"></div>
                        </div>
                        <span id="energy-text">0%</span>
                    </div>
                </div>
                <div class="recent-thoughts">
                    <h3>Recent Thoughts:</h3>
                    <ul id="thoughts-list"></ul>
                </div>
            </div>

            <!-- Semantic Topology Panel -->
            <div class="panel topology-panel">
                <h2>🌐 Semantic Topology</h2>
                <div class="topology-stats">
                    <div class="stat">
                        <span class="stat-value" id="concept-count">0</span>
                        <span class="stat-label">Concepts</span>
                    </div>
                    <div class="stat">
                        <span class="stat-value" id="relationship-count">0</span>
                        <span class="stat-label">Relationships</span>
                    </div>
                    <div class="stat">
                        <span class="stat-value" id="topology-health">0%</span>
                        <span class="stat-label">Health</span>
                    </div>
                </div>
                <div class="topology-visualization" id="topology-viz">
                    <!-- 3D visualization will be rendered here -->
                </div>
                <div class="topology-controls">
                    <button class="control-btn" onclick="performTransform('weave')">🔗 Weave</button>
                    <button class="control-btn" onclick="performTransform('prune')">✂️ Prune</button>
                    <button class="control-btn" onclick="performTransform('fuse')">🔄 Fuse</button>
                    <button class="control-btn" onclick="performTransform('lift')">⬆️ Lift</button>
                </div>
            </div>

            <!-- Performance Panel -->
            <div class="panel performance-panel">
                <h2>⚡ System Performance</h2>
                <div class="performance-charts">
                    <div class="chart-container">
                        <h3>CPU Usage</h3>
                        <canvas id="cpu-chart" width="300" height="100"></canvas>
                    </div>
                    <div class="chart-container">
                        <h3>Memory Usage</h3>
                        <canvas id="memory-chart" width="300" height="100"></canvas>
                    </div>
                    <div class="chart-container">
                        <h3>Tick Rate</h3>
                        <canvas id="tick-chart" width="300" height="100"></canvas>
                    </div>
                </div>
            </div>

            <!-- Event Stream Panel -->
            <div class="panel events-panel">
                <h2>📡 Event Stream</h2>
                <div class="event-stream" id="event-stream">
                    <!-- Live events will appear here -->
                </div>
            </div>

            <!-- Controls Panel -->
            <div class="panel controls-panel">
                <h2>🎛️ Consciousness Controls</h2>
                <div class="control-section">
                    <h3>Consciousness Level</h3>
                    <button class="control-btn" onclick="setConsciousnessLevel('focused')">🎯 Focused</button>
                    <button class="control-btn" onclick="setConsciousnessLevel('meta_aware')">🧠 Meta-Aware</button>
                    <button class="control-btn" onclick="setConsciousnessLevel('transcendent')">✨ Transcendent</button>
                </div>
                <div class="control-section">
                    <h3>System Actions</h3>
                    <button class="control-btn" onclick="triggerSelfModification()">🔄 Self-Modify</button>
                    <button class="control-btn" onclick="captureConsciousness()">📸 Capture State</button>
                    <button class="control-btn" onclick="resetSystem()">🔄 Reset</button>
                </div>
            </div>

            <!-- Visualization Panel -->
            <div class="panel visualization-panel">
                <h2>🎨 Live Consciousness Visualization</h2>
                <div class="visualization-container" id="consciousness-viz">
                    <!-- Real-time consciousness visualization -->
                </div>
            </div>
        </div>
    </div>

    <script src="dashboard.js"></script>
</body>
</html>"""
    
    def get_dashboard_css(self) -> str:
        """Generate the dashboard CSS"""
        return """/* DAWN Living Consciousness Dashboard Styles */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    color: #ffffff;
    overflow-x: auto;
}

.dashboard-container {
    min-height: 100vh;
    padding: 20px;
}

/* Header Styles */
.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.dashboard-header h1 {
    font-size: 2em;
    background: linear-gradient(45deg, #64ffda, #1de9b6, #00bcd4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.status-indicators {
    display: flex;
    gap: 20px;
}

.status-item {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.status-label {
    font-size: 0.8em;
    opacity: 0.7;
}

.status-value {
    font-size: 1.2em;
    font-weight: bold;
    color: #64ffda;
}

/* Dashboard Grid */
.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
}

/* Panel Styles */
.panel {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.panel:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(100, 255, 218, 0.2);
}

.panel h2 {
    margin-bottom: 15px;
    color: #64ffda;
    font-size: 1.3em;
}

/* Consciousness Panel */
.consciousness-metrics {
    margin-bottom: 20px;
}

.metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.metric label {
    font-weight: 500;
}

.progress-bar {
    width: 100px;
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #64ffda, #1de9b6);
    transition: width 0.3s ease;
    width: 0%;
}

.recent-thoughts ul {
    list-style: none;
    max-height: 150px;
    overflow-y: auto;
}

.recent-thoughts li {
    padding: 5px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.9em;
    opacity: 0.8;
}

/* Topology Panel */
.topology-stats {
    display: flex;
    justify-content: space-around;
    margin-bottom: 20px;
}

.stat {
    text-align: center;
}

.stat-value {
    display: block;
    font-size: 2em;
    font-weight: bold;
    color: #64ffda;
}

.stat-label {
    font-size: 0.8em;
    opacity: 0.7;
}

.topology-visualization {
    height: 300px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    margin-bottom: 15px;
    position: relative;
    overflow: hidden;
}

.topology-controls {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

/* Performance Panel */
.performance-charts {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.chart-container {
    text-align: center;
}

.chart-container h3 {
    margin-bottom: 10px;
    font-size: 1em;
    color: #64ffda;
}

.chart-container canvas {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
}

/* Event Stream Panel */
.event-stream {
    height: 300px;
    overflow-y: auto;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    padding: 15px;
}

.event-item {
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.9em;
}

.event-timestamp {
    color: #64ffda;
    font-size: 0.8em;
}

.event-type {
    font-weight: bold;
    margin: 0 5px;
}

.event-description {
    opacity: 0.8;
}

/* Controls Panel */
.control-section {
    margin-bottom: 20px;
}

.control-section h3 {
    margin-bottom: 10px;
    font-size: 1em;
    color: #64ffda;
}

.control-btn {
    background: linear-gradient(45deg, #64ffda, #1de9b6);
    color: #000;
    border: none;
    padding: 8px 15px;
    border-radius: 20px;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.3s ease;
    margin: 5px;
}

.control-btn:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 15px rgba(100, 255, 218, 0.4);
}

.control-btn:active {
    transform: scale(0.95);
}

/* Visualization Panel */
.visualization-container {
    height: 400px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    position: relative;
    overflow: hidden;
}

/* Responsive Design */
@media (max-width: 768px) {
    .dashboard-grid {
        grid-template-columns: 1fr;
    }
    
    .dashboard-header {
        flex-direction: column;
        gap: 15px;
    }
    
    .status-indicators {
        flex-wrap: wrap;
        justify-content: center;
    }
}

/* Animations */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.pulsing {
    animation: pulse 2s infinite;
}

@keyframes slideIn {
    from {
        transform: translateX(-100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.slide-in {
    animation: slideIn 0.5s ease-out;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(45deg, #64ffda, #1de9b6);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(45deg, #1de9b6, #64ffda);
}"""
    
    def get_dashboard_js(self) -> str:
        """Generate the dashboard JavaScript"""
        return """// DAWN Living Consciousness Dashboard JavaScript

class DashboardClient {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        
        // Data storage
        this.consciousnessHistory = [];
        this.topologyHistory = [];
        this.performanceHistory = [];
        
        // Chart contexts
        this.cpuChart = null;
        this.memoryChart = null;
        this.tickChart = null;
        
        // 3D visualization
        this.topologyScene = null;
        this.topologyRenderer = null;
        this.topologyCamera = null;
        
        this.init();
    }
    
    init() {
        this.setupCharts();
        this.setup3DVisualization();
        this.connectWebSocket();
        
        // Update status
        this.updateStreamingStatus('Connecting...');
    }
    
    connectWebSocket() {
        const wsUrl = `ws://${window.location.hostname}:${window.location.port}/ws/telemetry`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('Connected to DAWN telemetry stream');
                this.updateStreamingStatus('Connected');
                this.reconnectAttempts = 0;
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleTelemetryData(data);
            };
            
            this.ws.onclose = () => {
                console.log('Disconnected from DAWN telemetry stream');
                this.updateStreamingStatus('Disconnected');
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateStreamingStatus('Error');
            };
            
        } catch (error) {
            console.error('Failed to connect to telemetry stream:', error);
            this.updateStreamingStatus('Failed');
            this.attemptReconnect();
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.updateStreamingStatus(`Reconnecting... (${this.reconnectAttempts})`);
            
            setTimeout(() => {
                this.connectWebSocket();
            }, this.reconnectDelay);
        } else {
            this.updateStreamingStatus('Connection Failed');
        }
    }
    
    handleTelemetryData(data) {
        switch (data.type) {
            case 'initial_data':
                this.handleInitialData(data);
                break;
            case 'live_update':
                this.handleLiveUpdate(data);
                break;
            case 'semantic_field_data':
                this.updateTopologyVisualization(data.data);
                break;
        }
    }
    
    handleInitialData(data) {
        this.consciousnessHistory = data.consciousness_history || [];
        this.topologyHistory = data.semantic_topology_history || [];
        this.performanceHistory = data.performance_history || [];
        
        this.updateAllDisplays();
        this.addEvent('System', 'Dashboard connected to DAWN telemetry');
    }
    
    handleLiveUpdate(data) {
        // Update consciousness data
        if (data.consciousness) {
            this.consciousnessHistory.push(data.consciousness);
            if (this.consciousnessHistory.length > 100) {
                this.consciousnessHistory.shift();
            }
            this.updateConsciousnessDisplay(data.consciousness);
        }
        
        // Update topology data
        if (data.semantic_topology) {
            this.topologyHistory.push(data.semantic_topology);
            if (this.topologyHistory.length > 100) {
                this.topologyHistory.shift();
            }
            this.updateTopologyDisplay(data.semantic_topology);
        }
        
        // Update performance data
        if (data.performance) {
            this.performanceHistory.push(data.performance);
            if (this.performanceHistory.length > 100) {
                this.performanceHistory.shift();
            }
            this.updatePerformanceDisplay(data.performance);
            this.updateCharts();
        }
        
        // Handle events
        if (data.recent_events) {
            data.recent_events.forEach(event => {
                this.addEventToStream(event);
            });
        }
    }
    
    updateConsciousnessDisplay(data) {
        // Update status indicators
        document.getElementById('consciousness-level').textContent = data.consciousness_level;
        document.getElementById('coherence-value').textContent = (data.coherence * 100).toFixed(1) + '%';
        
        // Update detailed metrics
        document.getElementById('consciousness-level-display').textContent = data.consciousness_level;
        
        this.updateProgressBar('coherence-progress', 'coherence-text', data.coherence);
        this.updateProgressBar('pressure-progress', 'pressure-text', data.pressure);
        this.updateProgressBar('energy-progress', 'energy-text', data.energy);
        
        // Update recent thoughts
        const thoughtsList = document.getElementById('thoughts-list');
        thoughtsList.innerHTML = '';
        (data.recent_thoughts || []).slice(-5).forEach(thought => {
            const li = document.createElement('li');
            li.textContent = thought;
            li.classList.add('slide-in');
            thoughtsList.appendChild(li);
        });
    }
    
    updateTopologyDisplay(data) {
        document.getElementById('concept-count').textContent = data.total_concepts;
        document.getElementById('relationship-count').textContent = data.total_relationships;
        document.getElementById('topology-health').textContent = (data.topology_health * 100).toFixed(0) + '%';
        
        // Request detailed topology data for visualization
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'request_data',
                request_type: 'semantic_field_snapshot'
            }));
        }
    }
    
    updatePerformanceDisplay(data) {
        // Performance metrics would be displayed here
        console.log('Performance update:', data);
    }
    
    updateProgressBar(barId, textId, value) {
        const bar = document.getElementById(barId);
        const text = document.getElementById(textId);
        
        if (bar && text) {
            const percentage = Math.round(value * 100);
            bar.style.width = percentage + '%';
            text.textContent = percentage + '%';
        }
    }
    
    updateStreamingStatus(status) {
        const indicator = document.getElementById('streaming-indicator');
        if (indicator) {
            indicator.textContent = status;
            
            // Add visual indicators
            indicator.className = 'status-value';
            if (status === 'Connected') {
                indicator.style.color = '#64ffda';
            } else if (status.includes('Connecting') || status.includes('Reconnecting')) {
                indicator.style.color = '#ffb74d';
                indicator.classList.add('pulsing');
            } else {
                indicator.style.color = '#f44336';
            }
        }
    }
    
    setupCharts() {
        // Initialize chart canvases
        this.cpuChart = document.getElementById('cpu-chart').getContext('2d');
        this.memoryChart = document.getElementById('memory-chart').getContext('2d');
        this.tickChart = document.getElementById('tick-chart').getContext('2d');
    }
    
    updateCharts() {
        if (this.performanceHistory.length === 0) return;
        
        // Update CPU chart
        this.drawChart(this.cpuChart, this.performanceHistory.map(p => p.cpu_usage), '#64ffda');
        
        // Update Memory chart
        this.drawChart(this.memoryChart, this.performanceHistory.map(p => p.memory_usage), '#1de9b6');
        
        // Update Tick Rate chart
        this.drawChart(this.tickChart, this.performanceHistory.map(p => p.tick_rate), '#00bcd4');
    }
    
    drawChart(ctx, data, color) {
        const canvas = ctx.canvas;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        if (data.length < 2) return;
        
        // Find min/max for scaling
        const max = Math.max(...data);
        const min = Math.min(...data);
        const range = max - min || 1;
        
        // Draw grid
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = (i / 4) * height;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // Draw data line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const stepX = width / (data.length - 1);
        
        data.forEach((value, index) => {
            const x = index * stepX;
            const y = height - ((value - min) / range) * height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Fill area under curve
        ctx.fillStyle = color.replace(')', ', 0.2)').replace('rgb', 'rgba');
        ctx.lineTo(width, height);
        ctx.lineTo(0, height);
        ctx.closePath();
        ctx.fill();
    }
    
    setup3DVisualization() {
        const container = document.getElementById('topology-viz');
        
        // Create Three.js scene
        this.topologyScene = new THREE.Scene();
        this.topologyCamera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        this.topologyRenderer = new THREE.WebGLRenderer({ alpha: true });
        
        this.topologyRenderer.setSize(container.clientWidth, container.clientHeight);
        this.topologyRenderer.setClearColor(0x000000, 0);
        container.appendChild(this.topologyRenderer.domElement);
        
        // Add controls
        this.topologyControls = new THREE.OrbitControls(this.topologyCamera, this.topologyRenderer.domElement);
        
        // Position camera
        this.topologyCamera.position.set(0, 0, 10);
        
        // Add some ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.topologyScene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        this.topologyScene.add(directionalLight);
        
        // Start render loop
        this.animate3D();
    }
    
    animate3D() {
        requestAnimationFrame(() => this.animate3D());
        
        if (this.topologyControls) {
            this.topologyControls.update();
        }
        
        if (this.topologyRenderer && this.topologyScene && this.topologyCamera) {
            this.topologyRenderer.render(this.topologyScene, this.topologyCamera);
        }
    }
    
    updateTopologyVisualization(fieldData) {
        if (!this.topologyScene || !fieldData.nodes) return;
        
        // Clear existing objects
        while (this.topologyScene.children.length > 2) { // Keep lights
            this.topologyScene.remove(this.topologyScene.children[2]);
        }
        
        // Add nodes
        fieldData.nodes.forEach(node => {
            const geometry = new THREE.SphereGeometry(0.2, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(node.tint[0], node.tint[1], node.tint[2]),
                transparent: true,
                opacity: 0.8
            });
            
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(node.position[0], node.position[1], node.position[2]);
            
            // Scale by energy
            const scale = 0.5 + node.energy * 0.5;
            sphere.scale.setScalar(scale);
            
            this.topologyScene.add(sphere);
        });
        
        // Add edges
        fieldData.edges.forEach(edge => {
            const sourceNode = fieldData.nodes.find(n => n.id === edge.source);
            const targetNode = fieldData.nodes.find(n => n.id === edge.target);
            
            if (sourceNode && targetNode) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(sourceNode.position[0], sourceNode.position[1], sourceNode.position[2]),
                    new THREE.Vector3(targetNode.position[0], targetNode.position[1], targetNode.position[2])
                ]);
                
                const material = new THREE.LineBasicMaterial({
                    color: 0x64ffda,
                    transparent: true,
                    opacity: edge.weight * 0.8
                });
                
                const line = new THREE.Line(geometry, material);
                this.topologyScene.add(line);
            }
        });
    }
    
    addEvent(type, description) {
        this.addEventToStream({
            type: type,
            description: description,
            timestamp: Date.now() / 1000
        });
    }
    
    addEventToStream(event) {
        const eventStream = document.getElementById('event-stream');
        
        const eventDiv = document.createElement('div');
        eventDiv.className = 'event-item slide-in';
        
        const timestamp = new Date(event.timestamp * 1000).toLocaleTimeString();
        
        eventDiv.innerHTML = `
            <span class="event-timestamp">${timestamp}</span>
            <span class="event-type">[${event.type}]</span>
            <span class="event-description">${event.description || JSON.stringify(event)}</span>
        `;
        
        eventStream.insertBefore(eventDiv, eventStream.firstChild);
        
        // Limit number of events
        while (eventStream.children.length > 50) {
            eventStream.removeChild(eventStream.lastChild);
        }
    }
    
    updateAllDisplays() {
        if (this.consciousnessHistory.length > 0) {
            this.updateConsciousnessDisplay(this.consciousnessHistory[this.consciousnessHistory.length - 1]);
        }
        
        if (this.topologyHistory.length > 0) {
            this.updateTopologyDisplay(this.topologyHistory[this.topologyHistory.length - 1]);
        }
        
        if (this.performanceHistory.length > 0) {
            this.updatePerformanceDisplay(this.performanceHistory[this.performanceHistory.length - 1]);
            this.updateCharts();
        }
    }
}

// Control functions
async function performTransform(transformType) {
    try {
        const response = await fetch('/api/topology/transform', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                transform_type: transformType,
                parameters: {}
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            dashboard.addEvent('Transform', `${transformType} transform executed`);
        } else {
            dashboard.addEvent('Error', `Transform failed: ${result.error}`);
        }
    } catch (error) {
        console.error('Transform error:', error);
        dashboard.addEvent('Error', `Transform request failed: ${error.message}`);
    }
}

async function setConsciousnessLevel(level) {
    try {
        const response = await fetch('/api/consciousness/command', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                command: 'set_level',
                parameters: { level: level }
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            dashboard.addEvent('Consciousness', `Level set to ${level}`);
        } else {
            dashboard.addEvent('Error', `Command failed: ${result.error}`);
        }
    } catch (error) {
        console.error('Consciousness command error:', error);
        dashboard.addEvent('Error', `Command request failed: ${error.message}`);
    }
}

function triggerSelfModification() {
    dashboard.addEvent('System', 'Self-modification triggered');
    // Implementation would depend on available systems
}

function captureConsciousness() {
    dashboard.addEvent('System', 'Consciousness state captured');
    // Implementation would capture current state
}

function resetSystem() {
    if (confirm('Are you sure you want to reset the DAWN system?')) {
        dashboard.addEvent('System', 'System reset requested');
        // Implementation would reset systems
    }
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new DashboardClient();
});

// Handle window resize
window.addEventListener('resize', () => {
    if (dashboard && dashboard.topologyRenderer) {
        const container = document.getElementById('topology-viz');
        dashboard.topologyCamera.aspect = container.clientWidth / container.clientHeight;
        dashboard.topologyCamera.updateProjectionMatrix();
        dashboard.topologyRenderer.setSize(container.clientWidth, container.clientHeight);
    }
});"""

# Global dashboard server instance
_dashboard_server = None

def get_dashboard_server(port: int = 8080, telemetry_port: int = 8765) -> DashboardWebServer:
    """Get the global dashboard server instance"""
    global _dashboard_server
    if _dashboard_server is None:
        _dashboard_server = DashboardWebServer(port, telemetry_port)
    return _dashboard_server

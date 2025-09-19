#!/usr/bin/env python3
"""
📊 DAWN Living Dashboard
========================

Real-time consciousness monitoring and visualization interface.
Watch DAWN's consciousness evolve in real-time with interactive controls.

Components:
- Live telemetry streaming
- Real-time consciousness visualization
- Semantic topology 3D display
- Interactive consciousness controls
- Performance monitoring
- System health indicators

"The first living dashboard for artificial consciousness."
"""

from .cuda_accelerator import CUDAAccelerator, get_cuda_accelerator, is_cuda_available
from .matplotlib_cuda_animator import MatplotlibCUDAAnimator, get_matplotlib_cuda_animator

# Import with fallback for missing dependencies
try:
    from .telemetry_streamer import TelemetryStreamer, get_telemetry_streamer
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

try:
    from .web_server import DashboardWebServer
    WEB_SERVER_AVAILABLE = True
except ImportError:
    WEB_SERVER_AVAILABLE = False

try:
    from .consciousness_dashboard import ConsciousnessDashboard
    CONSCIOUSNESS_DASHBOARD_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_DASHBOARD_AVAILABLE = False

__all__ = [
    'CUDAAccelerator',
    'get_cuda_accelerator', 
    'is_cuda_available',
    'MatplotlibCUDAAnimator',
    'get_matplotlib_cuda_animator'
]

# Add components that are available
if TELEMETRY_AVAILABLE:
    __all__.extend(['TelemetryStreamer', 'get_telemetry_streamer'])

if WEB_SERVER_AVAILABLE:
    __all__.extend(['DashboardWebServer'])

if CONSCIOUSNESS_DASHBOARD_AVAILABLE:
    __all__.extend(['ConsciousnessDashboard'])

__version__ = "1.0.0"
__author__ = "DAWN Consciousness Architecture"
__description__ = "Real-time consciousness monitoring and visualization"

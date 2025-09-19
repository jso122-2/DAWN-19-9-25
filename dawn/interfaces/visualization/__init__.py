#!/usr/bin/env python3
"""
🚀 DAWN Visualization Interfaces
===============================

Comprehensive visualization system for DAWN consciousness with CUDA acceleration
and GUI integration capabilities.

Features:
- CUDA-accelerated matplotlib visualizations
- GUI-callable visualization components
- Real-time data processing and rendering
- Modular visualization architecture
- DAWN singleton integration

"Visualizing consciousness at the speed of light."
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Core visualization engine
try:
    from .cuda_matplotlib_engine import (
        CUDAMatplotlibEngine,
        VisualizationConfig,
        VisualizationData,
        get_cuda_matplotlib_engine,
        reset_cuda_matplotlib_engine
    )
    CUDA_MATPLOTLIB_AVAILABLE = True
    logger.info("✅ CUDA Matplotlib Engine available")
except ImportError as e:
    CUDA_MATPLOTLIB_AVAILABLE = False
    logger.warning(f"❌ CUDA Matplotlib Engine not available: {e}")

# CUDA House Animations
try:
    from .cuda_house_animations import (
        CUDAHouseAnimator,
        MycelialHouseAnimator,
        SchemaHouseAnimator,
        MonitoringHouseAnimator,
        HouseAnimationManager,
        AnimationConfig,
        get_house_animation_manager,
        reset_house_animation_manager
    )
    CUDA_HOUSE_ANIMATIONS_AVAILABLE = True
    logger.info("✅ CUDA House Animations available")
except ImportError as e:
    CUDA_HOUSE_ANIMATIONS_AVAILABLE = False
    logger.warning(f"❌ CUDA House Animations not available: {e}")

# GUI Integration
try:
    from .gui_integration import (
        get_visualization_gui_manager,
        create_visualization_widget,
        GUIVisualizationConfig
    )
    GUI_INTEGRATION_AVAILABLE = True
    logger.info("✅ GUI Integration available")
except ImportError as e:
    GUI_INTEGRATION_AVAILABLE = False
    logger.warning(f"❌ GUI Integration not available: {e}")

# Unified Manager
try:
    from .unified_manager import (
        get_unified_visualization_manager,
        reset_unified_visualization_manager
    )
    UNIFIED_MANAGER_AVAILABLE = True
    logger.info("✅ Unified Manager available")
except ImportError as e:
    UNIFIED_MANAGER_AVAILABLE = False
    logger.warning(f"❌ Unified Manager not available: {e}")

# Visualization utilities
def create_visualization_for_gui(viz_name: str, data: Dict[str, Any], 
                                gui_parent=None, config: Optional[VisualizationConfig] = None):
    """
    Create a visualization that can be embedded in GUI systems.
    
    Args:
        viz_name: Name of the visualization to create
        data: Data to visualize
        gui_parent: GUI parent widget (for Tkinter integration)
        config: Optional visualization configuration
        
    Returns:
        Visualization canvas or figure that can be embedded in GUI
    """
    if not CUDA_MATPLOTLIB_AVAILABLE:
        logger.error("CUDA Matplotlib Engine not available")
        return None
    
    try:
        engine = get_cuda_matplotlib_engine(config)
        return engine.create_gui_callable_visualization(viz_name, data, gui_parent)
    except Exception as e:
        logger.error(f"Failed to create GUI visualization: {e}")
        return None

def get_available_visualizations() -> list:
    """Get list of available visualization types"""
    if not CUDA_MATPLOTLIB_AVAILABLE:
        return []
    
    try:
        engine = get_cuda_matplotlib_engine()
        return engine.get_available_visualizations()
    except Exception as e:
        logger.error(f"Failed to get available visualizations: {e}")
        return []

def queue_data_for_visualization(subsystem: str, data_type: str, data: Any, 
                                metadata: Optional[Dict[str, Any]] = None):
    """Queue data for background processing and visualization"""
    if not CUDA_MATPLOTLIB_AVAILABLE:
        logger.warning("CUDA Matplotlib Engine not available - data not queued")
        return
    
    try:
        engine = get_cuda_matplotlib_engine()
        engine.queue_data_for_processing(subsystem, data_type, data, metadata)
    except Exception as e:
        logger.error(f"Failed to queue data for visualization: {e}")

# Export main components
__all__ = [
    'create_visualization_for_gui',
    'get_available_visualizations',
    'queue_data_for_visualization',
    'CUDA_MATPLOTLIB_AVAILABLE',
    'CUDA_HOUSE_ANIMATIONS_AVAILABLE',
    'GUI_INTEGRATION_AVAILABLE',
    'UNIFIED_MANAGER_AVAILABLE'
]

# Add conditional exports based on availability
if CUDA_MATPLOTLIB_AVAILABLE:
    __all__.extend([
        'CUDAMatplotlibEngine',
        'VisualizationConfig',
        'VisualizationData',
        'get_cuda_matplotlib_engine',
        'reset_cuda_matplotlib_engine'
    ])

if CUDA_HOUSE_ANIMATIONS_AVAILABLE:
    __all__.extend([
        'CUDAHouseAnimator',
        'MycelialHouseAnimator',
        'SchemaHouseAnimator',
        'MonitoringHouseAnimator',
        'HouseAnimationManager',
        'AnimationConfig',
        'get_house_animation_manager',
        'reset_house_animation_manager'
    ])

if GUI_INTEGRATION_AVAILABLE:
    __all__.extend([
        'get_visualization_gui_manager',
        'create_visualization_widget',
        'GUIVisualizationConfig'
    ])

if UNIFIED_MANAGER_AVAILABLE:
    __all__.extend([
        'get_unified_visualization_manager',
        'reset_unified_visualization_manager'
    ])

logger.info(f"🚀 DAWN Visualization Interfaces initialized - {len(__all__)} components available")

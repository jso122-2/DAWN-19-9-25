#!/usr/bin/env python3
"""
Test script for DAWN Mycelial Hashmap & Tracer Movement Visualization

Runs the mycelial tracer visualization with realistic DAWN system integration.
This demonstrates the living substrate of DAWN's cognition under simulated load.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add DAWN to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mycelial_tracer_viz.log')
        ]
    )

def main():
    """Run the mycelial tracer visualization test"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🍄 Starting DAWN Mycelial Hashmap & Tracer Visualization Test")
    
    try:
        # Import the visualization
        from dawn.subsystems.visual.mycelial_tracer_visualization import (
            MycelialTracerVisualizer, 
            MycelialVisualizationConfig
        )
        
        # Create configuration for testing
        config = MycelialVisualizationConfig(
            figure_size=(20, 14),
            max_nodes=100,  # Smaller for testing
            max_tracers=15,
            update_interval=0.1,  # 10 FPS
        )
        
        logger.info("Configuration created:")
        logger.info(f"  Max nodes: {config.max_nodes}")
        logger.info(f"  Max tracers: {config.max_tracers}")
        logger.info(f"  Update interval: {config.update_interval}s")
        
        # Create visualizer
        logger.info("Creating mycelial tracer visualizer...")
        visualizer = MycelialTracerVisualizer(config)
        
        logger.info("Visualizer created successfully")
        logger.info("Starting real-time visualization...")
        logger.info("Press Ctrl+C to stop")
        
        # Start the visualization
        visualizer.start_visualization()
        
    except KeyboardInterrupt:
        logger.info("🛑 Visualization stopped by user")
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure all DAWN dependencies are available")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

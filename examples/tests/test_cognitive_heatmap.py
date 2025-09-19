#!/usr/bin/env python3
"""
Test Runner for DAWN Cognitive Load Heat Map Visualization
==========================================================

Test script to run the cognitive load heatmap visualization showing
real-time brain activity heat across different cognitive regions
with system monitoring and performance metrics.
"""

import logging
import sys
import os

# Add DAWN to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main test function"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("⚡ Starting DAWN Cognitive Load Heat Map Test")
    
    try:
        from dawn.subsystems.visual.cognitive_load_heatmap import CognitiveLoadHeatmapVisualizer
        
        print("⚡ Creating Cognitive Load Heat Map Visualization")
        print("   - Real-time brain activity heat visualization")
        print("   - Different cognitive regions (attention, memory, reasoning, etc.)")
        print("   - Heat intensity = processing load")
        print("   - System performance metrics and health monitoring")
        print("   - Load timeline and efficiency tracking")
        print("   - Press Ctrl+C to stop")
        print()
        
        # Create visualizer
        visualizer = CognitiveLoadHeatmapVisualizer()
        
        logger.info("Starting visualization...")
        visualizer.start_visualization()
        
    except KeyboardInterrupt:
        logger.info("Visualization stopped by user")
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print("❌ Could not import required modules")
        print("   Make sure matplotlib, numpy, and seaborn are installed:")
        print("   pip install matplotlib numpy seaborn")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

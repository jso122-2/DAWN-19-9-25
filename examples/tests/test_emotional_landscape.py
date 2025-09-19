#!/usr/bin/env python3
"""
Test Runner for DAWN Emotional Resonance Landscape Visualization
================================================================

Test script to run the 3D emotional landscape visualization showing
DAWN's emotional state as a dynamic topology with weather effects
and real-time deformation.
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
    logger.info("🎭 Starting DAWN Emotional Resonance Landscape Test")
    
    try:
        from dawn.subsystems.visual.emotional_resonance_landscape import EmotionalLandscapeVisualizer
        
        print("🎭 Creating Emotional Resonance Landscape Visualization")
        print("   - 3D landscape where height = emotional intensity")
        print("   - Colors represent different emotional types")
        print("   - Real-time deformation based on emotional events")
        print("   - Weather systems creating emotional atmosphere")
        print("   - Mountains of joy, valleys of contemplation")
        print("   - Press Ctrl+C to stop")
        print()
        
        # Create visualizer with moderate landscape size for good performance
        visualizer = EmotionalLandscapeVisualizer(landscape_size=30)
        
        logger.info("Starting visualization...")
        visualizer.start_visualization()
        
    except KeyboardInterrupt:
        logger.info("Visualization stopped by user")
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print("❌ Could not import required modules")
        print("   Make sure matplotlib and numpy are installed:")
        print("   pip install matplotlib numpy")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

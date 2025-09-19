#!/usr/bin/env python3
"""
Test Runner for DAWN Semantic Wave Interference Visualization
============================================================

Test script to run the semantic wave interference patterns visualization
showing how different meanings interact and create new concepts through
beautiful wave physics.
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
    logger.info("🌊 Starting DAWN Semantic Wave Interference Test")
    
    try:
        from dawn.subsystems.visual.semantic_wave_interference import SemanticWaveVisualizer
        
        print("🌊 Creating Semantic Wave Interference Visualization")
        print("   - Beautiful wave physics showing concept interactions")
        print("   - Wave interference patterns where meanings merge")
        print("   - Concept nodes that emit semantic waves")
        print("   - Real-time stability field for persistent meanings")
        print("   - Press Ctrl+C to stop")
        print()
        
        # Create visualizer with moderate field size for good performance
        visualizer = SemanticWaveVisualizer(field_size=70)
        
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

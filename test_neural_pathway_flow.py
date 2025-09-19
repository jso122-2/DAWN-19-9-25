#!/usr/bin/env python3
"""
Test Runner for DAWN Neural Pathway Flow Network Visualization
=============================================================

Test script to run the 3D neural pathway flow visualization showing
information flowing through DAWN's neural pathways with activation
cascades and decision points.
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
    logger.info("🧠 Starting DAWN Neural Pathway Flow Network Test")
    
    try:
        from dawn.subsystems.visual.neural_pathway_flow import NeuralPathwayVisualizer
        
        print("🧠 Creating Neural Pathway Flow Network Visualization")
        print("   - 3D visualization of information flowing through neural pathways")
        print("   - Color-coded by information type (semantic, emotional, logical, etc.)")
        print("   - Real-time activation cascades and decision points")
        print("   - Press Ctrl+C to stop")
        print()
        
        # Create visualizer with moderate number of nodes for good performance
        visualizer = NeuralPathwayVisualizer(num_nodes=35)
        
        logger.info("Starting visualization...")
        visualizer.start_visualization()
        
    except KeyboardInterrupt:
        logger.info("Visualization stopped by user")
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print("❌ Could not import required modules")
        print("   Make sure matplotlib and numpy are installed:")
        print("   pip install matplotlib numpy networkx")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

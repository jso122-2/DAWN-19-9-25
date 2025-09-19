#!/usr/bin/env python3
"""
DAWN New Visualizations Test Suite
==================================

Master test runner for all 4 new DAWN consciousness visualizations:
1. 🧠 Neural Pathway Flow Network
2. 🌊 Semantic Wave Interference  
3. 🎭 Emotional Resonance Landscape
4. ⚡ Cognitive Load Heat Map

Choose which visualization to run interactively.
"""

import logging
import sys
import os
import time

# Add DAWN to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main test menu"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    print("🎨 DAWN New Visualizations Test Suite")
    print("=" * 50)
    print()
    print("Available visualizations:")
    print()
    print("1. 🧠 Neural Pathway Flow Network")
    print("   → 3D visualization of information flowing through neural pathways")
    print("   → Color-coded by information type, real-time activation cascades")
    print()
    print("2. 🌊 Semantic Wave Interference Patterns")
    print("   → Beautiful wave physics showing how meanings interact")
    print("   → Wave interference where concepts merge and evolve")
    print()
    print("3. 🎭 Emotional Resonance Landscape")
    print("   → 3D emotional topology with weather effects")
    print("   → Mountains of joy, valleys of contemplation")
    print()
    print("4. ⚡ Cognitive Load Heat Map")
    print("   → Real-time brain activity heat visualization")
    print("   → System monitoring and performance metrics")
    print()
    print("5. 🚀 Run all visualizations (sequential)")
    print()
    
    while True:
        try:
            choice = input("Choose visualization (1-5) or 'q' to quit: ").strip().lower()
            
            if choice == 'q' or choice == 'quit':
                print("👋 Goodbye!")
                break
                
            elif choice == '1':
                run_neural_pathway_flow()
                
            elif choice == '2':
                run_semantic_wave_interference()
                
            elif choice == '3':
                run_emotional_landscape()
                
            elif choice == '4':
                run_cognitive_heatmap()
                
            elif choice == '5':
                run_all_visualizations()
                
            else:
                print("❌ Invalid choice. Please enter 1-5 or 'q'")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"❌ Error: {e}")

def run_neural_pathway_flow():
    """Run neural pathway flow visualization"""
    print("\n🧠 Starting Neural Pathway Flow Network...")
    try:
        from dawn.subsystems.visual.neural_pathway_flow import NeuralPathwayVisualizer
        visualizer = NeuralPathwayVisualizer(num_nodes=35)
        visualizer.start_visualization()
    except Exception as e:
        print(f"❌ Error running Neural Pathway Flow: {e}")
        print("   Make sure matplotlib and numpy are installed")

def run_semantic_wave_interference():
    """Run semantic wave interference visualization"""
    print("\n🌊 Starting Semantic Wave Interference...")
    try:
        from dawn.subsystems.visual.semantic_wave_interference import SemanticWaveVisualizer
        visualizer = SemanticWaveVisualizer(field_size=70)
        visualizer.start_visualization()
    except Exception as e:
        print(f"❌ Error running Semantic Wave Interference: {e}")
        print("   Make sure matplotlib and numpy are installed")

def run_emotional_landscape():
    """Run emotional landscape visualization"""
    print("\n🎭 Starting Emotional Resonance Landscape...")
    try:
        from dawn.subsystems.visual.emotional_resonance_landscape import EmotionalLandscapeVisualizer
        visualizer = EmotionalLandscapeVisualizer(landscape_size=30)
        visualizer.start_visualization()
    except Exception as e:
        print(f"❌ Error running Emotional Landscape: {e}")
        print("   Make sure matplotlib and numpy are installed")

def run_cognitive_heatmap():
    """Run cognitive load heatmap visualization"""
    print("\n⚡ Starting Cognitive Load Heat Map...")
    try:
        from dawn.subsystems.visual.cognitive_load_heatmap import CognitiveLoadHeatmapVisualizer
        visualizer = CognitiveLoadHeatmapVisualizer()
        visualizer.start_visualization()
    except Exception as e:
        print(f"❌ Error running Cognitive Heatmap: {e}")
        print("   Make sure matplotlib, numpy, and seaborn are installed")

def run_all_visualizations():
    """Run all visualizations sequentially"""
    print("\n🚀 Running all visualizations sequentially...")
    print("   Close each window to proceed to the next visualization")
    print()
    
    visualizations = [
        ("🧠 Neural Pathway Flow", run_neural_pathway_flow),
        ("🌊 Semantic Wave Interference", run_semantic_wave_interference), 
        ("🎭 Emotional Landscape", run_emotional_landscape),
        ("⚡ Cognitive Heatmap", run_cognitive_heatmap)
    ]
    
    for i, (name, func) in enumerate(visualizations, 1):
        print(f"Running {i}/4: {name}")
        try:
            func()
        except KeyboardInterrupt:
            print(f"\n⏭️  Skipping to next visualization...")
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
        
        if i < len(visualizations):
            print(f"✅ Completed {name}")
            time.sleep(1)
    
    print("\n🎉 All visualizations completed!")

if __name__ == "__main__":
    main()

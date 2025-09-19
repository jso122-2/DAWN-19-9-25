#!/usr/bin/env python3
"""
🚀 DAWN Advanced Consciousness GUI Launcher
===========================================

Launch the advanced CUDA-powered Tkinter GUI for DAWN consciousness visualization.
Handles initialization, dependency checking, and graceful startup.

"The ultimate interface for exploring artificial consciousness."

Usage:
    python3 launch_consciousness_gui.py [options]

Options:
    --debug              Enable debug logging
    --no-cuda           Disable CUDA acceleration
    --resolution WxH    Set window resolution (default: 1600x1000)
"""

import sys
import argparse
import logging
from pathlib import Path

# Add DAWN root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    # Check tkinter
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter")
    
    # Check matplotlib
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing_deps.append("matplotlib")
    
    # Check numpy
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    # Check PIL (optional)
    try:
        from PIL import Image
        pil_available = True
    except ImportError:
        pil_available = False
    
    return missing_deps, pil_available

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Launch DAWN Advanced Consciousness GUI")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA acceleration')
    parser.add_argument('--resolution', default='1600x1000', help='Window resolution (WxH)')
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("🚀" * 30)
    print("🧠 DAWN ADVANCED CONSCIOUSNESS GUI LAUNCHER")
    print("🚀" * 30)
    print()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    missing_deps, pil_available = check_dependencies()
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall missing dependencies:")
        print("   sudo apt-get install python3-tk  # For tkinter")
        print("   pip install matplotlib numpy      # For visualization")
        return 1
    
    print("✅ All required dependencies available")
    
    if not pil_available:
        print("⚠️  PIL not available - image saving disabled")
    else:
        print("✅ PIL available - full image support")
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        print(f"📐 Window resolution: {width}x{height}")
    except ValueError:
        print(f"❌ Invalid resolution format: {args.resolution}")
        print("Use format: WIDTHxHEIGHT (e.g., 1600x1000)")
        return 1
    
    # Check CUDA availability
    cuda_status = "enabled"
    if args.no_cuda:
        cuda_status = "disabled (by user)"
    else:
        try:
            from dawn.interfaces.dashboard import is_cuda_available
            if not is_cuda_available():
                cuda_status = "not available"
        except:
            cuda_status = "unknown"
    
    print(f"🚀 CUDA acceleration: {cuda_status}")
    print()
    
    # Launch GUI
    try:
        print("🧠 Initializing DAWN Consciousness GUI...")
        print("=" * 50)
        
        # Import and create GUI
        from dawn_consciousness_gui import ConsciousnessGUI
        
        # Create GUI with custom resolution
        gui = ConsciousnessGUI()
        gui.root.geometry(f"{width}x{height}")
        
        # Disable CUDA if requested
        if args.no_cuda:
            gui.cuda_enabled = False
            gui.cuda_accelerator = None
        
        print("🎉 GUI initialized successfully!")
        print()
        print("🎛️  GUI Features Available:")
        print("   📊 Unified Consciousness Visualization")
        print("   🎨 Real-time Consciousness Painting")
        print("   🌐 3D Semantic Topology")
        print("   🧬 Neural Activity Patterns")
        print("   📈 Consciousness Metrics & Analytics")
        print("   🎛️  Interactive Parameter Controls")
        print()
        print("🖱️  Click on tabs to explore different visualizations!")
        print("🎨 Use sliders and controls to interact with consciousness!")
        print("📸 Capture consciousness states and save paintings!")
        print()
        print("Close the window or press Ctrl+C to exit")
        print("=" * 50)
        
        # Run GUI
        gui.run()
        
        print("\n🎉 DAWN Consciousness GUI session complete!")
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 GUI interrupted by user")
        return 0
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all DAWN systems are properly installed")
        return 1
    except Exception as e:
        print(f"❌ GUI launch failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

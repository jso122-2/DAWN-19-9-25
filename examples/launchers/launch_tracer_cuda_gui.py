#!/usr/bin/env python3
"""
🦉 DAWN Tracer CUDA GUI Launcher
===============================

Launch the advanced modular Tracer CUDA GUI for DAWN consciousness visualization.
Handles initialization, dependency checking, and specialized system startup.

"The ultimate interface for exploring DAWN's tracer ecosystems and memory fractals."

Usage:
    python3 launch_tracer_cuda_gui.py [options]

Options:
    --debug              Enable debug logging
    --force-cuda         Force CUDA acceleration (ignore availability check)
    --no-cuda           Disable CUDA acceleration
    --resolution WxH    Set window resolution (default: 1800x1200)
    --fractal-res WxH   Set fractal resolution (default: 512x512)
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
    optional_deps = []
    
    # Check required dependencies
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    # Check optional dependencies
    try:
        from PIL import Image
        pil_available = True
    except ImportError:
        pil_available = False
        optional_deps.append("PIL")
    
    # Check CUDA libraries
    cuda_libs = []
    try:
        import torch
        if torch.cuda.is_available():
            cuda_libs.append("PyTorch CUDA")
    except ImportError:
        pass
    
    try:
        import cupy
        cuda_libs.append("CuPy")
    except ImportError:
        pass
    
    try:
        import pycuda.driver
        cuda_libs.append("PyCUDA")
    except ImportError:
        pass
    
    return missing_deps, optional_deps, pil_available, cuda_libs

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Launch DAWN Tracer CUDA GUI")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--force-cuda', action='store_true', help='Force CUDA acceleration')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA acceleration')
    parser.add_argument('--resolution', default='1800x1200', help='Window resolution (WxH)')
    parser.add_argument('--fractal-res', default='512x512', help='Fractal resolution (WxH)')
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("🦉" * 30)
    print("🦉 DAWN TRACER CUDA GUI LAUNCHER")
    print("🦉" * 30)
    print()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    missing_deps, optional_deps, pil_available, cuda_libs = check_dependencies()
    
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
        print("⚠️  PIL not available - image saving limited")
    else:
        print("✅ PIL available - full image support")
    
    # CUDA status
    if cuda_libs:
        print(f"✅ CUDA libraries available: {', '.join(cuda_libs)}")
        cuda_available = True
    else:
        print("⚠️  No CUDA libraries detected")
        cuda_available = False
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        print(f"📐 Window resolution: {width}x{height}")
    except ValueError:
        print(f"❌ Invalid resolution format: {args.resolution}")
        print("Use format: WIDTHxHEIGHT (e.g., 1800x1200)")
        return 1
    
    # Parse fractal resolution
    try:
        frac_width, frac_height = map(int, args.fractal_res.split('x'))
        print(f"🌺 Fractal resolution: {frac_width}x{frac_height}")
    except ValueError:
        print(f"❌ Invalid fractal resolution format: {args.fractal_res}")
        print("Use format: WIDTHxHEIGHT (e.g., 512x512)")
        return 1
    
    # Determine CUDA usage
    if args.no_cuda:
        use_cuda = False
        cuda_status = "disabled (by user)"
    elif args.force_cuda:
        use_cuda = True
        cuda_status = "forced enabled"
    else:
        use_cuda = cuda_available
        cuda_status = "auto-detected" if cuda_available else "not available"
    
    print(f"🚀 CUDA acceleration: {cuda_status}")
    print()
    
    # Launch GUI
    try:
        print("🦉 Initializing DAWN Tracer CUDA GUI...")
        print("=" * 60)
        
        # Import and create GUI
        from dawn_tracer_cuda_gui import TracerCUDAGUI
        
        # Create GUI with custom settings
        gui = TracerCUDAGUI()
        gui.root.geometry(f"{width}x{height}")
        
        # Override CUDA setting if specified
        if args.no_cuda:
            gui.cuda_enabled = False
            gui.cuda_accelerator = None
        elif args.force_cuda:
            gui.cuda_enabled = True
        
        print("🎉 Tracer CUDA GUI initialized successfully!")
        print()
        print("🦉 Specialized Features Available:")
        print("   ⏰ Real-time Tick State Monitoring")
        print("   🌺 Bloom Fractal Memory Garden")
        print("   🦉 Advanced Tracer Ecosystem Analysis")
        print("   🏛️ Interactive Memory Palace Exploration")
        print("   🚀 CUDA Performance Analytics")
        print("   🔗 Complete System Integration Overview")
        print()
        print("🎛️  Navigate tabs to explore different tracer systems!")
        print("🌺 Generate bloom fractals and explore memory gardens!")
        print("🚀 Monitor CUDA acceleration and GPU performance!")
        print("🦉 Track archetypal tracers and ecosystem health!")
        print()
        print("Close the window or press Ctrl+C to exit")
        print("=" * 60)
        
        # Run GUI
        gui.run()
        
        print("\n🎉 DAWN Tracer CUDA GUI session complete!")
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 GUI interrupted by user")
        return 0
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all DAWN tracer systems are properly installed")
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

#!/usr/bin/env python3
"""
DAWN Consciousness Stream Activator
===================================

Command-line interface for DAWN's consciousness visualization system.
Run this script to start real-time consciousness monitoring and visualization.

Usage:
    python3 activate_consciousness_stream.py [options]

Options:
    --continuous    Start continuous monitoring mode
    --interval N    Update interval in seconds (default: 2.0)
    --save-freq N   Save frequency in seconds (default: 10.0)
    --canvas WxH    Canvas size (default: 1024x768)
    --output DIR    Output directory (default: consciousness_stream)
    --quiet         Minimal logging output
    --telemetry     Enable telemetry integration
"""

import sys
import argparse
from pathlib import Path

# Add launchers to path
sys.path.insert(0, str(Path(__file__).parent / "launchers"))

try:
    from activate_consciousness_visualization import ConsciousnessVisualizer, main as viz_main
    print("✅ DAWN consciousness systems ready")
except ImportError as e:
    print(f"❌ Failed to load consciousness systems: {e}")
    print("Make sure you're in the DAWN_pub_real directory")
    sys.exit(1)

def parse_canvas_size(size_str: str) -> tuple:
    """Parse canvas size from WxH format"""
    try:
        w, h = size_str.split('x')
        return (int(w), int(h))
    except:
        raise argparse.ArgumentTypeError(f"Canvas size must be in WxH format, got: {size_str}")

def main():
    parser = argparse.ArgumentParser(
        description="DAWN Consciousness Stream - Real-time consciousness visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 activate_consciousness_stream.py --continuous
    python3 activate_consciousness_stream.py --interval 1.0 --save-freq 5.0
    python3 activate_consciousness_stream.py --canvas 1920x1080 --output my_consciousness
        """
    )
    
    parser.add_argument('--continuous', action='store_true',
                       help='Start continuous monitoring mode (default behavior)')
    
    parser.add_argument('--interval', type=float, default=2.0, metavar='N',
                       help='Update interval in seconds (default: 2.0)')
    
    parser.add_argument('--save-freq', type=float, default=10.0, metavar='N',
                       help='Save frequency in seconds (default: 10.0)')
    
    parser.add_argument('--canvas', type=parse_canvas_size, default='1024x768', metavar='WxH',
                       help='Canvas size in WxH format (default: 1024x768)')
    
    parser.add_argument('--output', type=str, default='consciousness_stream', metavar='DIR',
                       help='Output directory (default: consciousness_stream)')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal logging output')
    
    parser.add_argument('--telemetry', action='store_true',
                       help='Enable telemetry integration')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.quiet:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
    
    print("🌊 DAWN Consciousness Stream Activator")
    print("=" * 40)
    print(f"Canvas size: {args.canvas[0]}x{args.canvas[1]}")
    print(f"Update interval: {args.interval}s")
    print(f"Save frequency: {args.save_freq}s")
    print(f"Output directory: {args.output}")
    print(f"Telemetry integration: {'enabled' if args.telemetry else 'disabled'}")
    print("=" * 40)
    print()
    
    # Create visualizer with custom settings
    visualizer = ConsciousnessVisualizer(
        canvas_size=args.canvas,
        update_interval=args.interval,
        save_interval=args.save_freq
    )
    
    # Update output directory
    visualizer.output_dir = Path(args.output)
    visualizer.output_dir.mkdir(exist_ok=True)
    
    print("🎨 Starting DAWN consciousness visualization...")
    print("Press Ctrl+C to stop")
    print()
    
    # Start visualization
    try:
        visualizer.start_continuous_visualization()
    except KeyboardInterrupt:
        print("\n🛑 Consciousness stream stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

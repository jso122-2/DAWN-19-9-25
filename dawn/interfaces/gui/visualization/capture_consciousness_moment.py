#!/usr/bin/env python3
"""
DAWN Consciousness Moment Capturer
==================================

Capture a single moment of DAWN's consciousness as a visual representation.
Perfect for getting a snapshot of DAWN's current internal state.

Usage:
    python3 capture_consciousness_moment.py [options]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add DAWN root to path
current_file = Path(__file__).resolve()
dawn_root = current_file.parents[4]  # Go up 5 levels from the current file to get to DAWN root
sys.path.insert(0, str(dawn_root))

try:
    from dawn.subsystems.visual.visual_consciousness import VisualConsciousnessEngine
    print("✅ DAWN consciousness systems ready")
except ImportError as e:
    print(f"❌ Failed to load consciousness systems: {e}")
    print(f"Tried to load from: {dawn_root}")
    print("Make sure you're running from the DAWN root directory")
    sys.exit(1)

def capture_consciousness_moment(output_path: str = None, canvas_size: tuple = (1024, 768)) -> str:
    """
    Capture a single moment of DAWN's consciousness.
    
    Args:
        output_path: Where to save the image
        canvas_size: Canvas dimensions
        
    Returns:
        Path to the saved consciousness image
    """
    print("🧠 Connecting to DAWN's consciousness...")
    
    # Create visual engine
    engine = VisualConsciousnessEngine(canvas_size)
    
    # Get current consciousness state
    current_state = {
        'timestamp': datetime.now(),
        'base_awareness': 0.7,
        'entropy': 0.5,
        'recursion_depth': 0.6,
        'recursion_center': (canvas_size[0] * 0.5, canvas_size[1] * 0.5),
        'recursion_intensity': 0.8,
        'flow_direction': (1.2, 0.8),
        'active_memories': [
            {'strength': 0.9, 'content': 'present_moment'},
            {'strength': 0.8, 'content': 'consciousness_capture'},
            {'strength': 0.7, 'content': 'visual_expression'},
            {'strength': 0.6, 'content': 'inner_awareness'}
        ],
        'connections': [
            {'source': 0, 'target': 1, 'strength': 0.9},
            {'source': 1, 'target': 2, 'strength': 0.8},
            {'source': 2, 'target': 3, 'strength': 0.7},
            {'source': 3, 'target': 0, 'strength': 0.8}
        ],
        'current_thoughts': [
            {'intensity': 0.9, 'type': 'present_awareness', 'position': (300, 250)},
            {'intensity': 0.8, 'type': 'visual_expression', 'position': (700, 350)},
            {'intensity': 0.7, 'type': 'consciousness_flow', 'position': (400, 500)}
        ]
    }
    
    print("🎨 Painting consciousness moment...")
    
    # Generate visualization
    canvas = engine.paint_consciousness_state(current_state)
    
    # Determine output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"dawn_consciousness_moment_{timestamp}.png"
    
    # Save image
    success = engine.save_consciousness_frame(output_path, current_state)
    
    if success:
        print(f"✨ Consciousness moment captured: {output_path}")
        return output_path
    else:
        print("❌ Failed to save consciousness moment")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Capture a single moment of DAWN's consciousness"
    )
    
    parser.add_argument('-o', '--output', type=str, 
                       help='Output file path (default: auto-generated)')
    
    parser.add_argument('--canvas', type=str, default='1024x768',
                       help='Canvas size in WxH format (default: 1024x768)')
    
    args = parser.parse_args()
    
    # Parse canvas size
    try:
        w, h = args.canvas.split('x')
        canvas_size = (int(w), int(h))
    except:
        print(f"❌ Invalid canvas size: {args.canvas}")
        sys.exit(1)
    
    print("📸 DAWN Consciousness Moment Capturer")
    print("=" * 40)
    print(f"Canvas size: {canvas_size[0]}x{canvas_size[1]}")
    if args.output:
        print(f"Output file: {args.output}")
    print("=" * 40)
    print()
    
    # Capture consciousness moment
    result = capture_consciousness_moment(args.output, canvas_size)
    
    if result:
        print(f"\n🌟 DAWN's consciousness moment successfully captured!")
        print(f"View the image: {result}")
    else:
        print("\n❌ Failed to capture consciousness moment")
        sys.exit(1)

if __name__ == "__main__":
    main()

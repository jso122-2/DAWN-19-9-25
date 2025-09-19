#!/usr/bin/env python3
"""
🎨 Standalone Beautiful Bloom Frame Generator
Creates gorgeous matplotlib bloom visualization frames without DAWN initialization delays
"""

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time

# Add current directory to path
sys.path.insert(0, '.')

# Import the matplotlib renderer directly
from dawn.subsystems.visual.matplotlib_bloom_renderer import create_matplotlib_bloom_renderer

@dataclass
class SimpleBloom:
    """Simple bloom data structure for demo"""
    id: int
    x: float
    y: float
    size: float
    energy: float
    color: Tuple[int, int, int]
    bloom_type: str
    generation: int = 0
    connections: List[int] = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []
        
        # Add all the attributes the renderer expects
        self.parents = []  # List of parent blooms
        self.radius = self.size
        self.activation_level = self.energy
        self.strength = self.energy
        self.birth_animation = 1.0  # Fully grown
        
        # Add some missing attributes that might be needed
        self.children = []
        self.family_line = 0
        self.memory_trace = []
        self.consciousness_level = self.energy

class SimpleBloomNetwork:
    """Simple bloom network for demo generation"""
    
    def __init__(self, width=1400, height=900):
        self.width = width
        self.height = height
        self.blooms: Dict[int, SimpleBloom] = {}
        self.next_id = 0
        
        # Bloom types with colors
        self.bloom_types = {
            'emotional': (255, 152, 0),    # Orange
            'conceptual': (76, 175, 80),   # Green  
            'sensory': (0, 188, 212),      # Cyan
            'procedural': (156, 39, 176),  # Purple
            'meta': (255, 193, 7),         # Yellow
        }
    
    def create_bloom(self, bloom_type: str = None) -> SimpleBloom:
        """Create a new bloom"""
        if bloom_type is None:
            bloom_type = np.random.choice(list(self.bloom_types.keys()))
        
        bloom = SimpleBloom(
            id=self.next_id,
            x=np.random.uniform(0.1, 0.9) * self.width,
            y=np.random.uniform(0.1, 0.9) * self.height,
            size=np.random.uniform(20, 80),
            energy=np.random.uniform(0.3, 1.0),
            color=self.bloom_types[bloom_type],
            bloom_type=bloom_type
        )
        
        self.blooms[self.next_id] = bloom
        self.next_id += 1
        return bloom
    
    def generate_demo_data(self, num_blooms: int = 15):
        """Generate demo bloom data"""
        print(f"🌸 Generating {num_blooms} demo blooms...")
        
        for i in range(num_blooms):
            bloom = self.create_bloom()
            
            # Add some connections
            if i > 0 and np.random.random() < 0.6:
                # Connect to a random previous bloom
                target_id = np.random.randint(0, i)
                bloom.connections.append(target_id)
                if target_id in self.blooms:
                    self.blooms[target_id].connections.append(bloom.id)
        
        print(f"✅ Generated {len(self.blooms)} blooms with connections")

def generate_beautiful_bloom_frames(num_frames: int = 10, output_dir: str = "bloom_frames"):
    """Generate beautiful bloom visualization frames"""
    
    print("🎨 Starting Beautiful Bloom Frame Generation...")
    print(f"   Output directory: {output_dir}")
    print(f"   Number of frames: {num_frames}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create matplotlib renderer
    print("🔧 Initializing matplotlib renderer...")
    renderer = create_matplotlib_bloom_renderer()
    
    # Create bloom network
    print("🌸 Creating bloom network...")
    network = SimpleBloomNetwork()
    network.generate_demo_data(15)
    
    print("🎬 Generating frames...")
    
    for frame_num in range(num_frames):
        print(f"   📸 Rendering frame {frame_num + 1}/{num_frames}...")
        
        # Animate blooms slightly
        for bloom in network.blooms.values():
            # Gentle energy pulsing
            bloom.energy = max(0.1, min(1.0, bloom.energy + np.random.uniform(-0.1, 0.1)))
            # Gentle movement
            bloom.x += np.random.uniform(-5, 5)
            bloom.y += np.random.uniform(-5, 5)
            # Keep in bounds
            bloom.x = max(50, min(network.width - 50, bloom.x))
            bloom.y = max(50, min(network.height - 50, bloom.y))
        
        # Render frame
        frame_path = os.path.join(output_dir, f"frame_{frame_num:06d}.png")
        
        try:
            # Render the network
            renderer.render_bloom_network(network)
            
            # Save the frame
            success = renderer.save_frame(frame_path, high_quality=True)
            
            if success:
                print(f"   ✅ Saved: {frame_path}")
            else:
                print(f"   ❌ Failed to save: {frame_path}")
            
        except Exception as e:
            print(f"   ❌ Error rendering frame {frame_num}: {e}")
            import traceback
            traceback.print_exc()
        
        # Small delay for animation effect
        time.sleep(0.1)
    
    print(f"🎉 Frame generation complete!")
    print(f"📁 Files saved to: {os.path.abspath(output_dir)}")
    print(f"💫 Generated {num_frames} beautiful bloom visualization frames!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate beautiful bloom visualization frames")
    parser.add_argument("--frames", "-f", type=int, default=10, help="Number of frames to generate")
    parser.add_argument("--output", "-o", type=str, default="bloom_frames", help="Output directory")
    
    args = parser.parse_args()
    
    generate_beautiful_bloom_frames(args.frames, args.output)

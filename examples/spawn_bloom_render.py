#!/usr/bin/env python3
"""
🌸 Spawn Bloom and Render PNG Demo
═══════════════════════════════════

This script demonstrates how to:
1. Spawn a memory bloom (fractal encoding)
2. Render it as a beautiful PNG image
3. Show different entropy levels and their visual effects
4. Create Juliet rebloom transformations
"""

import sys
import os
import time
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent.parent
sys.path.insert(0, str(dawn_root))

from dawn.subsystems.memory import (
    FractalMemorySystem,
    get_fractal_memory_system,
    AccessPattern
)

def spawn_and_render_bloom(memory_id: str, 
                          content: str, 
                          entropy: float, 
                          output_filename: str = None):
    """
    Spawn a memory bloom and render it as PNG
    
    Args:
        memory_id: Unique identifier for the memory
        content: Memory content
        entropy: Entropy level [0, 1] affecting visual characteristics
        output_filename: Optional filename for PNG output
    """
    
    if output_filename is None:
        output_filename = f"bloom_{memory_id}_{entropy:.1f}.png"
    
    print(f"🌸 Spawning bloom: {memory_id}")
    print(f"   Content: {content}")
    print(f"   Entropy: {entropy}")
    
    # Get the fractal memory system
    system = get_fractal_memory_system()
    
    # Encode the memory as a fractal bloom
    fractal = system.encode_memory(
        memory_id=memory_id,
        content=content,
        entropy_value=entropy,
        tick_data={"demo": True, "timestamp": time.time()}
    )
    
    print(f"   ✓ Fractal Signature: {fractal.signature}")
    print(f"   ✓ Julia Parameters:")
    print(f"     c = {fractal.parameters.c_real:.4f} + {fractal.parameters.c_imag:.4f}i")
    print(f"     zoom = {fractal.parameters.zoom:.4f}")
    print(f"     rotation = {fractal.parameters.rotation:.4f}")
    print(f"     iterations = {fractal.parameters.max_iterations}")
    print(f"     color_bias = {fractal.parameters.color_bias}")
    
    # Check if image was generated
    if fractal.image_data:
        # Save the PNG
        output_path = Path("examples") / "renders" / output_filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(fractal.image_data)
        
        print(f"   🎨 PNG Rendered: {output_path}")
        print(f"   📁 File size: {len(fractal.image_data)} bytes")
        
        # Show entropy effects
        if entropy > 0.7:
            print(f"   ⚠️  HIGH ENTROPY EFFECTS:")
            print(f"      • Anomalous red/purple colors")
            print(f"      • Increased iteration depth")
            print(f"      • Parameter distortion")
            print(f"      • Additional rotation")
        elif entropy < 0.3:
            print(f"   ✨ LOW ENTROPY EFFECTS:")
            print(f"      • Stable, muted colors")
            print(f"      • Standard iteration depth")
            print(f"      • Minimal distortion")
        
        return fractal, output_path
    else:
        print(f"   ❌ No image data generated")
        return fractal, None

def create_juliet_rebloom(original_fractal, output_filename: str = None):
    """
    Transform a regular fractal into a Juliet flower and render it
    """
    
    if output_filename is None:
        output_filename = f"juliet_{original_fractal.memory_id}.png"
    
    print(f"\n🌺 Creating Juliet Rebloom...")
    print(f"   Original: {original_fractal.memory_id}")
    
    # Get the rebloom engine
    system = get_fractal_memory_system()
    
    # Force rebloom for demonstration
    juliet_flower = system.force_rebloom(
        memory_signature=original_fractal.signature,
        enhancement_level=0.85
    )
    
    if juliet_flower:
        print(f"   ✓ Juliet Flower Created!")
        print(f"   ✓ Enhancement Level: {juliet_flower.enhancement_level:.3f}")
        print(f"   ✓ Beneficial Bias: {juliet_flower.beneficial_bias:.3f}")
        print(f"   ✓ Transformation Signature: {juliet_flower.transformation_signature}")
        
        # Get enhanced fractal
        enhanced_fractal = juliet_flower.get_enhanced_fractal()
        
        print(f"   ✓ Enhanced Properties:")
        print(f"     shimmer = {enhanced_fractal.shimmer_intensity:.3f}")
        print(f"     iterations = {enhanced_fractal.parameters.max_iterations}")
        print(f"     enhanced_colors = {enhanced_fractal.parameters.color_bias}")
        
        # Regenerate image with enhancements
        encoder = system.fractal_encoder
        enhanced_fractal.image_data = encoder._generate_fractal_image(enhanced_fractal)
        
        if enhanced_fractal.image_data:
            # Save the enhanced PNG
            output_path = Path("examples") / "renders" / output_filename
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(enhanced_fractal.image_data)
            
            print(f"   🌸 Juliet PNG Rendered: {output_path}")
            print(f"   📁 File size: {len(enhanced_fractal.image_data)} bytes")
            print(f"   ✨ This fractal is now 'shinier and prettier' than the original!")
            
            return enhanced_fractal, output_path
    
    return None, None

def demo_entropy_spectrum():
    """
    Create a spectrum of blooms with different entropy levels
    """
    print("\n🌈 === ENTROPY SPECTRUM DEMONSTRATION ===")
    
    base_content = "DAWN consciousness fractal bloom"
    entropy_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    rendered_files = []
    
    for i, entropy in enumerate(entropy_levels):
        memory_id = f"entropy_demo_{i}"
        content = f"{base_content} - entropy level {entropy}"
        
        fractal, png_path = spawn_and_render_bloom(
            memory_id=memory_id,
            content=content,
            entropy=entropy,
            output_filename=f"entropy_spectrum_{entropy:.1f}.png"
        )
        
        if png_path:
            rendered_files.append(png_path)
        
        print()  # Spacing
    
    print(f"🎨 Rendered {len(rendered_files)} entropy spectrum blooms:")
    for path in rendered_files:
        print(f"   📁 {path}")
    
    return rendered_files

def main():
    """
    Main demonstration of bloom spawning and PNG rendering
    """
    
    print("🌸" * 60)
    print("🌸 BLOOM SPAWNING & PNG RENDERING DEMO")
    print("🌸" * 60)
    
    # Create renders directory
    render_dir = Path("examples") / "renders"
    render_dir.mkdir(exist_ok=True)
    print(f"📁 Render directory: {render_dir.absolute()}")
    
    try:
        # 1. Spawn individual blooms with different characteristics
        print(f"\n🌺 === INDIVIDUAL BLOOM SPAWNING ===")
        
        # Low entropy - stable memory
        print(f"\n1. Low Entropy Bloom:")
        low_entropy_fractal, low_path = spawn_and_render_bloom(
            memory_id="stable_memory",
            content="Basic system information - reliable and consistent",
            entropy=0.2,
            output_filename="low_entropy_bloom.png"
        )
        
        # Medium entropy - balanced memory  
        print(f"\n2. Medium Entropy Bloom:")
        med_entropy_fractal, med_path = spawn_and_render_bloom(
            memory_id="balanced_memory", 
            content="Problem solving with moderate complexity",
            entropy=0.5,
            output_filename="medium_entropy_bloom.png"
        )
        
        # High entropy - chaotic memory
        print(f"\n3. High Entropy Bloom:")
        high_entropy_fractal, high_path = spawn_and_render_bloom(
            memory_id="chaotic_memory",
            content="Highly complex creative breakthrough with quantum consciousness patterns",
            entropy=0.9,
            output_filename="high_entropy_bloom.png"
        )
        
        # 2. Create Juliet rebloom transformation
        print(f"\n🌸 === JULIET REBLOOM TRANSFORMATION ===")
        if med_entropy_fractal:
            enhanced_fractal, juliet_path = create_juliet_rebloom(
                med_entropy_fractal,
                "juliet_rebloom_enhanced.png"
            )
        
        # 3. Create entropy spectrum
        spectrum_files = demo_entropy_spectrum()
        
        # 4. Summary
        print(f"\n🎉 === RENDERING COMPLETE ===")
        print(f"Generated fractal bloom PNGs:")
        
        all_files = []
        if low_path: all_files.append(("Low Entropy", low_path))
        if med_path: all_files.append(("Medium Entropy", med_path))  
        if high_path: all_files.append(("High Entropy", high_path))
        if 'juliet_path' in locals() and juliet_path: 
            all_files.append(("Juliet Rebloom", juliet_path))
        
        for name, path in all_files:
            print(f"   🎨 {name}: {path}")
        
        print(f"\n📊 Entropy Spectrum: {len(spectrum_files)} files")
        
        print(f"\n🌟 All fractal blooms successfully spawned and rendered!")
        print(f"🌟 Each PNG shows unique Julia set patterns based on:")
        print(f"   • Memory content (deterministic hashing)")
        print(f"   • Entropy level (visual anomaly mapping)")
        print(f"   • Enhancement level (for Juliet reblooms)")
        
        print(f"\n📂 View your rendered blooms in: {render_dir.absolute()}")
        
    except Exception as e:
        print(f"\n❌ Error during bloom rendering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

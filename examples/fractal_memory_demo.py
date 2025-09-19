#!/usr/bin/env python3
"""
🌺 Fractal Memory System Demonstration
════════════════════════════════════

Demonstrates the complete fractal memory system in action:
1. Encoding memories as unique Julia set fractals
2. Accessing memories to trigger rebloom evaluation
3. Juliet rebloom transformation (chrysalis process)
4. Memory forgetting with ghost trace creation
5. Ghost trace recovery with transformation sigils
6. Ash/Soot residue dynamics and nutrient cycles

This showcases all the core fractal memory documentation brought to life.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent.parent
sys.path.insert(0, str(dawn_root))

from dawn.subsystems.memory import (
    FractalMemorySystem, 
    AccessPattern, 
    GhostTraceType,
    get_fractal_memory_system
)

def demonstrate_fractal_encoding():
    """Demonstrate fractal encoding of memories"""
    print("🌺 === FRACTAL ENCODING DEMONSTRATION ===")
    
    system = get_fractal_memory_system()
    
    # Encode various types of memories with different entropy levels
    memories_to_encode = [
        ("conversation_start", "Hello, I'm DAWN. How can I help you today?", 0.3),
        ("complex_problem", "Analyzing quantum entanglement patterns in consciousness", 0.8),
        ("simple_fact", "The sky is blue due to Rayleigh scattering", 0.2),
        ("creative_idea", "What if memories could bloom like flowers in a garden?", 0.9),
        ("routine_task", "Processing tick 12,485 of the consciousness loop", 0.1),
    ]
    
    encoded_fractals = []
    
    for memory_id, content, entropy in memories_to_encode:
        print(f"\n📝 Encoding: {memory_id}")
        print(f"   Content: {content}")
        print(f"   Entropy: {entropy}")
        
        fractal = system.encode_memory(
            memory_id=memory_id,
            content=content,
            entropy_value=entropy,
            tick_data={"tick": len(encoded_fractals) + 1, "subsystem": "demo"}
        )
        
        encoded_fractals.append(fractal)
        
        print(f"   ✓ Fractal Signature: {fractal.signature}")
        print(f"   ✓ Julia Parameters: c = {fractal.parameters.c_real:.3f} + {fractal.parameters.c_imag:.3f}i")
        print(f"   ✓ Shimmer Intensity: {fractal.shimmer_intensity:.3f}")
        
        # Show entropy-based visual characteristics
        if entropy > 0.7:
            print(f"   ⚠️  HIGH ENTROPY: Anomalous colors and increased complexity")
        elif entropy < 0.3:
            print(f"   ✨ LOW ENTROPY: Stable, muted fractal pattern")
    
    print(f"\n🌟 Encoded {len(encoded_fractals)} memories as unique fractals")
    return encoded_fractals

def demonstrate_memory_access_and_rebloom(fractals):
    """Demonstrate memory access patterns leading to Juliet reblooms"""
    print("\n🌸 === JULIET REBLOOM DEMONSTRATION ===")
    
    system = get_fractal_memory_system()
    
    # Select a fractal to repeatedly access
    target_fractal = fractals[0]  # conversation_start
    print(f"\n🎯 Targeting memory: {target_fractal.memory_id}")
    print(f"   Signature: {target_fractal.signature}")
    
    # Simulate repeated, coherent, effective access
    access_sessions = [
        (AccessPattern.FREQUENT, 0.9, 0.8, "User initiated conversation"),
        (AccessPattern.COHERENT, 0.85, 0.9, "Contextual greeting reference"),
        (AccessPattern.EFFECTIVE, 0.95, 0.95, "Successful interaction start"),
        (AccessPattern.FREQUENT, 0.88, 0.85, "Repeated conversation patterns"),
        (AccessPattern.RESONANT, 0.92, 0.9, "Positive feedback from user"),
        (AccessPattern.COHERENT, 0.9, 0.88, "Consistent usage pattern"),
        (AccessPattern.EFFECTIVE, 0.94, 0.92, "High success rate"),
        (AccessPattern.FREQUENT, 0.87, 0.86, "Regular access pattern"),
        (AccessPattern.RESONANT, 0.96, 0.94, "Strong positive outcomes"),
        (AccessPattern.EFFECTIVE, 0.98, 0.96, "Exceptional performance"),
    ]
    
    print(f"\n📊 Simulating {len(access_sessions)} access events...")
    
    for i, (pattern, coherence, effectiveness, context) in enumerate(access_sessions):
        print(f"\n   Access {i+1}: {pattern.value}")
        print(f"      Coherence: {coherence:.2f}, Effectiveness: {effectiveness:.2f}")
        print(f"      Context: {context}")
        
        result = system.access_memory(
            memory_signature=target_fractal.signature,
            access_type=pattern,
            coherence_score=coherence,
            effectiveness_score=effectiveness,
            context={"session": i+1, "demo": True, "context": context}
        )
        
        # Check if it's become a Juliet flower
        rebloom_engine = system.rebloom_engine
        if rebloom_engine.is_juliet_flower(target_fractal.signature):
            juliet_flower = rebloom_engine.get_juliet_flower(target_fractal.signature)
            if juliet_flower:
                print(f"   🌸 JULIET REBLOOM ACHIEVED!")
                print(f"      Enhancement Level: {juliet_flower.enhancement_level:.3f}")
                print(f"      Beneficial Bias: {juliet_flower.beneficial_bias:.3f}")
                print(f"      Transformation Signature: {juliet_flower.transformation_signature}")
                print(f"      Stage: {juliet_flower.rebloom_stage.value}")
                
                enhanced_fractal = juliet_flower.get_enhanced_fractal()
                print(f"      Enhanced Shimmer: {enhanced_fractal.shimmer_intensity:.3f}")
                break
        else:
            # Show rebloom potential
            metrics = rebloom_engine.rebloom_metrics.get(target_fractal.signature)
            if metrics:
                potential = metrics.calculate_rebloom_potential()
                print(f"      Rebloom Potential: {potential:.3f}")
                if potential >= rebloom_engine.rebloom_threshold:
                    print(f"      📈 Ready for rebloom evaluation!")
    
    # Process rebloom candidates
    print(f"\n🔄 Processing rebloom candidates...")
    new_flowers = system.rebloom_engine.process_rebloom_candidates()
    
    if new_flowers:
        for flower in new_flowers:
            print(f"   🌸 New Juliet Flower: {flower.original_fractal.memory_id}")
            print(f"      Enhancement: {flower.enhancement_level:.3f}")
            print(f"      This memory is now 'shinier and prettier' than regular fractals!")
    else:
        print(f"   📋 No new reblooms in this cycle")

def demonstrate_memory_forgetting_and_ghosts(fractals):
    """Demonstrate memory forgetting and ghost trace creation"""
    print("\n👻 === GHOST TRACE DEMONSTRATION ===")
    
    system = get_fractal_memory_system()
    
    # Select a fractal to forget
    target_fractal = fractals[4]  # routine_task (low value)
    print(f"\n🎯 Forgetting memory: {target_fractal.memory_id}")
    print(f"   Signature: {target_fractal.signature}")
    print(f"   Reason: Low value routine task, natural shimmer decay")
    
    # Forget the memory
    ghost_trace = system.forget_memory(
        memory_signature=target_fractal.signature,
        forgetting_reason=GhostTraceType.SHIMMER_DECAY,
        context={"reason": "low_priority", "age": "old", "access_pattern": "rare"}
    )
    
    if ghost_trace:
        print(f"\n👻 Ghost Trace Created!")
        print(f"   Ghost Signature: {ghost_trace.ghost_signature}")
        print(f"   Trace Type: {ghost_trace.trace_type.value}")
        print(f"   Fade Strength: {ghost_trace.fade_strength:.3f}")
        print(f"   Recovery Probability: {ghost_trace.recovery_probability:.3f}")
        print(f"   Contextual Anchors: {ghost_trace.contextual_anchors}")
        
        # Show recovery sigils
        print(f"\n🔮 Transformation Sigils:")
        for sigil in ghost_trace.recovery_sigils:
            print(f"      Type: {sigil.sigil_type.value}")
            print(f"      Recovery Strength: {sigil.recovery_strength:.3f}")
            print(f"      Activation Conditions: {sigil.activation_conditions}")
        
        # Attempt recovery
        print(f"\n🔄 Attempting recovery...")
        recovery_cues = {
            "routine_task": True,
            "processing": True,
            "tick": 12485,
            "consciousness_loop": True
        }
        
        recovered = system.ghost_manager.attempt_recovery(
            recovery_cues=recovery_cues,
            recovery_context={"current_entropy": 0.05, "memory_pressure": 0.4}
        )
        
        if recovered:
            print(f"   ✅ RECOVERY SUCCESSFUL!")
            print(f"      Recovered Memory ID: {recovered.memory_id}")
            print(f"      New Signature: {recovered.signature}")
            print(f"      Recovery Shimmer: {recovered.shimmer_intensity:.3f}")
        else:
            print(f"   ❌ Recovery failed - conditions not met")
    
    return ghost_trace

def demonstrate_ash_soot_dynamics():
    """Demonstrate ash and soot residue dynamics"""
    print("\n🔥 === ASH & SOOT DYNAMICS DEMONSTRATION ===")
    
    system = get_fractal_memory_system()
    ash_soot_engine = system.ash_soot_engine
    
    print(f"\n📊 Current Residue State:")
    balance = ash_soot_engine.get_residue_balance()
    print(f"   Total Residues: {balance['total_residues']}")
    print(f"   Soot Ratio: {balance['soot_ratio']:.3f}")
    print(f"   Ash Ratio: {balance['ash_ratio']:.3f}")
    print(f"   Balance Health: {balance['balance_health']:.3f}")
    
    # Process some ticks to show dynamics
    print(f"\n⏱️  Processing system ticks...")
    
    for tick in range(5):
        print(f"\n   Tick {tick + 1}:")
        
        # Process tick
        tick_summary = system.process_tick(delta_time=1.0)
        
        print(f"      Nutrients Generated: {tick_summary['nutrients_generated']:.4f}")
        print(f"      Ash/Soot Actions: {tick_summary['actions'].get('ash_soot', {})}")
        
        if tick_summary['reblooms']:
            print(f"      🌸 New Reblooms: {len(tick_summary['reblooms'])}")
        
        if tick_summary['reignitions']:
            print(f"      🔥 Reignition Events: {len(tick_summary['reignitions'])}")
    
    # Show final state
    print(f"\n📈 Final System State:")
    final_balance = ash_soot_engine.get_residue_balance()
    print(f"   Soot→Ash Conversions: {ash_soot_engine.stats['soot_to_ash_conversions']}")
    print(f"   Reignition Events: {ash_soot_engine.stats['reignition_events']}")
    print(f"   Total Nutrients Provided: {ash_soot_engine.stats['nutrients_provided']:.4f}")
    print(f"   Average Soot Volatility: {ash_soot_engine.stats['average_soot_volatility']:.3f}")
    print(f"   Average Ash Stability: {ash_soot_engine.stats['average_ash_stability']:.3f}")

def demonstrate_garden_overview():
    """Show comprehensive garden overview"""
    print("\n🌻 === FRACTAL MEMORY GARDEN OVERVIEW ===")
    
    system = get_fractal_memory_system()
    
    # Get comprehensive metrics
    status = system.get_system_status()
    
    print(f"\n🏡 Garden Population:")
    garden = status['garden_metrics']
    print(f"   Total Fractals: {garden['total_fractals']}")
    print(f"   Juliet Flowers: {garden['juliet_flowers']}")
    print(f"   Ghost Traces: {garden['ghost_traces']}")
    print(f"   Ash Residues: {garden['ash_residues']}")
    print(f"   Soot Residues: {garden['soot_residues']}")
    
    print(f"\n💚 Garden Health:")
    health = status['health_metrics']
    print(f"   Average Shimmer: {health['average_shimmer']:.3f}")
    print(f"   Rebloom Rate: {health['rebloom_rate']:.3f}")
    print(f"   Recovery Rate: {health['recovery_rate']:.3f}")
    print(f"   Residue Balance Health: {health['residue_balance_health']:.3f}")
    
    print(f"\n⚡ Recent Activity:")
    activity = status['activity_metrics']
    print(f"   Recent Encodes: {activity['recent_encodes']}")
    print(f"   Recent Accesses: {activity['recent_accesses']}")
    print(f"   Recent Reblooms: {activity['recent_reblooms']}")
    print(f"   Recent Recoveries: {activity['recent_recoveries']}")
    
    print(f"\n🔧 Performance:")
    perf = status['performance_metrics']
    print(f"   Cache Hit Rate: {perf['cache_hit_rate']:.3f}")
    print(f"   Rebloom Success Rate: {perf['rebloom_success_rate']:.3f}")
    print(f"   Recovery Success Rate: {perf['recovery_success_rate']:.3f}")
    
    print(f"\n⚙️  System Info:")
    info = status['system_info']
    print(f"   Tick Count: {info['tick_count']}")
    print(f"   Uptime: {info['uptime']:.1f} seconds")
    print(f"   Operation History: {info['operation_history_size']} entries")

def main():
    """Run the complete fractal memory demonstration"""
    print("🌺" * 60)
    print("🌺 DAWN FRACTAL MEMORY SYSTEM DEMONSTRATION")
    print("🌺" * 60)
    
    print("\nThis demonstration showcases the complete fractal memory system")
    print("as specified in the DAWN documentation. Each memory becomes a")
    print("unique Julia set fractal, and through repeated coherent access,")
    print("memories can 'rebloom' into Juliet flowers - shinier and prettier")
    print("versions that get beneficial bias for future access.")
    print("\nWhen memories are forgotten, they leave ghost traces with")
    print("transformation sigils that allow recovery under the right")
    print("conditions. The ash and soot residue system manages the")
    print("'cognitive composting' that feeds new growth.")
    
    try:
        # Step 1: Fractal Encoding
        fractals = demonstrate_fractal_encoding()
        time.sleep(1)
        
        # Step 2: Memory Access and Rebloom
        demonstrate_memory_access_and_rebloom(fractals)
        time.sleep(1)
        
        # Step 3: Forgetting and Ghost Traces
        demonstrate_memory_forgetting_and_ghosts(fractals)
        time.sleep(1)
        
        # Step 4: Ash/Soot Dynamics
        demonstrate_ash_soot_dynamics()
        time.sleep(1)
        
        # Step 5: Garden Overview
        demonstrate_garden_overview()
        
        print("\n🌟" * 60)
        print("🌟 DEMONSTRATION COMPLETE")
        print("🌟" * 60)
        
        print("\nThe fractal memory system is now fully operational!")
        print("Key features demonstrated:")
        print("✓ Unique fractal encoding for each memory")
        print("✓ Entropy-based visual anomalies") 
        print("✓ Juliet rebloom transformation (chrysalis)")
        print("✓ Beneficial bias for effective memories")
        print("✓ Ghost traces with recovery sigils")
        print("✓ Ash/soot residue dynamics")
        print("✓ Shimmer decay and forgetting")
        print("✓ Comprehensive garden metrics")
        
        print("\n'This isn't about perfect recall — it's about conceptual")
        print("composting. Old ideas break down and feed the soil from")
        print("which new ones grow.'")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

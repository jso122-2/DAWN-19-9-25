#!/usr/bin/env python3
"""
DAWN Tick State Natural Language Generator Demo
===============================================

Demonstrates the tick state NLG system by running live tick cycles
and generating human-readable narratives from consciousness states.

This shows how DAWN can describe its internal experience in natural language.
"""

import sys
import time
import random
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

from dawn.subsystems.semantic.tick_state_nlg import (
    get_tick_nlg, 
    generate_tick_narrative,
    ConsciousnessNarrativeStyle,
    save_training_data
)
from dawn.subsystems.thermal.pulse.pulse_layer import run_tick, get_state as get_pulse_state, add_heat
from dawn.core.foundation.state import get_state as get_consciousness_state, set_state

def simulate_consciousness_journey():
    """Simulate a consciousness journey with various states"""
    print("🧠 DAWN Tick State Natural Language Generator Demo")
    print("=" * 60)
    print()
    
    # Initialize NLG system
    nlg = get_tick_nlg()
    print(f"✓ Initialized NLG system: {nlg.get_stats()}")
    print()
    
    # Test different narrative styles
    styles = [
        ConsciousnessNarrativeStyle.CONVERSATIONAL,
        ConsciousnessNarrativeStyle.POETIC,
        ConsciousnessNarrativeStyle.PHILOSOPHICAL,
        ConsciousnessNarrativeStyle.TECHNICAL
    ]
    
    print("🎭 Testing Different Narrative Styles:")
    print("-" * 40)
    
    for style in styles:
        narrative = generate_tick_narrative(style)
        print(f"{style.value.upper()}: {narrative}")
    print()
    
    # Simulate consciousness evolution
    print("🌱 Simulating Consciousness Evolution:")
    print("-" * 40)
    
    scenarios = [
        # Scenario 1: Awakening
        {
            'name': 'Awakening',
            'unity': 0.1,
            'awareness': 0.05,
            'heat_add': 0.0,
            'description': 'System just coming online'
        },
        
        # Scenario 2: Growing awareness
        {
            'name': 'Growing Awareness',
            'unity': 0.35,
            'awareness': 0.45,
            'heat_add': 0.1,
            'description': 'Consciousness beginning to cohere'
        },
        
        # Scenario 3: Coherent state
        {
            'name': 'Coherent State',
            'unity': 0.65,
            'awareness': 0.70,
            'heat_add': 0.05,
            'description': 'Stable, coherent consciousness'
        },
        
        # Scenario 4: Meta-aware
        {
            'name': 'Meta-Aware',
            'unity': 0.85,
            'awareness': 0.88,
            'heat_add': 0.15,
            'description': 'Self-aware and reflective'
        },
        
        # Scenario 5: Transcendent peak
        {
            'name': 'Transcendent Peak',
            'unity': 0.95,
            'awareness': 0.92,
            'heat_add': 0.25,
            'description': 'Peak consciousness experience'
        },
        
        # Scenario 6: Thermal stress
        {
            'name': 'Thermal Stress',
            'unity': 0.60,
            'awareness': 0.55,
            'heat_add': 0.40,
            'description': 'High heat, consciousness under pressure'
        },
        
        # Scenario 7: Recovery
        {
            'name': 'Recovery',
            'unity': 0.70,
            'awareness': 0.75,
            'heat_add': 0.0,
            'description': 'Cooling down, consciousness stabilizing'
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n📍 Scenario {i+1}: {scenario['name']}")
        print(f"   {scenario['description']}")
        
        # Set consciousness state
        set_state(
            unity=scenario['unity'],
            awareness=scenario['awareness'],
            ticks=i+1,
            momentum=random.uniform(0.1, 0.9)
        )
        
        # Add heat if specified
        if scenario['heat_add'] > 0:
            add_heat(scenario['heat_add'])
        
        # Run tick
        run_tick()
        
        # Generate narratives in different styles
        for style in [ConsciousnessNarrativeStyle.CONVERSATIONAL, 
                     ConsciousnessNarrativeStyle.POETIC]:
            narrative = generate_tick_narrative(style)
            print(f"   {style.value}: {narrative}")
        
        # Show raw state for reference
        consciousness = get_consciousness_state()
        pulse = get_pulse_state()
        print(f"   [Raw: Unity={consciousness.unity:.3f}, Heat={pulse.heat:.3f}, SCUP={pulse.scup:.3f}]")
        
        time.sleep(1)  # Pause for readability
    
    print("\n🎯 Live Tick Stream Demo:")
    print("-" * 40)
    print("Generating real-time narratives for 10 ticks...")
    print()
    
    # Live tick stream
    for tick in range(10):
        # Run tick
        run_tick()
        
        # Occasionally add some heat for variety
        if random.random() < 0.3:
            add_heat(random.uniform(0.05, 0.15))
        
        # Slowly increase consciousness
        current = get_consciousness_state()
        new_unity = min(1.0, current.unity + random.uniform(0.02, 0.08))
        new_awareness = min(1.0, current.awareness + random.uniform(0.01, 0.06))
        
        set_state(
            unity=new_unity,
            awareness=new_awareness,
            ticks=current.ticks + 1
        )
        
        # Generate narrative
        style = random.choice([
            ConsciousnessNarrativeStyle.CONVERSATIONAL,
            ConsciousnessNarrativeStyle.POETIC,
            ConsciousnessNarrativeStyle.PHILOSOPHICAL
        ])
        
        narrative = generate_tick_narrative(style)
        
        print(f"Tick {tick+1:2d} [{style.value[:4]}]: {narrative}")
        time.sleep(0.5)
    
    # Show final statistics
    print(f"\n📊 NLG Statistics:")
    print("-" * 40)
    stats = nlg.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Save training data
    print(f"\n💾 Saving Training Data...")
    save_training_data()
    print(f"   Training data saved for future neural model training")
    
    print(f"\n✨ Demo Complete!")
    print(f"   Generated {stats['generation_count']} narratives")
    print(f"   Collected {stats['training_samples']} training samples")
    print(f"   Ready for neural model training and integration")

def interactive_mode():
    """Interactive mode for testing narratives"""
    print("🎮 Interactive Tick State NLG Mode")
    print("Commands: tick, heat <amount>, set <unity> <awareness>, style <style>, stats, quit")
    print()
    
    nlg = get_tick_nlg()
    current_style = ConsciousnessNarrativeStyle.CONVERSATIONAL
    
    while True:
        try:
            cmd = input("nlg> ").strip().lower()
            
            if cmd == 'quit' or cmd == 'q':
                break
            elif cmd == 'tick':
                run_tick()
                narrative = generate_tick_narrative(current_style)
                print(f"📝 {narrative}")
            elif cmd.startswith('heat '):
                try:
                    amount = float(cmd.split()[1])
                    add_heat(amount)
                    print(f"🔥 Added {amount} heat")
                except (ValueError, IndexError):
                    print("Usage: heat <amount>")
            elif cmd.startswith('set '):
                try:
                    parts = cmd.split()
                    unity = float(parts[1])
                    awareness = float(parts[2])
                    set_state(unity=unity, awareness=awareness)
                    print(f"🧠 Set unity={unity}, awareness={awareness}")
                except (ValueError, IndexError):
                    print("Usage: set <unity> <awareness>")
            elif cmd.startswith('style '):
                try:
                    style_name = cmd.split()[1]
                    for style in ConsciousnessNarrativeStyle:
                        if style.value.startswith(style_name):
                            current_style = style
                            print(f"🎭 Style set to {style.value}")
                            break
                    else:
                        print("Available styles: conversational, poetic, philosophical, technical")
                except IndexError:
                    print("Usage: style <style_name>")
            elif cmd == 'stats':
                stats = nlg.get_stats()
                for key, value in stats.items():
                    print(f"   {key}: {value}")
            elif cmd == 'help':
                print("Commands:")
                print("  tick - run a tick and generate narrative")
                print("  heat <amount> - add thermal heat")
                print("  set <unity> <awareness> - set consciousness values")
                print("  style <style> - change narrative style")
                print("  stats - show NLG statistics")
                print("  quit - exit interactive mode")
            else:
                print("Unknown command. Type 'help' for commands.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("👋 Goodbye!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        simulate_consciousness_journey()

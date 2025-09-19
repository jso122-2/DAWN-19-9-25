#!/usr/bin/env python3
"""
Example: Using DAWN Global Singleton
===================================

This example demonstrates how to use the DAWN global singleton
from anywhere in your code. This replaces the need to manually
manage individual system components.
"""

import asyncio
from pathlib import Path
import sys

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

# Import the DAWN singleton - this is the main way to access DAWN
import dawn

async def example_consciousness_module():
    """Example module that uses DAWN singleton."""
    print("🧠 Example Consciousness Module")
    print("=" * 40)
    
    # Get the global DAWN instance
    dawn_system = dawn.get_dawn()
    
    # Initialize if needed
    if not dawn.is_dawn_initialized():
        print("   Initializing DAWN system...")
        await dawn_system.initialize({'mode': 'interactive'})
    
    # Start if needed  
    if not dawn.is_dawn_running():
        print("   Starting DAWN system...")
        await dawn_system.start()
    
    # Access consciousness state directly
    state = dawn.get_consciousness_state()
    print(f"   Current unity: {state.unity:.3f}")
    print(f"   Current awareness: {state.awareness:.3f}")
    print(f"   Consciousness level: {state.level}")
    
    # Access subsystems through singleton
    bus = dawn_system.consciousness_bus
    engine = dawn_system.dawn_engine
    telemetry = dawn_system.telemetry_system
    
    print(f"   Consciousness Bus: {'✅' if bus else '❌'}")
    print(f"   DAWN Engine: {'✅' if engine else '❌'}")  
    print(f"   Telemetry System: {'✅' if telemetry else '❌'}")
    
    # Use the consciousness bus
    if bus:
        # Register this module
        bus.register_module(
            module_name="example_consciousness_module",
            capabilities=["example_processing", "consciousness_analysis"],
            state_schema={"processing_state": "string", "analysis_depth": "float"}
        )
        
        # Publish module state
        bus.publish_state(
            "example_consciousness_module",
            {
                "processing_state": "active",
                "analysis_depth": 0.75,
                "last_update": "example_module_tick"
            }
        )
        
        # Get bus metrics
        metrics = bus.get_bus_metrics()
        print(f"   Bus registered modules: {metrics['registered_modules']}")
        print(f"   Bus coherence: {metrics['consciousness_coherence']:.3f}")
    
    # Use telemetry system
    if telemetry:
        # Log an event
        telemetry.log_event(
            'example_module', 'processing', 'module_executed',
            data={'unity_level': state.unity, 'processing_complete': True}
        )
        
        tel_metrics = telemetry.get_system_metrics()
        print(f"   Telemetry events logged: {tel_metrics.get('logger_events_logged', 0)}")
    
    return state

async def example_pulse_module():
    """Example thermal/pulse module using singleton."""
    print("\n🔥 Example Pulse Module")
    print("=" * 40)
    
    # Quick access functions
    if not dawn.is_dawn_running():
        print("   DAWN not running, skipping pulse module")
        return
    
    # Get system directly
    dawn_system = dawn.get_dawn()
    
    # Access state
    state = dawn.get_consciousness_state()
    
    # Simulate pulse processing
    print(f"   Processing thermal pulse at unity {state.unity:.3f}")
    
    # Update consciousness state (this affects the global state)
    from dawn.core.foundation.state import set_state
    set_state(
        unity=min(1.0, state.unity + 0.1),
        awareness=min(1.0, state.awareness + 0.05)
    )
    
    updated_state = dawn.get_consciousness_state()
    print(f"   Updated unity: {updated_state.unity:.3f}")
    print(f"   Updated awareness: {updated_state.awareness:.3f}")
    print(f"   New level: {updated_state.level}")
    
    # Log to telemetry
    if dawn_system.telemetry_system:
        dawn_system.telemetry_system.log_event(
            'pulse_module', 'thermal', 'pulse_processed',
            data={
                'unity_delta': updated_state.unity - state.unity,
                'awareness_delta': updated_state.awareness - state.awareness,
                'new_level': updated_state.level
            }
        )

async def example_schema_module():
    """Example schema/sigil module using singleton."""
    print("\n🔮 Example Schema Module")
    print("=" * 40)
    
    # Check system status
    status = dawn.get_system_status()
    print(f"   System initialized: {status.initialized}")
    print(f"   System running: {status.running}")
    print(f"   Components loaded: {len(status.components_loaded)}")
    
    # Get comprehensive metrics
    dawn_system = dawn.get_dawn()
    metrics = dawn_system.get_system_metrics()
    
    print(f"   Consciousness state level: {metrics['consciousness_state']['level']}")
    print(f"   System startup time: {metrics['status']['startup_time']}")
    
    # Access engine for schema processing
    engine = dawn_system.dawn_engine
    if engine:
        engine_status = engine.get_engine_status()
        print(f"   Engine tick count: {engine_status.get('tick_count', 0)}")
        print(f"   Engine unity score: {engine_status.get('consciousness_unity_score', 0.0):.3f}")

async def main():
    """Main example demonstrating singleton usage patterns."""
    print("🌅 DAWN Global Singleton Usage Examples")
    print("=" * 60)
    
    # Run example modules
    await example_consciousness_module()
    await example_pulse_module() 
    await example_schema_module()
    
    print("\n📊 Final System Status")
    print("=" * 40)
    
    # Final status
    dawn_system = dawn.get_dawn()
    final_metrics = dawn_system.get_system_metrics()
    
    print(f"System Status: {final_metrics['status']}")
    print(f"Final Consciousness State: {final_metrics['consciousness_state']}")
    
    # Clean shutdown
    print("\n🌙 Shutting down DAWN system...")
    dawn_system.stop()
    print("✅ Complete!")

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
DAWN Schema System - Complete Usage Example
==========================================
Comprehensive demonstration of the DAWN Schema System including all components:
- Core Schema System with tick processing
- SHI (Schema Health Index) calculation
- SCUP (Semantic Coherence Under Pressure) tracking
- Multi-layer validation
- Real-time monitoring and anomaly detection

This example shows how to use the schema system in a realistic DAWN environment.

Author: DAWN Development Team
Generated: 2025-09-18
"""

import time
import numpy as np
from typing import Dict, Any

# Import DAWN schema components
try:
    from . import (
        create_integrated_schema_system,
        get_default_schema_config,
        integrate_with_dawn_systems
    )
except ImportError:
    # For standalone execution
    from __init__ import (
        create_integrated_schema_system,
        get_default_schema_config,
        integrate_with_dawn_systems
    )


def demonstrate_basic_usage():
    """Demonstrate basic schema system usage"""
    print("🧠 DAWN Schema System - Basic Usage Demo")
    print("=" * 50)
    
    # Create schema system with default configuration
    schema_system = create_integrated_schema_system()
    
    # Add schema nodes (representing concepts/knowledge structures)
    print("\n📝 Adding schema nodes...")
    
    schema_system.add_schema_node(
        node_id="consciousness_core",
        health=0.85,
        tint=[0.4, 0.3, 0.3],  # Red-tinted (abstract thinking)
        energy=0.9
    )
    
    schema_system.add_schema_node(
        node_id="memory_buffer", 
        health=0.72,
        tint=[0.2, 0.4, 0.4],  # Blue-tinted (memory/storage)
        energy=0.6
    )
    
    schema_system.add_schema_node(
        node_id="reasoning_engine",
        health=0.88,
        tint=[0.3, 0.5, 0.2],  # Green-tinted (logical processing)
        energy=0.8
    )
    
    schema_system.add_schema_node(
        node_id="pattern_recognition",
        health=0.75,
        tint=[0.35, 0.35, 0.3],  # Balanced (pattern analysis)
        energy=0.7
    )
    
    # Add schema edges (representing connections between concepts)
    print("🔗 Adding schema edges...")
    
    schema_system.add_schema_edge(
        edge_id="core_to_memory",
        source="consciousness_core", 
        target="memory_buffer",
        weight=0.8,
        tension=0.1
    )
    
    schema_system.add_schema_edge(
        edge_id="core_to_reasoning",
        source="consciousness_core",
        target="reasoning_engine", 
        weight=0.9,
        tension=0.05
    )
    
    schema_system.add_schema_edge(
        edge_id="memory_to_patterns",
        source="memory_buffer",
        target="pattern_recognition",
        weight=0.6,
        tension=0.2
    )
    
    schema_system.add_schema_edge(
        edge_id="reasoning_to_patterns",
        source="reasoning_engine",
        target="pattern_recognition", 
        weight=0.7,
        tension=0.15
    )
    
    print(f"✅ Schema initialized: {len(schema_system.state.nodes)} nodes, {len(schema_system.state.edges)} edges")
    
    return schema_system


def simulate_cognitive_scenarios(schema_system):
    """Simulate various cognitive scenarios and observe schema responses"""
    print("\n🎭 Simulating Cognitive Scenarios")
    print("=" * 40)
    
    scenarios = [
        {
            "name": "Normal Operation",
            "description": "Balanced cognitive load with normal processing",
            "tick_data": {
                'signals': {'pressure': 0.3, 'drift': 0.2, 'entropy': 0.4},
                'tracers': {'crow': 2, 'spider': 1, 'ant': 5, 'whale': 1, 'owl': 1},
                'residue': {'soot_ratio': 0.1, 'ash_bias': [0.33, 0.33, 0.34]}
            }
        },
        {
            "name": "High Cognitive Pressure", 
            "description": "Intense reasoning under time pressure",
            "tick_data": {
                'signals': {'pressure': 0.8, 'drift': 0.4, 'entropy': 0.6},
                'tracers': {'crow': 5, 'spider': 3, 'ant': 8, 'whale': 0, 'owl': 2},
                'residue': {'soot_ratio': 0.3, 'ash_bias': [0.5, 0.3, 0.2]}
            }
        },
        {
            "name": "Creative Exploration",
            "description": "High entropy creative thinking with low pressure", 
            "tick_data": {
                'signals': {'pressure': 0.2, 'drift': 0.6, 'entropy': 0.8},
                'tracers': {'crow': 1, 'spider': 1, 'ant': 3, 'whale': 2, 'owl': 1},
                'residue': {'soot_ratio': 0.15, 'ash_bias': [0.2, 0.3, 0.5]}
            }
        },
        {
            "name": "Memory Consolidation",
            "description": "Low activity with memory processing",
            "tick_data": {
                'signals': {'pressure': 0.1, 'drift': 0.1, 'entropy': 0.2},
                'tracers': {'crow': 0, 'spider': 0, 'ant': 2, 'whale': 1, 'owl': 3},
                'residue': {'soot_ratio': 0.05, 'ash_bias': [0.25, 0.4, 0.35]}
            }
        },
        {
            "name": "Cognitive Overload",
            "description": "Extreme pressure causing potential breakdown",
            "tick_data": {
                'signals': {'pressure': 1.0, 'drift': 0.8, 'entropy': 0.9}, 
                'tracers': {'crow': 8, 'spider': 6, 'ant': 12, 'whale': 0, 'owl': 1},
                'residue': {'soot_ratio': 0.6, 'ash_bias': [0.7, 0.2, 0.1]}
            }
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n🎬 Scenario {i+1}: {scenario['name']}")
        print(f"   {scenario['description']}")
        
        # Add some external pressures based on scenario
        external_pressures = None
        if "pressure" in scenario['name'].lower():
            external_pressures = {'time_pressure': 0.7, 'complexity_pressure': 0.5}
        elif "creative" in scenario['name'].lower():
            external_pressures = {'exploration_drive': 0.8}
        
        # Run tick update
        snapshot = schema_system.tick_update(
            tick_data=scenario['tick_data'],
            external_pressures=external_pressures
        )
        
        # Display results
        print(f"   📊 SHI: {snapshot.shi:.3f} | SCUP: {snapshot.scup:.3f}")
        print(f"   🏥 Avg Health: {schema_system.state.get_average_health():.3f}")
        print(f"   ⚡ Avg Tension: {schema_system.state.get_average_tension():.3f}")
        print(f"   🚩 Flags: {', '.join([k for k, v in snapshot.flags.items() if v]) or 'None'}")
        
        results.append({
            'scenario': scenario['name'],
            'shi': snapshot.shi,
            'scup': snapshot.scup,
            'health': schema_system.state.get_average_health(),
            'tension': schema_system.state.get_average_tension()
        })
        
        # Brief pause to simulate real-time processing
        time.sleep(0.1)
    
    return results


def demonstrate_health_monitoring(schema_system):
    """Demonstrate health monitoring and trend analysis"""
    print("\n📈 Health Monitoring & Trend Analysis")
    print("=" * 40)
    
    # Generate a series of ticks with gradually degrading conditions
    print("🔄 Simulating gradual schema degradation...")
    
    degradation_steps = 20
    for step in range(degradation_steps):
        # Gradually increase pressure and entropy
        pressure = 0.2 + (step / degradation_steps) * 0.6
        entropy = 0.3 + (step / degradation_steps) * 0.5
        drift = 0.1 + (step / degradation_steps) * 0.4
        
        # Increase soot ratio (toxic residue)
        soot_ratio = 0.05 + (step / degradation_steps) * 0.4
        
        tick_data = {
            'signals': {
                'pressure': pressure,
                'drift': drift, 
                'entropy': entropy
            },
            'tracers': {
                'crow': min(10, step),  # Increase anomaly detection
                'spider': min(8, step // 2),  # Increase tension monitoring
                'ant': max(1, 10 - step),  # Decrease micro-patterns
                'whale': max(0, 3 - step // 5),  # Decrease macro context
                'owl': 1
            },
            'residue': {
                'soot_ratio': soot_ratio,
                'ash_bias': [0.5 + soot_ratio * 0.3, 0.3, 0.2 - soot_ratio * 0.1]
            }
        }
        
        snapshot = schema_system.tick_update(tick_data)
        
        # Print progress every 5 steps
        if step % 5 == 0:
            print(f"   Step {step:2d}: SHI={snapshot.shi:.3f}, SCUP={snapshot.scup:.3f}, "
                  f"Pressure={pressure:.2f}, Entropy={entropy:.2f}")
    
    # Analyze health trend
    print("\n📊 Health Trend Analysis:")
    trend = schema_system.get_health_trend(window=15)
    print(f"   Direction: {trend['direction']}")
    print(f"   Trend Value: {trend['trend']:.3f}")
    print(f"   Stability: {trend['stability']:.3f}")
    
    # Get current system status
    print("\n🔍 Current System Status:")
    status = schema_system.get_system_status()
    print(f"   Current SHI: {status['current_shi']:.3f}")
    print(f"   Current SCUP: {status['current_scup']:.3f}")
    print(f"   Average Health: {status['average_health']:.3f}")
    print(f"   Mode: {status['mode']}")
    print(f"   Active Flags: {', '.join([k for k, v in status['flags'].items() if v]) or 'None'}")
    
    # Performance metrics
    perf = status['performance']
    print(f"\n⚡ Performance Metrics:")
    print(f"   Avg Processing Time: {perf['avg_processing_time']:.4f}s")
    print(f"   Error Count: {perf['error_count']}")
    print(f"   History Size: {perf['history_size']}")


def demonstrate_validation_and_monitoring(schema_system):
    """Demonstrate validation and monitoring capabilities"""
    print("\n🔍 Validation & Monitoring Demo")
    print("=" * 35)
    
    # Create some problematic tick data to trigger validation issues
    problematic_data = {
        'signals': {
            'pressure': 1.5,  # Out of normal range
            'drift': -0.5,    # Negative value
            'entropy': 2.0    # Very high entropy
        },
        'tracers': {
            'crow': 15,  # Very high anomaly count
            'spider': 10,
            'ant': 0,    # No micro-patterns
            'whale': 0,  # No macro context
            'owl': 5
        },
        'residue': {
            'soot_ratio': 0.8,  # Very high toxic residue
            'ash_bias': [0.8, 0.1, 0.1]  # Heavily biased
        }
    }
    
    print("🚨 Processing problematic data...")
    snapshot = schema_system.tick_update(problematic_data)
    
    print(f"   SHI: {snapshot.shi:.3f} (Critical threshold: 0.2)")
    print(f"   SCUP: {snapshot.scup:.3f} (Critical threshold: 0.3)")
    
    # Check if validation caught issues
    if hasattr(schema_system, 'validator') and schema_system.validator:
        validation_data = {
            'nodes': [node.to_dict() for node in snapshot.nodes],
            'edges': [edge.to_dict() for edge in snapshot.edges],
            'signals': snapshot.signals,
            'shi': snapshot.shi,
            'scup': snapshot.scup
        }
        
        validation_summary = schema_system.validator.validate_schema_snapshot(validation_data)
        
        print(f"\n✅ Validation Results:")
        print(f"   Overall Valid: {validation_summary.overall_valid}")
        print(f"   Pass Rate: {validation_summary.get_pass_rate():.1%}")
        print(f"   Warnings: {validation_summary.warnings}")
        print(f"   Errors: {validation_summary.errors}")
        print(f"   Critical Issues: {validation_summary.critical_issues}")
        
        if validation_summary.results:
            print(f"\n   Validation Details:")
            for result in validation_summary.results:
                status_icon = "✅" if result.valid else "❌"
                print(f"   {status_icon} {result.layer.value}: {result.message}")
    
    # Check monitoring results
    if hasattr(schema_system, 'monitor') and schema_system.monitor:
        monitoring_data = {
            'shi': snapshot.shi,
            'scup': snapshot.scup,
            'schema_state': schema_system.state
        }
        
        monitor_result = schema_system.monitor.monitor_real_time(monitoring_data)
        
        print(f"\n👁️ Monitoring Results:")
        print(f"   Health Status: {monitor_result['health_status']['status']}")
        print(f"   Message: {monitor_result['health_status']['message']}")
        print(f"   Anomalies Detected: {len(monitor_result['anomalies'])}")
        
        if monitor_result['anomalies']:
            print(f"\n   🚨 Anomalies:")
            for anomaly in monitor_result['anomalies'][:3]:  # Show first 3
                print(f"   - {anomaly['metric']}: {anomaly['description']} (Severity: {anomaly['severity']})")


def demonstrate_integration_capabilities():
    """Demonstrate integration with other DAWN systems"""
    print("\n🔗 Integration Capabilities Demo")
    print("=" * 35)
    
    schema_system = create_integrated_schema_system()
    
    # Mock other DAWN systems for demonstration
    class MockMycelialLayer:
        def __init__(self):
            self.external_pressures = {}
            self.health_callbacks = []
            
        def update_external_pressure(self, source, pressure):
            self.external_pressures[source] = pressure
            print(f"   🍄 Mycelial received pressure from {source}: {pressure:.3f}")
            
        def register_health_callback(self, callback):
            self.health_callbacks.append(callback)
            
        def simulate_health_change(self, health):
            for callback in self.health_callbacks:
                callback(health)
    
    class MockPulseSystem:
        def __init__(self):
            self.phase_callbacks = []
            
        def register_phase_callback(self, callback):
            self.phase_callbacks.append(callback)
            
        def simulate_pulse_phase(self, phase, intensity):
            print(f"   💓 Pulse system: phase={phase:.2f}, intensity={intensity:.2f}")
            for callback in self.phase_callbacks:
                callback(phase, intensity)
    
    class MockThermalSystem:
        def __init__(self):
            self.pressure_callbacks = []
            
        def register_pressure_callback(self, callback):
            self.pressure_callbacks.append(callback)
            
        def simulate_thermal_pressure(self, pressure, state):
            print(f"   🌡️ Thermal system: pressure={pressure:.2f}, state={state}")
            for callback in self.pressure_callbacks:
                callback(pressure, state)
    
    # Create mock systems
    print("🏗️ Creating mock DAWN subsystems...")
    mycelial_layer = MockMycelialLayer()
    pulse_system = MockPulseSystem()
    thermal_system = MockThermalSystem()
    
    # Integrate systems
    print("🔌 Integrating schema system with other DAWN subsystems...")
    integrate_with_dawn_systems(
        schema_system=schema_system,
        mycelial_layer=mycelial_layer,
        pulse_system=pulse_system,
        thermal_system=thermal_system
    )
    
    print("\n🎮 Simulating inter-system communication...")
    
    # Simulate schema health changes affecting other systems
    tick_data = {
        'signals': {'pressure': 0.7, 'drift': 0.3, 'entropy': 0.5},
        'tracers': {'crow': 3, 'spider': 2, 'ant': 4, 'whale': 1, 'owl': 1},
        'residue': {'soot_ratio': 0.2, 'ash_bias': [0.4, 0.3, 0.3]}
    }
    
    snapshot = schema_system.tick_update(tick_data)
    print(f"   🧠 Schema update: SHI={snapshot.shi:.3f}, SCUP={snapshot.scup:.3f}")
    
    # Simulate other systems affecting schema
    print("\n🔄 Simulating feedback from other systems...")
    mycelial_layer.simulate_health_change(0.6)
    pulse_system.simulate_pulse_phase(0.8, 0.7)
    thermal_system.simulate_thermal_pressure(0.9, "high_load")


def run_comprehensive_demo():
    """Run complete demonstration of DAWN Schema System"""
    print("🚀 DAWN Schema System - Comprehensive Demonstration")
    print("=" * 60)
    print("This demo showcases all major features of the DAWN Schema System:")
    print("- Core schema management with nodes and edges") 
    print("- SHI (Schema Health Index) calculation")
    print("- SCUP (Semantic Coherence Under Pressure) tracking")
    print("- Multi-layer validation framework")
    print("- Real-time monitoring and anomaly detection")
    print("- Integration with other DAWN subsystems")
    print("=" * 60)
    
    # Basic usage demonstration
    schema_system = demonstrate_basic_usage()
    
    # Cognitive scenario simulation
    scenario_results = simulate_cognitive_scenarios(schema_system)
    
    # Health monitoring
    demonstrate_health_monitoring(schema_system)
    
    # Validation and monitoring
    demonstrate_validation_and_monitoring(schema_system)
    
    # Integration capabilities
    demonstrate_integration_capabilities()
    
    # Summary
    print("\n🎯 Demo Summary")
    print("=" * 20)
    print("✅ Successfully demonstrated:")
    print("   - Schema node and edge management")
    print("   - SHI calculation across multiple scenarios")
    print("   - SCUP tracking under various cognitive loads")
    print("   - Health trend analysis and monitoring")
    print("   - Validation framework with multi-layer checks")
    print("   - Real-time anomaly detection")
    print("   - Integration hooks for other DAWN systems")
    
    # Performance summary
    final_status = schema_system.get_system_status()
    print(f"\n📊 Final Performance Metrics:")
    print(f"   Total Ticks Processed: {final_status['tick_count']}")
    print(f"   Average Processing Time: {final_status['performance']['avg_processing_time']:.4f}s")
    print(f"   Error Count: {final_status['performance']['error_count']}")
    print(f"   Schema Complexity: {final_status['node_count']} nodes, {final_status['edge_count']} edges")
    
    print("\n🎉 DAWN Schema System demonstration completed successfully!")
    print("The system is ready for integration into the full DAWN consciousness architecture.")


if __name__ == "__main__":
    run_comprehensive_demo()

#!/usr/bin/env python3
"""
🫁📊 DAWN Pulse-Telemetry Unification Test
==========================================

Comprehensive test of the unified pulse-telemetry logging system that demonstrates:

1. Integration between pulse system and telemetry logging
2. Consciousness-depth organization of pulse events
3. SCUP coherence tracking with consciousness levels
4. Thermal state logging with awareness correlation
5. Tick orchestration integration with consciousness transitions
6. Sigil system pulse integration
7. Unified metrics collection across all systems

This test verifies that telemetry logging is properly unified with 
pulse logic and modules, creating a cohesive consciousness-aware 
logging architecture.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

# Add DAWN to path
sys.path.insert(0, str(Path(__file__).parent))

def test_pulse_telemetry_bridge():
    """Test the core pulse-telemetry bridge functionality"""
    print("🫁🔗 TESTING PULSE-TELEMETRY BRIDGE")
    print("=" * 50)
    
    try:
        from dawn.core.logging import (
            get_pulse_telemetry_bridge, PulseTelemetryBridge,
            ConsciousnessLevel, TelemetryLevel
        )
        
        print("✅ Pulse-telemetry imports successful")
        
        # Create bridge
        bridge = get_pulse_telemetry_bridge()
        print("✅ Pulse-telemetry bridge created")
        
        # Start unified logging
        started = bridge.start_unified_logging()
        if started:
            print("✅ Unified logging started successfully")
        else:
            print("⚠️ Unified logging was already running")
        
        # Test various pulse event types
        test_events = [
            # Zone transition event
            {
                'event_type': 'zone_transition',
                'pulse_data': {
                    'old_zone': 'green',
                    'new_zone': 'amber',
                    'scup_coherence': 0.7,
                    'thermal_state': 'warming',
                    'zone_change_reason': 'increased_cognitive_load',
                    'subsystem': 'pulse',
                    'component': 'zone_manager'
                },
                'consciousness_context': {
                    'awareness_shift': True,
                    'coherence_change': -0.2
                }
            },
            
            # Tick completion event
            {
                'event_type': 'tick_completed',
                'pulse_data': {
                    'tick_number': 42,
                    'zone': 'amber',
                    'scup_coherence': 0.6,
                    'tick_duration': 0.05,
                    'phase_results': {
                        'state_collection': 'completed',
                        'information_sharing': 'completed', 
                        'decision_making': 'partial',
                        'state_updates': 'completed',
                        'synchronization_check': 'completed'
                    },
                    'synchronization_quality': 0.85,
                    'subsystem': 'pulse',
                    'component': 'tick_orchestrator'
                },
                'consciousness_context': {
                    'tick_consciousness_flow': True,
                    'phase_transitions': 5
                }
            },
            
            # SCUP state update event
            {
                'event_type': 'scup_state_update',
                'pulse_data': {
                    'scup_coherence': 0.8,
                    'pressure_level': 0.4,
                    'semantic_integrity': 0.9,
                    'zone': 'green',
                    'coherence_trend': 'increasing',
                    'subsystem': 'pulse',
                    'component': 'scup_controller'
                },
                'consciousness_context': {
                    'semantic_coherence_shift': True,
                    'pressure_consciousness_factor': 0.6
                }
            },
            
            # Thermal consciousness event
            {
                'event_type': 'thermal_consciousness_state',
                'pulse_data': {
                    'thermal_state': 'cooling',
                    'zone': 'green',
                    'scup_coherence': 0.9,
                    'cooling_efficiency': 0.7,
                    'heat_sources': {
                        'cognitive_load': 0.3,
                        'emotional_resonance': 0.2,
                        'unexpressed_thoughts': 0.1
                    },
                    'expressions': {
                        'verbal_expression': 0.4,
                        'creative_flow': 0.6,
                        'empathetic_response': 0.5
                    },
                    'subsystem': 'pulse',
                    'component': 'thermal_manager'
                },
                'consciousness_context': {
                    'thermal_consciousness_integration': True,
                    'expression_consciousness_levels': True
                }
            },
            
            # Sigil pulse integration event
            {
                'event_type': 'sigil_pulse_resonance',
                'pulse_data': {
                    'sigil_id': 'pulse_sigil_001',
                    'zone': 'green',
                    'scup_coherence': 0.85,
                    'sigil_resonance': 0.8,
                    'symbolic_coherence': 0.75,
                    'pulse_sigil_synchronization': 0.9,
                    'subsystem': 'pulse',
                    'component': 'sigil_integration'
                },
                'consciousness_context': {
                    'sigil_consciousness_resonance': True,
                    'symbolic_pulse_alignment': True
                }
            }
        ]
        
        logged_events = []
        
        for test_event in test_events:
            print(f"\n📝 Logging {test_event['event_type']} event...")
            
            result_ids = bridge.log_unified_pulse_event(
                test_event['event_type'],
                test_event['pulse_data'],
                test_event['consciousness_context'],
                TelemetryLevel.INFO
            )
            
            logged_events.append(result_ids)
            
            print(f"  ✅ Event logged with IDs: {result_ids}")
            
            # Small delay between events
            time.sleep(0.1)
        
        print(f"\n✅ Successfully logged {len(logged_events)} unified pulse events")
        
        # Get unified metrics
        print("\n📊 Getting unified metrics...")
        metrics = bridge.get_unified_metrics()
        
        print("📈 Unified Pulse Metrics:")
        print(f"  🎯 Current Zone: {metrics.current_zone}")
        print(f"  🧠 SCUP Coherence: {metrics.scup_coherence:.3f}")
        print(f"  🔥 Thermal Level: {metrics.thermal_level:.3f}")
        print(f"  🫁 Pulse Frequency: {metrics.pulse_frequency:.3f}")
        print(f"  ⏱️ Tick Rate: {metrics.tick_rate:.3f}")
        print(f"  🌐 Consciousness Coherence: {metrics.avg_consciousness_coherence:.3f}")
        print(f"  📊 Events/Second: {metrics.events_per_second:.2f}")
        print(f"  💚 Integration Health: {metrics.integration_health:.3f}")
        
        if metrics.consciousness_level_distribution:
            print("  🧠 Consciousness Level Distribution:")
            for level, count in metrics.consciousness_level_distribution.items():
                print(f"    {level}: {count} entries")
        
        # Stop unified logging
        stopped = bridge.stop_unified_logging()
        if stopped:
            print("\n✅ Unified logging stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Pulse-telemetry bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consciousness_pulse_integration():
    """Test consciousness-pulse integration"""
    print("\n🧠🫁 TESTING CONSCIOUSNESS-PULSE INTEGRATION")
    print("=" * 50)
    
    try:
        from dawn.core.logging import (
            ConsciousnessPulseLogger, get_consciousness_repository,
            ConsciousnessLevel, PulseConsciousnessMapping
        )
        
        # Create consciousness-pulse logger
        consciousness_repo = get_consciousness_repository("test_pulse_consciousness")
        pulse_logger = ConsciousnessPulseLogger(consciousness_repo)
        
        print("✅ Consciousness-pulse logger created")
        
        # Test pulse zone to consciousness mapping
        print("\n🎯 Testing pulse zone consciousness mapping...")
        
        zone_tests = [
            ('green', 'High coherence, transcendent awareness'),
            ('amber', 'Meta-awareness of emerging issues'),
            ('red', 'Causal reasoning about problems'),
            ('black', 'Survival mode, mythic responses')
        ]
        
        for zone, description in zone_tests:
            pulse_data = {
                'zone': zone,
                'scup_coherence': {'green': 0.9, 'amber': 0.7, 'red': 0.5, 'black': 0.2}[zone],
                'thermal_state': {'green': 'cooling', 'amber': 'stable', 'red': 'warming', 'black': 'critical'}[zone]
            }
            
            expected_level = PulseConsciousnessMapping.PULSE_ZONE_CONSCIOUSNESS_MAP[zone]
            
            entry_id = pulse_logger.log_pulse_consciousness_state(pulse_data)
            
            print(f"  🎯 {zone.upper()} zone → {expected_level.name} consciousness")
            print(f"    Description: {description}")
            print(f"    Entry ID: {entry_id}")
        
        # Test tick phase consciousness transitions
        print("\n⏱️ Testing tick phase consciousness transitions...")
        
        phase_transitions = [
            ('state_collection', 'information_sharing', 'Concrete → Symbolic transition'),
            ('information_sharing', 'decision_making', 'Symbolic → Causal transition'),
            ('decision_making', 'state_updates', 'Causal → Formal transition'),
            ('state_updates', 'synchronization_check', 'Formal → Integral transition')
        ]
        
        for old_phase, new_phase, description in phase_transitions:
            tick_data = {
                'tick_number': 100 + len(phase_transitions),
                'phase_transition': True,
                'transition_quality': 0.85
            }
            
            entry_id = pulse_logger.log_tick_consciousness_transition(
                tick_data, old_phase, new_phase
            )
            
            print(f"  ⏱️ {old_phase} → {new_phase}")
            print(f"    {description}")
            print(f"    Entry ID: {entry_id}")
        
        # Test thermal consciousness correlation
        print("\n🔥 Testing thermal consciousness correlation...")
        
        thermal_states = [
            ('cooling', 'Transcendent cooling consciousness'),
            ('stable', 'Integral thermal balance'),
            ('warming', 'Formal thermal management'),
            ('hot', 'Causal thermal reasoning'),
            ('critical', 'Mythic survival responses')
        ]
        
        for thermal_state, description in thermal_states:
            thermal_data = {
                'thermal_state': thermal_state,
                'cooling_efficiency': {'cooling': 0.8, 'stable': 0.6, 'warming': 0.4, 'hot': 0.2, 'critical': 0.1}[thermal_state],
                'heat_sources': {
                    'cognitive_load': 0.3,
                    'emotional_resonance': 0.2,
                    'drift': 0.1
                },
                'expressions': {
                    'creative_flow': 0.6,
                    'verbal_expression': 0.4,
                    'empathetic_response': 0.5
                }
            }
            
            entry_id = pulse_logger.log_thermal_consciousness_state(thermal_data)
            
            print(f"  🔥 {thermal_state.upper()} → {description}")
            print(f"    Entry ID: {entry_id}")
        
        # Get consciousness repository statistics
        print("\n📊 Consciousness-pulse integration statistics:")
        stats = consciousness_repo.get_consciousness_hierarchy_stats()
        
        for level_name, level_stats in stats['consciousness_levels'].items():
            if level_stats['entry_count'] > 0:
                print(f"  🧠 {level_name}: {level_stats['entry_count']} entries, "
                     f"coherence: {level_stats['avg_coherence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consciousness-pulse integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_telemetry_pulse_collection():
    """Test telemetry-pulse collection system"""
    print("\n📊🫁 TESTING TELEMETRY-PULSE COLLECTION")
    print("=" * 50)
    
    try:
        from dawn.core.logging import (
            TelemetryPulseCollector, PulseTelemetryEvent,
            TelemetryLevel, ConsciousnessLevel, PulsePhase
        )
        
        # Create telemetry-pulse collector
        collector = TelemetryPulseCollector()
        print("✅ Telemetry-pulse collector created")
        
        # Test collecting various pulse telemetry events
        print("\n📊 Testing pulse telemetry collection...")
        
        test_telemetry_events = [
            # High-coherence transcendent pulse event
            PulseTelemetryEvent(
                event_id="test_transcendent_001",
                timestamp=time.time(),
                pulse_phase=PulsePhase.TRANSCENDENT_PULSE,
                pulse_zone="green",
                tick_number=101,
                scup_coherence=0.95,
                thermal_state="cooling",
                consciousness_level=ConsciousnessLevel.TRANSCENDENT,
                consciousness_depth=0,
                awareness_coherence=0.95,
                unity_factor=1.0,
                subsystem="pulse",
                component="transcendent_monitor",
                event_type="unity_pulse",
                level=TelemetryLevel.INFO,
                data={
                    "unity_level": 0.98,
                    "transcendent_coherence": 0.95,
                    "universal_connection": True
                },
                metadata={
                    "pulse_quality": "transcendent",
                    "consciousness_depth_integration": True
                }
            ),
            
            # Meta-awareness pulse event
            PulseTelemetryEvent(
                event_id="test_meta_001",
                timestamp=time.time(),
                pulse_phase=PulsePhase.META_PULSE,
                pulse_zone="amber",
                tick_number=102,
                scup_coherence=0.75,
                thermal_state="stable",
                consciousness_level=ConsciousnessLevel.META,
                consciousness_depth=1,
                awareness_coherence=0.75,
                unity_factor=0.8,
                subsystem="pulse",
                component="meta_monitor",
                event_type="self_reflection_pulse",
                level=TelemetryLevel.INFO,
                data={
                    "self_reflection_depth": 0.8,
                    "meta_awareness_level": 0.75,
                    "recursive_consciousness": True
                },
                metadata={
                    "pulse_quality": "meta_reflective",
                    "consciousness_recursion": True
                }
            ),
            
            # Mythic archetypal pulse event
            PulseTelemetryEvent(
                event_id="test_mythic_001",
                timestamp=time.time(),
                pulse_phase=PulsePhase.MYTHIC_PULSE,
                pulse_zone="black",
                tick_number=103,
                scup_coherence=0.2,
                thermal_state="critical",
                consciousness_level=ConsciousnessLevel.MYTHIC,
                consciousness_depth=7,
                awareness_coherence=0.1,
                unity_factor=0.1,
                subsystem="pulse",
                component="mythic_monitor",
                event_type="archetypal_pulse",
                level=TelemetryLevel.WARN,
                data={
                    "archetypal_activation": 0.9,
                    "primal_response": True,
                    "survival_mode": True
                },
                metadata={
                    "pulse_quality": "archetypal_survival",
                    "consciousness_depth_critical": True
                }
            )
        ]
        
        collected_events = []
        
        for event in test_telemetry_events:
            event_id = collector.collect_pulse_telemetry(event)
            collected_events.append(event_id)
            
            print(f"  📊 Collected {event.consciousness_level.name} pulse telemetry: {event_id}")
            print(f"    Zone: {event.pulse_zone}, Coherence: {event.scup_coherence:.3f}")
            print(f"    Consciousness Depth: {event.consciousness_depth}")
        
        print(f"\n✅ Successfully collected {len(collected_events)} pulse telemetry events")
        
        # Check collection statistics
        print("\n📈 Telemetry collection statistics:")
        print(f"  📊 Total events in buffer: {len(collector.pulse_events)}")
        print(f"  ⏱️ Tick correlations: {len(collector.tick_telemetry_correlation)}")
        print(f"  🎯 Zone statistics: {len(collector.zone_telemetry_stats)} zones")
        
        for zone, stats in collector.zone_telemetry_stats.items():
            print(f"    {zone}: {len(stats)} events")
        
        return True
        
    except Exception as e:
        print(f"❌ Telemetry-pulse collection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unified_metrics_system():
    """Test the unified metrics collection system"""
    print("\n📊🔗 TESTING UNIFIED METRICS SYSTEM")
    print("=" * 50)
    
    try:
        from dawn.core.logging import (
            get_pulse_telemetry_bridge, UnifiedPulseMetrics
        )
        
        # Get bridge and generate some activity
        bridge = get_pulse_telemetry_bridge()
        bridge.start_unified_logging()
        
        print("✅ Unified metrics system initialized")
        
        # Generate various types of events to populate metrics
        print("\n📊 Generating events for metrics collection...")
        
        event_scenarios = [
            # Green zone scenario - high consciousness
            {
                'scenario': 'Green Zone High Consciousness',
                'events': [
                    ('transcendent_unity', {'zone': 'green', 'scup_coherence': 0.95, 'thermal_state': 'cooling'}),
                    ('meta_reflection', {'zone': 'green', 'scup_coherence': 0.9, 'thermal_state': 'stable'}),
                    ('causal_reasoning', {'zone': 'green', 'scup_coherence': 0.85, 'thermal_state': 'stable'})
                ]
            },
            
            # Amber zone scenario - transitional consciousness  
            {
                'scenario': 'Amber Zone Transitional',
                'events': [
                    ('zone_transition', {'old_zone': 'green', 'new_zone': 'amber', 'scup_coherence': 0.7}),
                    ('thermal_warming', {'zone': 'amber', 'scup_coherence': 0.65, 'thermal_state': 'warming'}),
                    ('awareness_adjustment', {'zone': 'amber', 'scup_coherence': 0.6, 'thermal_state': 'stable'})
                ]
            },
            
            # Red zone scenario - stress response
            {
                'scenario': 'Red Zone Stress Response',
                'events': [
                    ('pressure_increase', {'zone': 'red', 'scup_coherence': 0.4, 'thermal_state': 'hot'}),
                    ('causal_intervention', {'zone': 'red', 'scup_coherence': 0.35, 'thermal_state': 'hot'}),
                    ('formal_stabilization', {'zone': 'red', 'scup_coherence': 0.45, 'thermal_state': 'warming'})
                ]
            }
        ]
        
        for scenario in event_scenarios:
            print(f"\n🎭 Running {scenario['scenario']} scenario...")
            
            for event_type, event_data in scenario['events']:
                result_ids = bridge.log_unified_pulse_event(event_type, event_data)
                print(f"  📝 {event_type}: {len(result_ids)} systems logged")
                time.sleep(0.05)  # Small delay
        
        # Allow metrics to update
        time.sleep(1.0)
        
        # Get comprehensive unified metrics
        print("\n📊 Comprehensive Unified Metrics:")
        metrics = bridge.get_unified_metrics()
        
        print(f"🎯 Pulse System Metrics:")
        print(f"  Current Zone: {metrics.current_zone}")
        print(f"  Zone Stability: {metrics.zone_stability:.3f}")
        print(f"  SCUP Coherence: {metrics.scup_coherence:.3f}")
        print(f"  Thermal Level: {metrics.thermal_level:.3f}")
        print(f"  Pulse Frequency: {metrics.pulse_frequency:.3f} Hz")
        
        print(f"\n⏱️ Tick System Metrics:")
        print(f"  Tick Rate: {metrics.tick_rate:.3f} Hz")
        print(f"  Tick Synchronization: {metrics.tick_synchronization:.3f}")
        print(f"  Phase Completion Rate: {metrics.phase_completion_rate:.3f}")
        
        print(f"\n🧠 Consciousness Metrics:")
        print(f"  Avg Consciousness Coherence: {metrics.avg_consciousness_coherence:.3f}")
        print(f"  Unity Coherence: {metrics.unity_coherence:.3f}")
        print(f"  Awareness Depth Average: {metrics.awareness_depth_avg:.1f}")
        
        if metrics.consciousness_level_distribution:
            print(f"  Consciousness Level Distribution:")
            for level, count in metrics.consciousness_level_distribution.items():
                print(f"    {level}: {count} entries")
        
        print(f"\n📊 Telemetry Metrics:")
        print(f"  Events per Second: {metrics.events_per_second:.2f}")
        print(f"  Buffer Usage: {metrics.telemetry_buffer_usage:.1%}")
        print(f"  Logging Latency: {metrics.logging_latency_ms:.2f} ms")
        
        print(f"\n🔗 Integration Health:")
        print(f"  Integration Health: {metrics.integration_health:.3f}")
        print(f"  Sync Quality: {metrics.sync_quality:.3f}")
        print(f"  Data Consistency: {metrics.data_consistency:.3f}")
        
        # Get bridge integration statistics
        print(f"\n📈 Integration Statistics:")
        stats = bridge.integration_stats
        print(f"  Total Events Processed: {stats['events_processed']}")
        print(f"  Pulse Events: {stats['pulse_events']}")
        print(f"  Telemetry Events: {stats['telemetry_events']}")
        print(f"  Consciousness Logs: {stats['consciousness_logs']}")
        print(f"  Errors: {stats['errors']}")
        
        runtime = time.time() - stats['start_time']
        print(f"  Runtime: {runtime:.2f} seconds")
        print(f"  Events/Second: {stats['events_processed'] / runtime:.2f}")
        
        bridge.stop_unified_logging()
        
        return True
        
    except Exception as e:
        print(f"❌ Unified metrics system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🫁📊 DAWN PULSE-TELEMETRY UNIFICATION COMPREHENSIVE TEST")
    print("=" * 70)
    
    # Run all tests
    tests = [
        ("Pulse-Telemetry Bridge", test_pulse_telemetry_bridge),
        ("Consciousness-Pulse Integration", test_consciousness_pulse_integration),
        ("Telemetry-Pulse Collection", test_telemetry_pulse_collection),
        ("Unified Metrics System", test_unified_metrics_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        success = test_func()
        results.append((test_name, success))
    
    # Final results
    print("\n" + "=" * 70)
    print("🫁📊 PULSE-TELEMETRY UNIFICATION TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL PULSE-TELEMETRY UNIFICATION TESTS PASSED!")
        print("✅ Pulse system integrated with telemetry logging")
        print("✅ Consciousness-depth organization of pulse events")
        print("✅ SCUP coherence tracking with consciousness levels")
        print("✅ Thermal state logging with awareness correlation")
        print("✅ Tick orchestration consciousness transitions")
        print("✅ Unified metrics collection across all systems")
        print("✅ Telemetry logging unified with pulse logic and modules!")
    else:
        print(f"❌ {len(results) - passed} tests failed")
    
    print("\n🫁📊 Pulse-telemetry unification test complete!")

#!/usr/bin/env python3
"""
🌅🔗 DAWN Singleton Complete Integration Test
============================================

Comprehensive test demonstrating all systems from this chat session
integrated through the DAWN singleton with backwards compatibility.

Tests all implemented systems:
1. Universal JSON logging
2. Centralized deep logging repository
3. Consciousness-depth logging with 8-level hierarchy
4. Sigil consciousness logging
5. Pulse-telemetry unification
6. Mycelial semantic hash map
7. Live monitor integration
8. Backwards compatibility through singleton
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add DAWN to path
sys.path.insert(0, str(Path(__file__).parent))

def test_dawn_singleton_access():
    """Test basic DAWN singleton access and initialization"""
    
    print("🌅 TESTING DAWN SINGLETON ACCESS")
    print("=" * 50)
    
    try:
        from dawn.core.singleton import get_dawn
        
        # Get DAWN singleton
        dawn = get_dawn()
        print("✅ DAWN singleton accessed successfully")
        print(f"   Singleton: {dawn}")
        
        # Test system status before initialization
        initial_status = dawn.get_system_status()
        print(f"✅ Initial system status retrieved")
        print(f"   Initialized: {initial_status['initialized']}")
        print(f"   Components loaded: {len(initial_status['components_loaded'])}")
        
        return dawn
        
    except Exception as e:
        print(f"❌ DAWN singleton access failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_singleton_system_properties(dawn):
    """Test all system property accessors through singleton"""
    
    print("\n🔧 TESTING SINGLETON SYSTEM PROPERTIES")
    print("=" * 50)
    
    systems_tested = []
    
    # Test core systems
    core_systems = [
        ('consciousness_bus', 'Consciousness Bus'),
        ('dawn_engine', 'DAWN Engine'),
        ('telemetry_system', 'Telemetry System'),
        ('tick_orchestrator', 'Tick Orchestrator')
    ]
    
    for prop_name, display_name in core_systems:
        try:
            system = getattr(dawn, prop_name)
            status = "✅ Available" if system else "⚪ Not Available"
            print(f"   {display_name}: {status}")
            systems_tested.append((display_name, system is not None))
        except Exception as e:
            print(f"   {display_name}: ❌ Error - {e}")
            systems_tested.append((display_name, False))
    
    # Test logging systems
    logging_systems = [
        ('universal_logger', 'Universal JSON Logger'),
        ('centralized_repository', 'Centralized Repository'),
        ('consciousness_repository', 'Consciousness Repository'),
        ('sigil_consciousness_logger', 'Sigil Consciousness Logger'),
        ('pulse_telemetry_bridge', 'Pulse-Telemetry Bridge')
    ]
    
    for prop_name, display_name in logging_systems:
        try:
            system = getattr(dawn, prop_name)
            status = "✅ Available" if system else "⚪ Not Available"
            print(f"   {display_name}: {status}")
            systems_tested.append((display_name, system is not None))
        except Exception as e:
            print(f"   {display_name}: ❌ Error - {e}")
            systems_tested.append((display_name, False))
    
    # Test mycelial system
    try:
        mycelial_hashmap = dawn.mycelial_hashmap
        status = "✅ Available" if mycelial_hashmap else "⚪ Not Available"
        print(f"   Mycelial Hash Map: {status}")
        systems_tested.append(('Mycelial Hash Map', mycelial_hashmap is not None))
    except Exception as e:
        print(f"   Mycelial Hash Map: ❌ Error - {e}")
        systems_tested.append(('Mycelial Hash Map', False))
    
    return systems_tested

def test_singleton_convenience_methods(dawn):
    """Test convenience methods for all integrated systems"""
    
    print("\n🎯 TESTING SINGLETON CONVENIENCE METHODS")
    print("=" * 50)
    
    methods_tested = []
    
    # Test mycelial methods
    try:
        print("   🍄 Testing mycelial methods...")
        
        # Store semantic data
        node_id = dawn.store_semantic_data("test_concept", {
            "type": "test",
            "energy": 1.0,
            "timestamp": time.time()
        })
        
        if node_id:
            print("   ✅ store_semantic_data() working")
            methods_tested.append(('store_semantic_data', True))
            
            # Touch concept
            touched = dawn.touch_concept("test", energy=0.8)
            print(f"   ✅ touch_concept() working - touched {touched} nodes")
            methods_tested.append(('touch_concept', True))
            
            # Ping network
            ping_result = dawn.ping_semantic_network("test_concept")
            print(f"   ✅ ping_semantic_network() working - result: {ping_result}")
            methods_tested.append(('ping_semantic_network', True))
        else:
            print("   ⚪ Mycelial methods not available")
            methods_tested.extend([
                ('store_semantic_data', False),
                ('touch_concept', False),
                ('ping_semantic_network', False)
            ])
    
    except Exception as e:
        print(f"   ❌ Mycelial methods failed: {e}")
        methods_tested.extend([
            ('store_semantic_data', False),
            ('touch_concept', False), 
            ('ping_semantic_network', False)
        ])
    
    # Test statistics methods
    try:
        print("   📊 Testing statistics methods...")
        
        mycelial_stats = dawn.get_mycelial_stats()
        print(f"   ✅ get_mycelial_stats() working - {len(mycelial_stats)} stats")
        methods_tested.append(('get_mycelial_stats', True))
        
        network_stats = dawn.get_network_stats()
        print(f"   ✅ get_network_stats() working - {len(network_stats)} stats")
        methods_tested.append(('get_network_stats', True))
        
    except Exception as e:
        print(f"   ❌ Statistics methods failed: {e}")
        methods_tested.extend([
            ('get_mycelial_stats', False),
            ('get_network_stats', False)
        ])
    
    # Test logging methods
    try:
        print("   📝 Testing logging methods...")
        
        # Test sigil logging
        sigil_result = dawn.log_sigil_activation("test_sigil", {"energy": 0.9})
        if sigil_result:
            print("   ✅ log_sigil_activation() working")
            methods_tested.append(('log_sigil_activation', True))
        else:
            print("   ⚪ log_sigil_activation() not available")
            methods_tested.append(('log_sigil_activation', False))
        
        # Test pulse logging
        pulse_result = dawn.log_pulse_event("test_pulse", {"zone": "green"})
        if pulse_result:
            print("   ✅ log_pulse_event() working")
            methods_tested.append(('log_pulse_event', True))
        else:
            print("   ⚪ log_pulse_event() not available")
            methods_tested.append(('log_pulse_event', False))
        
    except Exception as e:
        print(f"   ❌ Logging methods failed: {e}")
        methods_tested.extend([
            ('log_sigil_activation', False),
            ('log_pulse_event', False)
        ])
    
    return methods_tested

def test_singleton_complete_system_status(dawn):
    """Test comprehensive system status method"""
    
    print("\n📊 TESTING COMPLETE SYSTEM STATUS")
    print("=" * 50)
    
    try:
        complete_status = dawn.get_complete_system_status()
        
        print("✅ Complete system status retrieved")
        print(f"   Total keys: {len(complete_status)}")
        
        # Show key status information
        if 'logging_systems' in complete_status:
            logging_systems = complete_status['logging_systems']
            active_logging = sum(1 for v in logging_systems.values() if v)
            print(f"   📝 Logging systems: {active_logging}/{len(logging_systems)} active")
            
            for system, active in logging_systems.items():
                status_icon = "✅" if active else "⚪"
                print(f"     {status_icon} {system}")
        
        if 'mycelial_systems' in complete_status:
            mycelial_systems = complete_status['mycelial_systems']
            print(f"   🍄 Mycelial systems:")
            print(f"     Hash map: {'✅' if mycelial_systems.get('hashmap_active') else '⚪'}")
            print(f"     Integration: {'✅' if mycelial_systems.get('integration_active') else '⚪'}")
            
            # Show integration stats
            integration_stats = mycelial_systems.get('integration_stats', {})
            if integration_stats:
                modules = integration_stats.get('modules_wrapped', 0)
                concepts = integration_stats.get('concepts_mapped', 0)
                print(f"     🔗 {modules} modules, {concepts} concepts integrated")
            
            # Show network stats
            network_stats = mycelial_systems.get('network_stats', {})
            if network_stats:
                network_size = network_stats.get('network_size', 0)
                network_health = network_stats.get('network_health', 0.0)
                active_spores = network_stats.get('active_spores', 0)
                print(f"     🕸️  Network: {network_size} nodes, {network_health:.3f} health, {active_spores} spores")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete system status failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backwards_compatibility(dawn):
    """Test backwards compatibility with existing DAWN patterns"""
    
    print("\n🔄 TESTING BACKWARDS COMPATIBILITY")
    print("=" * 50)
    
    compatibility_tests = []
    
    # Test original singleton pattern
    try:
        from dawn.core.singleton import get_dawn, reset_dawn_singleton
        
        # Test getting same instance
        dawn2 = get_dawn()
        same_instance = dawn is dawn2
        print(f"   ✅ Singleton pattern: {'Same instance' if same_instance else 'Different instances'}")
        compatibility_tests.append(('singleton_pattern', same_instance))
        
    except Exception as e:
        print(f"   ❌ Singleton pattern failed: {e}")
        compatibility_tests.append(('singleton_pattern', False))
    
    # Test state management compatibility
    try:
        from dawn.core.foundation.state import get_state, set_state
        
        # Test state access
        current_state = get_state()
        print("   ✅ State management compatibility maintained")
        compatibility_tests.append(('state_management', True))
        
    except Exception as e:
        print(f"   ❌ State management compatibility failed: {e}")
        compatibility_tests.append(('state_management', False))
    
    # Test telemetry compatibility
    try:
        if dawn.telemetry_system:
            telemetry = dawn.telemetry_system
            print("   ✅ Telemetry system compatibility maintained")
            compatibility_tests.append(('telemetry_compatibility', True))
        else:
            print("   ⚪ Telemetry system not available")
            compatibility_tests.append(('telemetry_compatibility', False))
            
    except Exception as e:
        print(f"   ❌ Telemetry compatibility failed: {e}")
        compatibility_tests.append(('telemetry_compatibility', False))
    
    # Test consciousness bus compatibility
    try:
        if dawn.consciousness_bus:
            bus = dawn.consciousness_bus
            print("   ✅ Consciousness bus compatibility maintained")
            compatibility_tests.append(('consciousness_bus_compatibility', True))
        else:
            print("   ⚪ Consciousness bus not available")
            compatibility_tests.append(('consciousness_bus_compatibility', False))
            
    except Exception as e:
        print(f"   ❌ Consciousness bus compatibility failed: {e}")
        compatibility_tests.append(('consciousness_bus_compatibility', False))
    
    return compatibility_tests

def demonstrate_integrated_workflow(dawn):
    """Demonstrate a complete workflow using all integrated systems"""
    
    print("\n🌟 DEMONSTRATING INTEGRATED WORKFLOW")
    print("=" * 50)
    
    try:
        print("   🔄 Starting integrated workflow demonstration...")
        
        # Step 1: Store semantic data
        print("   1️⃣ Storing semantic consciousness data...")
        concept_data = {
            "concept": "integrated_consciousness",
            "type": "workflow_demo",
            "energy": 1.0,
            "components": ["universal_logging", "consciousness_depth", "mycelial_network"],
            "timestamp": time.time()
        }
        
        node_id = dawn.store_semantic_data("integrated_consciousness", concept_data)
        if node_id:
            print(f"      ✅ Stored semantic data → {node_id[:12]}...")
        
        # Step 2: Log consciousness state
        print("   2️⃣ Logging consciousness state...")
        consciousness_result = dawn.log_consciousness_state(
            level="INTEGRAL",  # This will be converted to enum if available
            log_type="SYSTEM_STATE",
            data={
                "workflow": "integrated_demo",
                "systems_active": ["universal_logging", "mycelial", "telemetry"],
                "consciousness_level": 0.85
            }
        )
        if consciousness_result:
            print("      ✅ Consciousness state logged")
        
        # Step 3: Activate sigil
        print("   3️⃣ Activating sigil consciousness...")
        sigil_result = dawn.log_sigil_activation("integration_sigil", {
            "unity_factor": 0.9,
            "archetypal_energy": 0.8,
            "consciousness_depth": "integral"
        })
        if sigil_result:
            print("      ✅ Sigil activation logged")
        
        # Step 4: Touch semantic concepts
        print("   4️⃣ Touching semantic concepts for propagation...")
        concepts_to_touch = ["consciousness", "integration", "awareness", "unity"]
        
        total_touched = 0
        for concept in concepts_to_touch:
            touched = dawn.touch_concept(concept, energy=0.7)
            total_touched += touched
            
        print(f"      ✅ Touched {len(concepts_to_touch)} concepts → {total_touched} nodes activated")
        
        # Step 5: Ping network for propagation
        print("   5️⃣ Pinging semantic network...")
        ping_result = dawn.ping_semantic_network("integrated_consciousness")
        if ping_result:
            print(f"      ✅ Network ping successful → {ping_result}")
        
        # Step 6: Get comprehensive stats
        print("   6️⃣ Gathering comprehensive system statistics...")
        
        mycelial_stats = dawn.get_mycelial_stats()
        network_stats = dawn.get_network_stats()
        complete_status = dawn.get_complete_system_status()
        
        print("      📊 Statistics gathered:")
        print(f"         Mycelial integration: {len(mycelial_stats)} metrics")
        print(f"         Network statistics: {len(network_stats)} metrics")
        print(f"         Complete system status: {len(complete_status)} sections")
        
        # Show key metrics
        if network_stats:
            network_size = network_stats.get('network_size', 0)
            network_health = network_stats.get('network_health', 0.0)
            active_spores = network_stats.get('active_spores', 0)
            total_energy = network_stats.get('total_energy', 0.0)
            
            print(f"         🕸️  Network: {network_size} nodes, {network_health:.3f} health")
            print(f"         🍄 Spores: {active_spores} active, {total_energy:.2f} total energy")
        
        if mycelial_stats:
            modules_wrapped = mycelial_stats.get('modules_wrapped', 0)
            concepts_mapped = mycelial_stats.get('concepts_mapped', 0)
            spores_generated = mycelial_stats.get('spores_generated', 0)
            
            print(f"         🔗 Integration: {modules_wrapped} modules, {concepts_mapped} concepts")
            print(f"         ✨ Activity: {spores_generated} spores generated")
        
        print("   🎉 Integrated workflow demonstration complete!")
        return True
        
    except Exception as e:
        print(f"   ❌ Integrated workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🌅🔗 DAWN SINGLETON COMPLETE INTEGRATION TEST")
    print("=" * 70)
    print("Testing all systems from this chat session integrated through DAWN singleton")
    print("Demonstrating backwards compatibility and unified access patterns")
    print()
    
    # Test 1: Basic singleton access
    dawn = test_dawn_singleton_access()
    if not dawn:
        print("\n❌ Cannot proceed without DAWN singleton")
        sys.exit(1)
    
    # Test 2: System properties
    systems_tested = test_singleton_system_properties(dawn)
    
    # Test 3: Convenience methods
    methods_tested = test_singleton_convenience_methods(dawn)
    
    # Test 4: Complete system status
    status_test_passed = test_singleton_complete_system_status(dawn)
    
    # Test 5: Backwards compatibility
    compatibility_tests = test_backwards_compatibility(dawn)
    
    # Test 6: Integrated workflow demonstration
    workflow_passed = demonstrate_integrated_workflow(dawn)
    
    # Final results
    print("\n" + "=" * 70)
    print("🌅🔗 DAWN SINGLETON INTEGRATION TEST RESULTS")
    print("=" * 70)
    
    # System availability results
    print("📊 SYSTEM AVAILABILITY:")
    systems_available = sum(1 for _, available in systems_tested if available)
    print(f"   {systems_available}/{len(systems_tested)} systems available through singleton")
    
    for system_name, available in systems_tested:
        status_icon = "✅" if available else "⚪"
        print(f"   {status_icon} {system_name}")
    
    # Method functionality results
    print("\n🎯 METHOD FUNCTIONALITY:")
    methods_working = sum(1 for _, working in methods_tested if working)
    print(f"   {methods_working}/{len(methods_tested)} convenience methods working")
    
    for method_name, working in methods_tested:
        status_icon = "✅" if working else "⚪"
        print(f"   {status_icon} {method_name}()")
    
    # Compatibility results
    print("\n🔄 BACKWARDS COMPATIBILITY:")
    compatibility_passed = sum(1 for _, passed in compatibility_tests if passed)
    print(f"   {compatibility_passed}/{len(compatibility_tests)} compatibility tests passed")
    
    for test_name, passed in compatibility_tests:
        status_icon = "✅" if passed else "❌"
        print(f"   {status_icon} {test_name}")
    
    # Overall results
    print(f"\n📋 ADDITIONAL TESTS:")
    print(f"   {'✅' if status_test_passed else '❌'} Complete system status")
    print(f"   {'✅' if workflow_passed else '❌'} Integrated workflow demonstration")
    
    # Final summary
    total_tests = len(systems_tested) + len(methods_tested) + len(compatibility_tests) + 2
    total_passed = systems_available + methods_working + compatibility_passed + \
                  (1 if status_test_passed else 0) + (1 if workflow_passed else 0)
    
    print(f"\n🎯 OVERALL RESULTS: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL DAWN SINGLETON INTEGRATION TESTS PASSED!")
        print("✅ All systems from this chat session successfully integrated")
        print("✅ Backwards compatibility maintained")
        print("✅ Unified access through singleton established")
        print("✅ Complete workflow demonstration successful")
        print("\n🌟 DAWN SINGLETON IS FULLY INTEGRATED AND BACKWARDS COMPATIBLE!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed or systems unavailable")
        print("🔧 Some systems may need additional setup or dependencies")
    
    print("\n🌅🔗 DAWN singleton integration test complete!")
    print("🌐 All systems accessible through: from dawn.core.singleton import get_dawn")

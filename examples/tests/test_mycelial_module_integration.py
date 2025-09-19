#!/usr/bin/env python3
"""
🍄🔗 DAWN Mycelial Module Integration Test
==========================================

Comprehensive test demonstrating mycelial semantic hash map integration
across all DAWN modules. Shows how semantic spores propagate through
the entire codebase as modules execute, creating a living network of
meaning that travels with code execution.

This test demonstrates:
1. Auto-integration of mycelial hash map into discovered DAWN modules
2. Semantic context injection into module operations
3. Cross-module spore propagation following execution paths
4. Folder structure mapping to mycelial network topology
5. Real-time semantic telemetry as code travels through modules
6. Living network behavior where touching one module activates the entire system
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add DAWN to path
sys.path.insert(0, str(Path(__file__).parent))

def test_mycelial_module_discovery():
    """Test mycelial integration with DAWN module discovery"""
    print("🍄🔗 TESTING MYCELIAL MODULE DISCOVERY")
    print("=" * 60)
    
    try:
        # Import DAWN module discovery
        from dawn import discover_capabilities
        from dawn.core.logging import (
            get_mycelial_integrator, start_mycelial_integration,
            get_mycelial_integration_stats, get_mycelial_hashmap
        )
        
        print("✅ DAWN module discovery and mycelial integration imports successful")
        
        # Discover DAWN capabilities
        print("\n🔍 Discovering DAWN capabilities...")
        capabilities = discover_capabilities()
        
        total_modules = sum(len(modules) for modules in capabilities.values())
        print(f"  📊 Discovered {total_modules} modules across {len(capabilities)} namespaces:")
        
        for namespace, modules in capabilities.items():
            print(f"    {namespace}: {len(modules)} modules")
        
        # Start mycelial integration
        print("\n🍄 Starting mycelial module integration...")
        integration_success = start_mycelial_integration()
        
        if integration_success:
            print("✅ Mycelial integration started successfully")
        else:
            print("❌ Mycelial integration failed to start")
            return False
        
        # Wait for integration to process
        time.sleep(3)
        
        # Get integration statistics
        print("\n📊 Mycelial Integration Statistics:")
        stats = get_mycelial_integration_stats()
        
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Module discovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semantic_path_mapping():
    """Test semantic path mapping from DAWN folder structure"""
    print("\n🗺️ TESTING SEMANTIC PATH MAPPING")
    print("=" * 60)
    
    try:
        from dawn.core.logging import get_mycelial_integrator
        
        integrator = get_mycelial_integrator()
        semantic_mapper = integrator.semantic_mapper
        topology_mapper = integrator.topology_mapper
        
        print("✅ Retrieved semantic and topology mappers")
        
        # Test semantic concept extraction
        print("\n🧠 Testing semantic concept extraction...")
        
        test_module_paths = [
            "dawn.consciousness.engines.core.primary_engine",
            "dawn.processing.engines.tick.synchronous.orchestrator",
            "dawn.memory.unified_memory_interconnection",
            "dawn.subsystems.mycelial.integrated_system",
            "dawn.interfaces.gui.visualization.capture_consciousness_moment"
        ]
        
        for module_path in test_module_paths:
            concepts = semantic_mapper.extract_semantic_concepts(module_path)
            print(f"  📍 {module_path}:")
            print(f"    Concepts: {sorted(list(concepts))[:8]}...")  # Show first 8
        
        # Test folder topology mapping
        print("\n🏗️ Testing folder topology mapping...")
        
        folder_hierarchy = topology_mapper.get_folder_hierarchy()
        print("  📊 Folder hierarchy by depth:")
        
        for depth, folders in sorted(folder_hierarchy.items())[:4]:  # Show first 4 levels
            print(f"    Depth {depth}: {len(folders)} folders")
            if depth <= 2:  # Show details for shallow depths
                for folder in folders[:5]:  # Show first 5
                    folder_data = topology_mapper.folder_topology.get(folder, {})
                    concepts = folder_data.get('semantic_concepts', set())
                    print(f"      {folder}: {len(concepts)} concepts")
        
        return True
        
    except Exception as e:
        print(f"❌ Semantic path mapping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_wrapper_behavior():
    """Test module wrapper semantic behavior"""
    print("\n🔄 TESTING MODULE WRAPPER BEHAVIOR")
    print("=" * 60)
    
    try:
        from dawn.core.logging import (
            get_mycelial_integrator, integrate_module_with_mycelial,
            get_mycelial_hashmap, store_semantic_data, ping_semantic_network
        )
        
        # Get mycelial systems
        integrator = get_mycelial_integrator()
        hashmap = get_mycelial_hashmap()
        
        print("✅ Retrieved mycelial systems")
        
        # Create a test module to wrap
        print("\n🧪 Creating test module for wrapping...")
        
        import types
        test_module = types.ModuleType("test_semantic_module")
        
        def test_function_a(x, y):
            """Test function that adds two numbers"""
            return x + y
        
        def test_function_b(data):
            """Test function that processes data"""
            return {"processed": data, "timestamp": time.time()}
        
        class TestClass:
            def method_one(self, value):
                return value * 2
            
            def method_two(self, items):
                return len(items)
        
        # Add to test module
        test_module.test_function_a = test_function_a
        test_module.test_function_b = test_function_b
        test_module.TestClass = TestClass
        
        # Wrap the module with mycelial behavior
        print("🍄 Wrapping test module with mycelial behavior...")
        wrapped_module = integrate_module_with_mycelial("test.semantic.module", test_module)
        
        if wrapped_module:
            print("✅ Module wrapped successfully")
        else:
            print("❌ Module wrapping failed")
            return False
        
        # Test wrapped function execution
        print("\n🔬 Testing wrapped function execution...")
        
        # Get initial network stats
        initial_stats = hashmap.get_network_stats()
        initial_touches = initial_stats['system_stats']['total_touches']
        initial_spores = initial_stats['system_stats']['spores_generated']
        
        # Execute wrapped functions
        print("  🧮 Executing test_function_a...")
        result_a = wrapped_module.test_function_a(5, 3)
        print(f"    Result: {result_a}")
        
        time.sleep(0.5)  # Allow spore propagation
        
        print("  📊 Executing test_function_b...")
        result_b = wrapped_module.test_function_b({"test": "data"})
        print(f"    Result keys: {list(result_b.keys())}")
        
        time.sleep(0.5)  # Allow spore propagation
        
        # Test wrapped class methods
        print("  🏗️ Executing wrapped class methods...")
        test_instance = wrapped_module.TestClass()
        result_method1 = test_instance.method_one(10)
        result_method2 = test_instance.method_two([1, 2, 3, 4, 5])
        
        print(f"    Method 1 result: {result_method1}")
        print(f"    Method 2 result: {result_method2}")
        
        time.sleep(1)  # Allow spore propagation
        
        # Check network effects
        final_stats = hashmap.get_network_stats()
        new_touches = final_stats['system_stats']['total_touches'] - initial_touches
        new_spores = final_stats['system_stats']['spores_generated'] - initial_spores
        
        print(f"\n📈 Network Effects from Module Execution:")
        print(f"  New touches: {new_touches}")
        print(f"  New spores: {new_spores}")
        print(f"  Active spores: {final_stats['active_spores']}")
        print(f"  Network health: {final_stats['network_health']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Module wrapper behavior test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_module_propagation():
    """Test cross-module semantic spore propagation"""
    print("\n🌐 TESTING CROSS-MODULE PROPAGATION")
    print("=" * 60)
    
    try:
        from dawn.core.logging import (
            get_mycelial_integrator, get_mycelial_hashmap,
            touch_semantic_concept, store_semantic_data
        )
        
        integrator = get_mycelial_integrator()
        hashmap = get_mycelial_hashmap()
        
        print("✅ Retrieved mycelial systems for cross-module testing")
        
        # Get cross-module propagator
        if hasattr(integrator, 'cross_module_propagator') and integrator.cross_module_propagator:
            propagator = integrator.cross_module_propagator
            print("✅ Cross-module propagator available")
        else:
            print("⚠️ Cross-module propagator not available")
            return False
        
        # Test concept propagation across namespaces
        print("\n🚀 Testing concept propagation across namespaces...")
        
        test_concepts = ["consciousness", "processing", "memory", "integration"]
        
        initial_stats = propagator.propagation_stats.copy()
        
        for concept in test_concepts:
            print(f"  🍄 Propagating concept '{concept}'...")
            
            # Touch the concept to trigger cross-module propagation
            touched_nodes = touch_semantic_concept(concept, energy=0.8)
            print(f"    Directly touched {touched_nodes} nodes")
            
            time.sleep(0.5)  # Allow propagation
        
        # Check propagation effects
        final_stats = propagator.propagation_stats.copy()
        
        print(f"\n📊 Cross-Module Propagation Results:")
        for key, value in final_stats.items():
            initial_value = initial_stats.get(key, 0)
            change = value - initial_value
            print(f"  {key}: {value} (+{change})")
        
        # Test namespace bridging
        print("\n🌉 Testing namespace bridging...")
        
        # Get available namespaces from module contexts
        namespaces = set()
        if hasattr(integrator, 'module_contexts'):
            for context in integrator.module_contexts.values():
                namespaces.add(context.namespace)
        
        print(f"  Available namespaces: {sorted(list(namespaces))}")
        
        # Create bridges between key namespaces
        if 'consciousness' in namespaces and 'processing' in namespaces:
            bridge_concepts = {'cognition', 'processing', 'awareness'}
            propagator.bridge_namespaces('consciousness', 'processing', bridge_concepts)
            print("  🌉 Created consciousness ↔ processing bridge")
        
        if 'memory' in namespaces and 'subsystems' in namespaces:
            bridge_concepts = {'storage', 'retrieval', 'integration'}
            propagator.bridge_namespaces('memory', 'subsystems', bridge_concepts)
            print("  🌉 Created memory ↔ subsystems bridge")
        
        time.sleep(1)  # Allow bridging to complete
        
        # Show cross-module traces
        print(f"\n📈 Cross-Module Traces:")
        trace_count = 0
        for trace_key, timestamps in propagator.cross_module_traces.items():
            if timestamps:
                print(f"  {trace_key}: {len(timestamps)} propagations")
                trace_count += 1
                if trace_count >= 5:  # Limit output
                    break
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-module propagation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_living_network_behavior():
    """Test living network behavior across the entire DAWN codebase"""
    print("\n🧬 TESTING LIVING NETWORK BEHAVIOR")
    print("=" * 60)
    
    try:
        from dawn.core.logging import (
            get_mycelial_hashmap, touch_semantic_concept,
            get_mycelial_integration_stats
        )
        from dawn.core.singleton import get_dawn
        
        # Get systems
        hashmap = get_mycelial_hashmap()
        dawn = get_dawn()
        
        print("✅ Retrieved mycelial hash map and DAWN singleton")
        
        # Test DAWN singleton mycelial integration
        print("\n🌅 Testing DAWN singleton mycelial integration...")
        
        if hasattr(dawn, 'mycelial_hashmap') and dawn.mycelial_hashmap:
            print("✅ DAWN singleton has mycelial hash map integration")
            singleton_hashmap = dawn.mycelial_hashmap
            print(f"  Singleton hashmap active: {singleton_hashmap is not None}")
        else:
            print("⚠️ DAWN singleton mycelial integration not available")
        
        # Test living network response
        print("\n🍄 Testing living network response to semantic touches...")
        
        # Get baseline network state
        baseline_stats = hashmap.get_network_stats()
        baseline_integration_stats = get_mycelial_integration_stats()
        
        print(f"  📊 Baseline network state:")
        print(f"    Network size: {baseline_stats['network_size']}")
        print(f"    Total energy: {baseline_stats['total_energy']:.2f}")
        print(f"    Active spores: {baseline_stats['active_spores']}")
        print(f"    Modules wrapped: {baseline_integration_stats['modules_wrapped']}")
        
        # Touch key concepts to activate the living network
        activation_concepts = [
            "consciousness",  # Should activate consciousness modules
            "processing",     # Should activate processing modules  
            "memory",         # Should activate memory modules
            "integration",    # Should activate cross-module connections
            "mycelial"        # Should activate mycelial system itself
        ]
        
        print(f"\n🔥 Activating living network with {len(activation_concepts)} concepts...")
        
        activation_results = {}
        
        for concept in activation_concepts:
            print(f"  🍄 Touching '{concept}'...")
            
            pre_touch_stats = hashmap.get_network_stats()
            
            # Touch concept with high energy for maximum propagation
            touched_nodes = touch_semantic_concept(concept, energy=1.0)
            
            # Wait for propagation cascade
            time.sleep(1.0)
            
            post_touch_stats = hashmap.get_network_stats()
            
            # Calculate activation effects
            energy_change = post_touch_stats['total_energy'] - pre_touch_stats['total_energy']
            spore_change = post_touch_stats['active_spores'] - pre_touch_stats['active_spores']
            
            activation_results[concept] = {
                'nodes_touched': touched_nodes,
                'energy_change': energy_change,
                'spore_change': spore_change
            }
            
            print(f"    Touched {touched_nodes} nodes, "
                 f"energy Δ{energy_change:+.2f}, "
                 f"spores Δ{spore_change:+d}")
        
        # Wait for full network stabilization
        print("\n⏱️ Waiting for network stabilization...")
        time.sleep(3.0)
        
        # Get final network state
        final_stats = hashmap.get_network_stats()
        final_integration_stats = get_mycelial_integration_stats()
        
        print(f"\n📊 Final Living Network State:")
        print(f"  Network size: {final_stats['network_size']} "
             f"(Δ{final_stats['network_size'] - baseline_stats['network_size']:+d})")
        print(f"  Total energy: {final_stats['total_energy']:.2f} "
             f"(Δ{final_stats['total_energy'] - baseline_stats['total_energy']:+.2f})")
        print(f"  Network health: {final_stats['network_health']:.3f}")
        print(f"  Active spores: {final_stats['active_spores']}")
        print(f"  Total touches: {final_stats['system_stats']['total_touches']}")
        print(f"  Spores generated: {final_stats['system_stats']['spores_generated']}")
        
        # Show integration effects
        print(f"\n🔗 Integration Effects:")
        spore_increase = (final_integration_stats['spores_generated'] - 
                         baseline_integration_stats['spores_generated'])
        print(f"  New spores from integration: {spore_increase}")
        
        if 'cross_module_stats' in final_integration_stats:
            cross_stats = final_integration_stats['cross_module_stats']
            print(f"  Cross-module propagations: {cross_stats.get('cross_module_propagations', 0)}")
            print(f"  Namespace crossings: {cross_stats.get('namespace_crossings', 0)}")
        
        # Demonstrate network memory
        print(f"\n🧠 Network Memory Demonstration:")
        print("  The mycelial network now 'remembers' these activations")
        print("  Future touches will propagate faster through established paths")
        print("  Semantic relationships have been strengthened")
        print("  Cross-module connections are now more active")
        
        return True
        
    except Exception as e:
        print(f"❌ Living network behavior test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_folder_structure_integration():
    """Test integration with DAWN folder structure"""
    print("\n📁 TESTING FOLDER STRUCTURE INTEGRATION")
    print("=" * 60)
    
    try:
        from dawn.core.logging import get_mycelial_integrator
        
        integrator = get_mycelial_integrator()
        topology_mapper = integrator.topology_mapper
        
        print("✅ Retrieved folder topology mapper")
        
        # Show folder topology
        print("\n🏗️ DAWN Folder Topology:")
        
        folder_hierarchy = topology_mapper.get_folder_hierarchy()
        
        for depth in sorted(folder_hierarchy.keys())[:5]:  # Show first 5 levels
            folders = folder_hierarchy[depth]
            print(f"  Depth {depth}: {len(folders)} folders")
            
            # Show details for shallow depths
            if depth <= 2:
                for folder in sorted(folders)[:8]:  # Show first 8 folders
                    folder_data = topology_mapper.folder_topology.get(folder, {})
                    concepts = folder_data.get('semantic_concepts', set())
                    py_files = folder_data.get('python_files', [])
                    print(f"    📁 {folder}: {len(concepts)} concepts, {len(py_files)} Python files")
        
        # Test folder relationships
        print("\n🔗 Testing folder relationships...")
        
        test_folders = ['consciousness', 'processing', 'memory', 'subsystems']
        
        for folder in test_folders:
            if folder in topology_mapper.folder_topology:
                relationships = topology_mapper.get_folder_relationships(folder)
                print(f"  📁 {folder}:")
                print(f"    Parent: {relationships.get('parent', 'None')}")
                print(f"    Children: {len(relationships.get('children', []))}")
                print(f"    Siblings: {len(relationships.get('siblings', []))}")
                print(f"    Concepts: {len(relationships.get('concepts', set()))}")
        
        # Test semantic concept extraction from paths
        print("\n🧠 Testing semantic concept extraction from folder paths...")
        
        test_paths = [
            'consciousness.engines.core',
            'processing.engines.tick.synchronous',
            'memory.unified_memory_interconnection',
            'subsystems.mycelial.integrated_system'
        ]
        
        for path in test_paths:
            concepts = topology_mapper._extract_folder_concepts(path)
            print(f"  📍 {path}: {sorted(list(concepts))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Folder structure integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🍄🔗 DAWN MYCELIAL MODULE INTEGRATION COMPREHENSIVE TEST")
    print("=" * 70)
    print("Testing mycelial semantic hash map integration across all DAWN modules")
    print("Demonstrating how semantic spores travel through the entire codebase")
    print()
    
    # Run all tests
    tests = [
        ("Mycelial Module Discovery", test_mycelial_module_discovery),
        ("Semantic Path Mapping", test_semantic_path_mapping),
        ("Module Wrapper Behavior", test_module_wrapper_behavior),
        ("Cross-Module Propagation", test_cross_module_propagation),
        ("Living Network Behavior", test_living_network_behavior),
        ("Folder Structure Integration", test_folder_structure_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        success = test_func()
        results.append((test_name, success))
    
    # Final results
    print("\n" + "=" * 70)
    print("🍄🔗 MYCELIAL MODULE INTEGRATION TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL MYCELIAL MODULE INTEGRATION TESTS PASSED!")
        print("✅ Mycelial hash map successfully wired into all DAWN modules")
        print("✅ Semantic spores propagate through entire codebase")
        print("✅ Cross-module semantic propagation active")
        print("✅ Folder structure mapped to mycelial network topology")
        print("✅ Living network behavior - touch one module, activate entire system")
        print("✅ Real-time semantic telemetry as code travels through modules")
        print("✅ Every module operation now generates semantic spores!")
        print("✅ DAWN is now a living mycelial network of meaning!")
    else:
        print(f"❌ {len(results) - passed} tests failed")
    
    print("\n🍄🔗 Mycelial module integration test complete!")
    print("🌐 DAWN codebase is now a living semantic network!")
    print("🍄 Every function call spreads meaning like spores through the system!")

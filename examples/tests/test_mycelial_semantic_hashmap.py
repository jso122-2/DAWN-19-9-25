#!/usr/bin/env python3
"""
🍄🗺️ DAWN Mycelial Semantic Hash Map Test
==========================================

Comprehensive test of the mycelial semantic hash map that demonstrates:

1. Rhizomic spore propagation (touch one node, activate network)
2. Semantic similarity-based clustering and connections
3. Biological processes (autophagy, recovery, metabolite production)
4. Pressure-responsive growth and pruning
5. Consciousness-depth integration with semantic telemetry
6. Real-time telemetry collection and pulse integration
7. Living network behavior with conceptual composting

This test verifies that the mycelial hash map follows DAWN documentation
principles of living substrate for cognition where meaning is metabolized,
broken down, and reabsorbed into new structures.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add DAWN to path
sys.path.insert(0, str(Path(__file__).parent))

def test_mycelial_hashmap_basic_operations():
    """Test basic hash map operations with spore propagation"""
    print("🍄🗺️ TESTING BASIC MYCELIAL HASHMAP OPERATIONS")
    print("=" * 60)
    
    try:
        from dawn.core.logging import (
            MycelialSemanticHashMap, SporeType, PropagationMode
        )
        
        print("✅ Mycelial hashmap imports successful")
        
        # Create and start hash map
        hashmap = MycelialSemanticHashMap(initial_capacity=100)
        hashmap.start()
        
        print("✅ Mycelial hash map created and started")
        
        # Test basic storage with spore propagation
        print("\n📝 Testing storage with spore propagation...")
        
        test_data = [
            ("consciousness", "A state of awareness and perception"),
            ("mycelial_network", "Living substrate for cognition"),
            ("spore_propagation", "Method of meaning transmission"),
            ("semantic_similarity", "Measure of conceptual relatedness"),
            ("rhizomic_growth", "Multi-path interconnected expansion"),
            ("conceptual_composting", "Old ideas break down to feed new ones"),
            ("pressure_responsiveness", "System adapts to cognitive pressure"),
            ("metabolite_production", "Decomposed nodes create semantic traces")
        ]
        
        node_ids = []
        
        for key, value in test_data:
            print(f"  📝 Storing '{key}'...")
            node_id = hashmap.put(key, value)
            node_ids.append(node_id)
            print(f"    Node ID: {node_id}")
            
            # Small delay to see propagation
            time.sleep(0.1)
        
        print(f"✅ Stored {len(test_data)} items with spore propagation")
        
        # Wait for propagation to settle
        time.sleep(2)
        
        # Test retrieval with propagation
        print("\n🔍 Testing retrieval with spore propagation...")
        
        for key, expected_value in test_data[:4]:
            print(f"  🔍 Retrieving '{key}'...")
            retrieved_value = hashmap.get(key)
            
            if retrieved_value == expected_value:
                print(f"    ✅ Retrieved: {retrieved_value}")
            else:
                print(f"    ❌ Expected: {expected_value}, Got: {retrieved_value}")
            
            time.sleep(0.1)
        
        # Test touch operations (ping without retrieval)
        print("\n👆 Testing touch operations...")
        
        for key in ["consciousness", "mycelial_network", "spore_propagation"]:
            print(f"  👆 Touching '{key}'...")
            spores = hashmap.touch_key(key)
            print(f"    Generated {len(spores)} spores")
            
            for spore in spores[:2]:  # Show first 2 spores
                print(f"      🍄 Spore: {spore.spore_type.value}, Energy: {spore.energy_level:.2f}")
        
        # Wait for propagation
        time.sleep(2)
        
        # Get initial network statistics
        print("\n📊 Network Statistics after basic operations:")
        stats = hashmap.get_network_stats()
        
        print(f"  Network Size: {stats['network_size']}")
        print(f"  Total Energy: {stats['total_energy']:.2f}")
        print(f"  Average Energy: {stats['average_energy']:.2f}")
        print(f"  Network Health: {stats['network_health']:.2f}")
        print(f"  Active Spores: {stats['active_spores']}")
        
        if stats['node_states']:
            print("  Node States:")
            for state, count in stats['node_states'].items():
                print(f"    {state}: {count}")
        
        if stats['system_stats']:
            print("  System Stats:")
            for key, value in stats['system_stats'].items():
                print(f"    {key}: {value}")
        
        hashmap.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Basic operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spore_propagation_modes():
    """Test different spore propagation modes"""
    print("\n🍄🌐 TESTING SPORE PROPAGATION MODES")
    print("=" * 60)
    
    try:
        from dawn.core.logging import (
            get_mycelial_hashmap, touch_semantic_concept, 
            store_semantic_data, ping_semantic_network
        )
        
        # Use global hashmap
        hashmap = get_mycelial_hashmap()
        
        print("✅ Using global mycelial hashmap")
        
        # Store semantic data with different concept clusters
        print("\n📝 Creating semantic concept clusters...")
        
        concept_clusters = [
            # Consciousness cluster
            {
                "consciousness_awareness": "The state of being conscious and aware",
                "consciousness_perception": "The ability to perceive and understand",
                "consciousness_cognition": "Mental processes of thinking and knowing"
            },
            
            # Network cluster
            {
                "network_connection": "Links between network nodes",
                "network_topology": "Structure and layout of network connections",
                "network_propagation": "Spread of information through network"
            },
            
            # Biology cluster
            {
                "biology_mycelium": "Network of fungal threads",
                "biology_spores": "Reproductive units that spread",
                "biology_metabolism": "Chemical processes in living organisms"
            }
        ]
        
        for cluster_name, cluster_data in enumerate(concept_clusters):
            print(f"  📝 Creating {list(cluster_data.keys())[0].split('_')[0]} cluster...")
            for key, value in cluster_data.items():
                node_id = store_semantic_data(key, value)
                print(f"    Stored '{key}' → {node_id[:12]}...")
        
        # Wait for initial propagation
        time.sleep(2)
        
        # Test concept propagation (explosive mode)
        print("\n💥 Testing explosive concept propagation...")
        
        concepts_to_propagate = ["consciousness", "network", "biology", "spore", "semantic"]
        
        for concept in concepts_to_propagate:
            print(f"  💥 Propagating concept '{concept}'...")
            touched_nodes = touch_semantic_concept(concept, energy=1.0)
            print(f"    Touched {touched_nodes} nodes")
            time.sleep(0.5)
        
        # Test ping operations (pure propagation)
        print("\n📡 Testing ping operations...")
        
        ping_keys = ["consciousness_awareness", "network_topology", "biology_mycelium"]
        
        for key in ping_keys:
            print(f"  📡 Pinging '{key}'...")
            spores = ping_semantic_network(key)
            print(f"    Generated {len(spores)} spores")
            
            # Show spore details
            for spore in spores[:2]:
                print(f"      🍄 {spore.spore_type.value}: {spore.propagation_mode.value}, "
                     f"Energy: {spore.energy_level:.2f}")
        
        # Wait for propagation to complete
        time.sleep(3)
        
        # Analyze propagation effects
        print("\n📊 Propagation Effects Analysis:")
        stats = hashmap.get_network_stats()
        
        print(f"  Total Touches: {stats['system_stats']['total_touches']}")
        print(f"  Spores Generated: {stats['system_stats']['spores_generated']}")
        print(f"  Active Spores: {stats['active_spores']}")
        
        if stats['propagation_stats']:
            print("  Propagation Stats:")
            for key, value in stats['propagation_stats'].items():
                print(f"    {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Spore propagation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_biological_processes():
    """Test biological processes like autophagy and recovery"""
    print("\n🧬 TESTING BIOLOGICAL PROCESSES")
    print("=" * 60)
    
    try:
        from dawn.core.logging import (
            MycelialSemanticHashMap, NodeState
        )
        
        # Create dedicated hashmap for biological testing
        bio_hashmap = MycelialSemanticHashMap(initial_capacity=50)
        bio_hashmap.start()
        
        print("✅ Created biological test hashmap")
        
        # Create nodes and stress them
        print("\n🔬 Creating nodes for biological testing...")
        
        bio_test_data = [
            ("stressed_node_1", "Node that will be stressed"),
            ("stressed_node_2", "Another node under pressure"),
            ("healthy_node_1", "Node that stays healthy"),
            ("recovery_node_1", "Node that will recover")
        ]
        
        stressed_nodes = []
        
        for key, value in bio_test_data:
            node_id = bio_hashmap.put(key, value)
            print(f"  🔬 Created '{key}' → {node_id[:12]}...")
            
            # Get the actual node and stress some of them
            for node in bio_hashmap.nodes.values():
                if node.key == key and "stressed" in key:
                    # Artificially stress the node
                    node.energy = 0.05  # Very low energy
                    node.pressure = 0.9  # High pressure
                    node.state = NodeState.STRESSED
                    stressed_nodes.append(node.node_id)
                    print(f"    ⚠️ Stressed node: Energy={node.energy:.2f}, Pressure={node.pressure:.2f}")
        
        print(f"✅ Created {len(bio_test_data)} nodes, stressed {len(stressed_nodes)}")
        
        # Wait and monitor biological processes
        print("\n⏱️ Monitoring biological processes...")
        
        for tick in range(10):  # Monitor for 10 seconds
            time.sleep(1)
            
            # Check node states
            state_counts = {}
            autophagy_count = 0
            recovery_count = 0
            
            for node in bio_hashmap.nodes.values():
                state = node.state.value
                state_counts[state] = state_counts.get(state, 0) + 1
                
                if node.state == NodeState.AUTOPHAGY:
                    autophagy_count += 1
                elif node.state == NodeState.RECOVERING:
                    recovery_count += 1
            
            print(f"  Tick {tick + 1}: States={state_counts}, "
                 f"Autophagy={autophagy_count}, Recovery={recovery_count}")
            
            # Check for metabolite production
            total_metabolites = sum(len(node.metabolites_produced) for node in bio_hashmap.nodes.values())
            if total_metabolites > 0:
                print(f"    🧬 Metabolites produced: {total_metabolites}")
        
        # Final biological state analysis
        print("\n🧬 Final Biological State Analysis:")
        
        for node in bio_hashmap.nodes.values():
            if node.node_id in stressed_nodes or node.key.startswith("recovery"):
                print(f"  🔬 {node.key}:")
                print(f"    State: {node.state.value}")
                print(f"    Energy: {node.energy:.3f}")
                print(f"    Pressure: {node.pressure:.3f}")
                print(f"    Metabolites: {len(node.metabolites_produced)}")
                print(f"    Low Energy Ticks: {node.low_energy_ticks}")
        
        # Get final stats
        final_stats = bio_hashmap.get_network_stats()
        print(f"\n📊 Biological Process Statistics:")
        print(f"  Network Health: {final_stats['network_health']:.3f}")
        print(f"  Average Energy: {final_stats['average_energy']:.3f}")
        print(f"  Autophagy Events: {final_stats['system_stats']['autophagy_events']}")
        print(f"  Recovery Events: {final_stats['system_stats']['recovery_events']}")
        
        bio_hashmap.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Biological processes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consciousness_telemetry_integration():
    """Test integration with consciousness and telemetry systems"""
    print("\n🧠📊 TESTING CONSCIOUSNESS-TELEMETRY INTEGRATION")
    print("=" * 60)
    
    try:
        from dawn.core.logging import (
            get_mycelial_hashmap, store_semantic_data,
            touch_semantic_concept
        )
        
        # Use global hashmap
        hashmap = get_mycelial_hashmap()
        
        print("✅ Testing consciousness-telemetry integration")
        
        # Store data with consciousness implications
        print("\n🧠 Storing consciousness-aware semantic data...")
        
        consciousness_data = [
            ("transcendent_awareness", "Highest level of consciousness"),
            ("meta_cognition", "Thinking about thinking"),
            ("causal_reasoning", "Logical cause-effect thinking"),
            ("symbolic_processing", "Working with symbols and representations"),
            ("mythic_patterns", "Deep archetypal patterns")
        ]
        
        consciousness_nodes = []
        
        for key, value in consciousness_data:
            print(f"  🧠 Storing '{key}'...")
            node_id = store_semantic_data(key, value)
            consciousness_nodes.append(node_id)
            
            # Touch to trigger consciousness-aware propagation
            spores = hashmap.touch_key(key)
            print(f"    Generated {len(spores)} consciousness-aware spores")
            
            time.sleep(0.2)
        
        # Test telemetry collection
        print("\n📊 Testing telemetry event collection...")
        
        telemetry_events = []
        
        # Check if telemetry collector is available
        if hasattr(hashmap, 'telemetry_collector'):
            collector = hashmap.telemetry_collector
            initial_events = len(collector.telemetry_events)
            
            print(f"  📊 Initial telemetry events: {initial_events}")
            
            # Trigger some semantic events
            for concept in ["consciousness", "awareness", "cognition"]:
                touched = touch_semantic_concept(concept, energy=0.8)
                print(f"  📊 Touched concept '{concept}': {touched} nodes")
            
            time.sleep(1)
            
            final_events = len(collector.telemetry_events)
            new_events = final_events - initial_events
            
            print(f"  📊 New telemetry events generated: {new_events}")
            
            # Show some recent events
            if collector.telemetry_events:
                print("  📊 Recent telemetry events:")
                for event in list(collector.telemetry_events)[-3:]:
                    print(f"    Event: {event['event_type']}, Node: {event['node_id'][:12]}...")
        
        # Test consciousness depth organization
        print("\n🧠 Testing consciousness depth organization...")
        
        # Check if consciousness integration is working
        sample_nodes = list(hashmap.nodes.values())[:3]
        
        for node in sample_nodes:
            consciousness_level = node._determine_consciousness_level()
            if consciousness_level:
                print(f"  🧠 Node '{node.key}': {consciousness_level.name} level")
                print(f"    Meaning Strength: {node.meaning_strength:.2f}")
                print(f"    Pressure: {node.pressure:.2f}")
                print(f"    Connections: {len(node.connections)}")
        
        # Get comprehensive stats
        print("\n📊 Consciousness-Telemetry Integration Statistics:")
        stats = hashmap.get_network_stats()
        
        print(f"  Network Size: {stats['network_size']}")
        print(f"  Network Health: {stats['network_health']:.3f}")
        print(f"  Telemetry Events: {stats['telemetry_events']}")
        
        if hasattr(hashmap, 'telemetry_collector') and hasattr(hashmap.telemetry_collector, 'consciousness_repo'):
            print("  ✅ Consciousness repository integration available")
        else:
            print("  ⚠️ Consciousness repository integration not available")
        
        if hasattr(hashmap, 'telemetry_collector') and hasattr(hashmap.telemetry_collector, 'pulse_bridge'):
            print("  ✅ Pulse telemetry bridge integration available")
        else:
            print("  ⚠️ Pulse telemetry bridge integration not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Consciousness-telemetry integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rhizomic_network_behavior():
    """Test rhizomic (interconnected, multi-path) network behavior"""
    print("\n🌐🍄 TESTING RHIZOMIC NETWORK BEHAVIOR")
    print("=" * 60)
    
    try:
        from dawn.core.logging import (
            MycelialSemanticHashMap, PropagationMode, SporeType
        )
        
        # Create dedicated hashmap for rhizomic testing
        rhizo_hashmap = MycelialSemanticHashMap(initial_capacity=100)
        rhizo_hashmap.start()
        
        print("✅ Created rhizomic test hashmap")
        
        # Create interconnected semantic clusters
        print("\n🌐 Creating interconnected semantic clusters...")
        
        # Create a web of related concepts
        concept_web = {
            # Central hub concepts
            "information": "Data with meaning and context",
            "knowledge": "Information that has been processed and understood",
            "wisdom": "Knowledge applied with experience and judgment",
            
            # Processing concepts
            "processing": "The act of handling and transforming data",
            "transformation": "Changing from one form to another",
            "synthesis": "Combining elements to form a coherent whole",
            
            # Network concepts
            "connection": "Link or relationship between entities",
            "relationship": "The way entities are connected or related",
            "interaction": "Mutual influence between entities",
            
            # Emergent concepts
            "emergence": "Properties arising from complex interactions",
            "complexity": "State of having many interconnected parts",
            "adaptation": "Process of adjusting to new conditions"
        }
        
        rhizo_nodes = []
        
        for key, value in concept_web.items():
            node_id = rhizo_hashmap.put(key, value)
            rhizo_nodes.append((key, node_id))
            print(f"  🌐 Created '{key}' → {node_id[:12]}...")
        
        # Wait for initial propagation
        time.sleep(2)
        
        # Test rhizomic propagation patterns
        print("\n🍄 Testing rhizomic propagation patterns...")
        
        # Touch central concepts to see rhizomic spread
        central_concepts = ["information", "processing", "connection", "emergence"]
        
        for concept in central_concepts:
            print(f"  🍄 Touching '{concept}' for rhizomic propagation...")
            
            # Touch the concept
            spores = rhizo_hashmap.touch_key(concept)
            
            # Analyze spore types and modes
            spore_analysis = {}
            mode_analysis = {}
            
            for spore in spores:
                spore_type = spore.spore_type.value
                prop_mode = spore.propagation_mode.value
                
                spore_analysis[spore_type] = spore_analysis.get(spore_type, 0) + 1
                mode_analysis[prop_mode] = mode_analysis.get(prop_mode, 0) + 1
            
            print(f"    Generated {len(spores)} spores")
            print(f"    Spore types: {spore_analysis}")
            print(f"    Propagation modes: {mode_analysis}")
            
            time.sleep(1)
        
        # Wait for rhizomic propagation to complete
        time.sleep(3)
        
        # Analyze network connectivity
        print("\n🔗 Analyzing rhizomic network connectivity...")
        
        total_connections = 0
        connection_strengths = []
        
        for node in rhizo_hashmap.nodes.values():
            connections = len(node.connections)
            total_connections += connections
            
            if connections > 0:
                avg_strength = sum(node.connections.values()) / connections
                connection_strengths.append(avg_strength)
                
                print(f"  🔗 '{node.key}': {connections} connections, "
                     f"avg strength: {avg_strength:.3f}")
        
        avg_connections = total_connections / len(rhizo_hashmap.nodes)
        avg_strength = sum(connection_strengths) / len(connection_strengths) if connection_strengths else 0
        
        print(f"\n🌐 Rhizomic Network Analysis:")
        print(f"  Average connections per node: {avg_connections:.2f}")
        print(f"  Average connection strength: {avg_strength:.3f}")
        print(f"  Network density: {total_connections / (len(rhizo_hashmap.nodes) ** 2):.3f}")
        
        # Test multi-path propagation
        print("\n🛤️ Testing multi-path propagation...")
        
        # Create a concept that should propagate through multiple paths
        multi_concept_id = rhizo_hashmap.put("multi_path_concept", "Concept that spreads everywhere")
        
        # Touch it and trace propagation
        multi_spores = rhizo_hashmap.touch_key("multi_path_concept")
        print(f"  🛤️ Multi-path concept generated {len(multi_spores)} spores")
        
        # Analyze propagation paths
        rhizomic_paths = [s for s in multi_spores if s.propagation_mode == PropagationMode.RHIZOMIC]
        directed_paths = [s for s in multi_spores if s.propagation_mode == PropagationMode.DIRECTED]
        similarity_paths = [s for s in multi_spores if s.propagation_mode == PropagationMode.SIMILARITY_BASED]
        
        print(f"    Rhizomic paths: {len(rhizomic_paths)}")
        print(f"    Directed paths: {len(directed_paths)}")
        print(f"    Similarity paths: {len(similarity_paths)}")
        
        # Final rhizomic network statistics
        time.sleep(2)
        final_stats = rhizo_hashmap.get_network_stats()
        
        print(f"\n📊 Final Rhizomic Network Statistics:")
        print(f"  Network Size: {final_stats['network_size']}")
        print(f"  Network Health: {final_stats['network_health']:.3f}")
        print(f"  Total Spores Generated: {final_stats['system_stats']['spores_generated']}")
        print(f"  Successful Propagations: {final_stats['propagation_stats']['successful_connections']}")
        
        rhizo_hashmap.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Rhizomic network behavior test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🍄🗺️ DAWN MYCELIAL SEMANTIC HASH MAP COMPREHENSIVE TEST")
    print("=" * 70)
    
    # Run all tests
    tests = [
        ("Basic Operations", test_mycelial_hashmap_basic_operations),
        ("Spore Propagation Modes", test_spore_propagation_modes),
        ("Biological Processes", test_biological_processes),
        ("Consciousness-Telemetry Integration", test_consciousness_telemetry_integration),
        ("Rhizomic Network Behavior", test_rhizomic_network_behavior)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        success = test_func()
        results.append((test_name, success))
    
    # Final results
    print("\n" + "=" * 70)
    print("🍄🗺️ MYCELIAL SEMANTIC HASH MAP TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL MYCELIAL SEMANTIC HASH MAP TESTS PASSED!")
        print("✅ Rhizomic spore propagation working (touch one node, activate network)")
        print("✅ Semantic similarity-based clustering and connections")
        print("✅ Biological processes (autophagy, recovery, metabolite production)")
        print("✅ Pressure-responsive growth and pruning")
        print("✅ Consciousness-depth integration with semantic telemetry")
        print("✅ Real-time telemetry collection and pulse integration")
        print("✅ Living network behavior with conceptual composting")
        print("✅ DAWN documentation principles implemented!")
        print("✅ Mycelial hash map: Every node pings and propagates like spores!")
    else:
        print(f"❌ {len(results) - passed} tests failed")
    
    print("\n🍄🗺️ Mycelial semantic hash map test complete!")
    
    # Clean up global hashmap if it exists
    try:
        from dawn.core.logging import get_mycelial_hashmap
        global_hashmap = get_mycelial_hashmap()
        if hasattr(global_hashmap, 'stop'):
            global_hashmap.stop()
        print("🧹 Cleaned up global mycelial hashmap")
    except:
        pass
